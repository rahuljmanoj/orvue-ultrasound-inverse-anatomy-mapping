"""
orvue_us_inverse.simulation.bmode - Real-time synthetic B-mode of a fixed virtual anatomy under a flat phantom.

Phantom frame (all units mm) - see anatomy.py for the anatomical meaning of the axes:
    origin at a corner of the 100 x 100 mm active region on the phantom surface,
    +x towards the patient's left (medial), +y caudal, +z down into tissue (posterior).

Probe frame convention (for a 4x4 pose from your tracker, expressed in the phantom frame):
    origin  = centre of the transducer face
    x-axis  = lateral (along the array)
    y-axis  = elevation
    z-axis  = beam/axial direction (into tissue)

Run standalone for an interactive mouse/keyboard demo (no D405 needed):
    python -m orvue_us_inverse sim [case_name]   e.g. python -m orvue_us_inverse sim parallel_cystic_duct
With the D405 camera tracker (see README.md for setup and calibration):
    python -m orvue_us_inverse sim [case_name] --track [--cam-view] [--mouse]
        --track     the tracked probe moves the virtual probe (camera control)
        --cam-view  also open the "tracking" window (camera feed + tracker top view)
        --mouse     start in mouse control
Controls: move mouse over the top view = probe position, left-click there = lock / unlock the probe
          (locked, it stays put while the mouse is used elsewhere, and ignores the tracker),
          q/e = rotate probe, n = next anatomy case, g = toggle ground-truth contours, c = toggle contact,
          Esc = quit. With the tracker: m = switch camera / mouse control, t = show / hide the tracking window.
Imaging:  click or drag the Focus / Gain / DR / Pers / TGC sliders in the B-mode window's
          left panel (settings are kept when switching case).

Dependencies: numpy, scipy, opencv-contrib-python (pyrealsense2 for the tracker)
"""
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter
from orvue_us_inverse.simulation.anatomy import (Anatomy, build_case, CASES, TISSUES, N_LAB, Z_TAB, ECHO_TAB, ATT_TAB, COL_TAB,
                     TEX_TAB, PT_TAB, calot_triangle)


# --------------------------------------------------------------------------------------
# Fixed scatterer field via coordinate hashing (no stored volume, speckle is tied to tissue)
# --------------------------------------------------------------------------------------
def _mix32(h):
    h ^= h >> np.uint32(16); h *= np.uint32(0x7FEB352D)
    h ^= h >> np.uint32(15); h *= np.uint32(0x846CA68B)
    h ^= h >> np.uint32(16)
    return h


def scatterers(P, res=0.1, seed=1234):
    """Complex Gaussian scatterer amplitude at points P, constant per res-sized cell.
    Deterministic in phantom coordinates, so speckle stays attached to the tissue."""
    q = (np.floor(P * (1.0 / res)).astype(np.int32) + (1 << 14)).astype(np.uint32)
    with np.errstate(over="ignore"):
        h = _mix32(q[:, 0] * np.uint32(73856093) ^ q[:, 1] * np.uint32(19349663)
                   ^ q[:, 2] * np.uint32(83492791) ^ np.uint32(seed))
        h2 = _mix32(h ^ np.uint32(0x9E3779B9))
        h3 = _mix32(h2 ^ np.uint32(0x85EBCA6B))
    u1 = np.clip((h.astype(np.float32) + 1.0) * np.float32(2.0 ** -32), 1e-7, 1.0)
    u2 = h2.astype(np.float32) * np.float32(2.0 ** -32)
    u3 = h3.astype(np.float32) * np.float32(2.0 ** -32)
    r = np.sqrt(-np.log(u1))
    ph = np.float32(2 * np.pi) * u2
    return (r * np.cos(ph)).astype(np.float32), (r * np.sin(ph)).astype(np.float32), u3


def value_noise(P, cell=0.8, seed=99):
    """Smooth, tissue-fixed random field (~unit std) with correlation length ~cell mm."""
    g = P * (1.0 / cell)
    i0 = np.floor(g).astype(np.int32)
    f = g - i0
    f = f * f * (3 - 2 * f)                                       # smoothstep
    out = np.zeros(P.shape[0], np.float32)
    base = (i0 + (1 << 14)).astype(np.uint32)
    with np.errstate(over="ignore"):
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    q = base + np.array([dx, dy, dz], np.uint32)
                    h = _mix32(q[:, 0] * np.uint32(73856093) ^ q[:, 1] * np.uint32(19349663)
                               ^ q[:, 2] * np.uint32(83492791) ^ np.uint32(seed))
                    val = h.astype(np.float32) * np.float32(2.0 ** -31) - 1.0
                    w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                         * (f[:, 2] if dz else 1 - f[:, 2]))
                    out += w * val
    return out * 2.2


# --------------------------------------------------------------------------------------
# Probe + simulator
# --------------------------------------------------------------------------------------
@dataclass
class LinearProbe:
    f0_mhz: float = 7.5
    width_mm: float = 30.0
    depth_mm: float = 50.0          # imaging depth = the full 50 mm anatomy volume
    px_mm: float = 0.1
    fnum: float = 3.0
    cycles: float = 2.5
    elev_sigma_mm: float = 0.5       # slice thickness (1-sigma)
    focus_mm: float = 20.0           # transmit focus depth
    rayleigh_mm: float = 10.0        # depth over which the beam widens by sqrt(2)
    c_mm_us: float = 1.54

    @property
    def wavelength(self):
        return self.c_mm_us / self.f0_mhz

    @property
    def psf_sigma_px(self):
        sig_ax = 0.5 * self.cycles * self.wavelength / 2.355          # ~ half pulse length FWHM
        sig_lat = self.fnum * self.wavelength / 2.355
        return sig_ax / self.px_mm, sig_lat / self.px_mm


class BModeSimulator:
    def __init__(self, anatomy: Anatomy, probe: LinearProbe = LinearProbe(),
                 dyn_range_db=50.0, gain_db=0.0, noise_db=-42.0, tgc_db_cm_mhz=0.5, specular_gain=4.0, n_elev=5,
                 compound_deg=(-7.0, 0.0, 7.0), band_mm=4.0, persistence=0.3, gamma=1.25, point_gain=5.0, sri=0.6):
        self.an, self.pr = anatomy, probe
        self.compound_deg, self.band_mm, self.persistence = compound_deg, band_mm, persistence
        self.point_gain = point_gain
        self.sri = sri          # speckle-reduction strength 0..1 (smoothing in the log domain)
        self._prev = None
        x = np.linspace(0, 1, 256)
        self.lut = (255 * x ** gamma).astype(np.uint8)            # grey map
        self.dr, self.gain_db, self.tgc, self.spec_gain = dyn_range_db, gain_db, tgc_db_cm_mhz, specular_gain
        p = probe
        self.lat = np.arange(-p.width_mm / 2, p.width_mm / 2 + 1e-6, p.px_mm, dtype=np.float32)
        self.ax = np.arange(0.0, p.depth_mm + 1e-6, p.px_mm, dtype=np.float32)
        self.nz, self.nx = len(self.ax), len(self.lat)
        e = np.linspace(-2, 2, n_elev) * p.elev_sigma_mm
        w = np.exp(-0.5 * (e / p.elev_sigma_mm) ** 2)
        self.elev, self.elev_w = e.astype(np.float32), (w / np.sqrt((w ** 2).sum())).astype(np.float32)
        L, D = np.meshgrid(self.lat, self.ax)
        self._L, self._D = L.reshape(-1, 1), D.reshape(-1, 1)
        Lc, Dc = np.meshgrid(self.lat[::2], self.ax[::2])          # 0.2 mm grid for tissue labels
        self._Lc, self._Dc = Lc.reshape(-1, 1), Dc.reshape(-1, 1)
        self._nzc, self._nxc = Lc.shape
        self.set_tgc(tgc_db_cm_mhz)
        self._build_psf_bank()
        self.ref_liver = self._calibrate()
        self.noise = self.ref_liver * 10 ** (noise_db / 20.0)
        self._rng = np.random.default_rng()

    # ---- live controls -----------------------------------------------------------------
    # gain_db, dr and persistence are read every frame and can be assigned directly;
    # focus and TGC feed precomputed tables, so they go through these setters.
    def set_tgc(self, db_cm_mhz):
        """TGC slope: two-way gain added per cm of depth, per MHz (compensates attenuation)."""
        self.tgc = float(db_cm_mhz)
        self.tgc_gain = 10 ** (2 * self.tgc * self.pr.f0_mhz * (self.ax / 10.0) / 20.0)

    def set_focus(self, focus_mm):
        """Move the transmit focus and rebuild the depth-dependent beam (PSF) table."""
        # copy the probe rather than mutating it: the constructor's default probe is shared
        self.pr = replace(self.pr, focus_mm=float(focus_mm))
        self._build_psf_bank()

    # ---- point spread function ---------------------------------------------------------
    def _build_psf_bank(self):
        """Separable PSF per depth band: axial Gaussian x lateral sinc^2 (with side lobes).
        Lateral width follows the beam: narrowest at the focus, wider above and below."""
        p = self.pr
        sig_ax = 0.5 * p.cycles * p.wavelength / 2.355 / p.px_mm
        ha = int(np.ceil(4 * sig_ax))
        t = np.arange(-ha, ha + 1, dtype=np.float32)
        self.k_ax = np.exp(-0.5 * (t / sig_ax) ** 2).astype(np.float32)
        self.k_ax /= np.sqrt((self.k_ax ** 2).sum())
        n_b = int(np.ceil(p.depth_mm / self.band_mm))
        edges = np.linspace(0.0, p.depth_mm, n_b + 1)
        self.bands = []
        for i, (z0, z1) in enumerate(zip(edges[:-1], edges[1:])):
            zc = 0.5 * (z0 + z1)
            w = p.wavelength * p.fnum * np.sqrt(1 + ((zc - p.focus_mm) / p.rayleigh_mm) ** 2)
            hl = int(np.ceil(3.0 * w / p.px_mm))
            x = np.arange(-hl, hl + 1, dtype=np.float32) * p.px_mm
            kl = (np.sinc(x / w) ** 2 * np.exp(-0.5 * (x / (1.5 * w)) ** 2)).astype(np.float32)
            kl /= np.sqrt((kl ** 2).sum())
            r0 = int(round(z0 / p.px_mm))
            r1 = self.nz if i == n_b - 1 else int(round(z1 / p.px_mm))     # last band covers every row
            self.bands.append((r0, r1, kl))
        # steering by shear: a beam steered by angle a samples x + z*tan(a)
        self.shears = []
        for ang in self.compound_deg:
            k = np.tan(np.radians(ang))
            c = -k * self.nz / 2.0
            M = np.float32([[1, k, c], [0, 1, 0]])
            Mi = np.float32([[1, -k, -c], [0, 1, 0]])
            self.shears.append((ang, M, Mi))

    def _envelope(self, fr, fi):
        """Depth-dependent PSF + spatial compounding (mean envelope over steering angles)."""
        import cv2
        env = np.zeros_like(fr)
        m = len(self.k_ax) // 2 + 1
        for ang, M, Mi in self.shears:
            if ang != 0:
                ar = cv2.warpAffine(fr, M, (self.nx, self.nz), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                ai = cv2.warpAffine(fi, M, (self.nx, self.nz), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            else:
                ar, ai = fr, fi
            e = np.zeros_like(fr)
            for r0, r1, kl in self.bands:
                a0, a1 = max(r0 - m, 0), min(r1 + m, self.nz)
                cr = cv2.sepFilter2D(ar[a0:a1], -1, kl, self.k_ax, borderType=cv2.BORDER_REFLECT)
                ci = cv2.sepFilter2D(ai[a0:a1], -1, kl, self.k_ax, borderType=cv2.BORDER_REFLECT)
                e[r0:r1] = np.hypot(cr, ci)[r0 - a0:r1 - a0]
            if ang != 0:
                e = cv2.warpAffine(e, Mi, (self.nx, self.nz), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            env += e
        return env / len(self.shears)

    def _calibrate(self):
        """Fixed display reference: median envelope of homogeneous liver (no per-frame auto-gain)."""
        T = self.pose_from_xy_yaw(-500.0, -500.0, 0.0).astype(np.float32)
        P0, v = self.plane_points(T), T[:3, 1]
        re_ = np.zeros(len(P0), np.float32); im_ = np.zeros(len(P0), np.float32)
        for e, w in zip(self.elev, self.elev_w):
            a, b, _ = scatterers(P0 + e * v); re_ += w * a; im_ += w * b
        env = self._envelope(ECHO_TAB[0] * re_.reshape(self.nz, self.nx),
                             ECHO_TAB[0] * im_.reshape(self.nz, self.nx))
        return float(np.median(env))

    # ---- poses -------------------------------------------------------------------------
    @staticmethod
    def pose_from_xy_yaw(x, y, yaw_deg, z=0.0):
        """Probe perpendicular to the surface at (x, y), rotated yaw_deg about the surface normal."""
        c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        T = np.eye(4, dtype=np.float32)
        T[:3, 0] = [c, s, 0]          # lateral
        T[:3, 1] = [-s, c, 0]         # elevation
        T[:3, 2] = [0, 0, 1]          # axial (down)
        T[:3, 3] = [x, y, z]
        return T

    def in_contact(self, T, tol_mm=1.0, max_tilt_deg=30.0):
        x, y, z = T[:3, 3]
        inside = 0 <= x <= self.an.size_xy[0] and 0 <= y <= self.an.size_xy[1]
        tilt = np.degrees(np.arccos(np.clip(T[2, 2], -1, 1)))
        return inside and abs(z) <= tol_mm and tilt <= max_tilt_deg

    def plane_points(self, T, elev=0.0, coarse=False):
        T = T.astype(np.float32)
        o, u, v, n = T[:3, 3], T[:3, 0], T[:3, 1], T[:3, 2]
        L, D = (self._Lc, self._Dc) if coarse else (self._L, self._D)
        return (o + elev * v) + L * u + D * n

    def labels_image(self, T):
        """Tissue labels on the image grid (computed at 0.2 mm, upsampled)."""
        labc = self.an.labels(self.plane_points(T, coarse=True)).reshape(self._nzc, self._nxc)
        return np.repeat(np.repeat(labc, 2, 0), 2, 1)[:self.nz, :self.nx]

    # ---- rendering ---------------------------------------------------------------------
    def render(self, T, return_labels=False):
        p = self.pr
        T = T.astype(np.float32)
        import cv2
        lab = self.labels_image(T)
        # mm-scale parenchymal texture, fixed in tissue (computed on the 0.2 mm grid, upsampled)
        nzc = value_noise(self.plane_points(T, coarse=True)).reshape(self._nzc, self._nxc)
        nz_ = cv2.resize(nzc, (self.nx, self.nz), interpolation=cv2.INTER_LINEAR)
        tex = TEX_TAB[lab]
        echo = ECHO_TAB[lab] * np.exp(tex * nz_ - 0.5 * tex ** 2)
        P0 = self.plane_points(T)
        v = T[:3, 1]
        pt_dens = PT_TAB[lab].ravel()

        # diffuse scattering, integrated over elevation (slice thickness)
        re_ = np.zeros(self.nz * self.nx, np.float32)
        im_ = np.zeros(self.nz * self.nx, np.float32)
        for e, w in zip(self.elev, self.elev_w):
            a, b, u3 = scatterers(P0 + e * v if e != 0 else P0)
            g = np.where(u3 < pt_dens, self.point_gain, 1.0).astype(np.float32)   # sparse strong scatterers
            re_ += w * a * g; im_ += w * b * g
        fr = echo * re_.reshape(self.nz, self.nx)
        fi = echo * im_.reshape(self.nz, self.nx)

        # specular reflections at impedance jumps along each scan line
        Z = Z_TAB[lab]
        R = np.zeros_like(Z)
        R[1:] = np.abs(Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])
        fr = fr + self.spec_gain * R

        # transmission loss below interfaces (gives stone shadowing)
        trans = np.cumprod(1.0 - R ** 2, axis=0)

        # depth-dependent attenuation integrated along each line, two-way
        att_db = 2.0 * np.cumsum(ATT_TAB[lab] * p.f0_mhz * (p.px_mm / 10.0), axis=0)
        amp = 10 ** (-att_db / 20.0) * trans * self.tgc_gain[:, None]

        # PSF blur (envelope domain), then envelope
        env = self._envelope(fr, fi) * amp
        # electronic noise floor (changes every frame, like a real scanner)
        env = np.hypot(env, self.noise * self._rng.standard_normal(env.shape, np.float32))

        # log compression with fixed gain: homogeneous liver sits at mid-grey
        db = 20 * np.log10(env / self.ref_liver + 1e-12) - 0.5 * self.dr + self.gain_db
        if self.sri > 0:
            # speckle reduction: smooth in log domain, anisotropic like the PSF, then blend
            sm = cv2.GaussianBlur(db, (0, 0), sigmaX=2.0, sigmaY=1.0)
            db = (1 - self.sri) * db + self.sri * sm
        img = np.nan_to_num(np.clip((db + self.dr) / self.dr, 0, 1)).astype(np.float32)
        # persistence (temporal averaging, as on clinical scanners); set persistence=0 to disable
        if self.persistence > 0 and self._prev is not None and self._prev.shape == img.shape:
            img = self.persistence * self._prev + (1 - self.persistence) * img
        self._prev = img
        img = self.lut[(255 * img).astype(np.uint8)]
        return (img, lab) if return_labels else img

    def ground_truth_overlay(self, img, lab):
        """Colour contours of lumens on top of the B-mode (BGR)."""
        import cv2
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for l, colour in [(3, (60, 220, 60)), (7, (60, 60, 255)), (9, (255, 120, 60)), (4, (0, 255, 255)),
                          (10, (200, 120, 255))]:
            m = (lab == l).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, colour, 1)
        return out


# --------------------------------------------------------------------------------------
# Clinical-style display: image on black with depth scale, focus marker and settings panel
# --------------------------------------------------------------------------------------
from orvue_us_inverse.paths import LOGO_PATH  # noqa: E402


@lru_cache(maxsize=None)
def load_logo(height, path=LOGO_PATH):
    """Company logo scaled to the given height with its aspect ratio kept (BGR), or None if missing."""
    import cv2
    logo = cv2.imread(path, cv2.IMREAD_COLOR)
    if logo is None:
        print(f"[WARN] Logo not found at {path}; continuing without it")
        return None
    width = max(1, int(round(logo.shape[1] * height / logo.shape[0])))
    return cv2.resize(logo, (width, height), interpolation=cv2.INTER_AREA)


def clinical_view(us, sim, status_lines=(), sliders=None, panel_w=200, scale_w=70, top_h=56, bottom_h=16,
                  logo_h=40):
    """Lay out a scanner-style screen around a B-mode image (BGR, any size, 0 mm depth at row 0).

    Right of the image: depth scale in cm (minor tick every 5 mm, major + number every 10 mm)
    with a triangle marking the transmit focus. Above the image's left edge: the orientation
    dot (marks the probe's -x / lateral-negative end, which is the image's left side).
    Left panel: the simulator's imaging settings, then the caller's status_lines.
    Top-left corner: the Orvue Surgical logo, logo_h pixels tall.
    With a SliderPanel, the adjustable settings are drawn as mouse sliders instead of text.
    """
    import cv2
    p = sim.pr
    h, w = us.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    grey, white, amber = (170, 170, 170), (235, 235, 235), (0, 200, 255)
    canvas = np.zeros((top_h + h + bottom_h, panel_w + w + scale_w, 3), np.uint8)
    x0, y0 = panel_w, top_h
    canvas[y0:y0 + h, x0:x0 + w] = us

    # top bar: logo on the left, probe / preset above the image, date-time on the right
    text_y = top_h // 2 + 6
    logo = load_logo(logo_h)
    if logo is not None:
        ly = (top_h - logo_h) // 2
        lw = min(logo.shape[1], panel_w - 20)
        canvas[ly:ly + logo_h, 10:10 + lw] = logo[:, :lw]
        cv2.putText(canvas, "Orvue Surgical", (10 + lw + 10, text_y), font, 0.5, white, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"L{p.width_mm:g}  {p.f0_mhz:g} MHz  B-mode", (x0 + 20, text_y), font, 0.55, white, 1,
                cv2.LINE_AA)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(stamp, font, 0.5, 1)
    cv2.putText(canvas, stamp, (canvas.shape[1] - tw - 10, text_y), font, 0.5, grey, 1, cv2.LINE_AA)

    # orientation marker
    cv2.circle(canvas, (x0 + 8, y0 - 8), 5, amber, -1, cv2.LINE_AA)

    # depth scale
    px_per_mm = h / (sim.nz * p.px_mm)
    sx = x0 + w + 10
    cv2.line(canvas, (sx, y0), (sx, y0 + h - 1), grey, 1)
    for d in range(0, int(p.depth_mm) + 1, 5):
        y = y0 + int(round(d * px_per_mm))
        if y >= y0 + h:
            break
        major = d % 10 == 0
        cv2.line(canvas, (sx, y), (sx + (12 if major else 6), y), white if major else grey, 1)
        if major and d > 0:
            cv2.putText(canvas, str(d // 10), (sx + 18, y + 6), font, 0.55, white, 1, cv2.LINE_AA)
    cv2.putText(canvas, "cm", (sx + 34, y0 + h + 12), font, 0.45, grey, 1, cv2.LINE_AA)   # clear of a bottom number
    fy = y0 + int(round(p.focus_mm * px_per_mm))
    tri = np.array([[sx - 2, fy], [sx - 10, fy - 6], [sx - 10, fy + 6]], np.int32)
    cv2.fillConvexPoly(canvas, tri, amber, cv2.LINE_AA)

    # left panel: imaging settings, then status
    def draw_lines(lines, y):
        for text, colour in lines:
            # "name   value": name at the left edge, value in a column so the numbers line up
            name, _, value = text.partition(" ")
            cv2.putText(canvas, name, (10, y), font, 0.5, colour, 1, cv2.LINE_AA)
            cv2.putText(canvas, value.strip(), (75, y), font, 0.5, colour, 1, cv2.LINE_AA)
            y += 24
        return y

    y = draw_lines([("SIMULATION", amber), ("B-MODE", white),
                    (f"Freq   {p.f0_mhz:g} MHz", grey), (f"Depth  {p.depth_mm / 10:.1f} cm", grey)], y0 + 22)
    if sliders is not None:
        y = sliders.draw(canvas, 10, y - 4, panel_w - 24) + 20
    else:
        y = draw_lines([(f"Focus  {p.focus_mm / 10:.1f} cm", grey), (f"Gain   {sim.gain_db:+g} dB", grey),
                        (f"DR     {sim.dr:g} dB", grey), (f"Pers   {sim.persistence:.1f}", grey),
                        (f"TGC    {sim.tgc:.1f} dB/cm/MHz", grey)], y)
    draw_lines([("", grey)] + [(s, grey) for s in status_lines], y)
    return canvas


class SliderPanel:
    """Mouse-driven sliders for the imaging settings, drawn into the clinical view's left panel.

    Click or drag on a slider's bar to set its value. `values` holds the current settings
    (same keys as get_controls); the demo applies them to the simulator every frame.
    """
    ROW_H = 44                  # label line + bar

    def __init__(self, values, depth_mm):
        # key: (label, min, max, step, format)
        self.specs = {
            "focus": ("Focus", 5.0, depth_mm - 5.0, 0.5, lambda v: f"{v / 10:.2f} cm"),
            "gain": ("Gain", *CONTROL_RANGES["gain"], 1.0, lambda v: f"{v:+.0f} dB"),
            "dr": ("DR", *CONTROL_RANGES["dr"], 1.0, lambda v: f"{v:.0f} dB"),
            "pers": ("Pers", *CONTROL_RANGES["pers"], 0.05, lambda v: f"{v:.2f}"),
            "tgc": ("TGC", *CONTROL_RANGES["tgc"], 0.05, lambda v: f"{v:.2f} dB/cm/MHz"),
        }
        self.values = dict(values)
        self.bars = {}          # key -> (x0, x1, y) of the bar, from the last draw
        self.active = None      # key of the slider being dragged

    def draw(self, canvas, x, y, width):
        """Draw all sliders starting at (x, y); returns the y below the last one."""
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        for key, (label, lo, hi, _step, fmt) in self.specs.items():
            v = self.values[key]
            colour = (235, 235, 235) if key == self.active else (170, 170, 170)
            cv2.putText(canvas, label, (x, y + 14), font, 0.5, colour, 1, cv2.LINE_AA)
            cv2.putText(canvas, fmt(v), (x + 65, y + 14), font, 0.5, colour, 1, cv2.LINE_AA)
            by = y + 30
            x0, x1 = x + 6, x + width - 6
            kx = int(round(x0 + (v - lo) / (hi - lo) * (x1 - x0)))
            cv2.line(canvas, (x0, by), (x1, by), (80, 80, 80), 3, cv2.LINE_AA)
            cv2.line(canvas, (x0, by), (kx, by), (0, 200, 255), 3, cv2.LINE_AA)
            cv2.circle(canvas, (kx, by), 7, (255, 255, 255) if key == self.active else (210, 210, 210), -1,
                       cv2.LINE_AA)
            self.bars[key] = (x0, x1, by)
            y += self.ROW_H
        return y

    def _set_from_x(self, key, mx):
        _label, lo, hi, step, _fmt = self.specs[key]
        x0, x1, _ = self.bars[key]
        v = lo + np.clip((mx - x0) / (x1 - x0), 0.0, 1.0) * (hi - lo)
        self.values[key] = float(np.clip(round(v / step) * step, lo, hi))

    def on_mouse(self, event, mx, my):
        """Feed mouse events from the window; returns True if a slider handled the event."""
        import cv2
        if event == cv2.EVENT_LBUTTONDOWN:
            for key, (x0, x1, by) in self.bars.items():
                if x0 - 10 <= mx <= x1 + 10 and abs(my - by) <= 12:
                    self.active = key
                    self._set_from_x(key, mx)
                    return True
        elif event == cv2.EVENT_MOUSEMOVE and self.active is not None:
            self._set_from_x(self.active, mx)
            return True
        elif event == cv2.EVENT_LBUTTONUP and self.active is not None:
            self.active = None
            return True
        return False


# --------------------------------------------------------------------------------------
# Top-down map of the anatomy (for the demo / for checking registration)
# --------------------------------------------------------------------------------------
# Top-view depth cues: each structure is shaded by its own depth and seen through a haze of the liver / fat above
# it, so structures under the liver look red-veiled and deep ones fade. Shared with draw_depth_legend.
TV_LIVER_BGR = np.float32([70, 60, 120])       # liver haze / background
TV_FAT_BGR = np.float32([90, 175, 205])        # fat haze / background
TV_LIVER_MM, TV_LIVER_MAX = 12.0, 0.65         # haze = MAX * (1 - exp(-thickness / MM))
TV_FAT_MM, TV_FAT_MAX = 25.0, 0.35


def _tv_shade(z, depth_mm):
    return 0.25 + 0.75 * (1.0 - z / depth_mm)


def _tv_haze(colour, liver_mm, fat_mm):
    """Blend colour(s) (..., 3) behind fat_mm of fat and liver_mm of liver."""
    a_fat = (TV_FAT_MAX * (1 - np.exp(-fat_mm / TV_FAT_MM)))[..., None]
    a_liv = (TV_LIVER_MAX * (1 - np.exp(-liver_mm / TV_LIVER_MM)))[..., None]
    colour = colour * (1 - a_fat) + TV_FAT_BGR * a_fat
    return colour * (1 - a_liv) + TV_LIVER_BGR * a_liv


def top_view(anatomy, px_mm=0.25, depth_mm=50.0, dz=0.5, palette=COL_TAB):
    """Depth-coded top view (BGR). Each pixel shows the shallowest structure below it, darker the deeper it lies
    and veiled by the tissue above it: red haze for liver, light yellow haze for fat. Where nothing lies below,
    the background is fat, or liver shaded darker red the thicker the liver is (showing its thin inferior edge).
    palette: BGR colour per tissue label for the structures (the figures pass their own)."""
    xs = np.arange(0, anatomy.size_xy[0], px_mm, dtype=np.float32)
    ys = np.arange(0, anatomy.size_xy[1], px_mm, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    top = np.full(X.shape, -1, np.int16)
    z_top = np.zeros(X.shape, np.float32)
    liver_mm = np.zeros(X.shape, np.float32)      # liver / fat thickness above the top structure (or in total)
    fat_mm = np.zeros(X.shape, np.float32)
    for z in np.arange(dz / 2, depth_mm, dz):
        P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z, np.float32)], 1)
        lab = anatomy.labels(P).reshape(X.shape)
        free = top < 0
        liver_mm[free & (lab == 0)] += dz
        fat_mm[free & (lab == 1)] += dz
        hit = free & (lab >= 2)
        top[hit], z_top[hit] = lab[hit], z
    img = np.empty(X.shape + (3,), np.float32)
    s = top >= 0
    base = np.asarray(palette, np.float32)[top[s]] * _tv_shade(z_top[s], depth_mm)[:, None]
    img[s] = _tv_haze(base, liver_mm[s], fat_mm[s])
    a = (1 - np.exp(-liver_mm[~s] / 15.0))[:, None]                    # background: fat -> thick liver
    img[~s] = TV_FAT_BGR * 0.55 * (1 - a) + TV_LIVER_BGR * (0.85 - 0.45 * a) * a
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_depth_legend(img, depth_mm=50.0, colour_lab=5):
    """Key in the bottom-right corner: a duct shaded by depth, as seen through fat (top bar) and through liver
    (bottom bar), assuming the tissue above it is all fat or all liver."""
    import cv2
    w, h, x0, y0 = 150, 10, img.shape[1] - 172, img.shape[0] - 92
    z = np.linspace(0, depth_mm, w, dtype=np.float32)
    base = COL_TAB[colour_lab] * _tv_shade(z, depth_mm)[:, None]
    bars = [("under fat", _tv_haze(base, np.zeros_like(z), z)), ("under liver", _tv_haze(base, z, np.zeros_like(z)))]
    cv2.rectangle(img, (x0 - 8, y0 - 26), (x0 + w + 12, y0 + 2 * (h + 16) + 16), (20, 20, 20), -1)
    cv2.putText(img, "depth of top structure", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1,
                cv2.LINE_AA)
    for i, (name, bar) in enumerate(bars):
        y = y0 + i * (h + 16)
        img[y:y + h, x0:x0 + w] = np.clip(bar, 0, 255).astype(np.uint8)[None]
        cv2.putText(img, name, (x0, y + h + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1, cv2.LINE_AA)
    y = y0 + 2 * (h + 16) + 8
    for d in (0, 15, 30):
        x = x0 + int(d / depth_mm * (w - 1))
        cv2.putText(img, f"{d}", (x - 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "mm", (x0 + w - 14, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def draw_calot_triangle(img, anatomy, px_per_mm, colour=(255, 255, 255)):
    """Outline and lightly fill Calot's triangle (anatomy.calot_triangle) on a top view, in place."""
    import cv2
    poly = np.round(calot_triangle(anatomy) * px_per_mm).astype(np.int32)
    fill = img.copy()
    cv2.fillPoly(fill, [poly], colour, cv2.LINE_AA)
    cv2.addWeighted(fill, 0.25, img, 0.75, 0, dst=img)
    cv2.polylines(img, [poly], True, colour, 1, cv2.LINE_AA)
    x, y = poly[:, 0].min(), poly[:, 1].min()          # label above-left of the triangle, clear of the ducts
    cv2.putText(img, "Calot's triangle", (max(int(x) - 150, 4), int(y) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                colour, 1, cv2.LINE_AA)
    return img


# Ranges of the live imaging controls (focus is limited to the image depth instead).
CONTROL_RANGES = {"gain": (-30.0, 30.0), "dr": (30.0, 90.0), "pers": (0.0, 0.9), "tgc": (0.0, 1.5)}


def get_controls(sim):
    return {"gain": sim.gain_db, "dr": sim.dr, "focus": sim.pr.focus_mm, "pers": sim.persistence, "tgc": sim.tgc}


def apply_controls(sim, c):
    """Clamp the settings in c and apply them to the simulator; returns the clamped values."""
    lo, hi = CONTROL_RANGES["gain"]; sim.gain_db = float(np.clip(c["gain"], lo, hi))
    lo, hi = CONTROL_RANGES["dr"]; sim.dr = float(np.clip(c["dr"], lo, hi))
    lo, hi = CONTROL_RANGES["pers"]; sim.persistence = round(float(np.clip(c["pers"], lo, hi)), 2)
    lo, hi = CONTROL_RANGES["tgc"]
    tgc = round(float(np.clip(c["tgc"], lo, hi)), 2)
    if tgc != sim.tgc:
        sim.set_tgc(tgc)
    focus = float(np.clip(c["focus"], 5.0, sim.pr.depth_mm - 5.0))    # keep the focus inside the image
    if focus != sim.pr.focus_mm:
        sim.set_focus(focus)
    return get_controls(sim)


SIM_WINDOW = "Ultrasound Imaging Simulator"
B_IMAGE_H = 900                 # display height (px) of the B-mode image; keeps the window within 1080p screens
MODE_TABS = [("B-MODE", True, True), ("DOPPLER", False, False), ("ELASTOGRAPHY", False, False)]  # (label, active,
#                                                                       enabled); disabled = placeholder (to do)


def left_panel_free_y(sliders, n_status, top_h=56):
    """First free y in clinical_view's left panel below the status lines (mirrors its layout: 4 header lines of
    24 px from top_h + 22, the sliders, then a blank line and the status lines)."""
    y = top_h + 22 + 4 * 24
    y = y - 4 + len(sliders.specs) * SliderPanel.ROW_H + 20
    return y + (n_status + 1) * 24
RIGHT_W = 1020                  # width of the area right of the B-mode display (anatomy map, readouts, camera view)


def camera_scene(tracker):
    """Scene of the camera view: the D405 frame with the tracking overlay, and the tracking state.
    Returns (scene, state); scene is None until the first frame. Replace this (demo's scene_source) to show a
    rendered phantom instead of the camera. Zoomed onto the board by tracking.viewer.ZOOM (z toggles)."""
    from orvue_us_inverse.tracking.viewer import ZOOM, annotate
    frame, det, s = tracker.get_frame()
    if frame is None:
        return None, s
    return ZOOM.apply(annotate(frame, det, s, tracker), s, tracker.K, tracker.dist), s


def simulator_keys(tracker):
    keys = [("n / p", "next / previous case"), ("q / e", "rotate"), ("g", "contours"), ("c", "contact"),
            ("click map", "freeze / release probe"), ("v", "3D viewer")]
    if tracker is not None:
        keys = [("m", "camera / mouse"), ("t", "camera view"), ("z", "fit board / full")] + keys
    return keys + [("Esc", "quit")]


def draw_image_orientation(img, x0, y0, width, yaw):
    """Patient directions at the B-mode image's top corners (image left = probe -x end = screen dot)."""
    from orvue_us_inverse.ui import clinical as ui
    left, right = ui.image_sides(yaw)
    ui.text(img, left, (x0 + 18, y0 - 4), ui.AMBER, 0.42)
    ui.text(img, right, (x0 + width - ui.text_w(right, 0.42) - 6, y0 - 4), ui.AMBER, 0.42)


def simulator_window(bmode, view, probe_px, anatomy, yaw, frozen, keys, camera=None, track_state=None,
                     tracking=None, zoom_sliders=None, modes_y=None, viewer_busy=False):
    """The whole simulator in one window.

    Left: the B-mode display (clinical_view output, unchanged) with the image's patient directions.
    Right: the anatomy map (seen from anterior; probe line, amber dot = image-left end, patient directions),
    status chips, image orientation, tracked probe and tracking quality (with the camera view), case description.
    camera: True / False = camera / mouse control (None: no tracker). track_state: the tracker's TrackState.
    With tracking=(scene, sim_pose, footprint_mm) a camera view row is added: scene + phantom map, and
    zoom_sliders (ui.clinical.Sliders for the camera view's zoom / pan) are drawn in the readout column.
    modes_y: y of the IMAGING MODE tabs (MODE_TABS) and the TOOLS buttons in the B-mode display's left panel, below
    its status lines. Returns (image, map rect, buttons {name: (x, y, w, h)}).
    Bottom: key bar. probe_px = ((x, y) of the probe's -x end, (x, y) of its +x end) on the 600 px map `view`.
    Returns (image, (map_x, map_y, map_size)) for the mouse.
    """
    import cv2
    from orvue_us_inverse.ui import clinical as ui
    g, top, cap = 14, 56, 26                             # gutter, clinical_view header height, caption height
    keys_h = 36
    bh, bw = bmode.shape[:2]
    W, H = bw + RIGHT_W, bh + keys_h
    img = np.zeros((H, W, 3), np.uint8)
    img[:bh, :bw] = bmode
    img[:top, bw - 200:bw] = 0                           # the clock moves to the right end of the header
    draw_image_orientation(img, 200, top, bw - 270, yaw)          # clinical_view: panel_w 200, scale_w 70
    ui.text(img, "ULTRASOUND IMAGING SIMULATOR", (bw + g, top // 2 + 6), ui.WHITE, 0.55)
    buttons = {}
    if modes_y is not None:                              # imaging modes and tools in the left panel, below the status
        ui.section(img, 10, modes_y, "IMAGING MODE")
        y_tools = ui.mode_tabs(img, 10, modes_y + 10, 176, MODE_TABS) + 14
        ui.section(img, 10, y_tools, "TOOLS")
        buttons["viewer"] = ui.button(img, (10, y_tools + 10, 176, 30), "3D anatomy viewer", "v", busy=viewer_busy)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    ui.text(img, stamp, (W - ui.text_w(stamp) - 10, top // 2 + 6), ui.GREY, 0.5)

    # anatomy map
    ms = 470 if tracking is not None else 600
    ox, oy = bw + g, top + cap
    ui.panel(img, (ox, oy, ms, ms), f"ANATOMY MAP  |  {anatomy.name}  |  view from anterior")
    m = cv2.resize(view, (ms, ms), interpolation=cv2.INTER_AREA) if ms != view.shape[0] else view.copy()
    k = ms / view.shape[0]
    a, b = (np.asarray(p, float) * k for p in probe_px)
    colour = (0, 140, 255) if frozen else (0, 255, 255)
    cv2.line(m, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), colour, 3 if frozen else 2,
             cv2.LINE_AA)
    cv2.circle(m, tuple(np.round(a).astype(int)), 6, ui.AMBER, -1, cv2.LINE_AA)      # image-left end
    cv2.circle(m, tuple(np.round(a).astype(int)), 6, (0, 0, 0), 1, cv2.LINE_AA)
    ui.draw_patient_directions(m)
    img[oy:oy + ms, ox:ox + ms] = m

    # readouts right of the map
    x = ox + ms + g + 4
    col_w = W - x - g
    y = ui.section(img, x, oy + 4, "STATUS")
    chips = []
    if camera is not None:
        chips.append(("CAMERA CONTROL", ui.GREEN) if camera else ("MOUSE CONTROL", ui.AMBER))
        chips.append(ui.tracking_status(track_state, camera))
    chips.append(("PROBE FROZEN", ui.AMBER) if frozen else ("PROBE LIVE", ui.GREY))
    y = ui.chip_rows(img, x, y, chips) + 8
    left, right = ui.image_sides(yaw)
    y = ui.section(img, x, y, "IMAGE ORIENTATION")
    y = ui.rows(img, x, y, [("Left", f"{left}  (dot)"), ("Right", right), ("Top", "ANT  (skin)"),
                            ("Bottom", "POST  (deep)")], value_x=70) + 8
    if tracking is not None:                             # tracked probe and quality in a second column
        if zoom_sliders is not None:                     # camera view zoom / pan under the orientation
            y = ui.section(img, x, y, "CAMERA ZOOM  (z: board / full)")
            y = zoom_sliders.draw(img, x, y - 6, col_w // 2 - 10) + 14
        x2 = x + col_w // 2 + 6
        y2 = ui.probe_face_rows(img, x2, oy + 4, track_state, value_x=60) + 8
        y2 = ui.quality_rows(img, x2, y2, track_state, value_x=60) + 8
        y = max(y, y2)
    y = ui.section(img, x, y, "CASE")
    for line in ui.wrap(anatomy.description, col_w, 0.46):
        if y > oy + ms:
            break
        ui.text(img, line, (x, y), ui.GREY, 0.46)
        y += 20

    if tracking is None:                                 # map legend in the space the camera view would use
        lx, ly = ox, oy + ms + 40
        ly = ui.section(img, lx, ly, "MAP LEGEND")
        items = [("dot", ui.AMBER, "probe end shown on the image's left (same as the screen dot)"),
                 ("line", (0, 255, 255), "probe face (transducer array), live"),
                 ("line", (0, 140, 255), "probe face, frozen (click the map to release)"),
                 ("box", (255, 255, 255), "Calot's triangle (projected on the surface)")]
        for kind, c, label in items:
            cy = ly - 5
            if kind == "dot":
                cv2.circle(img, (lx + 12, cy), 6, c, -1, cv2.LINE_AA)
            elif kind == "line":
                cv2.line(img, (lx, cy), (lx + 24, cy), c, 2, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (lx + 4, cy - 6), (lx + 20, cy + 6), c, 1, cv2.LINE_AA)
            ui.text(img, label, (lx + 36, ly), ui.GREY, 0.46)
            ly += 24
        ui.text(img, "Map seen from anterior: patient's left on the right, cranial at the top. Colours show the "
                "depth of", (lx, ly + 8), ui.DIM, 0.42)
        ui.text(img, "the top structure (legend in the map).", (lx, ly + 28), ui.DIM, 0.42)

    # camera view row
    if tracking is not None:
        scene, sim_pose, footprint = tracking
        sy = oy + ms + g + cap
        sh = bh - sy - g
        pm = sh
        sw = RIGHT_W - 3 * g - pm
        r_scene = (bw + g, sy, sw, sh)
        from orvue_us_inverse.tracking.viewer import ZOOM
        ui.panel(img, r_scene, f"CAMERA VIEW  |  D405  |  {ZOOM.label()}  (z)")
        if scene is not None:
            ui.fit(scene, r_scene, img)
        else:
            ui.text(img, "NO CAMERA FRAME", (bw + g + sw // 2 - 70, sy + sh // 2), ui.RED)
        r_map = (bw + g + sw + g, sy, pm, pm)
        ui.panel(img, r_map, "PHANTOM MAP  |  tracked probe")
        img[sy:sy + pm, r_map[0]:r_map[0] + pm] = ui.phantom_map(track_state, pm, footprint_mm=footprint,
                                                                 sim_pose=sim_pose, camera=bool(camera))

    ui.key_bar(img, keys, H - keys_h)
    return img, (ox, oy, ms), buttons



def demo(case="normal", tracker=None, cam_view=False, camera_control=True, scene_source=camera_scene):
    """Interactive demo in one "Ultrasound Imaging Simulator" window, B-mode tab (the Doppler tab is a placeholder):
    B-mode display, anatomy map, readouts, key bar and, with a tracker, an optional camera view row.

    tracker: optional tracking.tracker.ProbeTracker. In camera control it sets x, y and yaw every frame;
        mouse and keyboard take over when it loses the probe, in mouse control (m), or while the probe is
        frozen (click on the map). The camera keeps tracking in mouse control, so switching back is immediate.
    cam_view: show the camera view row (scene + phantom map + tracked probe readouts) at start; t toggles it.
    camera_control: start in camera control (True) or mouse control (False).
    scene_source: tracker -> (scene image, state) for the camera view (default: the D405 frame with overlay).
    """
    import cv2
    from orvue_us_inverse.ui import clinical as ui
    names = list(CASES)
    state = {"x": 55.0, "y": 70.0, "yaw": 0.0, "gt": False, "contact": True, "case": names.index(case),
             "locked": False, "cam_view": bool(cam_view and tracker is not None), "camera": bool(camera_control),
             "map_rect": None, "buttons": {}, "viewer_thread": None}
    controls = None             # imaging settings, kept when switching case

    def load(i):
        an = build_case(names[i])
        sim = BModeSimulator(an)
        if controls is not None:
            apply_controls(sim, controls)
        tv = top_view(an)
        sc = 600 / tv.shape[0]
        tv = cv2.resize(tv, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
        draw_calot_triangle(tv, an, sc / 0.25)
        draw_depth_legend(tv)
        print(f"[{names[i]}] {an.description}")
        return an, sim, tv, sc

    an, sim, tv, scale_tv = load(state["case"])
    controls = get_controls(sim)
    sliders = SliderPanel(controls, sim.pr.depth_mm)
    keys = simulator_keys(tracker)
    zoom = zoom_sliders = None
    if tracker is not None:                                  # camera view zoom / pan sliders
        from orvue_us_inverse.tracking.viewer import ZOOM as zoom
        zoom_sliders = ui.Sliders([
            ("zoom", "Zoom", 1.0, zoom.MAX_ZOOM, 0.1, lambda v: f"{v:.1f} x"),
            ("cx", "Pan x", 0.0, 1.0, 0.01, lambda v: f"{(v - 0.5) * 200:+.0f} %"),
            ("cy", "Pan y", 0.0, 1.0, 0.01, lambda v: f"{(v - 0.5) * 200:+.0f} %")])

    def open_viewer():
        """Open the 3D anatomy viewer (browser) on the current case with the probe plane at the probe's pose.
        Runs in the background: the first build of the page takes a few seconds; later it opens at once."""
        import threading
        if state["viewer_thread"] is not None and state["viewer_thread"].is_alive():
            return
        hash_state = (f"case={an.name}&probe=1&px={state['x']:.1f}&py={state['y']:.1f}"
                      f"&pyaw={state['yaw']:.0f}")

        def run():
            try:
                from orvue_us_inverse.viewer3d.viewer import open_page
                open_page(hash_state)
                print(f"[INFO] 3D anatomy viewer opened ({an.name})")
            except Exception as e:                   # never stop the simulator for the viewer
                print(f"[WARN] could not open the 3D anatomy viewer: {e}")

        state["viewer_thread"] = threading.Thread(target=run, daemon=True)
        state["viewer_thread"].start()

    def on_mouse(ev, mx, my, flags, _):
        if sliders.on_mouse(ev, mx, my):                     # left panel sliders (also while dragging)
            return
        if zoom_sliders is not None and state["cam_view"] and zoom_sliders.on_mouse(ev, mx, my):
            zoom.zoom, zoom.cx, zoom.cy = (zoom_sliders.values[k] for k in ("zoom", "cx", "cy"))
            return
        if ev == cv2.EVENT_LBUTTONDOWN:                      # TOOLS buttons in the left panel
            bx = state["buttons"].get("viewer")
            if bx and bx[0] <= mx <= bx[0] + bx[2] and bx[1] <= my <= bx[1] + bx[3]:
                open_viewer()
                return
        if state["map_rect"] is None:
            return
        ox, oy, ms = state["map_rect"]
        if not (0 <= mx - ox < ms and 0 <= my - oy < ms):
            return
        px, py = (mx - ox) * tv.shape[1] / ms, (my - oy) * tv.shape[0] / ms      # on the 600 px map
        # left-click on the map freezes / releases the probe; the click point becomes the probe position
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["locked"] = not state["locked"]
        elif state["locked"]:
            return
        state["x"], state["y"] = px / scale_tv * 0.25, py / scale_tv * 0.25

    cv2.namedWindow(SIM_WINDOW)
    cv2.setMouseCallback(SIM_WINDOW, on_mouse)
    t_last = time.time()
    while True:
        if sliders.values != controls:
            controls = apply_controls(sim, sliders.values)
            if sliders.active is None:
                sliders.values = dict(controls)
        use_camera = tracker is not None and state["camera"] and not state["locked"]
        tracked = tracker.get_xy_yaw() if use_camera else None
        if tracked is not None:
            state["x"], state["y"], state["yaw"] = tracked
        T = sim.pose_from_xy_yaw(state["x"], state["y"], state["yaw"], 0.0 if state["contact"] else 5.0)
        o, u = T[:3, 3], T[:3, 0]
        probe_px = ((o[:2] - u[:2] * sim.pr.width_mm / 2) / 0.25 * scale_tv,       # -x end (image left)
                    (o[:2] + u[:2] * sim.pr.width_mm / 2) / 0.25 * scale_tv)

        if sim.in_contact(T):
            img, lab = sim.render(T, return_labels=True)
            us = sim.ground_truth_overlay(img, lab) if state["gt"] else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            us = np.zeros((sim.nz, sim.nx, 3), np.uint8)
            cv2.putText(us, "no contact", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        us = cv2.resize(us, (int(round(B_IMAGE_H * sim.nx / sim.nz)), B_IMAGE_H))   # fixed height, true aspect
        t = time.time(); fps = 1 / max(t - t_last, 1e-6); t_last = t
        if tracker is None:
            track = "off"
        elif not state["camera"]:
            track = "mouse"
        else:
            track = "on" if tracked is not None else "lost"
        status = [f"FR     {fps:4.1f} Hz", "", f"Probe  {'frozen' if state['locked'] else 'live'}",
                  f"Track  {track}",
                  f"x      {state['x']:.1f} mm", f"y      {state['y']:.1f} mm", f"yaw    {state['yaw']:.0f} deg",
                  "", f"Case   {an.name}", f"GT     {'on' if state['gt'] else 'off'}"]
        tracking, track_state = None, (tracker.get_state() if tracker is not None else None)
        if state["cam_view"]:
            scene, track_state = scene_source(tracker)
            tracking = (scene, (state["x"], state["y"], state["yaw"]), (sim.pr.width_mm, 10.0))
        if zoom_sliders is not None and zoom_sliders.active is None:       # follow z / the first board fit
            zoom_sliders.values.update(zoom=zoom.zoom, cx=zoom.cx, cy=zoom.cy)
        busy = state["viewer_thread"] is not None and state["viewer_thread"].is_alive()
        window, state["map_rect"], state["buttons"] = simulator_window(
            clinical_view(us, sim, status, sliders), tv, probe_px, an, state["yaw"], state["locked"], keys,
            camera=state["camera"] if tracker is not None else None, track_state=track_state, tracking=tracking,
            zoom_sliders=zoom_sliders, modes_y=left_panel_free_y(sliders, len(status)) + 16, viewer_busy=busy)
        cv2.imshow(SIM_WINDOW, window)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: break
        if k == ord("q"): state["yaw"] -= 5
        if k == ord("e"): state["yaw"] += 5
        if k == ord("g"): state["gt"] = not state["gt"]
        if k == ord("c"): state["contact"] = not state["contact"]
        if k == ord("t") and tracker is not None:
            state["cam_view"] = not state["cam_view"]
        if k == ord("z") and tracker is not None:
            zoom.toggle()
        if k == ord("m") and tracker is not None:
            state["camera"] = not state["camera"]
            print(f"[INFO] control: {'camera' if state['camera'] else 'mouse'}")
        if k == ord("v"):
            open_viewer()
        if k in (ord("n"), ord("p")):
            state["case"] = (state["case"] + (1 if k == ord("n") else -1)) % len(names)
            an, sim, tv, scale_tv = load(state["case"])
    cv2.destroyAllWindows()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Interactive B-mode demo (mouse, or D405 camera tracking).")
    ap.add_argument("case", nargs="?", default="normal", choices=list(CASES), help="anatomy case")
    ap.add_argument("--track", action="store_true", help="drive the probe from the D405 tracker")
    ap.add_argument("--cam-view", action="store_true", help="show the camera view at start (implies --track)")
    ap.add_argument("--mouse", action="store_true", help="start in mouse control (implies --track; m switches)")
    a = ap.parse_args(argv)
    tracker = None
    if a.track or a.cam_view or a.mouse:
        try:
            from orvue_us_inverse.tracking.tracker import ProbeTracker
            tracker = ProbeTracker("realsense")
            print("[INFO] t = show / hide the camera view, m = switch camera / mouse control")
        except Exception as e:                       # no camera (or no pyrealsense2): keep the mouse demo usable
            print(f"[WARN] camera tracking unavailable ({e}); running with mouse control only")
    try:
        demo(a.case, tracker, cam_view=a.cam_view, camera_control=not a.mouse)
    finally:
        if tracker is not None:
            tracker.stop()


if __name__ == "__main__":
    main()
