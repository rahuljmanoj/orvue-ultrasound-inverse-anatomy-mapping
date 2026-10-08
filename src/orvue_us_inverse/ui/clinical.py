"""
orvue_us_inverse.ui.clinical - shared clinical-style drawing for the simulator and tracking windows (Orvue Surgical).

Same look as the B-mode window (simulation.bmode.clinical_view): black background, logo header, amber section
captions, white values, grey labels. Pure drawing on numpy BGR images; no windows are opened here.

    compose_tracking(scene, state, ...)   full "Probe tracking" window: scene panel, phantom map, readouts
    phantom_map(state, size, ...)         100 x 100 mm top-down map with the tracked (and virtual) probe
    header / panel / chip / rows          building blocks

The scene panel takes any BGR image: today the annotated D405 camera frame, later a rendered phantom.
"""
import math
import time
from functools import lru_cache

import cv2
import numpy as np

from orvue_us_inverse.paths import LOGO_PATH  # noqa: E402
COMPANY = "Orvue Surgical"

# palette (BGR), matching clinical_view
BG = (0, 0, 0)
PANEL = (16, 16, 18)
PANEL_LIGHT = (26, 27, 30)
BORDER = (55, 55, 60)
WHITE = (235, 235, 235)
GREY = (170, 170, 170)
DIM = (110, 110, 115)
AMBER = (0, 200, 255)
GREEN = (90, 205, 110)
RED = (70, 70, 230)
FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_H = 56
LOGO_H = 40


@lru_cache(maxsize=4)
def logo(height=LOGO_H):
    img = cv2.imread(LOGO_PATH, cv2.IMREAD_COLOR)
    if img is None:
        return None
    w = max(1, int(round(img.shape[1] * height / img.shape[0])))
    return cv2.resize(img, (w, height), interpolation=cv2.INTER_AREA)


def text(img, s, org, colour=WHITE, scale=0.5, thick=1):
    cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, colour, thick, cv2.LINE_AA)


def text_w(s, scale=0.5, thick=1):
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def header(img, title, subtitle="", height=HEADER_H):
    """Top bar: logo + company on the left, window title (and subtitle) after it, clock on the right."""
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, height - 1), BG, -1)
    y = height // 2 + 6
    x = 10
    lg = logo()
    if lg is not None:
        ly = (height - lg.shape[0]) // 2
        img[ly:ly + lg.shape[0], x:x + lg.shape[1]] = lg
        x += lg.shape[1] + 10
    text(img, COMPANY, (x, y), WHITE, 0.5)
    x += text_w(COMPANY) + 28
    text(img, title, (x, y), WHITE, 0.55)
    if subtitle:
        text(img, subtitle, (x + text_w(title, 0.55) + 16, y), GREY, 0.5)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    text(img, stamp, (img.shape[1] - text_w(stamp) - 10, y), GREY, 0.5)
    cv2.line(img, (0, height - 1), (img.shape[1] - 1, height - 1), BORDER, 1)


def button(img, rect, label, key="", busy=False):
    """Clickable button (x, y, w, h); key: keyboard shortcut shown on the right. busy: dimmed with '...'."""
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL_LIGHT, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), DIM if busy else AMBER, 1)
    text(img, label + ("..." if busy else ""), (x + 12, y + h // 2 + 6), GREY if busy else WHITE, 0.48)
    if key:
        kw = text_w(key, 0.42) + 10
        cv2.rectangle(img, (x + w - kw - 6, y + 6), (x + w - 6, y + h - 6), BORDER, 1)
        text(img, key, (x + w - kw - 1, y + h // 2 + 5), GREY, 0.42)
    return rect


def mode_tabs(img, x, y, width, items, height=30, gap=8):
    """Vertical list of mode tabs from (x, y). items: (label, active, enabled). The active tab is amber with an
    accent bar on its left; disabled tabs are dim and tagged "TO DO". Returns the y below the last tab."""
    for label, active, enabled in items:
        if active:
            cv2.rectangle(img, (x, y), (x + width, y + height), PANEL_LIGHT, -1)
            cv2.rectangle(img, (x, y), (x + width, y + height), AMBER, 1)
            cv2.rectangle(img, (x, y), (x + 3, y + height), AMBER, -1)
        else:
            cv2.rectangle(img, (x, y), (x + width, y + height), BORDER, 1)
        colour = AMBER if active else (GREY if enabled else DIM)
        text(img, label, (x + 12, y + height // 2 + 6), colour, 0.48)
        if not enabled:
            tag = "TO DO"
            text(img, tag, (x + width - text_w(tag, 0.38) - 8, y + height // 2 + 5), DIM, 0.38)
        y += height + gap
    return y


def panel(img, rect, caption=""):
    """Dark panel with a thin border and an amber caption above it; rect = (x, y, w, h) of the inside."""
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), PANEL, -1)
    cv2.rectangle(img, (x - 1, y - 1), (x + w, y + h), BORDER, 1)
    if caption:
        text(img, caption, (x, y - 8), AMBER, 0.45)


def fit(img, rect, canvas):
    """Paste img into rect of canvas, scaled to fit and centred (letterboxed)."""
    x, y, w, h = rect
    s = min(w / img.shape[1], h / img.shape[0])
    iw, ih = max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))
    small = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA)
    ox, oy = x + (w - iw) // 2, y + (h - ih) // 2
    canvas[oy:oy + ih, ox:ox + iw] = small


def chip(img, org, label, colour, scale=0.5):
    """Status chip: coloured dot and label in a rounded-looking dark box; org = left baseline. Returns width."""
    x, y = int(org[0]), int(org[1])
    w = text_w(label, scale, 1) + 30
    cv2.rectangle(img, (x, y - 16), (x + w, y + 7), PANEL_LIGHT, -1)
    cv2.rectangle(img, (x, y - 16), (x + w, y + 7), colour, 1)
    cv2.circle(img, (x + 11, y - 4), 4, colour, -1, cv2.LINE_AA)
    text(img, label, (x + 21, y), colour, scale)
    return w


def rows(img, x, y, items, value_x=95, step=22):
    """'name   value' rows like the B-mode side panel; items = (name, value[, colour]). Returns the next y."""
    for it in items:
        name, value = it[0], it[1]
        colour = it[2] if len(it) > 2 else WHITE
        text(img, name, (x, y), GREY, 0.48)
        text(img, value, (x + value_x, y), colour, 0.48)
        y += step
    return y


def section(img, x, y, title):
    text(img, title, (x, y), AMBER, 0.45)
    return y + 22


def fmt(v, spec, unit=""):
    return "---" if v is None else f"{format(v, spec)}{unit}"


def wrap(s, width_px, scale=0.5):
    """Split text into lines no wider than width_px."""
    lines, line = [], ""
    for word in s.split():
        trial = f"{line} {word}".strip()
        if line and text_w(trial, scale) > width_px:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])


def key_bar(img, keys, y, x=14):
    """Key bar from y down (about 34 px): a separator line, then [key] action pairs."""
    cv2.line(img, (0, y), (img.shape[1] - 1, y), BORDER, 1)
    for k, action in keys:
        kw = text_w(k, 0.45) + 14
        cv2.rectangle(img, (x, y + 8), (x + kw, y + 28), PANEL_LIGHT, -1)
        cv2.rectangle(img, (x, y + 8), (x + kw, y + 28), BORDER, 1)
        text(img, k, (x + 7, y + 23), WHITE, 0.45)
        text(img, action, (x + kw + 8, y + 23), GREY, 0.45)
        x += kw + 8 + text_w(action, 0.45) + 24


# ---------------------------------------------------------------- sliders
class Sliders:
    """Mouse-driven sliders in the clinical style (like the B-mode window's imaging sliders).

    specs: list of (key, label, lo, hi, step, fmt). draw() at absolute window coordinates, then feed the window's
    mouse events to on_mouse(); it returns True when a slider took the event (click or drag)."""
    ROW_H = 40

    def __init__(self, specs, values=None):
        self.specs = {k: (label, lo, hi, step, fmt) for k, label, lo, hi, step, fmt in specs}
        self.values = {k: (values or {}).get(k, lo) for k, (_, lo, _, _, _) in self.specs.items()}
        self.bars, self.active = {}, None

    def draw(self, img, x, y, width):
        for key, (label, lo, hi, _step, fmt) in self.specs.items():
            v = self.values[key]
            colour = WHITE if key == self.active else GREY
            text(img, label, (x, y + 14), colour, 0.48)
            text(img, fmt(v), (x + 70, y + 14), colour, 0.48)
            by, x0, x1 = y + 28, x + 6, x + width - 6
            kx = int(round(x0 + (v - lo) / (hi - lo) * (x1 - x0)))
            cv2.line(img, (x0, by), (x1, by), (80, 80, 80), 3, cv2.LINE_AA)
            cv2.line(img, (x0, by), (kx, by), AMBER, 3, cv2.LINE_AA)
            cv2.circle(img, (kx, by), 7, (255, 255, 255) if key == self.active else (210, 210, 210), -1, cv2.LINE_AA)
            self.bars[key] = (x0, x1, by)
            y += self.ROW_H
        return y

    def _set(self, key, mx):
        _label, lo, hi, step, _fmt = self.specs[key]
        x0, x1, _ = self.bars[key]
        v = lo + np.clip((mx - x0) / (x1 - x0), 0.0, 1.0) * (hi - lo)
        self.values[key] = float(np.clip(round(v / step) * step, lo, hi))

    def on_mouse(self, event, mx, my):
        if event == cv2.EVENT_LBUTTONDOWN:
            for key, (x0, x1, by) in self.bars.items():
                if x0 - 10 <= mx <= x1 + 10 and abs(my - by) <= 12:
                    self.active = key
                    self._set(key, mx)
                    return True
        elif event == cv2.EVENT_MOUSEMOVE and self.active is not None:
            self._set(self.active, mx)
            return True
        elif event == cv2.EVENT_LBUTTONUP and self.active is not None:
            self.active = None
            return True
        return False


# ---------------------------------------------------------------- patient directions
# Phantom frame -> patient: +x = patient's left (medial), +y = caudal, +z = posterior (depth). The maps are seen
# from anterior (from the probe / camera side), so patient left is on the right, cranial at the top.
def direction_name(dx, dy):
    """Patient direction of an in-plane phantom-frame vector: 'MED', 'CRAN-LAT', ..."""
    horiz = "MED" if dx > 0 else "LAT"
    vert = "CAUD" if dy > 0 else "CRAN"
    if abs(dx) >= 2.0 * abs(dy):
        return horiz
    if abs(dy) >= 2.0 * abs(dx):
        return vert
    return f"{vert}-{horiz}"


def image_sides(yaw_deg):
    """(left, right) patient directions of the B-mode image for a probe at yaw_deg. The image's left side is the
    probe's -x end (the screen orientation dot)."""
    a = math.radians(yaw_deg)
    ux, uy = math.cos(a), math.sin(a)
    return direction_name(-ux, -uy), direction_name(ux, uy)


def label_box(img, s, centre, colour=WHITE, scale=0.42):
    """Small label on a dark box, centred at `centre`."""
    w = text_w(s, scale)
    x, y = int(centre[0] - w / 2), int(centre[1] + 5)
    cv2.rectangle(img, (x - 5, y - 14), (x + w + 5, y + 5), (10, 10, 12), -1)
    text(img, s, (x, y), colour, scale)


def draw_patient_directions(img):
    """CRANIAL / CAUDAL / LAT (patient R) / MED (patient L) at the edges of a map seen from anterior."""
    h, w = img.shape[:2]
    label_box(img, "CRANIAL", (w / 2, 14))
    label_box(img, "CAUDAL", (w / 2, h - 16))
    label_box(img, "LAT  pt R", (42, h / 2))
    label_box(img, "MED  pt L", (w - 42, h / 2))


# ---------------------------------------------------------------- phantom map
def _probe_outline(x, y, yaw_deg, length, width):
    """Footprint polygon and the probe's -x end (image left, where the screen orientation dot is)."""
    a = math.radians(yaw_deg)
    u = np.array([math.cos(a), math.sin(a)])
    v = np.array([-math.sin(a), math.cos(a)])
    c = np.array([x, y])
    L, W = length / 2, width / 2
    return np.array([c - L * u - W * v, c + L * u - W * v, c + L * u + W * v, c - L * u + W * v]), c - L * u


def phantom_map(state, size=360, region_mm=100.0, footprint_mm=(30.0, 10.0), sim_pose=None, camera=True):
    """Top-down map of the 100 x 100 mm region with the tracked probe footprint (face pose; the amber dot marks
    the image-left end, like the B-mode screen dot). sim_pose=(x, y, yaw): also draw the simulator's virtual
    probe (shown in mouse control)."""
    img = np.full((size, size, 3), PANEL, np.uint8)
    m = 30                                               # margin for the tick labels
    s = (size - 2 * m) / region_mm

    def P(xy):
        return np.round(m + np.asarray(xy) * s).astype(np.int32)

    cv2.rectangle(img, tuple(P((0, 0))), tuple(P((region_mm, region_mm))), PANEL_LIGHT, -1)
    for i in range(0, int(region_mm) + 1, 10):
        c = (52, 53, 58) if i % 50 else (72, 73, 80)
        cv2.line(img, tuple(P((i, 0))), tuple(P((i, region_mm))), c, 1)
        cv2.line(img, tuple(P((0, i))), tuple(P((region_mm, i))), c, 1)
    cv2.rectangle(img, tuple(P((0, 0))), tuple(P((region_mm, region_mm))), (100, 100, 108), 1)
    for i in (0, 50, 100):
        t = str(i)
        px = P((i, 0))
        text(img, t, (px[0] - text_w(t, 0.38) // 2, m - 8), DIM, 0.38)
        py = P((0, i))
        text(img, t, (m - 6 - text_w(t, 0.38), py[1] + 4), DIM, 0.38)

    if sim_pose is not None and not camera:                 # virtual probe driven by the mouse
        poly, end = _probe_outline(*sim_pose, *footprint_mm)
        cv2.polylines(img, [P(poly)], True, AMBER, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(P(end)), 4, AMBER, -1, cv2.LINE_AA)
    if state is not None and state.valid:
        x, y, _ = state.face_filt
        poly, end = _probe_outline(x, y, state.yaw_filt, *footprint_mm)
        fill = img.copy()
        cv2.fillPoly(fill, [P(poly)], (70, 120, 70) if camera else (70, 70, 70))
        img[:] = cv2.addWeighted(fill, 0.55, img, 0.45, 0)
        cv2.polylines(img, [P(poly)], True, GREEN if camera else GREY, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(P(end)), 5, AMBER, -1, cv2.LINE_AA)            # image-left (-x) end
        cx, cy = P((x, y))
        cv2.line(img, (cx - 6, cy), (cx + 6, cy), WHITE, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - 6), (cx, cy + 6), WHITE, 1, cv2.LINE_AA)
    elif state is not None:
        t = "PROBE NOT TRACKED"
        text(img, t, ((size - text_w(t, 0.5)) // 2, size // 2), RED, 0.5)
    return img


# ---------------------------------------------------------------- readout blocks
def chip_rows(img, x, y, chips):
    """One chip per line; chips = (label, colour). Returns the next y."""
    for label, colour in chips:
        chip(img, (x, y + 4), label, colour)
        y += 30
    return y


def probe_face_rows(img, x, y, s, value_x=95):
    v = s is not None and s.valid
    y = section(img, x, y, "PROBE FACE  (tracked)")
    return rows(img, x, y, [
        ("x", fmt(s.face_filt[0] if v else None, ".1f", " mm")),
        ("y", fmt(s.face_filt[1] if v else None, ".1f", " mm")),
        ("z", fmt(s.face_filt[2] if v else None, ".1f", " mm")),
        ("yaw", fmt(s.yaw_filt if v else None, "+.1f", " deg")),
        ("tilt", fmt(s.tilt_deg if v else None, ".1f", " deg")),
    ], value_x=value_x)


def quality_rows(img, x, y, s, value_x=95):
    y = section(img, x, y, "TRACKING QUALITY")
    ref = "---" if s is None else ("held" if s.phantom_held else f"{s.n_reference_markers_used} / 4")
    reproj = "---" if s is None else f"{fmt(s.reproj_error_ref_px, '.2f')} / {fmt(s.reproj_error_probe_px, '.2f')} px"
    return rows(img, x, y, [("Board", ref), ("Reproj", reproj),
                            ("Camera", "---" if s is None else f"{s.fps:.0f} fps")], value_x=value_x)


# ---------------------------------------------------------------- tracking window
def tracking_status(state, camera):
    """(label, colour) of the tracking chip."""
    if state is None:
        return "NO CAMERA", RED
    if state.valid:
        return ("HELD BOARD" if state.phantom_held else "TRACKING"), (AMBER if state.phantom_held else GREEN)
    if not state.phantom_valid:
        return "BOARD NOT SEEN", RED
    return "PROBE NOT SEEN", RED


def compose_tracking(scene, state, title="PROBE TRACKING", subtitle="", camera=None, keys=(), banner=(),
                     footprint_mm=(30.0, 10.0), sim_pose=None, badges=(), scene_caption="SCENE  |  D405 CAMERA",
                     scene_h=360):
    """Full tracking window: header, scene panel (any BGR image), phantom map, readout column, optional
    footer banner (e.g. calibration prompts).

    camera: True / False shows the CAMERA / MOUSE control chip (None: no control chip).
    keys: (key, action) rows for the KEYS section. badges: extra (label, colour) chips, e.g. ("REC", RED).
    """
    g = 14                                               # gutter
    scene_w = int(round(scene_h * 16 / 9))
    map_s = scene_h
    col_w = 250
    banner = list(banner)
    banner_h = 14 + 22 * len(banner) if banner else 0
    keys_h = 30 if keys else 0
    W = g + scene_w + g + map_s + g + col_w + g
    H = HEADER_H + 26 + scene_h + g + (banner_h + g if banner else 0) + keys_h + (6 if keys else 0)
    img = np.full((H, W, 3), BG, np.uint8)
    header(img, title, subtitle)

    top = HEADER_H + 26
    r_scene = (g, top, scene_w, scene_h)
    r_map = (g + scene_w + g, top, map_s, map_s)
    panel(img, r_scene, scene_caption)
    if scene is not None:
        fit(scene, r_scene, img)
    else:
        t = "NO SCENE"
        text(img, t, (g + (scene_w - text_w(t)) // 2, top + scene_h // 2), RED)
    panel(img, r_map, "PHANTOM MAP  |  mm, x right, y down")
    img[top:top + map_s, r_map[0]:r_map[0] + map_s] = phantom_map(state, map_s, footprint_mm=footprint_mm,
                                                                   sim_pose=sim_pose,
                                                                   camera=True if camera is None else camera)

    # readout column
    x = r_map[0] + map_s + g + 4
    y = top + 4
    y = section(img, x, y, "STATUS")
    chips = []
    if camera is not None:
        chips.append(("CAMERA CONTROL", GREEN) if camera else ("MOUSE CONTROL", AMBER))
    chips.append(tracking_status(state, camera))
    y = chip_rows(img, x, y, chips) + 6
    if badges:
        bx = x
        for b_label, b_colour in badges:
            bx += chip(img, (bx, y + 4 - 6), b_label, b_colour) + 8
        y += 30
    y = probe_face_rows(img, x, y, state) + 6
    quality_rows(img, x, y, state)
    by = top + scene_h + g
    if banner:                                           # e.g. calibration prompts
        cv2.rectangle(img, (g, by), (W - g - 1, by + banner_h), PANEL_LIGHT, -1)
        cv2.rectangle(img, (g, by), (W - g - 1, by + banner_h), AMBER, 1)
        for i, line in enumerate(banner):
            text(img, line, (g + 12, by + 24 + 22 * i), AMBER if i == 0 else WHITE, 0.5)
        by += banner_h + g
    if keys:                                             # key bar along the bottom
        key_bar(img, keys, by - 4, g)
    return img
