"""
orvue_us_inverse.mapping.run_mouse - hand-guided sweep with the mouse over the hidden 100 x 100 mm region (S6).

    python -m orvue_us_inverse.mapping.run_mouse [--case normal] [--yaw 0] [--overlap 20] [--spacing 0.25]
                                                  [--no-images]
    python -m orvue_us_inverse mouse [options]

One window, "Mouse sweep":
  left    sweep top view at 4 px/mm (+x right, +y down), anatomy hidden (a reveals it): the mouse position is the probe
          face centre, hold the left button to record. Lane guides of the chosen strategy (--yaw / --overlap, S5
          default) as thin bands for the orientation nearest the probe's angle, the current lane highlighted. Coverage
          map (black = not scanned, dark purple -> yellow = imaged once -> many times), unscanned holes (enclosed by
          scanned area) cyan, speed gaps (consecutive frames > 2 x spacing apart) red lines; colour key underneath.
  middle  the latest captured frame: B-mode, or with B-mode off (button in the caption, key i, start with --no-images)
          the oracle labels in tissue colours with a key. B-mode off computes only labels_image (~13 ms instead of
          ~78 ms per frame), so the capture rate and the speed limit about double; the capture-rate measurement
          restarts at each switch. A sweep can mix frames with and without B-mode (sweep format 2).
  right   the 3D reconstruction (PyVista rendered off-screen into the panel; drag to rotate, wheel to zoom; view
          buttons Iso / Top / Axial / Sagittal or key v; faces labelled patient R / L, cranial / caudal, anterior /
          posterior, orientation triad in the corner; g adds the ground truth translucent). Rebuilt whenever the button is up and something changed (after each stroke,
          undo, reset or g), never during a stroke, so building the surfaces (~0.2 s) cannot cause speed gaps. Without
          PyVista the panel says so; b and 3 still work.
  bottom  status, speed meter (maximum speed = frame spacing x measured capture rate; green below 80 % of it, amber up
          to 100 %, red above; frames are still captured), message, keys.

Probe angle: the mouse wheel over the sweep or q / e turn the probe by 5 degrees, 0 / 9 set 0 / 90 degrees, at any
time (also during a stroke; a frame is captured for every 1 degree of turning).

Keys: left button hold = record; q / e / 0 / 9 angle; i B-mode on / off; c complete (S4 evaluation and report, then the 3D view shows the
evaluated volume); s save the sweep; u undo the last stroke; r reset; a reveal anatomy; g ground truth in the 3D
view; v next 3D view; 3 3D snapshot + STL; b browser 3D view; Esc quit (q turns the probe here).
"""
import argparse
import dataclasses
import os
import time
import webbrowser

import cv2
import numpy as np
from scipy import ndimage

from orvue_us_inverse.mapping.acquisition import Acquirer
from orvue_us_inverse.mapping.config import AcquisitionConfig, GridConfig, SweepConfig
from orvue_us_inverse.mapping.evaluate import complete_report, instance_masks, summary_table
from orvue_us_inverse.mapping.poses import MousePose, ScriptedSweep, axes
from orvue_us_inverse.mapping.probe import FRAME_SHAPE, PROBE, make_simulator, probe_metadata
from orvue_us_inverse.mapping.recon import LabelCompounder, VoxelGrid
from orvue_us_inverse.mapping.render import Recon3DOutputs, SliceView, surface_meshes
from orvue_us_inverse.paths import RESULTS_DIR, SWEEPS_DIR
from orvue_us_inverse.simulation.anatomy import CASES, COL_TAB
from orvue_us_inverse.simulation.bmode import top_view
from orvue_us_inverse.ui import clinical as ui

WIN = "Mouse sweep"
PX_PER_MM = 4.0
MARGIN = 40
TOP_W = int(round(100 * PX_PER_MM)) + 2 * MARGIN       # 480: top-view canvas (square)
GAP = 10
BMODE_W = 288                                           # 501 x 301 frame scaled to the canvas height
VIEW3D_W = 420
STATUS_H = 120
KEYS_H = 68
SPEED_OK = 0.8                     # green below SPEED_OK x maximum speed, amber up to 1.0, red above
GAP_FACTOR = 2.0                   # gap when consecutive frames of a stroke are > GAP_FACTOR x spacing apart
DEFAULT_PERIOD_S = {True: 0.12, False: 0.04}      # capture-loop period before it is measured (images / labels)
SPEED_BGR = {"green": ui.GREEN, "amber": ui.AMBER, "red": ui.RED}
HOLE_BGR = (255, 255, 0)          # cyan: never produced by the coverage heat map (inferno)
GAP_BGR = (40, 40, 255)
VIEW_ORDER = ("isometric", "top", "axial", "sagittal")      # live3d.VIEWS, buttons left to right
VIEW_LABELS = {"isometric": "Iso", "top": "Top", "axial": "Axial", "sagittal": "Sagittal"}
CAPTION_3D_H = 66                 # caption, view buttons and hint above the 3D image
TISSUE_NAMES = ("liver", "fat", "GB wall", "bile", "stone", "duct wall", "artery wall", "arterial blood", "vein wall",
                "venous blood", "lymph node")          # anatomy.TISSUES labels 0-10, for the key of the labels view


def wheel_delta(flags: int) -> int:
    """Signed wheel delta of an OpenCV mouse-wheel event (upper 16 bits of flags; the Python bindings have no
    getMouseWheelDelta). Positive = wheel forward / up."""
    return ((int(flags) >> 16) + 0x8000) % 0x10000 - 0x8000


def speed_class(speed_mm_s: float, max_speed_mm_s: float) -> str:
    """"green" below SPEED_OK x max, "amber" up to max, "red" above."""
    if speed_mm_s < SPEED_OK * max_speed_mm_s:
        return "green"
    return "amber" if speed_mm_s <= max_speed_mm_s else "red"


def find_gaps(frames, strokes: list[tuple[int, int]], spacing_mm: float, factor: float = GAP_FACTOR
              ) -> list[tuple[int, int, float]]:
    """(i, i + 1, distance) for consecutive frames of the same stroke whose face centres are > factor x spacing
    apart. strokes: [start, end) frame index ranges."""
    out = []
    for a, b in strokes:
        for i in range(a, b - 1):
            d = float(np.linalg.norm(frames[i + 1].T_true[:3, 3] - frames[i].T_true[:3, 3]))
            if d > factor * spacing_mm:
                out.append((i, i + 1, d))
    return out


def mouse_filename(case: str, cfg: SweepConfig, store_images: bool, stamp: str | None = None) -> str:
    settings = f"sp{cfg.frame_spacing_mm:g}" + ("" if store_images else "_labels")
    return f"{case}_mouse_{settings}_{stamp or time.strftime('%Y%m%d-%H%M%S')}.npz"


class MouseSession(Recon3DOutputs):
    """Hand-guided sweep state and the single window's drawing and mouse routing (no window is opened here)."""

    # panel rectangles (x, y, w, h) in the window
    RECT_TOP = (0, ui.HEADER_H, TOP_W, TOP_W)
    RECT_BMODE = (TOP_W + GAP, ui.HEADER_H, BMODE_W, TOP_W)
    RECT_3D = (TOP_W + GAP + BMODE_W + GAP, ui.HEADER_H, VIEW3D_W, TOP_W)
    WIDTH = TOP_W + GAP + BMODE_W + GAP + VIEW3D_W + GAP
    HEIGHT = ui.HEADER_H + TOP_W + STATUS_H + KEYS_H

    def __init__(self, case: str, cfg: SweepConfig, acq_cfg: AcquisitionConfig, grid_cfg: GridConfig | None = None,
                 clock=time.perf_counter):
        self.case, self.cfg, self.acq_cfg, self.clock = case, cfg, acq_cfg, clock
        self.sim = make_simulator(case)
        self.plan = ScriptedSweep(cfg)                       # lane guides
        self.grid = VoxelGrid(grid_cfg or GridConfig())
        self.slice_view = SliceView(self.grid)                # no window: error slices for the report; show_gt flag
        self.pose = MousePose(clock=clock)
        self.size = TOP_W
        self.view3d = None                                   # live3d.Embedded3D (init_view3d) or None
        self.view3d_error = ""
        self._orbit = None                                   # last drag position while rotating the 3D view
        self._revealed_img = None
        self.revealed = False
        self._gt = None
        self._gt_meshes = None
        self._instances = None
        self.period_s = None                                 # measured capture-loop period (EMA)
        self._prev_step_t = None
        self._prev_captured = False
        self.message = ""
        self.restart()

    # ---- state
    def restart(self) -> None:
        self.acq = Acquirer(self.sim, self.cfg, self.acq_cfg, mode="mouse")
        self.comp = LabelCompounder(self.grid, probe=probe_metadata())
        self.strokes: list[list[int]] = []                  # [start, end) frame indices
        self.gaps: list[tuple[int, int, float]] = []
        self.latest = None
        self.completed = False
        self.final_labels = None
        self.evaluation = None
        self.slice_view.errors = False
        self._coverage = (-1, None)
        self.observed_pct = 0.0
        self.pose.release()

    @property
    def frames(self):
        return self.acq.sweep.frames

    @property
    def capture_rate(self) -> float:
        """Frames per second the capture loop achieves (measured; a default before the first captures)."""
        return 1.0 / (self.period_s or DEFAULT_PERIOD_S[self.acq_cfg.store_images])

    @property
    def max_speed(self) -> float:
        return self.cfg.frame_spacing_mm * self.capture_rate

    def speed(self) -> tuple[float, str]:
        v = self.pose.speed_mm_s()
        return v, speed_class(v, self.max_speed)

    # ---- probe angle (wheel over the sweep, keys)
    def turn(self, sign: int) -> None:
        self.pose.rotate(sign)

    def set_angle(self, yaw_deg: float) -> None:
        self.pose.snap(yaw_deg)

    # ---- B-mode on / off
    @property
    def bmode_on(self) -> bool:
        return self.acq_cfg.store_images

    def toggle_bmode(self) -> bool:
        """Switch the B-mode rendering for the next frames (the Acquirer shares acq_cfg); restarts the capture-rate
        measurement. Returns the new state."""
        self.acq_cfg.store_images = not self.acq_cfg.store_images
        self.period_s, self._prev_captured = None, False
        self.message = ("B-mode on: realistic images, ~78 ms per frame, lower speed limit" if self.bmode_on else
                        "B-mode off: oracle labels only, ~13 ms per frame, about twice the speed limit")
        return self.bmode_on

    def bmode_button(self) -> tuple[int, int, int, int]:
        """Rect of the B-mode on / off button in the middle panel's caption."""
        x, y, w, _ = self.RECT_BMODE
        return x + w - 92, y + 3, 88, 17

    # ---- 3D view presets
    def view_buttons(self) -> list[tuple[str, tuple[int, int, int, int]]]:
        """(view name, rect in the window) of the preset buttons above the 3D image."""
        x, y, _, _ = self.RECT_3D
        out, bx = [], x + 6
        for name in VIEW_ORDER:
            w = ui.text_w(VIEW_LABELS[name], 0.42) + 18
            out.append((name, (bx, y + 24, w, 20)))
            bx += w + 6
        return out

    def set_view(self, name: str) -> None:
        if self.view3d is not None:
            self.view3d.set_view(name)

    def next_view(self) -> None:
        if self.view3d is not None:
            i = VIEW_ORDER.index(self.view3d.view) if self.view3d.view in VIEW_ORDER else -1
            self.view3d.set_view(VIEW_ORDER[(i + 1) % len(VIEW_ORDER)])

    # ---- mouse
    def mm(self, wx: int, wy: int) -> tuple[float, float]:
        """Window pixel -> phantom mm (the top view sits under the header, at the left)."""
        return (wx - MARGIN) / PX_PER_MM, (wy - ui.HEADER_H - MARGIN) / PX_PER_MM

    def window_px(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        return int(round(MARGIN + x_mm * PX_PER_MM)), int(round(ui.HEADER_H + MARGIN + y_mm * PX_PER_MM))

    @staticmethod
    def _inside(rect, x, y) -> bool:
        rx, ry, rw, rh = rect
        return rx <= x < rx + rw and ry <= y < ry + rh

    def route_mouse(self, event: int, wx: int, wy: int, flags: int = 0) -> None:
        """Window mouse events: the top view drives the probe; dragging / the wheel in the 3D panel turn / zoom the
        view. A stroke ends on button-up wherever it happens; leaving the top view while recording moves the probe
        off the region (no capture)."""
        if self._orbit is not None:                          # rotating the 3D view
            if event == cv2.EVENT_MOUSEMOVE and self.view3d is not None:
                self.view3d.orbit(wx - self._orbit[0], wy - self._orbit[1])
                self._orbit = (wx, wy)
            elif event == cv2.EVENT_LBUTTONUP:
                self._orbit = None
            return
        if self._inside(self.RECT_3D, wx, wy) and not self.pose.recording:
            if event == cv2.EVENT_LBUTTONDOWN:
                for name, rect in self.view_buttons():
                    if self._inside(rect, wx, wy):
                        self.set_view(name)
                        return
                self._orbit = (wx, wy)
            elif event == cv2.EVENT_MOUSEWHEEL and self.view3d is not None:
                self.view3d.zoom(1.1 if wheel_delta(flags) > 0 else 1 / 1.1)
            return
        if event == cv2.EVENT_LBUTTONDOWN and self._inside(self.bmode_button(), wx, wy) and not self.pose.recording:
            self.toggle_bmode()
            return
        if event == cv2.EVENT_MOUSEWHEEL and self._inside(self.RECT_TOP, wx, wy):
            self.turn(1 if wheel_delta(flags) > 0 else -1)                   # the wheel turns the probe
            return
        if self._inside(self.RECT_TOP, wx, wy) or self.pose.recording:
            self.on_mouse(event, wx, wy, flags)

    def on_mouse(self, event: int, wx: int, wy: int, flags: int = 0) -> None:
        """Top-view mouse events: move the probe, button down / up start / end a stroke."""
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
            self.pose.move(*self.mm(wx, wy))
        if event == cv2.EVENT_LBUTTONDOWN and not self.completed:
            self.pose.press()
            self.strokes.append([len(self.frames), len(self.frames)])
        elif event == cv2.EVENT_LBUTTONUP and self.pose.recording:
            self.pose.release()
            if self.strokes and self.strokes[-1][0] == self.strokes[-1][1]:
                self.strokes.pop()                           # empty stroke (no frame captured)

    # ---- capture
    def step(self):
        """Offer the current pose to the acquirer; a captured frame goes into the reconstruction. Also measures the
        capture-loop period (time from a capturing step to the next step). Returns the FrameRecord or None."""
        now = self.clock()
        if self._prev_captured and self._prev_step_t is not None:
            p = now - self._prev_step_t
            if p < 1.0:
                self.period_s = p if self.period_s is None else 0.8 * self.period_s + 0.2 * p
        self._prev_step_t = now
        frame = None
        if not self.completed:
            s = self.pose.sample()
            frame = self.acq.feed(s.T, s.t, s.recording)
        self._prev_captured = frame is not None
        if frame is not None:
            self.comp.insert(frame)
            self.latest = frame
            stroke = self.strokes[-1]
            if frame.index > stroke[0]:
                prev = self.frames[frame.index - 1]
                d = float(np.linalg.norm(frame.T_true[:3, 3] - prev.T_true[:3, 3]))
                if d > GAP_FACTOR * self.cfg.frame_spacing_mm:
                    self.gaps.append((frame.index - 1, frame.index, d))
            stroke[1] = frame.index + 1
        return frame

    def undo(self) -> bool:
        """Remove the last stroke (its frames leave the sweep and the reconstruction); not while recording."""
        if self.pose.recording or not self.strokes or self.completed:
            return False
        a, b = self.strokes.pop()
        removed = self.acq.truncate(a)
        for i in range(0, len(removed), 40):
            self.comp.remove_batch(removed[i:i + 40])
        self.gaps = [g for g in self.gaps if g[1] < a]
        self.latest = self.frames[-1] if self.frames else None
        self._coverage = (-1, None)
        self.message = f"undo: {len(removed)} frames removed"
        return True

    def save(self, folder: str = SWEEPS_DIR) -> str:
        sweep = self.acq.sweep
        sweep.metadata.update(strokes=[list(s) for s in self.strokes], gaps=len(self.gaps),
                              covered_pct=self.coverage()[2], frames_with_bmode=sweep.n_images)
        path = sweep.save(os.path.join(folder, mouse_filename(self.case, self.cfg, self.acq_cfg.store_images)))
        self.message = f"saved {os.path.basename(path)} ({len(sweep)} frames)"
        return path

    def complete(self, results: str = RESULTS_DIR, fill: bool = True, max_gap: int = 1) -> tuple[str, str]:
        """Stop recording, fill small holes, evaluate (S4) and write the report; the 3D view then shows the
        evaluated (hole-filled) volume."""
        self.pose.release()
        self.completed = True
        if self._instances is None:
            self._instances = instance_masks(self.sim.an, self.grid)
        folder = os.path.join(results, f"{self.case}_mouse_{time.strftime('%Y%m%d-%H%M%S')}")
        cov = self.coverage()
        self.evaluation, self.final_labels = complete_report(
            self.comp.result(), self.sim.an, self.grid, folder, self.case, gt=self.gt_volume(),
            instances=self._instances, fill=fill, max_gap=max_gap, slice_view=self.slice_view,
            extra=dict(mode="mouse", frames=len(self.frames), strokes=len(self.strokes), speed_gaps=len(self.gaps),
                       covered_pct=cov[2], sweep_config=dataclasses.asdict(self.cfg)),
            title=f"{self.case}: hand-guided (mouse), {len(self.frames)} frames in {len(self.strokes)} strokes, "
                  f"spacing {self.cfg.frame_spacing_mm:g} mm, {cov[2]:.1f}% of the region covered")
        r = self.evaluation.report
        self.message = (f"complete: {100 * r['accuracy_observed']:.1f}% correct, {r['counts']['detected']} detected, "
                        f"{r['counts']['not covered']} not covered, topology {r['topology']['status']}")
        return folder, summary_table(r)

    # ---- 3D view
    def meshes(self) -> dict:
        return surface_meshes(self.final_labels if self.completed else self.comp.result(), self.grid)

    def init_view3d(self, size=(VIEW3D_W, TOP_W - CAPTION_3D_H)) -> bool:
        """Create the embedded 3D view (PyVista off-screen); False with a reason when it is not available."""
        try:
            from orvue_us_inverse.mapping.live3d import Embedded3D
            self.view3d = Embedded3D(size)
            return True
        except Exception as e:                       # ImportError, blocked DLL, no OpenGL ...
            self.view3d = None
            self.view3d_error = f"{type(e).__name__}: {e}"
            return False

    def live3d_due(self, last_state) -> tuple[bool, tuple]:
        """Rebuild the 3D view now? Only while the button is up and when the frames (stroke, undo, reset), the
        ground-truth setting or the completion changed since last_state; returns (due, current state)."""
        state = (len(self.frames), self.slice_view.show_gt, self.completed)
        return (not self.pose.recording and state != last_state), state

    def refresh_view3d(self) -> None:
        if self.view3d is not None:
            self.update_live3d(self.view3d)

    # ---- coverage
    def coverage(self) -> tuple[np.ndarray, np.ndarray, float, int]:
        """(covered (ny, nx), holes (ny, nx), % of the region covered, number of holes). A hole is an unscanned
        area enclosed by scanned columns."""
        n = len(self.frames)
        if self._coverage[0] != n:
            hits = self.comp.column_hits().T
            covered = hits > 0
            holes = ndimage.binary_fill_holes(covered) & ~covered
            n_holes = int(ndimage.label(holes)[1]) if holes.any() else 0
            self._coverage = (n, (hits, covered, holes, 100.0 * covered.mean(), n_holes))
            self.observed_pct = 100.0 * np.count_nonzero(self.comp.hits) / self.grid.n_voxels
        hits, covered, holes, pct, n_holes = self._coverage[1]
        return covered, holes, pct, n_holes

    def coverage_image(self) -> np.ndarray:
        covered, holes, _, _ = self.coverage()
        hits = self._coverage[1][0].astype(np.float64)
        top = np.log1p(hits.max()) if hits.max() > 0 else 1.0
        img = cv2.applyColorMap((255 * np.log1p(hits) / top).astype(np.uint8), cv2.COLORMAP_INFERNO)
        img[~covered] = 0
        img[holes] = HOLE_BGR
        r = int(round(100 * PX_PER_MM))
        return cv2.resize(img, (r, r), interpolation=cv2.INTER_NEAREST)

    def toggle_revealed(self) -> None:
        if self._revealed_img is None:
            self._revealed_img = top_view(self.sim.an, px_mm=1.0 / PX_PER_MM)
        self.revealed = not self.revealed

    # ---- drawing
    def px(self, xy) -> tuple[int, int]:
        return int(round(MARGIN + xy[0] * PX_PER_MM)), int(round(MARGIN + xy[1] * PX_PER_MM))

    def guide_lanes(self):
        """(lanes of the orientation nearest the probe angle (mod 180), current lane or None)."""
        def d(a, b):
            return abs((a - b + 90.0) % 180.0 - 90.0)
        yaw = min(self.cfg.yaw_list_deg, key=lambda y: d(y, self.pose.yaw))
        lanes = self.plan.lanes_for(yaw)
        u, _ = axes(yaw)
        s = self.pose.x * u[0] + self.pose.y * u[1]
        cur = min(lanes, key=lambda ln: abs(ln.s_mm - s)) if lanes else None
        if cur is not None and abs(cur.s_mm - s) > PROBE.width_mm / 2:
            cur = None
        return lanes, cur

    def top_image(self) -> np.ndarray:
        n = self.size
        img = np.full((n, n, 3), 28, np.uint8)
        r0, r1 = MARGIN, MARGIN + int(round(100 * PX_PER_MM))
        cov = self.coverage_image()
        if self.revealed:
            base = self._revealed_img.copy()
            m = cov.any(axis=2)
            base[m] = (0.45 * cov[m] + 0.55 * base[m]).astype(np.uint8)
            img[r0:r1, r0:r1] = base
        else:
            img[r0:r1, r0:r1] = cov
        cv2.rectangle(img, (r0 - 1, r0 - 1), (r1, r1), ui.BORDER, 1)
        lanes, cur = self.guide_lanes()
        h = PROBE.width_mm / 2
        overlay = img.copy()
        for ln in lanes:
            u, _ = axes(ln.yaw_deg)
            a, b = np.array(ln.start), np.array(ln.end)
            quad = np.array([self.px(a - h * u), self.px(b - h * u), self.px(b + h * u), self.px(a + h * u)], np.int32)
            if ln is cur:
                cv2.fillConvexPoly(overlay, quad, ui.AMBER)
            cv2.polylines(img, [quad], True, ui.DIM, 1, cv2.LINE_AA)
        if cur is not None:
            img[:] = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
        for i, j, _ in self.gaps:
            p, q = self.px(self.frames[i].T_true[:2, 3]), self.px(self.frames[j].T_true[:2, 3])
            cv2.line(img, p, q, GAP_BGR, 2, cv2.LINE_AA)
            cv2.circle(img, q, 3, GAP_BGR, -1, cv2.LINE_AA)
        T = self.pose.sample().T
        o, u = T[:2, 3], T[:2, 0]
        a, b = self.px(o - h * u), self.px(o + h * u)
        rec = self.pose.recording
        cv2.line(img, a, b, ui.GREEN if rec else ui.GREY, 3, cv2.LINE_AA)
        cv2.circle(img, a, 5, ui.AMBER, -1, cv2.LINE_AA)
        ui.text(img, "x", (r1 - 12, r0 - 10), ui.GREY, 0.45)
        cv2.arrowedLine(img, (r1 - 60, r0 - 14), (r1 - 20, r0 - 14), ui.GREY, 1, cv2.LINE_AA, tipLength=0.25)
        ui.text(img, "y", (r0 - 30, r1 - 8), ui.GREY, 0.45)
        cv2.arrowedLine(img, (r0 - 16, r1 - 60), (r0 - 16, r1 - 20), ui.GREY, 1, cv2.LINE_AA, tipLength=0.25)
        ui.text(img, "SWEEP  " + ("anatomy revealed" if self.revealed else "anatomy hidden (a reveals)"),
                (r0, r0 - 10), ui.GREY, 0.42)
        self.draw_key(img, r0, r1 + 22)
        return img

    @staticmethod
    def draw_key(img: np.ndarray, x: int, y: int) -> None:
        """Colour key of the coverage map in one row from (x, y) (baseline)."""
        def swatch(x0, colour, label):
            cv2.rectangle(img, (x0, y - 9), (x0 + 12, y + 2), colour, -1)
            cv2.rectangle(img, (x0, y - 9), (x0 + 12, y + 2), ui.BORDER, 1)
            ui.text(img, label, (x0 + 16, y), ui.GREY, 0.38)
            return x0 + 22 + ui.text_w(label, 0.38)
        x = swatch(x, (0, 0, 0), "not scanned")
        bar = cv2.applyColorMap(np.linspace(40, 255, 48).astype(np.uint8)[None, :], cv2.COLORMAP_INFERNO)[0]
        for i, c in enumerate(bar):
            cv2.line(img, (x + i, y - 9), (x + i, y + 2), tuple(int(k) for k in c), 1)
        ui.text(img, "scanned 1x -> many", (x + 52, y), ui.GREY, 0.38)
        x += 58 + ui.text_w("scanned 1x -> many", 0.38)
        x = swatch(x, HOLE_BGR, "hole")
        cv2.line(img, (x, y - 4), (x + 14, y - 4), GAP_BGR, 2, cv2.LINE_AA)
        ui.text(img, "speed gap", (x + 18, y), ui.GREY, 0.38)

    def showing_bmode(self) -> bool:
        """True when the middle panel shows a B-mode image (B-mode on and the latest frame has one)."""
        return self.bmode_on and self.latest is not None and self.latest.image is not None

    def bmode_image(self) -> np.ndarray:
        """The latest frame (501 x 301, BGR): B-mode, or the labels in tissue colours when B-mode is off."""
        f = self.latest
        if f is None:
            img = np.zeros((*FRAME_SHAPE, 3), np.uint8)
            ui.text(img, "hold the left button", (60, 240), ui.GREY, 0.5)
            ui.text(img, "on the sweep to record", (52, 266), ui.GREY, 0.5)
            return img
        img = cv2.cvtColor(f.image, cv2.COLOR_GRAY2BGR) if self.showing_bmode() else COL_TAB[f.labels].copy()
        cv2.circle(img, (8, 30), 4, ui.AMBER, -1, cv2.LINE_AA)
        return img

    @staticmethod
    def draw_tissue_key(img: np.ndarray, x: int, y: int, w: int) -> None:
        """Tissue colours of the labels view, two columns, in a dark box from (x, y) (top-left)."""
        rows = (len(TISSUE_NAMES) + 1) // 2
        cv2.rectangle(img, (x, y), (x + w, y + 8 + 16 * rows), (0, 0, 0), -1)
        for k, name in enumerate(TISSUE_NAMES):
            cx, cy = x + 6 + (k // rows) * (w // 2), y + 6 + 16 * (k % rows)
            cv2.rectangle(img, (cx, cy), (cx + 10, cy + 10), tuple(int(c) for c in COL_TAB[k]), -1)
            ui.text(img, name, (cx + 15, cy + 10), ui.GREY, 0.36)

    def view3d_image(self) -> np.ndarray:
        """The 3D view below the caption strip: (TOP_W - CAPTION_3D_H) x VIEW3D_W, BGR."""
        w, h = VIEW3D_W, TOP_W - CAPTION_3D_H
        if self.view3d is not None:
            img = self.view3d.image().copy()
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        else:
            img = np.full((h, w, 3), 16, np.uint8)
            for k, line in enumerate(ui.wrap("3D view needs PyVista / VTK (pip install -e .[view3d]); "
                                             f"{self.view3d_error}. b opens the browser 3D view, 3 writes a snapshot.",
                                             w - 30, 0.45)[:8]):
                ui.text(img, line, (14, 160 + 20 * k), ui.GREY, 0.45)
        return img

    def speed_meter(self, img: np.ndarray, x: int, y: int, w: int) -> int:
        """Bar 0 .. 1.5 x max speed with green / amber / red zones and the current speed; returns the next y."""
        v, cls = self.speed()
        vmax = self.max_speed
        full = 1.5 * vmax
        h = 14
        zones = ((0.0, SPEED_OK, ui.GREEN), (SPEED_OK, 1.0, ui.AMBER), (1.0, 1.5, ui.RED))
        for lo, hi, c in zones:
            cv2.rectangle(img, (x + int(w * lo / 1.5), y), (x + int(w * hi / 1.5), y + h),
                          tuple(int(0.35 * k) for k in c), -1)
        fill = int(w * min(v, full) / full)
        cv2.rectangle(img, (x, y + 3), (x + fill, y + h - 3), SPEED_BGR[cls], -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), ui.BORDER, 1)
        ui.text(img, f"{v:.1f} mm/s  (max {vmax:.1f} = {self.cfg.frame_spacing_mm:g} mm x {self.capture_rate:.1f}/s)",
                (x, y + h + 16), SPEED_BGR[cls], 0.42)
        return y + h + 26

    def status_rows(self) -> tuple[list[tuple], list[tuple]]:
        _, _, pct, n_holes = self.coverage()
        state = "complete" if self.completed else ("recording" if self.pose.recording else "ready")
        lanes, cur = self.guide_lanes()
        left = [("state", state, ui.GREEN if state == "recording" else ui.AMBER),
                ("probe", f"({self.pose.x:.1f}, {self.pose.y:.1f}) mm"),
                ("angle", f"{self.pose.yaw:g} deg  (wheel or q / e: 5 deg, 0 / 9)"),
                ("B-mode", "on (i: off for speed)" if self.bmode_on else "off: labels only (i: on)"),
                ("lane", "---" if cur is None else f"{cur.k + 1}/{cur.n} (guides at {cur.yaw_deg:g} deg)")]
        right = [("frames", f"{len(self.frames)} in {len(self.strokes)} strokes, {self.acq.sweep.n_images} with B-mode"),
                 ("covered", f"{pct:.1f}% of the region, observed {self.observed_pct:.1f}% of the volume"),
                 ("holes", str(n_holes), ui.RED if n_holes else ui.WHITE),
                 ("speed gaps", str(len(self.gaps)), ui.RED if self.gaps else ui.WHITE)]
        return left, right

    def compose(self) -> np.ndarray:
        """The whole window: sweep | B-mode | 3D, then status / speed / message, then keys."""
        img = np.full((self.HEIGHT, self.WIDTH, 3), ui.BG, np.uint8)
        ui.header(img, "Mouse sweep", f"hand-guided, {self.case}")
        x, y, w, h = self.RECT_TOP
        img[y:y + h, x:x + w] = self.top_image()
        x, y, w, h = self.RECT_BMODE
        img[y:y + h, x:x + w] = cv2.resize(self.bmode_image(), (w, h), interpolation=cv2.INTER_AREA)
        cap = ("B-MODE" if self.showing_bmode() else "LABELS") + \
            (f"  frame {self.latest.index}" if self.latest is not None else "")
        cv2.rectangle(img, (x, y), (x + w - 1, y + 22), (0, 0, 0), -1)          # caption strip (readable on images)
        ui.text(img, cap, (x + 6, y + 16), ui.AMBER, 0.42)
        bx, by, bw, bh = self.bmode_button()
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), ui.PANEL_LIGHT, -1)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), ui.GREEN if self.bmode_on else ui.BORDER, 1)
        ui.text(img, f"B-mode {'ON' if self.bmode_on else 'OFF'} (i)", (bx + 5, by + 13),
                ui.GREEN if self.bmode_on else ui.GREY, 0.38)
        if self.latest is not None and not self.showing_bmode():
            self.draw_tissue_key(img, x + 4, y + h - 8 - 16 * 6 - 4, w - 8)
        x, y, w, h = self.RECT_3D
        img[y + CAPTION_3D_H:y + h, x:x + w] = self.view3d_image()
        cap3 = "3D RECONSTRUCTION" + ("  (evaluated)" if self.completed else "")
        cv2.rectangle(img, (x, y), (x + w - 1, y + CAPTION_3D_H), (0, 0, 0), -1)
        ui.text(img, cap3, (x + 6, y + 16), ui.AMBER, 0.42)
        if self.view3d is not None:
            for name, (bx, by, bw, bh) in self.view_buttons():
                active = self.view3d.view == name
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), ui.PANEL_LIGHT, -1)
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), ui.AMBER if active else ui.BORDER, 1)
                ui.text(img, VIEW_LABELS[name], (bx + 9, by + 15), ui.AMBER if active else ui.GREY, 0.42)
        ui.text(img, "drag: rotate  wheel: zoom  v: next view  g: truth" +
                (" (shown)" if self.slice_view.show_gt else ""), (x + 6, y + 60), ui.GREY, 0.38)
        y0 = ui.HEADER_H + TOP_W + 8
        cv2.line(img, (0, y0 - 4), (self.WIDTH - 1, y0 - 4), ui.BORDER, 1)
        left, right = self.status_rows()
        ui.rows(img, 14, y0 + 18, left, value_x=70, step=22)
        ui.rows(img, 380, y0 + 18, right, value_x=90, step=22)
        sx = 830
        ui.text(img, "SPEED", (sx, y0 + 18), ui.AMBER, 0.42)
        yy = self.speed_meter(img, sx, y0 + 26, self.WIDTH - sx - 20)
        for k, line in enumerate(ui.wrap(self.message, self.WIDTH - sx - 20, 0.42)[:2]):
            ui.text(img, line, (sx, yy + 10 + 18 * k), ui.GREY, 0.42)
        ui.key_bar(img, [("hold L", "record"), ("wheel/q/e", "turn 5 deg"), ("0/9", "0 / 90 deg"),
                         ("i", "B-mode on/off"), ("u", "undo stroke"), ("r", "reset"), ("Esc", "quit")],
                   self.HEIGHT - KEYS_H)
        ui.key_bar(img, [("c", "complete + report"), ("s", "save sweep"), ("a", "anatomy"), ("g", "truth in 3D"),
                         ("v", "3D view"), ("3", "snapshot + STL"), ("b", "browser 3D")], self.HEIGHT - KEYS_H // 2)
        return img


def parse_args(argv=None) -> argparse.Namespace:
    d = SweepConfig()
    p = argparse.ArgumentParser(description="Hand-guided (mouse) sweep over the hidden anatomy.")
    p.add_argument("--case", default="normal", choices=sorted(CASES))
    p.add_argument("--yaw", type=float, nargs="+", default=d.yaw_list_deg, help="orientations of the lane guides")
    p.add_argument("--overlap", type=float, default=d.overlap_pct, help="requested lane overlap in percent")
    p.add_argument("--spacing", type=float, default=d.frame_spacing_mm, help="frame spacing in mm")
    p.add_argument("--no-images", action="store_true", help="start with B-mode off (labels only, faster capture; "
                                                                "i or the button switches it during the sweep)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = SweepConfig(yaw_list_deg=a.yaw, overlap_pct=a.overlap, frame_spacing_mm=a.spacing)
    app = MouseSession(a.case, cfg, AcquisitionConfig(store_images=not a.no_images))
    if not app.init_view3d():
        print(f"[run_mouse] 3D view not available: {app.view3d_error}", flush=True)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, lambda ev, x, y, fl, _: app.route_mouse(ev, x, y, fl))
    view_state = None
    while True:
        app.step()
        due, state = app.live3d_due(view_state)              # after a stroke / undo / reset / g / c
        if due:
            app.refresh_view3d()
            view_state = state
        cv2.imshow(WIN, app.compose())
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord("q"), ord("e")):
            app.turn(-1 if key == ord("q") else 1)
        elif key in (ord("0"), ord("9")):
            app.set_angle(0.0 if key == ord("0") else 90.0)
        elif key == ord("v"):
            app.next_view()
        elif key == ord("i"):
            app.toggle_bmode()
        elif key == ord("u"):
            if not app.undo():
                app.message = "nothing to undo (or still recording)"
        elif key == ord("r"):
            app.restart()
            app.message = "reset"
        elif key == ord("s"):
            print(f"[run_mouse] {app.save()}", flush=True)
        elif key == ord("a"):
            if app._revealed_img is None:
                app.message = "computing the anatomy top view ..."
                cv2.imshow(WIN, app.compose())
                cv2.waitKey(1)
            app.toggle_revealed()
            app.message = ""
        elif key == ord("g"):
            app.toggle_gt_contours()                          # the flag the 3D view uses for the ground truth
        elif key == ord("3") and app.frames:
            png, stls = app.export_3d()
            print(f"[run_mouse] {png}\n[run_mouse] " + "\n[run_mouse] ".join(stls), flush=True)
        elif key == ord("b") and app.frames:
            path = app.browser_view()
            print(f"[run_mouse] {path}", flush=True)
            webbrowser.open("file:///" + os.path.abspath(path).replace(os.sep, "/"))
        elif key == ord("c") and not app.completed and app.frames:
            app.message = "complete: evaluating against the ground truth ..."
            cv2.imshow(WIN, app.compose())
            cv2.waitKey(1)
            folder, table = app.complete()
            print(table, flush=True)
            print(f"[run_mouse] report: {folder}", flush=True)
            if hasattr(os, "startfile"):
                os.startfile(folder)
        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    if app.view3d is not None:
        app.view3d.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
