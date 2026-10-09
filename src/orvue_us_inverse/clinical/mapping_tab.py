"""
orvue_us_inverse.clinical.mapping_tab - the INVERSE MAPPING imaging mode of the clinical window.

MappingSession (a TrackedSession: the sweep, reconstruction, coverage, undo, evaluation and 3D logic of the mouse /
tracked apps) with three pose sources and the clinical layout, 3D-centred:

  left panel   SOURCE (scripted sweep / mouse / camera probe), ACQUISITION buttons (record, undo, reset, complete,
               save, B-mode on / off, AR overlay), IMAGING MODE tabs (B-MODE / INVERSE MAPPING / ...), TOOLS
  B-mode       live image at the probe (also before recording: it shows where you are), the captured frame while
               recording; labels in tissue colours when B-mode is off. Under it: the reconstructed structures
               (volume per structure group; after Complete the evaluation per structure).
  3D           the reconstruction, large (PyVista off-screen; drag rotate, wheel zoom, view presets with patient
               directions). Refreshed after each stroke (during a scripted sweep every REFRESH_FRAMES frames).
  right        CAMERA VIEW (the D405 view of the B-mode tab) with the AR overlay of the reconstructed structures
               (clinical.ar: projected onto the phantom surface, depth-coded; o / button toggles), tracking line,
               SWEEP MAP (coverage heat map, holes, speed gaps; the mouse drives the probe here), status, speed.

Recording: SPACE or the RECORD button switches recording on and off for every source. SCRIPTED plays the S5
strategy (yaw 0, 20 % overlap, 0.25 mm) on its lanes (shown on the map), one frame per loop (start / pause). MOUSE: the
probe follows the mouse over the map (also holding the left button records, as in the mouse app); wheel / q / e
turn. CAMERA PROBE: the camera sets position and angle. The mouse and camera have no
lane guides: the probe may be moved in any direction and angle. Frames from several sources add up in one sweep.
"""
import cv2
import numpy as np

from orvue_us_inverse.clinical import ar
from orvue_us_inverse.mapping.config import AcquisitionConfig, SweepConfig
from orvue_us_inverse.mapping.poses import ScriptedPose
from orvue_us_inverse.mapping.probe import FRAME_SHAPE
from orvue_us_inverse.mapping.run_mouse import TOP_W, VIEW_LABELS, VIEW_ORDER, MouseSession, wheel_delta
from orvue_us_inverse.mapping.run_tracked import TrackedSession
from orvue_us_inverse.simulation.anatomy import COL_TAB
from orvue_us_inverse.simulation.bmode import B_IMAGE_H, RIGHT_W
from orvue_us_inverse.ui import clinical as ui

# window size: the same as the B-MODE tab (clinical_view: panel 200 + image + scale 70, plus RIGHT_W; header 56,
# image B_IMAGE_H, bottom 16, key bar 36)
WIDTH = 200 + int(round(B_IMAGE_H * FRAME_SHAPE[1] / FRAME_SHAPE[0])) + 70 + RIGHT_W
HEIGHT = ui.HEADER_H + B_IMAGE_H + 16 + 36
KEYS_H = 36
G = 14                                      # gutter
CAP = 26                                    # panel caption height
LEFT_W = 200
TAB_W = 176
BM_RECT = (LEFT_W + G, ui.HEADER_H + CAP, FRAME_SHAPE[1], FRAME_SHAPE[0])          # native 0.1 mm / px
BM_COL_W = FRAME_SHAPE[1] + 40                                                    # image + depth scale
RIGHT_X = WIDTH - G - 555
RIGHT_COL_W = 555
V3_RECT = (LEFT_W + G + BM_COL_W + G, ui.HEADER_H + CAP, RIGHT_X - G - (LEFT_W + G + BM_COL_W + G),
           HEIGHT - KEYS_H - G - ui.HEADER_H - CAP)
V3_BAR = 34                                 # view buttons strip at the top of the 3D panel
CAM_RECT = (RIGHT_X, ui.HEADER_H + CAP, RIGHT_COL_W, 312)
TRACK_Y = CAM_RECT[1] + CAM_RECT[3] + 20
MAP_SIZE = 400
MAP_RECT = (RIGHT_X, TRACK_Y + 16 + CAP, MAP_SIZE, MAP_SIZE)
STRUCT_Y = BM_RECT[1] + BM_RECT[3] + 16 + CAP
REFRESH_FRAMES = 80                         # scripted sweep: refresh the 3D view and overlay every so many frames
SOURCES = ("scripted", "mouse", "camera")
SOURCE_LABELS = {"scripted": "SCRIPTED SWEEP", "mouse": "MOUSE", "camera": "CAMERA PROBE"}
MODES = ("B-MODE", "INVERSE MAPPING", "DOPPLER", "ELASTOGRAPHY")
MODE_ENABLED = (True, True, False, False)
PREVIEW_MOVE_MM, PREVIEW_TURN_DEG = 0.2, 0.5
KEYS = [("Tab", "B-mode / mapping"), ("1 / 2 / 3", "scripted / mouse / camera"), ("SPACE", "record on / off"),
        ("wheel / q / e", "turn (mouse)"), ("u", "undo"), ("r", "reset"), ("c", "complete"), ("i", "B-mode"),
        ("o", "AR overlay"), ("v", "3D view"), ("g", "truth in 3D"), ("z", "camera zoom"), ("s", "save"), ("Esc", "quit")]


def mode_tab_rects(x: int, y: int, width: int = TAB_W, height: int = 30, gap: int = 8) -> list[tuple]:
    """Rects of the IMAGING MODE tabs as ui.mode_tabs draws them (one per MODES entry)."""
    return [(x, y + k * (height + gap), width, height) for k in range(len(MODES))]


def draw_mode_tabs(img: np.ndarray, x: int, y: int, active: str) -> int:
    """IMAGING MODE section with the tabs (B-MODE, INVERSE MAPPING, Doppler / Elastography to do); returns the y
    below the tabs."""
    ui.section(img, x, y, "IMAGING MODE")
    items = [(m, m == active, en) for m, en in zip(MODES, MODE_ENABLED)]
    return ui.mode_tabs(img, x, y + 10, TAB_W, items)


def hit_mode_tab(modes_y: int, mx: int, my: int, x: int = 10) -> str | None:
    """The enabled mode tab under (mx, my) (tabs drawn by draw_mode_tabs(img, x, modes_y, ...)), or None."""
    for (rx, ry, rw, rh), m, en in zip(mode_tab_rects(x, modes_y + 10), MODES, MODE_ENABLED):
        if en and rx <= mx <= rx + rw and ry <= my <= ry + rh:
            return m
    return None


class MappingSession(TrackedSession):
    """Inverse-mapping session of the clinical window: scripted / mouse / camera sources, 3D-centred layout."""

    WIDTH, HEIGHT = WIDTH, HEIGHT
    TITLE = ("INVERSE ANATOMY MAPPING", "tracked 2D ultrasound -> 3D anatomy")
    MODE = "mapping"
    RECT_TOP = MAP_RECT
    RECT_3D = V3_RECT
    request = None                                       # action for the app loop (complete / save / browser ...)

    def __init__(self, case: str, cfg: SweepConfig | None = None, acq_cfg: AcquisitionConfig | None = None,
                 tracker=None, modes_y: int = 690, source: str | None = None, **kw):
        self.scripted_pose = None
        self.modes_y = modes_y
        self.ar_on = True
        self._labels = (-1, None)                    # (frames, label volume) cache
        self._ar = None                              # (bgr, alpha) of the overlay
        self._groups = []                            # structure group volumes
        self._preview = None                         # (image BGR, T, kind) live image at the probe
        self._stroke_lane = None
        self.sources_used: set[str] = set()
        self.buttons: dict[str, tuple] = {}
        self.mouse_latched = False                   # mouse recording switched on by SPACE / the button
        kw.setdefault("key_state", None)             # SPACE toggles recording here (no hold-to-record)
        super().__init__(case, cfg or SweepConfig(), acq_cfg or AcquisitionConfig(), tracker=tracker, **kw)
        self.scripted_pose = ScriptedPose(self.cfg, clock=self.clock)
        start = source or ("camera" if tracker is not None else "mouse")
        if not self.set_source(start):
            self.set_source("mouse")
        self.message = ""

    # ---- state
    def restart(self) -> None:
        super().restart()
        if self.scripted_pose is not None:
            self.scripted_pose = ScriptedPose(self.cfg, clock=self.clock)
            if self.source == "scripted":
                self.pose = self.scripted_pose
        self.latched = False
        self.mouse_latched = False
        self._labels, self._ar, self._groups, self._preview = (-1, None), None, [], None
        self._stroke_lane = None
        self.sources_used = set()

    def set_source(self, source: str) -> bool:
        if source != "scripted":
            ok = super().set_source(source)
            if ok:
                self._stroke_lane = None
                self.mouse_latched = False
            return ok
        if self.pose.recording:
            self.message = "stop recording before switching the pose source"
            return False
        self.source, self.latched = "scripted", False
        self.pose = self.scripted_pose
        self.buffer.clear()
        self.acq.break_stretch()
        self.message = "SCRIPTED SWEEP: SPACE or the button starts / pauses the planned lanes"
        return True

    def toggle_record(self) -> None:
        """The RECORD button / SPACE: scripted: start / pause the sweep; camera and mouse: recording on / off."""
        if self.completed:
            self.message = "the sweep is complete: r resets for a new one"
            return
        if self.source == "scripted":
            if self.scripted_pose.playing:
                self.scripted_pose.release()
                self.message = "scripted sweep paused"
            else:
                self.scripted_pose.press()
                self.message = "scripted sweep running"
        elif self.source == "camera":
            self.latched = not self.latched
            self.message = "recording (camera probe)" if self.latched else "recording stopped"
        elif self.mouse_pose.recording:
            self.mouse_pose.release()
            self.mouse_latched = False
            if self.strokes and self.strokes[-1][0] == self.strokes[-1][1]:
                self.strokes.pop()
            self.message = "recording stopped"
        else:
            self.mouse_pose.press()
            self.mouse_latched = True
            self.strokes.append([len(self.frames), len(self.frames)])
            self.message = "recording (mouse): move over the sweep map; SPACE stops"

    def update_pose(self) -> None:
        if self.source != "scripted":
            super().update_pose()
            return
        sp = self.scripted_pose
        if not self.completed:
            sp.advance(self.acq)
        lane = sp.lane if sp.recording else None
        if lane is not self._stroke_lane:               # one stroke per lane (undo removes the last lane)
            if self.strokes and self.strokes[-1][0] == self.strokes[-1][1]:
                self.strokes.pop()
            if lane is not None:
                self.strokes.append([len(self.frames), len(self.frames)])
            self._stroke_lane = lane
        if not sp.playing and sp.i >= len(sp.samples) - 1 and self.frames and "finished" not in self.message:
            self.message = "scripted sweep finished: c evaluates, or add strokes with the mouse / camera probe"

    def step(self):
        frame = super().step()
        if frame is not None:
            self.sources_used.add(self.source)
        return frame

    def speed(self) -> tuple[float, str]:
        v, cls = super().speed()
        return (v, "green") if self.source == "scripted" else (v, cls)

    def guide_lanes(self):
        """Lane guides only for the scripted sweep (mouse / camera: free movement)."""
        if self.source != "scripted":
            return [], None
        lane = self.scripted_pose.lane
        yaw = lane.yaw_deg if lane is not None else self.cfg.yaw_list_deg[0]
        return self.plan.lanes_for(yaw), lane

    def report_extra(self) -> dict:
        d = super().report_extra()
        d["pose_source"] = "+".join(s for s in SOURCES if s in self.sources_used) or self.source
        return d

    def route_mouse(self, event: int, wx: int, wy: int, flags: int = 0) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self._orbit is None:
            for name, rect in self.buttons.items():
                if self._inside(rect, wx, wy):
                    self.press_button(name)
                    return
        if self.mouse_latched and self._orbit is None:     # recording switched on: the probe just follows the
            if event == cv2.EVENT_MOUSEMOVE and self._inside(self.RECT_TOP, wx, wy):      # mouse over the map
                self.mouse_pose.move(*self.mm(wx, wy))
            elif event == cv2.EVENT_MOUSEWHEEL and self._inside(self.RECT_TOP, wx, wy):
                self.turn(1 if wheel_delta(flags) > 0 else -1)
            return
        if self.source == "mouse" or self._orbit is not None:
            super().route_mouse(event, wx, wy, flags)
            return
        # scripted / camera: the plan or the camera drives the probe; the mouse only works the 3D view (also while
        # recording)
        if self._inside(self.RECT_3D, wx, wy):
            if event == cv2.EVENT_LBUTTONDOWN:
                for name, rect in self.view_buttons():
                    if self._inside(rect, wx, wy):
                        self.set_view(name)
                        return
                self._orbit = (wx, wy)
            elif event == cv2.EVENT_MOUSEWHEEL and self.view3d is not None:
                self.view3d.zoom(1.1 if wheel_delta(flags) > 0 else 1 / 1.1)

    def press_button(self, name: str) -> None:
        if name in SOURCES:
            self.set_source(name)
        elif name == "record":
            self.toggle_record()
        elif name == "undo":
            if not self.undo():
                self.message = "nothing to undo (or still recording)"
        elif name == "reset":
            self.restart()
            self.message = "reset: new sweep"
        elif name == "complete":
            self.request = "complete"                    # done by the app (it shows a message first)
        elif name == "save":
            self.request = "save"
        elif name == "bmode":
            self.toggle_bmode()
        elif name == "ar":
            self.toggle_ar()
        elif name == "browser":
            self.request = "browser"
        elif name == "export":
            self.request = "export"

    def toggle_ar(self) -> bool:
        self.ar_on = not self.ar_on
        self.message = "AR overlay on: reconstructed structures on the camera view" if self.ar_on else "AR overlay off"
        return self.ar_on

    def bmode_button(self):
        return (-10, -10, 0, 0)                          # B-mode on / off is a left-panel button here

    # ---- coordinates of the sweep map (MouseSession draws it at 4 px / mm in a TOP_W canvas; shown at MAP_SIZE)
    def mm(self, wx: int, wy: int) -> tuple[float, float]:
        k = TOP_W / MAP_SIZE
        return MouseSession.mm(self, (wx - MAP_RECT[0]) * k, (wy - MAP_RECT[1]) * k + ui.HEADER_H)

    def window_px(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        cx, cy = MouseSession.window_px(self, x_mm, y_mm)
        k = MAP_SIZE / TOP_W
        return int(round(MAP_RECT[0] + cx * k)), int(round(MAP_RECT[1] + (cy - ui.HEADER_H) * k))

    # ---- reconstruction products (labels, 3D, overlay, structure volumes)
    def labels(self) -> np.ndarray:
        n = len(self.frames)
        if self._labels[0] != n or self._labels[1] is None:
            self._labels = (n, self.comp.result())
        return self._labels[1]

    def meshes(self) -> dict:
        from orvue_us_inverse.mapping.render import surface_meshes
        return surface_meshes(self.final_labels if self.completed else self.labels(), self.grid)

    def init_view3d(self, size=None) -> bool:
        return super().init_view3d(size or (V3_RECT[2], V3_RECT[3] - V3_BAR))

    def live3d_due(self, last_state) -> tuple[bool, tuple]:
        """As the mouse app (after a stroke, undo, reset, g, complete); during a scripted sweep also every
        REFRESH_FRAMES frames (no speed limit there)."""
        n = len(self.frames)
        state = (n, self.slice_view.show_gt, self.completed)
        if last_state is None or state[1:] != last_state[1:]:
            return True, state
        if self.source == "scripted" and self.pose.recording:
            return n - last_state[0] >= REFRESH_FRAMES, state
        return (not self.pose.recording and n != last_state[0]), state

    def refresh_view3d(self) -> None:
        """Rebuild the 3D view, the AR overlay and the structure volumes from the current reconstruction."""
        lab = self.final_labels if self.completed else self.labels()
        if self.frames:
            self._ar = ar.ar_layer(*ar.surface_map(lab, self.grid))
            self._groups = ar.group_volumes_ml(lab, self.grid.voxel_mm)
        else:
            self._ar, self._groups = None, []
        super().refresh_view3d()

    def view_buttons(self):
        x, y, _, _ = V3_RECT
        out, bx = [], x + 8
        for name in VIEW_ORDER:
            w = ui.text_w(VIEW_LABELS[name], 0.42) + 18
            out.append((name, (bx, y + 7, w, 20)))
            bx += w + 6
        return out

    # ---- live image at the probe
    def live_image(self) -> tuple[np.ndarray, str]:
        """(BGR 501 x 301 image, caption) for the B-mode panel: the captured frame while recording, else a live
        render at the probe (B-mode or labels, re-rendered when the probe moved)."""
        if self.pose.recording and self.latest is not None:
            return self.bmode_image(), f"{'B-MODE' if self.showing_bmode() else 'LABELS'}  captured frame {self.latest.index}"
        smp = self.pose.sample()
        if smp is None:
            img = np.zeros((*FRAME_SHAPE, 3), np.uint8)
            ui.text(img, "TRACKING LOST", (70, 250), ui.RED, 0.6)
            return img, "NO POSE"
        T = np.asarray(smp.T, np.float64)
        kind = "bmode" if self.bmode_on else "labels"
        p = self._preview
        if p is None or p[2] != kind or np.linalg.norm(p[1][:3, 3] - T[:3, 3]) > PREVIEW_MOVE_MM or \
                np.degrees(np.arccos(np.clip(np.dot(p[1][:3, 0], T[:3, 0]), -1, 1))) > PREVIEW_TURN_DEG:
            if not self.sim.in_contact(T):
                img = np.zeros((*FRAME_SHAPE, 3), np.uint8)
                ui.text(img, "NO CONTACT", (80, 250), ui.RED, 0.6)
                ui.text(img, "probe outside the region", (52, 280), ui.GREY, 0.45)
            elif kind == "bmode":
                img = cv2.cvtColor(self.sim.render(T), cv2.COLOR_GRAY2BGR)
            else:
                img = COL_TAB[self.sim.labels_image(T)].copy()
            cv2.circle(img, (8, 30), 4, ui.AMBER, -1, cv2.LINE_AA)
            self._preview = p = (img, T, kind)
        return p[0], ("B-MODE  live" if kind == "bmode" else "LABELS  live")

    # ---- camera view with the AR overlay
    def camera_image(self) -> tuple[np.ndarray | None, object]:
        """(zoomed camera scene with the tracking overlay and the AR layer, tracker state) or (None, state)."""
        tr = self.tracker
        if tr is None or not hasattr(tr, "get_frame"):
            return None, None
        from orvue_us_inverse.tracking.viewer import ZOOM, annotate
        frame, det, s = tr.get_frame()
        if frame is None:
            return None, s
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        img = annotate(frame, det, s, tr)
        if self.ar_on and self._ar is not None and s is not None and s.T_cam_phantom is not None:
            img = ar.project_layer(img, *self._ar, s.T_cam_phantom, tr.K, tr.dist, self.grid)
        return ZOOM.apply(img, s, tr.K, tr.dist), s

    # ---- drawing
    def draw_left_panel(self, img: np.ndarray) -> None:
        x, b = 10, {}
        y = ui.section(img, x, ui.HEADER_H + 26, "SOURCE")
        for src in SOURCES:
            active = self.source == src
            enabled = src != "camera" or self.tracked_pose is not None
            ui.mode_tabs(img, x, y - 6, TAB_W, [(SOURCE_LABELS[src], active, enabled)])
            b[src] = (x, y - 6, TAB_W, 30)
            y += 38
        if self.source == "camera" and self.tracked_pose is not None:
            st = self.tracked_pose.state
            label, colour = ui.tracking_status(st, True)
            ui.chip(img, (x, y + 10), label, colour, 0.42)
            y += 34
        y = ui.section(img, x, y + 10, "ACQUISITION")
        rec = self.pose.recording or (self.source == "scripted" and self.scripted_pose.playing)
        if self.source == "scripted":
            label = "PAUSE SWEEP" if self.scripted_pose.playing else (
                "RESUME SWEEP" if 0 < self.scripted_pose.i < len(self.scripted_pose.samples) - 1 else "START SWEEP")
            hint = f"lanes {self.scripted_pose.progress * 100:.0f}% played"
        elif self.source == "camera":
            label, hint = ("STOP RECORDING" if self.latched else "RECORD"), "or SPACE: on / off"
        else:
            label = "STOP RECORDING" if self.mouse_latched else "RECORD"
            hint = "or SPACE; or hold left button on map"
        r = (x, y - 6, TAB_W, 38)
        cv2.rectangle(img, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (30, 30, 70) if rec else ui.PANEL_LIGHT, -1)
        cv2.rectangle(img, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), ui.RED if rec else ui.AMBER, 1)
        cv2.circle(img, (r[0] + 16, r[1] + 19), 6, ui.RED, -1 if rec else 1, cv2.LINE_AA)
        ui.text(img, label, (r[0] + 30, r[1] + 25), ui.WHITE, 0.48)
        b["record"] = r
        ui.text(img, hint, (x, y + 50), ui.GREY, 0.4)
        y += 64
        for name, text, key in (("undo", "Undo stroke", "u"), ("reset", "Reset", "r"),
                                ("complete", "Complete + report", "c"), ("save", "Save sweep", "s")):
            b[name] = ui.button(img, (x, y, TAB_W, 30), text, key)
            y += 38
        for name, text, key, on in (("bmode", "B-mode", "i", self.bmode_on), ("ar", "AR overlay", "o", self.ar_on)):
            r = ui.button(img, (x, y, TAB_W, 30), f"{text} {'ON' if on else 'OFF'}", key)
            cv2.rectangle(img, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), ui.GREEN if on else ui.DIM, 1)
            b[name] = r
            y += 38
        yt = draw_mode_tabs(img, x, self.modes_y, "INVERSE MAPPING") + 14
        ui.section(img, x, yt, "TOOLS")
        b["browser"] = ui.button(img, (x, yt + 10, TAB_W, 30), "Browser 3D view", "b")
        b["export"] = ui.button(img, (x, yt + 48, TAB_W, 30), "3D snapshot + STL", "x")
        self.buttons = b

    def draw_bmode(self, img: np.ndarray) -> None:
        x, y, w, h = BM_RECT
        live, cap = self.live_image()
        ui.panel(img, (x, y, w, h), cap)
        img[y:y + h, x:x + w] = live
        sides = ui.image_sides(self.pose.yaw)
        ui.text(img, sides[0], (x + 18, y + 16), ui.AMBER, 0.42)
        ui.text(img, sides[1], (x + w - ui.text_w(sides[1], 0.42) - 6, y + 16), ui.AMBER, 0.42)
        if not self.bmode_on:
            self.draw_tissue_key(img, x + 4, y + h - 8 - 16 * 6 - 4, w - 8)
        sx = x + w + 8                                   # depth scale (cm), 0.1 mm / px
        cv2.line(img, (sx, y), (sx, y + h - 1), ui.GREY, 1)
        for d in range(0, 51, 5):
            yy = y + min(int(round(d * 10)), h - 1)
            major = d % 10 == 0
            cv2.line(img, (sx, yy), (sx + (10 if major else 5), yy), ui.WHITE if major else ui.GREY, 1)
            if major and d:
                ui.text(img, str(d // 10), (sx + 14, yy + 5), ui.WHITE, 0.45)
        ui.text(img, "cm", (sx + 8, y + h + 14), ui.GREY, 0.4)

    def draw_structures(self, img: np.ndarray) -> None:
        x, y, w = BM_RECT[0], STRUCT_Y, BM_COL_W
        h = HEIGHT - KEYS_H - G - y
        if self.completed and self.evaluation is not None:
            ui.panel(img, (x, y, w, h), "STRUCTURES  (evaluated vs ground truth)")
            colours = {"detected": ui.GREEN, "missed": ui.RED, "not covered": ui.AMBER, "no voxels": ui.DIM}
            yy = y + 20
            for s in self.evaluation.report["structures"]:
                if yy > y + h - 8:
                    break
                ui.text(img, s["name"].replace("_", " "), (x + 8, yy), ui.WHITE, 0.4)
                ui.text(img, s["status"], (x + w - ui.text_w(s["status"], 0.4) - 8, yy), colours.get(s["status"],
                                                                                                    ui.GREY), 0.4)
                yy += 19
            return
        ui.panel(img, (x, y, w, h), "RECONSTRUCTED STRUCTURES")
        yy = y + 22
        if not self._groups:
            for k, line in enumerate(ui.wrap("Record a sweep: the structures found appear here, in the 3D view "
                                             "and on the camera view.", w - 16, 0.42)):
                ui.text(img, line, (x + 8, yy + 18 * k), ui.GREY, 0.42)
            return
        for name, colour, ml in self._groups:
            cv2.rectangle(img, (x + 8, yy - 10), (x + 20, yy + 2), colour, -1)
            ui.text(img, name, (x + 28, yy), ui.WHITE if ml > 0 else ui.DIM, 0.42)
            val = f"{ml:.2f} mL" if ml > 0 else "not found"
            ui.text(img, val, (x + w - ui.text_w(val, 0.42) - 8, yy), ui.GREY if ml > 0 else ui.DIM, 0.42)
            yy += 22
        _, _, pct, n_holes = self.coverage()
        yy += 6
        for line in ui.wrap(f"Region covered {pct:.0f}%, volume observed {self.observed_pct:.0f}%. Complete (c) "
                            "evaluates every structure against the simulator's ground truth.", w - 16, 0.4):
            ui.text(img, line, (x + 8, yy), ui.GREY, 0.4)
            yy += 17

    def view3d_image(self) -> np.ndarray:
        w, h = V3_RECT[2], V3_RECT[3] - V3_BAR
        if self.view3d is not None:
            img = self.view3d.image().copy()
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            return img
        img = np.full((h, w, 3), 16, np.uint8)
        for k, line in enumerate(ui.wrap("3D view needs PyVista / VTK (pip install -e .[view3d]); "
                                         f"{self.view3d_error}. The browser 3D view (b) still works.", w - 40,
                                         0.45)[:8]):
            ui.text(img, line, (20, h // 2 + 22 * k), ui.GREY, 0.45)
        return img

    def draw_3d(self, img: np.ndarray) -> None:
        x, y, w, h = V3_RECT
        ui.panel(img, (x, y, w, h), "3D RECONSTRUCTION" + ("  |  evaluated, small holes filled" if self.completed
                                                           else "  |  live"))
        img[y + V3_BAR:y + h, x:x + w] = self.view3d_image()
        cv2.rectangle(img, (x, y), (x + w - 1, y + V3_BAR - 1), ui.PANEL, -1)
        if self.view3d is not None:
            for name, (bx, by, bw, bh) in self.view_buttons():
                active = self.view3d.view == name
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), ui.PANEL_LIGHT, -1)
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), ui.AMBER if active else ui.BORDER, 1)
                ui.text(img, VIEW_LABELS[name], (bx + 9, by + 15), ui.AMBER if active else ui.GREY, 0.42)
        hint = "drag: rotate   wheel: zoom   g: ground truth" + (" (shown)" if self.slice_view.show_gt else "")
        ui.text(img, hint, (x + w - ui.text_w(hint, 0.4) - 10, y + 22), ui.GREY, 0.4)

    def tracking_line(self) -> list[tuple[str, tuple]]:
        if self.tracked_pose is None:
            msg = "camera not connected" + (f": {self.camera_error}" if self.camera_error else "")
            return [(msg[:80], ui.GREY)]
        st = self.tracked_pose.state if self.source == "camera" else self.tracker.get_state()
        if st is None:
            return [("waiting for the camera", ui.GREY)]
        out = [(f"reference {st.n_reference_markers_used}/4", ui.WHITE if st.n_reference_markers_used == 4
                else ui.AMBER),
               (f"probe {'seen' if st.probe_valid else 'not seen'}", ui.WHITE if st.probe_valid else ui.RED),
               (f"{st.fps:.0f} fps", ui.GREY)]
        if self.source == "camera" and self.tracked_pose.face_z_mm is not None:
            out.append((f"face z {self.tracked_pose.face_z_mm:+.1f} mm", ui.GREY))
        if self.outside_region():
            out.append(("outside the region: no capture", ui.RED))
        return out

    def draw_right(self, img: np.ndarray) -> None:
        x, y, w, h = CAM_RECT
        from orvue_us_inverse.tracking.viewer import ZOOM
        ui.panel(img, CAM_RECT, f"CAMERA VIEW  |  D405  |  {ZOOM.label()} (z)")
        scene, _ = self.camera_image()
        if scene is not None:
            ui.fit(scene, CAM_RECT, img)
            if self.ar_on and self._ar is not None:
                cv2.rectangle(img, (x, y + h - 22), (x + w - 1, y + h - 1), (0, 0, 0), -1)
                ar.draw_key(img, x + 6, y + h - 7)
        else:
            text = "NO CAMERA" if self.tracker is None else "NO CAMERA FRAME"
            ui.text(img, text, (x + w // 2 - ui.text_w(text, 0.6) // 2, y + h // 2), ui.RED, 0.6)
            ui.text(img, "scripted and mouse sources work without it", (x + w // 2 - 160, y + h // 2 + 26), ui.GREY,
                    0.45)
        cap = f"AR {'ON' if self.ar_on else 'OFF'} (o)"
        cw = ui.text_w(cap, 0.4) + 12
        ui.text(img, cap, (x + w - cw, y - 8), ui.GREEN if self.ar_on else ui.DIM, 0.4)
        tx = x
        for text, colour in self.tracking_line():
            ui.text(img, text, (tx, TRACK_Y), colour, 0.4)
            tx += ui.text_w(text, 0.4) + 16

        mx, my, ms, _ = MAP_RECT
        ui.panel(img, MAP_RECT, "SWEEP MAP  |  coverage" + ("  |  planned lanes" if self.source == "scripted" else ""))
        top = self.top_image()
        img[my:my + ms, mx:mx + ms] = cv2.resize(top, (ms, ms), interpolation=cv2.INTER_AREA)
        sx, sw = mx + ms + G, x + w - (mx + ms + G)       # status column right of the map
        _, _, pct, n_holes = self.coverage()
        state = "complete" if self.completed else ("recording" if self.pose.recording else "ready")
        rows = [("STATE", state.upper(), ui.RED if state == "recording" else (ui.GREEN if state == "complete"
                                                                                else ui.AMBER)),
                ("PROBE", f"{self.pose.x:.0f}, {self.pose.y:.0f} mm", ui.WHITE),
                ("ANGLE", f"{self.pose.yaw:.0f} deg", ui.WHITE),
                ("FRAMES", f"{len(self.frames)}", ui.WHITE),
                ("STROKES", f"{len(self.strokes)}", ui.WHITE),
                ("COVERED", f"{pct:.1f} %", ui.WHITE),
                ("HOLES", f"{n_holes}", ui.RED if n_holes else ui.WHITE),
                ("SPEED GAPS", f"{len(self.gaps)}", ui.RED if self.gaps else ui.WHITE)]
        yy = my + 12
        for label, value, colour in rows:
            ui.text(img, label, (sx, yy), ui.DIM, 0.38)
            ui.text(img, value, (sx, yy + 18), colour, 0.48)
            yy += 48
        y0 = my + ms + 30
        ui.text(img, "SPEED", (mx, y0), ui.AMBER, 0.42)
        yy = self.speed_meter(img, mx, y0 + 8, w - 10)
        for k, line in enumerate(ui.wrap(self.message, w - 10, 0.42)[:2]):
            ui.text(img, line, (mx, yy + 8 + 18 * k), ui.GREY, 0.42)

    def compose(self) -> np.ndarray:
        img = np.full((HEIGHT, WIDTH, 3), ui.BG, np.uint8)
        src = SOURCE_LABELS[self.source]
        ui.header(img, "INVERSE ANATOMY MAPPING", f"{self.case}  |  source: {src}")
        cv2.line(img, (LEFT_W, ui.HEADER_H), (LEFT_W, HEIGHT - KEYS_H), ui.BORDER, 1)
        self.draw_left_panel(img)
        self.draw_bmode(img)
        self.draw_structures(img)
        self.draw_3d(img)
        self.draw_right(img)
        ui.key_bar(img, KEYS, HEIGHT - KEYS_H)
        return img

