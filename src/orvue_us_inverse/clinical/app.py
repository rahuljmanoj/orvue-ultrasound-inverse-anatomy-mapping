"""
orvue_us_inverse.clinical.app - the Ultrasound Imaging Simulator: one window for B-mode imaging, inverse anatomy
mapping and the probe calibration.

    python -m orvue_us_inverse                    (opens this window)
    python -m orvue_us_inverse.clinical [--tab bmode | mapping] [--case normal] [--no-camera] [--source realsense]

One window, "Ultrasound Imaging Simulator":
  header           CASE selector (drop-down, both tabs; n / p in the B-MODE tab) and "Calibrate probe".
  IMAGING MODE tabs in the left panel (click, or Tab):
  B-MODE           the simulator of the developer menu's step 3 (simulation.bmode.demo, unchanged layout and keys):
                   B-mode display with sliders, anatomy map, readouts, D405 camera view; m camera / mouse, t camera
                   view, z zoom, wheel / q / e rotate, g contours, c contact, v 3D anatomy viewer.
  INVERSE MAPPING  clinical.mapping_tab.MappingSession: scripted sweep / mouse / camera probe as pose source (SPACE or
                   RECORD switches recording on / off), live B-mode, the 3D reconstruction in the centre, the camera
                   view with the AR overlay of the reconstructed structures, sweep map with coverage, status, speed.
  DOPPLER, ELASTOGRAPHY  placeholders (to do).
  Calibration      tracking.calibrate.CalibrationRoutine on the shared camera, shown in the same window (SPACE
                   capture, y / n save / discard, Esc abort, Enter back).

The camera (ProbeTracker) is opened once and shared; without it the window runs with the mouse. A case change starts
a new mapping (not while recording). Esc in an imaging tab closes the window.
"""
import argparse
import os
import threading
import time
import webbrowser

import cv2
import numpy as np

from orvue_us_inverse.clinical.mapping_tab import HEIGHT, WIDTH, MappingSession, draw_mode_tabs, hit_mode_tab
from orvue_us_inverse.mapping.run_mouse import wheel_delta
from orvue_us_inverse.simulation import bmode
from orvue_us_inverse.simulation.anatomy import CASES, build_case
from orvue_us_inverse.simulation.bmode import (B_IMAGE_H, BModeSimulator, SliderPanel, apply_controls, camera_scene,
                                              clinical_view, draw_calot_triangle, draw_depth_legend, get_controls,
                                              left_panel_free_y, simulator_keys, simulator_window, top_view)
from orvue_us_inverse.ui import clinical as ui

WIN = bmode.SIM_WINDOW                       # "Ultrasound Imaging Simulator"
N_STATUS = 10                                # status lines of the B-mode panel (as bmode.demo)
TAB_KEY = 9


class BModeTab:
    """The B-mode imaging mode: bmode.demo's loop body as an object (same layout, keys and mouse), with the
    IMAGING MODE tabs of the clinical window (INVERSE MAPPING enabled)."""

    def __init__(self, case: str = "normal", tracker=None, cam_view: bool = True, camera_control: bool = True):
        self.tracker = tracker
        self.names = list(CASES)
        self.state = {"x": 55.0, "y": 70.0, "yaw": 0.0, "gt": False, "contact": True,
                      "case": self.names.index(case), "locked": False, "cam_view": bool(cam_view and tracker),
                      "camera": bool(camera_control), "map_rect": None, "viewer_thread": None}
        self.controls = None
        self.load(self.state["case"])
        self.controls = get_controls(self.sim)
        self.sliders = SliderPanel(self.controls, self.sim.pr.depth_mm)
        self.keys = [("Tab", "inverse mapping")] + [("wheel / q / e", a) if k == "q / e" else (k, a)
                                                    for k, a in simulator_keys(tracker)]
        self.zoom = self.zoom_sliders = None
        if tracker is not None:
            from orvue_us_inverse.tracking.viewer import ZOOM
            self.zoom = ZOOM
            self.zoom_sliders = ui.Sliders([
                ("zoom", "Zoom", 1.0, ZOOM.MAX_ZOOM, 0.1, lambda v: f"{v:.1f} x"),
                ("cx", "Pan x", 0.0, 1.0, 0.01, lambda v: f"{(v - 0.5) * 200:+.0f} %"),
                ("cy", "Pan y", 0.0, 1.0, 0.01, lambda v: f"{(v - 0.5) * 200:+.0f} %")])
        self.modes_y = left_panel_free_y(self.sliders, N_STATUS) + 16
        self.viewer_button = None
        self.t_last = time.time()

    @property
    def case(self) -> str:
        return self.names[self.state["case"]]

    def load(self, i: int) -> None:
        an = build_case(self.names[i])
        sim = BModeSimulator(an)
        if self.controls is not None:
            apply_controls(sim, self.controls)
        tv = top_view(an)
        sc = 600 / tv.shape[0]
        tv = cv2.resize(tv, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
        draw_calot_triangle(tv, an, sc / 0.25)
        draw_depth_legend(tv)
        self.an, self.sim, self.tv, self.scale_tv = an, sim, tv, sc

    def open_viewer(self) -> None:
        """3D anatomy viewer (browser) on the current case with the probe plane, in the background."""
        st = self.state
        if st["viewer_thread"] is not None and st["viewer_thread"].is_alive():
            return
        hash_state = f"case={self.an.name}&probe=1&px={st['x']:.1f}&py={st['y']:.1f}&pyaw={st['yaw']:.0f}"

        def run():
            try:
                from orvue_us_inverse.viewer3d.viewer import open_page
                open_page(hash_state)
            except Exception as e:                       # never stop the window for the viewer
                print(f"[WARN] could not open the 3D anatomy viewer: {e}", flush=True)

        st["viewer_thread"] = threading.Thread(target=run, daemon=True)
        st["viewer_thread"].start()

    def on_mouse(self, ev, mx, my, flags=0) -> str | None:
        """Mouse event; returns the imaging mode clicked ('INVERSE MAPPING' ...) or None."""
        st = self.state
        if self.sliders.on_mouse(ev, mx, my):
            return None
        if self.zoom_sliders is not None and st["cam_view"] and self.zoom_sliders.on_mouse(ev, mx, my):
            self.zoom.zoom, self.zoom.cx, self.zoom.cy = (self.zoom_sliders.values[k] for k in ("zoom", "cx", "cy"))
            return None
        if ev == cv2.EVENT_MOUSEWHEEL:                     # wheel: turn the probe 5 deg (as q / e)
            st["yaw"] += 5 if wheel_delta(flags) > 0 else -5
            return None
        if ev == cv2.EVENT_LBUTTONDOWN:
            mode = hit_mode_tab(self.modes_y, mx, my)
            if mode is not None:
                return mode
            b = self.viewer_button
            if b and b[0] <= mx <= b[0] + b[2] and b[1] <= my <= b[1] + b[3]:
                self.open_viewer()
                return None
        if st["map_rect"] is None:
            return None
        ox, oy, ms = st["map_rect"]
        if not (0 <= mx - ox < ms and 0 <= my - oy < ms):
            return None
        px, py = (mx - ox) * self.tv.shape[1] / ms, (my - oy) * self.tv.shape[0] / ms
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["locked"] = not st["locked"]
        elif st["locked"]:
            return None
        st["x"], st["y"] = px / self.scale_tv * 0.25, py / self.scale_tv * 0.25
        return None

    def on_key(self, k: int) -> None:
        st = self.state
        if k == ord("q"):
            st["yaw"] -= 5
        elif k == ord("e"):
            st["yaw"] += 5
        elif k == ord("g"):
            st["gt"] = not st["gt"]
        elif k == ord("c"):
            st["contact"] = not st["contact"]
        elif k == ord("t") and self.tracker is not None:
            st["cam_view"] = not st["cam_view"]
        elif k == ord("z") and self.tracker is not None:
            self.zoom.toggle()
        elif k == ord("m") and self.tracker is not None:
            st["camera"] = not st["camera"]
        elif k == ord("v"):
            self.open_viewer()
        elif k in (ord("n"), ord("p")):
            st["case"] = (st["case"] + (1 if k == ord("n") else -1)) % len(self.names)
            self.load(st["case"])

    def frame(self) -> np.ndarray:
        """One loop of bmode.demo: pose (camera or mouse), render, the whole window."""
        st, sim, tracker = self.state, self.sim, self.tracker
        if self.sliders.values != self.controls:
            self.controls = apply_controls(sim, self.sliders.values)
            if self.sliders.active is None:
                self.sliders.values = dict(self.controls)
        use_camera = tracker is not None and st["camera"] and not st["locked"]
        tracked = tracker.get_xy_yaw() if use_camera else None
        if tracked is not None:
            st["x"], st["y"], st["yaw"] = tracked
        T = sim.pose_from_xy_yaw(st["x"], st["y"], st["yaw"], 0.0 if st["contact"] else 5.0)
        o, u = T[:3, 3], T[:3, 0]
        probe_px = ((o[:2] - u[:2] * sim.pr.width_mm / 2) / 0.25 * self.scale_tv,
                    (o[:2] + u[:2] * sim.pr.width_mm / 2) / 0.25 * self.scale_tv)
        if sim.in_contact(T):
            img, lab = sim.render(T, return_labels=True)
            us = sim.ground_truth_overlay(img, lab) if st["gt"] else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            us = np.zeros((sim.nz, sim.nx, 3), np.uint8)
            cv2.putText(us, "no contact", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        us = cv2.resize(us, (int(round(B_IMAGE_H * sim.nx / sim.nz)), B_IMAGE_H))
        t = time.time()
        fps = 1 / max(t - self.t_last, 1e-6)
        self.t_last = t
        track = "off" if tracker is None else ("mouse" if not st["camera"] else ("on" if tracked else "lost"))
        status = [f"FR     {fps:4.1f} Hz", "", f"Probe  {'frozen' if st['locked'] else 'live'}", f"Track  {track}",
                  f"x      {st['x']:.1f} mm", f"y      {st['y']:.1f} mm", f"yaw    {st['yaw']:.0f} deg", "",
                  f"Case   {self.an.name}", f"GT     {'on' if st['gt'] else 'off'}"]
        tracking, track_state = None, (tracker.get_state() if tracker is not None else None)
        if st["cam_view"]:
            scene, track_state = camera_scene(tracker)
            tracking = (scene, (st["x"], st["y"], st["yaw"]), (sim.pr.width_mm, 10.0))
        if self.zoom_sliders is not None and self.zoom_sliders.active is None:
            self.zoom_sliders.values.update(zoom=self.zoom.zoom, cx=self.zoom.cx, cy=self.zoom.cy)
        busy = st["viewer_thread"] is not None and st["viewer_thread"].is_alive()
        window, st["map_rect"], _ = simulator_window(
            clinical_view(us, sim, status, self.sliders), self.tv, probe_px, self.an, st["yaw"], st["locked"],
            self.keys, camera=st["camera"] if tracker is not None else None, track_state=track_state,
            tracking=tracking, zoom_sliders=self.zoom_sliders, modes_y=None)
        y_tools = draw_mode_tabs(window, 10, self.modes_y, "B-MODE") + 14       # imaging modes + tools
        ui.section(window, 10, y_tools, "TOOLS")
        self.viewer_button = ui.button(window, (10, y_tools + 10, 176, 30), "3D anatomy viewer", "v", busy=busy)
        return window


class CaseDropdown:
    """Case selector in the window header (both imaging tabs): a box with the current case; a click opens the list
    of anatomy cases, a click on one selects it."""

    RECT = (1130, 12, 260, 32)
    ROW_H = 28

    def __init__(self, names):
        self.names = list(names)
        self.open = False
        self.hover = None

    @staticmethod
    def label(name: str) -> str:
        return name.replace("_", " ")

    def rows(self) -> list[tuple[str, tuple]]:
        x, y, w, h = self.RECT
        return [(n, (x, y + h + 2 + k * self.ROW_H, w, self.ROW_H)) for k, n in enumerate(self.names)]

    @staticmethod
    def _inside(rect, mx, my) -> bool:
        x, y, w, h = rect
        return x <= mx <= x + w and y <= my <= y + h

    def on_mouse(self, ev, mx, my) -> tuple[bool, str | None]:
        """(event used, selected case or None)."""
        if ev == cv2.EVENT_MOUSEMOVE and self.open:
            self.hover = next((n for n, r in self.rows() if self._inside(r, mx, my)), None)
            return True, None
        if ev != cv2.EVENT_LBUTTONDOWN:
            return self.open and ev == cv2.EVENT_LBUTTONUP, None
        if self._inside(self.RECT, mx, my):
            self.open = not self.open
            return True, None
        if self.open:
            self.open = False
            pick = next((n for n, r in self.rows() if self._inside(r, mx, my)), None)
            return True, pick
        return False, None

    def draw(self, img: np.ndarray, current: str, enabled: bool = True) -> None:
        x, y, w, h = self.RECT
        cv2.rectangle(img, (x, y), (x + w, y + h), ui.PANEL_LIGHT, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), ui.AMBER if (self.open or enabled) else ui.BORDER, 1)
        ui.text(img, "CASE", (x + 10, y + 21), ui.AMBER, 0.42)
        ui.text(img, self.label(current), (x + 58, y + 21), ui.WHITE if enabled else ui.GREY, 0.48)
        tri = np.array([[x + w - 22, y + 13], [x + w - 10, y + 13], [x + w - 16, y + 21]], np.int32)
        if self.open:                                    # pointing up while the list is open
            tri = np.array([[x + w - 22, y + 21], [x + w - 10, y + 21], [x + w - 16, y + 13]], np.int32)
        cv2.fillConvexPoly(img, tri, ui.GREY, cv2.LINE_AA)
        if not self.open:
            return
        rows = self.rows()
        x0, y0 = rows[0][1][0], rows[0][1][1]
        y1 = rows[-1][1][1] + self.ROW_H
        cv2.rectangle(img, (x0, y0), (x0 + w, y1), ui.PANEL, -1)
        cv2.rectangle(img, (x0, y0), (x0 + w, y1), ui.AMBER, 1)
        for n, (rx, ry, rw, rh) in rows:
            if n == self.hover:
                cv2.rectangle(img, (rx + 1, ry), (rx + rw - 1, ry + rh), ui.PANEL_LIGHT, -1)
            colour = ui.AMBER if n == current else (ui.WHITE if n == self.hover else ui.GREY)
            ui.text(img, self.label(n), (rx + 12, ry + 19), colour, 0.46)


CALIBRATE_RECT = (1402, 12, 200, 32)          # header button, both imaging tabs
CAL_KEYS = [("SPACE", "capture 2 s"), ("y / n", "save / discard"), ("Esc", "abort calibration"),
            ("Enter", "back to imaging")]


class CalibrationView:
    """Probe calibration inside the clinical window (tracking.calibrate.CalibrationRoutine on the shared tracker):
    camera scene with the tracking overlay, phantom map, readouts and the prompts; SPACE captures, y / n save or
    discard the correction, Esc aborts; Back (Enter) returns to the imaging tab."""

    BACK_RECT = (14, HEIGHT - 56, 300, 36)

    def __init__(self, tracker, camera_error: str = ""):
        self.tracker = tracker
        self.camera_error = camera_error
        self.routine = None
        if tracker is not None:
            from orvue_us_inverse.tracking.calibrate import CalibrationRoutine
            self.routine = CalibrationRoutine(tracker)

    @property
    def finished(self) -> bool:
        return self.routine is None or self.routine.finished

    def on_key(self, k: int) -> bool:
        """Handle a key; True when the view should close (back to imaging)."""
        if self.finished:
            return k in (13, 27, ord("b"))
        self.routine.handle_key(k)
        return False

    def on_mouse(self, ev, mx, my) -> bool:
        """True when Back was clicked (aborts a running calibration first)."""
        x, y, w, h = self.BACK_RECT
        if ev == cv2.EVENT_LBUTTONDOWN and x <= mx <= x + w and y <= my <= y + h:
            if not self.finished:
                self.routine.handle_key(27)
            return True
        return False

    def frame(self) -> np.ndarray:
        from orvue_us_inverse.tracking.viewer import FOOTPRINT_MM, annotate
        canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        if self.tracker is None:
            ui.header(canvas, "PROBE CALIBRATION", "camera not connected")
            lines = ["The calibration needs the D405 camera and the tracking board.",
                     f"Camera: {self.camera_error or 'not available'}.",
                     "Connect the camera and restart the window (python -m orvue_us_inverse)."]
            for k, line in enumerate(lines):
                ui.text(canvas, line, (40, 160 + 32 * k), ui.GREY if k else ui.WHITE, 0.6)
        else:
            self.routine.update()
            frame, det, s = self.tracker.get_frame()
            scene = None
            if frame is not None:
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                scene = annotate(frame, det, s, self.tracker)
            img = ui.compose_tracking(scene, s, title="PROBE CALIBRATION", subtitle="yaw offset and face position",
                                      keys=CAL_KEYS, banner=self.routine.prompt_lines(),
                                      footprint_mm=FOOTPRINT_MM, scene_h=540)
            h, w = min(img.shape[0], HEIGHT - 70), min(img.shape[1], WIDTH)
            canvas[:h, :w] = img[:h, :w]
            cv2.line(canvas, (w, ui.HEADER_H - 1), (WIDTH - 1, ui.HEADER_H - 1), ui.BORDER, 1)
        label = "Back to imaging  (Enter)" if self.finished else "Abort and back  (Esc, then Enter)"
        ui.button(canvas, self.BACK_RECT, label)
        return canvas


class ClinicalApp:
    """The clinical window: B-MODE and INVERSE MAPPING tabs sharing one camera tracker, the case selector and the
    probe calibration in the header."""

    def __init__(self, case: str = "normal", tracker=None, tab: str = "bmode", camera_error: str = ""):
        self.tracker = tracker
        self.camera_error = camera_error
        self.bmode = BModeTab(case, tracker, cam_view=True)
        self.dropdown = CaseDropdown(self.bmode.names)
        self.calibration: CalibrationView | None = None
        self.mapping: MappingSession | None = None
        self.mapping_case = None
        self.view_state = None
        self.tab = "bmode"
        if tab == "mapping":
            self.switch("mapping")

    @property
    def recording(self) -> bool:
        return self.tab == "mapping" and self.mapping.pose.recording

    def switch(self, tab: str) -> None:
        """Show the tab; the mapping session is created on first use and anew after a case change (keeping the
        pose source)."""
        if tab == "mapping" and (self.mapping is None or self.mapping_case != self.bmode.case):
            source = None
            if self.mapping is not None:
                source = self.mapping.source
                if self.mapping.view3d is not None:
                    self.mapping.view3d.close()
            self.mapping = MappingSession(self.bmode.case, tracker=self.tracker, modes_y=self.bmode.modes_y,
                                          source=source)
            self.mapping.camera_error = self.camera_error
            if not self.mapping.init_view3d():
                print(f"[clinical] 3D view not available: {self.mapping.view3d_error}", flush=True)
            self.mapping_case, self.view_state = self.bmode.case, None
        self.tab = tab

    def set_case(self, name: str) -> bool:
        """Select the anatomy case for both tabs (a new mapping starts); not while recording."""
        if self.recording:
            self.mapping.message = "stop recording before changing the case"
            return False
        if name == self.bmode.case:
            return True
        self.bmode.state["case"] = self.bmode.names.index(name)
        self.bmode.load(self.bmode.state["case"])
        if self.tab == "mapping":
            self.switch("mapping")
            self.mapping.message = f"case {CaseDropdown.label(name)}: new mapping"
        return True

    def open_calibration(self) -> bool:
        if self.recording:
            self.mapping.message = "stop recording before the calibration"
            return False
        self.calibration = CalibrationView(self.tracker, self.camera_error)
        return True

    def close_calibration(self) -> None:
        self.calibration = None
        if self.tracker is not None:
            self.tracker.reset_filters()                 # the calibration may have changed

    def on_mouse(self, ev, x, y, flags=0) -> None:
        if self.calibration is not None:
            if self.calibration.on_mouse(ev, x, y):
                self.close_calibration()
            return
        used, pick = self.dropdown.on_mouse(ev, x, y)
        if pick is not None:
            self.set_case(pick)
        if used:
            return
        if ev == cv2.EVENT_LBUTTONDOWN and CaseDropdown._inside(CALIBRATE_RECT, x, y):
            self.open_calibration()
            return
        if self.tab == "bmode":
            mode = self.bmode.on_mouse(ev, x, y, flags)
            if mode == "INVERSE MAPPING":
                self.switch("mapping")
            return
        if ev == cv2.EVENT_LBUTTONDOWN and hit_mode_tab(self.mapping.modes_y, x, y) == "B-MODE":
            if self.recording:
                self.mapping.message = "stop recording before leaving the mapping"
            else:
                self.switch("bmode")
            return
        self.mapping.route_mouse(ev, x, y, flags)

    def frame(self) -> np.ndarray:
        if self.calibration is not None:
            return self.calibration.frame()
        if self.tab == "bmode":
            img = self.bmode.frame()
        else:
            m = self.mapping
            m.step()
            due, state = m.live3d_due(self.view_state)
            if due:
                m.refresh_view3d()
                self.view_state = state
            img = m.compose()
        if img.shape[:2] != (HEIGHT, WIDTH):                 # both tabs share one window size
            canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            h, w = min(img.shape[0], HEIGHT), min(img.shape[1], WIDTH)
            canvas[:h, :w] = img[:h, :w]
            img = canvas
        self.draw_header_controls(img)
        return img

    def draw_header_controls(self, img: np.ndarray) -> None:
        """Calibrate button and the case selector (drawn last: the open list lies over the panels)."""
        ui.button(img, CALIBRATE_RECT, "Calibrate probe")
        self.dropdown.draw(img, self.bmode.case, enabled=not self.recording)

    def show_now(self, message: str) -> None:
        """Show a message before a slow action (evaluation)."""
        self.mapping.message = message
        img = self.mapping.compose()
        self.draw_header_controls(img)
        cv2.imshow(WIN, img)
        cv2.waitKey(1)

    def run_request(self) -> None:
        """Actions the mapping tab's buttons asked for (they need the window: messages, files, browser)."""
        m = self.mapping
        req, m.request = m.request, None
        if req == "complete":
            self.complete()
        elif req == "save":
            if m.frames:
                print(f"[clinical] {m.save()}", flush=True)
            else:
                m.message = "nothing to save yet"
        elif req == "browser" and m.frames:
            path = m.browser_view()
            webbrowser.open("file:///" + os.path.abspath(path).replace(os.sep, "/"))
        elif req == "export" and m.frames:
            png, _ = m.export_3d()
            print(f"[clinical] {png}", flush=True)
        elif req in ("browser", "export"):
            m.message = "record a sweep first"

    def complete(self) -> None:
        m = self.mapping
        if m.completed or not m.frames:
            m.message = "record a sweep first" if not m.frames else "already complete: r starts a new sweep"
            return
        self.show_now("complete: filling small holes and evaluating against the ground truth ...")
        folder, table = m.complete()
        print(table, flush=True)
        print(f"[clinical] report: {folder}", flush=True)

    def on_key(self, k: int) -> bool:
        """Handle a key; False when the window should close (Esc in an imaging tab)."""
        if self.calibration is not None:
            if self.calibration.on_key(k):
                self.close_calibration()
            return True
        if k == 27:
            if self.dropdown.open:
                self.dropdown.open = False
                return True
            return False
        if k == TAB_KEY:
            if self.recording:
                self.mapping.message = "stop recording before leaving the mapping"
            else:
                self.switch("mapping" if self.tab == "bmode" else "bmode")
            return True
        if self.tab == "bmode":
            if k in (ord("n"), ord("p")):
                i = (self.bmode.state["case"] + (1 if k == ord("n") else -1)) % len(self.bmode.names)
                self.set_case(self.bmode.names[i])
            else:
                self.bmode.on_key(k)
            return True
        m = self.mapping
        keys = {ord("1"): "scripted", ord("2"): "mouse", ord("3"): "camera"}
        if k in keys:
            m.set_source(keys[k])
        elif k == ord(" "):
            m.toggle_record()
        elif k in (ord("q"), ord("e")):
            m.turn(-1 if k == ord("q") else 1)
        elif k in (ord("0"), ord("9")):
            m.set_angle(0.0 if k == ord("0") else 90.0)
        elif k == ord("i"):
            m.toggle_bmode()
        elif k == ord("o"):
            m.toggle_ar()
        elif k == ord("v"):
            m.next_view()
        elif k == ord("g"):
            m.toggle_gt_contours()
        elif k == ord("z") and self.tracker is not None:
            from orvue_us_inverse.tracking.viewer import ZOOM
            ZOOM.toggle()
        elif k == ord("u"):
            m.press_button("undo")
        elif k == ord("r"):
            m.press_button("reset")
        elif k == ord("a"):
            if m._revealed_img is None:
                self.show_now("computing the anatomy top view ...")
            m.toggle_revealed()
            m.message = ""
        elif k == ord("c"):
            self.complete()
        elif k == ord("s"):
            m.request = "save"
        elif k == ord("b"):
            m.request = "browser"
        elif k == ord("x"):
            m.request = "export"
        if m.request:
            self.run_request()
        return True

    def close(self) -> None:
        if self.mapping is not None and self.mapping.view3d is not None:
            self.mapping.view3d.close()


def open_tracker(source: str = "realsense"):
    """(ProbeTracker or None, reason when None)."""
    try:
        from orvue_us_inverse.tracking.tracker import ProbeTracker
        return ProbeTracker(source=source), ""
    except Exception as e:                           # no camera / pyrealsense2 / file
        return None, f"{type(e).__name__}: {e}"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ultrasound Imaging Simulator (clinical window): B-mode, inverse anatomy "
                                            "mapping and probe calibration in one window.")
    p.add_argument("--tab", default="bmode", choices=("bmode", "mapping"), help="imaging mode at start")
    p.add_argument("--case", default="normal", choices=list(CASES))
    p.add_argument("--source", default="realsense", help="camera: 'realsense' or a video file / image folder")
    p.add_argument("--no-camera", action="store_true", help="do not open the camera (mouse only)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    tracker, err = (None, "not opened (--no-camera)") if a.no_camera else open_tracker(a.source)
    if tracker is None:
        print(f"[clinical] camera not available ({err}); the mouse drives the probe", flush=True)
    app = ClinicalApp(a.case, tracker, tab=a.tab, camera_error=err)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, lambda ev, x, y, fl, _: app.on_mouse(ev, x, y, fl))
    try:
        while True:
            img = app.frame()
            if app.tab == "mapping" and app.calibration is None and app.mapping.request:
                app.run_request()
            cv2.imshow(WIN, img)
            k = cv2.waitKey(1) & 0xFF
            if k != 255 and not app.on_key(k):
                break
            try:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        app.close()
        if tracker is not None:
            tracker.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
