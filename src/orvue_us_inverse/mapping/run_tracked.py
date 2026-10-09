"""
orvue_us_inverse.mapping.run_tracked - sweep with the camera-tracked dummy probe (S7); the mouse stays available.

    python -m orvue_us_inverse.mapping.run_tracked [--case normal] [--source realsense | <video / image folder>]
                                                    [--mouse] [--inject jitter=0.5,yaw=0.5,latency=50] [--no-images]
                                                    [--yaw 0] [--overlap 20] [--spacing 0.25]
    python -m orvue_us_inverse tracked [options]

Same window and keys as the mouse sweep (run_mouse.py: sweep | B-mode | 3D, status, speed meter, c complete, s save,
u undo, i B-mode on / off, views, 3 / b), with:
  pose source  CAMERA: the tracked dummy probe (tracking.tracker.ProbeTracker, filtered pose, calibration from
               config/calibration.json; x, y and yaw used with the probe upright on the surface, as the simulator's
               own tracked mode does); hold SPACE to record (Windows: the key's held state while the window has the
               focus or a space event just arrived; elsewhere space toggles recording); the camera sets the angle. MOUSE: the S6 behaviour (hold the left button on the
               sweep, wheel / q / e turn). Key m switches between them at any time (also to fall back on the mouse
               when tracking is not available); --mouse starts with the mouse. If the camera cannot be opened the
               app starts with the mouse and says why.
  tracking     a status line: tracking OK / reference board held / LOST, reference markers used (of 4), probe marker,
               reprojection errors (reference, probe, upright fit) in px, camera frame rate, the one-euro filter's lag
               at the current speed (errors.filter_lag) and the injected error, if any. No capture while tracking is
               invalid (the stroke continues when it returns).
  --inject     a PoseErrorModel (errors.py) applied live: frames are rendered at the tracked (or mouse) pose and
               reconstructed at the erroneous pose; latency uses a buffer of the recent poses.
"""
import argparse
import ctypes
import os
import sys
import time
import webbrowser

import cv2

from orvue_us_inverse.mapping.config import AcquisitionConfig, SweepConfig
from orvue_us_inverse.mapping.errors import LatencyBuffer, PoseErrorModel, filter_lag
from orvue_us_inverse.mapping.poses import MousePose, TrackedPose
from orvue_us_inverse.mapping.run_mouse import KEYS_H, STATUS_H, TOP_W, MouseSession
from orvue_us_inverse.simulation.anatomy import CASES
from orvue_us_inverse.ui import clinical as ui

WIN = "Tracked sweep"
TRACK_H = 34
VK_SPACE = 0x20


SPACE_EVENT_S = 0.7                # a space key event this recent also proves the window has the keyboard focus


def space_held() -> bool | None:
    """True while the space bar is held (Windows, GetAsyncKeyState); None where this is not available."""
    if sys.platform != "win32":
        return None
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(VK_SPACE) & 0x8000)
    except Exception:
        return None


def window_focused(window_title: str = WIN) -> bool:
    """True when the OpenCV window is the foreground window (Windows; True elsewhere or when it cannot be told)."""
    if sys.platform != "win32":
        return True
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        return hwnd == 0 or user32.GetForegroundWindow() == hwnd
    except Exception:
        return True


class TrackedSession(MouseSession):
    """Mouse-sweep session with a camera pose source (switchable to the mouse), tracking status and error injection."""

    EXTRA_H = TRACK_H
    HEIGHT = ui.HEADER_H + TOP_W + STATUS_H + TRACK_H + KEYS_H
    TITLE = ("Tracked sweep", "camera-tracked probe")
    MODE = "tracked"
    KEYS_ROW1 = [("hold SPACE", "record (camera)"), ("hold L", "record (mouse)"), ("m", "camera / mouse"),
                 ("i", "B-mode on/off"), ("u", "undo stroke"), ("r", "reset"), ("Esc", "quit")]

    def __init__(self, case: str, cfg: SweepConfig, acq_cfg: AcquisitionConfig, tracker=None,
                 error_model: PoseErrorModel | None = None, key_state=space_held, focus=None,
                 start_with_mouse: bool = False, clock=time.perf_counter, **kw):
        self.tracker = tracker
        self.error_model = error_model or PoseErrorModel()
        self.key_state = key_state
        self.focus = focus                     # callable: window has the keyboard focus (None: assume it has)
        self._space_event_t = None             # last space key event OpenCV delivered (wall clock of `clock`)
        self.buffer = LatencyBuffer()
        self._space_toggle = False
        self.latched = False                   # recording switched on by a button (clinical window), no key held
        super().__init__(case, cfg, acq_cfg, clock=clock, **kw)
        self.mouse_pose = self.pose
        self.tracked_pose = TrackedPose(tracker, clock=clock) if tracker is not None else None
        self.source = "camera" if (tracker is not None and not start_with_mouse) else "mouse"
        if self.source == "camera":
            self.pose = self.tracked_pose
        self.camera_error = ""
        self._lag_cache = {}

    # ---- pose source
    def set_source(self, source: str) -> bool:
        """'camera' or 'mouse' (not while recording); False when the camera is not available."""
        if self.pose.recording:
            self.message = "release the record key / button before switching the pose source"
            return False
        if source == "camera" and self.tracked_pose is None:
            self.message = f"camera not available{': ' + self.camera_error if self.camera_error else ''}; using the mouse"
            return False
        self.source = source
        self.latched = False
        self.pose = self.tracked_pose if source == "camera" else self.mouse_pose
        self.buffer.clear()
        self.acq.break_stretch()
        self.message = "pose from the CAMERA (hold space to record)" if source == "camera" else \
            "pose from the MOUSE (hold the left button on the sweep)"
        return True

    def toggle_source(self) -> bool:
        return self.set_source("mouse" if self.source == "camera" else "camera")

    def update_pose(self) -> None:
        if self.source != "camera":
            self._buffer_add()
            return
        tp = self.tracked_pose
        tp.update()
        held = self.key_state() if self.key_state is not None else None     # None: SPACE toggles (or latched)
        if held:                                 # the key is down: only ours when our window has the focus
            recent = self._space_event_t is not None and self.clock() - self._space_event_t <= SPACE_EVENT_S
            held = recent or self.focus is None or self.focus()
        rec = (self._space_toggle if held is None else held) or self.latched
        if rec and not tp.recording and not self.completed:
            tp.press()
            self.strokes.append([len(self.frames), len(self.frames)])
        elif not rec and tp.recording:
            tp.release()
            if self.strokes and self.strokes[-1][0] == self.strokes[-1][1]:
                self.strokes.pop()
        self._buffer_add()

    def space_event(self) -> None:
        """OpenCV delivered a space key event: the window has the focus; where the held state cannot be read
        (non-Windows) it toggles recording."""
        self._space_event_t = self.clock()
        if self.key_state is None or self.key_state() is None:
            self._space_toggle = not self._space_toggle

    def toggle_space(self) -> None:
        """Space pressed where its held state cannot be read: toggles recording."""
        self._space_toggle = not self._space_toggle

    def outside_region(self) -> bool:
        """Recording with the probe centre off the 100 x 100 mm region (no capture there)."""
        return self.pose.recording and not (0.0 <= self.pose.x <= 100.0 and 0.0 <= self.pose.y <= 100.0)

    def _buffer_add(self) -> None:
        smp = self.pose.sample()
        if smp is not None:
            self.buffer.add(smp.t, smp.T)

    def angle_hint(self) -> str:
        return "(from the camera)" if self.source == "camera" else super().angle_hint()

    def report_extra(self) -> dict:
        return dict(pose_source=self.source, injected_error=self.error_model.to_dict(),
                    injected_error_text=self.error_model.describe())

    def measure(self, frame) -> None:
        if not self.error_model.is_zero:
            frame.T_measured = self.error_model.apply(frame.T_true, frame.t, self.buffer.pose_at)

    def route_mouse(self, event: int, wx: int, wy: int, flags: int = 0) -> None:
        if self.source == "camera" and self._inside(self.RECT_TOP, wx, wy) and self._orbit is None:
            return                                           # the camera drives the probe; the 3D panel still works
        super().route_mouse(event, wx, wy, flags)

    def turn(self, sign: int) -> None:
        if self.source == "camera":
            self.message = "the camera sets the probe angle (m switches to the mouse)"
            return
        super().turn(sign)

    def set_angle(self, yaw_deg: float) -> None:
        if self.source == "camera":
            self.message = "the camera sets the probe angle (m switches to the mouse)"
            return
        super().set_angle(yaw_deg)

    # ---- status
    def lag_ms(self, speed: float) -> float:
        key = round(max(speed, 0.5), 1)
        if key not in self._lag_cache:
            self._lag_cache[key] = filter_lag(key, duration_s=4.0)["lag_ms"]
        return self._lag_cache[key]

    def tracking_items(self) -> list[tuple[str, tuple]]:
        """(text, colour) items of the tracking line."""
        out = [(f"POSE: {'CAMERA' if self.source == 'camera' else 'MOUSE'}  (m)", ui.AMBER)]
        if self.pose.recording:
            out.append(("REC", ui.RED))
        st = self.tracked_pose.state if self.tracked_pose is not None else None
        if self.tracked_pose is None:
            out.append((f"camera not available{': ' + self.camera_error if self.camera_error else ''}", ui.GREY))
        elif st is None:
            out.append(("waiting for the camera", ui.GREY))
        else:
            if st.valid:
                out.append(("TRACKING: board held" if st.phantom_held else "TRACKING: OK",
                            ui.AMBER if st.phantom_held else ui.GREEN))
            else:
                out.append(("TRACKING: LOST", ui.RED))
            out.append((f"reference {st.n_reference_markers_used}/4", ui.WHITE if st.n_reference_markers_used == 4
                        else ui.AMBER))
            out.append((f"probe ID 0 {'seen' if st.probe_valid else 'not seen'}", ui.WHITE if st.probe_valid else ui.RED))
            rp = [f"{v:.2f}" if v is not None else "-" for v in (st.reproj_error_ref_px, st.reproj_error_probe_px,
                                                                 st.reproj_error_upright_px)]
            out.append((f"reproj ref / probe / upright {' / '.join(rp)} px", ui.GREY))
            tp = self.tracked_pose
            if tp.face_z_mm is not None:
                out.append((f"face z {tp.face_z_mm:+.1f} mm, tilt {tp.tilt_deg or 0:.0f} deg", ui.GREY))
            out.append((f"{st.fps:.0f} fps", ui.GREY))
            v, _ = self.speed()
            out.append((f"filter lag {self.lag_ms(v):.0f} ms", ui.GREY))
        if self.outside_region():
            out.append(("probe outside the 100 mm region: no capture", ui.RED))
        if not self.error_model.is_zero:
            out.append((f"inject: {self.error_model.describe()}", ui.RED))
        return out

    def compose(self):
        img = super().compose()
        y0 = ui.HEADER_H + TOP_W + STATUS_H
        img[y0:y0 + TRACK_H] = ui.BG
        cv2.line(img, (0, y0), (self.WIDTH - 1, y0), ui.BORDER, 1)
        x = 14
        for text, colour in self.tracking_items():
            ui.text(img, text, (x, y0 + 22), colour, 0.42)
            x += ui.text_w(text, 0.42) + 22
        return img


def parse_args(argv=None) -> argparse.Namespace:
    d = SweepConfig()
    p = argparse.ArgumentParser(description="Sweep with the camera-tracked dummy probe (the mouse stays available).")
    p.add_argument("--case", default="normal", choices=sorted(CASES))
    p.add_argument("--source", default="realsense", help="'realsense' or a video file / image folder")
    p.add_argument("--mouse", action="store_true", help="start with the mouse as pose source")
    p.add_argument("--inject", default="", help="pose errors, e.g. jitter=0.5,yaw=0.5,latency=50 (errors.py)")
    p.add_argument("--yaw", type=float, nargs="+", default=d.yaw_list_deg, help="orientations of the lane guides")
    p.add_argument("--overlap", type=float, default=d.overlap_pct)
    p.add_argument("--spacing", type=float, default=d.frame_spacing_mm)
    p.add_argument("--no-images", action="store_true", help="start with B-mode off")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = SweepConfig(yaw_list_deg=a.yaw, overlap_pct=a.overlap, frame_spacing_mm=a.spacing)
    model = PoseErrorModel.parse(a.inject)
    tracker, cam_err = None, ""
    try:
        from orvue_us_inverse.tracking.tracker import ProbeTracker
        tracker = ProbeTracker(source=a.source)
    except Exception as e:                       # no camera / pyrealsense2 / file
        cam_err = f"{type(e).__name__}: {e}"
        print(f"[run_tracked] camera not available ({cam_err}); starting with the mouse", flush=True)
    app = TrackedSession(a.case, cfg, AcquisitionConfig(store_images=not a.no_images), tracker=tracker,
                         error_model=model, focus=window_focused, start_with_mouse=a.mouse)
    app.camera_error = cam_err
    if not app.init_view3d():
        print(f"[run_tracked] 3D view not available: {app.view3d_error}", flush=True)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, lambda ev, x, y, fl, _: app.route_mouse(ev, x, y, fl))
    view_state = None
    try:
        while True:
            app.step()
            due, state = app.live3d_due(view_state)
            if due:
                app.refresh_view3d()
                view_state = state
            cv2.imshow(WIN, app.compose())
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord(" "):
                app.space_event()
            elif key == ord("m"):
                app.toggle_source()
            elif key in (ord("q"), ord("e")):
                app.turn(-1 if key == ord("q") else 1)
            elif key in (ord("0"), ord("9")):
                app.set_angle(0.0 if key == ord("0") else 90.0)
            elif key == ord("i"):
                app.toggle_bmode()
            elif key == ord("v"):
                app.next_view()
            elif key == ord("u"):
                if not app.undo():
                    app.message = "nothing to undo (or still recording)"
            elif key == ord("r"):
                app.restart()
                app.message = "reset"
            elif key == ord("s"):
                print(f"[run_tracked] {app.save()}", flush=True)
            elif key == ord("a"):
                if app._revealed_img is None:
                    app.message = "computing the anatomy top view ..."
                    cv2.imshow(WIN, app.compose())
                    cv2.waitKey(1)
                app.toggle_revealed()
                app.message = ""
            elif key == ord("g"):
                app.toggle_gt_contours()
            elif key == ord("3") and app.frames:
                png, stls = app.export_3d()
                print(f"[run_tracked] {png}", flush=True)
            elif key == ord("b") and app.frames:
                path = app.browser_view()
                webbrowser.open("file:///" + os.path.abspath(path).replace(os.sep, "/"))
            elif key == ord("c") and not app.completed and app.frames:
                app.message = "complete: evaluating against the ground truth ..."
                cv2.imshow(WIN, app.compose())
                cv2.waitKey(1)
                folder, table = app.complete()
                print(table, flush=True)
                print(f"[run_tracked] report: {folder}", flush=True)
                if hasattr(os, "startfile"):
                    os.startfile(folder)
            try:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        if tracker is not None:
            tracker.stop()
        if app.view3d is not None:
            app.view3d.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
