"""
orvue_us_inverse.tracking.viewer - live window for checking the probe tracking.

    python -m orvue_us_inverse viewer [--source realsense|video|folder] [--exposure-us 3000] [--no-second-marker]

Clinical-style window (see orvue_us_inverse.ui.clinical): Orvue Surgical header; the camera scene with the tracking overlay
(detected markers, the 100 x 100 mm region projected from the markers only - amber at z = H_MM on the printed
sheet, white at z = 0 when H_MM != 0 - and probe axes x red, y green, z blue, 30 mm, at the ID 0 marker and at
the computed face centre); the phantom map with the probe footprint; status, probe face, quality and keys.

Keys:  s  save the frame and state to output/captures/<timestamp>/ (frame.png, annotated.png, state.json,
          intrinsics.json; the folder can be replayed with --source)
       r  start / stop CSV logging of the state at frame rate (output/logs/track_<timestamp>.csv)
       c  run the probe calibration (see orvue_us_inverse.tracking.calibrate); its prompts appear in the footer
       z  camera scene: fit to the board (automatic the first time the board is seen) / full frame
       q  quit (Esc too)
"""
import argparse
import csv
import json
import math
import os
import time

import cv2
import numpy as np

from orvue_us_inverse.paths import CAPTURES_DIR, LOGS_DIR
from orvue_us_inverse.ui import clinical as ui
from orvue_us_inverse.tracking import markers as tm
from orvue_us_inverse.tracking.calibrate import CalibrationRoutine
from orvue_us_inverse.tracking.tracker import ProbeTracker, TrackerConfig, TrackState, save_intrinsics

WINDOW = "Probe tracking"
AXIS_MM = 30.0
FOOTPRINT_MM = (30.0, 10.0)          # probe footprint on the map (along x, across y); simulation.bmode.LinearProbe width
KEY_LEGEND = [("s", "save frame + state"),
              ("r", "CSV log on / off"),
              ("c", "calibrate probe"),
              ("z", "fit board / full frame"),
              ("q", "quit")]
CAL_KEY_LEGEND = [("SPACE", "capture 2 s"),
                  ("y / n", "save / discard"),
                  ("Esc", "abort calibration"),
                  ("q", "quit")]


def draw_axes(img, T_cam, K, dist, length=AXIS_MM):
    cv2.drawFrameAxes(img, K, dist, cv2.Rodrigues(T_cam[:3, :3])[0], T_cam[:3, 3], length, 2)


def draw_outline(img, T_cam_phantom, K, dist, z, colour):
    px = tm.region_outline_px(T_cam_phantom, K, dist, z=z)
    cv2.polylines(img, [np.round(px).astype(np.int32)], True, colour, 2, cv2.LINE_AA)


def draw_markers(img, corners, ids):
    """Detected markers: thin outline and ID, in the clinical palette (probe marker green, board grey)."""
    if ids is None:
        return
    for c, i in zip(corners, np.asarray(ids).ravel()):
        pts = np.round(c.reshape(4, 2)).astype(np.int32)
        colour = ui.GREEN if i == tm.PROBE_MARKER_ID else ui.GREY
        cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[0]), 3, colour, -1, cv2.LINE_AA)               # marker's top-left corner
        label = f"ID {i}"
        x, y = pts[:, 0].min(), pts[:, 1].min() - 6
        cv2.putText(img, label, (int(x), int(y)), ui.FONT, 0.45, colour, 1, cv2.LINE_AA)


def annotate(frame, detections, s, tracker):
    """Camera frame with the tracking overlay (graphics only; the numbers are in the readout column)."""
    img = frame.copy()
    corners, ids = detections
    K, dist = tracker.K, tracker.dist
    draw_markers(img, corners, ids)
    if s.T_cam_phantom is not None:
        draw_outline(img, s.T_cam_phantom, K, dist, tm.H_MM, ui.AMBER)
        if tm.H_MM != 0:
            draw_outline(img, s.T_cam_phantom, K, dist, 0.0, ui.WHITE)
    if s.valid:
        T_cam_marker = s.T_cam_probe.copy()                  # ID 0 marker: raw probe board pose
        T_cam_marker[:3, 3] = (s.T_cam_probe @ np.array([0, 0, -tm.PLATFORM_HEIGHT_MM, 1.0]))[:3]
        draw_axes(img, T_cam_marker, K, dist)
        draw_axes(img, s.T_cam_phantom @ s.T_phantom_probe_raw, K, dist)   # face centre (calibrated)
    return img


class CameraZoom:
    """Manual zoom and pan of the camera scene, with a fit-to-board preset.

    zoom (1 = full frame .. MAX_ZOOM) and the view centre (cx, cy, as fractions of the frame) are set by the
    simulator's Zoom / Pan sliders. fit_board() sets them so the printed sheet (markers.SHEET_*) projected
    through the tracked phantom pose, at the sheet (z = H_MM) and at the probe-marker height (z = -PLATFORM_HEIGHT_MM,
    so a lifted probe stays in view), plus margin_mm, fills the view. The view fits the board automatically the
    first time the board is seen; after that it only changes when you move the sliders or press z.
    """
    MAX_ZOOM = 5.0

    def __init__(self, margin_mm=12.0):
        self.zoom, self.cx, self.cy = 1.0, 0.5, 0.5
        self.margin_mm = margin_mm
        self.board = None                   # last projected board rect (x0, y0, x1, y1) in pixels
        self.shape = None
        self.fitted = False

    def label(self):
        return "full frame" if self.zoom <= 1.001 else f"zoom {self.zoom:.1f}x"

    def _board_rect(self, s, K, dist):
        if s is None or s.T_cam_phantom is None:
            return self.board
        (sx, sy), (sw, sh), m = tm.SHEET_TL, tm.SHEET_SIZE, self.margin_mm
        xy = [(sx - m, sy - m), (sx + sw + m, sy - m), (sx + sw + m, sy + sh + m), (sx - m, sy + sh + m)]
        P = np.array([(x, y, z) for z in (tm.H_MM, -tm.PLATFORM_HEIGHT_MM) for x, y in xy], np.float64)
        T = s.T_cam_phantom
        px = cv2.projectPoints(P, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)[0].reshape(-1, 2)
        (x0, y0), (x1, y1) = px.min(0), px.max(0)
        self.board = (float(x0), float(y0), float(x1), float(y1))
        return self.board

    def fit_board(self):
        """Zoom and centre so the board fills the view (keeps the frame's aspect ratio)."""
        if self.board is None or self.shape is None:
            return False
        H, W = self.shape[:2]
        x0, y0, x1, y1 = self.board
        self.zoom = float(np.clip(min(W / max(x1 - x0, 1.0), H / max(y1 - y0, 1.0)), 1.0, self.MAX_ZOOM))
        self.cx, self.cy = float(np.clip((x0 + x1) / 2 / W, 0, 1)), float(np.clip((y0 + y1) / 2 / H, 0, 1))
        return True

    def full_frame(self):
        self.zoom, self.cx, self.cy = 1.0, 0.5, 0.5

    def toggle(self):
        """z key: fit to board <-> full frame."""
        if self.zoom > 1.001 or not self.fit_board():
            self.full_frame()

    def apply(self, img, s, K, dist):
        """Crop img (the annotated camera frame) to the current zoom / pan."""
        self.shape = img.shape
        self._board_rect(s, K, dist)
        if not self.fitted and self.board is not None:
            self.fitted = self.fit_board()
        H, W = img.shape[:2]
        w, h = W / self.zoom, H / self.zoom
        x0 = int(round(np.clip(self.cx * W - w / 2, 0, W - w)))
        y0 = int(round(np.clip(self.cy * H - h / 2, 0, H - h)))
        return img[y0:y0 + int(round(h)), x0:x0 + int(round(w))]


ZOOM = CameraZoom()                      # shared by the viewer and the simulator's camera view


def viewer_window(frame, detections, s, tracker, cal=None, logging=False, zoom=ZOOM):
    """The whole viewer window: header, camera scene, phantom map, readouts, calibration prompts."""
    banner = cal.prompt_lines() if cal is not None else []
    scene = zoom.apply(annotate(frame, detections, s, tracker), s, tracker.K, tracker.dist)
    return ui.compose_tracking(scene, s, title="PROBE TRACKING",
                               subtitle="tracking check", keys=CAL_KEY_LEGEND if cal is not None else KEY_LEGEND,
                               banner=banner, footprint_mm=FOOTPRINT_MM,
                               badges=[("REC  CSV", ui.RED)] if logging else [],
                               scene_caption=f"SCENE  |  D405 CAMERA  |  {zoom.label()}  (z)")


def print_legend():
    print("[INFO] keys (viewer window must have focus):")
    for k, d in KEY_LEGEND:
        print(f"         {k:<8} {d}")
    print("       during calibration:")
    for k, d in CAL_KEY_LEGEND:
        print(f"         {k:<8} {d}")


def save_capture(frame, annotated, s, tracker, root=CAPTURES_DIR):
    d = os.path.join(root, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, "frame.png"), frame)
    cv2.imwrite(os.path.join(d, "annotated.png"), annotated)
    with open(os.path.join(d, "state.json"), "w") as f:
        json.dump(s.to_dict(), f, indent=2)
    save_intrinsics(os.path.join(d, "intrinsics.json"), tracker.K, tracker.dist)
    print(f"[INFO] saved {d}")


class CsvLogger:
    def __init__(self, root=LOGS_DIR):
        os.makedirs(root, exist_ok=True)
        self.path = os.path.join(root, time.strftime("track_%Y%m%d_%H%M%S.csv"))
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(TrackState.CSV_FIELDS)
        print(f"[INFO] logging to {self.path}")

    def write(self, s):
        self.w.writerow(s.to_row())

    def close(self):
        self.f.close()
        print(f"[INFO] log closed: {self.path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="realsense", help='"realsense", a video file or an image folder')
    ap.add_argument("--exposure-us", type=float, default=None, help="manual exposure (us); default auto")
    ap.add_argument("--gain", type=float, default=None, help="manual gain (with --exposure-us)")
    ap.add_argument("--no-second-marker", action="store_true", help="track ID 0 only")
    ap.add_argument("--no-upright", action="store_true", help="free 6-DoF probe pose (no upright refit)")
    ap.add_argument("--loop", action="store_true", help="replay a video / folder in a loop")
    a = ap.parse_args()

    cfg = TrackerConfig(exposure_us=a.exposure_us, gain=a.gain, upright_probe=not a.no_upright, loop_files=a.loop)
    tracker = ProbeTracker(a.source, use_second_marker=False if a.no_second_marker else None, config=cfg)
    print(f"[INFO] tracking: H_MM={tm.H_MM}, PLATFORM_HEIGHT_MM={tm.PLATFORM_HEIGHT_MM}, "
          f"second marker {'on' if tracker.use_second_marker else 'off'}")
    print_legend()
    logger, cal, last_frame = None, None, -1
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            frame, det, s = tracker.get_frame()
            if frame is not None:
                if cal is not None:
                    cal.update()
                    if cal.finished and cal.step == "done" and time.monotonic() - cal_done_t > 4.0:
                        cal = None
                img = viewer_window(frame, det, s, tracker, cal, logging=logger is not None)
                cv2.imshow(WINDOW, img)
                if logger is not None and s.frame_index != last_frame:
                    logger.write(s)
                last_frame = s.frame_index
            elif s.message.startswith("capture error"):
                print("[ERROR]", s.message)
                break
            if tracker.finished and not a.loop:
                print("[INFO] end of file source")
                cv2.waitKey(0)
                break

            key = cv2.waitKey(10) & 0xFF
            if cal is not None and key != 255 and key not in (ord("q"),):
                cal.handle_key(key)
                if cal.finished:
                    cal_done_t = time.monotonic()
                continue
            if key in (ord("q"), 27):
                break
            if key == ord("s") and frame is not None:
                save_capture(frame, img, s, tracker)
            elif key == ord("r"):
                if logger is None:
                    logger = CsvLogger()
                else:
                    logger.close()
                    logger = None
            elif key == ord("c"):
                cal, cal_done_t = CalibrationRoutine(tracker), math.inf
            elif key == ord("z"):
                ZOOM.toggle()
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 and frame is not None:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if logger is not None:
            logger.close()
        tracker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
