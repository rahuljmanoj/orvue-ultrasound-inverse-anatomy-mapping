"""
orvue_us_inverse.tracking.calibrate - probe calibration offsets for the tracker (stored in config/calibration.json).

    python -m orvue_us_inverse calibrate [--source realsense|video|folder] [--points 25,25 75,25 50,75]

Also run from the track viewer with the c key. Two steps, driven by on-screen prompts:

1. Yaw offset. Place the probe face on a known line on the region (default: along y = 50 with the
   array pointing +x, i.e. yaw 0, so the orientation mark at the probe's -x end faces x = 0). SPACE averages 2 s of yaw readings (circular mean) and stores
   yaw_offset_deg so the reading becomes the expected yaw. Saved immediately.
2. Face position check. Place the face centre on each known point in turn (default (25,25), (75,25),
   (50,75)) and press SPACE (2 s average each). The mean and max x/y error are shown together with
   the suggested probe-frame translation correction (dx, dy, dz; z target = 0, the gel surface).
   y saves the correction, n discards it.

Esc aborts at any time (a yaw offset already saved stays saved).
"""
import argparse
import math

import cv2
import numpy as np

from orvue_us_inverse.tracking.tracker import Calibration, ProbeTracker, TrackerConfig

DEFAULT_POINTS = [(25.0, 25.0), (75.0, 25.0), (50.0, 75.0)]
SPACE, ESC = 32, 27


def circular_mean_deg(angles):
    a = np.radians(np.asarray(angles, dtype=np.float64))
    return math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))


def suggest_translation(samples, targets, calibration):
    """Probe-frame correction from face measurements at known points.

    samples: list of (face_xyz_mean, R_phantom_probe) per point (current calibration applied);
    targets: list of (x, y) with z target 0. A change delta of (dx, dy, dz) moves the face by R @ delta,
    so each point asks for delta_i = -R_i.T @ error_i; the suggestion is their mean added to the current
    correction. Returns (errors Nx3, new (dx, dy, dz), predicted errors after the correction Nx3)."""
    errors, deltas = [], []
    for (face, R), (tx, ty) in zip(samples, targets):
        e = np.asarray(face) - np.array([tx, ty, 0.0])
        errors.append(e)
        deltas.append(-R.T @ e)
    errors = np.array(errors)
    delta = np.mean(deltas, axis=0)
    after = np.array([e + R @ delta for e, (_, R) in zip(errors, samples)])
    new = np.array([calibration.dx, calibration.dy, calibration.dz]) + delta
    return errors, new, after


class CalibrationRoutine:
    """Step machine: call update() every frame, handle_key() on key presses, draw prompt_lines()."""

    def __init__(self, tracker, points=None, yaw_line_y=50.0, expected_yaw=0.0, seconds=2.0):
        self.tr = tracker
        self.points = list(points or DEFAULT_POINTS)
        self.yaw_line_y, self.expected_yaw, self.seconds = yaw_line_y, expected_yaw, seconds
        self.step = "yaw_wait"                 # yaw_wait, yaw_collect, point_wait, point_collect, confirm, done
        self.i = 0
        self.buf, self.samples = [], []
        self._t0, self._last_frame = None, -1
        self.result_lines = []
        self.finished = False

    # ---- input
    def handle_key(self, key):
        if key == ESC:
            self._finish(["Calibration aborted."])
        elif key == SPACE and self.step in ("yaw_wait", "point_wait"):
            self.buf, self._t0 = [], None
            self.step = self.step.replace("wait", "collect")
        elif self.step == "confirm" and key in (ord("y"), ord("n")):
            if key == ord("y"):
                c = self.tr.calibration
                c.dx, c.dy, c.dz = (float(v) for v in self.new_xyz)
                c.save(self.tr.cfg.calibration_path)
                self.tr.reload_calibration()
                self._finish(self.result_lines[:-1] + [f"Saved to {self.tr.cfg.calibration_path}."])
            else:
                self._finish(self.result_lines[:-1] + ["Translation correction discarded."])

    def _finish(self, lines):
        self.result_lines = lines
        self.step, self.finished = "done", True
        print("[CAL]", lines[-1])                     # earlier lines were printed when computed

    # ---- per frame
    def update(self):
        if self.step not in ("yaw_collect", "point_collect"):
            return
        s = self.tr.get_state()
        if s.frame_index == self._last_frame:
            return
        self._last_frame = s.frame_index
        if self._t0 is None:
            self._t0 = s.timestamp
        if s.valid:
            if self.step == "yaw_collect":
                self.buf.append(s.yaw_uncalibrated)
            else:
                self.buf.append((np.array(s.face_raw), s.T_phantom_probe_raw[:3, :3].copy()))
        if s.timestamp - self._t0 < self.seconds:
            return
        if len(self.buf) < 5:
            self.step = self.step.replace("collect", "wait")        # too few valid frames: retry
            self.result_lines = [f"Only {len(self.buf)} valid frames - check the markers and press SPACE again."]
            return
        self.result_lines = []
        if self.step == "yaw_collect":
            self._finish_yaw()
        else:
            self._finish_point()

    def _finish_yaw(self):
        mean = circular_mean_deg(self.buf)
        c = self.tr.calibration
        c.yaw_offset_deg = (self.expected_yaw - mean + 180.0) % 360.0 - 180.0
        c.save(self.tr.cfg.calibration_path)
        self.tr.reload_calibration()
        spread = max(abs((a - mean + 180) % 360 - 180) for a in self.buf)
        msg = (f"Yaw offset {c.yaw_offset_deg:+.2f} deg saved (raw mean {mean:+.2f} deg, "
               f"spread +-{spread:.2f} deg, {len(self.buf)} frames).")
        print("[CAL]", msg)
        self.yaw_msg = msg
        self.step, self.i, self.samples = "point_wait", 0, []

    def _finish_point(self):
        faces = np.array([f for f, _ in self.buf])
        R = self.buf[len(self.buf) // 2][1]
        self.samples.append((faces.mean(0), R))
        self.i += 1
        if self.i < len(self.points):
            self.step = "point_wait"
            return
        errors, self.new_xyz, after = suggest_translation(self.samples, self.points, self.tr.calibration)
        exy = np.hypot(errors[:, 0], errors[:, 1])
        axy = np.hypot(after[:, 0], after[:, 1])
        lines = [self.yaw_msg]
        for (tx, ty), e in zip(self.points, errors):
            lines.append(f"  point ({tx:g},{ty:g}): error x {e[0]:+.2f}  y {e[1]:+.2f}  z {e[2]:+.2f} mm")
        lines += [f"x/y error: mean {exy.mean():.2f} mm, max {exy.max():.2f} mm",
                  f"Suggested correction dx {self.new_xyz[0]:+.2f}  dy {self.new_xyz[1]:+.2f}  "
                  f"dz {self.new_xyz[2]:+.2f} mm (probe frame)",
                  f"x/y error after it (predicted): mean {axy.mean():.2f} mm, max {axy.max():.2f} mm",
                  "Save this correction?  y = save,  n = discard"]
        for s in lines[:-1]:
            print("[CAL]", s)
        self.result_lines = lines
        self.step = "confirm"

    # ---- display
    def prompt_lines(self):
        if self.step == "yaw_wait":
            lines = [f"CALIBRATION 1/2 - yaw: place the face along y = {self.yaw_line_y:g} mm, array pointing +x "
                     f"(yaw {self.expected_yaw:g} deg):",
                     f"orientation mark towards x = 0 (left edge). SPACE = capture {self.seconds:g} s, Esc = abort"]
        elif self.step == "point_wait":
            x, y = self.points[self.i]
            lines = [f"CALIBRATION 2/2 - point {self.i + 1}/{len(self.points)}: place the face centre at "
                     f"({x:g}, {y:g}) mm.", f"SPACE = capture {self.seconds:g} s, Esc = abort"]
        elif self.step in ("yaw_collect", "point_collect"):
            lines = [f"Capturing... hold still ({len(self.buf)} frames)"]
        else:
            lines = []
        return lines + self.result_lines


def parse_points(values):
    return [tuple(float(v) for v in p.split(",")) for p in values]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="realsense")
    ap.add_argument("--points", nargs="+", default=None, help="face check points as x,y (mm)")
    ap.add_argument("--yaw-line-y", type=float, default=50.0)
    ap.add_argument("--expected-yaw", type=float, default=0.0)
    ap.add_argument("--exposure-us", type=float, default=None)
    ap.add_argument("--no-second-marker", action="store_true")
    a = ap.parse_args()

    tracker = ProbeTracker(a.source, use_second_marker=False if a.no_second_marker else None,
                           config=TrackerConfig(exposure_us=a.exposure_us))
    cal = CalibrationRoutine(tracker, parse_points(a.points) if a.points else None, a.yaw_line_y, a.expected_yaw)
    from orvue_us_inverse.ui import clinical as ui
    from orvue_us_inverse.tracking.viewer import CAL_KEY_LEGEND, FOOTPRINT_MM, annotate
    win = "Probe calibration"
    try:
        while not cal.finished:
            frame, det, s = tracker.get_frame()
            if frame is not None:
                img = ui.compose_tracking(annotate(frame, det, s, tracker), s, title="PROBE CALIBRATION",
                                          keys=CAL_KEY_LEGEND, banner=cal.prompt_lines(), footprint_mm=FOOTPRINT_MM)
                cv2.imshow(win, img)
            cal.update()
            key = cv2.waitKey(15) & 0xFF
            if key != 255:
                cal.handle_key(key)
        cv2.waitKey(1500)
    finally:
        tracker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
