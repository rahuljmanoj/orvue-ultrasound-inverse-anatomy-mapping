"""
orvue_us_inverse.tracking.tracker - live tracking of the ultrasound probe over the printed reference board.

The Intel RealSense D405 colour stream (or a video file / image folder) is read in a background
thread. Every frame the ArUco markers are detected once, then

    T_cam_phantom   reference board (IDs 1-4, any 1-4 visible)    markers.board_pose()
    T_cam_probe     probe board (ID 0, optionally ID 5)          markers.board_pose()
    T_phantom_probe = inv(T_cam_phantom) @ T_cam_probe @ T_calibration

Frames, marker layout and the measured heights all come from orvue_us_inverse.tracking.markers.

Reported per frame (TrackState):
    marker_*  centre of the ID 0 marker in the phantom frame (x, y, z)
    face_*    centre of the probe contact face in the phantom frame (x, y, z)
    yaw_*     atan2(u_y, u_x) of the probe +x axis u in the phantom frame (deg), the convention of
              BModeSimulator.pose_from_xy_yaw(). Seen from the camera (looking down +z, image x = +x,
              image y = +y), yaw grows CLOCKWISE: rotating the probe counter-clockwise as seen from
              the camera DECREASES yaw (YAW_SIGN_CCW_FROM_CAMERA = -1).
    tilt_deg  angle between probe +z and phantom +z
"raw" values are unfiltered; "filt" values come from one-euro filters (yaw via cos/sin). Both
include the calibration offsets from calibration.json.

Usage:
    tracker = ProbeTracker("realsense")         # or a video file / image folder
    T = tracker.get_pose()                      # 4x4 T_phantom_probe or None
    xy_yaw = tracker.get_xy_yaw()               # (x, y, yaw_deg) or None
    tracker.stop()
"""
import glob
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from orvue_us_inverse.paths import CALIBRATION_PATH
from orvue_us_inverse.tracking import markers as tm

YAW_SIGN_CCW_FROM_CAMERA = -1
CALIBRATION_FILE = CALIBRATION_PATH      # config/calibration.json
INTRINSICS_FILE = "intrinsics.json"
REFERENCE_IDS = sorted(tm.REF_MARKERS_TL)


@dataclass
class TrackerConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    exposure_us: float = None          # None = auto exposure; e.g. 3000 for a short manual exposure
    gain: float = None                 # manual gain (only used with exposure_us), None = leave as is
    max_reproj_px: float = 2.0         # reject a board pose above this RMS reprojection error
    phantom_hold_s: float = 2.0        # keep the last reference pose this long when the board is hidden
    pos_min_cutoff: float = 1.0        # one-euro filter, positions (Hz)
    pos_beta: float = 0.05             # one-euro speed coefficient, positions (1/mm)
    yaw_min_cutoff: float = 1.0        # one-euro filter on cos/sin(yaw) (Hz)
    yaw_beta: float = 0.5
    d_cutoff: float = 1.0              # cutoff of the derivative estimate (Hz)
    # Upright refinement: the probe is assumed perpendicular to the surface (as pose_from_xy_yaw does)
    # and x, y, z, yaw are re-fitted to the probe corners with tilt fixed at 0. The free 6-DoF fit of a
    # small marker seen face-on has 1-5 deg tilt noise, which the platform height turns into mm at the
    # face; the upright fit removes that. tilt_deg is still reported from the 6-DoF fit.
    upright_probe: bool = True
    calibration_path: str = CALIBRATION_FILE
    loop_files: bool = False           # replay a video file / image folder in a loop


# ---------------------------------------------------------------- filters
class OneEuroFilter:
    """One-euro filter (Casiez et al. 2012) for a scalar or a vector, with explicit timestamps."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self._x = self._dx = self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self._x is None:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return x.copy()
        dt = max(t - self._t, 1e-6)
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x.copy()


class AngleFilter:
    """One-euro filtering of an angle (deg) via cos and sin, recombined with atan2 (no wrap jumps)."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self._f = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self):
        self._f.reset()

    def __call__(self, angle_deg, t):
        a = math.radians(angle_deg)
        c, s = self._f(np.array([math.cos(a), math.sin(a)]), t)
        return math.degrees(math.atan2(s, c))


# ---------------------------------------------------------------- calibration offsets
@dataclass
class Calibration:
    yaw_offset_deg: float = 0.0
    dx: float = 0.0                    # probe-frame translation correction of the face centre (mm)
    dy: float = 0.0
    dz: float = 0.0

    @classmethod
    def load(cls, path=CALIBRATION_FILE):
        if path and os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            return cls(**{k: float(d.get(k, 0.0)) for k in ("yaw_offset_deg", "dx", "dy", "dz")})
        return cls()

    def save(self, path=CALIBRATION_FILE):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def transform(self):
        """T_rawprobe_probe: rotate about the probe z axis by the yaw offset, then translate by (dx, dy, dz)."""
        a = math.radians(self.yaw_offset_deg)
        T = np.eye(4)
        T[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
        T[:3, 3] = T[:3, :3] @ np.array([self.dx, self.dy, self.dz])
        return T


# ---------------------------------------------------------------- state
@dataclass
class TrackState:
    timestamp: float = 0.0
    frame_index: int = -1
    valid: bool = False                # phantom and probe both valid
    phantom_valid: bool = False
    phantom_held: bool = False         # reference board not seen; last pose reused (< phantom_hold_s)
    probe_valid: bool = False
    ids_seen: list = field(default_factory=list)
    n_reference_markers_used: int = 0
    n_probe_markers_used: int = 0
    reproj_error_ref_px: float = None
    reproj_error_probe_px: float = None
    reproj_error_upright_px: float = None   # of the upright refit (None when upright_probe is off)
    marker_raw: list = None           # ID 0 centre in the phantom frame (mm)
    marker_filt: list = None
    face_raw: list = None              # probe face centre in the phantom frame (mm)
    face_filt: list = None
    yaw_raw: float = None              # deg, calibration applied
    yaw_filt: float = None
    yaw_uncalibrated: float = None     # deg, before the calibration yaw offset (used by calibration)
    tilt_deg: float = None
    T_phantom_probe: np.ndarray = None       # filtered, calibration applied
    T_phantom_probe_raw: np.ndarray = None   # unfiltered, calibration applied
    T_cam_phantom: np.ndarray = None
    T_cam_probe: np.ndarray = None           # raw probe board pose (no calibration)
    fps: float = 0.0
    message: str = ""

    def to_dict(self):
        """JSON-friendly dict (arrays as nested lists)."""
        d = {}
        for k, v in asdict(self).items():
            d[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return d

    CSV_FIELDS = ["timestamp", "frame_index", "valid", "phantom_valid", "phantom_held", "probe_valid", "ids_seen",
                  "n_reference_markers_used", "n_probe_markers_used", "reproj_error_ref_px", "reproj_error_probe_px", "reproj_error_upright_px",
                  "marker_x", "marker_y", "marker_z", "marker_x_filt", "marker_y_filt", "marker_z_filt",
                  "face_x", "face_y", "face_z", "face_x_filt", "face_y_filt", "face_z_filt",
                  "yaw_raw", "yaw_filt", "yaw_uncalibrated", "tilt_deg", "fps"] + \
                 [f"T{r}{c}" for r in range(4) for c in range(4)]

    def to_row(self):
        """Flat row matching CSV_FIELDS."""
        def xyz(v):
            return list(v) if v is not None else [None] * 3
        T = self.T_phantom_probe.ravel().tolist() if self.T_phantom_probe is not None else [None] * 16
        return [self.timestamp, self.frame_index, self.valid, self.phantom_valid, self.phantom_held, self.probe_valid,
                " ".join(map(str, self.ids_seen)), self.n_reference_markers_used, self.n_probe_markers_used,
                self.reproj_error_ref_px, self.reproj_error_probe_px, self.reproj_error_upright_px,
                *xyz(self.marker_raw), *xyz(self.marker_filt), *xyz(self.face_raw), *xyz(self.face_filt),
                self.yaw_raw, self.yaw_filt, self.yaw_uncalibrated, self.tilt_deg, self.fps] + T


# ---------------------------------------------------------------- frame sources
def load_intrinsics(path):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["K"], dtype=np.float64), np.array(d["dist"], dtype=np.float64)


def save_intrinsics(path, K, dist):
    with open(path, "w") as f:
        json.dump({"K": np.asarray(K).tolist(), "dist": np.asarray(dist).ravel().tolist()}, f, indent=2)


class RealSenseSource:
    """D405 colour stream; K and distortion from the stream's factory intrinsics."""

    def __init__(self, cfg):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        rc = rs.config()
        rc.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
        profile = self.pipeline.start(rc)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
        self.dist = np.array(intr.coeffs, dtype=np.float64)
        self._set_exposure(profile.get_device(), cfg)

    def _set_exposure(self, device, cfg):
        rs = self.rs
        for sensor in device.query_sensors():
            if not any(p.stream_type() == rs.stream.color for p in sensor.get_stream_profiles()):
                continue
            try:
                if cfg.exposure_us is None:
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
                else:
                    sensor.set_option(rs.option.enable_auto_exposure, 0)
                    sensor.set_option(rs.option.exposure, float(cfg.exposure_us))
                    if cfg.gain is not None:
                        sensor.set_option(rs.option.gain, float(cfg.gain))
            except RuntimeError as e:
                print(f"[WARN] could not set exposure on {sensor.get_info(rs.camera_info.name)}: {e}")

    def read(self):
        frames = self.pipeline.wait_for_frames(5000)
        c = frames.get_color_frame()
        if not c:
            return None, None
        return np.asanyarray(c.get_data()).copy(), time.monotonic()

    def close(self):
        self.pipeline.stop()


class FileSource:
    """Video file or image folder, replayed at its frame rate. Intrinsics come from intrinsics.json in
    the folder / next to the video (written by the track viewer's 's' key) unless given explicitly."""

    def __init__(self, path, cfg, intrinsics=None):
        self.cfg, self.path = cfg, path
        if os.path.isdir(path):
            self.files = sorted(f for ext in ("png", "jpg", "jpeg", "bmp")
                                for f in glob.glob(os.path.join(path, f"*.{ext}"))
                                if not os.path.basename(f).startswith("annotated"))
            if not self.files:
                raise FileNotFoundError(f"no images in {path}")
            self.cap, self.fps = None, cfg.fps
            ipath = os.path.join(path, INTRINSICS_FILE)
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise FileNotFoundError(path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or cfg.fps
            ipath = os.path.join(os.path.dirname(os.path.abspath(path)), INTRINSICS_FILE)
        if intrinsics is None:
            if not os.path.exists(ipath):
                raise FileNotFoundError(f"camera intrinsics needed: {ipath} (or pass intrinsics=(K, dist))")
            intrinsics = load_intrinsics(ipath)
        self.K, self.dist = np.asarray(intrinsics[0], np.float64), np.asarray(intrinsics[1], np.float64)
        self.i = 0
        self._next = time.monotonic()

    def read(self):
        time.sleep(max(0.0, self._next - time.monotonic()))
        self._next = max(self._next + 1.0 / self.fps, time.monotonic())
        if self.cap is None:
            if self.i >= len(self.files):
                if not self.cfg.loop_files:
                    return None, None
                self.i = 0
            img = cv2.imread(self.files[self.i])
        else:
            ok, img = self.cap.read()
            if not ok and self.cfg.loop_files:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self.cap.read()
            if not ok:
                return None, None
        t = self.i / self.fps                 # file time, so filters behave as when recorded
        self.i += 1
        return img, t

    def close(self):
        if self.cap is not None:
            self.cap.release()


# ---------------------------------------------------------------- tracker
class ProbeTracker:
    """Probe pose in the phantom frame from the reference board and the probe marker(s).

    source: "realsense", a video file path, an image folder, or None (no capture thread; feed frames
    with process_frame(), as the tests do). intrinsics=(K, dist) is required for source=None and
    optional for files.
    """

    def __init__(self, source="realsense", use_second_marker=None, config=None, intrinsics=None, autostart=True):
        self.cfg = config or TrackerConfig()
        self.use_second_marker = tm.USE_SECOND_MARKER if use_second_marker is None else bool(use_second_marker)
        self.det = tm.detector()
        self.ref_board = tm.reference_board()
        self.probe_board = tm.probe_board(self.use_second_marker)
        self.probe_ids = self.probe_board.getIds().ravel().tolist()
        self.calibration = Calibration.load(self.cfg.calibration_path)

        c = self.cfg
        self._f_marker = OneEuroFilter(c.pos_min_cutoff, c.pos_beta, c.d_cutoff)
        self._f_face = OneEuroFilter(c.pos_min_cutoff, c.pos_beta, c.d_cutoff)
        self._f_yaw = AngleFilter(c.yaw_min_cutoff, c.yaw_beta, c.d_cutoff)

        self._T_cam_phantom = None             # last accepted reference pose and its time
        self._t_cam_phantom = None
        self._frame_index = 0
        self._fps, self._t_last = 0.0, None

        self._lock = threading.Lock()
        self._state = TrackState(message="starting")
        self._frame = None
        self._detections = (None, None)
        self._running = False
        self._thread = None
        self.finished = False                  # file source reached its end

        if source is None:
            if intrinsics is None:
                raise ValueError("source=None needs intrinsics=(K, dist)")
            self.source = None
            self.K, self.dist = np.asarray(intrinsics[0], np.float64), np.asarray(intrinsics[1], np.float64)
        else:
            self.source = (RealSenseSource(self.cfg) if source == "realsense"
                           else FileSource(source, self.cfg, intrinsics))
            self.K, self.dist = self.source.K, self.source.dist
            if autostart:
                self.start()

    # ---- thread
    def start(self):
        if self.source is None or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="ProbeTracker", daemon=True)
        self._thread.start()

    def _loop(self):
        try:
            while self._running:
                img, t = self.source.read()
                if img is None:
                    if isinstance(self.source, FileSource):
                        self.finished = True
                        break
                    continue
                self.process_frame(img, t)
        except Exception as e:                 # keep the error visible to the viewer
            with self._lock:
                self._state = TrackState(message=f"capture error: {e}")
            raise
        finally:
            self._running = False

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.source is not None:
            self.source.close()
            self.source = None

    def reload_calibration(self):
        self.calibration = Calibration.load(self.cfg.calibration_path)

    def reset_filters(self):
        for f in (self._f_marker, self._f_face, self._f_yaw):
            f.reset()

    # ---- results
    def get_state(self):
        """Latest TrackState (never blocks on the camera)."""
        with self._lock:
            return self._state

    def get_frame(self):
        """(latest BGR frame, (corners, ids), state) for display."""
        with self._lock:
            return self._frame, self._detections, self._state

    def get_pose(self):
        """Filtered 4x4 T_phantom_probe (calibration applied), or None if not valid."""
        s = self.get_state()
        return None if not s.valid else s.T_phantom_probe.copy()

    def get_xy_yaw(self):
        """(x, y, yaw_deg) of the probe face for pose_from_xy_yaw(), or None."""
        T = self.get_pose()
        return None if T is None else tm.xy_yaw(T)

    # ---- per frame
    def _board(self, board, corners, ids):
        T = tm.board_pose_refined(board, corners, ids, self.K, self.dist)
        err = tm.board_reprojection_error(board, corners, ids, T, self.K, self.dist)
        return T, err

    def _upright_fit(self, T_cam_phantom, T_phantom_probe, corners, ids):
        """Refine (x, y, z, yaw) of a probe held perpendicular to the surface (probe +z = phantom +z),
        minimising the reprojection error of the probe board's corners. Returns (T_phantom_probe, rms px)."""
        from scipy.optimize import least_squares
        obj, img = self.probe_board.matchImagePoints(corners, ids)
        obj = np.hstack([obj.reshape(-1, 3).astype(np.float64), np.ones((len(obj), 1))])
        img = img.reshape(-1, 2).astype(np.float64)
        P = self.K @ T_cam_phantom[:3]                      # 3x4, phantom -> image (no distortion)
        img_u = cv2.undistortPoints(img.reshape(-1, 1, 2), self.K, self.dist, P=self.K).reshape(-1, 2)

        def pose(q):
            c, s = math.cos(q[3]), math.sin(q[3])
            T = np.eye(4)
            T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            T[:3, 3] = q[:3]
            return T

        def residual(q):
            h = (P @ pose(q) @ obj.T).T
            return (h[:, :2] / h[:, 2:3] - img_u).ravel()

        u = T_phantom_probe[:3, 0]
        q0 = np.r_[T_phantom_probe[:3, 3], math.atan2(u[1], u[0])]
        r = least_squares(residual, q0, method="lm", x_scale="jac", xtol=1e-10, ftol=1e-10)
        rms = float(np.sqrt((r.fun.reshape(-1, 2) ** 2).sum(1).mean()))
        return pose(r.x), rms

    def process_frame(self, image, timestamp=None):
        """Detect markers once, estimate both boards, update the state. Returns the new TrackState."""
        t = time.monotonic() if timestamp is None else float(timestamp)
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        ids_seen = sorted(np.asarray(ids).ravel().tolist()) if ids is not None else []
        s = TrackState(timestamp=t, frame_index=self._frame_index, ids_seen=ids_seen)
        self._frame_index += 1
        if self._t_last is not None and t > self._t_last:
            inst = 1.0 / (t - self._t_last)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._t_last = t
        s.fps = self._fps
        msgs = []

        # reference board -> T_cam_phantom (held for phantom_hold_s when hidden or rejected)
        ref_seen = tm.board_markers_seen(self.ref_board, ids)
        s.n_reference_markers_used = len(ref_seen)
        T_ref, s.reproj_error_ref_px = self._board(self.ref_board, corners, ids) if ref_seen else (None, None)
        if T_ref is not None and s.reproj_error_ref_px <= self.cfg.max_reproj_px:
            self._T_cam_phantom, self._t_cam_phantom = T_ref, t
            s.phantom_valid = True
        else:
            if T_ref is not None:
                msgs.append(f"reference rejected ({s.reproj_error_ref_px:.1f} px)")
            s.n_reference_markers_used = 0
            if self._T_cam_phantom is not None and t - self._t_cam_phantom <= self.cfg.phantom_hold_s:
                s.phantom_valid = s.phantom_held = True
            else:
                msgs.append("reference board not visible")
        s.T_cam_phantom = self._T_cam_phantom if s.phantom_valid else None

        # probe board -> T_cam_probe (ID 0 required, no extrapolation)
        probe_seen = tm.board_markers_seen(self.probe_board, ids)
        s.n_probe_markers_used = len(probe_seen)
        if tm.PROBE_MARKER_ID in probe_seen:
            T_prb, s.reproj_error_probe_px = self._board(self.probe_board, corners, ids)
            if T_prb is not None and s.reproj_error_probe_px <= self.cfg.max_reproj_px:
                s.T_cam_probe, s.probe_valid = T_prb, True
            elif T_prb is not None:
                msgs.append(f"probe rejected ({s.reproj_error_probe_px:.1f} px)")
        else:
            msgs.append(f"probe marker ID {tm.PROBE_MARKER_ID} not visible")

        s.valid = s.phantom_valid and s.probe_valid
        if s.valid:
            T_raw_probe = np.linalg.inv(s.T_cam_phantom) @ s.T_cam_probe      # 6-DoF, no calibration
            s.tilt_deg = float(np.degrees(np.arccos(np.clip(T_raw_probe[2, 2], -1.0, 1.0))))
            if self.cfg.upright_probe:
                T_raw_probe, s.reproj_error_upright_px = self._upright_fit(s.T_cam_phantom, T_raw_probe,
                                                                           corners, ids)
            T = T_raw_probe @ self.calibration.transform()
            marker = (T_raw_probe @ np.array([0.0, 0.0, -tm.PLATFORM_HEIGHT_MM, 1.0]))[:3]
            s.T_phantom_probe_raw = T
            s.marker_raw, s.face_raw = marker.tolist(), T[:3, 3].tolist()
            s.yaw_uncalibrated = tm.xy_yaw(T_raw_probe)[2]
            s.yaw_raw = tm.xy_yaw(T)[2]
            s.marker_filt = self._f_marker(marker, t).tolist()
            face_f = self._f_face(T[:3, 3], t)
            s.face_filt = face_f.tolist()
            s.yaw_filt = self._f_yaw(s.yaw_raw, t)
            # filtered pose: raw rotation turned about phantom z by the yaw correction, filtered position
            a = math.radians(s.yaw_filt - s.yaw_raw)
            Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            Tf = np.eye(4)
            Tf[:3, :3] = Rz @ T[:3, :3]
            Tf[:3, 3] = face_f
            s.T_phantom_probe = Tf
        else:
            self.reset_filters()               # do not glide from a stale pose when tracking resumes
        s.message = "; ".join(msgs) if msgs else "ok"

        with self._lock:
            self._state = s
            self._frame = image
            self._detections = (corners, ids)
        return s
