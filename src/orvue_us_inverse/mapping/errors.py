"""
orvue_us_inverse.mapping.errors - pose-error injection and the tracking filter's lag (S7).

    model = PoseErrorModel(jitter_mm=0.5, jitter_yaw_deg=0.5, latency_ms=50)      # default: no error
    T_meas = model.apply(T_true, t, pose_at=plan.pose_at)        # one pose (pose_at(t) -> 4x4 for the latency)
    frames = model.apply_frames(frames, pose_at=...)            # FrameRecords with T_measured replaced
    model = PoseErrorModel.parse("jitter=0.5,yaw=0.5,latency=50")
    lag = filter_lag(speed_mm_s=10.0)                           # ms the tracker's one-euro filter trails behind

Error components (applied in this order to the true pose at time t):
    latency    the true pose from latency_ms earlier (pose_at(t - latency); a live LatencyBuffer of recent poses
               when there is no planned path)
    scale      positions scaled by (1 + scale_pct / 100) about SCALE_CENTRE (centre of the region on the surface),
               e.g. a wrong marker size or camera intrinsics
    bias       constant offset bias_mm (x, y, z) of the face centre and bias_yaw_deg about the probe axis
    jitter     independent Gaussian noise per frame: position sigma jitter_mm on x, y and z, yaw sigma jitter_yaw_deg,
               tilt sigma jitter_tilt_deg about the phantom x and y axes (seeded: reproducible)
With every component at 0, apply() returns the true pose unchanged (bit for bit).

Why errors must be injected: when the simulator renders from the tracked pose, image and pose are always
consistent - every frame shows the anatomy exactly at the pose it is reconstructed with - so the real tracking error of
the dummy probe is invisible in the reconstruction, however large it is. With a real probe the image comes from where
the probe really is and the tracker's error (noise, bias, latency, scale) moves each frame to the wrong place in the
volume. PoseErrorModel reproduces that: the frame is rendered at T_true and reconstructed at T_measured.

Tracking filter lag: ProbeTracker.get_pose() is one-euro filtered (TrackerConfig pos_min_cutoff, pos_beta), so on top of
the camera latency (exposure, transfer, detection) the filtered pose trails the probe. The filter's cutoff rises with
the speed it sees (min_cutoff + beta x |dx|, where dx is taken against the previous filtered value and so includes the
lag itself), so the lag shrinks with speed. Measured with the tracker's own filter at 30 frames/s (filter_lag, settled
after 4 s): 111 ms at 2 mm/s (0.22 mm behind), 85 ms at 5 mm/s (0.42 mm), 65 ms at 10 mm/s (0.65 mm), 47 ms at
20 mm/s (0.94 mm). tests/mapping/test_errors.py checks it end to end on synthetic camera frames. The camera latency
itself (D405 exposure + USB + detection) adds to this and can only be measured with the hardware.
"""
import math
import re
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from orvue_us_inverse.mapping.sweep_io import FrameRecord

SCALE_CENTRE = np.array([50.0, 50.0, 0.0])
PARSE_KEYS = {"jitter": "jitter_mm", "yaw": "jitter_yaw_deg", "tilt": "jitter_tilt_deg", "latency": "latency_ms",
              "scale": "scale_pct", "bias": "bias_x_mm", "biasx": "bias_x_mm", "biasy": "bias_y_mm",
              "biasz": "bias_z_mm", "biasyaw": "bias_yaw_deg", "seed": "seed"}


def _rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


def _ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def interpolate_pose(Ta: np.ndarray, Tb: np.ndarray, f: float) -> np.ndarray:
    """Pose between Ta (f = 0) and Tb (f = 1): position linear, rotation about the phantom z axis (yaw) linear;
    tilt taken from Ta (sweep poses are upright)."""
    ya, yb = math.atan2(Ta[1, 0], Ta[0, 0]), math.atan2(Tb[1, 0], Tb[0, 0])
    dyaw = (math.degrees(yb - ya) + 180.0) % 360.0 - 180.0
    T = np.array(Ta, np.float64)
    T[:3, :3] = _rz(f * dyaw) @ T[:3, :3]
    T[:3, 3] = (1 - f) * np.asarray(Ta)[:3, 3] + f * np.asarray(Tb)[:3, 3]
    return T


class LatencyBuffer:
    """Recent (t, pose) pairs of a live pose source; pose_at(t) interpolates (clamped to the oldest / newest)."""

    def __init__(self, keep_s: float = 2.0):
        self.keep_s = keep_s
        self._buf: deque = deque()

    def add(self, t: float, T: np.ndarray) -> None:
        self._buf.append((float(t), np.array(T, np.float64)))
        while len(self._buf) > 2 and self._buf[-1][0] - self._buf[1][0] > self.keep_s:
            self._buf.popleft()

    def clear(self) -> None:
        self._buf.clear()

    def pose_at(self, t: float) -> np.ndarray:
        if not self._buf:
            raise ValueError("LatencyBuffer is empty")
        b = self._buf
        if t <= b[0][0]:
            return b[0][1].copy()
        for (ta, Ta), (tb, Tb) in zip(b, list(b)[1:]):
            if ta <= t <= tb:
                return interpolate_pose(Ta, Tb, 0.0 if tb <= ta else (t - ta) / (tb - ta))
        return b[-1][1].copy()


@dataclass
class PoseErrorModel:
    jitter_mm: float = 0.0
    jitter_yaw_deg: float = 0.0
    jitter_tilt_deg: float = 0.0
    bias_x_mm: float = 0.0
    bias_y_mm: float = 0.0
    bias_z_mm: float = 0.0
    bias_yaw_deg: float = 0.0
    latency_ms: float = 0.0
    scale_pct: float = 0.0
    seed: int = 0
    _rng: np.random.Generator = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    @property
    def is_zero(self) -> bool:
        return all(getattr(self, k) == 0 for k in ("jitter_mm", "jitter_yaw_deg", "jitter_tilt_deg", "bias_x_mm",
                                                   "bias_y_mm", "bias_z_mm", "bias_yaw_deg", "latency_ms",
                                                   "scale_pct"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_rng", None)
        return d

    def describe(self) -> str:
        parts = [f"{k}={v:g}" for k, v in self.to_dict().items() if k != "seed" and v]
        return ", ".join(parts) if parts else "no error"

    @classmethod
    def parse(cls, text: str) -> "PoseErrorModel":
        """'jitter=0.5,yaw=0.5,latency=50' (keys: jitter mm, yaw deg, tilt deg, latency ms, scale %, bias / biasx /
        biasy / biasz mm, biasyaw deg, seed)."""
        kw = {}
        for part in filter(None, re.split(r"[,\s]+", text or "")):
            k, _, v = part.partition("=")
            k = k.strip().lower()
            if k not in PARSE_KEYS or not v:
                raise ValueError(f"unknown or empty error term {part!r}; use {', '.join(PARSE_KEYS)}")
            kw[PARSE_KEYS[k]] = int(v) if PARSE_KEYS[k] == "seed" else float(v)
        return cls(**kw)

    def reset(self) -> None:
        """Restart the jitter sequence (same seed)."""
        self._rng = np.random.default_rng(self.seed)

    def apply(self, T_true: np.ndarray, t: float = 0.0, pose_at: Callable[[float], np.ndarray] | None = None
              ) -> np.ndarray:
        """T_measured for the true pose T_true at time t (s). pose_at(t) gives the true pose at any time (needed for
        latency; without it the latency is ignored)."""
        T_true = np.asarray(T_true, np.float64)
        if self.is_zero:
            return T_true.copy()
        T = np.array(pose_at(t - self.latency_ms / 1000.0), np.float64) if (self.latency_ms and pose_at) \
            else T_true.copy()
        p, R = T[:3, 3].copy(), T[:3, :3].copy()
        if self.scale_pct:
            p = SCALE_CENTRE + (1.0 + self.scale_pct / 100.0) * (p - SCALE_CENTRE)
        p = p + np.array([self.bias_x_mm, self.bias_y_mm, self.bias_z_mm])
        if self.bias_yaw_deg:
            R = _rz(self.bias_yaw_deg) @ R
        if self.jitter_mm:
            p = p + self._rng.normal(0.0, self.jitter_mm, 3)
        if self.jitter_yaw_deg:
            R = _rz(self._rng.normal(0.0, self.jitter_yaw_deg)) @ R
        if self.jitter_tilt_deg:
            R = _rx(self._rng.normal(0.0, self.jitter_tilt_deg)) @ _ry(self._rng.normal(0.0, self.jitter_tilt_deg)) @ R
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = R, p
        return out

    def apply_frames(self, frames: list[FrameRecord], pose_at: Callable[[float], np.ndarray] | None = None
                     ) -> list[FrameRecord]:
        """Copies of the frames with T_measured = apply(T_true, t). Without pose_at the latency interpolates between
        the frames themselves (clamped at the first one)."""
        if self.is_zero:
            return list(frames)
        if self.latency_ms and pose_at is None:
            ts = np.array([f.t for f in frames])

            def pose_at(t):
                j = int(np.searchsorted(ts, t))
                if j <= 0:
                    return frames[0].T_true
                if j >= len(frames):
                    return frames[-1].T_true
                a, b = frames[j - 1], frames[j]
                return interpolate_pose(a.T_true, b.T_true, (t - a.t) / max(b.t - a.t, 1e-12))
        out = []
        for f in frames:
            out.append(FrameRecord(f.index, f.t, f.T_true, self.apply(f.T_true, f.t, pose_at), f.labels, f.image))
        return out


def filter_lag(speed_mm_s: float = 10.0, fps: float = 30.0, duration_s: float = 4.0, config=None) -> dict:
    """Steady-state lag of the tracker's one-euro position filter for a probe moving at a constant speed along y,
    sampled at fps: {'lag_ms', 'lag_mm', 'cutoff_hz'} (the tracker's own OneEuroFilter, TrackerConfig defaults)."""
    from orvue_us_inverse.tracking.tracker import OneEuroFilter, TrackerConfig

    c = config or TrackerConfig()
    f = OneEuroFilter(c.pos_min_cutoff, c.pos_beta, c.d_cutoff)
    n = int(duration_s * fps)
    err = 0.0
    for k in range(n):
        t = k / fps
        true = np.array([50.0, 20.0 + speed_mm_s * t, 0.0])
        err = float(true[1] - f(true, t)[1])
    return dict(lag_mm=err, lag_ms=1000.0 * err / speed_mm_s if speed_mm_s else 0.0,
                cutoff_hz=c.pos_min_cutoff + c.pos_beta * speed_mm_s)
