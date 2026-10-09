"""
orvue_us_inverse.mapping.poses - pose sources for a sweep (S1 scripted sweep, S6 mouse; S7 camera to follow).

ScriptedSweep(cfg) plans serpentine lanes over the region for every yaw in cfg.yaw_list_deg and gives the probe
pose as a function of simulated time at cfg.speed_mm_s.

Lane geometry for one yaw: lateral axis u = (cos yaw, sin yaw), sweep direction v = (-sin yaw, cos yaw) (the
columns of pose_from_xy_yaw). Lanes are lines along v at lateral positions s = p . u. With W the probe width,
stride = W * (1 - overlap) and n = ceil((E - W) / stride) + 1 lanes, where E is the region's extent along u
(100 mm at yaw 0 / 90); the n centres are then spread evenly so the outer lanes' image edges touch the region
edges exactly (actual overlap >= requested). Each lane runs the probe centre across the region along v, clipped to
where the face centre is on the region (= BModeSimulator.in_contact for these flat poses; at 0 / 90 the full
0..100 mm). Lane k runs along +v for even k and -v for odd k (serpentine), or always +v.

Between lanes and between orientations the probe is lifted: the transition moves at the same speed (and rotates
at ROTATION_DEG_S) but is not recorded.

    sweep = ScriptedSweep(SweepConfig(yaw_list_deg=[0, 90]))
    for s in sweep.samples():            # every 0.05 mm of travel
        acquirer.feed(s.T, s.t, s.recording)

MousePose (S6) is driven by mouse events on wall-clock time: position = probe face centre (mm), left button held =
recording, yaw turned in steps or set (any time, also during a stroke: only the user turns it). It keeps a short
position history for the speed meter.

    pose = MousePose()
    pose.move(x_mm, y_mm); pose.press(); ...; pose.release()
    s = pose.sample()                    # Sample(t, T, recording, lane=None)
"""
import bisect
import math
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from orvue_us_inverse.mapping.config import SweepConfig
from orvue_us_inverse.mapping.probe import PROBE
from orvue_us_inverse.simulation.bmode import BModeSimulator

ROTATION_DEG_S = 45.0          # rotation speed during a lift-off transition between orientations
STEP_MM = 0.05                 # default sample step along the path
_EPS = 1e-9                    # lane ends are clipped this far inside the region (keeps in_contact true)


@dataclass(frozen=True)
class Lane:
    """One planned lane: probe-centre path from start to end (x, y in mm) at a fixed yaw."""
    index: int                    # over the whole sweep
    yaw_deg: float
    k: int                        # lane number within this orientation (0 .. n-1)
    n: int                        # lanes in this orientation
    s_mm: float                   # lateral position p . u of the lane
    start: tuple[float, float]
    end: tuple[float, float]
    actual_overlap_pct: float

    @property
    def length_mm(self) -> float:
        return math.dist(self.start, self.end)


@dataclass(frozen=True)
class Segment:
    """A straight piece of the path: a lane (recorded) or a lift-off transition (lane is None)."""
    t0: float
    t1: float
    start: tuple[float, float, float]          # x, y, yaw_deg
    end: tuple[float, float, float]
    lane: Lane | None

    @property
    def recording(self) -> bool:
        return self.lane is not None


@dataclass(frozen=True)
class Sample:
    """A pose along the path at simulated time t; recording is False during transitions."""
    t: float
    T: np.ndarray
    recording: bool
    lane: Lane | None


def axes(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) in the phantom x-y plane for a yaw: lateral (along the array) and sweep (elevation) direction."""
    a = math.radians(yaw_deg)
    return np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])


def lane_positions(extent_lo: float, extent_hi: float, overlap_pct: float,
                   width_mm: float = PROBE.width_mm) -> tuple[np.ndarray, float]:
    """Evenly spread lane centres over [extent_lo, extent_hi] (outer image edges on the ends) and the actual
    overlap in percent."""
    extent = extent_hi - extent_lo
    if extent <= width_mm:
        return np.array([(extent_lo + extent_hi) / 2]), 100.0
    stride = width_mm * (1.0 - overlap_pct / 100.0)
    n = math.ceil((extent - width_mm) / stride - 1e-9) + 1
    centres = np.linspace(extent_lo + width_mm / 2, extent_hi - width_mm / 2, n)
    actual_stride = (extent - width_mm) / (n - 1)
    return centres, 100.0 * (width_mm - actual_stride) / width_mm


def _clip_line(p0: np.ndarray, d: np.ndarray, xr: tuple[float, float], yr: tuple[float, float]):
    """Parameter interval [t_lo, t_hi] where p0 + t d lies in the box, or None."""
    lo, hi = -math.inf, math.inf
    for p, dd, (a, b) in ((p0[0], d[0], xr), (p0[1], d[1], yr)):
        if abs(dd) < 1e-12:
            if not a <= p <= b:
                return None
            continue
        t1, t2 = sorted(((a - p) / dd, (b - p) / dd))
        lo, hi = max(lo, t1), min(hi, t2)
    return (lo, hi) if hi > lo else None


class ScriptedSweep:
    """Serpentine lanes for every orientation, joined by unrecorded transitions, on simulated time."""

    def __init__(self, cfg: SweepConfig | None = None):
        self.cfg = cfg or SweepConfig()
        self.width_mm = PROBE.width_mm
        self.lanes: list[Lane] = []
        self.overlap_pct: dict[float, float] = {}     # actual overlap per yaw
        for yaw in self.cfg.yaw_list_deg:
            self._plan_orientation(yaw)
        self.segments = self._build_segments()
        self._t0 = [s.t0 for s in self.segments]

    # ---- planning
    def _plan_orientation(self, yaw: float) -> None:
        u, v = axes(yaw)
        xr, yr = self.cfg.region_x_mm, self.cfg.region_y_mm
        corners = np.array([(x, y) for x in xr for y in yr])
        su = corners @ u
        centres, actual = lane_positions(su.min(), su.max(), self.cfg.overlap_pct, self.width_mm)
        self.overlap_pct[yaw] = actual
        lanes = []
        for s in centres:
            p0 = s * u
            iv = _clip_line(p0, v, xr, yr)
            if iv is None or iv[1] - iv[0] <= 2 * _EPS:
                continue
            a, b = p0 + (iv[0] + _EPS) * v, p0 + (iv[1] - _EPS) * v
            lanes.append((float(s), (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
        for k, (s, a, b) in enumerate(lanes):
            if self.cfg.serpentine and k % 2 == 1:
                a, b = b, a
            self.lanes.append(Lane(len(self.lanes), float(yaw), k, len(lanes), s, a, b, actual))

    def _build_segments(self) -> list[Segment]:
        speed = self.cfg.speed_mm_s
        segs: list[Segment] = []
        t = 0.0
        prev = None
        for lane in self.lanes:
            start = (*lane.start, lane.yaw_deg)
            if prev is not None and prev != start:
                dur = max(math.dist(prev[:2], start[:2]) / speed, abs(start[2] - prev[2]) / ROTATION_DEG_S)
                segs.append(Segment(t, t + dur, prev, start, None))
                t += dur
            end = (*lane.end, lane.yaw_deg)
            dur = lane.length_mm / speed
            segs.append(Segment(t, t + dur, start, end, lane))
            t += dur
            prev = end
        return segs

    # ---- queries
    @property
    def duration_s(self) -> float:
        return self.segments[-1].t1 if self.segments else 0.0

    def planned_lanes(self) -> list[Lane]:
        """Every lane in sweep order (for drawing: start, end, yaw)."""
        return list(self.lanes)

    def lanes_for(self, yaw_deg: float) -> list[Lane]:
        return [ln for ln in self.lanes if ln.yaw_deg == float(yaw_deg)]

    def expected_frames(self, spacing_mm: float | None = None) -> int:
        """Frames a distance-triggered acquisition captures: one at each lane start, then one per spacing."""
        sp = spacing_mm or self.cfg.frame_spacing_mm
        return sum(int(math.floor(ln.length_mm / sp + 1e-6)) + 1 for ln in self.lanes)

    def pose_at(self, t: float) -> Sample:
        """Pose at simulated time t (clamped to the sweep); at a lane / transition boundary the lane wins."""
        if not self.segments:
            raise ValueError("empty sweep")
        t = min(max(t, 0.0), self.duration_s)
        i = max(bisect.bisect_right(self._t0, t) - 1, 0)
        if i > 0 and not self.segments[i].recording and self.segments[i - 1].recording \
                and t <= self.segments[i - 1].t1 + 1e-12:
            i -= 1
        return self._sample(self.segments[i], t)

    def lane_at(self, t: float) -> Lane | None:
        return self.pose_at(t).lane

    @staticmethod
    def _sample(seg: Segment, t: float) -> Sample:
        f = 0.0 if seg.t1 <= seg.t0 else min(max((t - seg.t0) / (seg.t1 - seg.t0), 0.0), 1.0)
        x, y, yaw = (a + f * (b - a) for a, b in zip(seg.start, seg.end))
        return Sample(t, BModeSimulator.pose_from_xy_yaw(x, y, yaw), seg.recording, seg.lane)

    def samples(self, step_mm: float = STEP_MM) -> Iterator[Sample]:
        """Poses every step_mm of travel from each segment's start (transitions: travel or the equivalent rotation
        time), plus the segment's end. Each lane starts with a sample exactly at its start and ends exactly at its
        end, and its samples lie at whole multiples of step_mm, so a spacing that is a multiple of step_mm is hit
        exactly."""
        for j, seg in enumerate(self.segments):
            dur = seg.t1 - seg.t0
            d = dur * self.cfg.speed_mm_s                       # path length (equivalent for a rotation)
            n = int(math.floor(d / step_mm + 1e-9))
            fractions = [i * step_mm / d for i in range(n + 1)] if d > 0 else [0.0]
            if fractions[-1] < 1.0 - 1e-12:
                fractions.append(1.0)
            if not (seg.recording or j == 0):                  # a transition starts where the lane ended
                fractions = fractions[1:]
            for f in fractions:
                yield self._sample(seg, seg.t0 + dur * f)

    # ---- reports
    def coverage(self, px_mm: float = 0.25) -> dict:
        """Fraction of the region inside the planned lane footprints (each lane: width W along u over its
        clipped length), per yaw and for the union of all orientations."""
        xr, yr = self.cfg.region_x_mm, self.cfg.region_y_mm
        X, Y = np.meshgrid(np.arange(xr[0] + px_mm / 2, xr[1], px_mm), np.arange(yr[0] + px_mm / 2, yr[1], px_mm))
        union = np.zeros(X.shape, bool)
        out: dict = {}
        for yaw in self.cfg.yaw_list_deg:
            u, v = axes(yaw)
            S, Tt = X * u[0] + Y * u[1], X * v[0] + Y * v[1]
            cov = np.zeros(X.shape, bool)
            for ln in self.lanes_for(yaw):
                ta, tb = sorted((np.dot(ln.start, v), np.dot(ln.end, v)))
                cov |= (np.abs(S - ln.s_mm) <= self.width_mm / 2 + 1e-9) & (Tt >= ta - 1e-9) & (Tt <= tb + 1e-9)
            out[yaw] = float(cov.mean())
            union |= cov
        out["all"] = float(union.mean())
        return out

    def summary(self) -> list[str]:
        """One line per orientation: lanes, actual overlap, frames, planned coverage."""
        cov = self.coverage()
        lines = []
        for yaw in self.cfg.yaw_list_deg:
            lanes = self.lanes_for(yaw)
            frames = sum(int(math.floor(ln.length_mm / self.cfg.frame_spacing_mm + 1e-6)) + 1 for ln in lanes)
            lines.append(f"yaw {yaw:g}: {len(lanes)} lanes, overlap {self.cfg.overlap_pct:g}% requested / "
                         f"{self.overlap_pct[yaw]:.1f}% actual, {frames} frames, coverage {100 * cov[yaw]:.1f}%")
        lines.append(f"total: {len(self.lanes)} lanes, {self.expected_frames()} frames, "
                     f"coverage {100 * cov['all']:.1f}%, {self.duration_s:.1f} s simulated")
        return lines


# ---------------------------------------------------------------- mouse (S6)
class MousePose:
    """Hand-guided pose source: mouse position = face centre, left button = recording, yaw by keys (wall clock)."""

    ROTATE_STEP_DEG = 5.0
    SPEED_WINDOW_S = 0.3              # speed = path length over the last SPEED_WINDOW_S / elapsed time

    def __init__(self, x_mm: float = 50.0, y_mm: float = 50.0, yaw_deg: float = 0.0,
                 clock: Callable[[], float] = time.perf_counter):
        self.clock = clock
        self.t0 = clock()
        self.x, self.y, self.yaw = float(x_mm), float(y_mm), float(yaw_deg)
        self.button = False
        self._hist: deque = deque()        # (t, x, y)
        self._remember()

    @property
    def t(self) -> float:
        return self.clock() - self.t0

    @property
    def recording(self) -> bool:
        return self.button

    def _remember(self) -> None:
        t = self.t
        self._hist.append((t, self.x, self.y))
        while len(self._hist) > 2 and t - self._hist[1][0] > self.SPEED_WINDOW_S:
            self._hist.popleft()

    def move(self, x_mm: float, y_mm: float) -> None:
        self.x, self.y = float(x_mm), float(y_mm)
        self._remember()

    def press(self) -> None:
        self.button = True

    def release(self) -> None:
        self.button = False

    def rotate(self, sign: int) -> None:
        """Turn by +/- ROTATE_STEP_DEG (yaw kept in [-180, 180))."""
        self.yaw = (self.yaw + sign * self.ROTATE_STEP_DEG + 180.0) % 360.0 - 180.0

    def snap(self, yaw_deg: float) -> None:
        self.yaw = float(yaw_deg)

    def speed_mm_s(self) -> float:
        """Face-centre speed over the recent history (0 when the mouse has not moved for SPEED_WINDOW_S)."""
        self._remember()
        h = list(self._hist)
        if len(h) < 2 or h[-1][0] - h[0][0] <= 0:
            return 0.0
        path = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(h, h[1:]))
        return path / max(h[-1][0] - h[0][0], 1e-6)

    def sample(self) -> Sample:
        return Sample(self.t, BModeSimulator.pose_from_xy_yaw(self.x, self.y, self.yaw), self.recording, None)
