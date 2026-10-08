"""
orvue_us_inverse.mapping.acquisition - distance-triggered frame capture from the simulator (S1).

Acquirer(sim, sweep_cfg, acq_cfg) receives poses from any pose source (feed(T, t, recording)) and captures a
frame when, since the last captured frame, the face centre has moved >= frame_spacing_mm or the probe has rotated
>= angle_trigger_deg; the first pose of every recorded stretch (a lane after a lift-off) is always captured.
Nothing is captured while not recording or not in contact (sim.in_contact).

Capture: sim.render(T, return_labels=True) when acq_cfg.store_images, else sim.labels_image(T) only (~6x faster);
FrameRecord with T_true = T_measured = T. Time is the pose source's simulated time: feed() renders synchronously,
so a slow render slows playback instead of dropping frames.

    sim = make_simulator("normal")
    acq = Acquirer(sim, SweepConfig(), AcquisitionConfig())
    for s in ScriptedSweep(cfg).samples():
        acq.feed(s.T, s.t, s.recording)
    acq.sweep.save(path)
"""
import math

import numpy as np

from orvue_us_inverse.mapping.config import AcquisitionConfig, SweepConfig
from orvue_us_inverse.mapping.sweep_io import FrameRecord, Sweep, make_metadata

# poses from pose_from_xy_yaw are float32 (~1e-5 mm at 100 mm): a sample exactly one spacing away must trigger
_TOL_MM = 1e-4
_TOL_DEG = 1e-3


def rotation_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle of the rotation between two 3x3 rotation matrices, in degrees."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


class Acquirer:
    """Distance / angle-triggered capture into a Sweep (acq.sweep)."""

    def __init__(self, sim, sweep_cfg: SweepConfig | None = None, acq_cfg: AcquisitionConfig | None = None,
                 **metadata_extra):
        self.sim = sim
        self.sweep_cfg = sweep_cfg or SweepConfig()
        self.acq_cfg = acq_cfg or AcquisitionConfig()
        self.sweep = Sweep(make_metadata(sim, self.sweep_cfg, self.acq_cfg, **metadata_extra))
        self.last: FrameRecord | None = None
        self.n_not_in_contact = 0          # recorded poses skipped because the probe was off the region
        self._new_stretch = True           # next recorded pose starts a lane: capture it

    def triggered(self, T: np.ndarray) -> bool:
        """True when T is far enough (distance or angle) from the last captured frame, or starts a lane."""
        if self._new_stretch or self.last is None:
            return True
        L = self.last.T_true
        moved = float(np.linalg.norm(T[:3, 3] - L[:3, 3]))
        return (moved >= self.sweep_cfg.frame_spacing_mm - _TOL_MM
                or rotation_deg(L[:3, :3], T[:3, :3]) >= self.sweep_cfg.angle_trigger_deg - _TOL_DEG)

    def feed(self, T: np.ndarray, t: float, recording: bool) -> FrameRecord | None:
        """Offer a pose at simulated time t; returns the captured FrameRecord or None."""
        if not recording:
            self._new_stretch = True
            return None
        if not self.sim.in_contact(T):
            self.n_not_in_contact += 1
            return None
        if not self.triggered(T):
            return None
        image, labels = self._capture(T)
        T = np.array(T, np.float64)
        frame = FrameRecord(len(self.sweep), t, T, T.copy(), labels, image)
        self.sweep.append(frame)
        self.last, self._new_stretch = frame, False
        return frame

    def _capture(self, T: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        if self.acq_cfg.store_images:
            return self.sim.render(T, return_labels=True)
        return None, self.sim.labels_image(T)
