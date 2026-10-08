"""
orvue_us_inverse.mapping.config - settings for inverse mapping as dataclasses with defaults (S0).

    GridConfig         reconstruction grid: bounds in the phantom frame (mm), voxel size, shape, voxel centres
    SweepConfig        scripted sweep: orientations, lane overlap, frame spacing, triggers, speed, region
    AcquisitionConfig  what is stored per frame

Probe width and depth are not repeated here: they come from mapping/probe.py (SweepConfig.probe_width_mm,
SweepConfig.probe_depth_mm). Every config converts to a JSON-ready dict with dataclasses.asdict and back with
Config(**d).
"""
from dataclasses import dataclass, field

import numpy as np

from orvue_us_inverse.mapping.probe import PROBE


def _bounds(b) -> tuple[float, float]:
    lo, hi = (float(v) for v in b)
    if not hi > lo:
        raise ValueError(f"bounds must be (low, high) with high > low, got {b}")
    return lo, hi


@dataclass
class GridConfig:
    """Voxel grid over the phantom volume. Voxels are indexed [ix, iy, iz]; voxel i spans
    [lo + i * voxel_mm, lo + (i + 1) * voxel_mm) and its centre is lo + (i + 0.5) * voxel_mm."""
    x_mm: tuple[float, float] = (0.0, 100.0)
    y_mm: tuple[float, float] = (0.0, 100.0)
    z_mm: tuple[float, float] = (0.0, 50.0)
    voxel_mm: float = 0.5

    def __post_init__(self):
        self.x_mm, self.y_mm, self.z_mm = _bounds(self.x_mm), _bounds(self.y_mm), _bounds(self.z_mm)
        self.voxel_mm = float(self.voxel_mm)
        if self.voxel_mm <= 0:
            raise ValueError("voxel_mm must be > 0")
        for lo, hi in (self.x_mm, self.y_mm, self.z_mm):
            n = (hi - lo) / self.voxel_mm
            if abs(n - round(n)) > 1e-6:
                raise ValueError(f"extent {hi - lo} mm is not a whole number of {self.voxel_mm} mm voxels")

    @property
    def shape(self) -> tuple[int, int, int]:
        """(nx, ny, nz): (200, 200, 100) for the defaults."""
        return tuple(round((hi - lo) / self.voxel_mm) for lo, hi in (self.x_mm, self.y_mm, self.z_mm))

    @property
    def n_voxels(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    def centres(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Voxel-centre coordinates along x, y and z (1D arrays, mm)."""
        return tuple(lo + (np.arange(n) + 0.5) * self.voxel_mm
                     for (lo, _), n in zip((self.x_mm, self.y_mm, self.z_mm), self.shape))


@dataclass
class SweepConfig:
    """Scripted sweep: serpentine lanes over the region, one pass per yaw in yaw_list_deg.

    overlap_pct is the requested overlap between adjacent lanes ((width - stride) / width); a frame is taken
    every frame_spacing_mm of travel or angle_trigger_deg of rotation, whichever comes first.
    """
    yaw_list_deg: list[float] = field(default_factory=lambda: [0.0])
    overlap_pct: float = 20.0
    frame_spacing_mm: float = 0.5
    angle_trigger_deg: float = 1.0
    speed_mm_s: float = 10.0
    serpentine: bool = True
    region_x_mm: tuple[float, float] = (0.0, 100.0)
    region_y_mm: tuple[float, float] = (0.0, 100.0)

    def __post_init__(self):
        self.yaw_list_deg = [float(a) for a in self.yaw_list_deg]
        self.region_x_mm, self.region_y_mm = _bounds(self.region_x_mm), _bounds(self.region_y_mm)
        if not self.yaw_list_deg:
            raise ValueError("yaw_list_deg needs at least one orientation")
        if not 0.0 <= self.overlap_pct < 100.0:
            raise ValueError("overlap_pct must be in [0, 100)")
        for name in ("frame_spacing_mm", "angle_trigger_deg", "speed_mm_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")

    @property
    def probe_width_mm(self) -> float:
        return PROBE.width_mm

    @property
    def probe_depth_mm(self) -> float:
        return PROBE.depth_mm


@dataclass
class AcquisitionConfig:
    """Per-frame storage. Oracle labels are always stored; store_images adds the B-mode frame (render()),
    otherwise only labels_image() is computed (~6x faster). The simulator always comes from
    mapping.probe.make_simulator()."""
    store_images: bool = True
