"""
orvue_us_inverse.mapping.recon - reconstruction of a label volume from oracle labels (S2).

    grid = VoxelGrid(GridConfig())                           # 200 x 200 x 100 voxels of 0.5 mm
    comp = LabelCompounder.from_sweep(sweep, grid)          # probe geometry from the sweep metadata
    for frame in sweep.frames:
        comp.insert(frame)                                   # uses frame.T_measured
    labels = comp.result()                                   # int8 (nx, ny, nz), -1 = unobserved
    filled = fill_small_holes(labels, max_gap_voxels=1)

Each frame pixel is placed in 3D with pixel_points (the mapping of BModeSimulator.plane_points:
o + s*u + d*n, s lateral in [-W/2, W/2], d depth in [0, D], px_mm pixels) and its label is added to the class
votes of the voxel containing it (nearest-voxel binning, vectorised: one np.unique over voxel * n_classes + label
per frame, so a frame costs a sort of its ~151 000 pixels rather than an np.add.at per pixel). A voxel's label is the class
with the most votes; TIE RULE: the lowest class index wins (np.argmax), so the result is deterministic and does not
depend on insertion order. Voxels never hit stay -1. With images, the mean B-mode intensity per voxel is
accumulated alongside (NaN where no image pixel landed).

Optional elevation splat (splat_mm > 0): every pixel is also inserted at elevation offsets
-splat_mm/2 .. +splat_mm/2 (step voxel_mm / 2) along v, to model the slice thickness; default 0 = the image plane
only (pixel-nearest-neighbour).

Votes are uint16 and saturate at 65535 votes of one class in one voxel (a 0.5 mm voxel gets ~25 pixels per frame
through it, so this needs > 2600 frames through the same voxel). Hits are uint32.
"""
import dataclasses
import functools

import numpy as np

from orvue_us_inverse.mapping.config import GridConfig
from orvue_us_inverse.mapping.probe import probe_metadata
from orvue_us_inverse.mapping.sweep_io import FrameRecord, Sweep

N_CLASSES = 11               # anatomy.TISSUES labels 0-10
UNOBSERVED = -1


class VoxelGrid:
    """Axis-aligned voxel grid in the phantom frame; voxels indexed [ix, iy, iz], flat index in C order."""

    def __init__(self, cfg: GridConfig | None = None):
        self.cfg = cfg or GridConfig()
        self.shape = self.cfg.shape
        self.voxel_mm = self.cfg.voxel_mm
        self.lo = np.array([self.cfg.x_mm[0], self.cfg.y_mm[0], self.cfg.z_mm[0]], np.float64)
        self.n_voxels = self.cfg.n_voxels

    def index(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Voxel indices (N, 3) int64 of points P (N, 3) and the mask of points inside the grid."""
        ijk = np.floor((np.asarray(P, np.float64) - self.lo) / self.voxel_mm).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < np.array(self.shape)), axis=1)
        return ijk, inside

    def flat_index(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fast path for float32 points: flat voxel index (int64) of the points inside the grid, and the mask.
        Same voxels as index() up to float32 rounding at voxel faces."""
        q = (P - self.lo.astype(np.float32)) * np.float32(1.0 / self.voxel_mm)
        nx, ny, nz = self.shape
        inside = ((q[:, 0] >= 0) & (q[:, 0] < nx) & (q[:, 1] >= 0) & (q[:, 1] < ny)
                  & (q[:, 2] >= 0) & (q[:, 2] < nz))
        i = q[inside].astype(np.int64)                    # truncation = floor for q >= 0
        return (i[:, 0] * ny + i[:, 1]) * nz + i[:, 2], inside

    def flat(self, ijk: np.ndarray) -> np.ndarray:
        return np.ravel_multi_index(ijk.T, self.shape)

    def centre(self, ijk: np.ndarray) -> np.ndarray:
        """Centres (N, 3) of voxels ijk (N, 3)."""
        return self.lo + (np.asarray(ijk, np.float64) + 0.5) * self.voxel_mm

    def centres(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.cfg.centres()

    def centre_points(self, flat_idx: np.ndarray | None = None) -> np.ndarray:
        """Centres (N, 3) of the voxels with these flat indices (every voxel, in flat order, if None)."""
        if flat_idx is None:
            flat_idx = np.arange(self.n_voxels)
        return self.centre(np.stack(np.unravel_index(flat_idx, self.shape), 1))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self.cfg)

    @classmethod
    def from_dict(cls, d: dict) -> "VoxelGrid":
        return cls(GridConfig(**d))


@functools.lru_cache(maxsize=8)
def _lattice(width_mm: float, depth_mm: float, px_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Lateral (nx,) and depth (nz,) pixel offsets, float32, built exactly as BModeSimulator builds them."""
    lat = np.arange(-width_mm / 2, width_mm / 2 + 1e-6, px_mm, dtype=np.float32)
    ax = np.arange(0.0, depth_mm + 1e-6, px_mm, dtype=np.float32)
    lat.flags.writeable = ax.flags.writeable = False
    return lat, ax


def pixel_points(T: np.ndarray, probe: dict | None = None, elev_mm: float = 0.0) -> np.ndarray:
    """Phantom-frame points (nz * nx, 3) float32 of every image pixel for pose T (row-major: rows = depth), from
    the probe geometry (width_mm, depth_mm, px_mm; default mapping.probe). Same values as
    BModeSimulator.plane_points(T, elev_mm): ((o + e v) + s u) + d n in float32, evaluated per lateral column and
    per depth row, then broadcast."""
    p = probe or probe_metadata()
    lat, ax = _lattice(float(p["width_mm"]), float(p["depth_mm"]), float(p["px_mm"]))
    T = np.asarray(T).astype(np.float32)
    o, u, v, n = T[:3, 3], T[:3, 0], T[:3, 1], T[:3, 2]
    row = (o + np.float32(elev_mm) * v) + lat[:, None] * u          # (nx, 3)
    col = ax[:, None] * n                                            # (nz, 3)
    return (row[None, :, :] + col[:, None, :]).reshape(-1, 3)


class LabelCompounder:
    """Per-voxel class votes, hit counts and (with images) intensity sums; result() gives the label volume."""

    def __init__(self, grid: VoxelGrid, n_classes: int = N_CLASSES, probe: dict | None = None,
                 splat_mm: float = 0.0):
        self.grid = grid
        self.n_classes = n_classes
        self.probe = dict(probe or probe_metadata())
        self.splat_mm = float(splat_mm)
        if self.splat_mm > 0:
            h, step = self.splat_mm / 2, grid.voxel_mm / 2
            self.elev_offsets = np.arange(-h, h + 1e-9, step)
        else:
            self.elev_offsets = np.zeros(1)
        self.votes = np.zeros(grid.n_voxels * n_classes, np.uint16)       # [voxel * n_classes + class]
        self.hits = np.zeros(grid.n_voxels, np.uint32)
        self.intensity_sum = np.zeros(grid.n_voxels, np.float64)
        self.intensity_hits = np.zeros(grid.n_voxels, np.uint32)
        self.n_frames = 0

    @classmethod
    def from_sweep(cls, sweep: Sweep, grid: VoxelGrid, **kw) -> "LabelCompounder":
        """Compounder with the probe geometry stored in the sweep metadata."""
        return cls(grid, probe=sweep.metadata.get("probe"), **kw)

    def _binned(self, frames: list[FrameRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Flat voxel index, label and intensity (or None) of every pixel of the frames that lands in the grid."""
        flats, labs, imgs = [], [], []
        with_images = all(f.image is not None for f in frames)
        for f in frames:
            lab = f.labels.ravel()
            if lab.min() < 0 or lab.max() >= self.n_classes:
                raise ValueError(f"frame {f.index}: labels outside 0..{self.n_classes - 1}")
            for e in self.elev_offsets:
                flat, inside = self.grid.flat_index(pixel_points(f.T_measured, self.probe, e))
                flats.append(flat)
                labs.append(lab[inside])
                if with_images:
                    imgs.append(f.image.ravel()[inside])
        if not flats:
            return np.zeros(0, np.int64), np.zeros(0, np.int8), None
        return (np.concatenate(flats), np.concatenate(labs).astype(np.int64),
                np.concatenate(imgs).astype(np.float64) if with_images else None)

    def insert(self, frame: FrameRecord) -> None:
        """Add one frame (pose T_measured)."""
        self.insert_batch([frame])

    def insert_batch(self, frames: list[FrameRecord]) -> None:
        """Add several frames at once (same result as inserting them one by one)."""
        flat, lab, img = self._binned(list(frames))
        keys, counts = np.unique(flat * self.n_classes + lab, return_counts=True)
        v = self.votes[keys].astype(np.int64) + counts                 # keys are unique: plain fancy indexing
        self.votes[keys] = np.minimum(v, np.iinfo(np.uint16).max)
        np.add.at(self.hits, keys // self.n_classes, counts.astype(np.uint32))   # a few thousand entries
        if img is not None:
            uv, inv = np.unique(flat, return_inverse=True)
            self.intensity_sum[uv] += np.bincount(inv, weights=img)
            self.intensity_hits[uv] += np.bincount(inv).astype(np.uint32)
        self.n_frames += len(frames)

    @property
    def observed(self) -> np.ndarray:
        """Boolean volume (nx, ny, nz) of voxels hit at least once."""
        return (self.hits > 0).reshape(self.grid.shape)

    def result(self) -> np.ndarray:
        """int8 label volume (nx, ny, nz): majority class (ties: lowest class index), -1 where unobserved."""
        lab = np.argmax(self.votes.reshape(-1, self.n_classes), axis=1).astype(np.int8)
        lab[self.hits == 0] = UNOBSERVED
        return lab.reshape(self.grid.shape)

    def result_slice(self, axis: int, index: int) -> np.ndarray:
        """result() restricted to one slice (axis 0: x = index -> (ny, nz); 1: y -> (nx, nz); 2: z -> (nx, ny)),
        computed from that slice's votes only (fast enough for a live view)."""
        nx, ny, nz = self.grid.shape
        sel = [slice(None)] * 3
        sel[axis] = index
        votes = self.votes.reshape(nx, ny, nz, self.n_classes)[tuple(sel)]
        lab = np.argmax(votes, axis=-1).astype(np.int8)
        lab[self.hits.reshape(nx, ny, nz)[tuple(sel)] == 0] = UNOBSERVED
        return lab

    def column_hits(self) -> np.ndarray:
        """Hits summed over depth, (nx, ny): the coverage map seen from above."""
        nx, ny, nz = self.grid.shape
        return self.hits.reshape(nx, ny, nz).sum(axis=2, dtype=np.uint64)

    def intensity(self) -> np.ndarray | None:
        """float32 mean intensity volume (NaN where no image pixel landed), or None without images."""
        if not self.intensity_hits.any():
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            m = self.intensity_sum / self.intensity_hits
        m[self.intensity_hits == 0] = np.nan
        return m.astype(np.float32).reshape(self.grid.shape)


def _shift(a: np.ndarray, k: int, axis: int, fill) -> np.ndarray:
    """a shifted by k along axis (out[i] = a[i - k]), filled with `fill` where nothing shifts in."""
    out = np.full_like(a, fill)
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    if k > 0:
        src[axis], dst[axis] = slice(0, -k), slice(k, None)
    else:
        src[axis], dst[axis] = slice(-k, None), slice(0, k)
    out[tuple(dst)] = a[tuple(src)]
    return out


def fill_small_holes(labels: np.ndarray, max_gap_voxels: int = 1, n_classes: int = N_CLASSES) -> np.ndarray:
    """Fill unobserved voxels that lie in a gap of at most max_gap_voxels unobserved voxels between two observed
    voxels along at least one axis. The fill label is the majority of the bounding observed voxels over every
    qualifying axis (ties: lowest class index). One pass: filled voxels do not seed further filling, and a voxel
    without observed voxels on both sides along some axis is never filled, so nothing grows into unscanned
    regions. Observed voxels are unchanged. Returns a new volume."""
    if max_gap_voxels < 1:
        return labels.copy()
    unobs = labels < 0
    counts = np.zeros((n_classes,) + labels.shape, np.uint8)
    big = np.iinfo(np.int16).max
    for axis in range(labels.ndim):
        nearest = []
        for sign in (1, -1):                         # observed neighbour on the - side, then on the + side
            lab = np.full(labels.shape, UNOBSERVED, np.int8)
            dist = np.full(labels.shape, big, np.int16)
            for k in range(1, max_gap_voxels + 1):
                sh = _shift(labels, sign * k, axis, UNOBSERVED)
                new = (lab < 0) & (sh >= 0)
                lab[new], dist[new] = sh[new], k
            nearest.append((lab, dist))
        (la, da), (lb, db) = nearest
        ok = unobs & (la >= 0) & (lb >= 0) & (da.astype(np.int32) + db - 1 <= max_gap_voxels)
        for c in range(n_classes):
            counts[c] += (ok & (la == c)).astype(np.uint8) + (ok & (lb == c)).astype(np.uint8)
    fill = counts.sum(axis=0) > 0
    out = labels.copy()
    out[fill] = np.argmax(counts, axis=0)[fill].astype(np.int8)
    return out


def ground_truth(anatomy, grid: VoxelGrid, flat_idx: np.ndarray | None = None) -> np.ndarray:
    """Anatomy.labels at voxel centres: the whole volume (nx, ny, nz) if flat_idx is None, else the listed voxels.
    Reference for the S2 checks; the S4 evaluation builds its metrics on it."""
    lab = anatomy.labels(grid.centre_points(flat_idx).astype(np.float32))
    return lab.reshape(grid.shape) if flat_idx is None else lab


def interior_mask(gt: np.ndarray) -> np.ndarray:
    """Voxels whose 26 neighbours all have the same ground-truth label as the voxel (border voxels excluded)."""
    same = np.ones(gt.shape, bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                nb = np.roll(gt, (dx, dy, dz), axis=(0, 1, 2))
                same &= nb == gt
    same[[0, -1], :, :] = same[:, [0, -1], :] = same[:, :, [0, -1]] = False
    return same
