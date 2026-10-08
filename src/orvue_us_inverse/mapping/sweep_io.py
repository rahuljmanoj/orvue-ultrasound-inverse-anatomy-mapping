"""
orvue_us_inverse.mapping.sweep_io - frame records and the sweep container; save / load as compressed .npz (S0).

    sweep = Sweep(make_metadata(sim, SweepConfig(), AcquisitionConfig()))
    sweep.append(FrameRecord(index=0, t=0.0, T_true=T, T_measured=T, labels=lab, image=img))
    path = sweep.save(os.path.join(SWEEPS_DIR, "normal_yaw0.npz"))
    sweep = Sweep.load(path)

File layout (one .npz, every array stacked over the frames):
    index (N,) int64, t (N,) float64, T_true (N, 4, 4) float64, T_measured (N, 4, 4) float64,
    labels (N, 501, 301) int8, images (N, 501, 301) uint8 (only when the frames have images),
    metadata: JSON string, format_version: int.
"""
import dataclasses
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from orvue_us_inverse.mapping.probe import FRAME_SHAPE, probe_metadata
from orvue_us_inverse.paths import REPO_ROOT

FORMAT_VERSION = 1
SIMULATOR_SOURCE_REPOSITORY = "rahuljmanoj/orvue-ultrasound-simulator"
SIMULATOR_SOURCE_COMMIT = "e789389"          # UPSTREAM.md > Source (checked by tests/mapping/test_sweep_io.py)

# compressed bytes per frame, measured on 50-frame lanes at 0.5 mm (S0, STATUS.md): images 96-106 kB in every
# case; labels 1.4-1.6 kB over the anatomy, 150-200 B near the region edges (estimate = anatomy-rich upper value)
LABEL_BYTES_PER_FRAME = 1_500
IMAGE_BYTES_PER_FRAME = 100_000
POSE_BYTES_PER_FRAME = 2 * 16 * 8 + 16       # T_true, T_measured, index, t (stored, barely compressible)
FIXED_BYTES = 2_000                          # zip headers and metadata


@dataclass
class FrameRecord:
    """One acquired frame: pose used to make it, pose used to reconstruct it, oracle labels, optional B-mode."""
    index: int
    t: float                         # simulated time (s)
    T_true: np.ndarray               # 4x4 phantom <- probe, used to generate the frame
    T_measured: np.ndarray           # 4x4, used for reconstruction (= T_true unless errors are injected)
    labels: np.ndarray               # int8, FRAME_SHAPE
    image: np.ndarray | None = None  # uint8, FRAME_SHAPE

    def __post_init__(self):
        self.index, self.t = int(self.index), float(self.t)
        self.T_true = np.asarray(self.T_true, np.float64)
        self.T_measured = np.asarray(self.T_measured, np.float64)
        if self.T_true.shape != (4, 4) or self.T_measured.shape != (4, 4):
            raise ValueError("T_true and T_measured must be 4x4")
        if self.labels.shape != FRAME_SHAPE or self.labels.dtype != np.int8:
            raise ValueError(f"labels must be int8 {FRAME_SHAPE}, got {self.labels.dtype} {self.labels.shape}")
        if self.image is not None and (self.image.shape != FRAME_SHAPE or self.image.dtype != np.uint8):
            raise ValueError(f"image must be uint8 {FRAME_SHAPE}, got {self.image.dtype} {self.image.shape}")


class Sweep:
    """Metadata (JSON-ready dict) and the frames in acquisition order. Either every frame has an image or none."""

    def __init__(self, metadata: dict | None = None, frames: list[FrameRecord] | None = None):
        self.metadata = dict(metadata or {})
        self.frames: list[FrameRecord] = []
        for f in frames or []:
            self.append(f)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def has_images(self) -> bool:
        return bool(self.frames) and self.frames[0].image is not None

    def append(self, frame: FrameRecord) -> None:
        if self.frames and (frame.image is None) != (self.frames[0].image is None):
            raise ValueError("either every frame of a sweep has an image or none")
        self.frames.append(frame)

    def save(self, path: str) -> str:
        """Write a compressed .npz (folders created; '.npz' added if missing); returns the path written."""
        if not path.endswith(".npz"):
            path += ".npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fr = self.frames
        arrays = dict(
            index=np.array([f.index for f in fr], np.int64),
            t=np.array([f.t for f in fr], np.float64),
            T_true=np.array([f.T_true for f in fr], np.float64).reshape(-1, 4, 4),
            T_measured=np.array([f.T_measured for f in fr], np.float64).reshape(-1, 4, 4),
            labels=np.array([f.labels for f in fr], np.int8).reshape(-1, *FRAME_SHAPE),
            metadata=np.array(json.dumps(self.metadata)),
            format_version=np.array(FORMAT_VERSION),
        )
        if self.has_images:
            arrays["images"] = np.array([f.image for f in fr], np.uint8)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str) -> "Sweep":
        with np.load(path, allow_pickle=False) as z:
            version = int(z["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(f"{path}: sweep format {version}, expected {FORMAT_VERSION}")
            images = z["images"] if "images" in z.files else None
            frames = [FrameRecord(i, t, Tt, Tm, lab, None if images is None else images[k])
                      for k, (i, t, Tt, Tm, lab) in enumerate(zip(z["index"], z["t"], z["T_true"],
                                                                  z["T_measured"], z["labels"]))]
            return cls(json.loads(str(z["metadata"])), frames)


def git_commit(repo: str = REPO_ROOT) -> str | None:
    """This repository's commit (git rev-parse HEAD), or None when git or the repository is not available."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def provenance() -> dict:
    """This repository's git commit and the simulator source the copied code came from (UPSTREAM.md)."""
    return dict(repo_commit=git_commit(), simulator_source_repository=SIMULATOR_SOURCE_REPOSITORY,
                simulator_source_commit=SIMULATOR_SOURCE_COMMIT)


def make_metadata(sim, sweep_config, acquisition_config, **extra) -> dict:
    """Sweep metadata: case and simulator settings (from make_simulator), probe fields, configs, frame shape,
    UTC timestamp, provenance; extra keys are added as given (must be JSON-ready)."""
    meta = dict(
        case=sim.case,
        simulator=dict(sim.settings),
        probe=probe_metadata(),
        frame_shape=list(FRAME_SHAPE),
        sweep_config=dataclasses.asdict(sweep_config),
        acquisition_config=dataclasses.asdict(acquisition_config),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        provenance=provenance(),
    )
    meta.update(extra)
    return json.loads(json.dumps(meta))      # tuples -> lists, so the dict equals what load() returns


def estimate_size_mb(n_frames: int, store_images: bool) -> float:
    """Approximate .npz size in MB (1e6 bytes). B-mode speckle hardly compresses (~100 kB/frame, within ~10% in
    every case); labels compress to ~1.5 kB/frame over the anatomy and less elsewhere, so labels-only sweeps
    away from the anatomy come out smaller than estimated."""
    per_frame = LABEL_BYTES_PER_FRAME + POSE_BYTES_PER_FRAME + (IMAGE_BYTES_PER_FRAME if store_images else 0)
    return (n_frames * per_frame + FIXED_BYTES) / 1e6
