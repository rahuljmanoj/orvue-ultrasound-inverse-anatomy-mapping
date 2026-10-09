"""mapping/recon.py and mapping/reconstruct_sweep.py: reconstruction from oracle labels (S2).

Ground-truth test: oracle-label sweep at 0.25 mm, yaw 0, over the GB / hilum block x, y in [25, 75] mm (2 lanes of
201 frames, the full 50 mm depth; ~6 s of labels_image), reconstructed at 0.5 mm and compared with Anatomy.labels at
the voxel centres. The whole-region figures (4 lanes, 1604 frames) are in STATUS.md (S2).
"""
import json
import os
import time
from collections import Counter

import numpy as np
import pytest

from orvue_us_inverse.mapping.acquisition import Acquirer
from orvue_us_inverse.mapping.config import AcquisitionConfig, GridConfig, SweepConfig
from orvue_us_inverse.mapping.poses import ScriptedSweep
from orvue_us_inverse.mapping.probe import make_simulator, probe_metadata
from orvue_us_inverse.mapping.recon import (LabelCompounder, VoxelGrid, fill_small_holes, ground_truth,
                                            interior_mask, pixel_points)
from orvue_us_inverse.mapping.reconstruct_sweep import main as reconstruct_main
from orvue_us_inverse.mapping.sweep_io import Sweep
from orvue_us_inverse.simulation.anatomy import TISSUES
from orvue_us_inverse.simulation.bmode import BModeSimulator

NAMES = [TISSUES[i][0] for i in range(11)]
BLOCK = (25.0, 75.0)


@pytest.fixture(scope="module")
def sim():
    return make_simulator("normal")


@pytest.fixture(scope="module")
def block_sweep(sim):
    cfg = SweepConfig(frame_spacing_mm=0.25, region_x_mm=BLOCK, region_y_mm=BLOCK)
    acq = Acquirer(sim, cfg, AcquisitionConfig(store_images=False))
    for s in ScriptedSweep(cfg).samples():
        acq.feed(s.T, s.t, s.recording)
    assert len(acq.sweep) == 2 * 201
    return acq.sweep


@pytest.fixture(scope="module")
def block_recon(block_sweep):
    grid = VoxelGrid(GridConfig())
    comp = LabelCompounder.from_sweep(block_sweep, grid)
    t = []
    for f in block_sweep.frames:
        t0 = time.perf_counter()
        comp.insert(f)
        t.append(time.perf_counter() - t0)
    return comp, np.array(t) * 1000


@pytest.fixture(scope="module")
def gt(sim):
    return ground_truth(sim.an, VoxelGrid(GridConfig()))


def _random_pose(rng):
    """Random position / yaw, tilted up to 20 degrees about a random horizontal axis."""
    T = BModeSimulator.pose_from_xy_yaw(*rng.uniform(0, 100, 2), rng.uniform(-180, 180)).astype(np.float64)
    a = rng.uniform(0, 2 * np.pi)
    k = np.array([np.cos(a), np.sin(a), 0.0])
    th = np.radians(rng.uniform(0, 20))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    T[:3, :3] = R @ T[:3, :3]
    T[2, 3] = rng.uniform(0, 2)
    return T.astype(np.float32)


# ---------------------------------------------------------------- geometry
def test_pixel_points_equal_plane_points(sim):
    rng = np.random.default_rng(7)
    for _ in range(10):
        T = _random_pose(rng)
        assert np.array_equal(pixel_points(T, probe_metadata()), sim.plane_points(T))
        e = float(rng.uniform(-1, 1))
        assert np.array_equal(pixel_points(T, probe_metadata(), e), sim.plane_points(T, elev=e))


def test_voxel_grid():
    g = VoxelGrid(GridConfig())
    assert g.shape == (200, 200, 100) and g.n_voxels == 4_000_000
    rng = np.random.default_rng(3)
    ijk = np.stack([rng.integers(0, n, 1000) for n in g.shape], 1)
    c = g.centre(ijk)
    back, inside = g.index(c)
    assert inside.all() and np.array_equal(back, ijk)
    flat, inside32 = g.flat_index(c.astype(np.float32))
    assert inside32.all() and np.array_equal(flat, g.flat(ijk))
    assert np.allclose(g.centre_points(g.flat(ijk)), c)
    _, out = g.index(np.array([[-0.1, 5, 5], [5, 100.0, 5], [5, 5, 49.9]]))
    assert out.tolist() == [False, False, True]
    assert VoxelGrid.from_dict(g.to_dict()).shape == g.shape


# ---------------------------------------------------------------- ground truth
def test_reconstruction_matches_ground_truth(block_sweep, block_recon, gt):
    comp, t_ms = block_recon
    lab = comp.result()
    obs = lab >= 0
    inter = interior_mask(gt) & obs
    ok = lab == gt
    acc_all, acc_int = ok[obs].mean(), ok[inter].mean()
    conf = Counter(zip(gt[obs & ~ok].tolist(), lab[obs & ~ok].tolist()))
    print(f"\n[S2] {len(block_sweep)} frames at 0.25 mm over x, y {BLOCK} mm; insert mean {t_ms.mean():.1f} ms, "
          f"median {np.median(t_ms):.1f}, max {t_ms.max():.1f} ms/frame")
    print(f"[S2] observed {100 * obs.mean():.1f}% of the 100 x 100 x 50 mm volume; correct: "
          f"{100 * acc_all:.3f}% of {obs.sum()} observed voxels, {100 * acc_int:.3f}% of {inter.sum()} interior")
    print("[S2] mismatches (truth -> recon): " + ", ".join(f"{NAMES[a]}->{NAMES[b]} {n}"
                                                           for (a, b), n in conf.most_common(10)))
    assert acc_int >= 0.995
    assert acc_all >= 0.95
    # the observed block is the scanned region at full depth: x, y in [25, 75], z in [0, 50); the image edges and
    # lane ends at 75 mm lie on a voxel face and fall in voxel 150 ([75, 75.5) mm), the depth 50 mm is outside
    xs, ys, zs = np.nonzero(obs)
    assert (xs.min(), xs.max(), ys.min(), ys.max(), zs.min(), zs.max()) == (50, 150, 50, 150, 0, 99)
    assert t_ms.mean() < 30.0                    # target 15 ms on a laptop CPU (~9-10 ms measured); loose guard


def test_unobserved_stay_minus_one(block_recon):
    comp, _ = block_recon
    lab = comp.result()
    assert (lab[:50] == -1).all() and (lab[151:] == -1).all()          # x < 25 mm and x > 75.5 mm: never imaged
    assert ((lab == -1) == (comp.hits.reshape(lab.shape) == 0)).all()
    assert lab.dtype == np.int8


def test_incremental_equals_batch(sim):
    sim._rng = np.random.default_rng(0)
    frames = []
    cfg = SweepConfig(frame_spacing_mm=0.5)
    acq = Acquirer(sim, cfg, AcquisitionConfig(store_images=True))
    for s in ScriptedSweep(cfg).samples():
        if s.t > 2.0:
            break
        acq.feed(s.T, s.t, s.recording)
    frames = acq.sweep.frames
    assert len(frames) == 41
    grid = VoxelGrid(GridConfig())
    one, many = LabelCompounder(grid), LabelCompounder(grid)
    for f in frames:
        one.insert(f)
    many.insert_batch(frames[:15])
    many.insert_batch(frames[15:])
    for name in ("votes", "hits", "intensity_sum", "intensity_hits"):
        assert np.array_equal(getattr(one, name), getattr(many, name)), name
    assert np.array_equal(one.result(), many.result())
    inten = one.intensity()
    obs = one.observed
    assert np.isfinite(inten[obs]).all() and np.isnan(inten[~obs]).all()
    assert 0 <= np.nanmin(inten) and np.nanmax(inten) <= 255


def test_intensity_from_frames_with_images_only(sim):
    """Mixed sweep (B-mode on for some frames): votes use every frame, intensities only the frames with images."""
    import dataclasses
    sim._rng = np.random.default_rng(1)
    cfg = SweepConfig(frame_spacing_mm=0.5)
    acq = Acquirer(sim, cfg, AcquisitionConfig(store_images=True))
    for s in ScriptedSweep(cfg).samples():
        if s.t > 1.0:
            break
        acq.feed(s.T, s.t, s.recording)
    frames = acq.sweep.frames
    half = [f if k < len(frames) // 2 else dataclasses.replace(f, image=None) for k, f in enumerate(frames)]
    grid = VoxelGrid(GridConfig())
    mixed, images_only, labels_all = LabelCompounder(grid), LabelCompounder(grid), LabelCompounder(grid)
    mixed.insert_batch(half)
    images_only.insert_batch(frames[:len(frames) // 2])
    labels_all.insert_batch([dataclasses.replace(f, image=None) for f in frames])
    assert np.array_equal(mixed.votes, labels_all.votes) and np.array_equal(mixed.hits, labels_all.hits)
    assert np.array_equal(mixed.intensity_sum, images_only.intensity_sum)
    assert np.array_equal(mixed.intensity_hits, images_only.intensity_hits)


def test_tie_rule_lowest_class_wins():
    grid = VoxelGrid(GridConfig(x_mm=(0, 1), y_mm=(0, 1), z_mm=(0, 1), voxel_mm=0.5))
    c = LabelCompounder(grid)
    v = c.votes.reshape(-1, c.n_classes)
    v[0, 3] = v[0, 7] = 5                     # tie 3 / 7 -> 3
    v[1, 9], v[1, 2] = 4, 3                   # majority 9
    c.hits[:2] = (10, 7)
    lab = c.result().ravel()
    assert lab[0] == 3 and lab[1] == 9 and (lab[2:] == -1).all()


# ---------------------------------------------------------------- hole filling
def test_fill_small_holes_rules():
    lab = np.full((7, 3, 3), -1, np.int8)
    lab[0, 1, 1], lab[2, 1, 1] = 4, 4         # gap of 1 between two 4s -> filled
    lab[4, 1, 1] = 2                          # [3] lies between 4 and 2: one vote each, tie -> 2
    out = fill_small_holes(lab, 1)
    assert out[1, 1, 1] == 4 and out[3, 1, 1] == 2
    assert out[5, 1, 1] == -1 and out[6, 1, 1] == -1           # only one observed side: never filled
    assert (out[:, 0, :] == -1).all()                           # nothing grows sideways into empty space
    assert np.array_equal(out[lab >= 0], lab[lab >= 0])         # observed voxels unchanged
    gap2 = np.full((5, 1, 1), -1, np.int8)
    gap2[0], gap2[3] = 1, 1
    assert (fill_small_holes(gap2, 1)[1:3] == -1).all()         # a 2-voxel gap needs max_gap 2
    assert (fill_small_holes(gap2, 2)[1:3] == 1).all()


def test_fill_on_coarse_sweep_stays_inside(block_sweep, gt):
    """1 mm spacing (every 4th frame) leaves every other 0.5 mm layer along y empty; filling closes them only
    between observed layers and never outside the observed bounding box."""
    comp = LabelCompounder.from_sweep(block_sweep, VoxelGrid(GridConfig()))
    comp.insert_batch(block_sweep.frames[::4])
    lab = comp.result()
    obs = lab >= 0
    filled = fill_small_holes(lab, 1)
    new = (filled >= 0) & ~obs
    lo = [ax.min() for ax in np.nonzero(obs)]
    hi = [ax.max() for ax in np.nonzero(obs)]
    nz = np.nonzero(new)
    assert new.sum() > 0.5 * obs.sum()
    assert all(a.min() >= l and a.max() <= h for a, l, h in zip(nz, lo, hi))
    assert np.array_equal(filled[obs], lab[obs])
    acc_new = (filled[new] == gt[new]).mean()
    print(f"\n[S2] 1 mm spacing: observed {100 * obs.mean():.1f}% -> {100 * (filled >= 0).mean():.1f}% after "
          f"fill_small_holes(1); filled voxels {100 * acc_new:.2f}% correct")
    assert acc_new > 0.95


# ---------------------------------------------------------------- script
def test_reconstruct_sweep_script(block_sweep, tmp_path):
    short = Sweep(block_sweep.metadata, block_sweep.frames[:30])
    src = short.save(str(tmp_path / "sweeps" / "short.npz"))
    out = tmp_path / "results"
    assert reconstruct_main([src, "--out", str(out), "--fill"]) == 0
    files = sorted(os.listdir(out))
    assert files == ["recon_short.npz", "recon_short_xy.png", "recon_short_xz.png", "recon_short_yz.png"]
    with np.load(out / "recon_short.npz") as z:
        assert z["labels"].shape == (200, 200, 100) and z["labels"].dtype == np.int8
        assert z["hits"].shape == (200, 200, 100) and z["intensity"].size == 0
        assert VoxelGrid.from_dict(json.loads(str(z["grid"]))).shape == (200, 200, 100)
        meta = json.loads(str(z["metadata"]))
    assert meta["n_frames"] == 30 and meta["fill"] is True and meta["accuracy_observed"] > 0.95
