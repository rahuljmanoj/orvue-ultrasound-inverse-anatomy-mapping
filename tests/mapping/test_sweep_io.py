"""mapping/sweep_io.py and mapping/config.py: save / load round trip, metadata, size estimate (S0).

The sweep is 50 real frames along a short lane of the normal case (rendered once per module, noise seeded),
because labels compress far better than speckle: random arrays would calibrate the size estimate wrong.
"""
import dataclasses
import os

import numpy as np
import pytest

from orvue_us_inverse.mapping.config import AcquisitionConfig, GridConfig, SweepConfig
from orvue_us_inverse.mapping.probe import FRAME_SHAPE, PROBE, make_simulator, probe_metadata
from orvue_us_inverse.mapping.sweep_io import (SIMULATOR_SOURCE_COMMIT, FrameRecord, Sweep, estimate_size_mb,
                                               make_metadata)
from orvue_us_inverse.paths import UPSTREAM_PATH
from orvue_us_inverse.simulation.bmode import BModeSimulator

N_FRAMES = 50
SPACING_MM = 0.5
SPEED_MM_S = 10.0


@pytest.fixture(scope="module")
def lane():
    """50 frames at 0.5 mm along y through the GB / hilum (x = 50, y 30 -> 54.5, yaw 0)."""
    sim = make_simulator("normal")
    sim._rng = np.random.default_rng(0)
    frames = []
    for k in range(N_FRAMES):
        T = BModeSimulator.pose_from_xy_yaw(50.0, 30.0 + SPACING_MM * k, 0.0)
        T_meas = T.copy()
        T_meas[:3, 3] += (0.1, -0.05, 0.0)                   # distinct from T_true, to check both are kept
        img, lab = sim.render(T, return_labels=True)
        frames.append(FrameRecord(k, k * SPACING_MM / SPEED_MM_S, T, T_meas, lab, img))
    return sim, frames


def _sweep(sim, frames, store_images):
    acq = AcquisitionConfig(store_images=store_images)
    fr = frames if store_images else [dataclasses.replace(f, image=None) for f in frames]
    return Sweep(make_metadata(sim, SweepConfig(), acq, note="S0 test"), fr)


def _assert_same(a: Sweep, b: Sweep):
    assert a.metadata == b.metadata
    assert len(a) == len(b)
    for fa, fb in zip(a.frames, b.frames):
        assert (fa.index, fa.t) == (fb.index, fb.t)
        assert np.array_equal(fa.T_true, fb.T_true) and np.array_equal(fa.T_measured, fb.T_measured)
        assert np.array_equal(fa.labels, fb.labels) and fb.labels.dtype == np.int8
        if fa.image is None:
            assert fb.image is None
        else:
            assert np.array_equal(fa.image, fb.image) and fb.image.dtype == np.uint8


@pytest.mark.parametrize("store_images", [True, False])
def test_round_trip(lane, tmp_path, store_images):
    sweep = _sweep(*lane, store_images)
    path = sweep.save(str(tmp_path / "sub" / "lane"))
    assert path.endswith(".npz") and os.path.isfile(path)
    back = Sweep.load(path)
    assert back.has_images == store_images
    _assert_same(sweep, back)


def test_metadata(lane, tmp_path):
    sim, frames = lane
    meta = Sweep.load(_sweep(sim, frames, True).save(str(tmp_path / "m.npz"))).metadata
    assert meta["case"] == "normal"
    assert meta["probe"] == probe_metadata() and meta["probe"]["width_mm"] == PROBE.width_mm
    assert meta["frame_shape"] == list(FRAME_SHAPE) == [501, 301]
    assert meta["simulator"]["persistence"] == 0.0 and meta["simulator"]["compound_deg"] == [-7.0, 0.0, 7.0]
    assert SweepConfig(**meta["sweep_config"]) == SweepConfig()
    assert AcquisitionConfig(**meta["acquisition_config"]) == AcquisitionConfig()
    assert meta["note"] == "S0 test" and meta["created_utc"]
    prov = meta["provenance"]
    assert prov["simulator_source_commit"] == SIMULATOR_SOURCE_COMMIT
    assert prov["repo_commit"] is None or len(prov["repo_commit"]) == 40


def test_simulator_commit_matches_upstream():
    with open(UPSTREAM_PATH, encoding="utf-8") as f:
        assert f"`{SIMULATOR_SOURCE_COMMIT}`" in f.read()


@pytest.mark.parametrize("store_images", [True, False])
def test_size_estimate(lane, tmp_path, store_images):
    path = _sweep(*lane, store_images).save(str(tmp_path / "s.npz"))
    actual = os.path.getsize(path) / 1e6
    est = estimate_size_mb(N_FRAMES, store_images)
    assert abs(est - actual) <= 0.3 * actual, (est, actual)


def test_mixed_images_rejected(lane):
    sim, frames = lane
    sweep = Sweep({}, frames[:1])
    with pytest.raises(ValueError):
        sweep.append(dataclasses.replace(frames[1], image=None))


def test_frame_record_checks(lane):
    f = lane[1][0]
    with pytest.raises(ValueError):
        FrameRecord(0, 0.0, f.T_true, f.T_measured, f.labels.astype(np.int16))
    with pytest.raises(ValueError):
        FrameRecord(0, 0.0, f.T_true[:3], f.T_measured, f.labels)


def test_empty_sweep_round_trip(tmp_path):
    back = Sweep.load(Sweep({"case": "normal"}).save(str(tmp_path / "e.npz")))
    assert len(back) == 0 and back.metadata == {"case": "normal"}


def test_grid_config():
    g = GridConfig()
    assert g.shape == (200, 200, 100) and g.n_voxels == 4_000_000
    xc, yc, zc = g.centres()
    assert (xc[0], xc[-1], zc[0], zc[-1]) == (0.25, 99.75, 0.25, 49.75)
    assert len(yc) == 200
    assert GridConfig(voxel_mm=0.25).shape == (400, 400, 200)
    assert GridConfig(**dataclasses.asdict(g)) == g
    with pytest.raises(ValueError):
        GridConfig(voxel_mm=0.3)


def test_sweep_config():
    s = SweepConfig()
    assert (s.yaw_list_deg, s.overlap_pct, s.frame_spacing_mm, s.angle_trigger_deg, s.speed_mm_s, s.serpentine) \
        == ([0.0], 20.0, 0.5, 1.0, 10.0, True)
    assert (s.region_x_mm, s.region_y_mm) == ((0.0, 100.0), (0.0, 100.0))
    assert (s.probe_width_mm, s.probe_depth_mm) == (30.0, 50.0)
    assert AcquisitionConfig().store_images is True
    with pytest.raises(ValueError):
        SweepConfig(overlap_pct=100.0)
