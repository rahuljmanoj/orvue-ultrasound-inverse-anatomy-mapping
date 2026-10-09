"""mapping/errors.py (S7): zero error, latency, bias, jitter, parsing, live latency buffer, and the tracker's filter lag
(analytic filter run and end to end on synthetic camera frames)."""
import math

import numpy as np
import pytest

from orvue_us_inverse.mapping.acquisition import Acquirer
from orvue_us_inverse.mapping.config import AcquisitionConfig, GridConfig, SweepConfig
from orvue_us_inverse.mapping.errors import LatencyBuffer, PoseErrorModel, filter_lag, interpolate_pose
from orvue_us_inverse.mapping.poses import ScriptedSweep
from orvue_us_inverse.mapping.probe import make_simulator
from orvue_us_inverse.mapping.recon import LabelCompounder, VoxelGrid
from orvue_us_inverse.simulation.bmode import BModeSimulator

GRID = VoxelGrid(GridConfig())


@pytest.fixture(scope="module")
def block():
    """Oracle-label lane at 0.5 mm over the hilum (x 35-65, y 30-70 mm) and its plan."""
    cfg = SweepConfig(frame_spacing_mm=0.5, region_x_mm=(35, 65), region_y_mm=(30, 70))
    plan = ScriptedSweep(cfg)
    acq = Acquirer(make_simulator("normal"), cfg, AcquisitionConfig(store_images=False))
    for s in plan.samples():
        acq.feed(s.T, s.t, s.recording)
    return plan, acq.sweep.frames


def _recon(frames):
    c = LabelCompounder(GRID)
    c.insert_batch(frames)
    return c


def test_zero_error_is_identity(block):
    plan, frames = block
    m = PoseErrorModel()
    assert m.is_zero and m.describe() == "no error"
    T = frames[5].T_true
    assert np.array_equal(m.apply(T, 1.0, plan.pose_at), T)
    out = m.apply_frames(frames, pose_at=lambda t: plan.pose_at(t).T)
    a, b = _recon(frames), _recon(out)
    assert np.array_equal(a.votes, b.votes) and np.array_equal(a.hits, b.hits)


def test_latency_shifts_by_speed_times_latency(block):
    plan, frames = block
    v = plan.cfg.speed_mm_s                                  # 10 mm/s
    m = PoseErrorModel(latency_ms=100.0)
    f = frames[40]                                           # inside the lane, > 1 mm from its start
    Tm = m.apply(f.T_true, f.t, lambda t: plan.pose_at(t).T)
    d = Tm[:3, 3] - f.T_true[:3, 3]
    assert np.linalg.norm(d) == pytest.approx(v * 0.1, abs=1e-6)          # 1.0 mm
    direction = frames[41].T_true[:3, 3] - frames[39].T_true[:3, 3]
    assert np.dot(d, direction) < 0                                       # behind the probe
    # without a planned path: interpolated between the frames themselves
    out = m.apply_frames(frames[30:50])
    assert np.linalg.norm(out[10].T_measured[:3, 3] - out[10].T_true[:3, 3]) == pytest.approx(1.0, abs=1e-6)


def test_bias_shifts_reconstruction(block):
    plan, frames = block
    a = _recon(frames).result()
    b = _recon(PoseErrorModel(bias_x_mm=1.0).apply_frames(frames)).result()
    shifts = []
    for lab in (3, 9):                                       # bile, venous blood
        ca = np.argwhere(a == lab).mean(axis=0) * GRID.voxel_mm
        cb = np.argwhere(b == lab).mean(axis=0) * GRID.voxel_mm
        shifts.append(cb - ca)
    for s in shifts:
        assert s[0] == pytest.approx(1.0, abs=GRID.voxel_mm)                 # x: the bias, within one voxel
        assert abs(s[1]) <= GRID.voxel_mm and abs(s[2]) <= GRID.voxel_mm


def test_bias_yaw_scale_and_jitter():
    T = BModeSimulator.pose_from_xy_yaw(60.0, 40.0, 0.0).astype(np.float64)
    m = PoseErrorModel(bias_yaw_deg=10.0)
    assert math.degrees(math.atan2(m.apply(T)[1, 0], m.apply(T)[0, 0])) == pytest.approx(10.0)
    assert np.allclose(m.apply(T)[:3, 3], T[:3, 3])
    s = PoseErrorModel(scale_pct=10.0).apply(T)
    assert np.allclose(s[:3, 3], [61.0, 39.0, 0.0])                       # about (50, 50, 0)
    j = PoseErrorModel(jitter_mm=0.5, jitter_yaw_deg=1.0, seed=3)
    p = np.array([j.apply(T)[:3, 3] for _ in range(4000)])
    assert np.allclose(p.std(axis=0), 0.5, rtol=0.06) and np.allclose(p.mean(axis=0), T[:3, 3], atol=0.03)
    j.reset()
    first = j.apply(T)
    j.reset()
    assert np.array_equal(first, j.apply(T))                                # seeded: reproducible
    tilt = PoseErrorModel(jitter_tilt_deg=2.0, seed=1).apply(T)
    assert 0 < math.degrees(math.acos(np.clip(tilt[2, 2], -1, 1))) < 15


def test_parse():
    m = PoseErrorModel.parse("jitter=0.5,yaw=0.5,latency=50")
    assert (m.jitter_mm, m.jitter_yaw_deg, m.latency_ms) == (0.5, 0.5, 50.0)
    assert PoseErrorModel.parse("") == PoseErrorModel()
    assert PoseErrorModel.parse("bias=1 biasyaw=2 scale=1.5 tilt=1 seed=7").bias_x_mm == 1.0
    with pytest.raises(ValueError):
        PoseErrorModel.parse("wobble=3")


def test_latency_buffer():
    buf = LatencyBuffer()
    for k in range(11):                                      # 10 mm/s along y, 0.1 s apart
        buf.add(0.1 * k, BModeSimulator.pose_from_xy_yaw(50.0, 20.0 + k, 0.0))
    assert buf.pose_at(0.55)[1, 3] == pytest.approx(25.5)
    assert buf.pose_at(-1.0)[1, 3] == 20.0 and buf.pose_at(5.0)[1, 3] == 30.0
    T = interpolate_pose(BModeSimulator.pose_from_xy_yaw(0, 0, 350), BModeSimulator.pose_from_xy_yaw(2, 0, 10), 0.5)
    assert math.degrees(math.atan2(T[1, 0], T[0, 0])) == pytest.approx(0.0, abs=1e-6) and T[0, 3] == 1.0


def test_filter_lag_analytic_and_end_to_end():
    """The one-euro filter's lag at 10 mm/s from errors.filter_lag, and the same lag measured on the tracker with
    synthetic camera frames of a probe moving at 10 mm/s (30 frames/s, no calibration offsets)."""
    from tests.helpers import synthetic_scene as ss
    from orvue_us_inverse.tracking.tracker import ProbeTracker, TrackerConfig
    lag = filter_lag(10.0)
    assert 55 < lag["lag_ms"] < 75 and lag["lag_mm"] == pytest.approx(lag["lag_ms"] / 100.0)
    assert filter_lag(5.0)["lag_ms"] > lag["lag_ms"] > filter_lag(20.0)["lag_ms"]     # less lag at higher speed
    tr = ProbeTracker(source=None, config=TrackerConfig(calibration_path=None), intrinsics=(ss.K, ss.DIST))
    Tc = ss.camera_pose(300.0)
    diffs = []
    for k in range(75):                                      # 2.5 s at 30 frames/s, 10 mm/s along y
        t = k / 30.0
        st = tr.process_frame(ss.render(Tc, ss.pose_from_xy_yaw(50.0, 30.0 + 10.0 * t, 0.0), seed=k), t)
        assert st.valid
        if k >= 45:                                          # settled
            diffs.append(st.face_raw[1] - st.face_filt[1])
    measured_mm = float(np.mean(diffs))
    assert measured_mm == pytest.approx(lag["lag_mm"], abs=0.15)
