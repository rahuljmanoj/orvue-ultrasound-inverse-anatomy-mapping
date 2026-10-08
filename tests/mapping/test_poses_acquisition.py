"""mapping/poses.py and mapping/acquisition.py: lane layout, distance-triggered capture, sweep files (S1).

Full sweeps are checked with a trigger-only acquirer (the capture returns a shared empty label image), so
every pose of an 8-lane sweep is tested in well under a second; real rendering is tested on shorter stretches.
"""
import math
from collections import Counter

import numpy as np
import pytest

from orvue_us_inverse.mapping.acquisition import Acquirer
from orvue_us_inverse.mapping.config import AcquisitionConfig, SweepConfig
from orvue_us_inverse.mapping.poses import ScriptedSweep
from orvue_us_inverse.mapping.probe import FRAME_SHAPE, make_simulator
from orvue_us_inverse.mapping.sweep_io import Sweep
from orvue_us_inverse.simulation.bmode import BModeSimulator

W = 30.0
# requested overlap -> (lanes, centres, actual overlap %), PLAN_inverse_mapping.md 2.3
EXAMPLES = {0: (4, [15, 38.333, 61.667, 85], 22.2), 20: (4, [15, 38.333, 61.667, 85], 22.2),
            40: (5, [15, 32.5, 50, 67.5, 85], 41.7), 50: (6, [15, 29, 43, 57, 71, 85], 53.3)}
EMPTY = np.zeros(FRAME_SHAPE, np.int8)


class TriggerOnly(Acquirer):
    """Acquirer whose capture does no rendering: tests the trigger logic over whole sweeps quickly."""

    def _capture(self, T):
        return None, EMPTY


@pytest.fixture(scope="module")
def sim():
    return make_simulator("normal")


def run(acq, sweep, until_t=math.inf):
    fed = Counter()
    for s in sweep.samples():
        if s.t > until_t:
            break
        acq.feed(s.T, s.t, s.recording)
        fed[s.recording] += 1
    return fed


def lateral(yaw, T):
    """Lane coordinate of a pose: x for yaw 0, y for yaw 90."""
    return T[0, 3] if yaw == 0 else T[1, 3]


# ---------------------------------------------------------------- lane layout
@pytest.mark.parametrize("yaw", [0.0, 90.0])
@pytest.mark.parametrize("overlap", [0, 20, 40, 50])
def test_lane_layout_and_union(yaw, overlap):
    sw = ScriptedSweep(SweepConfig(yaw_list_deg=[yaw], overlap_pct=overlap))
    n, centres, actual = EXAMPLES[overlap]
    lanes = sw.planned_lanes()
    assert len(lanes) == n
    pos = sorted(lateral(yaw, BModeSimulator.pose_from_xy_yaw(*ln.start, yaw)) for ln in lanes)
    assert np.allclose(pos, centres, atol=1e-3)
    assert sw.overlap_pct[yaw] == pytest.approx(actual, abs=0.05) and sw.overlap_pct[yaw] >= overlap
    # image footprints [c - W/2, c + W/2] cover 0..100 with no gap
    assert pos[0] - W / 2 <= 1e-6 and pos[-1] + W / 2 >= 100 - 1e-6
    assert all(b - a <= W + 1e-6 for a, b in zip(pos, pos[1:]))
    # every lane runs across the full region along the sweep direction
    assert all(ln.length_mm == pytest.approx(100.0, abs=1e-6) for ln in lanes)
    assert sw.coverage()["all"] == 1.0


def test_serpentine_and_transitions():
    sw = ScriptedSweep(SweepConfig(yaw_list_deg=[0, 90]))
    lanes = sw.planned_lanes()
    assert [ln.yaw_deg for ln in lanes] == [0.0] * 4 + [90.0] * 4
    for a, b in zip(lanes, lanes[1:]):
        if a.yaw_deg == b.yaw_deg:          # serpentine: next lane starts at the end where the last one stopped
            assert np.dot(np.subtract(a.end, a.start), np.subtract(b.end, b.start)) < 0
    kinds = [s.recording for s in sw.segments]
    assert kinds == [True, False] * 7 + [True]        # lanes joined by lift-off transitions
    assert sw.expected_frames() == 8 * 401                      # default spacing 0.25 mm


def test_oblique_yaw_clipped_to_contact(sim):
    sw = ScriptedSweep(SweepConfig(yaw_list_deg=[45]))
    assert len(sw.lanes) == 6
    for ln in sw.lanes:
        for p in (ln.start, ln.end):
            assert sim.in_contact(BModeSimulator.pose_from_xy_yaw(*p, 45))
    cov = sw.coverage()[45.0]
    assert 0.9 < cov < 1.0             # the region corners outside the clipped lanes stay uncovered


def test_pose_at_and_speed():
    cfg = SweepConfig(speed_mm_s=10.0)
    sw = ScriptedSweep(cfg)
    first, last = sw.lanes[0], sw.lanes[-1]
    assert np.allclose(sw.pose_at(0.0).T[:2, 3], first.start)
    assert np.allclose(sw.pose_at(sw.duration_s).T[:2, 3], last.end, atol=1e-4)
    s = sw.pose_at(2.0)
    assert s.recording and s.lane is first and np.allclose(s.T[:2, 3], (15.0, 20.0), atol=1e-4)
    mid = (sw.segments[1].t0 + sw.segments[1].t1) / 2
    assert not sw.pose_at(mid).recording
    ts = [x.t for x in sw.samples()]
    assert all(b >= a for a, b in zip(ts, ts[1:]))
    assert sw.segments[0].t1 == pytest.approx(100.0 / 10.0)


# ---------------------------------------------------------------- acquisition (trigger only, full sweeps)
@pytest.mark.parametrize("yaws", [[0.0, 90.0], [45.0]])
def test_full_sweep_triggers(sim, yaws):
    cfg = SweepConfig(yaw_list_deg=yaws)
    sw = ScriptedSweep(cfg)
    acq = TriggerOnly(sim, cfg, AcquisitionConfig(store_images=False))
    fed = run(acq, sw)
    frames = acq.sweep.frames
    assert fed[False] > 0                                    # transitions were fed ...
    assert len(frames) == sw.expected_frames()
    assert acq.n_not_in_contact == 0
    per_lane = Counter()
    for f in frames:
        assert sim.in_contact(f.T_true)
        lane = sw.lane_at(f.t)
        assert lane is not None                               # ... but nothing was captured during them
        # the frame lies on its lane's line
        a, b, p = np.array(lane.start), np.array(lane.end), f.T_true[:2, 3]
        d = b - a
        assert abs(d[0] * (p - a)[1] - d[1] * (p - a)[0]) / np.linalg.norm(d) < 1e-3
        per_lane[lane.index] += 1
    for lane in sw.lanes:
        assert per_lane[lane.index] == math.floor(lane.length_mm / cfg.frame_spacing_mm + 1e-6) + 1
    if yaws == [0.0, 90.0]:
        assert all(per_lane[ln.index] == 401 for ln in sw.lanes)     # 100 mm / 0.25 mm + the start frame
    # spacing within each lane: frame_spacing_mm +/- 10 %
    for (fa, fb) in zip(frames, frames[1:]):
        if sw.lane_at(fa.t) is sw.lane_at(fb.t):
            gap = np.linalg.norm(fb.T_true[:3, 3] - fa.T_true[:3, 3])
            assert gap == pytest.approx(cfg.frame_spacing_mm, rel=0.10)
    assert [f.index for f in frames] == list(range(len(frames)))
    assert all(np.array_equal(f.T_true, f.T_measured) for f in frames)


def test_rotation_trigger_and_contact(sim):
    cfg = SweepConfig(angle_trigger_deg=1.0)
    acq = TriggerOnly(sim, cfg)
    for k in range(21):                                       # rotate on the spot in 0.25 degree steps
        acq.feed(BModeSimulator.pose_from_xy_yaw(50, 50, 0.25 * k), 0.1 * k, True)
    assert len(acq.sweep) == 6                                # 0, 1, 2, 3, 4, 5 degrees
    assert acq.feed(BModeSimulator.pose_from_xy_yaw(-1.0, 50, 0), 3.0, True) is None
    assert acq.n_not_in_contact == 1
    assert acq.feed(BModeSimulator.pose_from_xy_yaw(80, 80, 0), 3.1, False) is None   # not recording


# ---------------------------------------------------------------- real capture
def test_real_capture_lane_and_transition(sim):
    """Oracle labels only, 0.5 mm spacing: lane 1 in full, the lift-off, then 5 mm of lane 2."""
    cfg = SweepConfig(frame_spacing_mm=0.5)
    sw = ScriptedSweep(cfg)
    acq = Acquirer(sim, cfg, AcquisitionConfig(store_images=False))
    run(acq, sw, until_t=sw.segments[2].t0 + 5.0 / cfg.speed_mm_s + 1e-9)
    counts = Counter(sw.lane_at(f.t).index for f in acq.sweep.frames)
    assert counts == {0: 201, 1: 11}
    f = acq.sweep.frames[57]
    assert f.image is None and f.labels.shape == (501, 301)
    assert np.array_equal(f.labels, sim.labels_image(f.T_true.astype(np.float32)))


def test_short_sweep_save_load(sim, tmp_path):
    cfg = SweepConfig(yaw_list_deg=[0.0, 90.0], frame_spacing_mm=0.5)
    sw = ScriptedSweep(cfg)
    sim._rng = np.random.default_rng(0)
    acq = Acquirer(sim, cfg, AcquisitionConfig(store_images=True), note="S1 test")
    run(acq, sw, until_t=10.0 / cfg.speed_mm_s + 1e-9)       # first 10 mm: 21 frames with B-mode
    assert len(acq.sweep) == 21
    back = Sweep.load(acq.sweep.save(str(tmp_path / "short.npz")))
    assert back.metadata == acq.sweep.metadata
    assert SweepConfig(**back.metadata["sweep_config"]) == cfg
    assert back.has_images and len(back) == 21
    for a, b in zip(acq.sweep.frames, back.frames):
        assert np.array_equal(a.image, b.image) and np.array_equal(a.labels, b.labels)
        assert np.array_equal(a.T_true, b.T_true) and a.t == b.t
