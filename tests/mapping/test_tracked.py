"""mapping/poses.TrackedPose and mapping/run_tracked (S7) with a fake tracker (no camera): recording with the space
key, tracking loss, camera / mouse switch, live error injection, status line, report fields."""
import os

import cv2
import numpy as np
import pytest

from orvue_us_inverse.mapping.config import AcquisitionConfig, SweepConfig
from orvue_us_inverse.mapping.errors import PoseErrorModel
from orvue_us_inverse.mapping.poses import TrackedPose
from orvue_us_inverse.mapping.run_tracked import TrackedSession, parse_args
from orvue_us_inverse.mapping.sweep_io import Sweep
from orvue_us_inverse.simulation.bmode import BModeSimulator
from orvue_us_inverse.tracking.tracker import TrackState

MOVE, DOWN, UP = cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeTracker:
    """get_state() like ProbeTracker: a valid filtered pose at (x, y, yaw) unless valid is False."""

    def __init__(self):
        self.x, self.y, self.yaw, self.valid = 57.0, 20.0, 0.0, True

    def get_state(self):
        s = TrackState(valid=self.valid, probe_valid=self.valid, phantom_valid=True, n_reference_markers_used=4,
                       reproj_error_ref_px=0.3, reproj_error_probe_px=0.4, reproj_error_upright_px=0.5, fps=30.0)
        if self.valid:
            T = BModeSimulator.pose_from_xy_yaw(self.x, self.y, self.yaw).astype(np.float64)
            s.T_phantom_probe = s.T_phantom_probe_raw = T
        return s


class Keys:
    space = False

    def __call__(self):
        return self.space


@pytest.fixture()
def rig():
    clock, tracker, keys = FakeClock(), FakeTracker(), Keys()
    s = TrackedSession("normal", SweepConfig(), AcquisitionConfig(store_images=False), tracker=tracker,
                       key_state=keys, clock=clock)
    return s, clock, tracker, keys


def walk(s, clock, tracker, ys, dt=0.025):
    """Move the tracked probe along y (one step per call), stepping the session each time."""
    n = 0
    for y in ys:
        clock.t += dt
        tracker.y = y
        n += s.step() is not None
    return n


def test_tracked_pose(rig):
    s, clock, tracker, keys = rig
    tp = TrackedPose(tracker, clock=clock)
    assert tp.update() and tp.sample() is not None and tp.x == 57.0 and tp.yaw == 0.0
    tp.rotate(1)
    assert tp.yaw == 0.0                                     # the camera sets the angle
    tracker.valid = False
    assert not tp.update() and tp.sample() is None


def test_space_records_and_tracking_loss(rig):
    s, clock, tracker, keys = rig
    assert s.source == "camera"
    assert walk(s, clock, tracker, np.arange(20, 22, 0.25)) == 0          # space not held: nothing recorded
    keys.space = True
    n1 = walk(s, clock, tracker, np.arange(22, 27.01, 0.25))
    assert n1 == 21 and len(s.strokes) == 1
    tracker.valid = False                                    # tracking lost mid-stroke: no capture
    assert walk(s, clock, tracker, np.arange(27.25, 29, 0.25)) == 0
    tracker.valid = True
    n2 = walk(s, clock, tracker, np.arange(29, 31.01, 0.25))
    assert n2 == 9 and len(s.strokes) == 1                   # the stroke continues, restarting at the next pose
    keys.space = False
    s.step()
    assert not s.pose.recording and s.frames[-1].T_true[1, 3] == pytest.approx(31.0)
    assert all(np.array_equal(f.T_true, f.T_measured) for f in s.frames)   # no error injected


def test_camera_mouse_switch(rig):
    s, clock, tracker, keys = rig
    s.route_mouse(DOWN, *s.window_px(50, 50))                # camera mode: a click on the sweep does nothing
    assert not s.pose.recording and s.strokes == []
    s.turn(1)
    assert s.pose.yaw == 0.0 and "camera sets" in s.message
    assert s.toggle_source() and s.source == "mouse" and s.pose is s.mouse_pose
    s.route_mouse(DOWN, *s.window_px(40, 40))                # mouse mode: S6 behaviour
    for y in np.arange(40, 42.01, 0.25):
        clock.t += 0.05
        s.route_mouse(MOVE, *s.window_px(40, y))
        s.step()
    assert len(s.frames) == 9
    assert not s.toggle_source()                             # not while recording
    s.route_mouse(UP, *s.window_px(40, 42))
    assert s.toggle_source() and s.source == "camera"
    nocam = TrackedSession("normal", SweepConfig(), AcquisitionConfig(store_images=False), tracker=None,
                           key_state=Keys(), clock=clock)
    assert nocam.source == "mouse" and not nocam.set_source("camera") and "not available" in nocam.message


def test_injected_error_live(rig):
    s, clock, tracker, keys = rig
    s.error_model = PoseErrorModel(latency_ms=100.0)
    keys.space = True
    walk(s, clock, tracker, np.arange(20, 30.01, 0.25))      # 0.25 mm per 25 ms = 10 mm/s
    f = s.frames[-1]
    assert f.T_true[1, 3] - f.T_measured[1, 3] == pytest.approx(1.0, abs=0.02)   # 100 ms x 10 mm/s behind
    s.restart()
    s.error_model = PoseErrorModel(jitter_mm=0.5, seed=2)
    walk(s, clock, tracker, np.arange(30, 33.01, 0.25))
    d = np.array([f.T_measured[:3, 3] - f.T_true[:3, 3] for f in s.frames])
    assert np.abs(d).max() > 0.05 and np.abs(d).max() < 3.0
    # the reconstruction uses T_measured: undo removes exactly what was inserted
    keys.space = False
    s.step()
    assert s.undo() and not s.comp.hits.any()


def test_status_line_and_report(rig, tmp_path):
    s, clock, tracker, keys = rig
    s.error_model = PoseErrorModel(jitter_mm=0.5)
    s.step()
    items = [t for t, _ in s.tracking_items()]
    assert items[0].startswith("POSE: CAMERA") and "TRACKING: OK" in items and "reference 4/4" in items
    assert any(t.startswith("reproj ref / probe / upright 0.30 / 0.40 / 0.50") for t in items)
    assert any(t.startswith("filter lag") for t in items) and any(t.startswith("inject: jitter_mm=0.5") for t in items)
    tracker.valid = False
    s.step()
    assert "TRACKING: LOST" in [t for t, _ in s.tracking_items()]
    img = s.compose()
    assert img.shape == (TrackedSession.HEIGHT, TrackedSession.WIDTH, 3)
    tracker.valid = True
    keys.space = True
    walk(s, clock, tracker, np.arange(20, 40.01, 0.25))
    keys.space = False
    s.step()
    back = Sweep.load(s.save(str(tmp_path)))
    assert back.metadata["mode"] == "tracked" and back.metadata["pose_source"] == "camera"
    assert back.metadata["injected_error"]["jitter_mm"] == 0.5
    folder, table = s.complete(results=str(tmp_path / "results"))
    assert os.path.basename(folder).startswith("normal_tracked_") and s.evaluation.report["pose_source"] == "camera"


def test_cli():
    a = parse_args(["--inject", "jitter=0.5,yaw=0.5,latency=50", "--mouse", "--source", "clip.mp4"])
    assert a.inject == "jitter=0.5,yaw=0.5,latency=50" and a.mouse and a.source == "clip.mp4"
    assert PoseErrorModel.parse(a.inject).latency_ms == 50.0
    from orvue_us_inverse.__main__ import COMMANDS, MENU
    assert COMMANDS["tracked"][0] == ["-m", "orvue_us_inverse.mapping.run_tracked"]
    assert any(m[4] == "tracked" for m in MENU)


def test_real_tracker_with_committed_calibration_captures():
    """Regression (S7 review): with the committed probe calibration the tracked face comes out a few mm off the surface;
    the tracked pose must still capture (x, y, yaw used upright at z = 0, as the simulator's tracked mode does)."""
    from tests.helpers import synthetic_scene as ss
    from orvue_us_inverse.paths import CALIBRATION_PATH
    from orvue_us_inverse.tracking.tracker import ProbeTracker, TrackerConfig
    tr = ProbeTracker(source=None, config=TrackerConfig(calibration_path=CALIBRATION_PATH), intrinsics=(ss.K, ss.DIST))
    clock, keys = FakeClock(), Keys()
    s = TrackedSession("normal", SweepConfig(), AcquisitionConfig(store_images=True), tracker=tr, key_state=keys,
                       focus=lambda: False, clock=clock)
    Tc = ss.camera_pose(300.0)
    keys.space = True
    for k in range(12):                                      # 0.25 mm per frame along y
        clock.t = k / 30.0
        tr.process_frame(ss.render(Tc, ss.pose_from_xy_yaw(50.0, 40.0 + 0.25 * k, 0.0), seed=k), clock.t)
        if k == 0:
            s.space_event()                                  # OpenCV delivered the key: our window has the focus
        s.step()
    tp = s.tracked_pose
    assert tp.valid and abs(tp.face_z_mm) > 1.0              # the calibration puts the tracked face off the surface ...
    assert len(s.frames) >= 5                                # ... and frames are captured anyway (fewer than 12:
                                                             # the filtered pose lags at the start of the motion)
    f = s.frames[-1]
    assert f.T_true[2, 3] == 0.0 and f.image is not None     # upright on the surface, B-mode rendered
    assert any(t.startswith("face z") for t, _ in s.tracking_items())
    assert any(t == "REC" for t, _ in s.tracking_items())


def test_space_needs_focus_or_recent_key_event(rig):
    s, clock, tracker, keys = rig
    s.focus = lambda: False                                  # another window has the focus ...
    keys.space = True
    walk(s, clock, tracker, np.arange(20, 22, 0.25))
    assert len(s.frames) == 0 and not s.pose.recording       # ... so the held space bar is not ours
    s.space_event()                                          # a space event reached our window
    walk(s, clock, tracker, np.arange(22, 24.01, 0.25))
    assert len(s.frames) > 0
