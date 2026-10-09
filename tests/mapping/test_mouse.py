"""mapping/poses.MousePose and mapping/run_mouse (S6): synthetic mouse event sequences with a fake clock, no windows.

Oracle labels only (store_images False), so a captured frame costs ~13 ms of labels_image.
"""
import os

import cv2
import numpy as np
import pytest

from orvue_us_inverse.mapping.config import AcquisitionConfig, GridConfig, SweepConfig
from orvue_us_inverse.mapping.poses import MousePose
from orvue_us_inverse.mapping.recon import LabelCompounder
from orvue_us_inverse.mapping.run_mouse import (GAP_FACTOR, MouseSession, find_gaps,
                                                mouse_filename, parse_args, speed_class)
from orvue_us_inverse.mapping.sweep_io import Sweep

MOVE, DOWN, UP = cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float) -> None:
        self.t += dt


@pytest.fixture(scope="module")
def session_factory():
    cache = {}

    def make():
        clock = FakeClock()
        if "s" not in cache:
            cache["s"] = MouseSession("normal", SweepConfig(), AcquisitionConfig(store_images=False), clock=clock)
        s = cache["s"]
        s.clock = clock
        s.pose = MousePose(clock=clock)
        s.period_s, s._prev_step_t, s._prev_captured = None, None, False
        s.acq_cfg.store_images = False                     # every test starts with B-mode off (labels only)
        s.restart()
        return s, clock
    return make


def drag(s: MouseSession, clock: FakeClock, path_mm, dt: float = 0.05, button: bool = True) -> int:
    """Move along path_mm (list of (x, y)); press at the first point when button; step after every move."""
    n = 0
    x0, y0 = path_mm[0]
    s.on_mouse(MOVE, *s.window_px(x0, y0))
    if button:
        s.on_mouse(DOWN, *s.window_px(x0, y0))
    for x, y in path_mm:
        clock.tick(dt)
        s.on_mouse(MOVE, *s.window_px(x, y))
        n += s.step() is not None
    if button:
        s.on_mouse(UP, *s.window_px(*path_mm[-1]))
    return n


def line(x, y0, y1, step):
    return [(x, y) for y in np.arange(y0, y1 + 1e-9, step)]


# ---------------------------------------------------------------- pose source
def test_mouse_pose_turning_and_speed():
    clock = FakeClock()
    p = MousePose(clock=clock)
    p.rotate(1)
    assert p.yaw == 5.0
    p.snap(90.0)
    assert p.yaw == 90.0
    p.press()
    p.rotate(-1)                                           # turning works during a stroke too
    assert p.recording and p.yaw == 85.0
    p.rotate(1)
    p.release()
    p.snap(-180.0)
    p.rotate(-1)
    assert p.yaw == 175.0                                  # kept in [-180, 180)
    p.snap(90.0)
    assert p.sample().recording is False and np.allclose(p.sample().T[:2, 0], (np.cos(np.radians(90)),
                                                                             np.sin(np.radians(90))))
    for k in range(10):                                   # 1 mm every 0.1 s -> 10 mm/s
        clock.tick(0.1)
        p.move(50.0 + k + 1, 50.0)
    assert p.speed_mm_s() == pytest.approx(10.0, rel=0.02)
    clock.tick(1.0)
    assert p.speed_mm_s() == 0.0                          # standing still


def test_speed_classification():
    assert speed_class(3.9, 5.0) == "green"
    assert speed_class(4.0, 5.0) == "amber" and speed_class(5.0, 5.0) == "amber"
    assert speed_class(5.01, 5.0) == "red"


# ---------------------------------------------------------------- session
def test_recording_gated_by_button(session_factory):
    s, clock = session_factory()
    assert drag(s, clock, line(50, 30, 35, 0.25), button=False) == 0 and len(s.frames) == 0
    n = drag(s, clock, line(50, 30, 35, 0.25))
    assert n == len(s.frames) == 21 and len(s.strokes) == 1 and s.strokes[0] == [0, 21]
    assert s.comp.n_frames == 21 and s.comp.hits.any()
    assert drag(s, clock, line(60, 30, 35, 0.25), button=False) == 0 and len(s.frames) == 21
    s.on_mouse(DOWN, *s.window_px(70, 30))
    s.on_mouse(UP, *s.window_px(70, 30))                  # click without a capture: no empty stroke kept
    assert len(s.strokes) == 1
    assert drag(s, clock, line(-5, 30, 35, 0.25)) == 0  # recording outside the region: not in contact


def test_capture_rate_and_speed_meter(session_factory):
    s, clock = session_factory()
    assert s.max_speed == pytest.approx(0.25 / 0.04)       # default before measuring (labels: 25 frames/s)
    drag(s, clock, line(50, 30, 40, 0.25), dt=0.05)        # each capturing loop lasts 0.05 s -> 20 frames/s
    assert s.capture_rate == pytest.approx(20.0, rel=0.05)
    assert s.max_speed == pytest.approx(5.0, rel=0.05)
    v, cls = s.speed()                                     # 0.25 mm per 0.05 s = 5 mm/s: at the limit
    assert v == pytest.approx(5.0, rel=0.1) and cls in ("amber", "red")


def test_gap_detection(session_factory):
    s, clock = session_factory()
    path = line(50, 30, 32, 0.25) + [(50, 33.5)] + line(50, 33.75, 35, 0.25)    # a 1.5 mm jump (> 2 x 0.25)
    drag(s, clock, path)
    assert len(s.gaps) == 1
    i, j, d = s.gaps[0]
    assert j == i + 1 and d == pytest.approx(1.5, abs=1e-3) and d > GAP_FACTOR * s.cfg.frame_spacing_mm
    assert find_gaps(s.frames, [tuple(st) for st in s.strokes], s.cfg.frame_spacing_mm) == s.gaps
    drag(s, clock, line(70, 30, 32, 0.25))                 # a new stroke far away: no gap between strokes
    assert len(s.gaps) == 1


def test_undo_last_stroke(session_factory):
    s, clock = session_factory()
    drag(s, clock, line(45, 30, 40, 0.25))
    first = list(s.frames)
    ref = LabelCompounder(s.grid)
    ref.insert_batch(first)
    drag(s, clock, line(60, 30, 31, 0.25) + [(60, 33)] + line(60, 33.25, 40, 0.25))
    assert len(s.strokes) == 2 and len(s.gaps) == 1
    s.on_mouse(DOWN, *s.window_px(60, 40))
    assert not s.undo()                                    # not while recording
    s.on_mouse(UP, *s.window_px(60, 40))
    assert s.undo()
    assert s.frames == first and len(s.strokes) == 1 and s.gaps == []
    for name in ("votes", "hits"):
        assert np.array_equal(getattr(s.comp, name), getattr(ref, name)), name
    assert s.comp.n_frames == len(first) and s.latest is first[-1]
    n = drag(s, clock, line(45, 40.25, 41, 0.25))         # recording continues normally after an undo
    assert n == len(line(45, 40.25, 41, 0.25)) and len(s.strokes) == 2
    assert s.undo() and s.undo() and not s.undo()
    assert len(s.frames) == 0 and not s.comp.hits.any()


def test_coverage_holes_and_drawing(session_factory, tmp_path):
    """Small geometry at the default 0.25 mm spacing (steps of 1 mm would leave unscanned rows between frames,
    which the coverage map correctly reports as holes)."""
    s, clock = session_factory()
    for x in (40, 47.5, 55):                               # three overlapping lanes along y 30-40 at yaw 0
        drag(s, clock, line(x, 30, 40, 0.25))
    covered, holes, pct, n_holes = s.coverage()
    assert covered.shape == (200, 200) and 0 < pct < 100
    assert n_holes == 0                                     # lanes overlap: no enclosed hole
    s.restart()
    for x in (30, 70):                                     # lanes 40 mm apart (footprints leave x 45-55 open) ...
        s.pose.snap(0.0)
        drag(s, clock, line(x, 40, 60, 0.25))
    for y in (30, 70):                                     # ... closed by two rows at yaw 90 (y 45-55 open)
        s.pose.snap(90.0)
        drag(s, clock, [(xx, y) for xx in np.arange(40, 60.01, 0.25)])
    _, holes, _, n_holes = s.coverage()
    assert n_holes == 1 and holes[100, 100]                # centre (50, 50) mm is an enclosed hole
    img = s.compose()
    assert img.shape == (MouseSession.HEIGHT, MouseSession.WIDTH, 3)          # one window: sweep | B-mode | 3D
    assert s.bmode_image().shape == (501, 301, 3)
    assert s.view3d_image().shape == (480 - 66, 420, 3)                       # placeholder without a 3D view
    path = s.save(str(tmp_path))
    back = Sweep.load(path)
    assert len(back) == len(s.frames) and back.metadata["mode"] == "mouse" and len(back.metadata["strokes"]) == 4


def test_complete(session_factory, tmp_path):
    s, clock = session_factory()
    drag(s, clock, line(57, 20, 60, 0.25))                 # one stroke along the CBD
    folder, table = s.complete(results=str(tmp_path))
    assert s.completed and s.step() is None
    assert {"report.md", "report.json", "slices_errors.png"} <= set(os.listdir(folder))
    assert s.evaluation.report["mode"] == "mouse" and s.evaluation.report["strokes"] == 1
    st = {r["name"]: r["status"] for r in s.evaluation.report["structures"]}
    assert st["chd_cbd"] == "detected" and st["gallstone_1"] == "not covered"
    assert "| Class |" in table


def test_3d_outputs_and_live_policy(session_factory, tmp_path):
    s, clock = session_factory()
    due, state = s.live3d_due(None)
    assert due                                             # first update
    s.on_mouse(DOWN, *s.window_px(57, 20))
    for y in np.arange(20, 30.01, 0.25):
        clock.tick(0.05)
        s.on_mouse(MOVE, *s.window_px(57, y))
        s.step()
        assert not s.live3d_due(state)[0]                  # never while the button is held
    s.on_mouse(UP, *s.window_px(57, 30))
    due, state = s.live3d_due(state)
    assert due and not s.live3d_due(state)[0]              # once after the stroke, then not again
    s.undo()
    assert s.live3d_due(state)[0]                          # after an undo
    drag(s, clock, line(57, 20, 40, 0.25))                 # along the CBD
    png, stls = s.export_3d(results=str(tmp_path / "results"), export=str(tmp_path / "export"))
    assert os.path.isfile(png) and any(p.endswith("recon_bile.stl") for p in stls)
    assert os.path.isfile(s.browser_view(folder=str(tmp_path / "viewer3d")))
    pytest.importorskip("pyvista")
    assert s.init_view3d()                                 # embedded (off-screen) 3D panel
    s.refresh_view3d()
    assert "rec_bile" in s.view3d._names
    before = s.view3d_image().copy()
    assert before.shape == (480 - 66, 420, 3) and before.any()
    x, y, w, h = MouseSession.RECT_3D
    s.route_mouse(DOWN, x + 100, y + 200)                  # drag in the 3D panel: rotates the view ...
    s.route_mouse(MOVE, x + 160, y + 220, cv2.EVENT_FLAG_LBUTTON)
    s.route_mouse(UP, x + 160, y + 220)
    assert not s.pose.recording and not np.array_equal(s.view3d_image(), before)
    assert len(s.strokes) == 1                              # ... and does not record a stroke
    assert s.view3d.view == "isometric" and "patient R" in s.view3d.shown_labels
    for name, (bx, by, bw, bh) in s.view_buttons():        # the view buttons set the presets
        s.route_mouse(DOWN, bx + bw // 2, by + bh // 2)
        s.route_mouse(UP, bx + bw // 2, by + bh // 2)
        assert s.view3d.view == name and s._orbit is None
    assert s.view3d.shown_labels == ["cranial", "caudal", "anterior (surface)", "posterior"]     # sagittal
    s.next_view()
    assert s.view3d.view == "isometric"
    assert len(s.strokes) == 1
    s.view3d.close()
    s.view3d = None


def test_mouse_routing_and_turning(session_factory):
    s, clock = session_factory()
    bx, by, bw, bh = MouseSession.RECT_BMODE
    s.route_mouse(DOWN, bx + 50, by + 50)                  # a click on the B-mode panel does not start a stroke
    assert not s.pose.recording and s.strokes == []
    s.route_mouse(UP, bx + 50, by + 50)
    s.route_mouse(DOWN, *s.window_px(50, 50))              # on the sweep: records
    assert s.pose.recording
    clock.tick(0.05)
    s.step()                                               # first frame of the stroke
    s.route_mouse(cv2.EVENT_MOUSEWHEEL, *s.window_px(50, 50), 120 << 16)   # wheel over the sweep turns the probe
    assert s.pose.yaw == 5.0
    clock.tick(0.05)
    assert s.step() is not None                            # turning in place captures a frame (angle trigger)
    s.set_angle(90.0)
    assert s.pose.yaw == 90.0
    s.route_mouse(cv2.EVENT_MOUSEWHEEL, *s.window_px(50, 50), -120 << 16)
    assert s.pose.yaw == 85.0
    s.route_mouse(MOVE, bx + 10, by + 10)                  # leaving the sweep while recording: probe off the region
    assert s.pose.x > 100
    s.route_mouse(UP, bx + 10, by + 10)                    # button-up anywhere ends the stroke
    assert not s.pose.recording


def test_bmode_switch_during_a_sweep(session_factory, tmp_path):
    """B-mode off -> on -> off during one stroke: only the frames taken with B-mode on have an image; the caption
    button and key i switch it; the capture-rate measurement restarts; the sweep saves and loads mixed."""
    s, clock = session_factory()
    assert not s.bmode_on and s.max_speed == pytest.approx(0.25 / 0.04)
    bx, by, bw, bh = s.bmode_button()
    s.on_mouse(MOVE, *s.window_px(57, 20))
    s.on_mouse(DOWN, *s.window_px(57, 20))
    for k, y in enumerate(np.arange(20, 26.01, 0.25)):
        if k == 8:
            s.on_mouse(UP, *s.window_px(57, y))
            s.route_mouse(DOWN, bx + bw // 2, by + bh // 2)    # click the caption button (button up: allowed)
            s.route_mouse(UP, bx + bw // 2, by + bh // 2)
            assert s.bmode_on and s.period_s is None
            assert s.max_speed == pytest.approx(0.25 / 0.12)    # default for B-mode until measured
            s.on_mouse(MOVE, *s.window_px(57, y))
            s.on_mouse(DOWN, *s.window_px(57, y))
        if k == 16:
            s.toggle_bmode()                                   # key i (allowed while recording)
        clock.tick(0.05)
        s.on_mouse(MOVE, *s.window_px(57, y))
        s.step()
    s.on_mouse(UP, *s.window_px(57, 26))
    has = [f.image is not None for f in s.frames]
    assert has == [False] * 8 + [True] * 8 + [False] * (len(has) - 16)
    assert s.acq.sweep.n_images == 8 and not s.showing_bmode()
    img = s.compose()                                          # labels view with the tissue key
    assert img.shape == (MouseSession.HEIGHT, MouseSession.WIDTH, 3)
    back = Sweep.load(s.save(str(tmp_path)))
    assert [f.image is not None for f in back.frames] == has and back.metadata["frames_with_bmode"] == 8
    inten = s.comp.intensity()
    assert inten is not None and np.isfinite(inten).sum() > 0


def test_wheel_delta():
    from orvue_us_inverse.mapping.run_mouse import wheel_delta
    assert wheel_delta(120 << 16) == 120 and wheel_delta((120 << 16) | 8) == 120
    assert wheel_delta(-120 << 16) == -120 and wheel_delta(0xFF880000) == -120


def test_hole_colour_is_not_in_the_heat_map():
    from orvue_us_inverse.mapping.run_mouse import HOLE_BGR
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[None, :], cv2.COLORMAP_INFERNO)[0].astype(int)
    assert np.abs(lut - np.array(HOLE_BGR)).sum(axis=1).min() > 150


def test_cli_and_filename():
    a = parse_args([])
    assert (a.yaw, a.overlap, a.spacing, a.no_images) == ([0.0], 20.0, 0.25, False)
    assert parse_args(["--no-images"]).no_images is True
    assert mouse_filename("normal", SweepConfig(), False, stamp="20261009-100000") == \
        "normal_mouse_sp0.25_labels_20261009-100000.npz"
    from orvue_us_inverse.__main__ import COMMANDS, MENU
    assert COMMANDS["mouse"][0] == ["-m", "orvue_us_inverse.mapping.run_mouse"]
    assert any(m[4] == "mouse" for m in MENU)
