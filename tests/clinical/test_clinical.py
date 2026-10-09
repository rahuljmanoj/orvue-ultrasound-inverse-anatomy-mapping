"""clinical/ (clinical window): scripted pose source, AR overlay, the INVERSE MAPPING tab with a fake tracker, the
B-MODE tab, tab switching and the clinical menu (no camera, no window)."""
import subprocess
import sys

import cv2
import numpy as np
import pytest

from orvue_us_inverse import paths
from orvue_us_inverse.clinical import ar
from orvue_us_inverse.clinical.app import CALIBRATE_RECT, TAB_KEY, ClinicalApp
from orvue_us_inverse.clinical.mapping_tab import HEIGHT, MAP_RECT, WIDTH, MappingSession, hit_mode_tab
from orvue_us_inverse.mapping.acquisition import Acquirer
from orvue_us_inverse.mapping.config import AcquisitionConfig, SweepConfig
from orvue_us_inverse.mapping.poses import ScriptedPose
from orvue_us_inverse.mapping.probe import make_simulator
from orvue_us_inverse.mapping.recon import VoxelGrid
from orvue_us_inverse.simulation.bmode import BModeSimulator
from orvue_us_inverse.tracking.tracker import TrackState
from tests.helpers import synthetic_scene as ss

DOWN, UP, MOVE = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_MOUSEMOVE
SMALL = dict(region_x_mm=(35.0, 65.0), region_y_mm=(30.0, 40.0))     # one short lane: fast tests


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeTracker:
    def __init__(self):
        self.x, self.y, self.yaw, self.valid = 50.0, 35.0, 0.0, True

    def get_state(self):
        s = TrackState(valid=self.valid, probe_valid=self.valid, phantom_valid=True, n_reference_markers_used=4,
                       fps=30.0)
        if self.valid:
            s.T_phantom_probe = s.T_phantom_probe_raw = BModeSimulator.pose_from_xy_yaw(self.x, self.y, self.yaw)
        return s


def session(tracker=None, **kw):
    clock = FakeClock()
    s = MappingSession("normal", SweepConfig(**SMALL), AcquisitionConfig(store_images=False), tracker=tracker,
                       clock=clock, **kw)
    return s, clock


def click(s, rect):
    x, y, w, h = rect
    s.route_mouse(DOWN, x + w // 2, y + h // 2)
    s.route_mouse(UP, x + w // 2, y + h // 2)


def test_scripted_pose_captures_the_plan():
    cfg = SweepConfig(**SMALL)
    acq = Acquirer(make_simulator("normal"), cfg, AcquisitionConfig(store_images=False))
    p = ScriptedPose(cfg)
    assert not p.recording
    p.press()
    while p.playing:
        p.advance(acq)
        s = p.sample()
        acq.feed(s.T, s.t, s.recording)
    assert len(acq.sweep) == p.plan.expected_frames()
    x0 = acq.sweep.frames[0].T_true[0, 3]
    d = np.diff([f.T_true[1, 3] for f in acq.sweep.frames if f.T_true[0, 3] == x0])
    assert np.allclose(np.abs(d), cfg.frame_spacing_mm, atol=1e-3)     # every spacing, none skipped


def test_surface_map_and_projection():
    grid = VoxelGrid()
    lab = np.full(grid.shape, 0, np.int8)                 # liver everywhere (background)
    lab[20:40, 20:40, 30:50] = 3                          # bile 15-25 mm deep
    lab[20:40, 20:40, 60:70] = 7                          # artery under it (covered: dashed outline)
    lab[100:120, 100:120, 10:20] = 7                      # artery 5 mm deep
    l2, d2 = ar.surface_map(lab, grid)
    assert l2.shape == (grid.shape[1], grid.shape[0])
    assert l2[30, 30] == 3 and d2[30, 30] == pytest.approx(15.25) and l2[110, 110] == 7 and l2[0, 0] == -1
    fps = ar.footprints(lab)
    bgr, alpha = ar.ar_layer(l2, d2, fps)
    k = ar.UPSCALE
    assert is_artery(bgr, alpha, 30, 21) and not is_artery(bgr, alpha, 30, 30)     # covered: outline band only
    marks = ar.marked_columns(l2, fps, "artery")
    assert marks[30, 21] and not marks[30, 30] and marks[110, 110]
    assert alpha[0, 0] == 0 and alpha[30 * k + 2, 30 * k + 2] > 0.3
    assert alpha[110 * k + 2, 110 * k + 2] > alpha[30 * k + 2, 30 * k + 2]      # shallower: more opaque
    frame = np.full((ss.H, ss.W, 3), 128, np.uint8)
    T = ss.camera_pose(300.0)
    out = ar.project_layer(frame, bgr, alpha, T, ss.K, ss.DIST, grid)
    changed = np.any(out != frame, axis=2)
    pts, _ = cv2.projectPoints(np.array([[15.0, 15.0, 0.0], [55.0, 55.0, 0.0], [90.0, 10.0, 0.0]]),
                               cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], ss.K, ss.DIST)
    (u1, v1), (u2, v2), (u3, v3) = np.round(pts.reshape(-1, 2)).astype(int)
    assert changed[v1, u1] and changed[v2, u2] and not changed[v3, u3] and not changed[0, 0]
    vols = {n: v for n, _, v in ar.group_volumes_ml(lab, grid.voxel_mm)}
    assert vols["gallbladder / bile ducts"] == pytest.approx(20 * 20 * 20 * 0.125 / 1000)


def is_artery(bgr, alpha, row, col) -> bool:
    """True when the layer block of voxel column (row = y, col = x) holds an artery-coloured pixel."""
    k = ar.UPSCALE
    blk = bgr[row * k:(row + 1) * k, col * k:(col + 1) * k].astype(int)
    a = alpha[row * k:(row + 1) * k, col * k:(col + 1) * k]
    red = (blk[..., 2] > 1.8 * blk[..., 1]) & (blk[..., 2] > 1.8 * blk[..., 0]) & (a > 0)
    return bool(red.any())


def test_covered_cystic_arteries_are_shown():
    """Ground truth of the normal case (Anatomy.labels on the grid): the cystic artery and its two branches run under
    the gallbladder in places; >= 95 % of their footprints carry an artery outline or fill."""
    from orvue_us_inverse.mapping.evaluate import instance_masks
    from orvue_us_inverse.mapping.recon import ground_truth
    from orvue_us_inverse.simulation.anatomy import build_case
    grid, an = VoxelGrid(), build_case("normal")
    gt = ground_truth(an, grid)
    inst = instance_masks(an, grid)
    l2, d2 = ar.surface_map(gt, grid)
    fps = ar.footprints(gt)
    bgr, alpha = ar.ar_layer(l2, d2, fps)
    marks = ar.marked_columns(l2, fps, "artery")
    top = ar.group_map(l2) == ar.group_index("artery")
    for name in ("cystic_artery", "cystic_artery_superficial", "cystic_artery_deep"):
        ix, iy, _ = np.unravel_index(inst[name]["lumen"].flat, grid.shape)
        cols = sorted(set(zip(iy.tolist(), ix.tolist())))             # (row = y, col = x)
        assert np.mean([marks[r, c] for r, c in cols]) >= 0.95, name
        assert np.mean([is_artery(bgr, alpha, r, c) for r, c in cols]) >= 0.95, name
        assert np.mean([top[r, c] for r, c in cols]) < 0.95               # partly covered: the outlines matter


@pytest.mark.parametrize("tilt", [0.0, 15.0])
def test_ar_registration(tilt):
    """A single-voxel structure at (37.25, 61.75) mm lands within 2 px of its projection at z = 0."""
    grid = VoxelGrid()
    lab = np.zeros(grid.shape, np.int8)
    ix, iy = int(37.25 / grid.voxel_mm), int(61.75 / grid.voxel_mm)
    assert grid.centre(np.array([[ix, iy, 0]]))[0, :2] == pytest.approx((37.25, 61.75))
    lab[ix, iy, 10] = 7
    l2, d2 = ar.surface_map(lab, grid)
    bgr, alpha = ar.ar_layer(l2, d2)
    T = ss.camera_pose(300.0, tilt_x_deg=tilt)
    frame = np.zeros((ss.H, ss.W, 3), np.uint8)
    out = ar.project_layer(frame, bgr, alpha, T, ss.K, ss.DIST, grid).astype(float).sum(axis=2)
    assert out.sum() > 0
    v, u = np.indices(out.shape)
    centroid = np.array([(u * out).sum(), (v * out).sum()]) / out.sum()
    p, _ = cv2.projectPoints(np.array([[37.25, 61.75, 0.0]]), cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], ss.K, ss.DIST)
    assert np.linalg.norm(centroid - p.reshape(2)) <= 2.0


def test_sources_and_recording_buttons():
    tr = FakeTracker()
    s, clock = session(tr)
    assert s.source == "camera" and s.buttons == {}
    s.compose()                                          # draws (and registers) the buttons
    click(s, s.buttons["scripted"])
    assert s.source == "scripted" and s.guide_lanes()[0] == s.plan.lanes_for(0.0)
    click(s, s.buttons["record"])                        # start the scripted sweep
    for _ in range(30):
        clock.t += 0.05
        s.step()
    assert len(s.frames) == 30 and len(s.strokes) == 1 and s.sources_used == {"scripted"}
    assert not s.set_source("mouse")                     # not while the sweep is running
    click(s, s.buttons["record"])                        # pause
    s.step()
    assert s.set_source("mouse") and s.guide_lanes() == ([], None)          # no lane guides for the mouse
    s.route_mouse(DOWN, *s.window_px(40, 60))
    for y in np.arange(60, 62.01, 0.25):
        clock.t += 0.05
        s.route_mouse(MOVE, *s.window_px(40, y))
        s.step()
    s.route_mouse(UP, *s.window_px(40, 62))
    assert len(s.strokes) == 2 and s.frames[-1].T_true[1, 3] == pytest.approx(62.0, abs=0.3)
    assert s.set_source("camera")
    click(s, s.buttons["record"])                        # latch recording on (no key held)
    for y in np.arange(70, 72.01, 0.25):
        clock.t += 0.05
        tr.y = y
        s.step()
    click(s, s.buttons["record"])
    s.step()
    assert len(s.strokes) == 3 and not s.pose.recording
    assert s.report_extra()["pose_source"] == "scripted+mouse+camera"
    n, (a, b) = len(s.frames), s.strokes[-1]
    click(s, s.buttons["undo"])
    assert len(s.frames) == n - (b - a) and len(s.strokes) == 2
    click(s, s.buttons["reset"])
    assert not s.frames and s.scripted_pose.i == 0 and not s.strokes


def test_map_coordinates_and_layout():
    s, _ = session()
    for x, y in ((0.0, 0.0), (37.5, 81.0), (100.0, 100.0)):
        wx, wy = s.window_px(x, y)
        assert MAP_RECT[0] <= wx <= MAP_RECT[0] + MAP_RECT[2] and MAP_RECT[1] <= wy <= MAP_RECT[1] + MAP_RECT[3]
        assert np.allclose(s.mm(wx, wy), (x, y), atol=0.2)
    assert s.source == "mouse" and not s.set_source("camera")                  # no tracker
    assert s.compose().shape == (HEIGHT, WIDTH, 3)
    s.step()
    live, cap = s.live_image()
    assert live.shape[:2] == (501, 301) and "live" in cap


def test_overlay_and_structures_after_a_scripted_sweep(tmp_path):
    s, clock = session()
    s.set_source("scripted")
    s.toggle_record()
    while s.scripted_pose.playing:
        clock.t += 0.05
        s.step()
    s.step()
    assert s.scripted_pose.done and "finished" in s.message
    s.refresh_view3d()
    assert s._ar is not None and any(v > 0 for _, _, v in s._groups)
    s.complete(results=str(tmp_path))
    assert s.evaluation.report["pose_source"] == "scripted"
    assert s.compose().shape == (HEIGHT, WIDTH, 3)


def test_bmode_tab_and_switching():
    app = ClinicalApp("normal", tracker=None)
    assert app.tab == "bmode" and app.frame().shape == (HEIGHT, WIDTH, 3)
    y = app.bmode.modes_y + 10 + 38 + 15                 # INVERSE MAPPING tab
    assert hit_mode_tab(app.bmode.modes_y, 60, y) == "INVERSE MAPPING"
    assert hit_mode_tab(app.bmode.modes_y, 60, y + 38) is None                # Doppler: to do
    app.on_mouse(DOWN, 60, y)
    assert app.tab == "mapping" and app.mapping.case == "normal"
    assert app.frame().shape == (HEIGHT, WIDTH, 3)
    first = app.mapping
    app.on_key(TAB_KEY)
    assert app.tab == "bmode"
    app.on_key(ord("n"))                                  # next case in the B-mode tab
    app.on_key(TAB_KEY)
    assert app.tab == "mapping" and app.mapping is not first and app.mapping.case == app.bmode.case
    app.close()


def test_space_toggles_recording_for_mouse_and_camera():
    tr = FakeTracker()
    s, clock = session(tr)
    assert s.source == "camera" and s.key_state is None          # SPACE toggles, no hold-to-record
    s.toggle_record()                                            # SPACE: on
    for y in np.arange(35, 37.01, 0.25):
        clock.t += 0.05
        tr.y = y
        s.step()
    s.toggle_record()                                            # SPACE: off
    s.step()
    assert len(s.strokes) == 1 and len(s.frames) == 9 and not s.pose.recording
    assert s.set_source("mouse")
    s.route_mouse(MOVE, *s.window_px(40, 60))
    s.toggle_record()                                            # SPACE: recording without a button held
    assert s.pose.recording and s.mouse_latched
    s.route_mouse(DOWN, *s.window_px(40, 60))                    # clicks do not end the latched stroke
    s.route_mouse(UP, *s.window_px(40, 60))
    for y in np.arange(60, 62.01, 0.25):
        clock.t += 0.05
        s.route_mouse(MOVE, *s.window_px(40, y))
        s.step()
    s.route_mouse(MOVE, MAP_RECT[0] - 50, MAP_RECT[1] - 50)     # leaving the map: the probe stays on the map
    assert 0 <= s.pose.x <= 100 and 0 <= s.pose.y <= 100
    assert not s.set_source("camera")                            # not while recording
    s.toggle_record()
    assert not s.pose.recording and len(s.strokes) == 2 and len(s.frames) >= 9 + 6   # map pixels: ~0.3 mm
    s.route_mouse(DOWN, *s.window_px(40, 70))                    # holding the button still records (S6)
    assert s.pose.recording
    s.route_mouse(UP, *s.window_px(40, 70))
    assert not s.pose.recording


def test_case_dropdown_both_tabs():
    app = ClinicalApp("normal", tracker=None)
    app.frame()
    dd = app.dropdown
    x, y, w, h = dd.RECT
    app.on_mouse(DOWN, x + 10, y + 10)                           # open
    assert dd.open
    img = app.frame()
    assert img.shape == (HEIGHT, WIDTH, 3)
    name, (rx, ry, rw, rh) = dd.rows()[3]
    app.on_mouse(DOWN, rx + 20, ry + 10)                         # pick the 4th case
    assert not dd.open and app.bmode.case == name and app.bmode.an.name == name
    app.switch("mapping")
    assert app.mapping.case == name
    app.mapping.set_source("scripted")
    name2, (rx, ry, _, _) = dd.rows()[0]
    app.on_mouse(DOWN, x + 10, y + 10)
    app.on_mouse(DOWN, rx + 20, ry + 10)                         # from the mapping tab: new mapping, same source
    assert app.tab == "mapping" and app.mapping.case == name2 == app.bmode.case
    assert app.mapping.source == "scripted"
    app.on_key(ord(" "))                                         # SPACE starts the scripted sweep ...
    app.frame()
    assert app.recording and not app.set_case(name)              # ... and blocks a case change
    app.on_key(ord(" "))
    app.frame()
    app.on_key(TAB_KEY)
    app.on_key(ord("n"))                                         # n / p in the B-mode tab use the same selector
    assert app.bmode.case == app.bmode.names[1]
    assert app.on_key(27) is False                               # Esc closes the window
    app.close()


def test_calibration_in_the_window():
    app = ClinicalApp("normal", tracker=None)
    x, y, w, h = CALIBRATE_RECT
    app.on_mouse(DOWN, x + 10, y + 10)
    assert app.calibration is not None and app.frame().shape == (HEIGHT, WIDTH, 3)
    assert app.on_key(13) and app.calibration is None                # no camera: Enter goes back
    from orvue_us_inverse.paths import CALIBRATION_PATH
    from orvue_us_inverse.tracking.tracker import ProbeTracker, TrackerConfig
    tr = ProbeTracker(source=None, config=TrackerConfig(calibration_path=CALIBRATION_PATH), intrinsics=(ss.K, ss.DIST))
    for k in range(3):
        tr.process_frame(ss.render(ss.camera_pose(300.0), ss.pose_from_xy_yaw(50.0, 50.0, 0.0), seed=k), k / 30)
    app = ClinicalApp("normal", tracker=tr)
    assert app.open_calibration()
    assert app.frame().shape == (HEIGHT, WIDTH, 3)
    assert app.calibration.routine.step == "yaw_wait"
    app.on_key(ord(" "))                                             # capture starts
    assert app.calibration.routine.step == "yaw_collect"
    app.on_key(27)                                                   # abort (nothing saved) ...
    assert app.calibration.finished and app.calibration is not None
    app.on_key(13)                                                   # ... and back
    assert app.calibration is None and app.tab == "bmode"
    app.close()


def test_entry_point():
    from orvue_us_inverse.__main__ import COMMANDS, DEFAULT_COMMAND, MENU
    assert DEFAULT_COMMAND == "clinical" and COMMANDS["clinical"][0] == ["-m", "orvue_us_inverse.clinical"]
    assert all(m[4] in COMMANDS for m in MENU)
    out = subprocess.run([sys.executable, "-m", "orvue_us_inverse", "dev"], stdin=subprocess.DEVNULL,
                         capture_output=True, text=True, cwd=paths.REPO_ROOT, timeout=60)
    assert "Mouse sweep" in out.stdout and "0  Exit" in out.stdout


def test_bmode_wheel_turns_the_probe():
    app = ClinicalApp("normal", tracker=None)
    yaw = app.bmode.state["yaw"]
    app.on_mouse(cv2.EVENT_MOUSEWHEEL, 1000, 300, 120 << 16)     # wheel forward
    assert app.bmode.state["yaw"] == yaw + 5
    app.on_mouse(cv2.EVENT_MOUSEWHEEL, 1000, 300, (-120 & 0xFFFF) << 16)
    app.on_mouse(cv2.EVENT_MOUSEWHEEL, 1000, 300, (-120 & 0xFFFF) << 16)
    assert app.bmode.state["yaw"] == yaw - 5
    assert any(k == "wheel / q / e" for k, _ in app.bmode.keys)
    assert app.frame().shape == (HEIGHT, WIDTH, 3)
    app.close()
