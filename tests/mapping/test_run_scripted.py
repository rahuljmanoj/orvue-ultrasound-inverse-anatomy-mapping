"""mapping/run_scripted.py without windows: stepping on simulated time, drawing, file name, saving (S1)."""
import os
import re

import numpy as np

from orvue_us_inverse.__main__ import COMMANDS, MENU
from orvue_us_inverse.mapping.config import AcquisitionConfig, SweepConfig
from orvue_us_inverse.mapping.run_scripted import (KEYS_H, MARGIN, PANEL_W, VIEWS, ScriptedPlayback, parse_args,
                                                   sweep_filename)
from orvue_us_inverse.mapping.sweep_io import Sweep


def test_playback_steps_draws_and_saves(tmp_path):
    cfg = SweepConfig(yaw_list_deg=[0.0, 90.0])
    app = ScriptedPlayback("normal", cfg, AcquisitionConfig(store_images=False))
    assert app.expected == 1608
    n = 0
    while n < 5:                                   # each advance stops after one captured frame
        n += app.advance(1.0)
    assert len(app.acq.sweep) == 5 and app.latest.index == 4
    assert app.t == app.latest.t
    assert app.trace_mask.any()
    img = app.compose()
    assert img.shape == (56 + app.size + KEYS_H, app.size + PANEL_W, 3)
    assert app.size == 400 + 2 * MARGIN
    assert app.bmode_image().shape == (501, 301, 3)
    path = app.save(str(tmp_path))
    back = Sweep.load(path)
    assert len(back) == 5 and back.metadata["complete"] is False and back.metadata["expected_frames"] == 1608
    assert app.comp.n_frames == 5 and app.comp.hits.any()             # reconstruction grows with the frames
    for _ in VIEWS:                                                   # black box, coverage, revealed
        app.cycle_view()
        assert app.compose().shape == img.shape
    app.restart()
    assert len(app.acq.sweep) == 0 and not app.trace_mask.any() and app.comp.n_frames == 0


def test_sweep_filename_and_cli():
    cfg = SweepConfig(yaw_list_deg=[0.0, 90.0], overlap_pct=40.0)
    name = sweep_filename("normal", cfg, False, stamp="20261008-120000")
    assert name == "normal_yaw0-90_ov40_sp0.5_v10_labels_20261008-120000.npz"
    assert re.fullmatch(r"normal_yaw0_ov20_sp0\.5_v10_\d{8}-\d{6}\.npz", sweep_filename("normal", SweepConfig(), True))
    a = parse_args(["--case", "inflamed_obese", "--yaw", "0", "90", "--overlap", "40", "--spacing", "0.25",
                    "--speed", "5", "--no-images"])
    assert (a.case, a.yaw, a.overlap, a.spacing, a.speed, a.no_images) == ("inflamed_obese", [0.0, 90.0], 40.0,
                                                                          0.25, 5.0, True)
    assert parse_args(["--live3d"]).live3d is True and parse_args([]).live3d is False


def test_menu_step():
    assert COMMANDS["scripted"][0] == ["-m", "orvue_us_inverse.mapping.run_scripted"]
    assert any(m[4] == "scripted" for m in MENU)
    assert [m[1] for m in MENU] == [str(i) for i in range(1, len(MENU) + 1)]
