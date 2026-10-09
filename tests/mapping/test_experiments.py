"""mapping/experiments.py and run_experiments: a tiny grid completes headless and writes the CSV (S5).

Tiny grid: normal case, 2 mm spacing, 20 % overlap, yaw [0] and [0, 90] (two runs; three single-orientation
sweeps of 204 frames would be needed without reuse, two with it), sweeps cached in a temporary folder.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from orvue_us_inverse.mapping.experiments import (SETTING_COLUMNS, ExperimentGrid, RunSpec, estimate,  # noqa: E402
                                                  recommend, aggregate, run_experiments)

TINY = ExperimentGrid(spacings=(2.0,), overlaps=(20.0,), orientation_sets=((0.0,), (0.0, 90.0)), cases=("normal",),
                      default_extra_voxels=(), offset_check=False)


def test_grid_and_sweep_reuse():
    runs = TINY.runs()
    assert runs == [RunSpec("normal", 2.0, 20.0, (0.0,)), RunSpec("normal", 2.0, 20.0, (0.0, 90.0))]
    assert TINY.sweeps() == [("normal", 0.0, 2.0, 20.0), ("normal", 90.0, 2.0, 20.0)]     # [0] reused by [0, 90]
    full = ExperimentGrid()
    assert len(full.runs()) == 4 * 3 * 4 * 3 + 3 * 2 * 2          # grid + (0.25 mm voxels, half-voxel offset) extras
    assert len(full.sweeps()) == 3 * 4 * 4 * 3                     # case x yaw (0, 45, 90, 135) x spacing x overlap


def test_estimate(tmp_path):
    e = estimate(TINY, workers=2, cache_dir=str(tmp_path))
    assert e["runs"] == 2 and e["sweeps"] == 2 and e["frames_total"] == 2 * 204
    assert e["frames_to_acquire"] == 408 and e["frames_inserted"] == 204 + 408
    assert 0 < e["wall_min"] < e["serial_min"] + 1


def test_tiny_grid_runs_headless(tmp_path):
    out, rows = run_experiments(TINY, str(tmp_path / "out"), workers=2, cache_dir=str(tmp_path / "cache"),
                                log=lambda *a: None)
    files = sorted(os.listdir(out))
    for f in ("results.csv", "summary.md", "dice_vs_spacing.png", "hd95_vs_spacing.png", "vs_overlap.png",
              "vs_orientation.png", "time_vs_quality.png"):
        assert f in files, f
    with open(os.path.join(out, "results.csv"), encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols, data = reader.fieldnames, list(reader)
    assert len(data) == 2
    assert cols[:len(SETTING_COLUMNS)] == SETTING_COLUMNS
    for c in ("bile_dice", "bile_msd_mm", "bile_hd95_mm", "vein_wall_dice", "gallbladder_coverage",
              "gallbladder_recall", "gallbladder_dice", "gallbladder_msd_mm", "gallbladder_hd95_mm",
              "gallbladder_status", "cystic_artery_deep_recall", "topology", "scan_time_s", "frames"):
        assert c in cols, c
    a, b = data
    assert (a["orientations"], b["orientations"]) == ("0", "0+90")
    assert (int(a["frames"]), int(b["frames"])) == (204, 408)
    assert float(b["scan_time_s"]) > 2 * float(a["scan_time_s"])          # second pass + transitions
    assert a["fill_max_gap"] == "3"                                         # 2 mm spacing / 0.5 mm voxels
    assert float(a["observed_raw"]) < float(a["observed"]) <= 1.0
    assert float(a["gallbladder_dice"]) > 0.8 and a["gallbladder_status"] == "detected"
    pick, rule = recommend(aggregate(rows))
    assert pick["orientations"] in ("0", "0+90") and "cheapest" in rule
    with open(os.path.join(out, "summary.md"), encoding="utf-8") as fh:
        assert "## Recommendation" in fh.read()
    # cached sweeps are reused: a second run acquires nothing
    msgs = []
    run_experiments(TINY, str(tmp_path / "out2"), workers=1, cache_dir=str(tmp_path / "cache"), plots=False,
                    log=msgs.append)
    assert not any("sweeps" in m for m in msgs)


@pytest.mark.parametrize("spec,keys", [(RunSpec("normal", 0.5, 40.0, (0.0, 45.0)), 2),
                                       (RunSpec("normal", 0.5, 40.0, (90.0,), 0.25, -0.125), 1)])
def test_runspec(spec, keys):
    assert len(spec.sweep_keys()) == keys
    assert spec.strategy.startswith("sp0.5_ov40_yaw")


def test_error_study_tiny(tmp_path):
    """S7 error grid, 2 runs (no error / 1 mm jitter) on a coarse sweep: CSV, plots, summary, robust box."""
    from orvue_us_inverse.mapping.experiments import ERROR_KEYS, ErrorGrid, error_aggregate, robust_box, run_error_study
    g = ErrorGrid(jitter_mm=(0.0, 1.0), jitter_yaw_deg=(0.0,), latency_ms=(0.0,), bias_x_mm=(0.0,), cases=("normal",),
                  spacing_mm=2.0)
    runs = g.runs()
    assert len(runs) == 2 and runs[0].errors == () and runs[1].errors == (("jitter_mm", 1.0),)
    assert runs[0].error_model() is None and runs[1].error_model().jitter_mm == 1.0
    out, rows = run_error_study(g, str(tmp_path / "err"), workers=1, cache_dir=str(tmp_path / "cache"),
                                log=lambda *a: None)
    files = set(os.listdir(out))
    assert {"results_errors.csv", "summary_errors.md", "errors_dice.png", "errors_hd95.png",
            "errors_topology.png"} <= files
    with open(os.path.join(out, "results_errors.csv"), encoding="utf-8") as fh:
        cols = csv.DictReader(fh).fieldnames
    assert cols[:4] == list(ERROR_KEYS) and "cystic_duct_dice" in cols and "chd_cbd_status" in cols
    a0, a1 = error_aggregate(rows)
    assert a0["jitter_mm"] == 0.0 and a1["jitter_mm"] == 1.0 and a0["cystic_duct_dice"] > a1["cystic_duct_dice"]
    box = robust_box([dict(a0, pass_all=True), dict(a1, pass_all=False)], g)
    assert box["jitter_mm"] == 0.0 and box["combinations"] == 1
    with open(os.path.join(out, "summary_errors.md"), encoding="utf-8") as fh:
        assert "## Tracking accuracy needed" in fh.read()
