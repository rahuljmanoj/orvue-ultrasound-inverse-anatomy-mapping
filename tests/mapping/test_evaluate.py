"""mapping/evaluate.py, mapping/evaluate_sweep.py and "complete" in run_scripted (S4).

Synthetic reconstructions built from the normal case's ground truth on the 0.5 mm grid: perfect, shifted by one voxel
along x, cystic duct removed from the bile class, and observed only in part of the volume.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from orvue_us_inverse.mapping.acquisition import Acquirer  # noqa: E402
from orvue_us_inverse.mapping.config import AcquisitionConfig, GridConfig, SweepConfig  # noqa: E402
from orvue_us_inverse.mapping.evaluate import (CLASSES, evaluate, instance_masks, summary_table,  # noqa: E402
                                               surface_distances, write_report)
from orvue_us_inverse.mapping.evaluate_sweep import main as evaluate_main  # noqa: E402
from orvue_us_inverse.mapping.poses import ScriptedSweep  # noqa: E402
from orvue_us_inverse.mapping.probe import make_simulator  # noqa: E402
from orvue_us_inverse.mapping.recon import VoxelGrid, ground_truth  # noqa: E402
from orvue_us_inverse.mapping.run_scripted import ScriptedPlayback  # noqa: E402
from orvue_us_inverse.simulation.anatomy import build_case  # noqa: E402

GRID = VoxelGrid(GridConfig())
H = GRID.voxel_mm
LUMEN_CLASSES = ("bile", "arterial blood", "venous blood", "stone", "lymph node")


@pytest.fixture(scope="module")
def an():
    return build_case("normal")


@pytest.fixture(scope="module")
def gt(an):
    return ground_truth(an, GRID)


@pytest.fixture(scope="module")
def inst(an):
    return instance_masks(an, GRID)


def _eval(labels, an, gt, inst):
    return evaluate(labels, an, GRID, gt=gt, instances=inst, case="normal").report


def test_instance_masks(an, inst, gt):
    names = {t.name for t in an.tubes} | {b.name for b in an.blobs}
    assert set(inst) == names
    gb = inst["gallbladder"]
    assert set(gb) == {"lumen", "wall"} and gb["lumen"].label == 3 and gb["wall"].label == 2
    assert set(inst["gallstone_1"]) == {"interior"} and inst["gallstone_1"]["interior"].label == 4
    # lumen voxels of a tube mostly carry its lumen label in the ground truth (stones / other lumens overlap a few)
    for name in ("chd_cbd", "portal_vein", "proper_hepatic_artery"):
        lum = inst[name]["lumen"]
        assert (gt.ravel()[lum.flat] == lum.label).mean() > 0.9
        assert not np.intersect1d(lum.flat, inst[name]["wall"].flat).size        # lumen and wall disjoint


def test_perfect_reconstruction(an, gt, inst):
    r = _eval(gt.copy(), an, gt, inst)
    assert r["observed_fraction"] == 1.0 and r["accuracy_observed"] == 1.0
    for name, c in r["classes"].items():
        assert c["dice"] == 1.0 and c["iou"] == 1.0 and c["precision"] == 1.0 and c["recall"] == 1.0, name
        assert c["msd_mm"] == 0.0 and c["hd95_mm"] == 0.0, name
        assert c["coverage"] == 1.0
    assert all(s["status"] == "detected" and s["recall"] == 1.0 for s in r["structures"])
    assert r["counts"] == {"detected": len(inst), "missed": 0, "not covered": 0, "no voxels": 0}
    t = r["topology"]
    assert t["connected_truth"] and t["connected_recon"] and t["status"] == "ok"
    assert set(r["classes"]) == set(CLASSES.values())


def test_shift_one_voxel_along_x(an, gt, inst):
    """HD95 = one voxel; the symmetric mean surface distance is ~0.45-0.7 voxel, because surfaces parallel to the
    shift do not move (decided at S4: the standard metric, not a 'moved-surface' variant)."""
    sh = np.roll(gt, 1, axis=0)
    sh[0] = -1
    r = _eval(sh, an, gt, inst)
    for name in LUMEN_CLASSES:
        c = r["classes"][name]
        assert c["hd95_mm"] == pytest.approx(H, rel=0.3), name
        assert 0.4 * H <= c["msd_mm"] <= 0.8 * H, (name, c["msd_mm"])
        assert c["dice"] < 1.0
    print("\n[S4] one-voxel shift: " + ", ".join(f"{n} MSD {r['classes'][n]['msd_mm']:.3f} HD95 "
                                                 f"{r['classes'][n]['hd95_mm']:.2f} mm" for n in LUMEN_CLASSES))


def test_surface_distance_of_a_slab():
    """A wide, thin slab shifted across its large faces by 2 voxels (1 mm): those faces (~90 % of the boundary) move
    by the full shift, so MSD ~ 1 mm and HD95 = 1 mm (the small side faces parallel to the shift do not move)."""
    a = np.zeros((20, 120, 120), bool)
    a[5:12, 2:118, 2:118] = True
    b = np.roll(a, 2, axis=0)
    msd, hd95 = surface_distances(a, b, 0.5)
    assert msd == pytest.approx(1.0, rel=0.15) and hd95 == pytest.approx(1.0)
    assert surface_distances(a, np.zeros_like(a), 0.5) == (None, None)


def test_cystic_duct_removed_disconnects(an, gt, inst):
    cut = gt.copy()
    f = cut.ravel()
    cd = inst["cystic_duct"]["lumen"].flat
    f[cd[f[cd] == 3]] = 5                                   # cystic duct lumen -> duct wall: no bile path
    t = _eval(cut, an, gt, inst)["topology"]
    assert t["connected_truth"] and not t["connected_recon"]
    assert t["status"] == "disconnected"


def test_outside_observed_region_is_not_covered(an, gt, inst):
    part = gt.copy()
    part[60:] = -1                                          # only x < 30 mm observed
    r = _eval(part, an, gt, inst)
    st = {s["name"]: s for s in r["structures"]}
    for name in ("portal_vein", "chd_cbd", "proper_hepatic_artery", "calot_lymph_node", "left_portal_vein"):
        assert st[name]["status"] == "not covered", name
        assert st[name]["coverage"] < 0.2
    assert st["gallstone_2"]["status"] == "detected"          # at x = 14 mm, inside the observed part
    assert r["counts"]["missed"] == 0
    assert r["topology"]["status"] == "not covered"
    assert r["classes"]["venous blood"]["coverage"] < 0.05
    assert r["classes"]["bile"]["dice"] == 1.0              # metrics only inside the observed voxels


def test_missed_structure(an, gt, inst):
    lab = gt.copy()
    f = lab.ravel()
    f[inst["calot_lymph_node"]["interior"].flat] = 1         # node reconstructed as fat
    st = {s["name"]: s for s in _eval(lab, an, gt, inst)["structures"]}
    assert st["calot_lymph_node"]["status"] == "missed" and st["calot_lymph_node"]["recall"] == 0.0


def test_report_files(an, gt, inst, tmp_path):
    ev = evaluate(gt.copy(), an, GRID, gt=gt, instances=inst, case="normal")
    folder = write_report(ev, str(tmp_path / "rep"))
    assert sorted(os.listdir(folder)) == ["overlay_3d.png", "report.json", "report.md", "structures.png"]
    r = json.load(open(os.path.join(folder, "report.json"), encoding="utf-8"))
    assert r["classes"]["bile"]["dice"] == 1.0 and r["topology"]["status"] == "ok"
    md = open(os.path.join(folder, "report.md"), encoding="utf-8").read()
    assert "| bile (3) |" in md and "| gallbladder (bile) |" in md
    assert "Topology gallbladder - chd_cbd" in summary_table(r)


@pytest.fixture(scope="module")
def short_sweep(tmp_path_factory):
    """Oracle labels, one lane at the default 0.25 mm over the hilum (x 40-70 mm, y 20-60 mm: 161 frames), saved."""
    cfg = SweepConfig(region_x_mm=(40, 70), region_y_mm=(20, 60))
    acq = Acquirer(make_simulator("normal"), cfg, AcquisitionConfig(store_images=False))
    for s in ScriptedSweep(cfg).samples():
        acq.feed(s.T, s.t, s.recording)
    return acq.sweep.save(str(tmp_path_factory.mktemp("sweeps") / "hilum.npz"))


def test_evaluate_sweep_script(short_sweep, tmp_path):
    assert evaluate_main([short_sweep, "--fill", "--out", str(tmp_path)]) == 0
    (folder,) = os.listdir(tmp_path)
    assert folder.startswith("normal_")
    files = sorted(os.listdir(tmp_path / folder))
    assert files == ["overlay_3d.png", "report.json", "report.md", "slices_errors.png", "structures.png"]
    r = json.load(open(tmp_path / folder / "report.json", encoding="utf-8"))
    st = {s["name"]: s["status"] for s in r["structures"]}
    assert st["chd_cbd"] == "detected" and st["gallstone_1"] == "not covered"
    assert r["fill"] is True and r["frames"] == 161


def test_app_complete(tmp_path):
    app = ScriptedPlayback("normal", SweepConfig(region_x_mm=(40, 70), region_y_mm=(20, 60)),
                           AcquisitionConfig(store_images=False))
    while not app.done:
        app.advance(10.0)
    folder, table = app.complete(results=str(tmp_path))
    assert app.completed and app.advance(10.0) == 0                     # acquisition stopped
    assert "slices_errors.png" in os.listdir(folder) and "report.md" in os.listdir(folder)
    assert "| Class |" in table and app.slice_view.errors
    assert app.evaluation.report["sweep_finished"] is True and app.evaluation.report["frames"] == 161
    img = app.slice_image()
    assert img.shape[1] == app.slice_view.size[1]
    app.restart()
    assert not app.completed and not app.slice_view.errors
