"""mapping/render.py, mapping/live3d.py and the S3 parts of run_scripted (headless, matplotlib Agg) (S3).

Small reconstruction: one lane at 0.5 mm spacing over x 35-65 mm (61 frames of oracle labels). Timing uses the normal
case's ground-truth volume at 0.5 mm, i.e. a perfect whole-region reconstruction (same size and structures).
"""
import os
import re
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from orvue_us_inverse.mapping.acquisition import Acquirer  # noqa: E402
from orvue_us_inverse.mapping.config import AcquisitionConfig, GridConfig, SweepConfig  # noqa: E402
from orvue_us_inverse.mapping.poses import ScriptedSweep  # noqa: E402
from orvue_us_inverse.mapping.probe import make_simulator  # noqa: E402
from orvue_us_inverse.mapping.recon import LabelCompounder, VoxelGrid, ground_truth  # noqa: E402
from orvue_us_inverse.mapping.render import (GROUPS, SLICE_BGR, UNOBSERVED_BGR, SliceView, browser_payload,  # noqa: E402
                                             export_stl, snapshot_3d, surface_meshes, write_browser_view)
from orvue_us_inverse.mapping.run_scripted import ScriptedPlayback  # noqa: E402

GRID = VoxelGrid(GridConfig())


@pytest.fixture(scope="module")
def sim():
    return make_simulator("normal")


@pytest.fixture(scope="module")
def small(sim):
    cfg = SweepConfig(region_x_mm=(35, 65), region_y_mm=(35, 65), frame_spacing_mm=0.5)
    acq = Acquirer(sim, cfg, AcquisitionConfig(store_images=False))
    for s in ScriptedSweep(cfg).samples():
        acq.feed(s.T, s.t, s.recording)
    comp = LabelCompounder.from_sweep(acq.sweep, GRID)
    comp.insert_batch(acq.sweep.frames)
    return comp


@pytest.fixture(scope="module")
def gt(sim):
    return ground_truth(sim.an, GRID)


# ---------------------------------------------------------------- slices
def test_slice_shapes_colours_and_crosshair(small):
    lab = small.result()
    bile = np.argwhere(lab == 3)
    ix, iy, iz = bile[len(bile) // 2]
    view = SliceView(GRID, crosshair=(ix, iy, iz))
    img = view.update(small)
    assert img.shape == (*view.size, 3) and img.dtype == np.uint8
    s = view.s
    assert s == 2
    for name, (rx, ry, rw, rh) in view.rect.items():
        assert ry + rh <= img.shape[0] and rx + rw <= img.shape[1]
    assert view.rect["xy"][2:] == (400, 400) and view.rect["xz"][2:] == (400, 200)
    rx, ry = view.rect["xy"][:2]
    # a bile voxel next to the crosshair (off the crosshair lines) is drawn green
    for dx in range(2, 40):
        if lab[ix + dx, iy + 3, iz] == 3:
            px, py = rx + (ix + dx) * s, ry + (iy + 3) * s
            assert tuple(img[py, px]) == tuple(SLICE_BGR[3])
            break
    else:
        pytest.fail("no bile voxel near the crosshair")
    # far outside the scanned block: unobserved
    assert tuple(img[ry + 10 * s, rx + 10 * s]) == UNOBSERVED_BGR
    # slices from the compounder equal slices from the full volume
    assert np.array_equal(img, view.update(lab))
    # click sets the crosshair; move clips to the grid
    assert view.click(rx + 20 * s, ry + 30 * s) and view.cross[:2] == [20, 30]
    view.move(dz=-1000)
    assert view.cross[2] == 0
    assert not view.click(5, 5)


def test_slice_update_idempotent_and_gt(small, gt):
    view = SliceView(GRID, gt=gt)
    a = view.update(small)
    b = view.update(small)
    assert np.array_equal(a, b)
    view.show_gt = False
    c = view.update(small)
    assert not np.array_equal(a, c)            # the contours change the picture
    view.show_gt = True
    assert np.array_equal(view.update(small), a)


def test_result_slice_matches_result(small):
    lab = small.result()
    for axis, idx in ((0, 100), (1, 90), (2, 40)):
        sl = [slice(None)] * 3
        sl[axis] = idx
        assert np.array_equal(small.result_slice(axis, idx), lab[tuple(sl)])
    assert np.array_equal(small.column_hits(), small.hits.reshape(GRID.shape).sum(axis=2))


# ---------------------------------------------------------------- surfaces and outputs
def test_meshes_inside_grid(small):
    meshes = surface_meshes(small.result(), GRID)
    assert set(meshes) <= set(GROUPS) and "bile" in meshes
    lo, hi = GRID.lo, GRID.lo + np.array(GRID.shape) * GRID.voxel_mm
    for name, (v, f) in meshes.items():
        assert v.dtype == np.float32 and f.dtype == np.int32 and len(f) > 0
        assert (v >= lo - 1e-3).all() and (v <= hi + 1e-3).all(), name
        assert f.max() < len(v)
    # surfaces stay within the observed block (x, y 35-65 mm), padded by one voxel
    v = np.concatenate([m[0] for m in meshes.values()])
    assert v[:, 0].min() >= 35 - 0.5 and v[:, 0].max() <= 65.5 + 0.5
    empty = surface_meshes(np.full(GRID.shape, -1, np.int8), GRID)
    assert empty == {}


def test_timing_slice_and_meshes(gt):
    """Normal case at 0.5 mm: slice update <= 50 ms, mesh extraction <= 300 ms (targets; loose asserts)."""
    comp = LabelCompounder(GRID)
    flat = np.arange(GRID.n_voxels)
    comp.votes[flat * comp.n_classes + gt.ravel()] = 1           # a perfect reconstruction as votes
    comp.hits[:] = 1
    view = SliceView(GRID)
    t = []
    for k in range(10):
        view.move(dz=1)
        t0 = time.perf_counter()
        view.update(comp)
        t.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    meshes = surface_meshes(gt, GRID)
    tm = time.perf_counter() - t0
    print(f"\n[S3] slice update {1000 * np.mean(t):.1f} ms (max {1000 * np.max(t):.1f}); mesh extraction "
          f"{1000 * tm:.0f} ms ({sum(len(f) for _, f in meshes.values())} triangles)")
    assert np.mean(t) <= 0.050
    assert tm <= 0.6


def test_snapshot_stl_and_browser(small, gt, tmp_path):
    meshes = surface_meshes(small.result(), GRID)
    gtm = surface_meshes(gt, GRID)
    png = snapshot_3d(meshes, str(tmp_path / "snap.png"), gt=gtm, title="test")
    assert os.path.getsize(png) > 10_000
    import matplotlib.image as mpimg
    im = mpimg.imread(png)
    assert im.shape[:2] == (900, 1200)
    stls = export_stl(meshes, str(tmp_path / "stl"))
    assert sorted(os.path.basename(p) for p in stls) == sorted(f"recon_{n}.stl" for n in meshes)
    for p in stls:
        n = int.from_bytes(open(p, "rb").read()[80:84], "little")
        name = re.match(r"recon_(.+)\.stl", os.path.basename(p)).group(1)
        assert n == len(meshes[name][1]) and os.path.getsize(p) == 84 + 50 * n
    payload = browser_payload(meshes, gtm)
    assert [c["name"] for c in payload["cases"]] == ["reconstruction", "reconstruction_vs_truth", "ground_truth"]
    assert all(s["kind"] == "blob" for c in payload["cases"] for s in c["structures"])
    html = write_browser_view(meshes, str(tmp_path / "recon.html"), gt=gtm)
    text = open(html, encoding="utf-8").read()
    assert "reconstruction_vs_truth" in text and "/*__PAYLOAD__*/null" not in text


def test_app_outputs(tmp_path):
    app = ScriptedPlayback("normal", SweepConfig(frame_spacing_mm=0.5), AcquisitionConfig(store_images=False))
    n = 0
    while n < 150:                                     # lane 1 (x = 15 mm) up to y = 74.5 mm: through the GB
        n += app.advance(1.0)
    img = app.slice_image()
    assert img.shape[1] == app.slice_view.size[1] and app.observed_pct > 0
    app.cycle_view()                                   # coverage
    cov = app.coverage_image()
    assert cov.shape == (400, 400, 3) and cov.any() and not cov[:, -50:].any()     # x > 87.5 mm not yet imaged
    png, stls = app.export_3d(results=str(tmp_path / "results"), export=str(tmp_path / "export"))
    assert os.path.isfile(png) and stls and all(os.path.isfile(p) for p in stls)
    assert os.path.isfile(app.browser_view(folder=str(tmp_path / "viewer3d")))


# ---------------------------------------------------------------- optional live PyVista window
def test_live3d_off_screen(small, gt, tmp_path):
    pytest.importorskip("pyvista")
    from orvue_us_inverse.mapping.live3d import Live3D
    live = Live3D(off_screen=True, window_size=(400, 300))
    meshes = surface_meshes(small.result(), GRID)
    live.update(meshes, gt=surface_meshes(gt, GRID))
    live.update(meshes)                                # replacing the surfaces works repeatedly
    assert sorted(live._names) == sorted(f"rec_{n}" for n in meshes)
    p = live.screenshot(str(tmp_path / "live.png"))
    assert os.path.getsize(p) > 1000
    live.close()
