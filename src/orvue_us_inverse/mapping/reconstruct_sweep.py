"""
orvue_us_inverse.mapping.reconstruct_sweep - reconstruct a saved sweep into a label volume (S2, offline).

    python -m orvue_us_inverse.mapping.reconstruct_sweep [output/sweeps/<sweep>.npz] [--voxel 0.5] [--fill]
                                                          [--max-gap 1] [--splat 0]
    python -m orvue_us_inverse recon <sweep.npz> [options]

Writes to output/results/ (paths.RESULTS_DIR):
    recon_<name>.npz           labels int8 (nx, ny, nz; -1 unobserved), hits uint32, intensity float32 (NaN where
                               no image pixel; empty without images), grid (JSON), metadata (JSON: source sweep and
                               its metadata, settings, timings, observed fraction, accuracy)
    recon_<name>_xy.png        slice z = centre (top view: x right, y down)
    recon_<name>_xz.png        slice y = centre (x right, depth down)
    recon_<name>_yz.png        slice x = centre (y right, depth down)
Each PNG shows the reconstruction (left) and the ground truth Anatomy.labels at the voxel centres (right),
coloured by tissue (anatomy.COL_TAB), unobserved voxels hatched.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from orvue_us_inverse.mapping.config import GridConfig
from orvue_us_inverse.mapping.recon import (LabelCompounder, VoxelGrid, fill_small_holes, ground_truth,
                                            interior_mask)
from orvue_us_inverse.mapping.sweep_io import Sweep
from orvue_us_inverse.paths import RESULTS_DIR, SWEEPS_DIR
from orvue_us_inverse.simulation.anatomy import COL_TAB, TISSUES, build_case
from orvue_us_inverse.ui import clinical as ui

HATCH_BGR = ((15, 15, 15), (95, 95, 95))   # unobserved: diagonal hatching (liver is uniform grey 60)
SLICE_PX = 400                    # longest side of each slice image


def hatch(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.indices(shape)
    return np.where(((xx + yy) // 4 % 2 == 0)[..., None], HATCH_BGR[0], HATCH_BGR[1]).astype(np.uint8)


def colour(labels2d: np.ndarray, scale: int = 1) -> np.ndarray:
    """BGR image of a 2D label array (rows, cols), scaled up by `scale` (nearest); -1 = hatched."""
    lab = np.repeat(np.repeat(labels2d, scale, 0), scale, 1)
    img = hatch(lab.shape)
    obs = lab >= 0
    img[obs] = COL_TAB[lab[obs]]
    return img


def slices(vol: np.ndarray) -> dict[str, np.ndarray]:
    """Centre slices as 2D arrays with rows going down the picture: xy (rows y, cols x), xz and yz (rows z)."""
    nx, ny, nz = vol.shape
    return {"xy": vol[:, :, nz // 2].T, "xz": vol[:, ny // 2, :].T, "yz": vol[nx // 2, :, :].T}


def slice_titles(grid: VoxelGrid) -> dict[str, str]:
    xc, yc, zc = grid.centres()
    nx, ny, nz = grid.shape
    return {"xy": f"z = {zc[nz // 2]:g} mm (x right, y down)",
            "xz": f"y = {yc[ny // 2]:g} mm (x right, depth down)",
            "yz": f"x = {xc[nx // 2]:g} mm (y right, depth down)"}


def comparison_png(recon2d: np.ndarray, gt2d: np.ndarray, title: str) -> np.ndarray:
    """Reconstruction | ground truth side by side, scaled up (nearest), with captions and a tissue key."""
    s = max(1, SLICE_PX // max(recon2d.shape))
    a, b = colour(recon2d, s), colour(gt2d, s)
    h, w = a.shape[:2]
    pad, top, key_h = 16, 54, 46
    img = np.full((top + h + key_h, 3 * pad + 2 * w, 3), ui.BG, np.uint8)
    img[top:top + h, pad:pad + w] = a
    img[top:top + h, 2 * pad + w:2 * pad + 2 * w] = b
    obs = recon2d >= 0
    acc = f"{100 * (recon2d[obs] == gt2d[obs]).mean():.2f}% of observed voxels correct" if obs.any() else "nothing observed"
    ui.text(img, title, (pad, 22), ui.WHITE, 0.5)
    ui.text(img, f"reconstruction ({acc})", (pad, top - 8), ui.AMBER, 0.45)
    ui.text(img, "ground truth", (2 * pad + w, top - 8), ui.AMBER, 0.45)
    x, y = pad, top + h + 18
    for lab in range(len(TISSUES)):
        name = TISSUES[lab][0]
        cv2.rectangle(img, (x, y - 10), (x + 12, y + 2), tuple(int(c) for c in COL_TAB[lab]), -1)
        ui.text(img, name, (x + 16, y), ui.GREY, 0.38)
        x += 22 + ui.text_w(name, 0.38)
        if x > img.shape[1] - 110:
            x, y = pad, y + 18
    img[y - 10:y + 3, x:x + 13] = hatch((13, 13))
    ui.text(img, "unobserved", (x + 16, y), ui.GREY, 0.38)
    return img


def latest_sweep(folder: str = SWEEPS_DIR) -> str | None:
    """The most recently modified .npz in folder, or None."""
    if not os.path.isdir(folder):
        return None
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npz")]
    return max(files, key=os.path.getmtime) if files else None


def reconstruct(sweep: Sweep, voxel_mm: float = 0.5, fill: bool = False, max_gap: int = 1,
                splat_mm: float = 0.0) -> tuple[LabelCompounder, np.ndarray, dict]:
    """Insert every frame; returns the compounder, the (optionally hole-filled) label volume and timings."""
    grid = VoxelGrid(GridConfig(voxel_mm=voxel_mm))
    comp = LabelCompounder.from_sweep(sweep, grid, splat_mm=splat_mm)
    t = []
    for f in sweep.frames:
        t0 = time.perf_counter()
        comp.insert(f)
        t.append(time.perf_counter() - t0)
    labels = comp.result()
    timings = {"insert_ms_mean": 1000 * float(np.mean(t)) if t else None,
               "insert_ms_max": 1000 * float(np.max(t)) if t else None}
    if fill:
        t0 = time.perf_counter()
        labels = fill_small_holes(labels, max_gap)
        timings["fill_s"] = time.perf_counter() - t0
    return comp, labels, timings


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Reconstruct a saved sweep (oracle labels) into a label volume.")
    p.add_argument("sweep", nargs="?", help="sweep .npz (default: the newest file in output/sweeps/)")
    p.add_argument("--voxel", type=float, default=0.5, help="voxel size in mm")
    p.add_argument("--fill", action="store_true", help="fill small holes between observed voxels")
    p.add_argument("--max-gap", type=int, default=1, help="largest gap (voxels) --fill closes")
    p.add_argument("--splat", type=float, default=0.0, help="elevation splat thickness in mm (0 = plane only)")
    p.add_argument("--out", default=RESULTS_DIR, help="output folder")
    a = p.parse_args(argv)
    if a.sweep is None:
        a.sweep = latest_sweep()
        if a.sweep is None:
            print(f"[recon] no sweep in {SWEEPS_DIR}: save one with s in the scripted sweep (menu 4)")
            return 1

    sweep = Sweep.load(a.sweep)
    name = os.path.splitext(os.path.basename(a.sweep))[0]
    comp, labels, timings = reconstruct(sweep, a.voxel, a.fill, a.max_gap, a.splat)
    grid = comp.grid
    print(f"[recon] {name}: {len(sweep)} frames, insert {timings['insert_ms_mean']:.1f} ms/frame (mean)")

    case = sweep.metadata.get("case")
    gt = ground_truth(build_case(case), grid) if case else None
    obs = labels >= 0
    stats = {"observed_fraction": float(obs.mean())}
    if gt is not None and obs.any():
        inter = interior_mask(gt) & obs
        stats["accuracy_observed"] = float((labels[obs] == gt[obs]).mean())
        stats["accuracy_interior"] = float((labels[inter] == gt[inter]).mean()) if inter.any() else None
    print(f"[recon] observed {100 * stats['observed_fraction']:.1f}% of the volume"
          + (f"; correct: {100 * stats['accuracy_observed']:.2f}% of observed, "
             f"{100 * stats['accuracy_interior']:.3f}% of interior observed voxels"
             if "accuracy_observed" in stats else ""))

    os.makedirs(a.out, exist_ok=True)
    meta = dict(source_sweep=os.path.abspath(a.sweep), sweep_metadata=sweep.metadata, voxel_mm=a.voxel,
                fill=a.fill, max_gap=a.max_gap if a.fill else 0, splat_mm=a.splat, n_frames=len(sweep),
                timings=timings, **stats)
    inten = comp.intensity()
    out = os.path.join(a.out, f"recon_{name}.npz")
    np.savez_compressed(out, labels=labels, hits=comp.hits.reshape(grid.shape),
                        intensity=inten if inten is not None else np.zeros(0, np.float32),
                        grid=np.array(json.dumps(grid.to_dict())), metadata=np.array(json.dumps(meta)))
    print(f"[recon] {out}")
    if gt is not None:
        titles = slice_titles(grid)
        rs, gs = slices(labels), slices(gt)
        for k in ("xy", "xz", "yz"):
            png = os.path.join(a.out, f"recon_{name}_{k}.png")
            cv2.imwrite(png, comparison_png(rs[k], gs[k], f"{name}: {titles[k]}"))
            print(f"[recon] {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
