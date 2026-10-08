"""
orvue_us_inverse.mapping.evaluate_sweep - offline evaluation of a saved sweep against the ground truth (S4).

    python -m orvue_us_inverse.mapping.evaluate_sweep [output/sweeps/<sweep>.npz] [--voxel 0.5] [--fill]
                                                       [--max-gap 1] [--out output/results]
    python -m orvue_us_inverse evaluate [sweep.npz] [options]

Reconstructs the sweep (oracle labels, mapping.reconstruct_sweep.reconstruct), evaluates it inside the observed voxels
(mapping.evaluate) and writes output/results/<case>_<time>/: report.json, report.md, overlay_3d.png, structures.png and
slices_errors.png (centre slices, wrong voxels magenta). Without a path: the newest sweep in output/sweeps/.
"""
import argparse
import os
import time

import cv2

from orvue_us_inverse.mapping.evaluate import evaluate, summary_table, write_report
from orvue_us_inverse.mapping.reconstruct_sweep import latest_sweep, reconstruct
from orvue_us_inverse.mapping.render import SliceView
from orvue_us_inverse.mapping.sweep_io import Sweep
from orvue_us_inverse.paths import RESULTS_DIR, SWEEPS_DIR
from orvue_us_inverse.simulation.anatomy import build_case


def evaluate_file(path: str, voxel_mm: float = 0.5, fill: bool = False, max_gap: int = 1,
                  out: str = RESULTS_DIR) -> tuple[str, dict]:
    """Reconstruct and evaluate one sweep file; returns (report folder, report)."""
    sweep = Sweep.load(path)
    case = sweep.metadata.get("case")
    if not case:
        raise ValueError(f"{path}: no case in the sweep metadata")
    comp, labels, timings = reconstruct(sweep, voxel_mm, fill, max_gap)
    an = build_case(case)
    t0 = time.perf_counter()
    ev = evaluate(labels, an, comp.grid, case=case,
                  extra=dict(source_sweep=os.path.abspath(path), frames=len(sweep), fill=fill,
                             max_gap=max_gap if fill else 0, sweep_config=sweep.metadata.get("sweep_config"),
                             observed_fraction_before_fill=float((comp.result() >= 0).mean()), timings=timings))
    ev.report["timings"]["evaluate_s"] = time.perf_counter() - t0
    folder = os.path.join(out, f"{case}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(folder, exist_ok=True)
    view = SliceView(comp.grid, gt=ev.gt)
    view.errors, view.show_gt = True, False
    cv2.imwrite(os.path.join(folder, "slices_errors.png"), view.update(labels))
    write_report(ev, folder, title=f"{os.path.basename(path)}: {len(sweep)} frames, voxels {voxel_mm:g} mm"
                                   + (f", holes filled (max gap {max_gap})" if fill else ""))
    return folder, ev.report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a saved sweep against the ground truth.")
    p.add_argument("sweep", nargs="?", help="sweep .npz (default: the newest file in output/sweeps/)")
    p.add_argument("--voxel", type=float, default=0.5, help="voxel size in mm")
    p.add_argument("--fill", action="store_true", help="fill small holes before evaluating")
    p.add_argument("--max-gap", type=int, default=1, help="largest gap (voxels) --fill closes")
    p.add_argument("--out", default=RESULTS_DIR, help="output folder")
    a = p.parse_args(argv)
    path = a.sweep or latest_sweep()
    if path is None:
        print(f"[evaluate] no sweep in {SWEEPS_DIR}: save one with s in the scripted sweep (menu 4)")
        return 1
    folder, report = evaluate_file(path, a.voxel, a.fill, a.max_gap, a.out)
    print(summary_table(report))
    print(f"[evaluate] report: {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
