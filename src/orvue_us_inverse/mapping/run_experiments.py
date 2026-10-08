"""
orvue_us_inverse.mapping.run_experiments - run the sweep-strategy study headless (S5).

    python -m orvue_us_inverse.mapping.run_experiments [--workers 10] [--quick] [--yes] [--max-minutes 30]
    python -m orvue_us_inverse experiments [options]

Prints the estimated frames and runtime first. If the full grid is estimated above --max-minutes (wall clock with the
given workers), the reduced grid (experiments.quick_grid) runs instead and the full grid needs --yes. Results in
output/results/experiments_<time>/ (results.csv, plots, summary.md); sweeps cached in output/cache/experiments/.
--resummarize FOLDER rewrites summary.md and the plots of an existing folder from its results.csv (no runs).
"""
import argparse
import os
import time

from orvue_us_inverse.mapping.experiments import (ExperimentGrid, estimate, quick_grid, resummarize, run_experiments,
                                                  summary_tables)
from orvue_us_inverse.paths import RESULTS_DIR


def _print_estimate(name: str, e: dict) -> None:
    print(f"[experiments] {name}: {e['runs']} runs from {e['sweeps']} single-orientation sweeps; frames "
          f"{e['frames_total']} ({e['frames_to_acquire']} to acquire, the rest cached), {e['frames_inserted']} "
          f"inserted; estimated {e['serial_min']:.0f} min serial, ~{e['wall_min']:.0f} min with {e['workers']} workers",
          flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Headless sweep-strategy experiments.")
    p.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) - 2)))
    p.add_argument("--quick", action="store_true", help="run the reduced grid only")
    p.add_argument("--yes", action="store_true", help="run the full grid even above --max-minutes")
    p.add_argument("--max-minutes", type=float, default=30.0)
    p.add_argument("--out", default=RESULTS_DIR)
    p.add_argument("--resummarize", metavar="FOLDER", help="rewrite summary.md and plots from FOLDER/results.csv")
    a = p.parse_args(argv)
    if a.resummarize:
        rows = resummarize(a.resummarize)
        _print_summary(rows, a.resummarize)
        return 0
    full, quick = ExperimentGrid(), quick_grid()
    e_full = estimate(full, a.workers)
    _print_estimate("full grid", e_full)
    grid, name = full, "full grid"
    if a.quick:
        grid, name = quick, "reduced grid"
    elif e_full["wall_min"] > a.max_minutes and not a.yes:
        print(f"[experiments] the full grid exceeds {a.max_minutes:g} min: running the reduced grid first; "
              f"rerun with --yes for the full grid", flush=True)
        grid, name = quick, "reduced grid"
    if grid is quick:
        _print_estimate(name, estimate(quick, a.workers))
    out = os.path.join(a.out, f"experiments_{time.strftime('%Y%m%d-%H%M%S')}")
    out, rows = run_experiments(grid, out, workers=a.workers)
    _print_summary(rows, out)
    return 0


def _print_summary(rows: list[dict], out: str) -> None:
    table, pick, rule = summary_tables(rows)
    print(table)
    print(f"\n[experiments] recommended: spacing {pick['spacing_mm']:g} mm, overlap {pick['overlap_pct']:g} %, "
          f"orientations [{pick['orientations']}] ({rule})\n[experiments] {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
