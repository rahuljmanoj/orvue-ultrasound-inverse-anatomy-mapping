"""
orvue_us_inverse.mapping.run_experiments - run the sweep-strategy study headless (S5).

    python -m orvue_us_inverse.mapping.run_experiments [--workers 10] [--quick] [--yes] [--max-minutes 30]
    python -m orvue_us_inverse experiments [options]
    python -m orvue_us_inverse.mapping.run_experiments --resummarize output/results/experiments_<time>
    python -m orvue_us_inverse.mapping.run_experiments --errors [--quick] [--yes]     tracking-error study (S7)

Prints the estimated frames and runtime first. If the full grid is estimated above --max-minutes (wall clock with the
given workers), the reduced grid (experiments.quick_grid) runs instead and the full grid needs --yes. Results in
output/results/experiments_<time>/ (results.csv, plots, summary.md); sweeps cached in output/cache/experiments/.
--resummarize FOLDER rewrites summary.md and the plots of an existing folder from its results.csv (no runs).
--errors runs the tracking-error study instead (experiments.ErrorGrid: the default strategy with position jitter, yaw
jitter, latency and bias injected) -> output/results/errors_<time>/ (results_errors.csv, errors_*.png,
summary_errors.md with the tracking accuracy needed); same estimate / reduced-grid rule.
"""
import argparse
import os
import time

from orvue_us_inverse.mapping.experiments import (ErrorGrid, ExperimentGrid, error_tables, estimate, quick_error_grid,
                                                  quick_grid, resummarize, resummarize_errors, run_error_study,
                                                  run_experiments, summary_tables)
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
    p.add_argument("--errors", action="store_true", help="run the tracking-error study (S7)")
    a = p.parse_args(argv)
    if a.errors:
        return error_study(a)
    if a.resummarize:
        if os.path.exists(os.path.join(a.resummarize, "results_errors.csv")):
            rows = resummarize_errors(a.resummarize)
            _print_errors(rows, ErrorGrid(cases=tuple(dict.fromkeys(r["case"] for r in rows))), a.resummarize)
            return 0
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


def error_study(a) -> int:
    full, quick = ErrorGrid(), quick_error_grid()
    e_full = estimate(full, a.workers)
    _print_estimate("error grid", e_full)
    grid = full
    if a.quick or (e_full["wall_min"] > a.max_minutes and not a.yes):
        if not a.quick:
            print(f"[experiments] the error grid exceeds {a.max_minutes:g} min: running the reduced grid first; rerun "
                  f"with --yes for the full grid", flush=True)
        grid = quick
        _print_estimate("reduced error grid", estimate(quick, a.workers))
    out = os.path.join(a.out, f"errors_{time.strftime('%Y%m%d-%H%M%S')}")
    out, rows = run_error_study(grid, out, workers=a.workers)
    _print_errors(rows, grid, out)
    return 0


def _print_errors(rows: list[dict], grid, out: str) -> None:
    table, tol = error_tables(rows, grid)
    print(table)
    box = tol["robust_box"]
    print("\n[experiments] robust tracking requirement (every combination within passes): " +
          (", ".join(f"{k} <= {v:g}" for k, v in box.items() if k != "combinations") if box else "none") +
          "\n[experiments] one factor at a time: " +
          ", ".join(f"{k} <= {v:g}" if v is not None else f"{k}: fails at the smallest level"
                    for k, v in tol["per_factor"].items()) + f"\n[experiments] {out}", flush=True)


def _print_summary(rows: list[dict], out: str) -> None:
    table, pick, rule = summary_tables(rows)
    print(table)
    print(f"\n[experiments] recommended: spacing {pick['spacing_mm']:g} mm, overlap {pick['overlap_pct']:g} %, "
          f"orientations [{pick['orientations']}] ({rule})\n[experiments] {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
