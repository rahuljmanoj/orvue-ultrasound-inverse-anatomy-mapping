"""
orvue_us_inverse.mapping.experiments - headless parameter study of the sweep strategy (S5).

    grid = ExperimentGrid()                                   # spacing x overlap x orientations x cases (+ extras)
    print(estimate(grid, workers=10))
    out, rows = run_experiments(grid, out_dir, workers=10)    # results.csv, plots, summary.md

Sweeps: one oracle-label sweep per (case, yaw, spacing, overlap) (store_images False: labels_image only), cached in
output/cache/experiments/ and reused: an orientation set [0, 90] is the frames of the [0] and [90] sweeps (the frames
of a lane do not depend on the other lanes; transitions are never recorded), and every voxel size reuses the same
sweeps. Simulator always from mapping.probe.make_simulator().

Runs: reconstruct (mapping.recon, oracle labels, exact poses), fill gaps up to round(spacing / voxel) - 1 voxels
(at least 1) so sparse sweeps are interpolated between frames, evaluate (mapping.evaluate) and add per-structure local
Dice / MSD / HD95: the structure's effective voxels against the reconstructed voxels of its label within 2 voxels of
it, excluding other structures of the same label (e.g. the GB lumen next to the cystic duct). Sweeps and runs are
spread over worker processes (concurrent.futures, spawn); no windows.

Extras for the default strategy (spacing 0.5 mm, 20 % overlap, orientations [0] and [0, 90]): voxel 0.25 mm, and the
grid shifted by half a voxel in x and y (grid_offset_mm = -voxel / 2) so frames at multiples of 0.5 mm sample voxel
centres instead of voxel faces (S4 observation).

Error study (S7, ErrorGrid / run_error_study): the default strategy (SweepConfig: 0.25 mm, 20 %, yaw 0) with pose
errors injected (errors.PoseErrorModel: position jitter, yaw jitter, latency at the sweep speed of 10 mm/s via the
planned path, bias along x): frames rendered at the true pose, reconstructed at the erroneous one. Output
results_errors.csv, errors_dice.png, errors_hd95.png, errors_topology.png and summary_errors.md with the tracking
accuracy needed to keep the cystic duct and the CBD reconstructed (detected, local Dice >= NEED_DICE) and connected
(GB - CBD topology ok) in every case.
"""
import csv
import dataclasses
import functools
import math
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from orvue_us_inverse.mapping.acquisition import Acquirer
from orvue_us_inverse.mapping.config import AcquisitionConfig, GridConfig, SweepConfig
from orvue_us_inverse.mapping.errors import PoseErrorModel
from orvue_us_inverse.mapping.evaluate import CLASSES, evaluate, instance_masks, surface_distances
from orvue_us_inverse.mapping.poses import ScriptedSweep
from orvue_us_inverse.mapping.probe import make_simulator
from orvue_us_inverse.mapping.recon import LabelCompounder, VoxelGrid, fill_small_holes, ground_truth
from orvue_us_inverse.mapping.sweep_io import Sweep
from orvue_us_inverse.paths import EXPERIMENTS_CACHE_DIR
from orvue_us_inverse.simulation.anatomy import build_case

EXPERIMENTS_CACHE = EXPERIMENTS_CACHE_DIR
ERROR_KEYS = ("jitter_mm", "jitter_yaw_deg", "latency_ms", "bias_x_mm")
NEED_DICE = 0.5                        # error study: local Dice the cystic duct and the CBD must keep
SPEED_MM_S = 10.0
# runtime model (Prep / S2 / S4 timings on this laptop): labels_image, insert per frame; evaluation per run
LABELS_S, INSERT_S, EVAL_S = 0.0134, 0.0095, 4.0
EVAL_S_FINE = 30.0                     # 0.25 mm voxels (8x the voxels)
LUMEN_CLASSES = ("bile", "arterial blood", "venous blood", "stone", "lymph node")
WALL_CLASSES = ("GB wall", "duct wall", "artery wall", "vein wall")
THIN = ("cystic_duct", "cystic_artery", "cystic_artery_superficial", "cystic_artery_deep")
INSERT_CHUNK = 40                      # frames per insert_batch (memory)
LOCAL_DILATE = 2                       # voxels around a structure for its local metrics


def _key(x: float) -> str:
    return f"{x:g}"


@dataclass(frozen=True)
class RunSpec:
    case: str
    spacing_mm: float
    overlap_pct: float
    yaws: tuple[float, ...]
    voxel_mm: float = 0.5
    grid_offset_mm: float = 0.0
    errors: tuple = ()                 # sorted ((PoseErrorModel field, value), ...); () = exact poses

    @property
    def orientations(self) -> str:
        return "+".join(_key(y) for y in self.yaws)

    @property
    def strategy(self) -> str:
        return f"sp{_key(self.spacing_mm)}_ov{_key(self.overlap_pct)}_yaw{self.orientations}"

    def error_model(self) -> PoseErrorModel | None:
        """The run's PoseErrorModel (seeded from the run, reproducible), or None without errors."""
        if not self.errors:
            return None
        seed = zlib.crc32(repr((self.case, self.strategy, self.errors)).encode())
        return PoseErrorModel(**dict(self.errors), seed=seed)

    def sweep_keys(self) -> list[tuple[str, float, float, float]]:
        return [(self.case, y, self.spacing_mm, self.overlap_pct) for y in self.yaws]


@dataclass
class ExperimentGrid:
    spacings: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    overlaps: tuple[float, ...] = (20.0, 40.0, 50.0)
    orientation_sets: tuple[tuple[float, ...], ...] = ((0.0,), (90.0,), (0.0, 90.0), (0.0, 45.0, 90.0, 135.0))
    voxels: tuple[float, ...] = (0.5,)
    cases: tuple[str, ...] = ("normal", "parallel_cystic_duct", "anterior_cystic_artery")
    default_spacing: float = 0.5
    default_overlap: float = 20.0
    default_orientation_sets: tuple[tuple[float, ...], ...] = ((0.0,), (0.0, 90.0))
    default_extra_voxels: tuple[float, ...] = (0.25,)
    offset_check: bool = True          # default strategy on a grid shifted by half a voxel

    def runs(self) -> list[RunSpec]:
        out = [RunSpec(c, s, o, tuple(y), v) for c in self.cases for s in self.spacings for o in self.overlaps
               for y in self.orientation_sets for v in self.voxels]
        for c in self.cases:
            for y in self.default_orientation_sets:
                for v in self.default_extra_voxels:
                    out.append(RunSpec(c, self.default_spacing, self.default_overlap, tuple(y), v))
                if self.offset_check:
                    v = self.voxels[0]
                    out.append(RunSpec(c, self.default_spacing, self.default_overlap, tuple(y), v, -v / 2))
        return list(dict.fromkeys(out))

    def sweeps(self) -> list[tuple[str, float, float, float]]:
        return sorted({k for r in self.runs() for k in r.sweep_keys()})


def quick_grid() -> ExperimentGrid:
    """Reduced grid (normal case, 6 runs: spacing 0.5 / 2 mm x yaw [0] / [0, 90], plus the half-voxel grid offset)."""
    return ExperimentGrid(spacings=(0.5, 2.0), overlaps=(20.0,), orientation_sets=((0.0,), (0.0, 90.0)),
                          cases=("normal",), default_extra_voxels=(), offset_check=True)


# ---------------------------------------------------------------- sweeps
def sweep_config(yaws, spacing: float, overlap: float) -> SweepConfig:
    return SweepConfig(yaw_list_deg=list(yaws), overlap_pct=overlap, frame_spacing_mm=spacing, speed_mm_s=SPEED_MM_S)


def sweep_path(key, cache_dir: str = EXPERIMENTS_CACHE) -> str:
    case, yaw, spacing, overlap = key
    return os.path.join(cache_dir, f"{case}_yaw{_key(yaw)}_sp{_key(spacing)}_ov{_key(overlap)}.npz")


def acquire_sweep(key, cache_dir: str = EXPERIMENTS_CACHE) -> tuple[str, int, float]:
    """Oracle-label sweep for (case, yaw, spacing, overlap), cached; returns (path, frames, seconds)."""
    path = sweep_path(key, cache_dir)
    if os.path.exists(path):
        return path, -1, 0.0
    t0 = time.perf_counter()
    case, yaw, spacing, overlap = key
    cfg = sweep_config([yaw], spacing, overlap)
    acq = Acquirer(make_simulator(case), cfg, AcquisitionConfig(store_images=False), experiment_key=list(key))
    for s in ScriptedSweep(cfg).samples():
        acq.feed(s.T, s.t, s.recording)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path[:-4] + f".tmp{os.getpid()}.npz"
    acq.sweep.save(tmp)
    os.replace(tmp, path)
    return path, len(acq.sweep), time.perf_counter() - t0


# ---------------------------------------------------------------- runs
@functools.lru_cache(maxsize=4)
def _truth(case: str, voxel_mm: float, offset_mm: float):
    """(anatomy, grid, ground truth, instances) per worker, cached."""
    an = build_case(case)
    grid = VoxelGrid(GridConfig(x_mm=(offset_mm, 100.0 + offset_mm), y_mm=(offset_mm, 100.0 + offset_mm),
                                voxel_mm=voxel_mm))
    return an, grid, ground_truth(an, grid), instance_masks(an, grid)


def structure_local_metrics(labels: np.ndarray, gt: np.ndarray, instances: dict, mask: np.ndarray,
                            voxel_mm: float) -> dict[str, dict]:
    """Per structure: Dice, MSD, HD95 between its effective voxels and the reconstructed voxels of its label within
    LOCAL_DILATE voxels of it (other structures of the same label excluded, except voxels the structure shares with
    them), inside the evaluation mask."""
    eff = {}
    for name, parts in instances.items():
        main = parts.get("lumen") or parts["interior"]
        eff[name] = (main.label, main.flat[gt.ravel()[main.flat] == main.label])
    out = {}
    shape = gt.shape
    for name, (label, e) in eff.items():
        if len(e) == 0:
            out[name] = dict(dice=None, msd_mm=None, hd95_mm=None)
            continue
        ijk = np.stack(np.unravel_index(e, shape), 1)
        lo = np.maximum(ijk.min(0) - LOCAL_DILATE - 1, 0)
        hi = np.minimum(ijk.max(0) + LOCAL_DILATE + 2, shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        sub_shape = tuple(hi - lo)
        own = np.zeros(sub_shape, bool)
        own[tuple((ijk - lo).T)] = True
        other = np.zeros(sub_shape, bool)
        for oname, (olab, oe) in eff.items():
            if oname == name or olab != label or len(oe) == 0:
                continue
            o = np.stack(np.unravel_index(oe, shape), 1) - lo
            keep = np.all((o >= 0) & (o < np.array(sub_shape)), axis=1)
            other[tuple(o[keep].T)] = True
        other &= ~own                    # voxels shared with a same-label neighbour (e.g. duct / GB neck) stay own
        region = ndimage.binary_dilation(own, iterations=LOCAL_DILATE)
        m = mask[sl]
        a = (labels[sl] == label) & region & m & ~other
        b = own & m
        na, nb, inter = int(a.sum()), int(b.sum()), int((a & b).sum())
        msd, hd95 = surface_distances(a, b, voxel_mm)
        out[name] = dict(dice=None if na + nb == 0 else 2 * inter / (na + nb), msd_mm=msd, hd95_mm=hd95)
    return out


def run_one(spec: RunSpec, cache_dir: str = EXPERIMENTS_CACHE) -> dict:
    """Reconstruct, fill and evaluate one run; returns one CSV row (dict)."""
    t0 = time.perf_counter()
    an, grid, gt, inst = _truth(spec.case, spec.voxel_mm, spec.grid_offset_mm)
    comp = LabelCompounder(grid)
    n_frames = 0
    model = spec.error_model()
    for key in spec.sweep_keys():
        sw = Sweep.load(sweep_path(key, cache_dir))
        comp.probe = dict(sw.metadata["probe"])
        frames = sw.frames
        if model is not None:                            # rendered at T_true, reconstructed at T_measured
            plan1 = ScriptedSweep(sweep_config([key[1]], key[2], key[3]))
            frames = model.apply_frames(frames, pose_at=lambda t, p=plan1: p.pose_at(t).T)
        for i in range(0, len(frames), INSERT_CHUNK):
            comp.insert_batch(frames[i:i + INSERT_CHUNK])
        n_frames += len(sw)
    raw = comp.result()
    max_gap = max(1, round(spec.spacing_mm / spec.voxel_mm) - 1)
    labels = fill_small_holes(raw, max_gap)
    del comp
    ev = evaluate(labels, an, grid, gt=gt, instances=inst, case=spec.case)
    r = ev.report
    local = structure_local_metrics(labels, gt, inst, labels >= 0, grid.voxel_mm)
    plan = ScriptedSweep(sweep_config(spec.yaws, spec.spacing_mm, spec.overlap_pct))
    err = dict(spec.errors)
    row = dict(case=spec.case, strategy=spec.strategy, spacing_mm=spec.spacing_mm, overlap_pct=spec.overlap_pct,
               **{k: float(err.get(k, 0.0)) for k in ERROR_KEYS},
               overlap_actual_pct="/".join(f"{plan.overlap_pct[y]:.1f}" for y in spec.yaws),
               orientations=spec.orientations, n_orientations=len(spec.yaws), voxel_mm=spec.voxel_mm,
               grid_offset_mm=spec.grid_offset_mm, lanes=len(plan.lanes), frames=n_frames,
               scan_time_s=round(plan.duration_s, 2), fill_max_gap=max_gap,
               observed_raw=float((raw >= 0).mean()), observed=r["observed_fraction"],
               accuracy=r["accuracy_observed"], topology=r["topology"]["status"],
               detected=r["counts"]["detected"], missed=r["counts"]["missed"],
               not_covered=r["counts"]["not covered"], no_voxels=r["counts"]["no voxels"])
    for cname, c in r["classes"].items():
        k = cname.replace(" ", "_")
        row.update({f"{k}_dice": c["dice"], f"{k}_msd_mm": c["msd_mm"], f"{k}_hd95_mm": c["hd95_mm"]})
    for s in r["structures"]:
        n = s["name"]
        loc = local.get(n, {})
        row.update({f"{n}_coverage": s["coverage"], f"{n}_recall": s["recall"], f"{n}_dice": loc.get("dice"),
                    f"{n}_msd_mm": loc.get("msd_mm"), f"{n}_hd95_mm": loc.get("hd95_mm"), f"{n}_status": s["status"]})
    lumen = [r["classes"][c]["dice"] for c in LUMEN_CLASSES if r["classes"][c]["dice"] is not None]
    walls = [r["classes"][c]["dice"] for c in WALL_CLASSES if r["classes"][c]["dice"] is not None]
    thin = [s["recall"] or 0.0 for s in r["structures"] if s["name"] in THIN]
    hd = [r["classes"][c]["hd95_mm"] for c in LUMEN_CLASSES if r["classes"][c]["hd95_mm"] is not None]
    row.update(lumen_dice=float(np.mean(lumen)), wall_dice=float(np.mean(walls)), thin_recall=float(np.mean(thin)),
               min_recall=float(min(s["recall"] or 0.0 for s in r["structures"] if s["n_voxels"])),
               lumen_hd95_mm=float(np.mean(hd)), runtime_s=round(time.perf_counter() - t0, 2))
    return row


# ---------------------------------------------------------------- estimate and orchestration
def estimate(grid: ExperimentGrid, workers: int = 1, cache_dir: str = EXPERIMENTS_CACHE) -> dict:
    """Frames to acquire (sweeps not in the cache), frames to insert, runs, and runtime (serial and wall)."""
    frames = {}
    for key in grid.sweeps():
        case, yaw, sp, ov = key
        frames[key] = ScriptedSweep(sweep_config([yaw], sp, ov)).expected_frames()
    to_acquire = sum(n for k, n in frames.items() if not os.path.exists(sweep_path(k, cache_dir)))
    runs = grid.runs()
    inserts = sum(frames[k] for r in runs for k in r.sweep_keys())
    n_fine = sum(r.voxel_mm < 0.5 for r in runs)
    serial = to_acquire * LABELS_S + inserts * INSERT_S + (len(runs) - n_fine) * EVAL_S + n_fine * EVAL_S_FINE
    wall = serial / max(1, workers) * 1.15 + 20.0            # pool start-up and imbalance
    return dict(sweeps=len(frames), frames_total=sum(frames.values()), frames_to_acquire=to_acquire, runs=len(runs),
                frames_inserted=inserts, serial_min=serial / 60, wall_min=wall / 60, workers=workers)


def _pool_map(fn, items, workers: int, log=print, label="") -> list:
    out = []
    if workers <= 1:
        for i, it in enumerate(items, 1):
            out.append(fn(it))
            log(f"[experiments] {label} {i}/{len(items)}")
        return out
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for i, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if i % max(1, len(items) // 20) == 0 or i == len(items):
                log(f"[experiments] {label} {i}/{len(items)}")
    return out


def run_experiments(grid: ExperimentGrid, out_dir: str, workers: int = 8, cache_dir: str = EXPERIMENTS_CACHE,
                    fine_workers: int = 3, plots: bool = True, log=print) -> tuple[str, list[dict]]:
    """Acquire missing sweeps, run every RunSpec, write results.csv, plots and summary.md to out_dir; returns
    (out_dir, rows)."""
    t0 = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    keys = [k for k in grid.sweeps() if not os.path.exists(sweep_path(k, cache_dir))]
    _pool_map(functools.partial(acquire_sweep, cache_dir=cache_dir), keys, workers, log, "sweeps")
    t1 = time.perf_counter()
    runs = grid.runs()
    coarse = [r for r in runs if r.voxel_mm >= 0.5]
    fine = [r for r in runs if r.voxel_mm < 0.5]                # ~2.5 GB each: fewer at a time
    rows = _pool_map(functools.partial(run_one, cache_dir=cache_dir), coarse, workers, log, "runs")
    rows += _pool_map(functools.partial(run_one, cache_dir=cache_dir), fine, min(workers, fine_workers), log,
                      "runs (0.25 mm voxels)")
    order = {r: i for i, r in enumerate(runs)}
    rows.sort(key=lambda row: order[RunSpec(row["case"], row["spacing_mm"], row["overlap_pct"],
                                            tuple(float(y) for y in row["orientations"].split("+")),
                                            row["voxel_mm"], row["grid_offset_mm"])])
    write_csv(rows, os.path.join(out_dir, "results.csv"))
    timing = dict(sweeps_s=t1 - t0, runs_s=time.perf_counter() - t1, workers=workers, sweeps_acquired=len(keys))
    if plots:
        make_plots(rows, grid, out_dir)
    write_summary(rows, grid, out_dir, timing)
    log(f"[experiments] done in {(time.perf_counter() - t0) / 60:.1f} min -> {out_dir}")
    return out_dir, rows


# ---------------------------------------------------------------- outputs
SETTING_COLUMNS = ["case", "strategy", "spacing_mm", "overlap_pct", "overlap_actual_pct", "orientations",
                   "n_orientations", "voxel_mm", "grid_offset_mm", "lanes", "frames", "scan_time_s", "fill_max_gap",
                   "observed_raw", "observed", "accuracy", "topology", "detected", "missed", "not_covered",
                   "no_voxels", "lumen_dice", "wall_dice", "thin_recall", "min_recall", "lumen_hd95_mm", "runtime_s"]


def write_csv(rows: list[dict], path: str, first: list[str] | None = None) -> str:
    """One row per run; columns: first (if given), SETTING_COLUMNS, then every other key in order of appearance."""
    cols = list(first or []) + [c for c in SETTING_COLUMNS if c not in (first or [])]
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return path


def _main_rows(rows, **fixed) -> list[dict]:
    return [r for r in rows if r["voxel_mm"] == 0.5 and r["grid_offset_mm"] == 0.0
            and all(r[k] == v for k, v in fixed.items())]


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean over cases per strategy (0.5 mm voxels, no grid offset, and the extras listed separately)."""
    groups: dict = {}
    for r in rows:
        k = (r["spacing_mm"], r["overlap_pct"], r["orientations"], r["voxel_mm"], r["grid_offset_mm"])
        groups.setdefault(k, []).append(r)
    out = []
    for (sp, ov, orient, vox, off), rs in groups.items():
        out.append(dict(spacing_mm=sp, overlap_pct=ov, orientations=orient, voxel_mm=vox, grid_offset_mm=off,
                        cases=len(rs), frames=float(np.mean([r["frames"] for r in rs])),
                        scan_time_s=float(np.mean([r["scan_time_s"] for r in rs])),
                        lumen_dice=float(np.mean([r["lumen_dice"] for r in rs])),
                        wall_dice=float(np.mean([r["wall_dice"] for r in rs])),
                        thin_recall=float(np.mean([r["thin_recall"] for r in rs])),
                        min_recall=float(min(r["min_recall"] for r in rs)),
                        lumen_hd95_mm=float(np.mean([r["lumen_hd95_mm"] for r in rs])),
                        detected=sum(r["detected"] for r in rs), structures=sum(r["detected"] + r["missed"]
                                                                                + r["not_covered"] for r in rs),
                        topology_ok=sum(r["topology"] == "ok" for r in rs)))
    return out


def _eligible(agg: list[dict]) -> tuple[list[dict], list[dict]]:
    """(strategies at 0.5 mm voxels without grid offset, those detecting every structure with the right topology)."""
    main = [a for a in agg if a["voxel_mm"] == 0.5 and a["grid_offset_mm"] == 0.0]
    ok = [a for a in main if a["detected"] == a["structures"] and a["topology_ok"] == a["cases"]]
    return main, ok


def best_quality(agg: list[dict]) -> dict:
    """Highest mean lumen Dice (then thin-structure recall) among the eligible strategies."""
    main, ok = _eligible(agg)
    return max(ok or main, key=lambda a: (a["lumen_dice"], a["thin_recall"]))


def recommend(agg: list[dict], dice_tol: float = 0.01, thin_tol: float = 0.03) -> tuple[dict, str]:
    """Cheapest strategy (scan time at 10 mm/s, then fewer frames) at 0.5 mm voxels that detects every structure with
    the right topology in every case and is within dice_tol of the best lumen Dice and thin_tol of the best
    thin-structure recall (the knee of the quality / scan-time curve; tighter tolerances pick best_quality)."""
    main, ok = _eligible(agg)
    pool = ok or main
    best_dice = max(a["lumen_dice"] for a in pool)
    best_thin = max(a["thin_recall"] for a in pool)
    good = [a for a in pool if a["lumen_dice"] >= best_dice - dice_tol and a["thin_recall"] >= best_thin - thin_tol]
    pick = min(good, key=lambda a: (a["scan_time_s"], a["frames"]))       # scan time at 10 mm/s, then fewer frames
    rule = (f"cheapest scan among strategies that detect every structure with the correct GB - CBD topology in all "
            f"cases ({len(ok)} of {len(main)}) and are within {dice_tol:.3f} of the best mean lumen Dice "
            f"({best_dice:.3f}) and within {100 * thin_tol:.0f} points of the best thin-structure recall "
            f"({100 * best_thin:.1f}%)")
    return pick, rule


def _fmt_row(a: dict) -> str:
    return (f"| {_key(a['spacing_mm'])} | {_key(a['overlap_pct'])} | {a['orientations']} | {_key(a['voxel_mm'])}"
            f"{' (offset)' if a['grid_offset_mm'] else ''} | {a['frames']:.0f} | {a['scan_time_s']:.0f} | "
            f"{a['lumen_dice']:.3f} | {a['wall_dice']:.3f} | {100 * a['thin_recall']:.1f} | {100 * a['min_recall']:.1f} | "
            f"{a['lumen_hd95_mm']:.2f} | {a['detected']}/{a['structures']} | {a['topology_ok']}/{a['cases']} |")


TABLE_HEAD = ("| Spacing (mm) | Overlap (%) | Orientations | Voxel (mm) | Frames | Scan (s) | Lumen Dice | Wall Dice | "
              "Thin recall (%) | Min recall (%) | Lumen HD95 (mm) | Detected | Topology ok |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def summary_tables(rows: list[dict]) -> tuple[str, dict, str]:
    agg = aggregate(rows)
    main = sorted([a for a in agg if a["voxel_mm"] == 0.5 and a["grid_offset_mm"] == 0.0],
                  key=lambda a: (a["scan_time_s"]))
    extras = [a for a in agg if a["voxel_mm"] != 0.5 or a["grid_offset_mm"] != 0.0]
    pick, rule = recommend(agg)
    lines = [TABLE_HEAD] + [_fmt_row(a) for a in main]
    if extras:
        refs = [a for a in main if (a["spacing_mm"], a["overlap_pct"], a["orientations"])
                in {(e["spacing_mm"], e["overlap_pct"], e["orientations"]) for e in extras}]
        lines += ["", "Default strategy with 0.25 mm voxels and with the grid shifted by half a voxel (frames at voxel "
                      "centres), against the same strategy at 0.5 mm:", "", TABLE_HEAD]
        lines += [_fmt_row(a) for a in sorted(refs + extras, key=lambda a: (a["orientations"], -a["voxel_mm"],
                                                                             a["grid_offset_mm"]))]
    return "\n".join(lines), pick, rule


def write_summary(rows: list[dict], grid: ExperimentGrid, out_dir: str, timing: dict) -> str:
    table, pick, rule = summary_tables(rows)
    agg = {(_key(a["spacing_mm"]), _key(a["overlap_pct"]), a["orientations"], a["voxel_mm"], a["grid_offset_mm"]): a
           for a in aggregate(rows)}

    def a_(sp, ov, orient, vox=0.5, off=0.0):
        return agg.get((_key(sp), _key(ov), orient, vox, off))

    notes = []
    sp_line = [a_(s, grid.default_overlap, "0") for s in grid.spacings]
    if all(sp_line):
        notes.append("Spacing (20 % overlap, yaw 0): thin-structure recall " + ", ".join(
            f"{100 * a['thin_recall']:.1f}% at {_key(a['spacing_mm'])} mm" for a in sp_line) + "; lumen HD95 " +
            ", ".join(f"{a['lumen_hd95_mm']:.2f}" for a in sp_line) + " mm.")
    ov_line = [a_(grid.default_spacing, o, "0") for o in grid.overlaps]
    if all(ov_line):
        notes.append("Overlap (0.5 mm, yaw 0): lumen Dice " + ", ".join(
            f"{a['lumen_dice']:.3f} at {_key(a['overlap_pct'])} %" for a in ov_line) + "; scan time " +
            ", ".join(f"{a['scan_time_s']:.0f}" for a in ov_line) + " s (exact poses: overlap adds no information).")
    or_line = [a_(grid.default_spacing, grid.default_overlap, "+".join(_key(y) for y in o)) for o in
               grid.orientation_sets]
    if all(or_line):
        notes.append("Orientations (0.5 mm, 20 %): " + ", ".join(
            f"[{a['orientations']}] lumen Dice {a['lumen_dice']:.3f} / wall Dice {a['wall_dice']:.3f} / thin "
            f"{100 * a['thin_recall']:.1f}% in {a['scan_time_s']:.0f} s" for a in or_line) + ".")
    for orient in ("0", "0+90"):
        base, fine = a_(grid.default_spacing, grid.default_overlap, orient), None
        fine = a_(grid.default_spacing, grid.default_overlap, orient, 0.25)
        off = next((a for k, a in agg.items() if k[:3] == (_key(grid.default_spacing), _key(grid.default_overlap),
                                                             orient) and k[4] != 0.0), None)
        if base and (fine or off):
            parts = [f"0.5 mm grid: lumen Dice {base['lumen_dice']:.3f}, wall Dice {base['wall_dice']:.3f}"]
            if fine:
                parts.append(f"0.25 mm voxels: {fine['lumen_dice']:.3f} / {fine['wall_dice']:.3f}")
            if off:
                parts.append(f"0.5 mm grid shifted half a voxel: {off['lumen_dice']:.3f} / {off['wall_dice']:.3f}")
            notes.append(f"Default [{orient}]: " + "; ".join(parts) + ".")
    notes.append(f"Scan time is at a fixed {SPEED_MM_S:g} mm/s, so it does not depend on spacing; a scanner at "
                 f"f frames/s limits the speed to spacing x f (0.25 mm at 20 frames/s: 5 mm/s, twice the scan time of "
                 f"0.5 mm). The frame count is the cost of finer spacing here.")
    best = best_quality(aggregate(rows))

    def desc(a):
        return (f"spacing {_key(a['spacing_mm'])} mm, {_key(a['overlap_pct'])} % overlap, orientations "
                f"[{a['orientations']}], 0.5 mm voxels: {a['frames']:.0f} frames, {a['scan_time_s']:.0f} s at "
                f"{SPEED_MM_S:g} mm/s; lumen Dice {a['lumen_dice']:.3f}, wall Dice {a['wall_dice']:.3f}, thin-structure "
                f"recall {100 * a['thin_recall']:.1f}%, worst structure recall {100 * a['min_recall']:.1f}%, detected "
                f"{a['detected']}/{a['structures']}, topology ok {a['topology_ok']}/{a['cases']}")

    timing_text = (f" Sweeps {timing['sweeps_s'] / 60:.1f} min ({timing['sweeps_acquired']} acquired), runs "
                   f"{timing['runs_s'] / 60:.1f} min with {timing['workers']} workers." if timing else "")
    md = [f"# Sweep-strategy experiments", "",
          f"Cases {', '.join(grid.cases)}; oracle labels, exact poses; holes filled up to round(spacing / voxel) - 1 "
          f"voxels; metrics inside the observed voxels, means over the cases (thin = cystic duct and the cystic "
          f"artery branches; min recall = worst structure in any case). {len(rows)} runs.{timing_text}", "",
          "## Results (sorted by scan time)", "", table, "",
          "## Recommendation", "",
          f"**Recommended default: {desc(pick)}.**", "", f"Rule: {rule}.", "",
          f"Best quality (for comparison): {desc(best)}.", "", "Observations:", ""]
    md += [f"- {n}" for n in notes]
    md += ["", "Plots: dice_vs_spacing.png, hd95_vs_spacing.png, vs_overlap.png, vs_orientation.png, "
               "time_vs_quality.png; every run in results.csv.", ""]
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return path


TEXT_COLUMNS = {"case", "strategy", "orientations", "overlap_actual_pct", "topology"}


def load_results(path: str) -> list[dict]:
    """results.csv back as rows with numbers parsed (text columns and *_status kept as text, empty -> None)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row = {}
            for k, v in r.items():
                if v == "":
                    row[k] = None
                elif k in TEXT_COLUMNS or k.endswith("_status"):
                    row[k] = v
                else:
                    f = float(v)
                    row[k] = int(f) if f.is_integer() and k in ("frames", "lanes", "fill_max_gap", "n_orientations",
                                                                 "detected", "missed", "not_covered", "no_voxels") else f
            rows.append(row)
    return rows


def resummarize(out_dir: str, grid: ExperimentGrid | None = None, plots: bool = True) -> list[dict]:
    """Rewrite summary.md (and the plots) of an experiments folder from its results.csv."""
    rows = load_results(os.path.join(out_dir, "results.csv"))
    grid = grid or ExperimentGrid()
    if plots:
        make_plots(rows, grid, out_dir)
    write_summary(rows, grid, out_dir, {})
    return rows


def _structure_names(rows) -> list[str]:
    names = []
    for k in rows[0]:
        if k.endswith("_status"):
            names.append(k[:-len("_status")])
    return names


def make_plots(rows: list[dict], grid: ExperimentGrid, out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    names = _structure_names(rows)
    cmap = matplotlib.colormaps["tab20"]
    paths = []

    def per_case(xkey, xs, fixed, metric, ylabel, fname, categorical=False):
        fig, axes = plt.subplots(1, len(grid.cases), figsize=(5.2 * len(grid.cases), 4.6), sharey=True, squeeze=False)
        for ax, case in zip(axes[0], grid.cases):
            sel = [r for r in _main_rows(rows, case=case, **fixed)]
            for i, n in enumerate(names):
                pts = [(x, next((r[f"{n}_{metric}"] for r in sel if r[xkey] == x), None)) for x in xs]
                pts = [(j if categorical else x, y) for j, (x, y) in enumerate(pts) if y not in (None, "")]
                if pts:
                    ax.plot(*zip(*pts), "o-", ms=3, lw=1, color=cmap(i % 20), label=n)
            ax.set_title(case, fontsize=9)
            ax.set_xlabel(xkey)
            if categorical:
                ax.set_xticks(range(len(xs)), xs, fontsize=8)
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel(ylabel)
        if axes[0][-1].get_legend_handles_labels()[0]:
            axes[0][-1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.suptitle(f"{ylabel} per structure ({', '.join(f'{k} {v}' for k, v in fixed.items())})", fontsize=10)
        fig.tight_layout()
        p = os.path.join(out_dir, fname)
        fig.savefig(p, dpi=100)
        plt.close(fig)
        paths.append(p)

    d_or, d_ov, d_sp = "0", grid.default_overlap, grid.default_spacing
    per_case("spacing_mm", list(grid.spacings), dict(overlap_pct=d_ov, orientations=d_or), "dice", "Dice",
             "dice_vs_spacing.png")
    per_case("spacing_mm", list(grid.spacings), dict(overlap_pct=d_ov, orientations=d_or), "hd95_mm", "HD95 (mm)",
             "hd95_vs_spacing.png")
    per_case("overlap_pct", list(grid.overlaps), dict(spacing_mm=d_sp, orientations=d_or), "dice", "Dice",
             "vs_overlap.png")
    orients = ["+".join(_key(y) for y in o) for o in grid.orientation_sets]
    per_case("orientations", orients, dict(spacing_mm=d_sp, overlap_pct=d_ov), "dice", "Dice", "vs_orientation.png",
             categorical=True)

    agg = [a for a in aggregate(rows) if a["voxel_mm"] == 0.5 and a["grid_offset_mm"] == 0.0]
    pick, _ = recommend(aggregate(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, label in ((axes[0], "thin_recall", "thin-structure recall"), (axes[1], "lumen_dice", "lumen Dice")):
        for j, o in enumerate(orients):
            pts = [a for a in agg if a["orientations"] == o]
            ax.scatter([a["scan_time_s"] for a in pts], [a[metric] for a in pts], s=[30 + 40 * (2.0 / a["spacing_mm"])
                       ** 0.5 for a in pts], color=cmap(2 * j), alpha=0.75, label=f"[{o}]")
        ax.scatter([pick["scan_time_s"]], [pick[metric]], s=260, facecolors="none", edgecolors="#E24B4A", lw=1.5,
                   label="recommended")
        ax.set_xscale("log")
        ax.set_xlabel("scan time at 10 mm/s (s, log)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, title="orientations (marker size: finer spacing)", title_fontsize=7)
    fig.suptitle("Scan time vs quality (mean over cases, 0.5 mm voxels)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, "time_vs_quality.png")
    fig.savefig(p, dpi=100)
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------- error study (S7)
@dataclass
class ErrorGrid:
    """Default strategy x pose errors (full factorial) x cases."""
    jitter_mm: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    jitter_yaw_deg: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    latency_ms: tuple[float, ...] = (0.0, 33.0, 100.0)
    bias_x_mm: tuple[float, ...] = (0.0, 1.0)
    cases: tuple[str, ...] = ("normal", "parallel_cystic_duct", "anterior_cystic_artery")
    spacing_mm: float = SweepConfig().frame_spacing_mm
    overlap_pct: float = SweepConfig().overlap_pct
    yaws: tuple[float, ...] = tuple(SweepConfig().yaw_list_deg)

    def runs(self) -> list[RunSpec]:
        out = []
        for c in self.cases:
            for j in self.jitter_mm:
                for y in self.jitter_yaw_deg:
                    for lat in self.latency_ms:
                        for b in self.bias_x_mm:
                            err = tuple(sorted((k, float(v)) for k, v in zip(ERROR_KEYS, (j, y, lat, b)) if v))
                            out.append(RunSpec(c, self.spacing_mm, self.overlap_pct, self.yaws, errors=err))
        return out

    def sweeps(self) -> list[tuple[str, float, float, float]]:
        return sorted({k for r in self.runs() for k in r.sweep_keys()})


def quick_error_grid() -> ErrorGrid:
    """Reduced error grid (normal case, 8 runs)."""
    return ErrorGrid(jitter_mm=(0.0, 1.0), jitter_yaw_deg=(0.0, 1.0), latency_ms=(0.0, 100.0), bias_x_mm=(0.0,),
                     cases=("normal",))


def needs_met(row: dict) -> bool:
    """Cystic duct and CBD detected with local Dice >= NEED_DICE, and the GB - CBD connection as in the truth."""
    ok = row.get("topology") == "ok"
    for n in ("cystic_duct", "chd_cbd"):
        ok = ok and row.get(f"{n}_status") == "detected" and (row.get(f"{n}_dice") or 0.0) >= NEED_DICE
    return bool(ok)


def error_aggregate(rows: list[dict]) -> list[dict]:
    """Per error combination, over the cases: all cases meet the needs, mean Dice / HD95 of the duct and the CBD,
    topology ok count."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in ERROR_KEYS), []).append(r)
    out = []
    for key, rs in groups.items():
        a = dict(zip(ERROR_KEYS, key), cases=len(rs), pass_all=all(needs_met(r) for r in rs),
                 topology_ok=sum(r["topology"] == "ok" for r in rs), lumen_dice=float(np.mean([r["lumen_dice"] for r in rs])))
        for n in ("cystic_duct", "chd_cbd"):
            a[f"{n}_dice"] = float(np.mean([r.get(f"{n}_dice") or 0.0 for r in rs]))
            hd = [r.get(f"{n}_hd95_mm") for r in rs if r.get(f"{n}_hd95_mm") is not None]
            a[f"{n}_hd95_mm"] = float(np.mean(hd)) if hd else None
            a[f"{n}_detected"] = sum(r.get(f"{n}_status") == "detected" for r in rs)
        out.append(a)
    return sorted(out, key=lambda a: tuple(a[k] for k in ERROR_KEYS))


def tolerances(agg: list[dict], grid: ErrorGrid) -> dict:
    """Per error factor (the others at 0): the largest level up to which every level meets the needs in every case;
    and whether the combination of those levels (if in the grid) still does."""
    by_key = {tuple(a[k] for k in ERROR_KEYS): a for a in agg}
    levels = dict(zip(ERROR_KEYS, (grid.jitter_mm, grid.jitter_yaw_deg, grid.latency_ms, grid.bias_x_mm)))
    tol = {}
    for i, k in enumerate(ERROR_KEYS):
        best = None
        for lv in sorted(levels[k]):
            key = tuple(lv if j == i else 0.0 for j in range(len(ERROR_KEYS)))
            a = by_key.get(key)
            if a is None or not a["pass_all"]:
                break
            best = lv
        tol[k] = best
    combo = tuple((tol[k] if tol[k] is not None else 0.0) for k in ERROR_KEYS)
    a = by_key.get(combo)
    return dict(per_factor=tol, combination=dict(zip(ERROR_KEYS, combo)),
                combination_passes=None if a is None else a["pass_all"], robust_box=robust_box(agg, grid))


def robust_box(agg: list[dict], grid: "ErrorGrid") -> dict | None:
    """The largest box of error levels (every factor from 0 up to a limit) in which EVERY combination meets the needs
    in every case: {factor: limit, 'combinations': n}; ties broken towards the larger position jitter. None when even
    the error-free run fails."""
    import itertools
    by_key = {tuple(a[k] for k in ERROR_KEYS): a for a in agg}
    levels = [sorted(v) for v in (grid.jitter_mm, grid.jitter_yaw_deg, grid.latency_ms, grid.bias_x_mm)]
    best = None
    for idx in itertools.product(*[range(len(lv)) for lv in levels]):
        box = [lv[:i + 1] for lv, i in zip(levels, idx)]
        combos = list(itertools.product(*box))
        if all(by_key.get(c, {}).get("pass_all", False) for c in combos):
            score = (len(combos), idx[0])
            if best is None or score > best[0]:
                best = (score, {k: lv[i] for k, lv, i in zip(ERROR_KEYS, levels, idx)} | {"combinations": len(combos)})
    return None if best is None else best[1]


def run_error_study(grid: ErrorGrid, out_dir: str, workers: int = 8, cache_dir: str = EXPERIMENTS_CACHE,
                    plots: bool = True, log=print) -> tuple[str, list[dict]]:
    """Acquire missing sweeps, run every error combination, write results_errors.csv, plots and summary_errors.md."""
    t0 = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    keys = [k for k in grid.sweeps() if not os.path.exists(sweep_path(k, cache_dir))]
    _pool_map(functools.partial(acquire_sweep, cache_dir=cache_dir), keys, workers, log, "sweeps")
    t1 = time.perf_counter()
    runs = grid.runs()
    rows = _pool_map(functools.partial(run_one, cache_dir=cache_dir), runs, workers, log, "error runs")
    order = {(r.case, r.errors): i for i, r in enumerate(runs)}
    rows.sort(key=lambda row: order[(row["case"], tuple(sorted((k, row[k]) for k in ERROR_KEYS if row[k])))])
    write_csv(rows, os.path.join(out_dir, "results_errors.csv"), first=list(ERROR_KEYS))
    timing = dict(sweeps_s=t1 - t0, runs_s=time.perf_counter() - t1, workers=workers, sweeps_acquired=len(keys))
    if plots:
        make_error_plots(rows, grid, out_dir)
    write_error_summary(rows, grid, out_dir, timing)
    log(f"[experiments] error study done in {(time.perf_counter() - t0) / 60:.1f} min -> {out_dir}")
    return out_dir, rows


def error_tables(rows: list[dict], grid: ErrorGrid) -> tuple[str, dict]:
    agg = error_aggregate(rows)
    tol = tolerances(agg, grid)
    head = ("| Jitter (mm) | Yaw jitter (deg) | Latency (ms) | Bias x (mm) | Cystic duct Dice | CBD Dice | "
            "Cystic duct HD95 (mm) | CBD HD95 (mm) | Topology ok | Needs met |\n|---|---|---|---|---|---|---|---|---|---|")

    def fmt(a):
        h = [f"{a[n]:.2f}" if a[n] is not None else "-" for n in ("cystic_duct_hd95_mm", "chd_cbd_hd95_mm")]
        return (f"| {_key(a['jitter_mm'])} | {_key(a['jitter_yaw_deg'])} | {_key(a['latency_ms'])} | "
                f"{_key(a['bias_x_mm'])} | {a['cystic_duct_dice']:.3f} | {a['chd_cbd_dice']:.3f} | {h[0]} | {h[1]} | "
                f"{a['topology_ok']}/{a['cases']} | {'yes' if a['pass_all'] else 'NO'} |")
    one = [a for a in agg if sum(a[k] != 0 for k in ERROR_KEYS) <= 1]
    lines = ["One factor at a time (the others 0):", "", head] + [fmt(a) for a in one]
    lines += ["", f"All {len(agg)} combinations: needs met in every case for {sum(a['pass_all'] for a in agg)}.", "",
              head] + [fmt(a) for a in agg]
    return "\n".join(lines), tol


def write_error_summary(rows: list[dict], grid: ErrorGrid, out_dir: str, timing: dict) -> str:
    from orvue_us_inverse.mapping.errors import filter_lag
    table, tol = error_tables(rows, grid)
    t = tol["per_factor"]

    def lvl(k, unit):
        return "below the smallest tested level" if t[k] is None else f"<= {_key(t[k])} {unit}"
    lag10 = filter_lag(10.0)
    comb = tol["combination"]
    box = tol["robust_box"]
    comb_txt = ("passes" if tol["combination_passes"] else "does NOT pass" if tol["combination_passes"] is not None
                else "is not in the grid")
    md = ["# Tracking-error study", "",
          f"Default strategy (spacing {_key(grid.spacing_mm)} mm, {_key(grid.overlap_pct)} % overlap, yaw "
          f"{'+'.join(_key(y) for y in grid.yaws)}), cases {', '.join(grid.cases)}; oracle labels rendered at the true "
          f"pose and reconstructed at the pose with the injected error (errors.PoseErrorModel); latency at the sweep "
          f"speed of {SPEED_MM_S:g} mm/s (100 ms = 1 mm along the sweep). Needs: the cystic duct and the CBD detected "
          f"with local Dice >= {NEED_DICE} and the GB - CBD connection kept, in every case. {len(rows)} runs" +
          (f"; sweeps {timing['sweeps_s'] / 60:.1f} min, runs {timing['runs_s'] / 60:.1f} min with "
           f"{timing['workers']} workers." if timing else "."), "",
          "## Tracking accuracy needed", "",
          ("**Robust requirement** (every combination of errors up to these limits keeps both structures "
           f"reconstructed and connected in every case, {box['combinations']} combinations): position jitter <= "
           f"{_key(box['jitter_mm'])} mm (1 sigma per axis), yaw jitter <= {_key(box['jitter_yaw_deg'])} deg, latency "
           f"<= {_key(box['latency_ms'])} ms at {SPEED_MM_S:g} mm/s, bias <= {_key(box['bias_x_mm'])} mm."
           if box else "**No robust requirement found**: even the error-free run fails the needs."), "",
          "One factor at a time (the others 0), for comparison - optimistic, because errors combine:", "",
          f"- Position jitter (1 sigma, per axis): {lvl('jitter_mm', 'mm')}",
          f"- Yaw jitter (1 sigma): {lvl('jitter_yaw_deg', 'deg')}",
          f"- Latency at {SPEED_MM_S:g} mm/s: {lvl('latency_ms', 'ms')}",
          f"- Constant position bias: {lvl('bias_x_mm', 'mm')}",
          f"- These four limits together ({', '.join(f'{k} {_key(v)}' for k, v in comb.items())}): {comb_txt} "
          f"(one draw of the jitter; see the full table for the neighbouring combinations).", "",
          f"For reference: the tracker's one-euro filter alone lags {lag10['lag_ms']:.0f} ms at 10 mm/s "
          f"({lag10['lag_mm']:.2f} mm); the camera latency adds to it. A bias moves the whole reconstruction rigidly "
          f"(shapes and connections survive; the S4 metrics count it as a shift), whereas jitter and latency differences "
          f"between lanes break thin structures apart.", "", "## Results (means over the cases)", "", table, "",
          "Plots: errors_dice.png, errors_hd95.png, errors_topology.png; every run in results_errors.csv.", ""]
    path = os.path.join(out_dir, "summary_errors.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return path


def resummarize_errors(out_dir: str, grid: ErrorGrid | None = None, plots: bool = True) -> list[dict]:
    """Rewrite summary_errors.md (and the plots) of an error-study folder from its results_errors.csv."""
    rows = load_results(os.path.join(out_dir, "results_errors.csv"))
    grid = grid or ErrorGrid(cases=tuple(dict.fromkeys(r["case"] for r in rows)))
    if plots:
        make_error_plots(rows, grid, out_dir)
    write_error_summary(rows, grid, out_dir, {})
    return rows


def make_error_plots(rows: list[dict], grid: ErrorGrid, out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    agg = error_aggregate(rows)

    def pick(**fixed):
        return [a for a in agg if all(a[k] == v for k, v in fixed.items())]
    cmap = matplotlib.colormaps["viridis"]
    paths = []
    for metric, label, fname in (("dice", "local Dice", "errors_dice.png"), ("hd95_mm", "HD95 (mm)", "errors_hd95.png")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        ax = axes[0]
        for i, y in enumerate(grid.jitter_yaw_deg):
            pts = sorted(pick(jitter_yaw_deg=y, latency_ms=0.0, bias_x_mm=0.0), key=lambda a: a["jitter_mm"])
            c = cmap(i / max(1, len(grid.jitter_yaw_deg) - 1))
            for n, ls in (("cystic_duct", "-"), ("chd_cbd", "--")):
                vals = [(a["jitter_mm"], a[f"{n}_{metric}"]) for a in pts if a[f"{n}_{metric}"] is not None]
                if vals:
                    ax.plot(*zip(*vals), ls, marker="o", color=c, label=f"{'duct' if n == 'cystic_duct' else 'CBD'}, "
                                                                        f"yaw {_key(y)} deg")
        ax.set_xlabel("position jitter sigma (mm)")
        ax.set_ylabel(label)
        ax.set_title("vs jitter (latency 0, bias 0)", fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        ax = axes[1]
        for i, b in enumerate(grid.bias_x_mm):
            pts = sorted(pick(jitter_mm=0.0, jitter_yaw_deg=0.0, bias_x_mm=b), key=lambda a: a["latency_ms"])
            for n, ls in (("cystic_duct", "-"), ("chd_cbd", "--")):
                vals = [(a["latency_ms"], a[f"{n}_{metric}"]) for a in pts if a[f"{n}_{metric}"] is not None]
                if vals:
                    ax.plot(*zip(*vals), ls, marker="o", color=cmap(0.85 * i),
                            label=f"{'duct' if n == 'cystic_duct' else 'CBD'}, bias {_key(b)} mm")
        ax.set_xlabel(f"latency (ms) at {SPEED_MM_S:g} mm/s")
        ax.set_title("vs latency and bias (no jitter)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.suptitle(f"Cystic duct (solid) and CBD (dashed): {label} vs tracking error (mean over cases)", fontsize=10)
        fig.tight_layout()
        p = os.path.join(out_dir, fname)
        fig.savefig(p, dpi=100)
        plt.close(fig)
        paths.append(p)
    lats = list(grid.latency_ms)
    fig, axes = plt.subplots(1, len(lats), figsize=(4.2 * len(lats), 4.0), squeeze=False)
    for ax, lat in zip(axes[0], lats):
        m = np.full((len(grid.jitter_yaw_deg), len(grid.jitter_mm)), np.nan)
        for a in pick(latency_ms=lat, bias_x_mm=0.0):
            m[list(grid.jitter_yaw_deg).index(a["jitter_yaw_deg"]), list(grid.jitter_mm).index(a["jitter_mm"])] = \
                a["topology_ok"] / a["cases"]
        im = ax.imshow(m, origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(grid.jitter_mm)), [_key(v) for v in grid.jitter_mm])
        ax.set_yticks(range(len(grid.jitter_yaw_deg)), [_key(v) for v in grid.jitter_yaw_deg])
        ax.set_xlabel("position jitter (mm)")
        ax.set_ylabel("yaw jitter (deg)")
        ax.set_title(f"latency {_key(lat)} ms, bias 0", fontsize=9)
        for (i, j), v in np.ndenumerate(m):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="fraction of cases with the GB - CBD connection")
    fig.suptitle("Topology (GB - CBD connected through bile) vs tracking error", fontsize=10)
    p = os.path.join(out_dir, "errors_topology.png")
    fig.savefig(p, dpi=100)
    plt.close(fig)
    paths.append(p)
    return paths
