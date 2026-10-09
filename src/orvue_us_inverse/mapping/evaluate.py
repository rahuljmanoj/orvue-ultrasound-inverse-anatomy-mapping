"""
orvue_us_inverse.mapping.evaluate - comparison of a reconstructed label volume with the ground truth (S4).

    ev = evaluate(labels, anatomy, grid, case="normal")     # labels: int8 (nx, ny, nz), -1 = unobserved
    print(summary_table(ev.report))
    folder = write_report(ev, os.path.join(RESULTS_DIR, f"normal_{stamp}"))

Ground truth
    Label volume: Anatomy.labels at the voxel centres (recon.ground_truth). Instance masks (new code; anatomy.py is
    unchanged): for every Tube its lumen (tube.sd < 0) and wall (0 <= sd < wall), for every Blob its interior
    (blob.inside), evaluated at the voxel centres inside the primitive's bounding box. Where primitives overlap the
    label volume decides (e.g. a stone inside the gallbladder lumen), so an instance's "effective" voxels are its
    geometric voxels whose ground-truth label is the instance's own label.

Everything is measured inside the evaluation mask = observed voxels (labels >= 0, after optional hole filling):
    classes     per tissue class (bile 3, arterial blood 7, venous blood 9, stone 4, lymph node 10, walls 2 / 5 / 6 / 8):
                Dice, IoU, precision, recall; symmetric mean surface distance (MSD) and 95th-percentile Hausdorff
                distance (HD95) in mm between the boundary voxels of the reconstructed and the true class (boundary =
                voxels of the class with a 6-neighbour outside it; distances from scipy.ndimage distance transforms),
                and the class coverage (% of its true voxels observed).
    structures  per instance: coverage (% of its effective lumen / interior voxels observed), recall (observed
                effective voxels reconstructed with the correct class); status "detected" (recall >= 50 %), "missed",
                "not covered" (coverage < 20 %: outside the scanned region, not a miss) or "no voxels" (thinner than
                the grid); wall coverage / recall alongside.
    topology    are the gallbladder lumen and the CHD / CBD lumen connected through bile voxels (26-connectivity) in
                the reconstruction, as in the ground truth? "not covered" when either lumen was not observed.

A one-voxel shift gives HD95 = 1 voxel but MSD ~ 0.45-0.7 voxel (normal case): surfaces parallel to the shift do not move, so
the mean over a closed surface is about half the shift (a sphere: exactly half).
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from orvue_us_inverse.mapping.recon import VoxelGrid, ground_truth
from orvue_us_inverse.mapping.render import GROUPS, snapshot_3d, surface_meshes
from orvue_us_inverse.simulation.anatomy import TISSUES

# evaluated classes, in report order
CLASSES = {3: "bile", 7: "arterial blood", 9: "venous blood", 4: "stone", 10: "lymph node",
           2: "GB wall", 5: "duct wall", 6: "artery wall", 8: "vein wall"}
MIN_COVERAGE = 0.20           # below: "not covered"
DETECT_RECALL = 0.50          # at or above: "detected"
TOPOLOGY_PAIR = ("gallbladder", "chd_cbd")
BILE = 3
DIST_VMAX_MM = 2.0            # colour scale of the distance overlay


# ---------------------------------------------------------------- ground-truth instances
@dataclass
class Instance:
    """Voxels of one structure part on the grid (flat indices, sorted)."""
    name: str
    kind: str                 # "tube" or "blob"
    part: str                 # "lumen", "wall" or "interior"
    label: int
    flat: np.ndarray          # geometric voxels (centre inside the part)


def _box_points(grid: VoxelGrid, lo, hi) -> tuple[np.ndarray, np.ndarray]:
    """Flat indices and centres (float32) of the voxels whose centres lie in the box [lo, hi]."""
    h = grid.voxel_mm
    i0 = np.clip(np.ceil((np.asarray(lo) - grid.lo) / h - 0.5), 0, None).astype(int)
    i1 = np.minimum(np.floor((np.asarray(hi) - grid.lo) / h - 0.5).astype(int), np.array(grid.shape) - 1)
    if np.any(i1 < i0):
        return np.zeros(0, np.int64), np.zeros((0, 3), np.float32)
    ax = [np.arange(a, b + 1) for a, b in zip(i0, i1)]
    I, J, K = np.meshgrid(*ax, indexing="ij")
    ijk = np.stack([I.ravel(), J.ravel(), K.ravel()], 1)
    return grid.flat(ijk), grid.centre(ijk).astype(np.float32)


def instance_masks(anatomy, grid: VoxelGrid) -> dict[str, dict[str, Instance]]:
    """{structure name: {"lumen": Instance, "wall": Instance}} for tubes, {"interior": Instance} for blobs."""
    out: dict[str, dict[str, Instance]] = {}
    for t in anatomy.tubes:
        flat, P = _box_points(grid, *t.bbox())
        sd = t.sd(P) if len(P) else np.zeros(0)
        out[t.name] = {"lumen": Instance(t.name, "tube", "lumen", int(t.lumen_label), flat[sd < 0]),
                       "wall": Instance(t.name, "tube", "wall", int(t.wall_label), flat[(sd >= 0) & (sd < t.wall)])}
    for b in anatomy.blobs:
        c, r = np.asarray(b.centre, float), np.asarray(b.radii, float)
        flat, P = _box_points(grid, c - r, c + r)
        inside = b.inside(P) if len(P) else np.zeros(0, bool)
        out[b.name] = {"interior": Instance(b.name, "blob", "interior", int(b.label), flat[inside])}
    return out


# ---------------------------------------------------------------- metrics
def _boundary(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m, border_value=0)


def surface_distances(a: np.ndarray, b: np.ndarray, voxel_mm: float) -> tuple[float | None, float | None]:
    """Symmetric mean surface distance and 95th-percentile Hausdorff distance (mm) between the boundary voxels of
    two boolean volumes (None when either is empty). Computed on the bounding box of a | b, padded."""
    if not a.any() or not b.any():
        return None, None
    idx = np.nonzero(a | b)
    lo = np.maximum(np.array([i.min() for i in idx]) - 2, 0)
    hi = np.array([i.max() for i in idx]) + 3
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    ba, bb = _boundary(a[sl]), _boundary(b[sl])
    d_ab = ndimage.distance_transform_edt(~bb, sampling=voxel_mm)[ba]
    d_ba = ndimage.distance_transform_edt(~ba, sampling=voxel_mm)[bb]
    d = np.concatenate([d_ab, d_ba])
    return float(d.mean()), float(np.percentile(d, 95))


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def class_metrics(labels: np.ndarray, gt: np.ndarray, mask: np.ndarray, voxel_mm: float) -> dict:
    out = {}
    for c, name in CLASSES.items():
        a, b = (labels == c) & mask, (gt == c) & mask
        na, nb, inter = int(a.sum()), int(b.sum()), int((a & b).sum())
        msd, hd95 = surface_distances(a, b, voxel_mm)
        n_true = int((gt == c).sum())
        out[name] = dict(label=c, n_true=n_true, n_true_observed=nb, n_recon=na,
                         coverage=_ratio(nb, n_true), dice=_ratio(2 * inter, na + nb), iou=_ratio(inter, na + nb - inter),
                         precision=_ratio(inter, na), recall=_ratio(inter, nb), msd_mm=msd, hd95_mm=hd95)
    return out


def _part_stats(inst: Instance, labels: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    eff = inst.flat[gt.ravel()[inst.flat] == inst.label]
    obs = mask.ravel()[eff]
    n, n_obs = len(eff), int(obs.sum())
    correct = int((labels.ravel()[eff[obs]] == inst.label).sum())
    return dict(n_voxels=n, coverage=_ratio(n_obs, n), recall=_ratio(correct, n_obs))


def structure_metrics(instances: dict, labels: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> list[dict]:
    rows = []
    for name, parts in instances.items():
        main = parts.get("lumen") or parts["interior"]
        st = _part_stats(main, labels, gt, mask)
        if st["n_voxels"] == 0:
            status = "no voxels"
        elif st["coverage"] < MIN_COVERAGE:
            status = "not covered"
        else:
            status = "detected" if st["recall"] >= DETECT_RECALL else "missed"
        row = dict(name=name, kind=main.kind, part=main.part, label=main.label, tissue=TISSUES[main.label][0], **st,
                   status=status)
        if "wall" in parts:
            w = _part_stats(parts["wall"], labels, gt, mask)
            row.update(wall_voxels=w["n_voxels"], wall_coverage=w["coverage"], wall_recall=w["recall"])
        rows.append(row)
    return rows


def bile_connected(labels: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """True when some 26-connected component of bile voxels contains voxels of both flat index sets a and b."""
    bile = labels == BILE
    if not bile.any():
        return False
    comp, _ = ndimage.label(bile, structure=np.ones((3, 3, 3), bool))
    ca, cb = set(np.unique(comp.ravel()[a])) - {0}, set(np.unique(comp.ravel()[b])) - {0}
    return bool(ca & cb)


def topology(instances: dict, labels: np.ndarray, gt: np.ndarray, mask: np.ndarray, pair=TOPOLOGY_PAIR) -> dict:
    if not all(p in instances for p in pair):
        return dict(pair=list(pair), status="not applicable")
    eff = [instances[p]["lumen"].flat[gt.ravel()[instances[p]["lumen"].flat] == BILE] for p in pair]
    covered = all(mask.ravel()[e].mean() >= MIN_COVERAGE if len(e) else False for e in eff)
    gt_conn = bile_connected(gt, *eff)
    rec_conn = bile_connected(np.where(mask, labels, -1), *eff)
    if not covered:
        status = "not covered"
    else:
        status = "ok" if rec_conn == gt_conn else ("disconnected" if gt_conn else "spuriously connected")
    return dict(pair=list(pair), connected_truth=gt_conn, connected_recon=rec_conn, covered=covered, status=status)


# ---------------------------------------------------------------- evaluation
@dataclass
class Evaluation:
    report: dict
    labels: np.ndarray
    gt: np.ndarray
    grid: VoxelGrid
    instances: dict = field(repr=False, default_factory=dict)


def evaluate(labels: np.ndarray, anatomy, grid: VoxelGrid, gt: np.ndarray | None = None,
             instances: dict | None = None, case: str | None = None, extra: dict | None = None) -> Evaluation:
    """All metrics of a reconstructed label volume inside its observed voxels."""
    gt = ground_truth(anatomy, grid) if gt is None else gt
    instances = instance_masks(anatomy, grid) if instances is None else instances
    mask = labels >= 0
    n_obs = int(mask.sum())
    structures = structure_metrics(instances, labels, gt, mask)
    counts = {s: sum(r["status"] == s for r in structures) for s in ("detected", "missed", "not covered", "no voxels")}
    report = dict(
        case=case or getattr(anatomy, "name", None),
        grid=grid.to_dict(), voxel_mm=grid.voxel_mm,
        observed_fraction=n_obs / grid.n_voxels, n_observed=n_obs,
        accuracy_observed=_ratio(int((labels[mask] == gt[mask]).sum()), n_obs),
        classes=class_metrics(labels, gt, mask, grid.voxel_mm),
        structures=structures,
        topology=topology(instances, labels, gt, mask),
        counts=counts,
        thresholds=dict(min_coverage=MIN_COVERAGE, detect_recall=DETECT_RECALL),
        **(extra or {}),
    )
    return Evaluation(report, labels, gt, grid, instances)


# ---------------------------------------------------------------- report
def _f(v, spec=".3f", pct=False) -> str:
    if v is None:
        return "-"
    return f"{100 * v:.1f}%" if pct else format(v, spec)


def summary_table(r: dict) -> str:
    """Plain-text / Markdown tables of a report: overview, classes, structures, topology."""
    lines = [f"Case {r['case']}, voxels {r['voxel_mm']:g} mm: observed {_f(r['observed_fraction'], pct=True)} of the "
             f"volume, {_f(r['accuracy_observed'], pct=True)} of observed voxels correct; structures detected "
             f"{r['counts']['detected']}, missed {r['counts']['missed']}, not covered {r['counts']['not covered']}, "
             f"no voxels {r['counts']['no voxels']}; topology {r['topology'].get('status')}",
             "",
             "| Class | Coverage | Dice | IoU | Precision | Recall | MSD (mm) | HD95 (mm) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, c in r["classes"].items():
        lines.append(f"| {name} ({c['label']}) | {_f(c['coverage'], pct=True)} | {_f(c['dice'])} | {_f(c['iou'])} | "
                     f"{_f(c['precision'])} | {_f(c['recall'])} | {_f(c['msd_mm'], '.2f')} | {_f(c['hd95_mm'], '.2f')} |")
    lines += ["", "| Structure | Voxels | Coverage | Recall | Status | Wall coverage | Wall recall |",
              "|---|---|---|---|---|---|---|"]
    for s in r["structures"]:
        lines.append(f"| {s['name']} ({s['tissue']}) | {s['n_voxels']} | {_f(s['coverage'], pct=True)} | "
                     f"{_f(s['recall'], pct=True)} | {s['status']} | {_f(s.get('wall_coverage'), pct=True)} | "
                     f"{_f(s.get('wall_recall'), pct=True)} |")
    t = r["topology"]
    lines += ["", f"Topology {' - '.join(t['pair'])} through bile: truth "
                  f"{'connected' if t.get('connected_truth') else 'not connected'}, reconstruction "
                  f"{'connected' if t.get('connected_recon') else 'not connected'} -> {t['status']}"]
    return "\n".join(lines)


def vertex_distances(v: np.ndarray, target: np.ndarray, grid: VoxelGrid) -> np.ndarray:
    """Distance (mm) from mesh vertices to the boundary of the boolean volume `target` (nearest voxel)."""
    if not target.any():
        return np.full(len(v), np.nan)
    d = ndimage.distance_transform_edt(~_boundary(target), sampling=grid.voxel_mm)
    ijk = np.clip(np.round((v - grid.lo) / grid.voxel_mm - 0.5).astype(int), 0, np.array(grid.shape) - 1)
    return d[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def overlay_3d(ev: Evaluation, path: str, title: str = "") -> str:
    """Reconstruction surfaces coloured by distance to the ground-truth surface (mm), ground truth translucent."""
    labels = np.where(ev.labels >= 0, ev.labels, -1)
    meshes = surface_meshes(labels, ev.grid)
    gt_meshes = surface_meshes(np.where(ev.labels >= 0, ev.gt, -1), ev.grid)
    values = {}
    for name, (v, f) in meshes.items():
        target = np.isin(ev.gt, GROUPS[name][0]) & (ev.labels >= 0)
        d = vertex_distances(v, target, ev.grid)
        values[name] = d[f].mean(axis=1)
    return snapshot_3d(meshes, path, gt=gt_meshes, title=title, face_values=values, vmax=DIST_VMAX_MM,
                       value_label="distance to ground-truth surface (mm)")


def structure_chart(report: dict, path: str) -> str:
    """Horizontal bars per structure: coverage and recall, with the status."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    rows = report["structures"]
    names = [f"{s['name']} ({s['status']})" for s in rows]
    cov = [100 * (s["coverage"] or 0) for s in rows]
    rec = [100 * (s["recall"] or 0) for s in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.6), dpi=100)
    ax.barh(y - 0.2, cov, 0.38, label="coverage (% of voxels observed)", color="#378ADD")
    ax.barh(y + 0.2, rec, 0.38, label="recall (% of observed voxels correct)", color="#639922")
    ax.axvline(100 * DETECT_RECALL, color="#E24B4A", lw=0.8, ls="--")
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("%")
    ax.set_title(f"{report['case']}: structures", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.07 - 0.6 / len(rows)), ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def complete_report(raw: np.ndarray, anatomy, grid: VoxelGrid, folder: str, case: str, gt: np.ndarray | None = None,
                    instances: dict | None = None, fill: bool = True, max_gap: int = 1, extra: dict | None = None,
                    title: str = "", slice_view=None) -> tuple[Evaluation, np.ndarray]:
    """The "complete" step shared by the apps: fill small holes, evaluate, write the report folder; with a
    render.SliceView, switch it to error colouring and save slices_errors.png first. Returns (evaluation, labels)."""
    import cv2
    from orvue_us_inverse.mapping.recon import fill_small_holes

    labels = fill_small_holes(raw, max_gap) if fill else raw
    gt = ground_truth(anatomy, grid) if gt is None else gt
    extra = dict(extra or {}, fill=fill, max_gap=max_gap if fill else 0,
                 observed_fraction_before_fill=float((raw >= 0).mean()))
    ev = evaluate(labels, anatomy, grid, gt=gt, instances=instances, case=case, extra=extra)
    os.makedirs(folder, exist_ok=True)
    if slice_view is not None:
        slice_view.gt, slice_view.errors = gt, True
        cv2.imwrite(os.path.join(folder, "slices_errors.png"), slice_view.update(labels))
    write_report(ev, folder, title=title)
    return ev, labels


def write_report(ev: Evaluation, folder: str, title: str = "") -> str:
    """report.json, report.md, overlay_3d.png and structures.png in folder; returns the folder."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(ev.report, fh, indent=1)
    overlay_3d(ev, os.path.join(folder, "overlay_3d.png"),
               title=title or f"{ev.report['case']}: reconstruction coloured by distance to the ground truth")
    structure_chart(ev.report, os.path.join(folder, "structures.png"))
    md = [f"# Inverse mapping evaluation: {ev.report['case']}", "", title, "", summary_table(ev.report), "",
          "![3D overlay](overlay_3d.png)", "", "![Structures](structures.png)", ""]
    if os.path.exists(os.path.join(folder, "slices_errors.png")):
        md += ["![Slices, errors](slices_errors.png)", ""]
    with open(os.path.join(folder, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return folder
