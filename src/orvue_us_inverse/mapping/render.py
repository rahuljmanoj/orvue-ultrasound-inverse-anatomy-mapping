"""
orvue_us_inverse.mapping.render - views of the reconstruction: slices, surfaces, snapshots, exports (S3).

    view = SliceView(grid)                          # OpenCV image, no window here
    img = view.update(comp)                         # comp: LabelCompounder (only the 3 slices are computed) or volume
    meshes = surface_meshes(labels, grid)           # {group: (vertices mm, faces)}, marching cubes + Taubin
    snapshot_3d(meshes, "snap.png", gt=gt_meshes)   # matplotlib (Agg) off-screen PNG
    export_stl(meshes, folder)                      # one binary STL per group
    write_browser_view(meshes, "recon.html", gt=gt_meshes)   # page of the 3D anatomy viewer (three.js)

Slice colours (BGR from the hex values): bile lumen 3 green #639922, arterial blood 7 red #E24B4A, venous blood 9 blue
#378ADD, stone 4 amber #FAC775, lymph node 10 purple #AFA9EC; liver, fat and the walls in muted greys; unobserved
voxels near-black. Surfaces are built for those five coloured groups.

Slice layout (crosshair at voxel (ix, iy, iz)): left the depth slice z = iz seen from above (x right, y down); right
top the x-z slice y = iy and right bottom the y-z slice x = ix, both with depth downwards. Scale: the larger of nx, ny
fills SLICE_PX pixels (2 px per voxel at 0.5 mm).
"""
import os

import cv2
import numpy as np

from orvue_us_inverse.mapping.recon import VoxelGrid
from orvue_us_inverse.ui import clinical as ui


def _bgr(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


# surface groups: name -> (labels, hex colour)
GROUPS = {
    "bile": ((3,), "#639922"),
    "arterial_blood": ((7,), "#E24B4A"),
    "venous_blood": ((9,), "#378ADD"),
    "stone": ((4,), "#FAC775"),
    "lymph_node": ((10,), "#AFA9EC"),
}
UNOBSERVED_BGR = (14, 14, 14)
# slice colour per tissue label 0-10 (BGR)
SLICE_BGR = np.array([
    (92, 92, 92),          # 0 liver
    (56, 56, 56),          # 1 fat
    (150, 150, 150),       # 2 GB wall
    _bgr("#639922"),       # 3 bile
    _bgr("#FAC775"),       # 4 stone
    (170, 170, 170),       # 5 duct wall
    (128, 128, 128),       # 6 artery wall
    _bgr("#E24B4A"),       # 7 arterial blood
    (112, 112, 112),       # 8 vein wall
    _bgr("#378ADD"),       # 9 venous blood
    _bgr("#AFA9EC"),       # 10 lymph node
], np.uint8)
SNAPSHOT_BILE_ALPHA = 0.45
SLICE_PX = 400
GAP = 12
CAPTION_H = 22
CROSS_BGR = (0, 200, 255)          # amber (ui.AMBER)
GT_BGR = (255, 255, 255)
ERROR_BGR = (255, 0, 255)          # magenta: reconstructed label differs from the ground truth


def colour_slice(lab2d: np.ndarray) -> np.ndarray:
    """BGR image (same shape) of a 2D label array; -1 = unobserved."""
    img = np.empty(lab2d.shape + (3,), np.uint8)
    obs = lab2d >= 0
    img[obs] = SLICE_BGR[lab2d[obs]]
    img[~obs] = UNOBSERVED_BGR
    return img


def _edges(lab2d: np.ndarray) -> np.ndarray:
    """Pixels where the label differs from the right or lower neighbour (boundary of the regions)."""
    e = np.zeros(lab2d.shape, bool)
    e[:, :-1] |= lab2d[:, 1:] != lab2d[:, :-1]
    e[:-1, :] |= lab2d[1:, :] != lab2d[:-1, :]
    return e


class SliceView:
    """Three orthogonal slices through a crosshair, as one BGR image; click() / move() set the crosshair."""

    def __init__(self, grid: VoxelGrid, crosshair: tuple[int, int, int] | None = None, gt: np.ndarray | None = None):
        self.grid = grid
        nx, ny, nz = grid.shape
        self.s = max(1, SLICE_PX // max(nx, ny))
        self.cross = list(crosshair) if crosshair is not None else [nx // 2, ny // 2, nz // 2]
        self.gt = gt
        self.show_gt = gt is not None
        self.errors = False                  # error colouring (needs gt): wrong voxels magenta, correct ones dimmed
        s = self.s
        # panel rectangles (x, y, w, h) in the image
        top = CAPTION_H
        self.rect = {"xy": (0, top, nx * s, ny * s),
                     "xz": (nx * s + GAP, top, nx * s, nz * s),
                     "yz": (nx * s + GAP, top + nz * s + GAP + CAPTION_H, ny * s, nz * s)}
        self.size = (max(top + ny * s, top + 2 * nz * s + GAP + CAPTION_H) + 4, nx * s + GAP + max(nx, ny) * s)

    # ---- crosshair
    def move(self, dx: int = 0, dy: int = 0, dz: int = 0) -> None:
        for i, d in enumerate((dx, dy, dz)):
            self.cross[i] = int(np.clip(self.cross[i] + d, 0, self.grid.shape[i] - 1))

    def click(self, x: int, y: int) -> bool:
        """Set the crosshair from a click at image pixel (x, y); True when it hit a slice."""
        for name, (rx, ry, rw, rh) in self.rect.items():
            if rx <= x < rx + rw and ry <= y < ry + rh:
                a, b = (x - rx) // self.s, (y - ry) // self.s
                if name == "xy":
                    self.cross[0], self.cross[1] = a, b
                elif name == "xz":
                    self.cross[0], self.cross[2] = a, b
                else:
                    self.cross[1], self.cross[2] = a, b
                self.move()                        # clip
                return True
        return False

    def crosshair_mm(self) -> np.ndarray:
        return self.grid.centre(np.array([self.cross]))[0]

    # ---- slices
    def slices(self, src) -> dict[str, np.ndarray]:
        """2D label arrays with rows going down the picture. src: label volume (nx, ny, nz) or an object with
        result_slice(axis, index) (LabelCompounder: computes only these slices)."""
        ix, iy, iz = self.cross
        if isinstance(src, np.ndarray):
            xy, xz, yz = src[:, :, iz], src[:, iy, :], src[ix, :, :]
        else:
            xy, xz, yz = src.result_slice(2, iz), src.result_slice(1, iy), src.result_slice(0, ix)
        return {"xy": xy.T, "xz": xz.T, "yz": yz.T}

    def update(self, src) -> np.ndarray:
        """The slice image for the current crosshair (pure function of src, crosshair, gt and error settings)."""
        h, w = self.size
        img = np.full((h, w, 3), ui.BG, np.uint8)
        sl = self.slices(src)
        gsl = self.slices(self.gt) if (self.show_gt and self.gt is not None) else None
        esl = self.slices(self.gt) if (self.errors and self.gt is not None) else None
        s = self.s
        ix, iy, iz = self.cross
        c = self.crosshair_mm()
        captions = {"xy": f"depth z = {c[2]:.2f} mm (from above: x right, y down)",
                    "xz": f"y = {c[1]:.2f} mm (x right, depth down)",
                    "yz": f"x = {c[0]:.2f} mm (y right, depth down)"}
        cross_px = {"xy": (ix, iy), "xz": (ix, iz), "yz": (iy, iz)}
        for name, (rx, ry, rw, rh) in self.rect.items():
            col = colour_slice(sl[name])
            if esl is not None:
                obs = sl[name] >= 0
                wrong = obs & (sl[name] != esl[name])
                col[obs & ~wrong] = (0.45 * col[obs & ~wrong]).astype(np.uint8)
                col[wrong] = ERROR_BGR
            pane = np.repeat(np.repeat(col, s, 0), s, 1)
            if gsl is not None:
                e = np.repeat(np.repeat(_edges(gsl[name]), s, 0), s, 1)
                pane[e] = (0.35 * pane[e] + 0.65 * np.array(GT_BGR)).astype(np.uint8)
            cx, cy = cross_px[name]
            px, py = cx * s + s // 2, cy * s + s // 2
            pane[py, :] = CROSS_BGR
            pane[:, px] = CROSS_BGR
            img[ry:ry + rh, rx:rx + rw] = pane
            ui.text(img, captions[name], (rx + 2, ry - 7), ui.GREY, 0.42)
        return img


# ---------------------------------------------------------------- surfaces
def surface_meshes(labels: np.ndarray, grid: VoxelGrid, groups: dict | None = None, smooth: int = 5
                   ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Marching-cubes surface (vertices float32 mm in the phantom frame, faces int32, outward) of each group's
    voxels, cropped to the group's bounding box (padded 1 voxel so the surfaces close at the grid border), with
    `smooth` iterations of Taubin smoothing (viewer3d.geometry.taubin_smooth) and vertices clamped to the grid
    bounds. Empty groups are skipped."""
    from skimage.measure import marching_cubes

    from orvue_us_inverse.viewer3d.geometry import signed_volume, taubin_smooth

    out = {}
    h = grid.voxel_mm
    for name, (labs, _) in (groups or GROUPS).items():
        mask = np.isin(labels, labs)
        if not mask.any():
            continue
        idx = np.nonzero(mask)
        lo = np.array([a.min() for a in idx]) - 1
        hi = np.array([a.max() for a in idx]) + 2
        sub = np.zeros(tuple(hi - lo), np.float32)
        clo, chi = np.maximum(lo, 0), np.minimum(hi, grid.shape)
        sub[tuple(slice(a - l, b - l) for a, b, l in zip(clo, chi, lo))] = \
            mask[tuple(slice(a, b) for a, b in zip(clo, chi))]
        v, f, _, _ = marching_cubes(sub, level=0.5)
        v = (grid.lo + (v + lo + 0.5) * h).astype(np.float32)
        f = f.astype(np.int32)
        if smooth:
            v = taubin_smooth(v, f, iterations=smooth)
        # surfaces closed at the grid border (e.g. the portal vein below 50 mm) stay on it after smoothing
        v = np.clip(v, grid.lo, grid.lo + np.array(grid.shape) * h).astype(np.float32)
        if signed_volume(v, f) < 0:
            f = f[:, ::-1].copy()
        out[name] = (v, f)
    return out


def snapshot_3d(meshes: dict, path: str, gt: dict | None = None, title: str = "", region=(100.0, 100.0, 50.0),
                size_px=(1200, 900), face_values: dict | None = None, vmax: float = 2.0,
                value_label: str = "") -> str:
    """Off-screen PNG (matplotlib Agg): isometric, equal axes, x right, y towards the viewer (down), depth downwards
    with the scanning surface z = 0 on top. gt meshes are drawn translucent. face_values {group: per-face values}
    colours those groups' faces on a 0..vmax scale (plasma) with a colour bar instead of the group colour."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(size_px[0] / 100, size_px[1] / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    colours = {k: c for k, (_, c) in GROUPS.items()}
    order = [n for n in GROUPS if n != "bile"] + ["bile"]          # bile last (translucent, drawn over its contents)
    meshes = {n: meshes[n] for n in order if n in meshes} | {n: m for n, m in meshes.items() if n not in GROUPS}
    light = matplotlib.colors.LightSource(azdeg=225, altdeg=45)
    cmap = matplotlib.colormaps["plasma"]
    for set_, base_alpha, valued in ((gt or {}, 0.12, False), (meshes, 1.0, True)):
        for name, (v, f) in set_.items():
            if len(f) == 0:
                continue
            alpha = base_alpha * (SNAPSHOT_BILE_ALPHA if name == "bile" else 1.0)   # stones show inside the GB
            if valued and face_values is not None and name in face_values:
                fc = cmap(np.clip(np.nan_to_num(face_values[name]) / vmax, 0, 1))
            else:
                fc = colours.get(name, "#B4B2A9")
            pc = Poly3DCollection(v[f], facecolors=fc, alpha=alpha, linewidths=0, shade=True, lightsource=light)
            ax.add_collection3d(pc)
    if face_values is not None:
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, vmax))
        fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, label=value_label)
    rx, ry, rz = region
    ax.set_xlim(0, rx)
    ax.set_ylim(ry, 0)                 # y grows towards the viewer (down the print)
    ax.set_zlim(rz, 0)                 # depth downwards, surface on top
    ax.set_box_aspect((rx, ry, rz))
    ax.view_init(elev=30, azim=-60)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("depth z (mm)")
    names = [n for n in GROUPS if n in meshes]
    ax.set_title(title + ("\n" if title else "") + ", ".join(n.replace("_", " ") for n in names)
                 + ("; ground truth translucent" if gt else ""), fontsize=10)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def export_stl(meshes: dict, folder: str, prefix: str = "recon") -> list[str]:
    """One binary STL per group: <folder>/<prefix>_<group>.stl (vertices in mm, phantom frame)."""
    from orvue_us_inverse.viewer3d.geometry import write_stl

    os.makedirs(folder, exist_ok=True)
    paths = []
    for name, (v, f) in meshes.items():
        p = os.path.join(folder, f"{prefix}_{name}.stl")
        write_stl(p, v, f, name=f"{prefix} {name}")
        paths.append(p)
    return paths


# ---------------------------------------------------------------- browser view (viewer3d page)
def browser_payload(meshes: dict, gt: dict | None = None, title: str = "reconstruction") -> dict:
    """Payload for viewer3d's three.js page: cases 'reconstruction', 'reconstruction_vs_truth' (truth translucent)
    and 'ground_truth', each with one structure per group."""
    from orvue_us_inverse.viewer3d import geometry as geo
    from orvue_us_inverse.viewer3d import viewer as v3d

    packed = {}

    def structures(src: dict, tag: str, opacity: float) -> list[dict]:
        out = []
        for name, (v, f) in src.items():
            if len(f) == 0:
                continue
            key = f"{tag}_{name}"
            packed[key] = v3d.encode_mesh(v, f)
            hex_c = GROUPS.get(name, ((), "#B4B2A9"))[1]
            pretty = ("truth: " if tag == "gt" else "") + name.replace("_", " ")
            out.append(dict(id=key, name=key, kind="blob", part="", mesh=key, pretty=pretty,
                            desc=f"{'Ground truth' if tag == 'gt' else 'Reconstructed'} {name.replace('_', ' ')} "
                                 f"surface ({len(f)} triangles)",
                            colour=hex_c, opacity=opacity, cls=name, anchor=[float(a) for a in v[np.argmin(v[:, 2])]]))
        return out

    cases = [dict(name="reconstruction", description=f"{title}: reconstructed surfaces",
                  structures=structures(meshes, "rec", 1.0))]
    if gt:
        cases.append(dict(name="reconstruction_vs_truth",
                          description=f"{title}: reconstruction (solid) and ground truth (translucent)",
                          structures=structures(meshes, "rec", 1.0) + structures(gt, "gt", 0.25)))
        cases.append(dict(name="ground_truth", description="ground truth surfaces from the anatomy voxelised on the "
                                                           "same grid", structures=structures(gt, "gt", 1.0)))
    w, d = geo.probe_size()
    toggles = {k: v3d.TOGGLES[k] for k in ("labels", "surface", "probe")}
    return dict(cases=cases, meshes=packed, presets=v3d.PRESETS, default_preset=v3d.DEFAULT_PRESET,
                default_parallel=v3d.DEFAULT_PARALLEL, target=v3d.TARGET, toggles=toggles,
                probe=dict(width=w, depth=d), region=[100.0, 100.0, 50.0])


def write_browser_view(meshes: dict, path: str, gt: dict | None = None, title: str = "reconstruction") -> str:
    """Self-contained HTML page (viewer3d template; three.js from a CDN) with the reconstruction (and truth)."""
    from orvue_us_inverse.viewer3d import viewer as v3d

    start = "reconstruction_vs_truth" if gt else "reconstruction"
    return v3d.write_html(path, browser_payload(meshes, gt, title), start_case=start)
