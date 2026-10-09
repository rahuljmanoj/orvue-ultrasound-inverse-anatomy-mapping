"""
orvue_us_inverse.clinical.ar - augmented-reality overlay: the reconstructed structures drawn onto the live camera
image of the phantom, where they lie under the surface.

surface_map(labels, grid) takes the reconstructed label volume and keeps, for every (x, y) column, the shallowest
structure (gallbladder / bile ducts, stone, artery, vein, lymph node; liver and fat are background) and its depth.
ar_layer() colours it by structure group (the colours of the 3D view) and codes the depth: shallow structures are
bright and opaque, deep ones darker and more transparent (DEPTH_RANGE_MM), with a solid outline around each group.
A structure under another one would be hidden by that fill, so the bile and artery groups (OUTLINE_GROUPS) also get
an outline of their footprint at any depth (footprints(): the group anywhere in the column): solid where the group
is the shallowest structure, dashed where something shallower covers it; arteries are drawn last. The outline is a
band OUTLINE_VOX voxels wide inside the footprint, so thin vessels (the cystic artery and its branches, ~2-4 mm
across) show as a whole. This is a surface map: a vertical projection of the volume to z = 0 (not a perspective
view from the camera). project_layer() maps the layer from the phantom's x-y plane at the gel surface (z = 0) into the camera image with
the tracked board pose (T_cam_phantom, K, dist) and blends it in: the surgeon sees on the phantom where each
structure lies below the surface.

    labels = comp.result()
    lab2d, depth2d = surface_map(labels, grid)
    bgr, alpha = ar_layer(lab2d, depth2d, footprints(labels))
    frame = project_layer(frame, bgr, alpha, state.T_cam_phantom, tracker.K, tracker.dist, grid)
"""
import cv2
import numpy as np
from scipy import ndimage

from orvue_us_inverse.ui import clinical as ui

DEPTH_RANGE_MM = 50.0              # depth at which the colour is darkest / most transparent
ALPHA_SHALLOW, ALPHA_DEEP = 0.70, 0.30
OUTLINE_ALPHA = 0.95
UPSCALE = 4                        # layer pixels per voxel before warping (smoother edges)
OUTLINE_GROUPS = ("gallbladder / bile ducts", "artery")      # outlined at any depth, in this order (arteries last)
OUTLINE_VOX = 5                    # outline band width in voxels (2.5 mm at 0.5 mm voxels)
DASH_PX = 4                        # dash length of a covered outline, layer pixels (diagonal hatch)
SEGMENTATION = "SEGMENTATION: ORACLE (simulator labels)"     # what the labels come from (shown with the overlay)

# structure groups: (name, tissue labels, BGR), colours as in the 3D view (render.GROUPS)
GROUPS = (
    ("gallbladder / bile ducts", (2, 3, 5), (34, 153, 99)),
    ("stone", (4,), (117, 199, 250)),
    ("artery", (6, 7), (74, 75, 226)),
    ("vein", (8, 9), (221, 138, 55)),
    ("lymph node", (10,), (236, 169, 175)),
)
STRUCTURE_LABELS = tuple(lab for _, labels, _ in GROUPS for lab in labels)
_GROUP_OF = np.full(256, -1, np.int16)
for _g, (_, _labels, _) in enumerate(GROUPS):
    _GROUP_OF[list(_labels)] = _g
_GROUP_BGR = np.array([c for _, _, c in GROUPS], np.float32)


def surface_map(labels: np.ndarray, grid) -> tuple[np.ndarray, np.ndarray]:
    """(label, depth) of the shallowest structure in every column, as images (ny, nx) with rows = y and
    columns = x: label -1 and depth NaN where the column holds no structure (or was not observed)."""
    mask = np.isin(labels, STRUCTURE_LABELS)
    found = mask.any(axis=2)
    first = np.argmax(mask, axis=2)
    lab = np.take_along_axis(labels, first[..., None], axis=2)[..., 0].astype(np.int16)
    lab[~found] = -1
    depth = grid.lo[2] + (first + 0.5) * grid.voxel_mm
    depth = np.where(found, depth, np.nan)
    return lab.T.copy(), depth.T.copy()


def group_map(lab2d: np.ndarray) -> np.ndarray:
    """Structure group index per pixel (-1: none)."""
    return np.where(lab2d >= 0, _GROUP_OF[np.clip(lab2d, 0, 255)], -1)


def group_index(name: str) -> int:
    return next(i for i, (n, _, _) in enumerate(GROUPS) if n == name)


def footprints(labels: np.ndarray, names=OUTLINE_GROUPS) -> dict[str, np.ndarray]:
    """Columns (ny, nx; rows = y) holding the group at any depth, per group name."""
    return {n: np.isin(labels, GROUPS[group_index(n)][1]).any(axis=2).T.copy() for n in names}


def outline_marks(lab2d: np.ndarray, fps: dict[str, np.ndarray], width_vox: int = OUTLINE_VOX
                  ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(solid, dashed) outline columns per group: the band of width_vox voxels inside the footprint, solid where
    the group is the shallowest structure, dashed where it is covered."""
    g = group_map(lab2d)
    out = {}
    for name, fp in fps.items():
        band = fp & ~ndimage.binary_erosion(fp, iterations=width_vox, border_value=0)
        top = g == group_index(name)
        out[name] = (band & top, band & ~top)
    return out


def marked_columns(lab2d: np.ndarray, fps: dict[str, np.ndarray], name: str) -> np.ndarray:
    """Columns where the overlay shows the group: its fill (shallowest) or its outline (solid or dashed)."""
    solid, dashed = outline_marks(lab2d, {name: fps[name]})[name]
    return (group_map(lab2d) == group_index(name)) | solid | dashed


def ar_layer(lab2d: np.ndarray, depth2d: np.ndarray, fps: dict[str, np.ndarray] | None = None,
             depth_range_mm: float = DEPTH_RANGE_MM) -> tuple[np.ndarray, np.ndarray]:
    """BGR layer (uint8) and alpha (float32 0..1) of the surface map, UPSCALE pixels per voxel; with footprints
    (footprints()) also the solid / dashed outlines of OUTLINE_GROUPS."""
    g = group_map(lab2d)
    has = g >= 0
    f = np.clip(np.nan_to_num(depth2d, nan=0.0) / depth_range_mm, 0.0, 1.0)
    shade = (1.0 - 0.55 * f)[..., None]
    bgr = np.zeros((*g.shape, 3), np.float32)
    bgr[has] = _GROUP_BGR[g[has]]
    bgr *= shade
    alpha = np.where(has, ALPHA_SHALLOW + (ALPHA_DEEP - ALPHA_SHALLOW) * f, 0.0).astype(np.float32)
    bgr = cv2.resize(bgr, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_NEAREST)
    alpha = cv2.resize(alpha, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_NEAREST)
    gu = cv2.resize(g.astype(np.float32), None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_NEAREST)
    edge = np.zeros(gu.shape, bool)                      # outline: group differs from a neighbour
    edge[:, 1:] |= gu[:, 1:] != gu[:, :-1]
    edge[:, :-1] |= gu[:, 1:] != gu[:, :-1]
    edge[1:, :] |= gu[1:, :] != gu[:-1, :]
    edge[:-1, :] |= gu[1:, :] != gu[:-1, :]
    edge &= gu >= 0
    gi = gu.astype(np.int16)
    bgr[edge] = _GROUP_BGR[gi[edge]]
    alpha[edge] = OUTLINE_ALPHA
    if fps:
        rr, cc = np.indices(alpha.shape)
        dash = ((rr + cc) // DASH_PX) % 2 == 0                # diagonal dashes; every voxel block holds some
        marks = outline_marks(lab2d, fps)
        for name in OUTLINE_GROUPS:                           # in order: arteries last (on top)
            if name not in marks:
                continue
            solid, dashed = (cv2.resize(m.astype(np.uint8), None, fx=UPSCALE, fy=UPSCALE,
                                        interpolation=cv2.INTER_NEAREST).astype(bool) for m in marks[name])
            on = solid | (dashed & dash)
            bgr[on] = _GROUP_BGR[group_index(name)]
            alpha[on] = OUTLINE_ALPHA
    return np.clip(bgr, 0, 255).astype(np.uint8), alpha


def layer_homography(T_cam_phantom: np.ndarray, K: np.ndarray, dist: np.ndarray, grid, shape: tuple[int, int],
                     z_mm: float = 0.0) -> np.ndarray:
    """3x3 homography from layer pixels (shape = (rows, cols), covering the grid's x-y extent) to camera pixels,
    for the plane z = z_mm of the phantom (lens distortion taken into account at the four corners)."""
    (x0, x1), (y0, y1) = grid.cfg.x_mm, grid.cfg.y_mm
    corners = np.array([[x0, y0, z_mm], [x1, y0, z_mm], [x1, y1, z_mm], [x0, y1, z_mm]], np.float64)
    R, t = np.asarray(T_cam_phantom, np.float64)[:3, :3], np.asarray(T_cam_phantom, np.float64)[:3, 3]
    dst, _ = cv2.projectPoints(corners, cv2.Rodrigues(R)[0], t, np.asarray(K, np.float64),
                               np.asarray(dist, np.float64))
    h, w = shape
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32) - 0.5
    return cv2.getPerspectiveTransform(src, dst.reshape(4, 2).astype(np.float32))


def project_layer(frame: np.ndarray, bgr: np.ndarray, alpha: np.ndarray, T_cam_phantom: np.ndarray, K: np.ndarray,
                  dist: np.ndarray, grid, z_mm: float = 0.0) -> np.ndarray:
    """The camera frame with the layer blended in where the phantom's surface region appears."""
    H = layer_homography(T_cam_phantom, K, dist, grid, alpha.shape, z_mm)
    size = (frame.shape[1], frame.shape[0])
    wb = cv2.warpPerspective(bgr, H, size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    wa = cv2.warpPerspective(alpha, H, size, flags=cv2.INTER_LINEAR, borderValue=0.0)[..., None]
    out = frame.astype(np.float32) * (1.0 - wa) + wb.astype(np.float32) * wa
    return out.astype(np.uint8)


def group_volumes_ml(labels: np.ndarray, voxel_mm: float) -> list[tuple[str, tuple, float]]:
    """(group name, BGR, reconstructed volume in mL) per structure group."""
    counts = np.bincount(labels[labels >= 0].ravel().astype(np.int64), minlength=11)
    v = voxel_mm ** 3 / 1000.0
    return [(name, colour, float(counts[list(labs)].sum()) * v) for name, labs, colour in GROUPS]


KEY_H = 70                         # height of the colour key strip (draw_key)


def segmentation_chip(img: np.ndarray, org: tuple[int, int], scale: float = 0.4) -> int:
    """Fixed chip saying where the labels come from (oracle = the simulator's ground truth); returns its width."""
    return ui.chip(img, org, SEGMENTATION, ui.AMBER, scale)


def draw_key(img: np.ndarray, x: int, y: int, width: int, depth_range_mm: float = DEPTH_RANGE_MM) -> int:
    """Colour key of the overlay in a dark strip from (x, y) (top-left), KEY_H high: title "surface map" with the
    segmentation chip, the structure groups, the depth shading and the outline styles. Returns the y below it."""
    cv2.rectangle(img, (x, y), (x + width - 1, y + KEY_H - 1), (0, 0, 0), -1)
    x0, y1, y2, y3 = x + 6, y + 18, y + 40, y + 62
    ui.text(img, "SURFACE MAP", (x0, y1), ui.AMBER, 0.4)
    ui.text(img, "vertical projection to z = 0", (x0 + ui.text_w("SURFACE MAP", 0.4) + 8, y1), ui.GREY, 0.38)
    cw = ui.text_w(SEGMENTATION, 0.36) + 30
    segmentation_chip(img, (x + width - cw - 6, y1 + 2), 0.36)
    xx = x0
    for name, _, colour in GROUPS:
        cv2.rectangle(img, (xx, y2 - 9), (xx + 11, y2 + 2), colour, -1)
        ui.text(img, name, (xx + 15, y2), ui.GREY, 0.38)
        xx += 26 + ui.text_w(name, 0.38)
    w = 44
    for i in range(w):
        c = tuple(int(255 * (1.0 - 0.55 * i / (w - 1))) for _ in range(3))
        cv2.line(img, (x0 + i, y3 - 9), (x0 + i, y3 + 2), c, 1)
    label = f"surface -> {depth_range_mm:g} mm deep"
    ui.text(img, label, (x0 + w + 5, y3), ui.GREY, 0.38)
    xx = x0 + w + 25 + ui.text_w(label, 0.38)
    red = tuple(int(c) for c in _GROUP_BGR[group_index("artery")])
    cv2.line(img, (xx, y3 - 4), (xx + 22, y3 - 4), red, 2)
    ui.text(img, "on top", (xx + 27, y3), ui.GREY, 0.38)
    xx += 40 + ui.text_w("on top", 0.38)
    for k in range(0, 22, 8):
        cv2.line(img, (xx + k, y3 - 4), (xx + k + 4, y3 - 4), red, 2)
    ui.text(img, "covered (bile, artery outlines)", (xx + 27, y3), ui.GREY, 0.38)
    return y + KEY_H
