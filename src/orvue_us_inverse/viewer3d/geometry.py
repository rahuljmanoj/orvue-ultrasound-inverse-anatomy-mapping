"""
orvue_us_inverse.viewer3d.geometry - triangle meshes built from the simulator's own anatomy (simulation.anatomy).

Nothing is hard-coded: every Tube and Blob of the loaded case is meshed from its own definition.

    Tube   tube.sd(P) is sampled on a regular grid over tube.bbox() (padded 1 mm) and iso-surfaces are
           extracted with marching cubes (scikit-image): outer wall at sd = tube.wall, lumen at sd = 0.
           A light Taubin smoothing removes the voxel steps without shrinking (checked < 0.1 mm).
    Blob   ellipsoid from centre and radii.
    Liver  { 0<=x<=100, 0<=y, fat_t <= z <= depth, y < y0 + slope*(x-50) - wedge*max(z-fat_t, 0) },
           fat_t = fat_mean + fat_amp*sin(x/7)*cos(y/9), from the Anatomy attributes (closed surface).
    Fat    the sheet 0 <= z < fat_t over the 100 x 100 mm region.
    Calot  triangle: cystic duct first vertex (GB neck), its last vertex (junction with the CHD), and the
           chd_cbd centreline point interpolated at the neck's y.

A mesh is (vertices float32 (N, 3) in mm, faces int32 (M, 3)), outward-facing. Meshes are cached per
structure under output/cache/viewer3d/, keyed by a hash of the structure's geometry and the build settings,
so cases that share structures share cache files and switching cases is fast after the first build.
"""
import hashlib
import json
import math
import os
import struct

import numpy as np

from orvue_us_inverse import paths  # noqa: E402
from orvue_us_inverse.simulation.anatomy import Blob, Tube  # noqa: E402

GRID_MM = 0.4                    # default sampling of the implicit surfaces
LAYER_GRID_MM = 0.5              # liver / fat volume sampling (x, y); z uses GRID_MM
# quality presets: 'full' for tests / export (default), 'display' for the browser viewer (lighter meshes)
QUALITY = {
    "full": dict(tube=GRID_MM, layer=LAYER_GRID_MM, liver_z=GRID_MM, fat_z=0.2),
    "display": dict(tube=0.7, layer=1.5, liver_z=0.8, fat_z=0.25),
}
TAUBIN = (10, 0.5, -0.53)        # iterations, lambda, mu
CACHE_DIR = os.path.join(paths.CACHE_DIR, "viewer3d")
BUILD_VERSION = 3                # bump when the meshing changes, to invalidate the cache


# ---------------------------------------------------------------- helpers
def _marching_cubes(field, level, origin, h):
    """Iso-surface of a (nx, ny, nz) field at `level`, outward = towards increasing field values."""
    from skimage.measure import marching_cubes
    if not (field.min() < level < field.max()):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    v, f, _, _ = marching_cubes(field, level=level, spacing=(h, h, h), gradient_direction="ascent")
    return (v + np.asarray(origin, np.float32)).astype(np.float32), f.astype(np.int32)


def _flip_inward(v, f):
    """skimage's 'ascent' winding gives normals towards increasing values; reverse if needed so the signed
    volume is positive (outward normals)."""
    if len(f) and signed_volume(v, f) < 0:
        f = f[:, ::-1].copy()
    return v, f


def taubin_smooth(v, f, iterations=TAUBIN[0], lam=TAUBIN[1], mu=TAUBIN[2]):
    """Taubin lambda/mu smoothing (shrink-free low-pass) with uniform Laplacian weights."""
    if len(f) == 0:
        return v
    import scipy.sparse as sp
    n = len(v)
    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    A = sp.coo_matrix((np.ones(len(i), np.float32), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1.0
    W = sp.diags(1.0 / deg) @ A
    x = v.astype(np.float64)
    for _ in range(iterations):
        x = x + lam * (W @ x - x)
        x = x + mu * (W @ x - x)
    return x.astype(np.float32)


def signed_volume(v, f):
    a, b, c = v[f[:, 0]].astype(np.float64), v[f[:, 1]].astype(np.float64), v[f[:, 2]].astype(np.float64)
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _grid_points(lo, hi, h):
    axes = [np.arange(lo[k], hi[k] + h * 0.5, h, dtype=np.float32) for k in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1), tuple(len(a) for a in axes), axes


# ---------------------------------------------------------------- structures
def tube_meshes(tube: Tube, h=GRID_MM, smooth=True):
    """{'outer': mesh, 'lumen': mesh} of a Tube (outer wall surface sd = wall, lumen surface sd = 0)."""
    lo, hi = tube.bbox()
    lo, hi = np.asarray(lo, np.float32) - 1.0, np.asarray(hi, np.float32) + 1.0
    P, shape, _ = _grid_points(lo, hi, h)
    sd = tube.sd(P).reshape(shape)
    out = {}
    for part, level in (("outer", float(tube.wall)), ("lumen", 0.0)):
        v, f = _marching_cubes(sd, level, lo, h)
        if smooth:
            v = taubin_smooth(v, f)
        out[part] = _flip_inward(v, f)
    return out


def ellipsoid_mesh(centre, radii, n_lat=24, n_lon=48):
    c, r = np.asarray(centre, np.float32), np.asarray(radii, np.float32)
    th = np.linspace(0, math.pi, n_lat + 1)[1:-1]
    ph = np.linspace(0, 2 * math.pi, n_lon, endpoint=False)
    T, Ph = np.meshgrid(th, ph, indexing="ij")
    ring = np.stack([np.sin(T) * np.cos(Ph), np.sin(T) * np.sin(Ph), np.cos(T)], -1).reshape(-1, 3)
    v = np.vstack([[0, 0, 1], ring, [0, 0, -1]]) * r + c
    faces = []
    top, bot = 0, len(v) - 1
    idx = lambda a, b: 1 + a * n_lon + (b % n_lon)
    for b in range(n_lon):
        faces.append([top, idx(0, b), idx(0, b + 1)])
        faces.append([bot, idx(n_lat - 2, b + 1), idx(n_lat - 2, b)])
    for a in range(n_lat - 2):
        for b in range(n_lon):
            faces.append([idx(a, b), idx(a + 1, b), idx(a + 1, b + 1)])
            faces.append([idx(a, b), idx(a + 1, b + 1), idx(a, b + 1)])
    return _flip_inward(v.astype(np.float32), np.asarray(faces, np.int32))


def fat_thickness(an, x, y):
    return an.fat_mean + an.fat_amp * np.sin(x / 7.0) * np.cos(y / 9.0)


def liver_mesh(an, h=LAYER_GRID_MM, hz=GRID_MM):
    """Closed surface of the liver volume (see module docstring), from the Anatomy attributes."""
    X0, Y0, Z0 = an.size_xyz
    y_max = an.liver_edge_y0 + abs(an.liver_edge_slope) * X0 / 2 + 2.0
    xs = np.arange(-1.0, X0 + 1.0 + h / 2, h, dtype=np.float32)
    ys = np.arange(-1.0, y_max + h / 2, h, dtype=np.float32)
    zs = np.arange(-1.0, Z0 + 1.0 + hz / 2, hz, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    ft = fat_thickness(an, X, Y)
    edge = an.liver_edge_y0 + an.liver_edge_slope * (X - 50.0) - an.liver_edge_wedge * np.maximum(Z - ft, 0.0)
    fld = np.maximum.reduce([ft - Z, Z - Z0, -X, X - X0, -Y, Y - edge])            # <= 0 inside
    return _layer_surface(-fld, xs, ys, zs, h, hz)


def fat_mesh(an, h=LAYER_GRID_MM, hz=0.2):
    """Closed surface of the fat sheet 0 <= z < fat_t over the region."""
    X0, Y0, _ = an.size_xyz
    z_top = an.fat_mean + abs(an.fat_amp) + 1.0
    xs = np.arange(-1.0, X0 + 1.0 + h / 2, h, dtype=np.float32)
    ys = np.arange(-1.0, Y0 + 1.0 + h / 2, h, dtype=np.float32)
    zs = np.arange(-1.0, z_top + hz / 2, hz, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    fld = np.maximum.reduce([-Z, Z - fat_thickness(an, X, Y), -X, X - X0, -Y, Y - Y0])
    return _layer_surface(-fld, xs, ys, zs, h, hz)


def _layer_surface(inside_pos, xs, ys, zs, h, hz):
    """Iso-surface (level 0) of a field that is positive inside, on an anisotropic grid."""
    from skimage.measure import marching_cubes
    v, f, _, _ = marching_cubes(inside_pos, level=0.0, spacing=(h, h, hz), gradient_direction="descent")
    v = (v + np.array([xs[0], ys[0], zs[0]], np.float32)).astype(np.float32)
    return _flip_inward(v, f.astype(np.int32))


def calot_triangle_mesh(an):
    """Calot's triangle (GB neck, cystic-duct / CHD junction, CHD centreline at the neck's y), or None when
    the case has no cystic_duct or chd_cbd."""
    names = {t.name for t in an.tubes}
    if not {"cystic_duct", "chd_cbd"} <= names:
        return None
    cd, chd = an.tube("cystic_duct"), an.tube("chd_cbd")
    neck, junction = cd.pts[0].astype(np.float32), cd.pts[-1].astype(np.float32)
    p = chd.pts.astype(np.float64)
    order = np.argsort(p[:, 1])
    ys = p[order, 1]
    on_chd = np.array([np.interp(neck[1], ys, p[order, 0]), neck[1], np.interp(neck[1], ys, p[order, 2])],
                      np.float32)
    v = np.vstack([neck, junction, on_chd]).astype(np.float32)
    return v, np.array([[0, 1, 2]], np.int32)


def plane_mesh(T, width_mm=None, depth_mm=None):
    """B-mode image plane of a probe pose T (4x4, phantom frame, BModeSimulator.pose_from_xy_yaw convention:
    column 0 = lateral axis u, column 2 = axial (beam) axis, column 3 = centre of the transducer face).
    A width x depth rectangle from the face downwards. Defaults: simulation.bmode.LinearProbe width and depth."""
    if width_mm is None or depth_mm is None:
        w, d = probe_size()
        width_mm, depth_mm = width_mm or w, depth_mm or d
    T = np.asarray(T, np.float64)
    o, u, n = T[:3, 3], T[:3, 0], T[:3, 2]
    hw = width_mm / 2.0
    v = np.array([o - hw * u, o + hw * u, o + hw * u + depth_mm * n, o - hw * u + depth_mm * n], np.float32)
    return v, np.array([[0, 1, 2], [0, 2, 3]], np.int32)


def probe_size():
    """(width, depth) of the simulator's LinearProbe (mm), read from simulation.bmode without changing it."""
    from orvue_us_inverse.simulation.bmode import LinearProbe
    p = LinearProbe()
    return float(p.width_mm), float(p.depth_mm)


def tube_anchor(tube):
    """Point on the centreline half way along its length (label anchor)."""
    p = tube.pts.astype(np.float64)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    if seg.sum() == 0:
        return p[0].astype(np.float32)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = s[-1] / 2
    i = min(int(np.searchsorted(s, t)) - 1, len(seg) - 1)
    i = max(i, 0)
    a = (t - s[i]) / seg[i] if seg[i] > 0 else 0.0
    return (p[i] + a * (p[i + 1] - p[i])).astype(np.float32)


# ---------------------------------------------------------------- cache
def _key(kind, payload, q=None):
    h = hashlib.sha1()
    h.update(json.dumps({"kind": kind, "v": BUILD_VERSION, "q": q or QUALITY["full"],
                         "taubin": TAUBIN}, sort_keys=True).encode())
    for a in payload:
        h.update(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes())
    return f"{kind}_{h.hexdigest()[:16]}"


def _cached(key, build, cache=True):
    path = os.path.join(CACHE_DIR, key + ".npz")
    if cache and os.path.exists(path):
        d = np.load(path)
        return {k[:-2]: (d[k[:-2] + "_v"], d[k[:-2] + "_f"]) for k in d.files if k.endswith("_v")}
    meshes = build()
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(path, **{f"{k}_{s}": m[i] for k, m in meshes.items() for i, s in ((0, "v"), (1, "f"))})
    return meshes


def tube_key(t, q=None):
    return _key("tube", [t.pts, t.r, [t.wall, t.lumen_label, t.wall_label]], q)


def case_meshes(an, cache=True, quality="full"):
    """All meshes of a case: list of dicts {id, name, kind, part, mesh, key, prim} plus layers and Calot.

    kind: 'tube' (part 'outer' / 'lumen'), 'blob', 'liver', 'fat', 'calot'. quality: 'full' (0.4 mm, for tests
    and export) or 'display' (lighter, for the browser)."""
    q = QUALITY[quality]
    items = []
    for t in an.tubes:
        key = tube_key(t, q)
        ms = _cached(key, lambda t=t: tube_meshes(t, h=q["tube"]), cache)
        for part in ("outer", "lumen"):
            items.append(dict(id=f"{t.name}:{part}", name=t.name, kind="tube", part=part, mesh=ms[part],
                              key=f"{key}:{part}", prim=t))
    for b in an.blobs:
        key = _key("blob", [b.centre, b.radii, [b.label]], q)
        ms = _cached(key, lambda b=b: {"surface": ellipsoid_mesh(b.centre, b.radii)}, cache)
        items.append(dict(id=b.name, name=b.name, kind="blob", part="surface", mesh=ms["surface"],
                          key=f"{key}:surface", prim=b))
    layer = [an.size_xyz, [an.fat_mean, an.fat_amp, an.liver_edge_y0, an.liver_edge_slope, an.liver_edge_wedge]]
    builds = (("liver", lambda: liver_mesh(an, h=q["layer"], hz=q["liver_z"])),
              ("fat", lambda: fat_mesh(an, h=q["layer"], hz=q["fat_z"])))
    for kind, build in builds:
        key = _key(kind, layer, q)
        ms = _cached(key, lambda build=build: {"surface": build()}, cache)
        items.append(dict(id=kind, name=kind, kind=kind, part="surface", mesh=ms["surface"],
                          key=f"{key}:surface", prim=None))
    tri = calot_triangle_mesh(an)
    if tri is not None:
        items.append(dict(id="calot_triangle", name="calot_triangle", kind="calot", part="surface", mesh=tri,
                          key=_key("calot", [tri[0]]) + ":surface", prim=None))
    return items


# ---------------------------------------------------------------- export
def write_stl(path, v, f, name="mesh"):
    """Binary STL."""
    tri = v[f].astype(np.float32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(name.encode()[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(f)))
        fh.write(rec.tobytes())


def write_vtp(path, v, f):
    """VTK XML PolyData (ASCII), readable by ParaView / VTK without needing VTK here."""
    pts = " ".join(f"{a:.4f}" for a in v.ravel())
    con = " ".join(str(int(i)) for i in f.ravel())
    off = " ".join(str(3 * (k + 1)) for k in range(len(f)))
    with open(path, "w") as fh:
        fh.write('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
                 f'<PolyData><Piece NumberOfPoints="{len(v)}" NumberOfPolys="{len(f)}">\n'
                 f'<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">{pts}</DataArray></Points>\n'
                 f'<Polys><DataArray type="Int32" Name="connectivity" format="ascii">{con}</DataArray>\n'
                 f'<DataArray type="Int32" Name="offsets" format="ascii">{off}</DataArray></Polys>\n'
                 '</Piece></PolyData>\n</VTKFile>\n')


def export_case(an, folder, cache=True):
    """Write every mesh of the case as STL and VTP into folder/<case>/; returns the written paths."""
    out = os.path.join(folder, an.name)
    os.makedirs(out, exist_ok=True)
    paths = []
    for it in case_meshes(an, cache):
        v, f = it["mesh"]
        if len(f) == 0:
            continue
        stem = os.path.join(out, it["id"].replace(":", "_"))
        write_stl(stem + ".stl", v, f, it["id"])
        write_vtp(stem + ".vtp", v, f)
        paths += [stem + ".stl", stem + ".vtp"]
    return paths
