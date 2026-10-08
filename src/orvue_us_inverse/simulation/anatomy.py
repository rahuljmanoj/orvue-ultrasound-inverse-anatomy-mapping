"""
orvue_us_inverse.simulation.anatomy - Virtual hepatocystic-triangle anatomy for the B-mode simulator (bmode), with named variant test cases.

PHANTOM / ANATOMICAL FRAME (mm)
    origin : corner of the 100 x 100 mm active region on the phantom surface
    +x     : towards the patient's LEFT  (medial). Gallbladder is at small x, CBD/artery/portal vein at larger x.
    +y     : CAUDAL (towards the feet). Liver hilum at small y, distal CBD at large y.
    +z     : POSTERIOR = depth below the scanning surface (probe on the anterior surface).

    Zone y < liver edge (~52 mm at the surface, receding cranially with depth, i.e. a thin inferior margin):
        scanning through liver segments IV/V onto the gallbladder bed and hilum.
    Zone y > liver edge: scanning on the hepatoduodenal ligament / omental fat (background = fat).

Dimensions follow typical adult values (gallbladder ~80 x 30 mm, wall 1.5 mm, cystic duct ~3 mm x ~22 mm,
CHD ~5 mm, CBD 6 mm, proper hepatic artery ~4 mm, RHA ~3 mm, cystic artery ~1.8 mm, portal vein ~12 mm).
Relationships: CBD anterolateral (right), proper hepatic artery anteromedial (left), portal vein posterior
("Mickey Mouse" view); RHA crosses posterior to the CHD; cystic artery arises from the RHA inside Calot's triangle.

Usage:
    from orvue_us_inverse.simulation.anatomy import build_case, CASES
    an = build_case("parallel_cystic_duct")
"""
from dataclasses import dataclass, field
import numpy as np

# --------------------------------------------------------------------------------------
# Tissue table: label -> (name, impedance [MRayl], echogenicity, attenuation [dB/cm/MHz], BGR debug colour)
# --------------------------------------------------------------------------------------
TISSUES = {
    0:  ("liver",      1.65, 0.70,  0.50, (60, 60, 60)),
    1:  ("fat",        1.38, 1.10,  0.60, (80, 200, 230)),
    2:  ("gb_wall",    1.70, 1.60,  0.80, (40, 140, 40)),
    3:  ("bile",       1.52, 0.003, 0.05, (60, 220, 60)),
    4:  ("stone",      6.00, 2.50,  12.0, (255, 255, 255)),
    5:  ("duct_wall",  1.70, 1.50,  0.80, (40, 160, 40)),
    6:  ("art_wall",   1.75, 1.80,  1.00, (40, 40, 200)),
    7:  ("art_blood",  1.61, 0.008, 0.15, (60, 60, 255)),
    8:  ("vein_wall",  1.68, 1.20,  0.80, (200, 80, 40)),
    9:  ("vein_blood", 1.61, 0.008, 0.15, (255, 120, 60)),
    10: ("lymph_node", 1.60, 0.20,  0.60, (180, 120, 200)),
}
N_LAB = len(TISSUES)
Z_TAB = np.array([TISSUES[i][1] for i in range(N_LAB)], np.float32)
ECHO_TAB = np.array([TISSUES[i][2] for i in range(N_LAB)], np.float32)
ATT_TAB = np.array([TISSUES[i][3] for i in range(N_LAB)], np.float32)
COL_TAB = np.array([TISSUES[i][4] for i in range(N_LAB)], np.uint8)
# parenchymal texture strength (log-amplitude std of mm-scale echogenicity variation)
TEX_TAB = np.array([0.35, 0.60, 0.20, 0.0, 0.0, 0.20, 0.20, 0.0, 0.20, 0.0, 0.15], np.float32)
# density of sparse strong point scatterers (fraction of 0.1 mm cells), e.g. fibrous septa / portal triads
PT_TAB = np.array([0.004, 0.012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)


# --------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------
@dataclass
class Tube:
    """Hollow tube along a polyline. r_lumen may be a scalar or one radius per vertex (tapering)."""
    name: str
    pts: object
    r_lumen: object
    wall: float
    lumen_label: int
    wall_label: int

    def __post_init__(self):
        self.pts = np.asarray(self.pts, np.float32)
        r = np.asarray(self.r_lumen, np.float32)
        self.r = np.full(len(self.pts), float(r), np.float32) if r.ndim == 0 else r

    def sd(self, P):
        """Distance to the lumen surface (negative inside the lumen)."""
        best = np.full(P.shape[0], np.inf, np.float32)
        for i in range(len(self.pts) - 1):
            a, b = self.pts[i], self.pts[i + 1]
            ab = b - a
            AP = P - a
            t = np.clip((AP @ ab) / (ab @ ab), 0.0, 1.0)
            R = AP - t[:, None] * ab
            d = np.sqrt(np.einsum("ij,ij->i", R, R))
            best = np.minimum(best, d - (self.r[i] + (self.r[i + 1] - self.r[i]) * t))
        return best

    def bbox(self):
        m = float(self.r.max()) + self.wall
        return self.pts.min(0) - m, self.pts.max(0) + m


@dataclass
class Blob:
    """Solid ellipsoid (axis-aligned), e.g. a gallstone or a lymph node."""
    name: str
    centre: object
    radii: object
    label: int

    def __post_init__(self):
        self.centre = np.asarray(self.centre, np.float32)
        self.radii = np.broadcast_to(np.asarray(self.radii, np.float32), (3,)).copy()

    def inside(self, P):
        return (((P - self.centre) / self.radii) ** 2).sum(1) < 1.0

    def bbox(self):
        return self.centre - self.radii, self.centre + self.radii


@dataclass
class Anatomy:
    name: str = "normal"
    description: str = ""
    size_xyz: tuple = (100.0, 100.0, 50.0)
    fat_mean: float = 2.5          # superficial fat / peritoneum thickness (mm)
    fat_amp: float = 0.8
    liver_edge_y0: float = 52.0    # liver inferior edge: y_edge(x) = y0 + slope * (x - 50)
    liver_edge_slope: float = -0.1
    liver_edge_wedge: float = 0.5  # deeper liver ends more cranially (mm per mm depth): thin inferior margin
    tubes: list = field(default_factory=list)
    blobs: list = field(default_factory=list)

    @property
    def size_xy(self):
        return self.size_xyz[:2]

    def tube(self, name):
        return next(t for t in self.tubes if t.name == name)

    def labels(self, P):
        """Tissue label for each point P (N,3) in the phantom frame."""
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        lab = np.ones(P.shape[0], np.int8)                                   # fat / ligament background
        fat_t = self.fat_mean + self.fat_amp * np.sin(x / 7.0) * np.cos(y / 9.0)
        liver = y < (self.liver_edge_y0 + self.liver_edge_slope * (x - 50.0)
                     - self.liver_edge_wedge * np.maximum(z - fat_t, 0.0))
        lab[liver & (z >= fat_t)] = 0

        def near(prim):
            lo, hi = prim.bbox()
            return np.flatnonzero(np.all((P >= lo) & (P <= hi), axis=1))

        sds = [(t, idx, t.sd(P[idx])) for t in self.tubes for idx in [near(t)]]
        # walls first, then lumens (so connected lumens open into each other), then solid blobs
        for t, idx, d in sds:
            lab[idx[d < t.wall]] = t.wall_label
        for t, idx, d in sds:
            lab[idx[d < 0]] = t.lumen_label
        for b in self.blobs:
            idx = near(b)
            lab[idx[b.inside(P[idx])]] = b.label
        return lab


# --------------------------------------------------------------------------------------
# Normal anatomy (most common configuration)
# --------------------------------------------------------------------------------------
def _normal():
    T = []
    # Gallbladder as a tapered tube: fundus (caudal, lateral, superficial) -> body -> Hartmann's pouch -> neck.
    # ~80 mm long, up to 29 mm diameter (~40 ml), neck ~1.5 cm lateral to the CHD.
    T.append(Tube("gallbladder",
                  [(13, 78, 19), (22, 62.4, 22.9), (31, 46.8, 26.8), (37, 36.4, 29.4), (40.6, 30.2, 31.0), (43, 26, 32)],
                  [14, 14.5, 12, 8.5, 5.5, 3.5], 1.5, 3, 2))
    # Cystic duct: tortuous ~24 mm, calibre varies (spiral valves of Heister), lateral insertion, middle third of CHD
    T.append(Tube("cystic_duct",
                  [(43, 26, 32), (43, 31, 33), (46, 35, 32.5), (49.5, 38.5, 31), (52.5, 41.5, 28.5), (55, 44.5, 27)],
                  [1.4, 1.6, 1.2, 1.6, 1.3, 1.4], 0.5, 3, 5))
    # CHD from the hilum -> CBD caudally in the hepatoduodenal ligament (5 mm -> 6 mm)
    T.append(Tube("chd_cbd",
                  [(60, 8, 36), (58, 28, 30), (57, 45, 26), (56, 62, 23), (56, 80, 24), (57, 100, 28)],
                  [2.3, 2.4, 2.6, 3.0, 3.0, 3.0], 0.6, 3, 5))
    # Proper hepatic artery: anteromedial to the CBD, bifurcates near the hilum
    T.append(Tube("proper_hepatic_artery", [(65, 100, 26), (64.5, 75, 22), (64.5, 55, 23), (64.5, 40, 27)], 2.0, 0.6, 7, 6))
    T.append(Tube("left_hepatic_artery", [(64.5, 40, 27), (71, 30, 28), (78, 18, 30)], 1.4, 0.5, 7, 6))
    # Right hepatic artery: crosses POSTERIOR to the CHD into Calot's triangle, then into the right liver
    T.append(Tube("right_hepatic_artery",
                  [(64.5, 40, 27), (61, 36, 33), (59, 33, 35.5), (56, 28, 37), (52, 20, 39.5), (47, 12, 41), (42, 4, 42)],
                  1.6, 0.6, 7, 6))
    # Cystic artery: from the RHA inside Calot's triangle, posteromedial to the cystic duct, to the GB neck
    T.append(Tube("cystic_artery", [(56, 28, 37), (52, 30.5, 36.5), (48, 32.5, 36), (44.5, 33.5, 36.5)], 0.9, 0.4, 7, 6))
    # Portal vein: posterior, between CBD and artery; bifurcates deep at the hilum
    T.append(Tube("portal_vein", [(61, 100, 38), (60, 75, 34), (60, 55, 35), (60, 32, 47), (50, 20, 49)], 6.0, 0.4, 9, 8))
    T.append(Tube("left_portal_vein", [(60, 32, 47), (72, 24, 45), (82, 16, 43)], 4.5, 0.4, 9, 8))
    B = [Blob("gallstone_1", (19, 68, 29), 5.0, 4),
         Blob("gallstone_2", (14, 75, 27), 3.5, 4),
         # Calot's (cystic) node rests directly on the anterior surface of the cystic artery in Calot's
         # triangle, ~0.2 mm from its outer wall (centre straight anterior of the artery axis at
         # (51.59, 30.71, 36.45)); it clears the GB, cystic duct, CHD, RHA and portal vein by >= 0.6 mm.
         Blob("calot_lymph_node", (51.59, 30.71, 32.41), (3.5, 2.5, 2.5), 10)]
    return T, B


def _replace(T, name, new):
    return [new if t.name == name else t for t in T]


# --------------------------------------------------------------------------------------
# Cystic artery -> gallbladder connection and its superficial / deep branches
# --------------------------------------------------------------------------------------
BRANCH_R_LUMEN = 0.65          # mm, lumen radius of the superficial / deep branches
BRANCH_WALL = 0.3              # mm


def _gb_frame(gb, s):
    """Gallbladder centreline at arc length s measured from the NECK (last vertex) towards the fundus.
    Returns (centre, unit tangent towards the fundus, lumen radius, e1, e2): e1 is the direction
    perpendicular to the axis pointing most posterior (+z), e2 = tangent x e1."""
    pts, r = gb.pts[::-1].astype(np.float64), gb.r[::-1].astype(np.float64)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
    t = float(np.clip((s - cum[i]) / seg_len[i], 0.0, 1.0))
    tan = seg[i] / seg_len[i]
    e1 = np.array([0.0, 0.0, 1.0]) - tan[2] * tan
    e1 /= np.linalg.norm(e1)
    return pts[i] + t * seg[i], tan, r[i] + t * (r[i + 1] - r[i]), e1, np.cross(tan, e1)


def _gb_locate(gb, p, step=0.05):
    """(arc length from the neck, angle around the axis; 0 = posterior, pi = anterior) of point p."""
    total = float(np.linalg.norm(np.diff(gb.pts, axis=0), axis=1).sum())
    ss = np.arange(0.0, total, step)
    cs = np.array([_gb_frame(gb, s)[0] for s in ss])
    s = float(ss[np.argmin(np.linalg.norm(cs - p, axis=1))])
    c, _tan, _r, e1, e2 = _gb_frame(gb, s)
    d = np.asarray(p, np.float64) - c
    return s, float(np.arctan2(d @ e2, d @ e1))


def _gb_surface_point(gb, s, theta, offset):
    """Point at `offset` mm outside the lumen surface (offset = wall/2 -> middle of the GB wall)."""
    c, _tan, r, e1, e2 = _gb_frame(gb, s)
    return c + (r + offset) * (np.cos(theta) * e1 + np.sin(theta) * e2)


def _connect_to_gb(art, gb, step=0.05, max_mm=10.0):
    """Make the artery's lumen enter the GB wall. If its terminal vertex is outside the wall, extend it
    along its last segment until the new terminal vertex lies in the middle of the wall, tapering the
    tip to the branch radius. Arteries that already end inside the wall are returned unchanged."""
    p = art.pts[-1].astype(np.float64)
    if gb.sd(p[None].astype(np.float32))[0] <= gb.wall:
        return art
    d = art.pts[-1].astype(np.float64) - art.pts[-2]
    d /= np.linalg.norm(d)
    for k in range(1, int(max_mm / step) + 1):
        q = p + k * step * d
        if gb.sd(q[None].astype(np.float32))[0] <= gb.wall / 2:
            break
    else:
        raise ValueError(f"{art.name} does not reach the gallbladder wall within {max_mm} mm")
    pts = np.vstack([art.pts, q])
    r = np.append(art.r, min(BRANCH_R_LUMEN, float(art.r[-1])))
    return Tube(art.name, pts, r, art.wall, art.lumen_label, art.wall_label)


def _on_mid_wall(gb, s, theta):
    """Point at angle theta around the GB axis (at arc length s from the neck) where gb.sd = wall/2,
    i.e. in the middle of the wall. Found along the ray from the axis, so it also holds near the joins
    between the GB's tapered segments, where radius + wall/2 alone can land inside the next segment."""
    c, _tan, r, e1, e2 = _gb_frame(gb, s)
    d = np.cos(theta) * e1 + np.sin(theta) * e2
    lo, hi = 0.0, r + gb.wall + 10.0                     # sd < wall/2 at the axis, > wall/2 far out
    for _ in range(40):
        mid = (lo + hi) / 2
        if gb.sd((c + mid * d)[None].astype(np.float32))[0] < gb.wall / 2:
            lo = mid
        else:
            hi = mid
    return c + hi * d


def _surface_path(gb, s0, th0, dth, length, turn, step, tol=0.1, max_rounds=4):
    """Vertices of a path in the middle of the GB wall: arc length s0 -> s0 + length from the neck, angle
    th0 -> th0 + dth over the first `turn` mm. Segments whose midpoint strays more than `tol` mm from
    mid-wall are split (a few rounds), so curved stretches get more vertices and straight ones stay coarse."""
    def at(u):
        return _on_mid_wall(gb, s0 + u, th0 + dth * min(u / turn, 1.0))

    us = list(np.unique(np.concatenate([np.arange(0.0, length, step), [turn, length]])))
    pts = np.array([at(u) for u in us])
    for _ in range(max_rounds):
        mid = (pts[:-1] + pts[1:]) / 2
        bad = np.flatnonzero(np.abs(gb.sd(mid.astype(np.float32)) - gb.wall / 2) > tol)
        if len(bad) == 0:
            break
        us = sorted(us + [(us[i] + us[i + 1]) / 2 for i in bad])
        pts = np.array([at(u) for u in us])
    return pts


def _clearance(path, others, radius):
    """Smallest gap (mm) between a tube of outer radius `radius` along `path` and the other tubes' walls."""
    P = np.concatenate([a + np.linspace(0, 1, 8)[:, None] * (b - a) for a, b in zip(path[:-1], path[1:])])
    P = P.astype(np.float32)
    return min(float((t.sd(P) - t.wall).min()) for t in others) - radius


def _cystic_branches(gb, start, others, length=26.0, turn=12.0, step=3.0):
    """Superficial (posterior, free peritoneal surface) and deep (anterior, liver-bed side) branches of
    the cystic artery. Both start at the artery's terminal vertex at the GB neck, turn around the GB
    axis to the posterior / anterior side over the first `turn` mm, then follow the tapering GB
    surface towards the fundus in the middle of the wall (radius + wall/2 from the axis).
    Each branch goes round whichever side of the GB keeps it further from the `others` tubes."""
    s0, th0 = _gb_locate(gb, start)
    out = []
    for name, target in (("cystic_artery_superficial", 0.0), ("cystic_artery_deep", np.pi)):
        dth = (target - th0 + np.pi) % (2 * np.pi) - np.pi          # shortest way round the axis
        options = [dth, dth - np.sign(dth) * 2 * np.pi] if abs(dth) > np.pi / 4 else [dth]
        paths = [_surface_path(gb, s0, th0, d, length, turn, step) for d in options]
        best = max(paths, key=lambda p: _clearance(p, others, BRANCH_R_LUMEN + BRANCH_WALL))
        best[0] = start                                              # start exactly at the bifurcation
        out.append(Tube(name, best, BRANCH_R_LUMEN, BRANCH_WALL, 7, 6))
    return out


def _attach_cystic_arteries(T):
    """Connect the cystic artery (or the short cystic branch of a caterpillar hump) to the GB wall,
    and split the cystic artery into its superficial and deep branches at the neck."""
    gb = next(t for t in T if t.name == "gallbladder")
    names = [t.name for t in T]
    if "cystic_branch_1" in names:
        T = _replace(T, "cystic_branch_1", _connect_to_gb(T[names.index("cystic_branch_1")], gb))
    if "cystic_artery" in names:
        art = _connect_to_gb(T[names.index("cystic_artery")], gb)
        T = _replace(T, "cystic_artery", art)
        others = [t for t in T if t.name not in ("gallbladder", "cystic_artery")]
        T = T + _cystic_branches(gb, art.pts[-1], others)
    return T


# --------------------------------------------------------------------------------------
# Named test cases (prevalences are approximate ranges reported in MRCP / surgical series)
# --------------------------------------------------------------------------------------
CASES = {
    "normal": "Typical anatomy: lateral cystic duct insertion at the middle third of the CHD, cystic artery from "
              "the RHA inside Calot's triangle, RHA posterior to the CHD, two gallstones.",
    "parallel_cystic_duct": "Cystic duct runs parallel to the CHD/CBD for >2 cm and inserts low (reported ~1-25%, "
                            "typically ~7-10%). Classic bile-duct-injury trap: the CBD can be taken for the cystic duct.",
    "medial_spiral_insertion": "Cystic duct spirals posterior to the CHD and inserts on its medial side "
                               "(reported ~5-16%).",
    "short_cystic_duct": "Very short cystic duct (<5 mm, ~1%): GB neck lies almost against the CHD; "
                         "risk of clipping or tenting the CHD.",
    "anterior_cystic_artery": "Cystic artery arises from the proper hepatic artery and crosses ANTERIOR to the "
                              "CHD (reported ~2-8%).",
    "caterpillar_hump": "Tortuous right hepatic artery loops into Calot's triangle against the GB neck "
                        "(Moynihan's hump, reported ~3-13%), giving short cystic branches; risk of RHA injury.",
    "inflamed_obese": "Difficult case: acute cholecystitis with 5 mm wall, distended GB, stone impacted in "
                      "Hartmann's pouch, and 9 mm of fat over the field.",
    "choledocholithiasis": "Dilated CBD (8 mm) with a 6 mm stone in the distal CBD (shadowing in the duct).",
}


def build_case(name="normal"):
    if name not in CASES:
        raise KeyError(f"unknown case {name!r}; choose from {list(CASES)}")
    T, B = _normal()
    kw = {}

    if name == "parallel_cystic_duct":
        T = _replace(T, "cystic_duct", Tube("cystic_duct",
                     [(43, 26, 32), (44, 32, 33), (48.5, 40, 31.5), (51.3, 50, 27.5), (50.5, 62, 24.5), (50.5, 71, 24),
                      (55, 77, 24)], [1.4, 1.6, 1.3, 1.5, 1.3, 1.5, 1.4], 0.5, 3, 5))

    elif name == "medial_spiral_insertion":
        T = _replace(T, "cystic_duct", Tube("cystic_duct",
                     [(43, 26, 32), (44, 32, 31), (49, 37, 31), (55.5, 41, 32), (57.5, 44, 31.5), (61, 47, 29),
                      (58.5, 50, 25.8)], [1.4, 1.6, 1.3, 1.5, 1.3, 1.5, 1.4], 0.5, 3, 5))

    elif name == "short_cystic_duct":
        T = _replace(T, "gallbladder", Tube("gallbladder",
                     [(13, 78, 19), (22, 62.4, 22.9), (31, 46.8, 26.8), (38, 38, 29), (44, 35.5, 29), (49.5, 36.5, 28.5)],
                     [14, 14.5, 12, 8.5, 4.5, 2.6], 1.5, 3, 2))
        T = _replace(T, "cystic_duct", Tube("cystic_duct", [(49.5, 36.5, 28.5), (55.5, 37, 28)], 1.4, 0.5, 3, 5))
        T = _replace(T, "cystic_artery", Tube("cystic_artery",
                     [(56, 28, 37), (52, 30.5, 35.5), (47.5, 32.5, 33.5)], 0.9, 0.4, 7, 6))
        B = [b for b in B if b.name != "calot_lymph_node"]

    elif name == "anterior_cystic_artery":
        T = _replace(T, "cystic_artery", Tube("cystic_artery",
                     [(64.5, 55, 23), (61, 50, 19), (56, 46, 18.5), (50, 42, 20), (43, 39, 23)], 0.9, 0.4, 7, 6))
        # node rests on the anterior surface of this artery, close to where it reaches the GB
        B = [Blob("calot_lymph_node", (45.41, 40.03, 17.54), (3.5, 2.5, 2.5), 10) if b.name == "calot_lymph_node"
             else b for b in B]

    elif name == "caterpillar_hump":
        T = _replace(T, "right_hepatic_artery", Tube("right_hepatic_artery",
                     [(64.5, 40, 27), (61, 36, 33.5), (57.5, 33, 35.5), (53, 32, 33), (50, 30.5, 28.5), (50.5, 26, 30),
                      (52, 20, 38), (47, 12, 41), (42, 4, 42)], 1.7, 0.6, 7, 6))
        T = [t for t in T if t.name != "cystic_artery"]
        T.append(Tube("cystic_branch_1", [(50, 30.5, 28.5), (46.5, 30, 29)], 0.6, 0.3, 7, 6))
        B = [b for b in B if b.name != "calot_lymph_node"]

    elif name == "inflamed_obese":
        gb = T[0]
        shift = np.float32([[0, 0, 6], [0, 0, 5], [0, 0, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        r = gb.r * np.float32([1.1, 1.1, 1.08, 1.0, 1.0, 1.0])
        T = _replace(T, "gallbladder", Tube("gallbladder", gb.pts + shift, r, 5.0, 3, 2))
        B = [Blob("gallstone_1", (19, 68, 34.5), 5.0, 4), Blob("gallstone_2", (14, 75, 32.5), 3.5, 4),
             Blob("impacted_stone_hartmann", (37, 36.4, 31), 4.5, 4),
             # enlarged node (same 9 x 6 x 6 mm, long axis turned from x to y to fit beside the 5 mm GB wall),
             # resting on the antero-caudal side of the cystic artery, ~0.2 mm from its outer wall
             Blob("calot_lymph_node", (53.33, 35.91, 34.99), (3.0, 4.5, 3.0), 10)]
        kw = dict(fat_mean=9.0, fat_amp=1.5)

    elif name == "choledocholithiasis":
        T = _replace(T, "chd_cbd", Tube("chd_cbd",
                     [(60, 8, 36), (58, 28, 30), (57, 45, 26), (56, 62, 23), (56, 80, 24), (57, 100, 28)],
                     [2.3, 2.4, 2.6, 3.5, 3.8, 3.3], 0.7, 3, 5))
        B.append(Blob("cbd_stone", (56, 86, 24.8), 3.0, 4))

    T = _attach_cystic_arteries(T)
    return Anatomy(name=name, description=CASES[name], tubes=T, blobs=B, **kw)


# --------------------------------------------------------------------------------------
# Geometry check: report structures that intersect when they should not
# --------------------------------------------------------------------------------------
CONNECTED = {frozenset(p) for p in [
    ("gallbladder", "cystic_duct"), ("cystic_duct", "chd_cbd"),
    ("proper_hepatic_artery", "left_hepatic_artery"), ("proper_hepatic_artery", "right_hepatic_artery"),
    ("left_hepatic_artery", "right_hepatic_artery"),
    ("right_hepatic_artery", "cystic_artery"), ("cystic_artery", "gallbladder"),
    ("proper_hepatic_artery", "cystic_artery"),
    ("portal_vein", "left_portal_vein"),
    ("right_hepatic_artery", "cystic_branch_1"), ("right_hepatic_artery", "cystic_branch_2"),
    ("cystic_branch_1", "gallbladder"), ("cystic_branch_2", "gallbladder"),
    # superficial / deep branches of the cystic artery: run in the GB wall, split from the artery at the
    # neck, and touch each other at the bifurcation
    ("cystic_artery_superficial", "gallbladder"), ("cystic_artery_deep", "gallbladder"),
    ("cystic_artery_superficial", "cystic_artery"), ("cystic_artery_deep", "cystic_artery"),
    ("cystic_artery_superficial", "cystic_artery_deep"),
    # Calot's node rests on the cystic artery
    ("calot_lymph_node", "cystic_artery"),
]}


def _densify_xy(pts, step=0.25):
    """Polyline resampled every ~step mm, projected on the surface (x, y)."""
    out = [pts[0, :2]]
    for a, b in zip(pts[:-1, :2], pts[1:, :2]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out, np.float32)


def calot_triangle(an):
    """Calot's (hepatocystic) triangle projected on the phantom surface, as an (N, 2) polygon of (x, y) mm.

    Borders: the cystic duct from the gallbladder neck to where it meets the CHD in projection (lumens touching,
    so a duct passing behind the CHD is cut where it first reaches it), the CHD from there cranially to the
    hilum (its first vertex), and a straight line from the hilum back to the neck standing in for the inferior
    surface of the liver. Follows each case's own duct geometry, so a parallel or short cystic duct shows as a
    long thin or small triangle.
    """
    cd, chd = an.tube("cystic_duct"), an.tube("chd_cbd")
    cd_xy, chd_xy = _densify_xy(cd.pts), _densify_xy(chd.pts)
    d = np.linalg.norm(cd_xy[:, None] - chd_xy[None], axis=2)            # (cystic, CHD) distances
    touch = np.flatnonzero(d.min(1) < float(chd.r.mean() + cd.r.mean()))
    j = int(touch[0]) if len(touch) else len(cd_xy) - 1
    k = int(np.argmin(d[j]))
    return np.vstack([cd_xy[:j + 1], chd_xy[k::-1]])


def validate(an, res=0.4, min_gap=0.0):
    """Return list of (a, b, overlap_mm3, centre_xyz) for unconnected structures whose outer walls overlap."""
    sx, sy, sz = an.size_xyz
    g = [np.arange(0, s + res, res, dtype=np.float32) for s in (sx, sy, sz)]
    X, Y, Z = np.meshgrid(*g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    masks = {}
    for t in an.tubes:
        lo, hi = t.bbox()
        idx = np.flatnonzero(np.all((P >= lo) & (P <= hi), axis=1))
        masks[t.name] = set(idx[t.sd(P[idx]) < t.wall + min_gap].tolist())
    for b in an.blobs:
        lo, hi = b.bbox()
        idx = np.flatnonzero(np.all((P >= lo) & (P <= hi), axis=1))
        masks[b.name] = set(idx[b.inside(P[idx])].tolist())
    names = list(masks)
    bad = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if frozenset((a, b)) in CONNECTED:
                continue
            # stones and nodes may sit inside the gallbladder/duct they belong to
            if ("stone" in a or "stone" in b) and ({a, b} & {"gallbladder", "chd_cbd"}):
                continue
            ov = masks[a] & masks[b]
            if ov:
                c = P[list(ov)].mean(0).round(1).tolist()
                bad.append((a, b, round(len(ov) * res ** 3, 2), c))
    return bad


if __name__ == "__main__":
    for c in CASES:
        print(c, validate(build_case(c)))
