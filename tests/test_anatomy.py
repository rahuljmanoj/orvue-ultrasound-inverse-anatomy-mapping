"""Acceptance tests for the cholecystectomy anatomy (anatomy.py).

    python -m pytest tests -q
"""

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, distance_transform_edt

from orvue_us_inverse.simulation.anatomy import CASES, build_case, validate  # noqa: E402

BRANCHES = ("cystic_artery_superficial", "cystic_artery_deep")


def _names(an):
    return [t.name for t in an.tubes]


def _main_artery(an):
    """The artery that supplies the GB: the cystic artery, or the short cystic branch of a caterpillar hump."""
    for name in ("cystic_artery", "cystic_branch_1"):
        if name in _names(an):
            return an.tube(name)
    return None


def _node(an):
    return next((b for b in an.blobs if b.name == "calot_lymph_node"), None)


def _polyline_length(pts):
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _centreline_samples(gb, step=0.05):
    """Dense points along the gallbladder centreline."""
    out = []
    for a, b in zip(gb.pts[:-1], gb.pts[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
    return np.concatenate(out)


def _centreline_z_at_y(gb, p, tol=0.25):
    """GB centreline z at the y of point p. Where the centreline passes that y more than once (the neck can
    curve back), the crossing nearest to p is used; beyond the centreline's y-range, its nearest point."""
    cs = _centreline_samples(gb)
    same_y = cs[np.abs(cs[:, 1] - p[1]) <= tol]
    cand = same_y if len(same_y) else cs
    return float(cand[np.argmin(np.linalg.norm(cand - p, axis=1))][2])


def _ellipsoid_surface(blob, n_theta=200, n_phi=400):
    th, ph = np.meshgrid(np.linspace(0, np.pi, n_theta), np.linspace(0, 2 * np.pi, n_phi))
    unit = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3)
    return (unit * blob.radii + blob.centre).astype(np.float32)


CASES_WITH_ARTERY = [c for c in CASES if _main_artery(build_case(c)) is not None]
CASES_WITH_BRANCHES = [c for c in CASES if set(BRANCHES) <= set(_names(build_case(c)))]
CASES_WITH_NODE = [c for c in CASES if _node(build_case(c)) is not None and "cystic_artery" in _names(build_case(c))]


# T1 --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", list(CASES))
def test_t1_no_unexpected_intersections(case):
    assert validate(build_case(case)) == []


# T2 --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES_WITH_ARTERY)
def test_t2_cystic_artery_ends_in_gb_wall(case):
    an = build_case(case)
    gb, art = an.tube("gallbladder"), _main_artery(an)
    sd = float(gb.sd(art.pts[-1:])[0])
    assert sd <= gb.wall, f"{art.name} ends {sd - gb.wall:.2f} mm outside the GB wall"


# T3 --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES_WITH_BRANCHES)
def test_t3_branches_length_and_side(case):
    an = build_case(case)
    gb = an.tube("gallbladder")
    for name in BRANCHES:
        assert _polyline_length(an.tube(name).pts) >= 20.0, f"{name} shorter than 20 mm"
    sup, deep = an.tube("cystic_artery_superficial"), an.tube("cystic_artery_deep")
    sup_ref = np.mean([_centreline_z_at_y(gb, p) for p in sup.pts])
    deep_ref = np.mean([_centreline_z_at_y(gb, p) for p in deep.pts])
    assert sup.pts[:, 2].mean() > sup_ref, "superficial branch is not posterior (deeper) to the GB axis"
    assert deep.pts[:, 2].mean() < deep_ref, "deep branch is not anterior (shallower) to the GB axis"


# T4 --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES_WITH_NODE)
def test_t4_node_rests_on_cystic_artery(case):
    an = build_case(case)
    node, art = _node(an), an.tube("cystic_artery")
    S = _ellipsoid_surface(node)
    gap = art.sd(S) - art.wall
    assert gap.min() <= 0.5, f"node is {gap.min():.2f} mm from the cystic artery"
    # the node lies anterior to (shallower than) the artery where they meet
    contact = S[np.argmin(gap)]
    assert node.centre[2] < contact[2], "node is not anterior to the artery at the contact point"


# T5 --------------------------------------------------------------------------------------------------
def test_t5_render_plane_node_on_artery_and_artery_on_gb():
    from orvue_us_inverse.simulation.bmode import BModeSimulator          # imported, not modified
    sim = BModeSimulator(build_case("normal"))
    lab = sim.labels_image(sim.pose_from_xy_yaw(49.5, 31.7, -25))
    px = sim.pr.px_mm
    node, blood = lab == 10, lab == 7
    assert node.any() and blood.any(), "node or arterial blood missing from the plane"
    dist_to_blood = distance_transform_edt(~blood) * px
    assert dist_to_blood[node].min() <= 1.0, f"node is {dist_to_blood[node].min():.2f} mm from arterial blood"
    artery = (lab == 6) | (lab == 7)
    touching = binary_dilation(artery, structure=np.ones((3, 3), bool)) & (lab == 2)
    assert touching.any(), "artery labels do not touch the GB wall in this plane"
