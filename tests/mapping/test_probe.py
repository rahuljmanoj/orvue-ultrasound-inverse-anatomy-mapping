"""mapping/probe.py: one probe for the whole project, frames 501 x 301, persistence 0."""
import numpy as np
import pytest

from orvue_us_inverse.mapping.probe import PROBE, make_simulator, probe_metadata
from orvue_us_inverse.simulation.bmode import BModeSimulator

EXPECTED = dict(f0_mhz=7.5, width_mm=30.0, depth_mm=50.0, px_mm=0.1, fnum=3.0, cycles=2.5,
                elev_sigma_mm=0.5, focus_mm=20.0, rayleigh_mm=10.0, c_mm_us=1.54)


@pytest.fixture(scope="module")
def sim():
    return make_simulator("normal")


def test_probe_fields():
    assert probe_metadata() == EXPECTED
    assert all(getattr(PROBE, k) == v for k, v in EXPECTED.items())


def test_image_and_label_shapes(sim):
    T = BModeSimulator.pose_from_xy_yaw(50.0, 50.0, 0.0)
    img, lab = sim.render(T, return_labels=True)
    assert img.shape == lab.shape == (501, 301)
    assert img.dtype == np.uint8
    assert sim.labels_image(T).shape == (501, 301)


def test_persistence_zero(sim):
    assert sim.persistence == 0.0
    with pytest.raises(TypeError):
        make_simulator("normal", persistence=0.3)


def test_frame_independent_of_previous_pose(sim):
    """With the noise seeded, the image at B is the same whether or not another pose was rendered first."""
    A = BModeSimulator.pose_from_xy_yaw(25.0, 70.0, 90.0)
    B = BModeSimulator.pose_from_xy_yaw(50.0, 50.0, 0.0)
    fresh = make_simulator("normal")
    fresh._rng = np.random.default_rng(0)
    expected = fresh.render(B)
    sim.render(A)
    sim._rng = np.random.default_rng(0)
    assert np.array_equal(sim.render(B), expected)
