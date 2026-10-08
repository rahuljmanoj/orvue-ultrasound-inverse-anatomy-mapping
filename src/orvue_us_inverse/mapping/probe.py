"""
orvue_us_inverse.mapping.probe - the probe and simulator used for inverse mapping (the only probe source).

Every LinearProbe field is written out, so a change of the simulator's defaults cannot change this project.
Frames are 501 x 301 pixels (depth x lateral): 50 mm deep, 30 mm wide, 0.1 mm pixels.

    from orvue_us_inverse.mapping.probe import PROBE, make_simulator, probe_metadata
    sim = make_simulator("normal")
    img, lab = sim.render(T, return_labels=True)
"""
import inspect
from dataclasses import asdict

from orvue_us_inverse.simulation.anatomy import build_case
from orvue_us_inverse.simulation.bmode import BModeSimulator, LinearProbe

PROBE = LinearProbe(f0_mhz=7.5, width_mm=30.0, depth_mm=50.0, px_mm=0.1, fnum=3.0, cycles=2.5,
                    elev_sigma_mm=0.5, focus_mm=20.0, rayleigh_mm=10.0, c_mm_us=1.54)

# (rows, columns) = (depth, lateral) of every frame; the same grid as BModeSimulator (0 .. depth inclusive)
FRAME_SHAPE = (round(PROBE.depth_mm / PROBE.px_mm) + 1, round(PROBE.width_mm / PROBE.px_mm) + 1)


def simulator_settings(**kw) -> dict:
    """The BModeSimulator settings make_simulator(**kw) uses: the constructor defaults, kw, persistence 0.

    Anatomy and probe are not included (case name and probe_metadata() are stored separately). Tuples become
    lists so the result is JSON-ready.
    """
    params = inspect.signature(BModeSimulator.__init__).parameters
    unknown = set(kw) - set(params)
    if unknown:
        raise TypeError(f"unknown BModeSimulator arguments: {sorted(unknown)}")
    out = {name: p.default for name, p in params.items() if name not in ("self", "anatomy", "probe")}
    out.update(kw, persistence=0.0)
    return {k: list(v) if isinstance(v, tuple) else v for k, v in out.items()}


def make_simulator(case: str = "normal", **kw) -> BModeSimulator:
    """BModeSimulator for an anatomy case with PROBE and persistence 0 (each frame belongs to its own pose).

    Other BModeSimulator arguments (gain_db, dyn_range_db ...) pass through; persistence cannot be overridden.
    The returned simulator carries `case` and `settings` (simulator_settings(**kw)) for the sweep metadata.
    """
    if "persistence" in kw:
        raise TypeError("make_simulator: persistence is fixed at 0 for inverse mapping")
    sim = BModeSimulator(build_case(case), PROBE, persistence=0.0, **kw)
    sim.case, sim.settings = case, simulator_settings(**kw)
    return sim


def probe_metadata() -> dict:
    """Every LinearProbe field of PROBE as a dict (stored with each sweep)."""
    return asdict(PROBE)
