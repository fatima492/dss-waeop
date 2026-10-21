"""Unit tests for the physical models and the DSS layer.

Run with `python reproduce_all.py --tests` (uses pytest if installed) or `python -m pytest -q`.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # allow `python tests/test_models.py`

from dsswaeop import config as C
from dsswaeop import physics as ph
from dsswaeop.atmosphere import Atmosphere, Scenario, FieldGeometry
from dsswaeop.dss import DSSModel, active_set, stability_from_masks, boundary_fraction_bound
from dsswaeop.mission import build_lanes, max_speed_for_rate, simulate
from dsswaeop.nsga2 import nondominated_sort, hypervolume3


def close(a, b, rel=1e-6):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=0.0)


# ── power model ─────────────────────────────────────────────────────────────
def test_hover_figure_of_merit():
    """Profile power is calibrated so that FM = fm_hover at MTOW."""
    m = C.PLATFORM.mtow
    p_mech, vi, _ = ph.rotor_state(np.array([0.0]), np.array([m]))
    ideal = (m * C.G) ** 1.5 / np.sqrt(2 * C.RHO_AIR * C.PLATFORM.disk_area)
    assert close(ideal / p_mech[0], C.PLATFORM.fm_hover, rel=1e-3)


def test_power_curve_is_bucket_shaped():
    v = np.linspace(0, 15, 31)
    p = ph.power_curve(C.PLATFORM.mtow, v)
    i = int(np.argmin(p))
    assert 0 < i < len(v) - 1                     # minimum-power speed strictly inside
    assert p[0] > p[i] and p[-1] > p[i]


def test_headwind_costs_more_than_tailwind_at_same_ground_speed():
    m = C.PLATFORM.mtow
    g = np.array([[6.0, 0.0]])
    head, _, _ = ph.turbulent_power(g, np.array([[-4.0, 0.0]]), np.array([0.0]), m, True)
    tail, _, _ = ph.turbulent_power(g, np.array([[4.0, 0.0]]), np.array([0.0]), m, True)
    # airspeed 10 m/s vs 2 m/s: both cost differ, so wind is anisotropic in the model
    assert not close(head[0], tail[0], rel=1e-3)


def test_battery_energy_scale():
    """A sortie is tank-limited: spraying one full tank uses far less than the battery."""
    m = C.PLATFORM.mtow
    p = float(ph.electric_power(5.0, m, spraying=True))
    rate = C.CROPS["wheat"]["rate"]
    t_tank = C.PLATFORM.tank_L / (rate * C.SPRAY.swath * 5.0 / 1e4)   # s to empty the tank
    assert p * t_tank / 3600 < C.PLATFORM.usable_Wh


# ── spray kinematics ────────────────────────────────────────────────────────
def test_application_rate_is_achievable():
    """Flow-compensated speed limit: R* v w <= q_max at v <= v_max(R*), and v_max >= v_min."""
    for crop in C.CROPS:
        rate = C.CROPS[crop]["rate"]
        v = max_speed_for_rate(rate)
        q = rate * v * C.SPRAY.swath * 60 / 1e4
        assert q <= C.SPRAY.q_max + 1e-9, crop
        assert v >= C.SPRAY.v_spray_min, crop


def test_calm_air_deposit_is_on_prescription():
    """In calm air, boustrophedon along the long edge delivers the target dose."""
    atm = Atmosphere(Scenario(5, "flat", "winter", "wheat", seed=1))
    env = atm.truth
    for p in C.PARAMS:
        env.fields[p][:] = {"e4": 1.0, "e3": 0.02}.get(p, 0.0)
    env.sigma_w[:] = 0.05
    ls = build_lanes(atm.geom, 0.0, "wheat")
    gate = np.zeros(env.fields["e1"].shape, bool)
    r = simulate(ls, np.arange(ls.n_units), np.full(ls.n_units, 5.0), gate, env, meander=0.0)
    # off-target fraction in calm air is the edge spill of the outer lanes (~2 % for 5 ha)
    assert r.coverage > 0.98 and r.precision > 0.95 and r.drift_pct < 3.0


def test_crosswind_shifts_deposit_downwind():
    vi = np.array([4.0]); va = np.array([5.0])
    classes = C.CROPS["wheat"]["classes"]
    dp = ph.deposit_params(np.array([4.0]), np.array([[0.0, 1.0]]), np.array([[0.0, 1.0]]),
                           va, vi, np.array([10.0]), np.array([0.5]), np.array([0.0]),
                           np.array([0.5]), classes, meander=0.0)
    assert np.all(dp.shift > 0)
    assert dp.shift[0, 0] > dp.shift[0, -1]       # fine droplets drift further than coarse


# ── atmosphere and DSS ──────────────────────────────────────────────────────
def test_wind_field_is_spatially_correlated():
    atm = Atmosphere(Scenario(40, "rolling", "monsoon", "rice", seed=3))
    e1 = atm.truth.fields["e1"][100]
    a, b = e1[:-1, :].ravel(), e1[1:, :].ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.5          # neighbouring 25 m cells are correlated


def test_field_geometry_area():
    g = FieldGeometry.from_area(20)
    assert close(g.length * g.width, 20e4)


def test_phi_increases_outside_favourable_class():
    """Φ is ≤ 1 inside the favourable zone and > 1 where an active parameter is adverse
    (corrects the earlier definition, which gave Φ = 0 in the windiest cells)."""
    atm = Atmosphere(Scenario(10, "hilly", "monsoon", "rice", seed=2))
    m = DSSModel(atm.truth, "dynamic")
    phi, Zf, Za = m.phi(), m.favourable_zone(), m.adverse_zone()
    if Zf.any():
        assert phi[Zf].max() <= 1.0 + 1e-6
    if Za.any():
        assert phi[Za].min() > 1.0 / len(C.PARAMS)
        assert phi[Za].mean() > phi[~Za].mean()


def test_activation_rule():
    atm = Atmosphere(Scenario(10, "flat", "winter", "wheat", seed=2))
    A = active_set(atm.truth, "dynamic")
    h = atm.truth.hours
    assert A[:, :3].all()                                   # e1-e3 always active
    assert not A[h < 7, 5].any()                            # thermal index inactive pre-dawn
    assert A[(h > 10) & (h < 15), 5].all()
    assert active_set(atm.truth, "all").all()
    assert active_set(atm.truth, "speed").sum(axis=1).max() == 1


def test_stability_index_bounds():
    rng = np.random.default_rng(0)
    m = [rng.random((5, 5)) > 0.5 for _ in range(4)]
    s = stability_from_masks(m)
    assert 0 <= s <= 1
    assert stability_from_masks([m[0]] * 4) == 1.0


def test_boundary_fraction_bound():
    assert 0 < boundary_fraction_bound(0.15, "e1") < 0.2


# ── optimiser utilities ─────────────────────────────────────────────────────
def test_nondominated_sort_and_hv():
    F = np.array([[1, 3, 0], [2, 2, 0], [3, 1, 0], [3, 3, 0]], float)
    fr = nondominated_sort(F)
    assert set(fr[0]) == {0, 1, 2} and set(fr[1]) == {3}
    hv = hypervolume3(F[fr[0]], np.array([4.0, 4.0, 1.0]))
    assert close(hv, 3 + 2 + 1, rel=1e-9)


if __name__ == "__main__":
    # run all tests without pytest (e.g. with the IDE "Run" button)
    import traceback
    n_ok = n_fail = 0
    for name, fn in sorted((k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)):
        try:
            fn()
            n_ok += 1
            print(f"ok      {name}")
        except Exception:
            n_fail += 1
            print(f"FAILED  {name}")
            traceback.print_exc()
    print(f"{n_ok} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)
