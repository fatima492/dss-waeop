"""
Planners compared in the paper.

Decision vector of every NSGA-II-based planner, x ∈ [0,1)^(3+2B)  (Section 4.6):
  x[0]               lane heading ψ ∈ HEADINGS_DEG (geometry of the coverage path)
  x[1]               global spraying ground speed s ∈ [v_min, v_max(R*)]
  x[2]               swath-offset compensation fraction c ∈ [0,1]: every lane of block b
                     is flown c·Δ̂_b upwind of its nominal centre line, where Δ̂_b is the
                     forecast cross-track drift displacement (aerial-application practice)
  x[3 .. 3+B)        block priority keys (random-key permutation of the B lane blocks)
  x[3+B .. 3+2B)     block speed offsets δ_b ∈ [−1,1]: v_b = v_min + clip(s + 0.25 δ_b)(v_max−v_min)
Objectives (minimised): f1 = energy per treated hectare [Wh/ha], f2 = −spray precision,
f3 = −coverage, all evaluated by simulating the full mission on the forecast.

Methods
  BST       boustrophedon along the long field edge at 5 m/s; wind-unaware
  WA-BST    boustrophedon with lanes aligned to the forecast wind axis
  EA-CPP    energy-aware coverage: heading and speed minimising calm-air mission energy
  GRD       greedy receding horizon: at every refill picks the block with the least measured
            crosswind (plus distance); skips lanes where spraying is not permitted
  NSGA-C    NSGA-II on the continuous forecast fields; wind-speed gate; no DSS
  T-FSS     NSGA-II with a fixed-parameter (A = E) temporal fuzzy-soft-set gate
  DSS-WAEOP NSGA-II + DSS gate with dynamic A_t + DSS-biased mutation + lane skipping +
            σ-triggered replanning
  Ablations A1: A_t = {e1};  A2: no skipping/replanning (same plan as DSS-WAEOP);
            A3: no biased mutation;  A4: A_t = E (no dynamic activation)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):  # also allow `python dsswaeop/<module>.py` (e.g. the IDE "Run" button)
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "dsswaeop"

from . import config as C
from . import physics as ph
from .atmosphere import Atmosphere, Environment
from .dss import DSSModel, stability_from_masks
from .mission import LaneSet, Policy, build_lanes, max_speed_for_rate, simulate, MissionResult
from .nsga2 import NSGA2, nondominated_sort, hypervolume3

B = C.N_BLOCKS
NH = len(C.HEADINGS_DEG)
IH, IS, IC, IK, IV = 0, 1, 2, 3, 3 + B       # gene offsets
N_VAR = 3 + 2 * B
SPEED_SPREAD = 0.25


def _medium_tf(v: float, mass: float) -> float:
    """Fall time (s) of the median droplet class released at ground speed v in calm air."""
    _, vi, _ = ph.rotor_state(np.array([v]), np.array([mass]))
    chi = np.arctan2(v, max(float(vi[0]), 1e-3))
    vz = C.settling_velocity(200) + C.SPRAY.c_downwash * float(vi[0]) * np.cos(chi)
    return C.SPRAY.height / vz


# ─────────────────────────────────────────────────────────────────────────────
# Shared context for one scenario
# ─────────────────────────────────────────────────────────────────────────────
class Context:
    def __init__(self, atm: Atmosphere, thr_scale: float = 1.0, noise_scale: float = 1.0,
                 meander: Optional[float] = None):
        self.atm = atm
        self.geom = atm.geom
        self.crop = atm.sc.crop
        self.rate = C.CROPS[self.crop]["rate"]
        self.vmax = max_speed_for_rate(self.rate)
        self.vmin = C.SPRAY.v_spray_min
        self.k0 = C.step_of_hour(C.OP_START_H)
        self.k1 = C.step_of_hour(C.OP_END_H)
        self.thr_scale = thr_scale
        self.meander = meander
        self.truth = atm.truth
        self.forecast = atm.forecast(self.k0)
        self.measured = atm.measured(noise_scale=noise_scale)
        self._lanes: Dict[float, LaneSet] = {}
        self._gates: Dict[Tuple[str, str], np.ndarray] = {}
        self._dss: Dict[Tuple[str, str], DSSModel] = {}
        self._bias_cache: Dict[Tuple[str, float], Tuple[np.ndarray, np.ndarray]] = {}
        self._ucross: Dict[float, np.ndarray] = {}
        self._tf_v = np.linspace(self.vmin, self.vmax, 9)
        mid_mass = C.PLATFORM.m_dry + 0.5 * C.PLATFORM.tank_L
        self._tf = np.array([_medium_tf(v, mid_mass) for v in self._tf_v])

    def lanes(self, heading: float) -> LaneSet:
        if heading not in self._lanes:
            self._lanes[heading] = build_lanes(self.geom, heading, self.crop, seed=self.atm.sc.seed)
        return self._lanes[heading]

    def env(self, view: str) -> Environment:
        return {"forecast": self.forecast, "measured": self.measured, "truth": self.truth}[view]

    def dss(self, view: str, activation: str) -> DSSModel:
        key = (view, activation)
        if key not in self._dss:
            self._dss[key] = DSSModel(self.env(view), activation, thr_scale=self.thr_scale)
        return self._dss[key]

    def gate(self, view: str, rule: str) -> np.ndarray:
        """True where spraying is not permitted.

        rule 'wind'          : regulatory wind-speed limit (e1 adverse) or gust speed > limit
        rule 'tfss'          : adverse class of ANY of the six parameters (fixed A = E), hard
        rule 'dss:<act>'     : regulatory rule, plus DSS temporal deferral: a cell in the
                               adverse zone of the active parameters is deferred only while
                               the DSS forecast still shows a non-adverse slot for it before
                               OP_END − DEFER_MARGIN_H
        """
        key = (view, rule)
        if key not in self._gates:
            g = _gate(self, view, rule)
            self._gates[key] = g
        return self._gates[key]

    # ── predicted cross-track wind per block (for swath-offset compensation) ─
    def block_ucross(self, heading: float, env: Optional[Environment] = None,
                     k_from: Optional[int] = None) -> np.ndarray:
        if env is None and heading in self._ucross:
            return self._ucross[heading]
        env = env or self.forecast
        k0 = self.k0 if k_from is None else k_from
        ls = self.lanes(heading)
        wto = env.wind_to_unit()[k0:self.k1 + 1]                     # (t,2)
        out = np.zeros(len(ls.block_lanes))
        for b in range(len(ls.block_lanes)):
            us = np.flatnonzero(ls.u_block == b)
            cells = np.unique(ls.u_cell[us], axis=0)
            U = env.fields["e1"][k0:self.k1 + 1, cells[:, 0], cells[:, 1]].mean(axis=1)
            e5 = env.fields["e5"][k0:self.k1 + 1, cells[:, 0], cells[:, 1]].mean(axis=1)
            ueff = np.maximum(U - 0.5 * e5 * C.SPRAY.height, 0)
            out[b] = float(np.mean(ueff * (wto @ ls.n)))
        if k_from is None and env is self.forecast:
            self._ucross[heading] = out
        return out

    # ── decision-vector decoding ────────────────────────────────────────────
    def decode(self, x: np.ndarray, heading: Optional[float] = None,
               ucross: Optional[np.ndarray] = None):
        h = C.HEADINGS_DEG[min(int(x[IH] * NH), NH - 1)] if heading is None else heading
        ls = self.lanes(h)
        nb = len(ls.block_lanes)
        keys = x[IK:IK + nb]
        delta = 2.0 * x[IV:IV + nb] - 1.0
        frac = np.clip(x[IS] + SPEED_SPREAD * delta, 0.0, 1.0)
        vb = self.vmin + frac * (self.vmax - self.vmin)
        order = np.concatenate([np.flatnonzero(ls.u_block == b) for b in np.argsort(keys, kind="stable")])
        speeds = vb[ls.u_block]
        uc = self.block_ucross(h) if ucross is None else ucross
        tf = np.interp(vb, self._tf_v, self._tf)
        offsets = (-x[IC] * uc * tf)[ls.u_block]
        return ls, order, speeds, offsets

    def encode(self, heading: float, speed_frac: float = 0.5, comp: float = 0.0) -> np.ndarray:
        x = np.full(N_VAR, 0.5)
        x[IH] = (C.HEADINGS_DEG.index(heading) + 0.5) / NH
        x[IS] = speed_frac
        x[IC] = comp
        x[IK:IK + B] = np.linspace(0.05, 0.95, B)
        return x

    # ── DSS-biased mutation ingredients (per heading and activation) ───────
    def block_bias(self, activation: str, heading: float):
        key = (activation, heading)
        if key not in self._bias_cache:
            m = self.dss("forecast", activation)
            self._bias_cache[key] = _block_tau_phi(self, self.lanes(heading), m.adverse_zone(),
                                                   m.phi(), self.k0)
        return self._bias_cache[key]


def _block_tau_phi(ctx: Context, ls: LaneSet, Za, Phi, k_from: int):
    """Per-block deadline τ_b (steps until the adverse zone Z_a covers > 50 % of the block's
    cells) and mean energy potential Φ_b over the remaining operating window."""
    nb = len(ls.block_lanes)
    tau, phib = np.zeros(nb), np.zeros(nb)
    for b in range(nb):
        us = np.flatnonzero(ls.u_block == b)
        cells = np.unique(ls.u_cell[us], axis=0)
        frac = Za[k_from:ctx.k1 + 1, cells[:, 0], cells[:, 1]].mean(axis=1)
        bad = np.flatnonzero(frac > 0.5)
        tau[b] = bad[0] if len(bad) else ctx.k1 - k_from
        phib[b] = float(Phi[k_from:ctx.k1 + 1, cells[:, 0], cells[:, 1]].mean())
    return tau, phib


def _bias_operator(tau, phib, nb):
    """DSS-biased mutation (Definition 5.1).

    Biased step, one of (equal probability):
      (i)  re-key the blocks earliest-deadline-first (EDF): blocks that the DSS forecast
           shows entering the adverse zone Z_a soonest are sprayed first;
      (ii) set block speed offsets from Φ_b: slower (more downwash, shorter droplet fall
           time) where the energy potential indicates marginal conditions.
    """
    tmax = max(float(tau.max()), 1e-9)
    z = (phib - phib.mean()) / (phib.std() + 1e-9)

    def op(x, rng):
        y = x.copy()
        if rng.random() < 0.5:
            y[IK:IK + nb] = np.clip(tau / tmax + rng.normal(0, 0.05, nb), 0, 1 - 1e-9)
        else:
            y[IV:IV + nb] = np.clip(0.5 - 0.25 * z + rng.normal(0, 0.1, nb), 0, 1 - 1e-9)
        return y
    return op


# ─────────────────────────────────────────────────────────────────────────────
# Execution policies
# ─────────────────────────────────────────────────────────────────────────────
class SkipPolicy(Policy):
    skip_gated = True


class GreedyPolicy(Policy):
    """GRD: at each refill choose the block with the least measured crosswind + distance."""
    skip_gated = True

    def __init__(self, ctx: Context, ls: LaneSet):
        self.ctx, self.ls = ctx, ls

    def on_base(self, t_s, queue, speeds, state):
        ls, env = self.ls, self.ctx.measured
        k = min(max(int((t_s / 3600 - C.DAY_START_H) * 60 / C.DT_MIN), 0), C.N_STEPS - 1)
        cross = abs(float(env.wind_to_unit()[k] @ ls.n))
        rem = np.array(queue)
        blocks = np.unique(ls.u_block[rem])
        if len(blocks) <= 1:
            return None
        best, bscore = None, np.inf
        for b in blocks:
            us = rem[ls.u_block[rem] == b]
            cx, cy = ls.u_cell[us, 0], ls.u_cell[us, 1]
            u = float(env.fields["e1"][k, cx, cy].mean())
            dist = float(np.min(np.hypot(ls.u_start[us, 0], ls.u_start[us, 1])))
            score = u * cross + dist / 500.0
            if score < bscore:
                best, bscore = b, score
        if ls.u_block[queue[0]] == best:
            return None
        first = [u for u in queue if ls.u_block[u] == best]
        return first + [u for u in queue if ls.u_block[u] != best], speeds


class DSSReplanPolicy(Policy):
    """Module 4: σ-triggered warm-start re-optimisation of the remaining blocks."""
    skip_gated = True

    def __init__(self, ctx: Context, ls: LaneSet, activation: str, x_best: np.ndarray,
                 rng: np.random.Generator, biased: bool = True, sigma_min: float = C.OPT.sigma_min):
        self.ctx, self.ls, self.activation = ctx, ls, activation
        self.x = x_best.copy()
        self.rng = rng
        self.biased = biased
        self.sigma_min = sigma_min
        self.last_replan = -1e9
        self.last_k = -1
        self.meas = ctx.dss("measured", activation)
        self._Wm = {p: self.meas.W(p) for p in C.PARAMS}
        self.triggers = 0
        self.sigma_log = []
        self.offsets = None
        self.replan_s = []

    def sigma_hat(self, k: int) -> float:
        """min over active e of σ̂(e) on measured favourable masks over the sliding window
        (window GATE: C.OPT.sigma_window steps), computed on the cells of the remaining work."""
        W = C.OPT.sigma_window
        A = self.meas.A[k]
        vals = []
        for i, p in enumerate(C.PARAMS):
            if A[i]:
                m = self._Wm[p][max(k - W, 0):k + 1]
                vals.append(stability_from_masks(list(m)))
        return min(vals) if vals else 1.0

    def _online_offsets(self, k: int, speeds: np.ndarray):
        """Refresh the upwind swath offsets of every unit from the crosswind measured now."""
        ctx, ls = self.ctx, self.ls
        env = ctx.measured
        wto = env.wind_to_unit()[k]
        cross = float(wto @ ls.n)
        U = env.fields["e1"][k][ls.u_cell[:, 0], ls.u_cell[:, 1]].astype(float)
        e5 = env.fields["e5"][k][ls.u_cell[:, 0], ls.u_cell[:, 1]].astype(float)
        ueff = np.maximum(U - 0.5 * e5 * C.SPRAY.height, 0.0)
        # block-mean crosswind (smooths the sensor network) times fall time at the unit speed
        uc = np.zeros(len(ls.block_lanes))
        for b in range(len(uc)):
            uc[b] = ueff[ls.u_block == b].mean() * cross
        tf = np.interp(speeds, ctx._tf_v, ctx._tf)
        self.offsets = -self.x[IC] * uc[ls.u_block] * tf

    def on_base(self, t_s, queue, speeds, state):
        ctx, ls = self.ctx, self.ls
        k = min(max(int((t_s / 3600 - C.DAY_START_H) * 60 / C.DT_MIN), 0), C.N_STEPS - 1)
        if k == self.last_k:
            return None
        self.last_k = k
        self._online_offsets(k, speeds)
        s = self.sigma_hat(k)
        self.sigma_log.append((t_s / 3600, s))
        if s >= self.sigma_min or t_s - self.last_replan < C.OPT.replan_min_gap_h * 3600:
            return None
        if len(set(ls.u_block[queue])) < 2:
            return None
        self.last_replan = t_s
        self.triggers += 1
        _t0 = time.time()
        now = ctx.atm.nowcast(k)
        nmodel = DSSModel(now, self.activation, thr_scale=ctx.thr_scale)
        gate = _dss_gate(now, nmodel, nmodel, ctx.thr_scale)
        done = state["sprayed"].copy()
        heading = ls.heading
        uc = ctx.block_ucross(heading, env=now, k_from=k)

        def evaluate(xx):
            _, order, sp, off = ctx.decode(xx, heading, uc)
            r = simulate(ls, order, sp, gate, now, t_start_s=t_s, policy=SkipPolicy(),
                         deposit="fast", done_mask=done, xi_set=1, offsets=off,
                         meander=0.0)
            return _objectives(r)

        bias = None
        if self.biased:
            tau, phib = _block_tau_phi(ctx, ls, nmodel.adverse_zone(), nmodel.phi(), k)
            bias = _bias_operator(tau, phib, len(ls.block_lanes))
        X0 = np.vstack([self.x] + [_jitter(self.x, self.rng) for _ in range(C.OPT.replan_pop - 1)])
        opt = NSGA2(N_VAR, evaluate, C.OPT.replan_pop, C.OPT.replan_gens, self.rng,
                    biased=bias, p_bias=C.OPT.p_bias)
        X, F = opt.run(X0)
        self.x = select_solution(X, F)
        _, order, sp, off = ctx.decode(self.x, heading, uc)
        self._online_offsets(k, sp)
        self.replan_s.append(time.time() - _t0)
        rem = set(queue)
        return [u for u in order if u in rem], sp


def _regulatory(env: Environment, thr_scale: float = 1.0) -> np.ndarray:
    gust = env.fields["e1"] * env.fields["e4"] > C.PLATFORM.gust_limit
    return (env.fields["e1"] > C.THRESHOLDS["e1"][2] * thr_scale) | gust


def _dss_gate(env: Environment, model_now: DSSModel, model_fc: DSSModel, thr_scale: float):
    k_last = C.step_of_hour(C.OP_END_H - C.DEFER_MARGIN_H)
    return _regulatory(env, thr_scale) | (model_now.adverse_zone() & model_fc.future_window(k_last))


def _gate(ctx: "Context", view: str, rule: str) -> np.ndarray:
    env = ctx.env(view)
    if rule == "wind":
        return _regulatory(env, ctx.thr_scale)
    if rule == "none":
        return np.zeros(env.fields["e1"].shape, dtype=bool)
    if rule == "tfss":
        return _regulatory(env, ctx.thr_scale) | ctx.dss(view, "all").adverse_zone()
    act = rule.split(":")[1]
    return _dss_gate(env, ctx.dss(view, act), ctx.dss("forecast", act), ctx.thr_scale)


def _jitter(x, rng):
    y = x.copy()
    y[IS:] = np.clip(y[IS:] + rng.normal(0, 0.12, len(y) - IS), 0, 1 - 1e-9)
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Optimisation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _objectives(r: MissionResult) -> np.ndarray:
    e = r.energy_per_treated_ha if np.isfinite(r.energy_per_treated_ha) else 1e4
    return np.array([e, -r.precision, -r.coverage])


def select_solution(X, F):
    """Executed plan (knee rule, Section 5.2): the first-front solution minimising
        sqrt( (E/E_min − 1)^2 + (P_max − P)^2 + (C_max − C)^2 ),
    i.e. 10 % extra energy is traded one-for-one against 10 points of precision or of
    coverage. E, P, C are the plan-stage (forecast) objective values."""
    front = nondominated_sort(F)[0]
    Ff, Xf = F[front], X[front]
    E, P_, Cv = Ff[:, 0], -Ff[:, 1], -Ff[:, 2]
    d = np.sqrt((E / max(E.min(), 1e-9) - 1) ** 2 + (P_.max() - P_) ** 2 + (Cv.max() - Cv) ** 2)
    return Xf[int(np.argmin(d))]


@dataclass
class PlanOutcome:
    method: str
    result: MissionResult
    heading: float
    plan_time_s: float
    n_eval: int = 0
    front: Optional[np.ndarray] = None
    front_X: Optional[np.ndarray] = None
    hv_history: Optional[List[float]] = None
    triggers: int = 0
    extra: dict = field(default_factory=dict)


HV_REF = np.array([1500.0, 1.0, 1.0])     # (Wh/ha, 1 − precision, 1 − coverage)


def hv_of(F: np.ndarray) -> float:
    G = np.column_stack([F[:, 0], 1 + F[:, 1], 1 + F[:, 2]])
    return hypervolume3(G, HV_REF) / (HV_REF[0] * 1.0 * 1.0)


def run_nsga(ctx: Context, gate_rule: str, activation: Optional[str], biased: bool,
             skip: bool, seed: int, pop=C.OPT.pop, gens=C.OPT.gens, track_hv=False):
    """Plan on the forecast. Returns (selected x, final X, final F, optimiser)."""
    rng = np.random.default_rng(seed)
    gate = ctx.gate("forecast", gate_rule)
    env = ctx.forecast
    pol = SkipPolicy() if skip else Policy()

    def evaluate(x):
        ls, order, sp, off = ctx.decode(x)
        # planners predict the expected (ensemble-mean) deposit: meander fraction 0
        r = simulate(ls, order, sp, gate, env, policy=pol, deposit="fast", xi_set=1,
                     offsets=off, meander=0.0)
        return _objectives(r)

    bias = None
    if biased:
        def bias(x, rng_):
            h = C.HEADINGS_DEG[min(int(x[IH] * NH), NH - 1)]
            nb = len(ctx.lanes(h).block_lanes)
            tau, phib = ctx.block_bias(activation, h)
            return _bias_operator(tau, phib, nb)(x, rng_)

    # identical seeds for every NSGA-II planner: long-edge and wind-aligned boustrophedon
    # at slow / medium / fast speed, without offset compensation
    seeds = [ctx.encode(h, sf) for h in (0.0, wind_aligned_heading(ctx)) for sf in (0.0, 0.5, 1.0)]
    opt = NSGA2(N_VAR, evaluate, pop, gens, rng, p_cx=C.OPT.p_cx, p_mut=C.OPT.p_mut,
                biased=bias, p_bias=C.OPT.p_bias, seeds=seeds,
                hv_ref=hv_of if track_hv else None)
    X, F = opt.run()
    return select_solution(X, F), X, F, opt


def wind_aligned_heading(ctx: Context) -> float:
    """Lane heading closest to the forecast mean wind axis over the operating window."""
    v = ctx.forecast.wind_to_unit()[ctx.k0:ctx.k1 + 1].mean(axis=0)
    ang = np.rad2deg(np.arctan2(v[1], v[0])) % 180.0
    cand = np.array(C.HEADINGS_DEG, dtype=float)
    diff = np.minimum(np.abs(cand - ang), 180 - np.abs(cand - ang))
    return float(cand[int(np.argmin(diff))])


# ─────────────────────────────────────────────────────────────────────────────
# Method runners
# ─────────────────────────────────────────────────────────────────────────────
def _execute(ctx, ls, order, speeds, gate_rule, policy, keep=False, offsets=None) -> MissionResult:
    gate = ctx.gate("measured", gate_rule)
    return simulate(ls, order, speeds, gate, ctx.truth, policy=policy, deposit="full",
                    keep_dose=keep, keep_paths=keep, offsets=offsets, meander=ctx.meander)


def run_bst(ctx: Context, heading=0.0, name="BST", keep=False) -> PlanOutcome:
    t0 = time.time()
    ls = ctx.lanes(heading)
    v = min(5.0, ctx.vmax)
    r = _execute(ctx, ls, np.arange(ls.n_units), np.full(ls.n_units, v), "wind", Policy(), keep)
    return PlanOutcome(name, r, heading, time.time() - t0)


def run_wabst(ctx: Context, keep=False) -> PlanOutcome:
    return run_bst(ctx, wind_aligned_heading(ctx), "WA-BST", keep)


def _calm_env(ctx: Context) -> Environment:
    f = {p: np.zeros_like(ctx.forecast.fields[p]) for p in C.PARAMS}
    f["e4"][:] = 1.0
    f["e3"][:] = C.PARAM_RANGE["e3"][0]
    return Environment(ctx.geom, ctx.atm.sc.season, ctx.atm.sc.terrain, ctx.crop,
                       ctx.forecast.hours, f, ctx.forecast.dir_from,
                       np.full_like(ctx.forecast.sigma_w, 0.05), "calm")


def run_eacpp(ctx: Context, keep=False) -> PlanOutcome:
    """Energy-aware coverage path (wind-unaware): heading and speed minimising the calm-air
    energy of the full mission (spraying, turns, ferries and sorties)."""
    t0 = time.time()
    calm = _calm_env(ctx)
    nogate = np.zeros_like(ctx.gate("forecast", "wind"))
    best, best_e = None, np.inf
    for h in C.HEADINGS_DEG:
        ls = ctx.lanes(h)
        for v in np.linspace(ctx.vmin, ctx.vmax, 5):
            r = simulate(ls, np.arange(ls.n_units), np.full(ls.n_units, v), nogate, calm,
                         deposit="fast", xi_set=1, meander=0.0)
            e = r.energy_per_treated_ha
            if e < best_e:
                best, best_e = (h, v), e
    h, v = best
    ls = ctx.lanes(h)
    r = _execute(ctx, ls, np.arange(ls.n_units), np.full(ls.n_units, v), "wind", Policy(), keep)
    return PlanOutcome("EA-CPP", r, h, time.time() - t0, extra=dict(speed=v))


def run_grd(ctx: Context, keep=False) -> PlanOutcome:
    t0 = time.time()
    h = wind_aligned_heading(ctx)
    ls = ctx.lanes(h)
    v = min(5.0, ctx.vmax)
    r = _execute(ctx, ls, np.arange(ls.n_units), np.full(ls.n_units, v), "wind",
                 GreedyPolicy(ctx, ls), keep)
    return PlanOutcome("GRD", r, h, time.time() - t0)


#: NSGA-based method specifications
SPECS = {
    "NSGA-C":    dict(rule="wind",        act=None,      biased=False, skip=False, replan=False),
    "T-FSS":     dict(rule="tfss",        act="all",     biased=False, skip=False, replan=False),
    "DSS-WAEOP": dict(rule="dss:dynamic", act="dynamic", biased=True,  skip=True,  replan=True),
    "A1":        dict(rule="dss:speed",   act="speed",   biased=True,  skip=True,  replan=True),
    "A3":        dict(rule="dss:dynamic", act="dynamic", biased=False, skip=True,  replan=True),
    "A4":        dict(rule="dss:all",     act="all",     biased=True,  skip=True,  replan=True),
}


def run_nsga_method(ctx: Context, name: str, seed: int, keep=False, track_hv=False,
                    also_no_replan: bool = False, sigma_min: Optional[float] = None,
                    pop=C.OPT.pop, gens=C.OPT.gens) -> List[PlanOutcome]:
    sp = SPECS[name]
    t0 = time.time()
    xb, X, F, opt = run_nsga(ctx, sp["rule"], sp["act"], sp["biased"], sp["skip"], seed,
                             pop=pop, gens=gens, track_hv=track_hv)
    t_plan = time.time() - t0
    ls, order, speeds, off = ctx.decode(xb)
    front_idx = nondominated_sort(F)[0]
    if sp["replan"]:
        pol = DSSReplanPolicy(ctx, ls, sp["act"], xb, np.random.default_rng(seed + 7),
                              biased=sp["biased"],
                              sigma_min=C.OPT.sigma_min if sigma_min is None else sigma_min)
    elif sp["skip"]:
        pol = SkipPolicy()
    else:
        pol = Policy()
    t1 = time.time()
    r = _execute(ctx, ls, order, speeds, sp["rule"], pol, keep, offsets=off)
    t_exec = time.time() - t1
    outs = [PlanOutcome(name, r, ls.heading, t_plan, opt.n_eval, F[front_idx], X[front_idx],
                        opt.history, getattr(pol, "triggers", 0),
                        extra=dict(x=xb, plan_s=t_plan, exec_s=t_exec,
                                   sigma_log=getattr(pol, "sigma_log", None),
                                   replan_s=getattr(pol, "replan_s", []),
                                   F_all=F, hv=hv_of(F[front_idx])))]
    if also_no_replan:
        r2 = _execute(ctx, ls, order, speeds, sp["rule"], Policy(), keep, offsets=off)
        outs.append(PlanOutcome("A2", r2, ls.heading, t_plan, opt.n_eval))
    return outs


if __name__ == "__main__":
    # Short demonstration: fly the boustrophedon baseline (BST) on one 5 ha scenario.
    from .atmosphere import Scenario
    ctx = Context(Atmosphere(Scenario(5, "flat", "winter", "wheat", seed=1)))
    out = run_bst(ctx)
    r = out.result
    print("dsswaeop.planners: BST mission on a 5 ha field")
    print(f"  energy per treated ha {r.energy_per_treated_ha:.0f} Wh/ha, coverage {100 * r.coverage:.1f}%, "
          f"precision {100 * r.precision:.1f}%, {r.sorties} sorties, {r.duration_h:.2f} h")
    print("  (the full comparison of all planners is run by reproduce_all.py)")
