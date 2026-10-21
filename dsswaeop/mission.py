"""
Coverage geometry and mission simulation.

Geometry.  For a lane heading ψ the field rectangle is covered by parallel spray
lanes spaced one swath apart.  Lanes are grouped into N_BLOCKS contiguous blocks;
lanes are split into pieces of at most 100 m ("units") so that a sortie can stop mid-lane and resume after refilling (breakpoint continuation).
A unit is the smallest piece of work: it is sprayed in one pass at one speed.
Lanes alternate direction (boustrophedon).  The dose is accumulated on a grid in the
lane frame (ALONG_BIN m along track x CROSS_BIN m across).

Mission.  Starting at the base station at OP_START_H, the UAV takes off, ferries to
the next unit, sprays it, turns or ferries to the next one, and returns to refill and
swap batteries when the tank cannot hold the next unit or the battery projection
would breach the reserve.  Before each unit the go/no-go rule (`gate`) is checked at
the unit's start cell; when spraying is not permitted the policy either waits at
base (5-min steps) or, if `skip_gated`, moves on to the next permitted unit.  A policy
may also re-order the remaining queue each time the UAV is at base (replanning).
Operations end at OP_END_H; units not sprayed by then leave their area untreated.

The timing pass is sequential and cheap; flight power, drift and deposition are then
computed in one vectorised pass on the environment given as `phys_env`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import math

import numpy as np
from scipy.special import ndtr

if __package__ in (None, ""):  # also allow `python dsswaeop/<module>.py` (e.g. the IDE "Run" button)
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "dsswaeop"

from . import config as C
from . import physics as ph
from .atmosphere import Environment, FieldGeometry

P = C.PLATFORM
S = C.SPRAY
PIECE_LEN = 100.0   # m; lanes are split into pieces so a sortie can stop mid-lane and resume


# ─────────────────────────────────────────────────────────────────────────────
# Lane geometry
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LaneSet:
    heading: float
    d: np.ndarray                 # (2,) along-track unit vector (lane +direction)
    n: np.ndarray                 # (2,) cross-track unit vector
    crop: str
    rate: float                   # L/ha target
    # units
    u_lane: np.ndarray            # (U,) lane index
    u_block: np.ndarray           # (U,)
    u_sign: np.ndarray            # (U,) +1 travels along +d, -1 along -d
    u_len: np.ndarray             # (U,) m
    u_vol: np.ndarray             # (U,) L
    u_start: np.ndarray           # (U,2) field-frame start point
    u_end: np.ndarray             # (U,2)
    u_cell: np.ndarray            # (U,2) start cell indices
    u_seg: np.ndarray             # (U+1,) CSR pointer into segment arrays
    # segments
    s_row: np.ndarray             # (Sg,) dose-grid row
    s_len: np.ndarray             # (Sg,) m
    s_dist: np.ndarray            # (Sg,) m from unit start to segment midpoint
    s_xy: np.ndarray              # (Sg,2) field-frame midpoint
    s_cell: np.ndarray            # (Sg,2) atmospheric cell
    s_xc: np.ndarray              # (Sg,) cross coordinate of the lane centre (dose grid col units)
    s_fr0: np.ndarray             # (Sg,) first fine dose-grid row covered by the segment
    s_frac: np.ndarray            # (Sg, 6) covered length / DOSE_ALONG for consecutive fine rows
    s_xi: np.ndarray              # (2, Sg) N(0,1) meander draws: [0] truth, [1] planner's model
    # dose grid
    n_rows: int
    n_cols: int
    x0: float
    mask: np.ndarray              # (n_rows, n_cols) bool, cell centre inside the field
    n_lanes: int
    block_lanes: List[np.ndarray]

    @property
    def n_units(self):
        return len(self.u_len)


def _interval_in_rect(p0, d, L, W):
    """u-interval of the line p0 + u d inside [0,L]x[0,W] (empty -> None)."""
    lo, hi = -np.inf, np.inf
    for k, ext in ((0, L), (1, W)):
        if abs(d[k]) < 1e-12:
            if not (0 <= p0[k] <= ext):
                return None
            continue
        a = (0 - p0[k]) / d[k]
        b = (ext - p0[k]) / d[k]
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return (lo, hi) if hi > lo else None


def build_lanes(geom: FieldGeometry, heading_deg: float, crop: str,
                n_blocks: int = C.N_BLOCKS, seed: int = 0) -> LaneSet:
    psi = np.deg2rad(heading_deg)
    d = np.array([np.cos(psi), np.sin(psi)])
    n = np.array([-np.sin(psi), np.cos(psi)])
    L, W = geom.length, geom.width
    corners = np.array([[0, 0], [L, 0], [L, W], [0, W]], dtype=float)
    xs = corners @ n
    us = corners @ d
    xmin, xmax = xs.min(), xs.max()
    umin, umax = us.min(), us.max()
    w = S.swath
    rate = C.CROPS[crop]["rate"]

    centres = np.arange(xmin + w / 2, xmax, w)
    lanes = []
    for xc in centres:
        iv = _interval_in_rect(xc * n, d, L, W)
        if iv is None or iv[1] - iv[0] < 10.0:
            continue
        lanes.append((xc, iv[0], iv[1]))
    n_lanes = len(lanes)

    margin = 40.0
    x0 = xmin - margin
    n_cols = int(np.ceil((xmax - xmin + 2 * margin) / C.CROSS_BIN))
    n_rows = int(np.ceil((umax - umin) / C.DOSE_ALONG))
    rc = umin + (np.arange(n_rows) + 0.5) * C.DOSE_ALONG
    n_coarse = int(np.ceil((umax - umin) / C.ALONG_BIN))
    cc = x0 + (np.arange(n_cols) + 0.5) * C.CROSS_BIN
    P_ = rc[:, None, None] * d[None, None, :] + cc[None, :, None] * n[None, None, :]
    mask = (P_[..., 0] >= 0) & (P_[..., 0] <= L) & (P_[..., 1] >= 0) & (P_[..., 1] <= W)

    block_of_lane = np.zeros(n_lanes, dtype=int)
    block_lanes = np.array_split(np.arange(n_lanes), min(n_blocks, max(n_lanes, 1)))
    for b, ls in enumerate(block_lanes):
        block_of_lane[ls] = b

    U = dict(lane=[], block=[], sign=[], len=[], vol=[], start=[], end=[], seg=[0])
    Sg = dict(row=[], len=[], dist=[], xy=[], xc=[], fr0=[], frac=[])
    for j, (xc, ua, ub) in enumerate(lanes):
        sign = 1 if j % 2 == 0 else -1
        length = ub - ua
        vol = rate * w * length / 1e4
        k = max(1, int(np.ceil(length / PIECE_LEN)))   # breakpoint-resume pieces
        edges = np.linspace(ua, ub, k + 1)
        pieces = [(edges[i], edges[i + 1]) for i in range(k)]
        if sign < 0:
            pieces = [(b, a) for a, b in reversed(pieces)]
        for (s0, s1) in pieces:
            plen = abs(s1 - s0)
            U["lane"].append(j); U["block"].append(block_of_lane[j]); U["sign"].append(sign)
            U["len"].append(plen); U["vol"].append(rate * w * plen / 1e4)
            U["start"].append(xc * n + s0 * d); U["end"].append(xc * n + s1 * d)
            lo, hi = min(s0, s1), max(s0, s1)
            r0 = int(np.floor((lo - umin) / C.ALONG_BIN))
            r1 = int(np.floor((hi - umin - 1e-9) / C.ALONG_BIN))
            for r in range(max(r0, 0), min(r1, n_coarse - 1) + 1):
                a = max(lo, umin + r * C.ALONG_BIN)
                b = min(hi, umin + (r + 1) * C.ALONG_BIN)
                if b - a <= 1e-6:
                    continue
                mid = 0.5 * (a + b)
                Sg["row"].append(r); Sg["len"].append(b - a)
                Sg["dist"].append((mid - s0) if sign > 0 else (s0 - mid))
                Sg["xy"].append(xc * n + mid * d)
                Sg["xc"].append((xc - x0) / C.CROSS_BIN)
                f0 = int(np.floor((a - umin) / C.DOSE_ALONG))
                fr = np.zeros(6)
                for q in range(6):
                    ra = umin + (f0 + q) * C.DOSE_ALONG
                    fr[q] = max(0.0, min(b, ra + C.DOSE_ALONG) - max(a, ra)) / C.DOSE_ALONG
                Sg["fr0"].append(f0)
                Sg["frac"].append(fr)
            U["seg"].append(len(Sg["row"]))

    s_xy = np.array(Sg["xy"], dtype=float).reshape(-1, 2)
    six, siy = geom.cell_of(s_xy[:, 0], s_xy[:, 1])
    u_start = np.array(U["start"], dtype=float).reshape(-1, 2)
    uix, uiy = geom.cell_of(u_start[:, 0], u_start[:, 1])
    xrng = np.random.default_rng([seed, int(round(heading_deg * 10)), 17])
    s_xi = xrng.standard_normal((2, len(Sg["row"])))
    return LaneSet(
        heading=heading_deg, d=d, n=n, crop=crop, rate=rate,
        u_lane=np.array(U["lane"]), u_block=np.array(U["block"]), u_sign=np.array(U["sign"]),
        u_len=np.array(U["len"]), u_vol=np.array(U["vol"]), u_start=u_start,
        u_end=np.array(U["end"], dtype=float).reshape(-1, 2),
        u_cell=np.stack([uix, uiy], axis=1), u_seg=np.array(U["seg"]),
        s_row=np.array(Sg["row"]), s_len=np.array(Sg["len"]), s_dist=np.array(Sg["dist"]),
        s_xy=s_xy, s_cell=np.stack([six, siy], axis=1), s_xc=np.array(Sg["xc"]),
        s_fr0=np.array(Sg["fr0"], dtype=int), s_frac=np.array(Sg["frac"]).reshape(-1, 6),
        s_xi=s_xi,
        n_rows=n_rows, n_cols=n_cols, x0=x0, mask=mask, n_lanes=n_lanes,
        block_lanes=block_lanes,
    )


def max_speed_for_rate(rate: float) -> float:
    """Flow-compensated spraying: Q = R v w 60 / 1e4 <= q_max."""
    return min(S.v_spray_max, S.q_max * 1e4 / (60.0 * rate * S.swath))


# ─────────────────────────────────────────────────────────────────────────────
# Mission simulation
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MissionResult:
    energy_Wh: float
    e_spray: float
    e_ferry: float
    e_turn: float
    e_tol: float
    sorties: int
    duration_h: float
    wait_h: float
    sprayed: np.ndarray            # (U,) bool
    coverage: float
    precision: float
    drift_pct: float
    shift_cm: float
    treated_ha: float
    energy_per_treated_ha: float
    battery_violations: int
    replans: int = 0
    max_sortie_Wh: float = 0.0
    mean_sortie_Wh: float = 0.0
    unit_tstart: Optional[np.ndarray] = None
    unit_speed: Optional[np.ndarray] = None
    dose: Optional[np.ndarray] = None
    sortie_paths: Optional[list] = None

    def as_row(self) -> Dict[str, float]:
        return dict(energy_Wh=self.energy_Wh, e_spray=self.e_spray, e_ferry=self.e_ferry,
                    e_turn=self.e_turn, e_tol=self.e_tol, sorties=self.sorties,
                    duration_h=self.duration_h, wait_h=self.wait_h, coverage=self.coverage,
                    precision=self.precision, drift_pct=self.drift_pct, shift_cm=self.shift_cm,
                    treated_ha=self.treated_ha, energy_per_treated_ha=self.energy_per_treated_ha,
                    battery_violations=self.battery_violations, replans=self.replans,
                    max_sortie_Wh=self.max_sortie_Wh, mean_sortie_Wh=self.mean_sortie_Wh,
                    units_done=int(self.sprayed.sum()), units_total=int(len(self.sprayed)))


def _step(t_s: float) -> int:
    k = int((t_s / 3600.0 - C.DAY_START_H) * 60.0 / C.DT_MIN)
    return min(max(k, 0), C.N_STEPS - 1)


# Conservative power bound used only for the battery projection inside the timing pass:
# hover power (the maximum of the calm-air power curve for v <= 8 m/s) with a 15 % margin
# for wind and turbulence, fitted as a + b m^1.5 over the platform mass range.
_MS = np.linspace(P.m_dry, P.mtow, 8)
_PH = np.array([float(ph.electric_power(0.0, m, True)) for m in _MS]) * 1.15
_B, _A0 = np.polyfit(_MS ** 1.5, _PH, 1)


def _p_est(v, m):
    return _A0 + _B * m ** 1.5


class Policy:
    """Execution policy hooks. Subclasses implement replanning."""
    skip_gated: bool = False

    def on_base(self, t_s: float, queue: List[int], speeds: np.ndarray, state: dict):
        return None


def simulate(ls: LaneSet, order, speeds: np.ndarray, gate: np.ndarray, phys_env: Environment,
             t_start_s: float = C.OP_START_H * 3600.0, policy: Optional[Policy] = None,
             deposit: str = "full", keep_dose: bool = False, keep_paths: bool = False,
             done_mask: Optional[np.ndarray] = None, t_end_s: float = C.OP_END_H * 3600.0,
             xi_set: int = 0, meander: Optional[float] = None,
             offsets: Optional[np.ndarray] = None) -> MissionResult:
    """Simulate one mission.

    order    : sequence of unit indices (the planned queue)
    speeds   : (U,) ground speed of every unit (m/s)
    gate     : (T, nx, ny) bool, True where spraying is NOT permitted (decision view)
    phys_env : environment used for power and deposition (truth in evaluation,
               forecast/nowcast inside the planner)
    done_mask: units already sprayed before t_start_s (planner re-evaluations)
    """
    policy = policy or Policy()
    base = np.array(C.BASE_CORNER, dtype=float)
    queue = [int(u) for u in order if done_mask is None or not done_mask[u]]
    speeds = np.asarray(speeds, dtype=float).copy()
    nU = ls.n_units
    t = t_start_s
    pos = base.copy()
    tank = P.tank_L
    batt = 0.0
    in_air = False
    sorties = 0
    wait = 0.0
    replans = 0
    sprayed = np.zeros(nU, dtype=bool)
    tstart = np.full(nU, np.nan)
    uspeed = np.full(nU, np.nan)
    tank_at = np.zeros(nU)
    sortie_of = np.zeros(nU, dtype=int)
    legs = []        # (x0,y0,x1,y1,t_mid,mass)
    turns = []       # (x,y,t,mass)
    paths = [] if keep_paths else None
    cur_path = None
    last_end_unit = -1

    def go_base():
        nonlocal t, pos, in_air, tank, batt, cur_path
        dist = math.hypot(base[0] - pos[0], base[1] - pos[1])
        if dist > 0:
            m = P.m_dry + tank * C.RHO_LIQ
            dt = dist / P.v_ferry
            legs.append((pos[0], pos[1], base[0], base[1], t + dt / 2, m, sorties - 1))
            batt += _p_est(P.v_ferry, m) * dt / 3600
            t += dt
        pos = base.copy()
        in_air = False
        if cur_path is not None:
            cur_path.append((base[0], base[1], t, 0))
            paths.append(cur_path)
            cur_path = None

    head_unit, head_wait = -1, 0.0
    last_skip_t = -1.0
    guard = 0
    while queue:
        guard += 1
        if guard > 200000:
            raise RuntimeError("mission simulation did not terminate")
        if not in_air:
            res = policy.on_base(t, queue, speeds, dict(sprayed=sprayed))
            if res is not None:
                queue, speeds = res
                speeds = np.asarray(speeds, dtype=float).copy()
                replans += 1
            new_off = getattr(policy, "offsets", None)
            if new_off is not None and queue:
                offsets = offsets.copy() if offsets is not None else np.zeros(nU)
                rem = np.array(queue, dtype=int)
                offsets[rem] = new_off[rem]
                if not queue:
                    break
        u = queue[0]
        v = speeds[u]
        start = ls.u_start[u]
        ix, iy = ls.u_cell[u]
        dur = ls.u_len[u] / v
        # time needed to reach the unit
        if in_air:
            gap = math.hypot(start[0] - pos[0], start[1] - pos[1])
            if gap < 1.0:
                reach = 0.0
            elif gap <= 1.6 * S.swath:
                reach = P.t_turn + gap / P.v_ferry
            else:
                reach = gap / P.v_ferry
        else:
            gap = math.hypot(start[0] - base[0], start[1] - base[1])
            reach = gap / P.v_ferry
        k_arr = _step(t + reach)
        if gate[k_arr, ix, iy]:
            alt = None
            if policy.skip_gated and len(queue) > 1 and last_skip_t != t:
                # nearest unit (from the current position) where spraying is permitted
                rem = np.asarray(queue[1:])
                q_ = pos if in_air else base
                dist = np.hypot(ls.u_start[rem, 0] - q_[0], ls.u_start[rem, 1] - q_[1])
                ks = ((t + dist / P.v_ferry) / 3600.0 - C.DAY_START_H) * 60.0 / C.DT_MIN
                ks = np.clip(ks.astype(int), 0, C.N_STEPS - 1)
                okm = ~gate[ks, ls.u_cell[rem, 0], ls.u_cell[rem, 1]]
                if okm.any():
                    cand = np.flatnonzero(okm)
                    alt = 1 + int(cand[np.argmin(dist[cand])])
            if alt is not None:
                # at most one skip per instant: if the new head is still gated (e.g. the
                # headland turn shifts its arrival into a gated step) the UAV waits instead
                last_skip_t = t
                queue.insert(0, queue.pop(alt))
                continue
            if in_air:
                go_base()
            if queue[0] != head_unit:
                head_unit, head_wait = queue[0], 0.0
            t += C.DT_MIN * 60.0
            wait += C.DT_MIN * 60.0
            head_wait += C.DT_MIN * 60.0
            if head_wait >= C.MAX_WAIT_MIN * 60.0 and len(queue) > 1:
                # operator defers the rest of this lane to the end of the queue
                lane = ls.u_lane[queue[0]]
                queue = [q for q in queue if ls.u_lane[q] != lane] + \
                        [q for q in queue if ls.u_lane[q] == lane]
                head_unit, head_wait = -1, 0.0
            if t >= t_end_s:
                break
            continue
        # resources: tank and battery for this unit + return
        m_now = P.m_dry + tank * C.RHO_LIQ
        ret = math.hypot(ls.u_end[u][0] - base[0], ls.u_end[u][1] - base[1]) / P.v_ferry
        need_batt = (_p_est(P.v_ferry, m_now) * reach + _p_est(v, m_now) * dur
                     + _p_est(P.v_ferry, P.m_dry) * ret) / 3600
        if in_air and (ls.u_vol[u] > tank + 1e-9 or batt + need_batt > P.usable_Wh):
            go_base()
            continue
        if t + reach + dur + ret > t_end_s:
            break
        if not in_air:
            # service, then take off
            if tank < P.tank_L - 1e-9 or batt > 0:
                t += P.t_service
            tank = P.tank_L
            batt = P.e_takeoff_land
            sorties += 1
            in_air = True
            m_now = P.m_dry + tank * C.RHO_LIQ
            if keep_paths:
                cur_path = [(base[0], base[1], t, 0)]
            dist = math.hypot(start[0] - base[0], start[1] - base[1])
            legs.append((base[0], base[1], start[0], start[1], t + reach / 2, m_now, sorties - 1))
            batt += _p_est(P.v_ferry, m_now) * reach / 3600
            t += reach
        else:
            gap = math.hypot(start[0] - pos[0], start[1] - pos[1])
            if gap < 1.0:
                pass
            elif gap <= 1.6 * S.swath:
                turns.append((pos[0], pos[1], t, m_now, sorties - 1))
                batt += _p_est(0.0, m_now) * P.t_turn / 3600
            else:
                legs.append((pos[0], pos[1], start[0], start[1], t + reach / 2, m_now, sorties - 1))
                batt += _p_est(P.v_ferry, m_now) * reach / 3600
            t += reach
        if cur_path is not None:
            cur_path.append((start[0], start[1], t, 0))
        # spray the unit
        tstart[u] = t
        uspeed[u] = v
        tank_at[u] = tank
        sortie_of[u] = sorties - 1
        batt += _p_est(v, m_now) * dur / 3600
        t += dur
        tank -= ls.u_vol[u]
        pos = ls.u_end[u].copy()
        sprayed[u] = True
        queue.pop(0)
        if cur_path is not None:
            cur_path.append((pos[0], pos[1], t, 1))
    if in_air:
        go_base()

    res = _physics(ls, sprayed, tstart, uspeed, tank_at, sortie_of, legs, turns, sorties, phys_env,
                   deposit, keep_dose, xi_set, meander, offsets)
    res.duration_h = (t - t_start_s) / 3600.0
    res.wait_h = wait / 3600.0
    res.replans = replans
    res.unit_tstart = tstart
    res.unit_speed = uspeed
    res.sortie_paths = paths
    return res


def _physics(ls: LaneSet, sprayed, tstart, uspeed, tank_at, sortie_of, legs, turns, sorties,
             env: Environment, deposit: str, keep_dose: bool, xi_set: int = 0,
             meander: Optional[float] = None, offsets=None) -> MissionResult:
    wto = env.wind_to_unit()
    per_sortie = np.zeros(max(sorties, 1))
    classes = C.CROPS[ls.crop]["classes"]
    rate = ls.rate
    w = S.swath

    # ── spraying segments ──────────────────────────────────────────────────
    idx = np.flatnonzero(sprayed)
    if len(idx):
        counts = ls.u_seg[idx + 1] - ls.u_seg[idx]
        seg = np.concatenate([np.arange(ls.u_seg[i], ls.u_seg[i + 1]) for i in idx])
        uu = np.repeat(idx, counts)
        v = uspeed[uu]
        tseg = tstart[uu] + ls.s_dist[seg] / v
        k = ((tseg / 3600.0 - C.DAY_START_H) * 60.0 / C.DT_MIN).astype(int).clip(0, C.N_STEPS - 1)
        cx, cy = ls.s_cell[seg, 0], ls.s_cell[seg, 1]
        U = env.fields["e1"][k, cx, cy].astype(float)
        e2 = env.fields["e2"][k, cx, cy].astype(float)
        e3 = env.fields["e3"][k, cx, cy].astype(float)
        e5 = env.fields["e5"][k, cx, cy].astype(float)
        sw = env.sigma_w[k, cx, cy].astype(float)
        wu = wto[k]
        wind = U[:, None] * wu
        gvel = (ls.u_sign[uu] * v)[:, None] * ls.d[None, :]
        # mass at segment midpoint (tank drains along the sortie) — reconstruct per unit
        m = P.m_dry + (tank_at[uu] - rate * w * ls.s_dist[seg] / 1e4) * C.RHO_LIQ
        m = np.clip(m, P.m_dry, P.mtow)
        su, _ = ph.sigma_uv(e3)
        pw, va, vi = ph.turbulent_power(gvel, wind, su, m, True)
        es = pw * ls.s_len[seg] / v / 3600.0
        e_spray = float(np.sum(es))
        np.add.at(per_sortie, sortie_of[uu], es)
        dp = ph.deposit_params(U, wu, np.broadcast_to(ls.n, wu.shape), va, vi, e2, e3, e5, sw,
                               classes, xi=ls.s_xi[xi_set, seg], meander=meander)
        frac = np.array([f for _, f in classes])
        amount = rate * w * frac[None, :] * dp.keep
        emitted = float(np.sum(ls.u_vol[idx]))
        shift = dp.shift if offsets is None else dp.shift + offsets[uu][:, None]
        # volume-weighted mean |net displacement| of the deposit from the intended lane centre
        wgt = frac[None, :] * dp.keep * ls.s_len[seg][:, None]
        shift_cm = float(100 * np.sum(np.abs(shift) * wgt) / max(np.sum(wgt), 1e-9))
        dose = _deposit(ls, seg, shift / C.CROSS_BIN, dp.sigma / C.CROSS_BIN, amount, deposit)
    else:
        e_spray = 0.0
        shift_cm = 0.0
        emitted = 0.0
        dose = np.zeros((ls.n_rows, ls.n_cols))

    # ── ferry legs and turns ───────────────────────────────────────────────
    e_ferry = 0.0
    if legs:
        Lg = np.array(legs)
        p0, p1 = Lg[:, 0:2], Lg[:, 2:4]
        dist = np.linalg.norm(p1 - p0, axis=1)
        ok = dist > 1e-6
        if ok.any():
            p0, p1, dist, tm, mm = p0[ok], p1[ok], dist[ok], Lg[ok, 4], Lg[ok, 5]
            sid = Lg[ok, 6].astype(int)
            k = ((tm / 3600.0 - C.DAY_START_H) * 60.0 / C.DT_MIN).astype(int).clip(0, C.N_STEPS - 1)
            mid = 0.5 * (p0 + p1)
            cx, cy = env.geom.cell_of(mid[:, 0], mid[:, 1])
            U = env.fields["e1"][k, cx, cy].astype(float)
            su, _ = ph.sigma_uv(env.fields["e3"][k, cx, cy].astype(float))
            gvel = (p1 - p0) / dist[:, None] * P.v_ferry
            pw, _, _ = ph.turbulent_power(gvel, U[:, None] * wto[k], su, mm, False)
            ef = pw * dist / P.v_ferry / 3600.0
            e_ferry = float(np.sum(ef))
            np.add.at(per_sortie, np.clip(sid, 0, len(per_sortie) - 1), ef)
    e_turn = 0.0
    if turns:
        Tn = np.array(turns)
        k = ((Tn[:, 2] / 3600.0 - C.DAY_START_H) * 60.0 / C.DT_MIN).astype(int).clip(0, C.N_STEPS - 1)
        cx, cy = env.geom.cell_of(Tn[:, 0], Tn[:, 1])
        U = env.fields["e1"][k, cx, cy].astype(float)
        su, _ = ph.sigma_uv(env.fields["e3"][k, cx, cy].astype(float))
        pw, _, _ = ph.turbulent_power(np.zeros((len(Tn), 2)), U[:, None] * wto[k], su,
                                      Tn[:, 3], False)
        et = pw * P.t_turn / 3600.0
        e_turn = float(np.sum(et))
        np.add.at(per_sortie, np.clip(Tn[:, 4].astype(int), 0, len(per_sortie) - 1), et)
    e_tol = sorties * P.e_takeoff_land
    energy = e_spray + e_ferry + e_turn + e_tol

    # ── dose metrics ────────────────────────────────────────────────────────
    df = dose[ls.mask]
    cov = float(np.mean(df >= C.COVERAGE_MIN * rate))
    lo, hi = C.PRECISION_BAND
    prec = float(np.mean((df >= lo * rate) & (df <= hi * rate)))
    infield = float(np.sum(df) * C.DOSE_ALONG * C.CROSS_BIN / 1e4)
    drift = 100.0 * (1.0 - infield / emitted) if emitted > 0 else 0.0
    area_ha = ls.mask.sum() * C.DOSE_ALONG * C.CROSS_BIN / 1e4
    treated = cov * area_ha
    # battery check with the evaluation physics: sorties whose energy breaches the reserve
    per_sortie = per_sortie[:sorties] + P.e_takeoff_land if sorties else per_sortie[:0]
    viol = int(np.sum(per_sortie > P.usable_Wh))
    return MissionResult(energy, e_spray, e_ferry, e_turn, e_tol, sorties, 0.0, 0.0, sprayed,
                         cov, prec, drift, shift_cm, treated,
                         energy / treated if treated > 1e-9 else np.nan, viol,
                         max_sortie_Wh=float(per_sortie.max()) if len(per_sortie) else 0.0,
                         mean_sortie_Wh=float(per_sortie.mean()) if len(per_sortie) else 0.0,
                         dose=dose if keep_dose else None)


def _deposit(ls: LaneSet, seg, shift_c, sigma_c, amount, mode: str) -> np.ndarray:
    """Accumulate Gaussian deposits on the lane-frame grid (cols in CROSS_BIN units).

    mode 'full': every droplet class separately; 'fast': one moment-matched Gaussian
    per segment (used inside the optimiser).
    """
    if mode == "fast":
        a = amount.sum(axis=1)
        wgt = amount / np.maximum(a[:, None], 1e-12)
        mu = np.sum(wgt * shift_c, axis=1)
        var = np.sum(wgt * (sigma_c ** 2 + shift_c ** 2), axis=1) - mu ** 2
        shift_c = mu[:, None]
        sigma_c = np.sqrt(np.maximum(var, 1e-6))[:, None]
        amount = a[:, None]
    nS, nC = amount.shape
    xc = ls.s_xc[seg]
    mu = (xc[:, None] + shift_c).ravel()
    sg = sigma_c.ravel()
    am = amount.ravel()
    fr0 = np.repeat(ls.s_fr0[seg], nC)
    frac = np.repeat(ls.s_frac[seg], nC, axis=0)              # (nS*nC, 6)
    half = int(np.ceil(4.0 * np.quantile(sg, 0.995))) + 1
    edges = np.arange(-half, half + 2)
    base = np.floor(mu).astype(int)
    cdf = ndtr((base[:, None] + edges[None, :] - mu[:, None]) / sg[:, None])
    mass = np.diff(cdf, axis=1) * am[:, None] / C.CROSS_BIN
    cols = base[:, None] + edges[None, :-1]
    nq = int(np.max(np.flatnonzero(frac.any(axis=0)))) + 1
    rows = fr0[:, None] + np.arange(nq)[None, :]                # (N, nq)
    okc = (cols >= 0) & (cols < ls.n_cols)
    okr = (rows >= 0) & (rows < ls.n_rows) & (frac[:, :nq] > 0)
    flat = rows[:, :, None] * ls.n_cols + cols[:, None, :]       # (N, nq, K)
    wts = frac[:, :nq, None] * mass[:, None, :]
    ok = okr[:, :, None] & okc[:, None, :]
    dose = np.bincount(flat[ok], weights=wts[ok], minlength=ls.n_rows * ls.n_cols)
    return dose.reshape(ls.n_rows, ls.n_cols)


if __name__ == "__main__":
    # Short demonstration: lanes of a 5 ha field at three headings.
    from .atmosphere import FieldGeometry
    g = FieldGeometry.from_area(5)
    print("dsswaeop.mission: lane decomposition of a 5 ha field")
    for psi in (0, 30, 90):
        ls = build_lanes(g, psi, "wheat")
        print(f"  heading {psi:3d} deg: {ls.n_lanes} lanes, {len(ls.block_lanes)} blocks, {ls.n_units} pieces")
    print(f"  maximum spraying speed for wheat: {max_speed_for_rate(C.CROPS['wheat']['rate']):.2f} m/s")
