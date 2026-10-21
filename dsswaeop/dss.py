"""
Dynamic soft set (DSS) layer over a gridded atmosphere.

Universe U   : atmospheric grid cells u (WIND_CELL x WIND_CELL m) covering the field.
Parameters E : {e1..e6} (config.PARAMS).
Time index T : atmospheric time steps.

For each t, the DSS is the pair (A_t, F_t) with
  A_t  ⊆ E         active parameters (activation rule, `active_set`)
  F_t(e) = W_t(e)  cells whose value of e lies in the favourable class (Eq. 8).

Derived objects used by the planner:
  Z_f(t) = ∩_{e∈A_t} W_t(e)                    favourable zone (Eq. 9)
  Z_a(t) = ∪_{e∈A_t} {u : e(u,t) in class γ}    adverse zone (spraying suspended)
  Φ(u,t) = Σ_{e∈A_t} w_e ε_e(u,t)               energy-potential surrogate (Eq. 2)
           ε_e = (e − θ_lo)/(θ_fav − θ_lo) clipped to [0, ε_max]
           (Φ ≤ 1 inside Z_f, Φ > 1 when some active parameter leaves its
            favourable class; it increases monotonically with every exceedance)
  σ(e, t_i, t_j)                                temporal stability index (Eq. 1)

Variants used in the ablations:
  activation="dynamic"  : DSS rule (default)
  activation="all"      : A_t = E for all t (fixed parameter set, as in T-FSS)
  activation="speed"    : A_t = {e1} (single-parameter wind model)
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

if __package__ in (None, ""):  # also allow `python dsswaeop/<module>.py` (e.g. the IDE "Run" button)
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "dsswaeop"

from . import config as C
from .atmosphere import Environment

EPS_MAX = 3.0


def thresholds(scale: float = 1.0) -> Dict[str, tuple]:
    """Threshold table, optionally scaled about the lower bound (sensitivity analysis)."""
    out = {}
    for p, (lo, fav, adv) in C.THRESHOLDS.items():
        out[p] = (lo, lo + (fav - lo) * scale, lo + (adv - lo) * scale)
    return out


def active_set(env: Environment, mode: str = "dynamic") -> np.ndarray:
    """Boolean (T, 6) activation mask A_t.

    Dynamic rule (physically motivated, see Section 4.3):
      e1, e2, e3 always active;
      e4 (gust factor) active only when the field-mean wind is >= 2 m/s, because
         G = peak/mean is ill-conditioned in light winds;
      e5 (shear) active only without surface heating (S(h) <= 0.1), i.e. in the
         stable early-morning / evening layer where strong near-surface shear signals
         an inversion; under daytime mixing the 0-3 m shear has no separate pathway to
         drift beyond wind speed and turbulence;
      e6 (thermal index, a column CAPE property) active only while surface heating
         can couple it to near-surface convection, S(h) > 0.1.
    """
    T = env.T
    A = np.ones((T, len(C.PARAMS)), dtype=bool)
    if mode == "all":
        return A
    if mode == "speed":
        A[:] = False
        A[:, 0] = True
        return A
    if mode != "dynamic":
        raise ValueError(mode)
    A[:, 3] = env.field_mean("e1") >= C.GUST_ACTIVE_MIN_WIND
    A[:, 4] = C.solar_factor(env.hours) <= C.THERMAL_ACTIVE_MIN_S
    A[:, 5] = C.solar_factor(env.hours) > C.THERMAL_ACTIVE_MIN_S
    return A


class DSSModel:
    """Dynamic soft set built on an Environment view (forecast, nowcast or measured)."""

    def __init__(self, env: Environment, activation: str = "dynamic",
                 weights: Optional[Dict[str, float]] = None, thr_scale: float = 1.0):
        self.env = env
        self.activation = activation
        self.A = active_set(env, activation)
        self.thr = thresholds(thr_scale)
        self.weights = weights
        T, nx, ny = env.fields["e1"].shape
        self.shape = (T, nx, ny)
        # class maps: 0 favourable (alpha), 1 marginal (beta), 2 adverse (gamma)
        self.cls = {}
        self.eps = {}
        self._W = {}
        for p in C.PARAMS:
            lo, fav, adv = self.thr[p]
            v = env.fields[p]
            c = np.zeros(v.shape, dtype=np.int8)
            c[v > fav] = 1
            c[v > adv] = 2
            self.cls[p] = c
            self.eps[p] = np.clip((v - lo) / (fav - lo), 0.0, EPS_MAX).astype(np.float32)

    # ── soft-set mappings ──────────────────────────────────────────────────
    def W(self, p: str) -> np.ndarray:
        """Favourable set W_t(e) as a boolean (T, nx, ny) array (Eq. 8)."""
        if p not in self._W:
            self._W[p] = self.cls[p] == 0
        return self._W[p]

    def favourable_zone(self) -> np.ndarray:
        Z = np.ones(self.shape, dtype=bool)
        for i, p in enumerate(C.PARAMS):
            a = self.A[:, i][:, None, None]
            Z &= np.where(a, self.W(p), True)
        return Z

    def adverse_zone(self) -> np.ndarray:
        Z = np.zeros(self.shape, dtype=bool)
        for i, p in enumerate(C.PARAMS):
            a = self.A[:, i][:, None, None]
            Z |= a & (self.cls[p] == 2)
        return Z

    def future_window(self, k_last: int) -> np.ndarray:
        """Boolean (T, nx, ny): True where a non-adverse slot exists at some later step
        k' in (k, k_last] (reverse cumulative OR of the complement of the adverse zone)."""
        ok = ~self.adverse_zone()
        T = ok.shape[0]
        fut = np.zeros_like(ok)
        acc = np.zeros(ok.shape[1:], dtype=bool)
        for k in range(min(k_last, T - 1), -1, -1):
            fut[k] = acc
            acc = acc | ok[k]
        return fut

    def class_count(self) -> np.ndarray:
        """Number of active parameters in the marginal or adverse class per cell."""
        n = np.zeros(self.shape, dtype=np.int8)
        for i, p in enumerate(C.PARAMS):
            a = self.A[:, i][:, None, None]
            n += (a & (self.cls[p] >= 1)).astype(np.int8)
        return n

    def phi(self) -> np.ndarray:
        """Energy-potential surrogate Φ(u,t) (Eq. 2, corrected definition)."""
        out = np.zeros(self.shape, dtype=np.float32)
        nact = np.maximum(self.A.sum(axis=1), 1).astype(np.float32)
        for i, p in enumerate(C.PARAMS):
            if self.weights is None:
                w = (self.A[:, i] / nact)[:, None, None]
            else:
                w = (self.A[:, i] * self.weights[p])[:, None, None]
                norm = (self.A * np.array([self.weights[q] for q in C.PARAMS])).sum(axis=1)
                w = w / np.maximum(norm, 1e-9)[:, None, None]
            out += w * self.eps[p]
        return out

    def stability(self, p: str, ti: int, tj: int, cells: Optional[np.ndarray] = None) -> float:
        """Temporal stability index σ(e, t_i, t_j) of Eq. (1) on the favourable set W_t(e)."""
        ti = max(int(ti), 0)
        tj = min(int(tj), self.shape[0] - 1)
        if tj <= ti:
            return 1.0
        Wp = self.W(p)[ti:tj + 1]
        if cells is not None:
            Wp = Wp[:, cells]
        n = Wp[0].size
        flips = np.logical_xor(Wp[1:], Wp[:-1]).reshape(tj - ti, -1).sum(axis=1) / max(n, 1)
        return float(1.0 - flips.mean())


def stability_from_masks(masks: Sequence[np.ndarray]) -> float:
    """σ computed from a sequence of reconstructed favourable masks (online form)."""
    if len(masks) < 2:
        return 1.0
    m = np.stack(masks)
    flips = np.logical_xor(m[1:], m[:-1]).reshape(len(masks) - 1, -1).mean(axis=1)
    return float(1.0 - flips.mean())


def boundary_fraction_bound(eps_k: float, p: str) -> float:
    """Proposition 3.10 (Eq. 6): expected fraction of cells within eps_k of a class
    boundary under a uniform marginal over [θ_lo, θ_adv + (θ_adv − θ_fav)]."""
    lo, fav, adv = C.THRESHOLDS[p]
    span = adv + (adv - fav) - lo
    return min(1.0, 4.0 * eps_k / span)   # two boundaries (fav, adv), each ±eps


if __name__ == "__main__":
    # Short demonstration: build the dynamic soft set on the forecast of one scenario.
    from .atmosphere import Atmosphere, Scenario
    atm = Atmosphere(Scenario(5, "rolling", "pre_monsoon", "wheat", seed=1))
    m = DSSModel(atm.forecast(C.step_of_hour(6.0)), "dynamic")
    Za = m.adverse_zone()
    print("dsswaeop.dss: dynamic soft set on the 06:00 forecast of a 5 ha field")
    for h in (7, 10, 13, 16):
        k = C.step_of_hour(h)
        act = [p for j, p in enumerate(C.PARAMS) if m.A[k, j]]
        print(f"  {h:02d}:00  active attributes A_t = {{{', '.join(act)}}}, "
              f"adverse zone = {100 * Za[k].mean():.0f}% of cells")
