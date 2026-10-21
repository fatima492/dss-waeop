"""
Flight-energy and spray-deposition physics.

Power (Section 4.4).  Momentum theory with Glauert's forward-flight inflow
(Leishman 2006), applied to the total rotor disk of a multirotor:

    D   = ½ ρ CdA V_a²                       parasitic drag at airspeed V_a
    T   = sqrt((m g)² + D²),  α = atan(D / m g)   thrust and disk tilt
    v_i = T / (2 ρ A sqrt((V_a cos α)² + (V_a sin α + v_i)²))   (fixed point)
    P_e = [κ T v_i + T V_a sin α + P_0] / η_e + P_av (+ P_pump when spraying)

P_0 is the profile power, calibrated so that the hover figure of merit at MTOW
equals `fm_hover`.  V_a is the AIRSPEED, i.e. |v_ground − wind|, so headwind,
tailwind and crosswind change the power drawn.  Turbulence is handled by averaging
P over longitudinal gusts u' ~ N(0, σ_u²) with 3-point Gauss-Hermite quadrature;
because P(V_a) is non-linear this produces a turbulence penalty (or benefit) without
any additional fitted coefficient.

Deposition (Section 4.5).  Each droplet class c with settling velocity v_s,c falls
the release height h with vertical speed v_z,c = v_s,c + c_dw v_i cos χ, where χ is
the wake skew angle (rotor downwash entrains droplets, less so at high airspeed).
During the fall time t_f = h / v_z the mean cross-track wind displaces the deposit
by Δ = U_eff,⊥ t_f.  Lateral turbulence and direction variability add a turbulent
displacement with variance s_t² = (σ_v t_f)² + (U_eff sin σ_θ t_f)².  Following the
fluctuating-plume picture (Gifford 1959), a fraction a of s_t² is realised as a random
shift of each 25-m segment (ξ ~ N(0,1), common random numbers across planners) and the
remainder broadens it:

    Δ_c = U_eff,⊥ t_f + sqrt(a) s_t ξ,     σ_c² = σ_0² + (1 − a) s_t²

The fraction of a class that escapes deposition (fine droplets lofted by vertical
turbulence) is p_esc = exp(−v_z / σ_w).  U_eff = U − ½ e5 h is the mean wind over
the fall layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

if __package__ in (None, ""):  # also allow `python dsswaeop/<module>.py` (e.g. the IDE "Run" button)
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "dsswaeop"

from . import config as C

P = C.PLATFORM
S = C.SPRAY
_RHO = C.RHO_AIR
_A = P.disk_area


def _ideal_hover(m: float) -> float:
    T = m * C.G
    return T ** 1.5 / np.sqrt(2 * _RHO * _A)


#: Profile power calibrated so that FM(hover, MTOW) = fm_hover
P_PROFILE = _ideal_hover(P.mtow) * (1.0 / P.fm_hover - P.kappa_ind)

# Gauss-Hermite nodes/weights for E[f(X)], X ~ N(0,1)
_GH_X = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
_GH_W = np.array([1 / 6, 2 / 3, 1 / 6])


def rotor_state(va, mass) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mechanical power, induced velocity and disk tilt at airspeed `va` (vectorised)."""
    va = np.asarray(va, dtype=float)
    mass = np.asarray(mass, dtype=float)
    W = mass * C.G
    D = 0.5 * _RHO * P.cda * va ** 2
    T = np.sqrt(W ** 2 + D ** 2)
    alpha = np.arctan2(D, W)
    vh = np.sqrt(T / (2 * _RHO * _A))
    vi = vh.copy()
    vx = va * np.cos(alpha)
    vz = va * np.sin(alpha)
    for _ in range(25):
        vi_new = T / (2 * _RHO * _A * np.sqrt(vx ** 2 + (vz + vi) ** 2))
        vi = 0.5 * vi + 0.5 * vi_new
    p_mech = P.kappa_ind * T * vi + T * vz + P_PROFILE
    return p_mech, vi, alpha


def electric_power(va, mass, spraying=False) -> np.ndarray:
    p_mech, _, _ = rotor_state(va, mass)
    return p_mech / P.eta_e + P.p_avionics + (P.p_pump if spraying else 0.0)


def turbulent_power(ground_vel: np.ndarray, wind_vel: np.ndarray, sigma_u: np.ndarray,
                    mass, spraying) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expected electric power with longitudinal turbulence; also mean airspeed and v_i.

    ground_vel, wind_vel : (..., 2) vectors in the field frame (m/s)
    sigma_u              : (...) longitudinal turbulence std (m/s)
    """
    U = np.linalg.norm(wind_vel, axis=-1)
    uhat = np.where(U[..., None] > 1e-6, wind_vel / np.maximum(U[..., None], 1e-6), 0.0)
    pw = 0.0
    for x, w in zip(_GH_X, _GH_W):
        wv = wind_vel + (x * sigma_u)[..., None] * uhat
        va = np.linalg.norm(ground_vel - wv, axis=-1)
        pw = pw + w * electric_power(va, mass, spraying)
    va0 = np.linalg.norm(ground_vel - wind_vel, axis=-1)
    _, vi0, _ = rotor_state(va0, mass)
    return pw, va0, vi0


def sigma_uv(tke):
    ustar = np.sqrt(np.asarray(tke) / 5.47)
    return 2.4 * ustar, 1.9 * ustar


@dataclass
class DepositParams:
    """Per (segment, class) deposit description (arrays of equal shape)."""
    shift: np.ndarray      # m, cross-track displacement Δ
    sigma: np.ndarray      # m, lateral std
    keep: np.ndarray       # fraction deposited (1 - escape)


def deposit_params(U, wind_unit, cross_unit, va, vi, e2_deg, e3, e5, sigma_w,
                   classes, xi=None, meander=None) -> DepositParams:
    """Deposit shift/spread/retention for each droplet class.

    All atmospheric inputs have shape (n,); output arrays have shape (n, n_classes).
    """
    h = S.height
    chi = np.arctan2(va, np.maximum(vi, 1e-3))                  # wake skew from vertical
    dw = S.c_downwash * vi * np.cos(chi)
    ueff = np.maximum(U - 0.5 * e5 * h, 0.0)
    ucross = ueff * np.sum(wind_unit * cross_unit, axis=-1)     # signed, along +cross axis
    _, sv = sigma_uv(e3)
    meander_v = ueff * np.sin(np.deg2rad(e2_deg))
    vs = np.array([C.settling_velocity(d) for d, _ in classes])
    vz = vs[None, :] + dw[:, None]
    tf = h / vz
    a = C.MEANDER_FRACTION if meander is None else meander
    turb = np.sqrt((sv[:, None] * tf) ** 2 + (meander_v[:, None] * tf) ** 2)
    shift = ucross[:, None] * tf
    if xi is not None and a > 0:
        shift = shift + np.sqrt(a) * turb * xi[:, None]
    sig = np.sqrt(S.sigma0 ** 2 + (1.0 - a) * turb ** 2)
    keep = 1.0 - np.exp(-vz / np.maximum(sigma_w, 0.05)[:, None])
    return DepositParams(shift, sig, keep)


def power_curve(mass: float, speeds: np.ndarray) -> np.ndarray:
    """Calm-air electric power vs airspeed (for Fig. 3 and the energy-aware baseline)."""
    return electric_power(speeds, np.full_like(speeds, mass), spraying=False)


if __name__ == "__main__":
    # Short demonstration of the power model (Section 4.3 of the manuscript).
    m = C.PLATFORM.mtow
    v = np.array([0.0, 3.0, 5.0, 7.0, 10.0])
    p = power_curve(m, v)
    print("dsswaeop.physics: electric power of the full-tank UAV in calm air")
    for vi, pi in zip(v, p):
        print(f"  airspeed {vi:4.1f} m/s  ->  {pi:7.0f} W")
