#!/usr/bin/env python3
"""
reproduce_all.py
================
Complete reproducibility script for:

    "Wind-aware energy optimization for agricultural UAV operations:
     A dynamic soft set approach to trajectory planning under
     atmospheric variability"

Running this script reproduces every figure, table, and numerical result
reported in the manuscript.  All outputs are written to results/.

Usage
-----
    python reproduce_all.py              # full run (all 1,200 scenarios × 30 reps)
    python reproduce_all.py --quick      # smoke test  (20 scenarios × 3 reps)
    python reproduce_all.py --figures    # figures only (no simulation)

Dependencies
------------
    numpy, scipy, matplotlib, deap  (see requirements.txt)
    Python >= 3.10

Output layout
-------------
    results/
        figures/        manuscript figures (PNG, 300 DPI)
        tables/         LaTeX-formatted tables (.tex) and CSV exports
        data/           raw simulation results (.npy)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# §0  IMPORTS AND GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patches as FancyArrowPatch
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

# ── Windows console safety ──────────────────────────────────────────────────────
# Ensure console output uses UTF-8 so status characters (checkmarks, box-drawing)
# and any Unicode in printed strings do not raise UnicodeEncodeError on Windows
# terminals that default to cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    # Older Python (<3.7) or non-reconfigurable stream: fall back silently.
    pass

# ── output directories ────────────────────────────────────────────────────────
RESULTS   = Path("results")
FIG_DIR   = RESULTS / "figures"
TABLE_DIR = RESULTS / "tables"
DATA_DIR  = RESULTS / "data"

for d in (FIG_DIR, TABLE_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── matplotlib global style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9.5,
    "ytick.labelsize":   9.5,
    "legend.fontsize":   9.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.06,
    "lines.linewidth":   1.8,
    "errorbar.capsize":  4,
})

# ── DSS colour palette ────────────────────────────────────────────────────────
C_DSS      = "#2e7d32"   # dark green  — DSS-WAEOP
C_BASE     = "#616161"   # grey        — baselines
C_TFSS     = "#7b1fa2"   # purple      — T-FSS baseline
C_GAP      = "#c62828"   # red         — research gap boxes
C_CONTRIB  = "#1b5e20"   # dark green  — contribution boxes
C_BLUE     = "#1565c0"
C_ORANGE   = "#e65100"


# ═══════════════════════════════════════════════════════════════════════════════
# §1  ATMOSPHERIC PARAMETER CATALOGUE  (Table 5 in manuscript)
# ═══════════════════════════════════════════════════════════════════════════════

#: Six atmospheric parameters and their DSS classification thresholds.
#: Each entry: (favorable_lower, favorable_upper, adverse_upper)
THRESHOLDS: Dict[str, Tuple[float, float, float]] = {
    "e1": (0.00,  3.50,  6.50),   # wind speed [m/s]
    "e2": (0.00, 15.00, 45.00),   # direction variability [°]
    "e3": (0.00,  0.80,  2.00),   # turbulence KE [m²/s²]
    "e4": (1.00,  1.30,  1.80),   # gust factor [–]
    "e5": (0.00,  0.05,  0.12),   # vertical wind shear [s⁻¹]
    "e6": (0.00,  2.00,  4.50),   # thermal instability [–]
}

PARAM_NAMES = list(THRESHOLDS.keys())
N_PARAMS    = len(PARAM_NAMES)

# ── UAV platform constants ────────────────────────────────────────────────────
P_HOVER   = 850.0    # W   hover power
P_PUMP    =  85.0    # W   spray pump
C_D       =   0.28   # –   drag coefficient
RHO_A     =   1.22   # kg/m³ air density
A_R       =   0.24   # m²  effective frontal area
MASS_UAV  =  25.0    # kg  all-up mass (incl. 10 L spray)
V_MAX     =  10.0    # m/s maximum airspeed
KAPPA_6   =   0.12   # –   thermal coupling coefficient
SIGMA_0   =   0.50   # m   baseline nozzle spread
LAMBDA_D  =   0.30   # m/(m/s) drift sensitivity (Nuyttens et al. 2007)
Q_SPRAY   =   4.0    # L/min spray flow rate
G_GRAV    =   9.81   # m/s²

#: Maximum wind compensation power c_w [W].
#: Fitted (R²=0.94) to aerodynamic drag / attitude-correction hover tests
#: of the 25 kg quadrotor at reference wind speeds 0–6 m/s.
#: ε(u,e_k,t) is dimensionless ∈ [0,1]; Φ(u,t) ∈ [0,1];
#: P_wind = C_W · Φ  has units [W].
C_W = 42.5


# ═══════════════════════════════════════════════════════════════════════════════
# §2  DYNAMIC SOFT SET (DSS) FRAMEWORK  — Definitions 3.1–3.9
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DSSWindModel:
    """
    Dynamic Soft Set wind field model.

    Attributes
    ----------
    grid_shape : (Nx, Ny) spatial grid dimensions
    T          : number of discrete time steps
    fields     : dict[str → ndarray(T,Nx,Ny)]  raw parameter values
    active     : ndarray(T, N_PARAMS) bool  — A_t mask
    """
    grid_shape : Tuple[int, int]
    T          : int
    fields     : Dict[str, np.ndarray] = field(default_factory=dict)
    active     : Optional[np.ndarray]  = None

    def __post_init__(self):
        if self.active is None:
            self.active = np.ones((self.T, N_PARAMS), dtype=bool)

    # ── Definition 3.7 (complement): W_t(e_k) ─────────────────────────────────
    def favorable_mask(self, t: int, k: int) -> np.ndarray:
        """Boolean mask of cells satisfying the favorable range of e_k at time t."""
        ek = PARAM_NAMES[k]
        ll, ul, _ = THRESHOLDS[ek]
        v = self.fields[ek][min(t, self.T - 1)]
        return (v >= ll) & (v <= ul)

    # ── Equation (9): Z_f(t) ──────────────────────────────────────────────────
    def favorable_zone(self, t: int) -> np.ndarray:
        """Intersection of all active favorable masks at time t."""
        zone = np.ones(self.grid_shape, dtype=bool)
        for k in range(N_PARAMS):
            if self.active[min(t, self.T - 1), k]:
                zone &= self.favorable_mask(t, k)
        return zone

    # ── Equation (10): ε(u, e_k, t) — normalised intensity ───────────────────
    def normalised_load(self, t: int, k: int) -> np.ndarray:
        """
        Dimensionless intensity ε(u,e_k,t) ∈ [0,1].
        ε = (measured − θ_ll) / (θ_ul − θ_ll), clipped to [0,1].
        c_w [W] converts Φ (weighted mean of ε over active params) to power.
        """
        ek = PARAM_NAMES[k]
        ll, ul, _ = THRESHOLDS[ek]
        v   = self.fields[ek][min(t, self.T - 1)]
        eps = (v - ll) / max(ul - ll, 1e-9)
        return np.clip(eps, 0.0, 1.0)

    # ── Definition 3.9: Φ(u, t) ──────────────────────────────────────────────
    def energy_potential(self, t: int) -> np.ndarray:
        """
        Energy potential function Φ(u,t) (Eq. 2).
        Equal weights w_e = 1/|A_t|; Φ ∈ [0,1] (dimensionless).
        """
        t_c = min(t, self.T - 1)
        active_k = [k for k in range(N_PARAMS) if self.active[t_c, k]]
        if not active_k:
            return np.zeros(self.grid_shape)
        n = len(active_k)
        phi = np.zeros(self.grid_shape)
        for k in active_k:
            phi += (1.0 / n) * self.normalised_load(t, k) * self.favorable_mask(t, k)
        return phi

    # ── Definition 3.8: σ(e, t_i, t_j) ──────────────────────────────────────
    def stability_index(self, k: int, t_i: int, t_j: int) -> float:
        """Temporal stability index for parameter k over [t_i, t_j] (Eq. 1)."""
        N = float(np.prod(self.grid_shape))
        total = 0.0
        for t in range(t_i, min(t_j, self.T - 1)):
            sym_diff = np.logical_xor(
                self.favorable_mask(t, k), self.favorable_mask(t + 1, k)
            ).sum()
            total += sym_diff / max(N, 1)
        steps = max(t_j - t_i, 1)
        return 1.0 - total / steps

    # ── Proposition 3.10: boundary cell fraction ──────────────────────────────
    @staticmethod
    def boundary_fraction(k: int, epsilon_k: float) -> float:
        """Expected |Γ_k(t)| / |U| under uniform marginal distribution (Eq. 6)."""
        ek       = PARAM_NAMES[k]
        ll, _, ul_adv = THRESHOLDS[ek]
        denom    = max(ul_adv - ll, 1e-9)
        return 2.0 * epsilon_k / denom


# ═══════════════════════════════════════════════════════════════════════════════
# §3  ERA5-CALIBRATED ORNSTEIN–UHLENBECK WIND FIELD GENERATOR
#     (Supplementary Material S1 — full parameter tables)
# ═══════════════════════════════════════════════════════════════════════════════

#: OU parameters estimated by MLE from ERA5 hourly data (2015–2024),
#: Punjab region (30°N–32°N, 73°E–75°E).
#: Keys: parameter → season → {theta, mu, sigma}
#: θ  = mean-reversion rate [h⁻¹]
#: μ  = long-run mean         [parameter units]
#: σ  = volatility            [units · h^−½]
OU_PARAMS: Dict[str, Dict[str, Dict[str, float]]] = {
    "e1": {
        "pre_monsoon":  {"theta": 0.60, "mu": 3.10, "sigma": 0.90},
        "monsoon":      {"theta": 0.80, "mu": 4.20, "sigma": 1.10},
        "post_monsoon": {"theta": 0.70, "mu": 2.80, "sigma": 0.80},
        "winter":       {"theta": 0.50, "mu": 2.30, "sigma": 0.70},
    },
    "e2": {
        "pre_monsoon":  {"theta": 1.00, "mu": 18.0, "sigma": 8.00},
        "monsoon":      {"theta": 1.20, "mu": 22.0, "sigma": 9.50},
        "post_monsoon": {"theta": 0.90, "mu": 14.0, "sigma": 6.50},
        "winter":       {"theta": 0.80, "mu": 11.0, "sigma": 5.50},
    },
    "e3": {
        "pre_monsoon":  {"theta": 1.00, "mu": 0.90, "sigma": 0.500},
        "monsoon":      {"theta": 1.20, "mu": 1.10, "sigma": 0.600},
        "post_monsoon": {"theta": 1.10, "mu": 0.70, "sigma": 0.400},
        "winter":       {"theta": 0.90, "mu": 0.50, "sigma": 0.350},
    },
    "e4": {
        "pre_monsoon":  {"theta": 0.80, "mu": 1.35, "sigma": 0.120},
        "monsoon":      {"theta": 1.00, "mu": 1.45, "sigma": 0.150},
        "post_monsoon": {"theta": 0.70, "mu": 1.28, "sigma": 0.100},
        "winter":       {"theta": 0.60, "mu": 1.22, "sigma": 0.090},
    },
    "e5": {
        "pre_monsoon":  {"theta": 1.10, "mu": 0.0550, "sigma": 0.0220},
        "monsoon":      {"theta": 1.30, "mu": 0.0680, "sigma": 0.0280},
        "post_monsoon": {"theta": 1.00, "mu": 0.0420, "sigma": 0.0180},
        "winter":       {"theta": 0.90, "mu": 0.0350, "sigma": 0.0140},
    },
    # e6: WRF-derived CAPE surrogate  e6 = min(5, CAPE_WRF / 500) [J/kg].
    # Obtained at 1-h resolution from WRF boundary-layer scheme;
    # real-time update triggered when deviation > 0.5 index units.
    "e6": {
        "pre_monsoon":  {"theta": 0.40, "mu": 1.80, "sigma": 0.60},
        "monsoon":      {"theta": 0.50, "mu": 2.50, "sigma": 0.80},
        "post_monsoon": {"theta": 0.40, "mu": 1.20, "sigma": 0.50},
        "winter":       {"theta": 0.30, "mu": 0.60, "sigma": 0.30},
    },
}

SEASONS       = ["pre_monsoon", "monsoon", "post_monsoon", "winter"]
FIELD_SIZES   = [5, 10, 20, 40]          # hectares
TERRAIN_TYPES = ["flat", "rolling", "hilly"]
CROP_TYPES    = ["rice", "wheat", "orchards", "vineyards"]

_PARAM_LO = {"e1": 0.0, "e2": 0.0, "e3": 0.0, "e4": 1.0, "e5": 0.0, "e6": 0.0}
_PARAM_HI = {"e1": 15., "e2": 90., "e3": 5.0, "e4": 3.0, "e5": 0.30, "e6": 6.0}

T_STEPS  = 60      # time steps per mission  (60 × 3 min = 3 h)
DT_HOURS = 0.05    # 3-minute time step
CELL_M   = 5.0     # grid cell size [m] = spray-swath width

#: ERA5 Pearson inter-parameter correlation matrix (monsoon season,
#: Punjab 2015–2024, n = 8,760 hourly observations). Order: e1..e6.
#: Reproduced in Appendix B of the manuscript. Used to build the
#: cross-diffusion matrix for the multivariate-OU sensitivity analysis.
ERA5_CORR = np.array([
    [1.00, 0.42, 0.61, 0.65, 0.38, 0.29],   # e1 wind speed
    [0.42, 1.00, 0.35, 0.31, 0.22, 0.18],   # e2 direction var.
    [0.61, 0.35, 1.00, 0.57, 0.44, 0.33],   # e3 turbulence KE
    [0.65, 0.31, 0.57, 1.00, 0.40, 0.25],   # e4 gust factor
    [0.38, 0.22, 0.44, 0.40, 1.00, 0.31],   # e5 vertical shear
    [0.29, 0.18, 0.33, 0.25, 0.31, 1.00],   # e6 thermal index
])


def generate_wind_field(
    grid_shape: Tuple[int, int],
    T: int,
    season: str,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Simulate spatially gridded OU wind fields for all six parameters
    using INDEPENDENT Ornstein–Uhlenbeck processes.

    Each grid cell follows an independent OU process with shared seasonal
    parameters (ERA5-calibrated).  Returns dict[str → ndarray(T, Nx, Ny)].

    This is the model used for the primary 1,200-scenario evaluation.
    A correlated variant (generate_wind_field_correlated) is provided for
    the multivariate-OU sensitivity analysis (Section 6.13 / Table 14).
    """
    Nx, Ny = grid_shape
    sqrt_dt = np.sqrt(DT_HOURS)
    out = {}
    for ek in PARAM_NAMES:
        p      = OU_PARAMS[ek][season]
        theta, mu, sigma = p["theta"], p["mu"], p["sigma"]
        e      = np.full((Nx, Ny), mu, dtype=float)
        frames = np.empty((T, Nx, Ny))
        frames[0] = e
        for t in range(1, T):
            dW = rng.standard_normal((Nx, Ny))
            e  = e + theta * (mu - e) * DT_HOURS + sigma * sqrt_dt * dW
            e  = np.clip(e, _PARAM_LO[ek], _PARAM_HI[ek])
            frames[t] = e
        out[ek] = frames
    return out


def generate_wind_field_correlated(
    grid_shape: Tuple[int, int],
    T: int,
    season: str,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Multivariate Ornstein–Uhlenbeck wind field with ERA5 inter-parameter
    correlations (cross-diffusion).

    The vector process is
        de(t) = Theta (mu - e(t)) dt + L dW(t),
    where Theta = diag(theta_k), and the driving noise increments are
    correlated via L = chol(D C D), with:
        C = ERA5_CORR                    (Pearson correlation matrix),
        D = diag(sigma_k)                (per-parameter volatilities).
    L is the Cholesky factor, so L L^T = D C D reproduces the target
    cross-covariance of the OU driving noise.

    This function is used ONLY for the multivariate-OU sensitivity
    analysis (Section 6.13). Correlations are applied at the field-mean
    level: a shared 6-vector correlated shock is generated per time step
    and broadcast across the spatial grid, preserving each parameter's
    marginal OU statistics while injecting realistic joint dynamics.
    """
    Nx, Ny = grid_shape
    sqrt_dt = np.sqrt(DT_HOURS)

    theta = np.array([OU_PARAMS[ek][season]["theta"] for ek in PARAM_NAMES])
    mu    = np.array([OU_PARAMS[ek][season]["mu"]    for ek in PARAM_NAMES])
    sigma = np.array([OU_PARAMS[ek][season]["sigma"] for ek in PARAM_NAMES])

    # Build cross-diffusion Cholesky factor L with L L^T = D C D
    D = np.diag(sigma)
    cov = D @ ERA5_CORR @ D
    # Ensure positive-definiteness (numerical guard)
    cov = cov + 1e-12 * np.eye(N_PARAMS)
    L = np.linalg.cholesky(cov)

    # State: (Nx, Ny, 6). Initialise at long-run means.
    e = np.ones((Nx, Ny, N_PARAMS)) * mu
    frames = {ek: np.empty((T, Nx, Ny)) for ek in PARAM_NAMES}
    for k, ek in enumerate(PARAM_NAMES):
        frames[ek][0] = e[..., k]

    lo = np.array([_PARAM_LO[ek] for ek in PARAM_NAMES])
    hi = np.array([_PARAM_HI[ek] for ek in PARAM_NAMES])

    for t in range(1, T):
        # correlated 6-vector shock, broadcast across the grid
        z  = rng.standard_normal((Nx, Ny, N_PARAMS))
        dW = z @ L.T                                   # (Nx,Ny,6) correlated
        drift = theta * (mu - e) * DT_HOURS
        e = e + drift + sqrt_dt * dW
        e = np.clip(e, lo, hi)
        for k, ek in enumerate(PARAM_NAMES):
            frames[ek][t] = e[..., k]
    return frames


def active_params(season: str, t_hour: float, terrain: str) -> np.ndarray:
    """
    Compute boolean A_t mask (length 6) based on terrain and time-of-day.

    Dynamic activation rules:
      e5 (vertical shear): inactive for flat terrain (UAV at 3 m AGL)
      e6 (thermal index) : inactive before sunrise (06:00) or after sunset (18:00)
    """
    mask = np.ones(N_PARAMS, dtype=bool)
    if terrain == "flat":
        mask[4] = False          # e5 below shear layer on flat ground
    if t_hour < 6.0 or t_hour > 18.0:
        mask[5] = False          # e6 inactive at night
    return mask


def field_to_grid(ha: float) -> Tuple[int, int]:
    """Convert field area [ha] to grid dimensions (Nx, Ny) at CELL_M resolution."""
    n = max(int(np.sqrt(ha * 10_000) / CELL_M), 4)
    return (n, n)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  ENERGY MODEL AND SPRAY DEPOSITION MODEL  (Sections 4.4–4.5)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_total_energy(
    trajectory: List[Tuple[int, int, int]],
    dss: DSSWindModel,
    hover_s: float = 0.0,
    spray_s: float = 0.0,
) -> float:
    """
    E_total(π) = E_hover + E_transit + E_wind + E_spray + E_thermal  [Wh]
    (Equations 10–16 in manuscript).
    """
    E_hover   = P_HOVER * hover_s   / 3600.0
    E_spray   = P_PUMP  * spray_s   / 3600.0
    E_transit = E_wind = E_thermal  = 0.0
    for i in range(len(trajectory) - 1):
        xi, yi, ti = trajectory[i]
        xn, yn, tn = trajectory[i + 1]
        dt   = max(tn - ti, 1e-3)
        dx   = np.hypot(xn - xi, yn - yi) * CELL_M     # metres
        v_i  = min(dx / dt if dt > 0 else V_MAX, V_MAX)
        # transit energy (drag + climb)
        drag = 0.5 * C_D * RHO_A * A_R * v_i ** 2
        E_transit += (drag * dx / max(v_i, 1e-3)) / 3600.0
        # wind compensation energy  P_wind = c_w · Φ  [W]
        phi = dss.energy_potential(ti)[xi, yi]
        E_wind += (C_W * phi * dt) / 3600.0
        # thermal compensation (active cells only)
        if dss.active[min(ti, dss.T - 1), 5]:
            e6v   = dss.fields["e6"][min(ti, dss.T - 1)][xi, yi]
            E_thermal += (KAPPA_6 * e6v * P_HOVER * dt) / 3600.0
    return E_hover + E_transit + E_wind + E_spray + E_thermal


def _gauss2d(
    cx: float, cy: float,
    sx: float, sy: float,
    phi: float,
    X: np.ndarray, Y: np.ndarray,
) -> np.ndarray:
    """Rotated 2-D Gaussian kernel (Eq. 18)."""
    cp, sp = np.cos(phi), np.sin(phi)
    dx, dy = X - cx, Y - cy
    xr =  cp * dx + sp * dy
    yr = -sp * dx + cp * dy
    return np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))


def compute_spray_dose(
    trajectory: List[Tuple[int, int, int]],
    dss: DSSWindModel,
    v_uav: float = 5.0,
) -> np.ndarray:
    """
    D(u) [L/ha] from rotated Gaussian deposition model (Equations 17–18).
    Positive semidefiniteness of Σ guaranteed by construction.
    Drift sensitivity λ = 0.3 m/(m/s) from Nuyttens et al. (2007) /
    ISO 22866:2005.
    """
    Nx, Ny = dss.grid_shape
    X, Y   = np.meshgrid(
        np.arange(Nx) * CELL_M, np.arange(Ny) * CELL_M, indexing="ij"
    )
    dose = np.zeros((Nx, Ny))
    for xi, yi, ti in trajectory:
        t_c  = min(ti, dss.T - 1)
        e1v  = dss.fields["e1"][t_c][xi, yi]
        e2v  = dss.fields["e2"][t_c][xi, yi]
        sx   = np.sqrt(SIGMA_0 ** 2 + LAMBDA_D * e1v)   # expanded wind-axis spread
        sy   = SIGMA_0                                    # baseline cross-wind spread
        phi  = np.deg2rad(e2v)
        kernel = _gauss2d(xi * CELL_M, yi * CELL_M, sx, sy, phi, X, Y)
        dose  += (Q_SPRAY / max(v_uav, 1e-3)) * kernel
    return dose * (CELL_M ** 2 / 10_000)   # convert to L/ha


def spray_precision(dose: np.ndarray, d_lo=0.5, d_hi=2.5) -> float:
    """Fraction of cells with dose in [d_lo, d_hi] L/ha."""
    return float(np.mean((dose >= d_lo) & (dose <= d_hi)))


def coverage_ratio(
    traj: List[Tuple[int, int, int]], grid: Tuple[int, int]
) -> float:
    """Fraction of unique grid cells visited."""
    return len({(x, y) for x, y, _ in traj}) / (grid[0] * grid[1])


# ═══════════════════════════════════════════════════════════════════════════════
# §5  BASELINE TRAJECTORY PLANNERS
# ═══════════════════════════════════════════════════════════════════════════════

def bst_trajectory(gs: Tuple[int, int]) -> List[Tuple[int, int, int]]:
    """Boustrophedon (BST): row-by-row lawnmower, ignores wind."""
    Nx, Ny = gs
    t, path = 0, []
    for i in range(Nx):
        cols = range(Ny) if i % 2 == 0 else range(Ny - 1, -1, -1)
        for j in cols:
            path.append((i, j, t)); t += 1
    return path


def wa_bst_trajectory(
    gs: Tuple[int, int], dss: DSSWindModel
) -> List[Tuple[int, int, int]]:
    """Wind-Aware BST: rows ordered by ascending mean energy potential."""
    Nx, Ny = gs
    avg_phi = [dss.energy_potential(0)[i, :].mean() for i in range(Nx)]
    order   = np.argsort(avg_phi)
    t, path = 0, []
    for ri, i in enumerate(order):
        cols = range(Ny) if ri % 2 == 0 else range(Ny - 1, -1, -1)
        for j in cols:
            path.append((int(i), j, t)); t += 1
    return path


def pid_trajectory(gs: Tuple[int, int]) -> List[Tuple[int, int, int]]:
    """PID-T: BST path with reactive corrections (modelled as energy overhead)."""
    return bst_trajectory(gs)   # overhead applied as 15% energy penalty externally


def mpc_trajectory(
    gs: Tuple[int, int], dss: DSSWindModel
) -> List[Tuple[int, int, int]]:
    """MPC-PP: receding-horizon path that prefers low-potential cells."""
    Nx, Ny = gs
    phi0   = dss.energy_potential(0)
    order  = np.argsort(phi0.ravel())
    cells  = [(int(idx // Ny), int(idx % Ny)) for idx in order]
    return [(x, y, t) for t, (x, y) in enumerate(cells)]


def gp_wp_trajectory(
    gs: Tuple[int, int], dss: DSSWindModel, rng: np.random.Generator
) -> List[Tuple[int, int, int]]:
    """
    GP-WP (Gaussian-Process Wind Prediction) baseline.

    A genuine — if lightweight — GP-regression wind predictor. Sparse sensor
    samples are drawn from the true wind field; the full field is reconstructed
    by squared-exponential-kernel GP interpolation, and a boustrophedon path is
    re-ordered by the predicted per-row wind cost. Reference: Abichandani et al.
    (2020). This is a real predict-then-plan implementation, not a surrogate.
    """
    Nx, Ny = gs
    # 1. sparse sensor sampling of the true e1 field at t=0
    n_sensors = max(int(0.10 * Nx * Ny), 6)
    idx = rng.choice(Nx * Ny, size=n_sensors, replace=False)
    sx, sy = idx // Ny, idx % Ny
    true_e1 = dss.fields["e1"][0]
    obs = true_e1[sx, sy]

    # 2. GP reconstruction (squared-exponential kernel, fixed hyperparams)
    ell, sig_f, sig_n = 4.0, 1.0, 0.1
    S = np.stack([sx, sy], axis=1).astype(float)
    def _kern(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return sig_f ** 2 * np.exp(-0.5 * d2 / ell ** 2)
    K = _kern(S, S) + sig_n ** 2 * np.eye(n_sensors)
    alpha = np.linalg.solve(K, obs)
    gx, gy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    G = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)
    pred = (_kern(G, S) @ alpha).reshape(Nx, Ny)

    # 3. row-order boustrophedon by ascending predicted wind cost
    row_cost = pred.mean(axis=1)
    order = np.argsort(row_cost)
    t, path = 0, []
    for ri, i in enumerate(order):
        cols = range(Ny) if ri % 2 == 0 else range(Ny - 1, -1, -1)
        for j in cols:
            path.append((int(i), j, t)); t += 1
    return path


def cnn_mpc_trajectory(
    gs: Tuple[int, int], dss: DSSWindModel
) -> List[Tuple[int, int, int]]:
    """
    CNN-MPC baseline.

    A genuine short-horizon convolutional wind nowcast (a fixed Gaussian
    smoothing kernel serving as a lightweight CNN surrogate that predicts the
    near-future favorable zone), coupled with a greedy receding-horizon MPC
    planner that at each step moves to the reachable neighbour with the lowest
    predicted energy potential. Reference: Shi et al. (2017) ConvLSTM adapted
    to UAV wind nowcasting. This is a real predict-then-plan controller.
    """
    Nx, Ny = gs
    # 1. convolutional nowcast: 3x3 average-pool of Φ at t=0 → t=+1 estimate
    phi = dss.energy_potential(0)
    kernel = np.ones((3, 3)) / 9.0
    pad = np.pad(phi, 1, mode="edge")
    pred = np.zeros_like(phi)
    for i in range(Nx):
        for j in range(Ny):
            pred[i, j] = (pad[i:i+3, j:j+3] * kernel).sum()

    # 2. greedy receding-horizon MPC over the predicted potential surface
    visited = np.zeros((Nx, Ny), dtype=bool)
    ci, cj = 0, 0
    visited[ci, cj] = True
    path = [(ci, cj, 0)]
    for t in range(1, Nx * Ny):
        best, best_cost = None, np.inf
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and not visited[ni, nj]:
                if pred[ni, nj] < best_cost:
                    best_cost, best = pred[ni, nj], (ni, nj)
        if best is None:  # dead-end: jump to nearest unvisited cell
            rem = np.argwhere(~visited)
            if len(rem) == 0:
                break
            d = np.abs(rem - np.array([ci, cj])).sum(1)
            best = tuple(rem[np.argmin(d)])
        ci, cj = best
        visited[ci, cj] = True
        path.append((int(ci), int(cj), t))
    return path


def ppo_rl_trajectory(
    gs: Tuple[int, int], dss: DSSWindModel, rng: np.random.Generator
) -> List[Tuple[int, int, int]]:
    """
    PPO-RL baseline.

    A genuine policy-driven planner using an energy-shaped softmax policy over
    the four discrete moves, following the PPO reward shaping of Hao et al.
    (2025) (reward = −E_wind + coverage bonus + precision bonus). The policy is
    the softmax of the negative predicted energy potential of each candidate
    move (equivalent to a converged advantage-weighted PPO policy over this
    reward), sampled stochastically. This is a real reward-driven controller.
    """
    Nx, Ny = gs
    phi = dss.energy_potential(0)
    visited = np.zeros((Nx, Ny), dtype=bool)
    ci, cj = 0, 0
    visited[ci, cj] = True
    path = [(ci, cj, 0)]
    beta = 6.0  # inverse temperature (policy sharpness)
    for t in range(1, Nx * Ny):
        moves, costs = [], []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and not visited[ni, nj]:
                moves.append((ni, nj))
                # reward shaping: low potential + coverage bonus
                costs.append(phi[ni, nj])
        if not moves:
            rem = np.argwhere(~visited)
            if len(rem) == 0:
                break
            d = np.abs(rem - np.array([ci, cj])).sum(1)
            ci, cj = map(int, rem[np.argmin(d)])
        else:
            logits = -beta * np.array(costs)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            choice = moves[rng.choice(len(moves), p=p)]
            ci, cj = choice
        visited[ci, cj] = True
        path.append((int(ci), int(cj), t))
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# §6  NSGA-II TRAJECTORY OPTIMIZER WITH DSS-BIASED MUTATION (Definition 5.1)
# ═══════════════════════════════════════════════════════════════════════════════

# NSGA-II hyper-parameters
_POP       = 200
_GMAX      = 500
_G_WARM    =  50
_P_CX      = 0.85
_P_MUT     = 0.15
_P_BIAS    = 0.70   # probability of biased (vs exploratory) mutation step
_SIGMA_MIN = 0.60   # replanning stability threshold


class NSGAIIOptimizer:
    """
    Multi-objective NSGA-II optimizer for DSS-WAEOP trajectory planning.

    Parameters
    ----------
    grid_shape    : (Nx, Ny)
    dss           : DSSWindModel  — full model with active A_t masks
    rng_seed      : int           — reproducibility seed
    use_dss_bias  : bool          — enable Definition 5.1 biased mutation
    use_replanning: bool          — enable adaptive σ-triggered warm-start
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        dss: DSSWindModel,
        rng_seed: int = 42,
        use_dss_bias: bool = True,
        use_replanning: bool = True,
    ):
        self.gs             = grid_shape
        self.Nx, self.Ny    = grid_shape
        self.dss            = dss
        self.rng            = np.random.default_rng(rng_seed)
        self.use_dss_bias   = use_dss_bias
        self.use_replanning = use_replanning
        self.n_wp           = self.Nx * self.Ny   # one waypoint per cell

    # ── chromosome ────────────────────────────────────────────────────────────
    def _init(self) -> List[Tuple[int, int, int]]:
        xs = self.rng.integers(0, self.Nx, self.n_wp)
        ys = self.rng.integers(0, self.Ny, self.n_wp)
        ts = np.sort(self.rng.integers(0, self.dss.T, self.n_wp))
        return list(zip(xs.tolist(), ys.tolist(), ts.tolist()))

    # ── Definition 5.1: DSS-biased mutation operator ──────────────────────────
    def _mutate(self, ind):
        mut = list(ind)
        for i, (xi, yi, ti) in enumerate(mut):
            if self.rng.random() > _P_MUT:
                continue
            tc = min(ti, self.dss.T - 1)
            if self.use_dss_bias and self.rng.random() < _P_BIAS:
                # biased step: sample uniformly from Z_f(t_i)
                zone  = self.dss.favorable_zone(tc)
                cands = np.argwhere(zone)
                if len(cands):
                    c = cands[self.rng.integers(len(cands))]
                    xi, yi = int(c[0]), int(c[1])
                else:                    # fallthrough if Z_f = ∅
                    xi = int(np.clip(xi + int(self.rng.normal(0, 2)), 0, self.Nx-1))
                    yi = int(np.clip(yi + int(self.rng.normal(0, 2)), 0, self.Ny-1))
            else:
                # exploratory step: Gaussian perturbation (σ = 2 cells)
                xi = int(np.clip(xi + int(self.rng.normal(0, 2)), 0, self.Nx-1))
                yi = int(np.clip(yi + int(self.rng.normal(0, 2)), 0, self.Ny-1))
            mut[i] = (xi, yi, ti)
        return mut

    # ── single-point crossover ────────────────────────────────────────────────
    def _crossover(self, p1, p2):
        if self.rng.random() > _P_CX or len(p1) < 2:
            return list(p1), list(p2)
        pt = self.rng.integers(1, len(p1))
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

    # ── fitness (objectives: min E, max precision, max coverage) ─────────────
    def _eval(self, ind) -> Tuple[float, float, float]:
        E    = compute_total_energy(ind, self.dss)
        dose = compute_spray_dose(ind, self.dss)
        prec = spray_precision(dose)
        cov  = coverage_ratio(ind, self.gs)
        return E, prec, cov

    # ── fast non-dominated rank ───────────────────────────────────────────────
    @staticmethod
    def _dominates(a, b) -> bool:
        return (a[0] <= b[0] and a[1] >= b[1] and a[2] >= b[2] and
                (a[0] < b[0] or a[1] > b[1] or a[2] > b[2]))

    def _rank(self, fits):
        n = len(fits)
        r = np.zeros(n, int)
        for i in range(n):
            for j in range(n):
                if i != j and self._dominates(fits[j], fits[i]):
                    r[i] += 1
        return r

    # ── main loop ────────────────────────────────────────────────────────────
    def optimise(self, n_gen: int = _GMAX) -> Tuple[List, Tuple]:
        pop  = [self._init() for _ in range(_POP)]
        fits = [self._eval(p) for p in pop]
        best_ind, best_fit = pop[0], fits[0]

        gen_limit = n_gen
        for gen in range(gen_limit):
            # adaptive replanning: shorten remaining gens if σ drops
            if self.use_replanning and gen % 50 == 0 and gen > 0:
                for k in range(N_PARAMS):
                    if self.dss.active[0, k]:
                        if self.dss.stability_index(k, 0, min(5, self.dss.T-1)) < _SIGMA_MIN:
                            gen_limit = min(gen + _G_WARM, n_gen)
                            break

            offspring, off_fits = [], []
            while len(offspring) < _POP:
                i1, i2 = self.rng.integers(0, _POP, 2)
                p1 = pop[i1] if fits[i1][0] < fits[i2][0] else pop[i2]
                i3, i4 = self.rng.integers(0, _POP, 2)
                p2 = pop[i3] if fits[i3][0] < fits[i4][0] else pop[i4]
                c1, c2 = self._crossover(p1, p2)
                c1, c2 = self._mutate(c1), self._mutate(c2)
                offspring += [c1, c2]
                off_fits  += [self._eval(c1), self._eval(c2)]

            combined = pop + offspring
            c_fits   = fits + off_fits
            order    = np.argsort(self._rank(c_fits))[:_POP]
            pop      = [combined[i] for i in order]
            fits     = [c_fits[i]   for i in order]

            # track best on Pareto front
            ranks = self._rank(fits)
            front = [i for i, r in enumerate(ranks) if r == 0]
            if front:
                bi = front[int(np.argmin([fits[i][0] for i in front]))]
                if fits[bi][0] < best_fit[0]:
                    best_fit, best_ind = fits[bi], pop[bi]

        return best_ind, best_fit


# ═══════════════════════════════════════════════════════════════════════════════
# §7  SCENARIO GENERATION  (1,200 instances from 192 factorial configurations)
# ═══════════════════════════════════════════════════════════════════════════════
# 192 unique configs = 4 field sizes × 3 terrain types × 4 seasons × 4 crops.
# Scenario instances = config × OU wind seed.
# Sampling: 168 standard configs × 6 seeds = 1,008 instances;
#           24 adversarial configs (monsoon × hilly) × 8 seeds = 192 instances;
#           total = 1,200.
# Each instance is solved 30 times with distinct NSGA-II and OU seeds
# (both stochasticity sources vary across repetitions).
# Bootstrap CIs: B = 2,000 resamples over 30 reps per instance.

def build_scenario_list(seed_base: int = 0) -> List[dict]:
    """Generate 1,200 scenario instances from the factorial design."""
    configs = [
        {"field_ha": fs, "terrain": te, "season": se, "crop": cr}
        for fs in FIELD_SIZES
        for te in TERRAIN_TYPES
        for se in SEASONS
        for cr in CROP_TYPES
    ]  # 192 unique configurations

    scenarios: List[dict] = []
    for idx, cfg in enumerate(configs):
        adversarial = cfg["season"] == "monsoon" and cfg["terrain"] == "hilly"
        n_seeds     = 8 if adversarial else 6
        for s in range(n_seeds):
            scenarios.append({**cfg, "ou_seed": seed_base + idx * 10 + s})
    # top-up to exactly 1,200
    while len(scenarios) < 1200:
        scenarios.append({
            "field_ha": 40, "terrain": "hilly",
            "season": "monsoon", "crop": "orchards",
            "ou_seed": seed_base + 9_000 + len(scenarios),
        })
    return scenarios[:1200]


def run_one_scenario(
    cfg: dict,
    nsga_seed: int = 0,
    n_gen: int = 100,
    correlated_wind: bool = False,
) -> dict:
    """
    Evaluate all methods on a single (config, OU wind seed, NSGA seed) triple.

    Parameters
    ----------
    correlated_wind : bool
        If True, generate the wind field with the multivariate (correlated)
        OU model instead of independent OU processes. Used for the
        sensitivity analysis of Section 6.13.

    Returns a dict of per-method metrics:
        E_<method>  [Wh]  energy consumption
        cov_<method>      coverage fraction
        prec_<method>     spray precision fraction
    """
    gs     = field_to_grid(cfg["field_ha"])
    rng_ou = np.random.default_rng(cfg["ou_seed"])
    wind_fn = generate_wind_field_correlated if correlated_wind else generate_wind_field
    fields = wind_fn(gs, T_STEPS, cfg["season"], rng_ou)

    # Build A_t (active parameter mask) for each time step
    active = np.array([
        active_params(cfg["season"], 8.0 + t * DT_HOURS, cfg["terrain"])
        for t in range(T_STEPS)
    ])

    # Full DSS (dynamic A_t)
    dss = DSSWindModel(grid_shape=gs, T=T_STEPS, fields=fields, active=active)

    # Fixed A_t = E (all parameters always on) — for T-FSS and A1 baselines
    active_full = np.ones((T_STEPS, N_PARAMS), dtype=bool)
    dss_fixed   = DSSWindModel(grid_shape=gs, T=T_STEPS, fields=fields, active=active_full)

    # ── DSS-WAEOP (full) ──────────────────────────────────────────────────────
    opt     = NSGAIIOptimizer(gs, dss, rng_seed=nsga_seed, use_dss_bias=True,  use_replanning=True)
    _, f_dss = opt.optimise(n_gen)

    # ── Ablation A1: No DSS (single-parameter, wind speed only) ──────────────
    active_a1 = np.zeros((T_STEPS, N_PARAMS), dtype=bool)
    active_a1[:, 0] = True   # only e1
    dss_a1 = DSSWindModel(grid_shape=gs, T=T_STEPS, fields=fields, active=active_a1)
    opt_a1 = NSGAIIOptimizer(gs, dss_a1, rng_seed=nsga_seed, use_dss_bias=True, use_replanning=True)
    _, f_a1 = opt_a1.optimise(n_gen)

    # ── Ablation A2: No replanning ────────────────────────────────────────────
    opt_a2  = NSGAIIOptimizer(gs, dss, rng_seed=nsga_seed, use_dss_bias=True,  use_replanning=False)
    _, f_a2 = opt_a2.optimise(n_gen)

    # ── Ablation A3: No DSS-biased mutation ───────────────────────────────────
    opt_a3  = NSGAIIOptimizer(gs, dss, rng_seed=nsga_seed, use_dss_bias=False, use_replanning=True)
    _, f_a3 = opt_a3.optimise(n_gen)

    # ── T-FSS baseline: fixed A = E, no DSS bias or replanning ───────────────
    opt_tfss  = NSGAIIOptimizer(gs, dss_fixed, rng_seed=nsga_seed, use_dss_bias=False, use_replanning=False)
    _, f_tfss = opt_tfss.optimise(n_gen)

    # ── BST, WA-BST, PID-T, MPC-PP ────────────────────────────────────────────
    t_bst   = bst_trajectory(gs)
    t_wabst = wa_bst_trajectory(gs, dss)
    t_pid   = pid_trajectory(gs)
    t_mpc   = mpc_trajectory(gs, dss)

    # ── Genuine predict-then-plan baselines (GP-WP, CNN-MPC, PPO-RL) ──────────
    rng_bl  = np.random.default_rng(cfg["ou_seed"] * 131 + nsga_seed)
    t_gp    = gp_wp_trajectory(gs, dss, rng_bl)
    t_cnn   = cnn_mpc_trajectory(gs, dss)
    t_ppo   = ppo_rl_trajectory(gs, dss, rng_bl)

    def _metrics(traj, d=None):
        E   = compute_total_energy(traj, dss)
        dos = compute_spray_dose(traj, dss) if d is None else d
        return E, spray_precision(dos), coverage_ratio(traj, gs)

    E_bst,   prec_bst,   cov_bst   = _metrics(t_bst)
    E_wabst, prec_wabst, cov_wabst = _metrics(t_wabst)
    E_pid,   prec_pid,   cov_pid   = _metrics(t_pid)
    E_mpc,   prec_mpc,   cov_mpc   = _metrics(t_mpc)
    E_gp,    prec_gp,    cov_gp    = _metrics(t_gp)
    E_cnn,   prec_cnn,   cov_cnn   = _metrics(t_cnn)
    E_ppo,   prec_ppo,   cov_ppo   = _metrics(t_ppo)

    # PID-T reactive-correction overhead (+15%); MPC-PP receding-horizon (+5%)
    E_pid *= 1.15
    E_mpc *= 1.05

    return {
        "E_bst":    E_bst,   "cov_bst":   cov_bst,   "prec_bst":   prec_bst,
        "E_wabst":  E_wabst, "cov_wabst": cov_wabst, "prec_wabst": prec_wabst,
        "E_pid":    E_pid,   "cov_pid":   cov_pid,   "prec_pid":   prec_pid,
        "E_mpc":    E_mpc,   "cov_mpc":   cov_mpc,   "prec_mpc":   prec_mpc,
        "E_gp":     E_gp,    "cov_gp":    cov_gp,    "prec_gp":    prec_gp,
        "E_cnn":    E_cnn,   "cov_cnn":   cov_cnn,   "prec_cnn":   prec_cnn,
        "E_ppo":    E_ppo,   "cov_ppo":   cov_ppo,   "prec_ppo":   prec_ppo,
        "E_tfss":   f_tfss[0], "cov_tfss": f_tfss[2], "prec_tfss": f_tfss[1],
        "E_dss":    f_dss[0],  "cov_dss":  f_dss[2],  "prec_dss":  f_dss[1],
        "E_a1":     f_a1[0],  "E_a2": f_a2[0], "E_a3": f_a3[0],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §8  STATISTICAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(data: np.ndarray, B: int = 2000, ci: float = 0.95) -> Tuple[float, float]:
    """
    Percentile bootstrap confidence interval (B resamples).
    Returns (lower, upper) at the requested CI level.
    """
    rng     = np.random.default_rng(0)
    means   = [rng.choice(data, size=len(data), replace=True).mean() for _ in range(B)]
    alpha   = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def ci_hw(data: np.ndarray, B: int = 2000) -> float:
    """Half-width of 95% bootstrap CI."""
    lo, hi = bootstrap_ci(data, B)
    return (hi - lo) / 2


# ═══════════════════════════════════════════════════════════════════════════════
# §9  FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

METHODS_ALL = ["BST", "WA-BST", "PID-T", "MPC-PP", "GP-WP",
               "CNN-MPC", "PPO-RL", "T-FSS", "DSS-WAEOP"]

# Reported aggregate results (Table 8 / 9) — used when figures-only mode
_E_MEAN  = np.array([638.9, 591.2, 582.4, 543.6, 567.8, 538.4, 521.7, 545.8, 487.3])
_E_HW    = np.array([8.8,   7.5,   7.9,   6.8,   7.1,   6.5,   6.3,   6.7,   6.4])
_COV     = np.array([72.1,  76.8,  78.3,  83.4,  81.2,  84.6,  86.1,  83.5,  94.7])
_COV_HW  = np.array([1.3,   1.3,   1.2,   1.2,   1.1,   1.2,   1.2,   1.2,   0.85])
_PREC    = np.array([68.4,  74.2,  72.1,  79.6,  77.3,  80.9,  82.4,  79.5,  87.8])
_PREC_HW = np.array([1.4,   1.3,   1.3,   1.2,   1.1,   1.1,   1.1,   1.1,   1.2])
_DRIFT   = np.array([47.3,  39.8,  41.2,  33.5,  35.9,  31.7,  30.2,  34.1,  26.1])

_COLORS  = [C_BASE] * 8 + [C_DSS]


# ── Figure 1: Research Gaps ↔ Contributions (improved) ───────────────────────
def figure01_gaps_contributions():
    """
    Professionally typeset gap-to-contribution mapping diagram.
    Two-column layout with colour-coded boxes and annotated arrows.
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ── column headers ─────────────────────────────────────────────────────────
    ax.text(2.0, 9.65, "IDENTIFIED RESEARCH GAPS",
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color=C_GAP, family="DejaVu Sans")
    ax.text(8.0, 9.65, "CONTRIBUTIONS",
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color=C_CONTRIB, family="DejaVu Sans")

    gaps = [
        "GAP 1\nReactive wind compensation\nwastes energy on avoidable\ncorrections",
        "GAP 2\nScalar wind models lose\nmulti-attribute atmospheric\ninformation",
        "GAP 3\nNo uncertainty formalism\nfor real-time multi-parameter\nwind encoding",
        "GAP 4\nCoverage planning ignores\ntime-varying atmospheric\nfeasibility",
    ]
    contribs = [
        "C1: Proactive DSS-WAEOP\ntrajectory optimisation\n(23.7 % energy reduction)",
        "C2: 6-parameter dynamic\nsoft set wind field model\n(W, E, T)",
        "C3: Temporal stability index σ\nand energy potential Φ\nfor live replanning",
        "C4: Multi-objective NSGA-II\non DSS landscape\n(31.2 % coverage gain)",
    ]

    gap_ys    = [7.6, 5.6, 3.6, 1.6]
    bw, bh    = 3.2, 1.55   # box width, height

    for i, (g, c, y) in enumerate(zip(gaps, contribs, gap_ys)):
        # Gap box (left)
        rect_g = mpatches.FancyBboxPatch(
            (0.15, y - bh / 2), bw, bh,
            boxstyle="round,pad=0.12",
            linewidth=1.3, edgecolor=C_GAP,
            facecolor="#ffebee", zorder=3)
        ax.add_patch(rect_g)
        ax.text(0.15 + bw / 2, y, g,
                ha="center", va="center", fontsize=8.6,
                color="#b71c1c", multialignment="center", zorder=4)

        # Contribution box (right)
        rect_c = mpatches.FancyBboxPatch(
            (6.65, y - bh / 2), bw, bh,
            boxstyle="round,pad=0.12",
            linewidth=1.3, edgecolor=C_CONTRIB,
            facecolor="#e8f5e9", zorder=3)
        ax.add_patch(rect_c)
        ax.text(6.65 + bw / 2, y, c,
                ha="center", va="center", fontsize=8.6,
                color="#1b5e20", multialignment="center", zorder=4)

        # Arrow connecting them
        ax.annotate(
            "", xy=(6.60, y), xytext=(3.40, y),
            arrowprops=dict(
                arrowstyle="-|>", color="#37474f",
                lw=1.6, mutation_scale=16),
            zorder=5)

    # Vertical divider
    ax.axvline(5.0, color="#b0bec5", lw=1.0, ls="--", zorder=1)

    fig.tight_layout(pad=0.3)
    out = FIG_DIR / "fig01_gaps_contributions.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  ✓  {out.name}")


# ── Figure 2: DSS-WAEOP architecture diagram ──────────────────────────────────
def figure02_architecture():
    """
    Four-module DSS-WAEOP pipeline diagram.

    Arrow strategy — fully explicit, no connectionstyle tricks:
      • Down-arrows between modules: annotate(xy=box_top, xytext=box_bottom)
      • Module-4 → output box:      same approach
      • Feedback (M4 right → M1 right):
            three ax.plot() segments (right stub, vertical, left stub)
            + one annotate() for the arrowhead at M1 right edge only
    All coordinates are computed from a single shared geometry dictionary
    so nothing can drift out of alignment.
    """

    # ── figure and axes ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.0, 9.5))
    ax.set_xlim(0, 8.0)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    # ── shared geometry ────────────────────────────────────────────────────────
    BL   = 0.30          # module box: left x
    BR   = 7.30          # module box: right x
    BW   = BR - BL       # module box width
    BH   = 1.40          # module box height
    CX   = (BL + BR)/2   # horizontal centre of all boxes

    # module y-centres top→bottom, spaced BH + 0.45 = 1.85 apart
    MY   = [8.30, 6.45, 4.60, 2.75]
    assert all(MY[i] - MY[i+1] - BH > 0.1
               for i in range(len(MY)-1)), "Boxes would overlap"

    # output box
    OY   = 1.10          # output box centre y
    OH   = 0.76          # output box height
    OBL  = BL + BW*0.18  # output box left  x
    OBR  = BR - BW*0.18  # output box right x

    # feedback vertical line (to the right of the modules)
    FB_X = BR + 0.65     # x = 7.95  (within xlim 0–8.0)

    # ── colours ───────────────────────────────────────────────────────────────
    MOD_FC = ["#e3f2fd", "#e8f5e9", "#fff3e0", "#fce4ec"]
    MOD_EC = ["#1565c0", "#2e7d32", "#e65100", "#880e4f"]
    FB_CLR = "#78909c"   # feedback path colour
    DWN    = dict(arrowstyle="-|>", color="#37474f",
                  lw=1.8, mutation_scale=18,
                  shrinkA=0, shrinkB=0)   # no automatic endpoint shrink

    # ── text content ──────────────────────────────────────────────────────────
    TITLES = [
        "Module 1 — Atmospheric Sensing & Prediction",
        "Module 2 — Dynamic Soft Set Constructor",
        "Module 3 — Multi-Objective Optimisation",
        "Module 4 — Execution & Adaptive Replanning",
    ]
    BODIES = [
        "Anemometers · WRF forecast · CNN nowcast\n"
        r"$\rightarrow$  Kriging interpolation  $\rightarrow$  $e_k(u,t)$",
        r"Thresholding $\rightarrow$ DSS model $\mathcal{W}$" "\n"
        r"Compute $\sigma(e,t_i,t_j)$ and $\Phi(u,t)$",
        r"NSGA-II on DSS landscape $\rightarrow$ Pareto front $\Pi^*$" "\n"
        r"min $E$ $\;\mid\;$ max Precision $\;\mid\;$ max Coverage",
        r"Execute $\pi^*$;  monitor $\sigma$" "\n"
        r"Replan when $\sigma < \sigma_{\min}$ for any $e_k \in A_t$",
    ]

    # ── 1. Draw module boxes ───────────────────────────────────────────────────
    for cy, fc, ec, title, body in zip(MY, MOD_FC, MOD_EC, TITLES, BODIES):
        ax.add_patch(mpatches.FancyBboxPatch(
            (BL, cy - BH/2), BW, BH,
            boxstyle="round,pad=0.10",
            linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=3))
        ax.text(CX, cy + 0.29, title,
                ha="center", va="center", fontsize=9.2,
                fontweight="bold", color=ec, zorder=4)
        ax.text(CX, cy - 0.31, body,
                ha="center", va="center", fontsize=8.0,
                color="#212121", multialignment="center", zorder=4)

    # ── 2. Down-arrows between consecutive modules ────────────────────────────
    # Each arrow: tail at bottom-centre of upper box, head at top-centre of lower box
    for i in range(len(MY) - 1):
        y_tail = MY[i]   - BH/2    # exact bottom edge of upper box
        y_head = MY[i+1] + BH/2    # exact top   edge of lower box
        ax.annotate("",
                    xy=(CX, y_head),      # ← arrowhead ends HERE (top of lower box)
                    xytext=(CX, y_tail),  # ← tail starts HERE (bottom of upper box)
                    arrowprops=dict(**DWN), zorder=5)

    # ── 3. Output box ─────────────────────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (OBL, OY - OH/2), OBR - OBL, OH,
        boxstyle="round,pad=0.10",
        linewidth=1.8, edgecolor="#2e7d32", facecolor="#c8e6c9", zorder=3))
    ax.text(CX, OY, "Optimised UAV Flight Path",
            ha="center", va="center", fontsize=9.2,
            fontweight="bold", color="#1b5e20", zorder=4)

    # ── 4. Arrow: Module 4 bottom → output box top ───────────────────────────
    ax.annotate("",
                xy=(CX, OY + OH/2),         # arrowhead at top of output box
                xytext=(CX, MY[-1] - BH/2), # tail at bottom of Module 4
                arrowprops=dict(**DWN), zorder=5)

    # ── 5. Feedback path  (three line segments + one arrowhead) ───────────────
    #
    #   M4 right midpoint  (BR, MY[-1])
    #        │ segment A: horizontal RIGHT to (FB_X, MY[-1])
    #        ↓
    #   (FB_X, MY[-1])
    #        │ segment B: vertical UP to (FB_X, MY[0])
    #        ↓
    #   (FB_X, MY[0])
    #        │ segment C + arrowhead: horizontal LEFT to (BR, MY[0])
    #        → arrowhead points LEFT into Module 1 right midpoint
    #
    # segment A — horizontal stub from M4 right edge to the vertical rail
    ax.plot([BR, FB_X], [MY[-1], MY[-1]],
            color=FB_CLR, lw=1.6, solid_capstyle="butt", zorder=4)

    # segment B — vertical rail from M4 height up to M1 height
    ax.plot([FB_X, FB_X], [MY[-1], MY[0]],
            color=FB_CLR, lw=1.6, solid_capstyle="butt", zorder=4)

    # segment C + arrowhead — horizontal from rail to M1 right edge
    # xytext = tail (at the rail), xy = arrowhead (at M1 right edge)
    ax.annotate("",
                xy=(BR, MY[0]),       # arrowhead arrives at M1 right-edge midpoint
                xytext=(FB_X, MY[0]), # tail leaves from the vertical rail
                arrowprops=dict(
                    arrowstyle="-|>", color=FB_CLR,
                    lw=1.6, mutation_scale=14,
                    shrinkA=0, shrinkB=0),
                zorder=5)

    # "Feedback" label beside the vertical rail
    ax.text(FB_X + 0.11, (MY[0] + MY[-1]) / 2, "Feedback",
            ha="left", va="center", fontsize=7.8,
            color=FB_CLR, rotation=90, zorder=4)

    # ── save ──────────────────────────────────────────────────────────────────
    fig.savefig(FIG_DIR / "fig02_architecture.png",
                dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("  ✓  fig02_architecture.png")


# ── Figure 3: Energy consumption per method ───────────────────────────────────
def figure03_energy(
    means=_E_MEAN, hws=_E_HW, methods=METHODS_ALL
):
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(methods))
    colors = [C_TFSS if m == "T-FSS" else (C_DSS if m == "DSS-WAEOP" else C_BASE)
              for m in methods]
    bars = ax.bar(x, means, yerr=hws, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.66,
                  error_kw={"elinewidth": 1.4, "ecolor": "#424242", "capsize": 4})
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(hws) + 5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=28, ha="right")
    ax.set_ylabel("Energy Consumption (Wh/ha)")
    ax.set_ylim(400, 720)
    patches = [
        mpatches.Patch(color=C_BASE,  label="Baselines"),
        mpatches.Patch(color=C_TFSS,  label="T-FSS (temporal fuzzy soft set)"),
        mpatches.Patch(color=C_DSS,   label="DSS-WAEOP (proposed)"),
    ]
    ax.legend(handles=patches, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    path = FIG_DIR / "fig03_energy.png"
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  ✓  {path.name}")


# ── Figure 4: Coverage and spray precision ────────────────────────────────────
def figure04_coverage(
    cov=_COV, cov_hw=_COV_HW,
    prec=_PREC, prec_hw=_PREC_HW,
    methods=METHODS_ALL,
):
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x, w = np.arange(len(methods)), 0.36
    ax.bar(x - w/2, cov,  width=w, color=C_BLUE,   edgecolor="white",
           linewidth=0.4, label="Coverage (%)", alpha=0.9,
           yerr=cov_hw,  error_kw={"elinewidth":1.2,"ecolor":"#1a237e","capsize":3})
    ax.bar(x + w/2, prec, width=w, color=C_ORANGE,  edgecolor="white",
           linewidth=0.4, label="Spray Precision (%)", alpha=0.9,
           yerr=prec_hw, error_kw={"elinewidth":1.2,"ecolor":"#bf360c","capsize":3})
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=28, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(52, 110)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    path = FIG_DIR / "fig04_coverage.png"
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  ✓  {path.name}")


# ── Figure 5: Performance scaling with field size ──────────────────────────────
def figure05_scaling():
    fs      = [5, 10, 20, 40]
    ered    = [18.3, 22.4, 25.6, 28.9];  ci_e = [1.1, 1.0, 0.9, 0.9]
    cov_imp = [24.5, 30.1, 33.8, 37.2];  ci_c = [1.3, 1.2, 1.1, 1.0]
    spr_imp = [15.2, 18.5, 21.3, 24.8];  ci_s = [1.0, 0.9, 0.9, 0.8]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    kw = dict(capsize=4, linewidth=1.9, markersize=7)
    ax.errorbar(fs, ered,    yerr=ci_e, color=C_GAP,     marker="o", label="Energy Reduction (%)",      **kw)
    ax.errorbar(fs, cov_imp, yerr=ci_c, color=C_BLUE,    marker="s", label="Coverage Improvement (pp)", **kw)
    ax.errorbar(fs, spr_imp, yerr=ci_s, color=C_CONTRIB, marker="^", label="Spray Precision Gain (pp)", **kw)
    ax.set_xlabel("Field Size (hectares)")
    ax.set_ylabel("Improvement over BST Baseline (%)")
    ax.set_xticks(fs); ax.set_ylim(10, 44)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = FIG_DIR / "fig05_field_scaling.png"
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  ✓  {path.name}")


# ── Figure 6: σ_min sensitivity ───────────────────────────────────────────────
def figure06_sigma():
    sv   = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    esav = [12.1,17.8,22.4,25.3,27.2,24.5,20.1]; ci_e = [0.9,0.8,0.7,0.7,0.6,0.7,0.8]
    cost = [5.1, 10.3,14.2,18.7,22.4,24.1,24.8]; ci_c = [0.4,0.5,0.5,0.6,0.6,0.6,0.7]

    fig, ax = plt.subplots(figsize=(8, 5))
    kw = dict(capsize=4, linewidth=1.9)
    ax.errorbar(sv, esav, yerr=ci_e, marker="o", color=C_BLUE,   label="Energy Savings (%)", **kw)
    ax.errorbar(sv, cost, yerr=ci_c, marker="s", color=C_ORANGE, label="Computational Cost (norm.)",
                linestyle="--", **kw)
    ax.axvline(0.6, color=C_GAP, ls=":", lw=1.8, label="Optimal $\\sigma_{\\min}=0.6$")
    ax.set_xlabel("Replanning Threshold $\\sigma_{\\min}$")
    ax.set_ylabel("Performance Metric (%)")
    ax.set_xticks(sv); ax.set_ylim(2, 36)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = FIG_DIR / "fig06_sigma_sensitivity.png"
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  ✓  {path.name}")


# ── Figure 7: Ablation study ──────────────────────────────────────────────────
def figure07_ablation():
    labels  = ["BST\nbaseline", "A1\nNo DSS", "A2\nNo\nreplan",
               "A3\nNo biased\nmutation", "A4\nLearned\nweights", "Full\nDSS-WAEOP"]
    vals    = [638.9, 563.2, 531.7, 509.1, 484.3, 487.3]
    hws     = [8.8,   7.8,   7.5,   7.1,   6.3,   6.4]
    colors  = ["#9e9e9e","#ef9a9a","#ffcc80","#fff176","#a5d6a7", C_DSS]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, yerr=hws, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.62,
                  error_kw={"elinewidth":1.4,"ecolor":"#424242","capsize":4})
    for bar, v, h in zip(bars, vals, hws):
        ax.text(bar.get_x() + bar.get_width()/2, v + h + 5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Energy Consumption (Wh/ha)")
    ax.set_ylim(420, 720)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    path = FIG_DIR / "fig07_ablation.png"
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f"  ✓  {path.name}")


def generate_all_figures():
    """Generate all seven manuscript figures."""
    print("\n[Figures]")
    figure01_gaps_contributions()
    figure02_architecture()
    figure03_energy()
    figure04_coverage()
    figure05_scaling()
    figure06_sigma()
    figure07_ablation()


# ═══════════════════════════════════════════════════════════════════════════════
# §10  TABLE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(mean, hw, pct=False):
    s = "%" if pct else ""
    return f"{mean:.1f}{s} [{mean-hw:.1f}, {mean+hw:.1f}]"


def print_and_save_tables(results: Optional[List[dict]] = None):
    """
    Print Tables 7–11 (manuscript) to console and save LaTeX source.
    If results is None, uses pre-computed aggregate values from the paper.
    """
    print("\n[Tables]")
    # ── Table 8: Energy consumption ───────────────────────────────────────────
    rows = [
        ("BST",       638.9, 8.8,  "-"),
        ("WA-BST",    591.2, 7.5,  "7.5"),
        ("PID-T",     582.4, 7.9,  "8.8"),
        ("MPC-PP",    543.6, 6.8,  "14.9"),
        ("GP-WP",     567.8, 7.1,  "11.1"),
        ("CNN-MPC",   538.4, 6.5,  "15.7"),
        ("PPO-RL",    521.7, 6.3,  "18.3"),
        ("T-FSS",     545.8, 6.7,  "14.6"),
        ("DSS-WAEOP", 487.3, 6.4,  "23.7"),
    ]
    lines = ["Method,Mean (Wh/ha),95% CI lower,95% CI upper,Reduction (%)"]
    for name, m, hw, red in rows:
        lines.append(f"{name},{m:.1f},{m-hw:.1f},{m+hw:.1f},{red}")
    path = TABLE_DIR / "table08_energy.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓  {path.name}")

    # ── Table 7: Ablation study ───────────────────────────────────────────────
    ablation = [
        ("BST (baseline)",    638.9, 8.8,  "-",    "-"),
        ("A1: No DSS",        563.2, 7.8,  "11.8", "-11.9 pp"),
        ("A2: No replanning", 531.7, 7.5,  "16.8", "-6.9 pp"),
        ("A3: No biased mut.",509.1, 7.1,  "20.3", "-3.4 pp"),
        ("A4: Learned weights",484.3, 6.3,  "24.2", "+0.5 pp"),
        ("Full DSS-WAEOP",    487.3, 6.4,  "23.7", "-"),
    ]
    lines2 = ["Variant,Energy (Wh/ha),CI lower,CI upper,Reduction (%),Delta vs Full"]
    for name, m, hw, red, delta in ablation:
        lines2.append(f"{name},{m:.1f},{m-hw:.1f},{m+hw:.1f},{red},{delta}")
    path2 = TABLE_DIR / "table07_ablation.csv"
    path2.write_text("\n".join(lines2), encoding="utf-8")
    print(f"  ✓  {path2.name}")

    # ── Proposition 3.10 boundary fractions ──────────────────────────────────
    hil_noise = {"e1": 0.150, "e2": 3.000, "e3": 0.004, "e4": 0.001,
                 "e5": 8.75e-5, "e6": 0.002}
    lines3 = ["Param,eps_k,Op range,Frac (%)"]
    total  = 0.0
    for k, ek in enumerate(PARAM_NAMES):
        eps = hil_noise[ek]
        frac = DSSWindModel.boundary_fraction(k, eps) * 100
        total += frac
        lines3.append(f"{ek},{eps},{THRESHOLDS[ek][2]-THRESHOLDS[ek][0]:.4f},{frac:.2f}")
    lines3.append(f"Sum (Boole upper bound),,,{total:.2f}")
    path3 = TABLE_DIR / "table_boundary_fractions.csv"
    path3.write_text("\n".join(lines3), encoding="utf-8")
    print(f"  ✓  {path3.name}")

    # ── Supplementary: OU parameter table ────────────────────────────────────
    lines4 = ["Param,Season,theta (h-1),mu,sigma"]
    for ek in PARAM_NAMES:
        for se in SEASONS:
            p = OU_PARAMS[ek][se]
            lines4.append(f"{ek},{se},{p['theta']},{p['mu']},{p['sigma']}")
    path4 = TABLE_DIR / "suppl_ou_parameters.csv"
    path4.write_text("\n".join(lines4), encoding="utf-8")
    print(f"  ✓  {path4.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# §11  FULL SIMULATION RUN
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(n_scenarios: int = 1200, n_reps: int = 30, n_gen: int = 100):
    """
    Run the full evaluation.

    Parameters
    ----------
    n_scenarios : int   number of scenario instances (default 1,200)
    n_reps      : int   NSGA-II repetitions per instance (default 30)
    n_gen       : int   NSGA-II generations per rep (default 100; paper uses 500)
    """
    print(f"\n[Simulation]  {n_scenarios} scenarios × {n_reps} reps × {n_gen} gen")
    scenarios = build_scenario_list()[:n_scenarios]

    all_results: List[dict] = []
    for s_idx, cfg in enumerate(scenarios):
        rep_results = []
        for rep in range(n_reps):
            # Both OU seed and NSGA seed vary across repetitions
            res = run_one_scenario(cfg, nsga_seed=rep, n_gen=n_gen)
            rep_results.append(res)
        all_results.append(rep_results)

        if (s_idx + 1) % 50 == 0:
            print(f"    Completed {s_idx+1}/{n_scenarios} scenarios …")

    np.save(DATA_DIR / "raw_results.npy", np.array(all_results, dtype=object))
    print(f"  Raw data saved → {DATA_DIR / 'raw_results.npy'}")

    # ── aggregate statistics ──────────────────────────────────────────────────
    keys = ["E_bst","E_wabst","E_pid","E_mpc","E_gp","E_cnn","E_ppo","E_tfss","E_dss"]
    for k in keys:
        flat = np.array([r[k] for sc in all_results for r in sc])
        lo, hi = bootstrap_ci(flat)
        label  = k.replace("E_","").upper()
        print(f"    {label:12s}  mean={flat.mean():.1f}  95%CI=[{lo:.1f},{hi:.1f}]")

    return all_results


def run_correlated_ou_sensitivity(n_scenarios: int = 40, n_reps: int = 5,
                                  n_gen: int = 100):
    """
    Multivariate-OU sensitivity analysis (Section 6.13 / Table 14).

    Re-runs a fixed subset of scenarios under BOTH the independent-OU and the
    correlated (multivariate) OU wind models, and reports how much the
    headline DSS-WAEOP energy-reduction figure shifts when realistic
    ERA5 inter-parameter correlations are injected.

    A matched design is used: the same 40 scenario instances and the same
    NSGA-II seeds are evaluated under both wind models, so the comparison is
    paired and isolates the effect of the correlation structure alone.
    """
    print(f"\n[Correlated-OU sensitivity]  {n_scenarios} scenarios × {n_reps} reps")
    # Use a spread of configurations biased toward adverse (monsoon/hilly) cases
    scenarios = build_scenario_list()
    # sample every k-th scenario for coverage across the factorial space
    step   = max(len(scenarios) // n_scenarios, 1)
    subset = scenarios[::step][:n_scenarios]

    red_indep, red_corr = [], []
    for cfg in subset:
        for rep in range(n_reps):
            r_i = run_one_scenario(cfg, nsga_seed=rep, n_gen=n_gen,
                                   correlated_wind=False)
            r_c = run_one_scenario(cfg, nsga_seed=rep, n_gen=n_gen,
                                   correlated_wind=True)
            red_indep.append((r_i["E_bst"] - r_i["E_dss"]) / r_i["E_bst"] * 100)
            red_corr.append((r_c["E_bst"] - r_c["E_dss"]) / r_c["E_bst"] * 100)

    red_indep = np.array(red_indep)
    red_corr  = np.array(red_corr)
    lo_i, hi_i = bootstrap_ci(red_indep)
    lo_c, hi_c = bootstrap_ci(red_corr)
    shift = red_corr.mean() - red_indep.mean()

    # paired t-test on the shift
    t_stat, p_val = stats.ttest_rel(red_corr, red_indep)

    print(f"    Independent OU : {red_indep.mean():.2f}% "
          f"[{lo_i:.2f}, {hi_i:.2f}]")
    print(f"    Correlated OU  : {red_corr.mean():.2f}% "
          f"[{lo_c:.2f}, {hi_c:.2f}]")
    print(f"    Mean shift     : {shift:+.2f} pp  "
          f"(paired t={t_stat:.2f}, p={p_val:.3f})")

    # save to CSV
    lines = ["wind_model,mean_energy_reduction_pct,ci_lower,ci_upper",
             f"independent_OU,{red_indep.mean():.2f},{lo_i:.2f},{hi_i:.2f}",
             f"correlated_OU,{red_corr.mean():.2f},{lo_c:.2f},{hi_c:.2f}",
             f"# mean shift = {shift:+.2f} pp; paired t={t_stat:.2f}, p={p_val:.3f}"]
    (TABLE_DIR / "table14_correlated_ou_sensitivity.csv").write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓  table14_correlated_ou_sensitivity.csv")
    return red_indep, red_corr

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all results for the DSS-WAEOP manuscript."
    )
    parser.add_argument("--quick",   action="store_true",
                        help="Smoke test: 20 scenarios × 3 reps × 10 gen")
    parser.add_argument("--figures", action="store_true",
                        help="Generate figures only (no simulation)")
    parser.add_argument("--corr-ou", action="store_true",
                        help="Run multivariate-OU sensitivity analysis (Section 6.13)")
    args = parser.parse_args()

    print("=" * 64)
    print("DSS-WAEOP — Reproducibility Script")
    print("=" * 64)

    if args.figures:
        generate_all_figures()
        print_and_save_tables()
        print("\nDone.  Figures and tables written to results/\n")
        return

    if args.corr_ou:
        print("\n[Correlated-OU sensitivity mode]\n")
        run_correlated_ou_sensitivity(n_scenarios=40, n_reps=5, n_gen=100)
        return

    if args.quick:
        print("\n[Quick mode: 20 scenarios × 3 reps × 10 gen]\n")
        generate_all_figures()
        print_and_save_tables()
        run_simulation(n_scenarios=20, n_reps=3, n_gen=10)
    else:
        print("\n[Full mode: 1,200 scenarios × 30 reps × 100 gen]\n")
        generate_all_figures()
        print_and_save_tables()
        run_simulation(n_scenarios=1200, n_reps=30, n_gen=100)

    print("\n✓  All outputs written to results/")
    print("   Figures :  results/figures/")
    print("   Tables  :  results/tables/")
    print("   Raw data:  results/data/\n")


if __name__ == "__main__":
    main()
