"""
Global configuration: platform, spray system, crops, atmosphere, DSS thresholds,
scenario design and optimiser settings.

Every physical constant used anywhere in the package is defined here, with its
unit and its source or rationale, so that the manuscript's Table of parameters
(Table 3) can be generated directly from this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

G = 9.81            # m s^-2, gravitational acceleration
RHO_AIR = 1.20      # kg m^-3, air density (30 degC, near sea level)
RHO_LIQ = 1.00      # kg L^-1, spray liquid density (water-based mix)


# ─────────────────────────────────────────────────────────────────────────────
# Platform: a 10-L class hexacopter sprayer
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Platform:
    m_dry: float = 16.0          # kg, airframe + battery + empty tank
    tank_L: float = 10.0         # L, spray tank capacity
    n_rotor: int = 6
    rotor_d: float = 0.762       # m, 30-inch propellers
    kappa_ind: float = 1.15      # induced-power correction factor (non-ideal inflow)
    fm_hover: float = 0.62       # rotor figure of merit at MTOW hover (sets profile power)
    eta_e: float = 0.85          # motor + ESC efficiency
    cda: float = 0.30            # m^2, equivalent flat-plate drag area (body + tank)
    p_avionics: float = 40.0     # W, flight controller, GNSS, radio
    p_pump: float = 60.0         # W, spray pump while spraying
    battery_Wh: float = 621.6    # Wh, 12 000 mAh x 51.8 V (14S Li-ion/LiPo)
    reserve: float = 0.20        # fraction of battery kept as landing reserve
    va_max: float = 15.0         # m s^-1, maximum airspeed
    v_ferry: float = 8.0         # m s^-1, ground speed when not spraying
    t_turn: float = 5.0          # s, headland turn (decelerate-yaw-accelerate) at hover power
    t_service: float = 150.0     # s, refill + battery swap on the ground
    e_takeoff_land: float = 8.0  # Wh, take-off and landing transients per sortie
    gust_limit: float = 11.0     # m s^-1, peak gust above which flight is suspended

    @property
    def disk_area(self) -> float:
        return self.n_rotor * np.pi * (self.rotor_d / 2.0) ** 2

    @property
    def mtow(self) -> float:
        return self.m_dry + self.tank_L * RHO_LIQ

    @property
    def usable_Wh(self) -> float:
        return self.battery_Wh * (1.0 - self.reserve)


PLATFORM = Platform()


# ─────────────────────────────────────────────────────────────────────────────
# Spray system and crops
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SpraySystem:
    swath: float = 5.0           # m, effective swath = lane spacing
    height: float = 3.0          # m, release height above canopy
    q_max: float = 4.0           # L min^-1, pump capacity
    sigma0: float = 2.0          # m, lateral std of a single-pass deposit in calm air
    c_downwash: float = 0.7      # fraction of induced velocity entraining droplets downward
    v_spray_min: float = 3.0     # m s^-1, lowest admissible spraying ground speed
    v_spray_max: float = 7.0     # m s^-1, highest admissible spraying ground speed


SPRAY = SpraySystem()

#: Crop-specific target application volume R* [L/ha] and droplet spectrum as three
#: size classes (diameter um, volume fraction). Targets follow typical UAV
#: low-volume practice; finer spectra are used for orchard/vine canopy penetration.
CROPS: Dict[str, Dict] = {
    "rice":      {"rate": 15.0, "classes": ((100, 0.15), (200, 0.55), (350, 0.30))},
    "wheat":     {"rate": 12.0, "classes": ((100, 0.15), (200, 0.55), (350, 0.30))},
    "orchards":  {"rate": 30.0, "classes": ((100, 0.25), (180, 0.55), (300, 0.20))},
    "vineyards": {"rate": 20.0, "classes": ((100, 0.25), (180, 0.55), (300, 0.20))},
}
#: Canopy effects at the release height (rougher tall canopies: less wind, more TKE)
CANOPY = {
    "rice": (1.00, 1.00), "wheat": (1.00, 1.00),
    "orchards": (0.85, 1.30), "vineyards": (0.92, 1.15),
}

#: Terminal settling velocity of water droplets in still air (Gunn & Kinzer 1949).
_GK_D_UM = np.array([50, 100, 200, 300, 400, 500, 600])
_GK_V = np.array([0.072, 0.27, 0.72, 1.17, 1.62, 2.06, 2.47])


def settling_velocity(d_um: float) -> float:
    return float(np.interp(d_um, _GK_D_UM, _GK_V))


#: Fraction of the turbulent displacement variance realised as a random shift of each
#: 25-m deposit segment (fluctuating-plume / meandering; Gifford 1959). The remainder
#: broadens the deposit. 0 = ensemble-average Gaussian plume.
MEANDER_FRACTION = 0.5
#: Go/no-go decisions use a 3-step (15-min) running median of measurements
GATE_MEDIAN_STEPS = 3
#: An operator defers a lane after waiting this long for it to become sprayable
MAX_WAIT_MIN = 30.0
#: DSS temporal deferral: an adverse cell is deferred only if the DSS forecast shows a
#: non-adverse slot for it at least this long before the end of operations
DEFER_MARGIN_H = 1.0

# Dose-quality bands, as fractions of the crop target rate R*
COVERAGE_MIN = 0.50              # a field cell counts as treated if dose >= 0.5 R*
PRECISION_BAND = (0.80, 1.20)    # a cell is on-prescription if 0.8 R* <= dose <= 1.2 R*


# ─────────────────────────────────────────────────────────────────────────────
# Atmosphere: six DSS parameters and their thresholds
# ─────────────────────────────────────────────────────────────────────────────
PARAMS = ("e1", "e2", "e3", "e4", "e5", "e6")
PARAM_LABEL = {
    "e1": "Mean wind speed at release height",
    "e2": "Wind-direction variability (standard deviation)",
    "e3": "Turbulence kinetic energy",
    "e4": "Gust factor (peak / mean)",
    "e5": "Vertical wind shear (0-10 m)",
    "e6": "Thermal instability index",
}
PARAM_UNIT = {"e1": "m/s", "e2": "deg", "e3": "m2/s2", "e4": "-", "e5": "1/s", "e6": "-"}

#: (favourable upper bound, adverse lower bound); favourable lower bound is the
#: physical minimum (0, or 1 for the gust factor). Class alpha: e <= fav;
#: beta: fav < e <= adv; gamma: e > adv.
THRESHOLDS: Dict[str, Tuple[float, float, float]] = {
    #        (lower, favourable_upper, adverse_lower)
    "e1": (0.00, 3.50, 6.50),
    "e2": (0.00, 15.0, 45.0),
    "e3": (0.00, 0.80, 2.00),
    "e4": (1.00, 1.30, 1.80),
    "e5": (0.00, 0.05, 0.12),
    "e6": (0.00, 2.00, 4.50),
}

#: Physical clipping ranges for simulated values
PARAM_RANGE = {"e1": (0.0, 15.0), "e2": (1.0, 90.0), "e3": (0.02, 6.0),
               "e4": (1.0, 4.0), "e5": (0.0, 0.40), "e6": (0.0, 5.0)}

#: Onboard / station measurement noise (1 sigma) used for gating and nowcasts.
SENSOR_NOISE = {"e1": 0.15, "e2": 3.0, "e3": 0.08, "e4": 0.03, "e5": 0.010, "e6": 0.25}

# Operational wind-speed rule used by all non-DSS planners (spraying suspended above it)
WIND_SPEED_LIMIT = THRESHOLDS["e1"][2]


# ─────────────────────────────────────────────────────────────────────────────
# Seasonal climatology (representative values for central Punjab; see Appendix A)
# OU anomaly parameters: theta [1/h], sigma [unit / sqrt(h)].  Diurnal mean:
#   mu(h) = mean * (1 + amp * sin(2 pi (h - 9) / 24))   (peak ~15:00, minimum ~03:00)
# ─────────────────────────────────────────────────────────────────────────────
SEASONS = ("pre_monsoon", "monsoon", "post_monsoon", "winter")
SEASON_LABEL = {"pre_monsoon": "Pre-monsoon", "monsoon": "Monsoon",
                "post_monsoon": "Post-monsoon", "winter": "Winter"}

CLIMATE: Dict[str, Dict[str, Dict[str, float]]] = {
    # e1 wind speed
    "e1": {"pre_monsoon": dict(mean=3.1, amp=0.45, theta=0.60, sigma=0.90),
           "monsoon":     dict(mean=4.2, amp=0.40, theta=0.80, sigma=1.10),
           "post_monsoon": dict(mean=2.8, amp=0.45, theta=0.70, sigma=0.80),
           "winter":      dict(mean=2.3, amp=0.50, theta=0.50, sigma=0.70)},
    # e2 direction variability: larger in light (night) winds -> negative amplitude
    "e2": {"pre_monsoon": dict(mean=18.0, amp=-0.25, theta=1.00, sigma=8.0),
           "monsoon":     dict(mean=22.0, amp=-0.25, theta=1.20, sigma=9.5),
           "post_monsoon": dict(mean=14.0, amp=-0.25, theta=0.90, sigma=6.5),
           "winter":      dict(mean=11.0, amp=-0.25, theta=0.80, sigma=5.5)},
    # e3 turbulence kinetic energy
    "e3": {"pre_monsoon": dict(mean=0.90, amp=0.60, theta=1.00, sigma=0.50),
           "monsoon":     dict(mean=1.10, amp=0.55, theta=1.20, sigma=0.60),
           "post_monsoon": dict(mean=0.70, amp=0.60, theta=1.10, sigma=0.40),
           "winter":      dict(mean=0.50, amp=0.60, theta=0.90, sigma=0.35)},
    # e5 shear: larger in stable night/morning conditions -> negative amplitude
    "e5": {"pre_monsoon": dict(mean=0.055, amp=-0.40, theta=1.10, sigma=0.022),
           "monsoon":     dict(mean=0.068, amp=-0.40, theta=1.30, sigma=0.028),
           "post_monsoon": dict(mean=0.042, amp=-0.40, theta=1.00, sigma=0.018),
           "winter":      dict(mean=0.035, amp=-0.40, theta=0.90, sigma=0.014)},
    # e6 thermal instability index (CAPE / 500 J kg^-1); column property, weak diurnal cycle
    "e6": {"pre_monsoon": dict(mean=1.8, amp=0.20, theta=0.40, sigma=0.60),
           "monsoon":     dict(mean=2.5, amp=0.20, theta=0.50, sigma=0.80),
           "post_monsoon": dict(mean=1.2, amp=0.20, theta=0.40, sigma=0.50),
           "winter":      dict(mean=0.6, amp=0.20, theta=0.30, sigma=0.30)},
}
#: Prevailing direction the wind blows FROM (deg clockwise from north) and the slow
#: wander of the mean direction (OU: theta [1/h], sigma [deg/sqrt(h)]).
PREVAILING_FROM = {"pre_monsoon": 290.0, "monsoon": 120.0, "post_monsoon": 330.0, "winter": 315.0}
DIR_WANDER = dict(theta=0.30, sigma=25.0)

#: Assumed correlation of the OU driving noise for (e1, e2, e3, e5, e6).
#: e4 is derived from e1 and e3, so it is not driven independently.
NOISE_CORR = np.array([
    #  e1     e2     e3     e5     e6
    [1.00, -0.35,  0.60,  0.30,  0.20],
    [-0.35, 1.00,  0.10,  0.10,  0.10],
    [0.60,  0.10,  1.00,  0.30,  0.35],
    [0.30,  0.10,  0.30,  1.00,  0.10],
    [0.20,  0.10,  0.35,  0.10,  1.00],
])
DRIVEN = ("e1", "e2", "e3", "e5", "e6")

GUST_PEAK_FACTOR = 2.5   # g in  G = 1 + g * sigma_u / U   (peak gust over ~3 s)

# Terrain: (target mean slope, speed-up gain s, lee TKE gain b, shear gain c, GRF std)
TERRAINS = ("flat", "rolling", "hilly")
TERRAIN = {
    "flat":    dict(slope=0.000, speedup=0.00, lee_tke=0.0, shear=0.0, grf_std=0.10),
    "rolling": dict(slope=0.035, speedup=0.12, lee_tke=6.0, shear=8.0, grf_std=0.13),
    "hilly":   dict(slope=0.085, speedup=0.25, lee_tke=6.0, shear=8.0, grf_std=0.16),
}
GRF_LENGTH = 150.0      # m, spatial correlation length of mesoscale perturbations
GRF_TAU_H = 0.5         # h, AR(1) time constant of the spatial perturbation field


# ─────────────────────────────────────────────────────────────────────────────
# Time and space discretisation
# ─────────────────────────────────────────────────────────────────────────────
DT_MIN = 5.0                      # min, atmospheric time step
DAY_START_H = 5.0                 # h, first simulated time step
DAY_END_H = 19.0                  # h, last simulated time step
N_STEPS = int((DAY_END_H - DAY_START_H) * 60 / DT_MIN) + 1
OP_START_H = 6.0                  # h, operations may start
OP_END_H = 18.5                   # h, operations must finish (daylight)
WIND_CELL = 25.0                  # m, atmospheric grid resolution
ALONG_BIN = 25.0                  # m, along-track length of a physics segment (= wind cell)
DOSE_ALONG = 5.0                  # m, along-track resolution of the dose grid
CROSS_BIN = 1.0                   # m, cross-track resolution of the dose grid
FIELD_ASPECT = 2.0                # length / width of the rectangular fields
BASE_CORNER = (0.0, 0.0)          # refill / battery station at the SW corner
HEADINGS_DEG = tuple(range(0, 180, 15))   # admissible lane headings (12 options)
N_BLOCKS = 8                      # lanes are grouped into contiguous blocks


def hour_of_step(k: int) -> float:
    return DAY_START_H + k * DT_MIN / 60.0


def step_of_hour(h: float) -> int:
    return int(round((h - DAY_START_H) * 60.0 / DT_MIN))


def solar_factor(hour) -> np.ndarray:
    """Surface heating factor S(h) in [0,1]; zero before 07:00 and after 18:00."""
    h = np.asarray(hour, dtype=float)
    return np.clip(np.sin(np.pi * (h - 7.0) / 11.0), 0.0, None) * ((h >= 7.0) & (h <= 18.0))


# ─────────────────────────────────────────────────────────────────────────────
# DSS activation rules (A_t) and planner settings
# ─────────────────────────────────────────────────────────────────────────────
GUST_ACTIVE_MIN_WIND = 2.0   # m/s; the gust factor is ill-conditioned in lighter winds
THERMAL_ACTIVE_MIN_S = 0.10  # e6 active only while surface heating S(h) exceeds this

@dataclass(frozen=True)
class OptimiserSettings:
    pop: int = 32
    gens: int = 24
    p_cx: float = 0.90
    p_mut: float = 0.25
    p_bias: float = 0.70
    sigma_min: float = 0.75       # replanning threshold on σ̂ (≈ 15th percentile on development seeds)
    sigma_window: int = 3         # steps (15 min) in the sliding stability window
    replan_pop: int = 16
    replan_gens: int = 6
    replan_min_gap_h: float = 0.5  # at most one replan every 30 min
    cov_tolerance: float = 0.01    # solution selection: keep fronts within 1 pp of best coverage


OPT = OptimiserSettings()

# Scenario factorial design
FIELD_SIZES_HA = (5, 10, 20, 40)
CROP_LIST = tuple(CROPS.keys())


if __name__ == "__main__":
    # This module only defines constants; running it prints a short summary of the main settings.
    print("dsswaeop.config: constants of the model (no computation).")
    print(f"  platform: {PLATFORM.m_dry:g}/{PLATFORM.mtow:g} kg, battery {PLATFORM.battery_Wh:g} Wh "
          f"({PLATFORM.usable_Wh:.0f} Wh usable)")
    print(f"  spray: swath {SPRAY.swath:g} m, release height {SPRAY.height:g} m, pump {SPRAY.q_max:g} L/min")
    print(f"  attributes: {', '.join(PARAMS)}; seasons: {', '.join(SEASONS)}; terrains: {', '.join(TERRAINS)}")
    print(f"  NSGA-II: population {OPT.pop}, generations {OPT.gens}, sigma_min {OPT.sigma_min:g}")
