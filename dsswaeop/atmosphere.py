"""
Atmospheric environment: the 'truth' wind field used for evaluation, and the
forecast / nowcast / measured views that planners are allowed to see.

Structure of the truth field for parameter e_k at cell u and time t:

    e_k(u, t) = [mu_k(h(t)) + X_k(t)] * M_k(u, t)

* mu_k(h): seasonal diurnal climatology (config.CLIMATE)
* X_k(t):  field-mean anomaly, a 5-dimensional Ornstein-Uhlenbeck process with
           correlated driving noise (config.NOISE_CORR)
* M_k(u,t): spatial modulation by terrain (speed-up over crests, lee-side TKE,
           slope-enhanced shear), canopy roughness, and a spatially correlated
           Gaussian random field (GRF) that evolves as an AR(1) process.

The gust factor e4 is not driven independently; it is derived from wind speed and
turbulence as G = 1 + g sigma_u / U.  The thermal index e6 is a column property
(CAPE-based) and is spatially uniform; its effect on near-surface vertical
turbulence is gated by surface heating S(h).

Planners never see the truth.  They see
* forecast(t0):  climatology + the anomaly observed at t0 decaying at the OU rate,
                 terrain modulation known, GRF unknown (set to its mean);
* nowcast(t):    the same, re-issued at time t from the base-station measurement;
* measured:      truth + sensor noise, used for the go/no-go spray decision at a lane.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

if __package__ in (None, ""):  # also allow `python dsswaeop/<module>.py` (e.g. the IDE "Run" button)
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "dsswaeop"

from . import config as C


# ─────────────────────────────────────────────────────────────────────────────
# Field geometry
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FieldGeometry:
    area_ha: float
    length: float      # m, along x (east)
    width: float       # m, along y (north)
    nx: int            # atmospheric grid cells along x
    ny: int

    @classmethod
    def from_area(cls, area_ha: float, aspect: float = C.FIELD_ASPECT) -> "FieldGeometry":
        a = area_ha * 1e4
        length = float(np.sqrt(a * aspect))
        width = a / length
        nx = int(np.ceil(length / C.WIND_CELL))
        ny = int(np.ceil(width / C.WIND_CELL))
        return cls(area_ha, length, width, nx, ny)

    def cell_of(self, x, y):
        ix = np.clip((np.asarray(x) / C.WIND_CELL).astype(int), 0, self.nx - 1)
        iy = np.clip((np.asarray(y) / C.WIND_CELL).astype(int), 0, self.ny - 1)
        return ix, iy


# ─────────────────────────────────────────────────────────────────────────────
# Random-field helpers
# ─────────────────────────────────────────────────────────────────────────────
def _smooth_noise(shape, length_cells: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance Gaussian random field with Gaussian covariance (FFT filtering)."""
    nx, ny = shape
    px, py = 2 * nx, 2 * ny                     # zero-padded to avoid wrap-around
    kx = np.fft.fftfreq(px)[:, None]
    ky = np.fft.fftfreq(py)[None, :]
    filt = np.exp(-2.0 * (np.pi * length_cells) ** 2 * (kx ** 2 + ky ** 2))
    w = rng.standard_normal((px, py))
    f = np.real(np.fft.ifft2(np.fft.fft2(w) * filt))[:nx, :ny]
    s = f.std()
    return (f - f.mean()) / (s if s > 1e-12 else 1.0)


def make_terrain(geom: FieldGeometry, terrain: str, rng: np.random.Generator):
    """Elevation z (m) on the atmospheric grid and its gradient (dimensionless slope)."""
    spec = C.TERRAIN[terrain]
    shape = (geom.nx, geom.ny)
    if spec["slope"] <= 0:
        z = np.zeros(shape)
        return z, np.zeros(shape), np.zeros(shape)
    z = _smooth_noise(shape, length_cells=120.0 / C.WIND_CELL, rng=rng)
    gx, gy = np.gradient(z, C.WIND_CELL)
    mean_slope = np.mean(np.hypot(gx, gy))
    scale = spec["slope"] / max(mean_slope, 1e-9)
    z *= scale
    return z, gx * scale, gy * scale


def diurnal_mean(param: str, season: str, hours) -> np.ndarray:
    p = C.CLIMATE[param][season]
    h = np.asarray(hours, dtype=float)
    return p["mean"] * (1.0 + p["amp"] * np.sin(2 * np.pi * (h - 9.0) / 24.0))


def _ou_anomalies(season: str, n: int, dt_h: float, rng: np.random.Generator,
                  correlated: bool = True) -> np.ndarray:
    """Correlated OU anomalies for the driven parameters; shape (n, len(DRIVEN))."""
    th = np.array([C.CLIMATE[k][season]["theta"] for k in C.DRIVEN])
    sg = np.array([C.CLIMATE[k][season]["sigma"] for k in C.DRIVEN])
    corr = C.NOISE_CORR if correlated else np.eye(len(C.DRIVEN))
    L = np.linalg.cholesky(corr)
    # exact discretisation of the OU process
    a = np.exp(-th * dt_h)
    sd_step = sg * np.sqrt((1 - a ** 2) / (2 * th))
    sd_stat = sg / np.sqrt(2 * th)
    x = np.empty((n, len(C.DRIVEN)))
    x[0] = sd_stat * (L @ rng.standard_normal(len(C.DRIVEN)))
    for k in range(1, n):
        x[k] = a * x[k - 1] + sd_step * (L @ rng.standard_normal(len(C.DRIVEN)))
    return x


def _ou_scalar(theta, sigma, n, dt_h, rng):
    a = np.exp(-theta * dt_h)
    sd_step = sigma * np.sqrt((1 - a ** 2) / (2 * theta))
    x = np.empty(n)
    x[0] = rng.normal(0, sigma / np.sqrt(2 * theta))
    for k in range(1, n):
        x[k] = a * x[k - 1] + sd_step * rng.standard_normal()
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Environment container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Environment:
    """Gridded atmosphere over one operating day. Arrays are (T, nx, ny) unless noted."""
    geom: FieldGeometry
    season: str
    terrain: str
    crop: str
    hours: np.ndarray                 # (T,)
    fields: Dict[str, np.ndarray]     # e1..e6
    dir_from: np.ndarray              # (T,) deg, direction the wind blows from
    sigma_w: np.ndarray               # (T, nx, ny) vertical turbulent velocity, m/s
    kind: str = "truth"               # truth | forecast | nowcast | measured

    @property
    def T(self) -> int:
        return len(self.hours)

    def wind_to_unit(self) -> np.ndarray:
        """(T, 2) unit vector of the direction the wind blows TO, in the x(E)-y(N) frame."""
        to_rad = np.deg2rad((self.dir_from + 180.0) % 360.0)
        # meteorological bearing (clockwise from N) -> math frame
        return np.stack([np.sin(to_rad), np.cos(to_rad)], axis=1)

    def field_mean(self, p: str) -> np.ndarray:
        return self.fields[p].reshape(self.T, -1).mean(axis=1)


@dataclass
class Scenario:
    field_ha: float
    terrain: str
    season: str
    crop: str
    seed: int
    correlated: bool = True

    @property
    def key(self) -> str:
        return f"{self.field_ha:g}ha_{self.terrain}_{self.season}_{self.crop}_s{self.seed}"


class Atmosphere:
    """Generates and holds the truth for one scenario and produces planner views."""

    def __init__(self, sc: Scenario):
        self.sc = sc
        rng = np.random.default_rng(sc.seed)
        self.geom = FieldGeometry.from_area(sc.field_ha)
        g = self.geom
        T = C.N_STEPS
        dt_h = C.DT_MIN / 60.0
        self.hours = C.DAY_START_H + np.arange(T) * dt_h

        # terrain (independent stream so that terrain is identical across seasons/crops)
        trng = np.random.default_rng(10_000 + int(sc.field_ha) * 7 + C.TERRAINS.index(sc.terrain))
        self.z, self.gx, self.gy = make_terrain(g, sc.terrain, trng)
        spec = C.TERRAIN[sc.terrain]
        zs = self.z.std()
        self.zhat = np.clip(self.z / zs, -2, 2) / 2 if zs > 1e-9 else np.zeros_like(self.z)
        self.slope = np.hypot(self.gx, self.gy)
        self.spec = spec

        # field-mean anomalies and mean direction
        self.X = _ou_anomalies(sc.season, T, dt_h, rng, correlated=sc.correlated)
        self.D = _ou_scalar(C.DIR_WANDER["theta"], C.DIR_WANDER["sigma"], T, dt_h, rng)

        # spatial GRF (AR(1) in time); two correlated fields for speed and TKE
        rho = np.exp(-dt_h / C.GRF_TAU_H)
        lc = C.GRF_LENGTH / C.WIND_CELL
        eta1 = np.empty((T, g.nx, g.ny))
        eta3 = np.empty((T, g.nx, g.ny))
        a = _smooth_noise((g.nx, g.ny), lc, rng)
        b = _smooth_noise((g.nx, g.ny), lc, rng)
        for k in range(T):
            if k > 0:
                a = rho * a + np.sqrt(1 - rho ** 2) * _smooth_noise((g.nx, g.ny), lc, rng)
                b = rho * b + np.sqrt(1 - rho ** 2) * _smooth_noise((g.nx, g.ny), lc, rng)
            eta1[k] = a
            eta3[k] = 0.6 * a + 0.8 * b
        self.eta1, self.eta3 = eta1, eta3

        # sensor-noise realisation shared by all planners (fair comparison)
        self.noise = {p: rng.standard_normal((T, g.nx, g.ny)) * C.SENSOR_NOISE[p] for p in C.PARAMS}
        self.dir_noise = rng.standard_normal(T) * 5.0     # deg, vane noise

        self.truth = self._build(self.X, self.D, eta1, eta3, kind="truth")

    # ── construction ────────────────────────────────────────────────────────
    def _build(self, X, D, eta1, eta3, kind: str) -> Environment:
        sc, g, spec = self.sc, self.geom, self.spec
        T = len(self.hours)
        h = self.hours
        cu, ct = C.CANOPY[sc.crop]
        idx = {k: i for i, k in enumerate(C.DRIVEN)}

        def meso(p):
            lo, hi = C.PARAM_RANGE[p]
            return np.clip(diurnal_mean(p, sc.season, h) + X[:, idx[p]], lo, hi)

        e1m, e2m, e3m, e5m, e6m = (meso(p) for p in ("e1", "e2", "e3", "e5", "e6"))
        dir_from = (C.PREVAILING_FROM[sc.season] + D) % 360.0

        # wind-direction-dependent lee factor: downslope gradient along the wind
        to_rad = np.deg2rad((dir_from + 180.0) % 360.0)
        wx, wy = np.sin(to_rad), np.cos(to_rad)
        along = wx[:, None, None] * self.gx[None] + wy[:, None, None] * self.gy[None]
        lee = 1.0 + spec["lee_tke"] * np.clip(-along, 0, None)

        speedup = 1.0 + spec["speedup"] * self.zhat
        grf = spec["grf_std"]
        e1 = e1m[:, None, None] * cu * speedup[None] * (1 + grf * eta1)
        e1 = np.clip(e1, *C.PARAM_RANGE["e1"])
        e3 = (e3m[:, None, None] * ct * lee * speedup[None] * (1 + 1.5 * grf * eta3))
        e3 = np.clip(e3, *C.PARAM_RANGE["e3"])
        e2 = np.clip(e2m[:, None, None] * (1 + 2.0 * self.slope)[None] * np.ones_like(e1),
                     *C.PARAM_RANGE["e2"])
        e5 = np.clip(e5m[:, None, None] * (1 + spec["shear"] * self.slope)[None] * np.ones_like(e1),
                     *C.PARAM_RANGE["e5"])
        e6 = np.clip(e6m[:, None, None] * np.ones_like(e1), *C.PARAM_RANGE["e6"])
        ustar = np.sqrt(e3 / 5.47)
        sigma_u = 2.4 * ustar
        e4 = np.clip(1.0 + C.GUST_PEAK_FACTOR * sigma_u / np.maximum(e1, 0.5), *C.PARAM_RANGE["e4"])
        s = C.solar_factor(h)[:, None, None]
        wstar = 0.5 * e6 * s
        sigma_w = np.sqrt((1.25 * ustar) ** 2 + (0.6 * wstar) ** 2)
        f = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5, "e6": e6}
        f = {k: v.astype(np.float32) for k, v in f.items()}
        return Environment(g, sc.season, sc.terrain, sc.crop, h.copy(), f, dir_from,
                           sigma_w.astype(np.float32), kind)

    # ── planner views ───────────────────────────────────────────────────────
    def _anomaly_view(self, k0: int, X0: np.ndarray, D0: float, kind: str) -> Environment:
        """Climatology + anomaly observed at step k0 decaying at the OU rate; GRF unknown."""
        T = len(self.hours)
        dt = np.clip(self.hours - self.hours[k0], 0, None)
        th = np.array([C.CLIMATE[k][self.sc.season]["theta"] for k in C.DRIVEN])
        Xf = X0[None, :] * np.exp(-th[None, :] * dt[:, None])
        Df = D0 * np.exp(-C.DIR_WANDER["theta"] * dt)
        zeros = np.zeros((T, self.geom.nx, self.geom.ny))
        return self._build(Xf, Df, zeros, zeros, kind=kind)

    def station_anomaly(self, k: int) -> (np.ndarray, float):
        """Anomaly estimated from the base-station (cell 0,0) measurement at step k.

        The station measures local values; the planner inverts the known terrain and
        canopy modulation (but not the unknown GRF) to estimate field-mean anomalies.
        """
        sc, spec = self.sc, self.spec
        cu, ct = C.CANOPY[sc.crop]
        h = self.hours[k]
        m = {p: float(self.truth.fields[p][k, 0, 0] + self.noise[p][k, 0, 0]) for p in C.PARAMS}
        su = 1.0 + spec["speedup"] * self.zhat[0, 0]
        est = {
            "e1": m["e1"] / (cu * su),
            "e2": m["e2"] / (1 + 2.0 * self.slope[0, 0]),
            "e3": m["e3"] / (ct * su),
            "e5": m["e5"] / (1 + spec["shear"] * self.slope[0, 0]),
            "e6": m["e6"],
        }
        X0 = np.array([est[p] - float(diurnal_mean(p, sc.season, h)) for p in C.DRIVEN])
        D0 = float(((self.truth.dir_from[k] + self.dir_noise[k]) - C.PREVAILING_FROM[sc.season]
                    + 180) % 360 - 180)
        return X0, D0

    def forecast(self, k0: int) -> Environment:
        X0, D0 = self.station_anomaly(k0)
        return self._anomaly_view(k0, X0, D0, "forecast")

    def nowcast(self, k: int) -> Environment:
        X0, D0 = self.station_anomaly(k)
        return self._anomaly_view(k, X0, D0, "nowcast")

    def measured(self, noise_scale: float = 1.0, smooth: bool = True) -> Environment:
        """Truth + sensor noise, smoothed by a causal GATE_MEDIAN_STEPS running median
        (the go/no-go input).  noise_scale multiplies the sensor noise (robustness study)."""
        f = {}
        n = C.GATE_MEDIAN_STEPS if smooth else 1
        for p in C.PARAMS:
            lo, hi = C.PARAM_RANGE[p]
            raw = np.clip(self.truth.fields[p] + noise_scale * self.noise[p], lo, hi)
            if n > 1:
                stack = np.stack([np.concatenate([raw[:1].repeat(i, axis=0), raw[:len(raw) - i]])
                                  for i in range(n)])
                raw = np.median(stack, axis=0)
            f[p] = raw.astype(np.float32)
        return Environment(self.geom, self.sc.season, self.sc.terrain, self.sc.crop,
                           self.hours.copy(), f, (self.truth.dir_from + self.dir_noise) % 360,
                           self.truth.sigma_w, "measured")


if __name__ == "__main__":
    # Short demonstration: generate one scenario and summarise the true wind field.
    sc = Scenario(5, "rolling", "pre_monsoon", "wheat", seed=1)
    atm = Atmosphere(sc)
    e1 = atm.truth.fields["e1"]
    print(f"dsswaeop.atmosphere: scenario {sc.key}")
    print(f"  field {atm.geom.length:.0f} m x {atm.geom.width:.0f} m, {atm.geom.nx} x {atm.geom.ny} cells, "
          f"{e1.shape[0]} time steps")
    for h in (7, 10, 13, 16):
        k = C.step_of_hour(h)
        print(f"  {h:02d}:00  field-mean wind speed e1 = {e1[k].mean():.2f} m/s")
