#!/usr/bin/env python3
"""
Re-estimate the seasonal climatology and OU parameters of config.CLIMATE from ERA5.

The manuscript uses representative values (Appendix A); this script lets users replace them
with values fitted to ERA5 hourly single-level data for any location.

Input: a CSV with one row per hour (UTC) for one grid point and the columns
    time, u10, v10, u100, v100, i10fg, zust, cape
(10-m and 100-m wind components [m/s], instantaneous 10-m wind gust [m/s], friction velocity
[m/s], CAPE [J/kg]).  Such a file can be produced with the CDS API, e.g.

    import cdsapi
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis", "format": "netcdf",
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind",
                     "100m_u_component_of_wind", "100m_v_component_of_wind",
                     "instantaneous_10m_wind_gust", "friction_velocity",
                     "convective_available_potential_energy"],
        "year": [str(y) for y in range(2015, 2025)], "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)], "time": [f"{h:02d}:00" for h in range(24)],
        "area": [31.5, 73.5, 31.5, 73.5]}, "era5_point.nc")

and converted to CSV with xarray (`ds.to_dataframe().reset_index().to_csv(...)`).

Mapping to the DSS attributes (release height h_r = 3 m, roughness z0 = 0.05 m):
  e1 = |U10| * ln(h_r/z0) / ln(10/z0)              (log profile)
  e2 = std of the hourly wind direction over a centred 3-h window   [deg] (proxy)
  e3 = 5.47 * zust^2                                  (surface-layer TKE)
  e4 = i10fg / |U10|  (|U10| >= 1 m/s)
  e5 = (|U100| - |U10|) / 90
  e6 = CAPE / 500, capped at 5
Seasons: pre-monsoon Mar-May, monsoon Jun-Sep, post-monsoon Oct-Nov, winter Dec-Feb.
For each attribute e_k and season the diurnal climatology mu_k(h) = mean_k * (1 + r_k sin(2 pi (h-9)/24))
(manuscript Eq. 12) is fitted by least squares to the local-time hourly means, and the OU parameters
(manuscript Eq. 13) from the lag-1 autocorrelation rho of the hourly anomalies:
lambda_k = -ln(rho) [1/h] and varsigma_k = sd * sqrt(2 lambda_k).
Output keys per attribute and season, as in config.CLIMATE: "mean" (bar mu_k), "amp" (r_k),
"theta" (lambda_k) and "sigma" (varsigma_k).

Usage: python scripts/fit_climatology_era5.py era5_point.csv --utc-offset 5 > climate_fit.json
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

SEASON = {3: "pre_monsoon", 4: "pre_monsoon", 5: "pre_monsoon", 6: "monsoon", 7: "monsoon",
          8: "monsoon", 9: "monsoon", 10: "post_monsoon", 11: "post_monsoon", 12: "winter",
          1: "winter", 2: "winter"}


def derive(df: pd.DataFrame, hr=3.0, z0=0.05) -> pd.DataFrame:
    ws10 = np.hypot(df.u10, df.v10)
    ws100 = np.hypot(df.u100, df.v100)
    out = pd.DataFrame(index=df.index)
    out["e1"] = ws10 * np.log(hr / z0) / np.log(10 / z0)
    ang = np.rad2deg(np.arctan2(df.u10, df.v10))
    unw = np.rad2deg(np.unwrap(np.deg2rad(ang)))
    out["e2"] = pd.Series(unw, index=df.index).rolling(3, center=True).std()
    out["e3"] = 5.47 * df.zust ** 2
    out["e4"] = (df.i10fg / ws10).where(ws10 >= 1.0)
    out["e5"] = ((ws100 - ws10) / 90.0).clip(lower=0)
    out["e6"] = (df.cape / 500.0).clip(upper=5)
    return out


def fit(series: pd.Series, hours: pd.Series):
    hm = series.groupby(hours).mean().reindex(range(24))
    h = np.arange(24)
    X = np.column_stack([np.ones(24), np.sin(2 * np.pi * (h - 9) / 24)])
    ok = np.isfinite(hm.values)
    beta, *_ = np.linalg.lstsq(X[ok], hm.values[ok], rcond=None)
    m, amp = beta[0], beta[1] / beta[0] if beta[0] != 0 else 0.0
    anom = series - hm.reindex(hours.values).values
    a = anom.dropna()
    rho = np.corrcoef(a.values[:-1], a.values[1:])[0, 1]
    rho = min(max(rho, 1e-3), 0.999)
    theta = -np.log(rho)
    sigma = a.std() * np.sqrt(2 * theta)
    return dict(mean=round(float(m), 4), amp=round(float(amp), 3), theta=round(float(theta), 3),
                sigma=round(float(sigma), 4))


def _demo_data(n_days: int = 730) -> pd.DataFrame:
    """Synthetic hourly series in the ERA5 CSV format, used only to demonstrate the script."""
    rng = np.random.default_rng(0)
    t = pd.date_range("2020-01-01", periods=24 * n_days, freq="h")
    h = (t.hour + 5) % 24
    diurnal = 1 + 0.4 * np.sin(2 * np.pi * (h - 9) / 24)
    n = len(t)

    def ar1(sd, rho=0.5):          # hourly anomalies with lag-1 autocorrelation rho
        x = np.empty(n)
        x[0] = rng.normal(0, sd)
        e = rng.normal(0, sd * np.sqrt(1 - rho ** 2), n)
        for i in range(1, n):
            x[i] = rho * x[i - 1] + e[i]
        return x
    u10 = 2.5 * diurnal + ar1(0.8)
    v10 = 1.0 + ar1(0.8)
    return pd.DataFrame({"time": t, "u10": u10, "v10": v10, "u100": 1.4 * u10, "v100": 1.4 * v10,
                         "i10fg": 1.6 * np.hypot(u10, v10) + rng.normal(0, 0.3, n),
                         "zust": 0.25 * diurnal + ar1(0.03),
                         "cape": np.clip(400 * (diurnal - 0.8) + ar1(50.0), 0, None)})


def main():
    ap = argparse.ArgumentParser(description="Fit the climatology and OU parameters to an ERA5 point "
                                 "time series (CSV). Without a CSV file, a demonstration on synthetic "
                                 "data is run.")
    ap.add_argument("csv", nargs="?", help="ERA5 hourly CSV (columns time,u10,v10,u100,v100,i10fg,zust,cape)")
    ap.add_argument("--utc-offset", type=float, default=5.0)
    a = ap.parse_args()
    if a.csv is None:
        print("No ERA5 file given: running a DEMONSTRATION on synthetic data.\n"
              "For real use, download ERA5 hourly data for your site (see the header of this script)\n"
              "and run:  python scripts/fit_climatology_era5.py era5_point.csv --utc-offset 5\n",
              file=sys.stderr)
        df = _demo_data().set_index("time")
    else:
        df = pd.read_csv(a.csv, parse_dates=["time"]).sort_values("time").set_index("time")
    d = derive(df)
    local = d.index + pd.Timedelta(hours=a.utc_offset)
    season = pd.Series(local.month, index=d.index).map(SEASON)
    hours = pd.Series(local.hour, index=d.index)
    out = {}
    for p in ("e1", "e2", "e3", "e5", "e6"):
        out[p] = {}
        for s in ("pre_monsoon", "monsoon", "post_monsoon", "winter"):
            sel = season == s
            out[p][s] = fit(d[p][sel], hours[sel])
    # correlation of hourly anomalies of the driven parameters (all seasons pooled)
    an = d[["e1", "e2", "e3", "e5", "e6"]].copy()
    for p in an:
        an[p] = an[p] - an[p].groupby([season, hours]).transform("mean")
    out["noise_corr"] = an.dropna().corr().round(2).values.tolist()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
