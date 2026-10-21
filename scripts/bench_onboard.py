#!/usr/bin/env python3
"""
Onboard timing benchmark (e.g. on an NVIDIA Jetson companion computer).

Measures, on the machine it runs on, the wall-clock time of
  * the DSS layer (M1) on a forecast,
  * the full planning stage (M1 + M2: NSGA-II, default budget),
  * one stability-triggered replan (M4),
for each field size, and prints a JSON summary.  The manuscript reports times measured on a
cloud CPU core only (Table 9); this script is provided so that the onboard feasibility claim can
be checked on the target hardware before any field deployment.

Usage:  python scripts/bench_onboard.py [--sizes 5 10 20 40] [--repeats 3]
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsswaeop import config as C, planners as PL                       # noqa: E402
from dsswaeop.atmosphere import Atmosphere, Scenario                     # noqa: E402
from dsswaeop.dss import DSSModel                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=list(C.FIELD_SIZES_HA))
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    print("Timing benchmark of DSS-WAEOP on this computer (this is normal output, not an error).\n"
          "For each field size it reports the mean time, over --repeats runs, of\n"
          "  DSS    - building the dynamic soft set on the forecast (module M1),\n"
          "  plan   - the full NSGA-II planning stage at 06:00 (module M2),\n"
          "  replan - one stability-triggered replanning at a refill (module M4).\n"
          "Compare with Table 12 of the manuscript. With the default settings (field sizes 5-40 ha,\n"
          "3 repeats) this takes several minutes; e.g. --sizes 5 --repeats 1 is quick.\n"
          "A JSON summary is printed at the end.\n",
          file=sys.stderr)
    out = dict(machine=platform.platform(), processor=platform.processor(), python=platform.python_version(),
               numpy=np.__version__, results={})
    for fs in a.sizes:
        rec = dict(dss_s=[], plan_s=[], replan_s=[])
        for r in range(a.repeats):
            ctx = PL.Context(Atmosphere(Scenario(fs, "rolling", "monsoon", "wheat", seed=900 + r)))
            t0 = time.perf_counter()
            m = DSSModel(ctx.forecast, "dynamic"); m.adverse_zone(); m.phi(); m.favourable_zone()
            rec["dss_s"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            xb, X, F, opt = PL.run_nsga(ctx, "dss:dynamic", "dynamic", True, True, seed=r)
            rec["plan_s"].append(time.perf_counter() - t0)
            ls, order, sp, off = ctx.decode(xb)
            pol = PL.DSSReplanPolicy(ctx, ls, "dynamic", xb, np.random.default_rng(r), sigma_min=2.0)
            sprayed = np.zeros(ls.n_units, bool)
            sprayed[order[: len(order) // 3]] = True
            queue = [int(u) for u in order[len(order) // 3:]]
            t0 = time.perf_counter()
            pol.on_base(10.0 * 3600, queue, sp, dict(sprayed=sprayed))
            rec["replan_s"].append(time.perf_counter() - t0)
        out["results"][f"{fs}ha"] = {k: dict(mean=float(np.mean(v)), max=float(np.max(v))) for k, v in rec.items()}
        print(f"{fs:>3} ha  DSS {np.mean(rec['dss_s']):.3f} s  plan {np.mean(rec['plan_s']):.1f} s  "
              f"replan {np.mean(rec['replan_s']):.1f} s", file=sys.stderr)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
