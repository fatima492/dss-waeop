"""
Experiment campaign.  Every number, table and figure in the paper is computed from the
JSON/NPZ files written here (results/raw), by `dsswaeop.analysis` and `dsswaeop.figures`.

Experiments (Section 6):
  main   192 scenarios = full factorial 4 field sizes x 3 terrains x 4 seasons x 4 crops,
         one independent weather realisation each (seed 5000+i); methods BST, WA-BST,
         EA-CPP, GRD, NSGA-C, T-FSS, DSS-WAEOP and A2 (same plan as DSS-WAEOP, executed
         without lane skipping / replanning).  All NSGA-II planners of a scenario share the
         optimiser seed (common random numbers).
  abl    48-scenario stratified subset (every terrain x season x crop; field size cycled):
         ablations A1, A3, A4.
  sens   12-scenario subset (every terrain x season): σ_min, threshold scale, sensor noise,
         independent-vs-correlated OU driving noise, deposition meander fraction.
  seed   8 scenarios x 4 additional optimiser seeds (NSGA-C and DSS-WAEOP).
  conv   6 scenarios: hypervolume convergence of the planning stage, with vs without the
         DSS-biased mutation (60 generations).
  illus  one scenario stored in detail (wind, DSS sets, trajectories, dose maps, fronts).

Development seeds used while designing the method (1, 101-103) are disjoint from the
evaluation seeds used here (>= 5000).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np

if __package__ in (None, ""):  # also allow `python dsswaeop/<module>.py` (e.g. the IDE "Run" button)
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "dsswaeop"

from . import config as C
from . import planners as PL
from .atmosphere import Atmosphere, Scenario
from .dss import DSSModel
from .mission import simulate

ROOT = Path(os.environ.get("DSSWAEOP_RESULTS", Path(__file__).resolve().parents[1] / "results"))
RAW = ROOT / "raw"
SEED0 = 5000


# ─────────────────────────────────────────────────────────────────────────────
# Scenario design
# ─────────────────────────────────────────────────────────────────────────────
def main_scenarios() -> List[Scenario]:
    out = []
    i = 0
    for fs in C.FIELD_SIZES_HA:
        for te in C.TERRAINS:
            for se in C.SEASONS:
                for cr in C.CROP_LIST:
                    out.append(Scenario(fs, te, se, cr, seed=SEED0 + i))
                    i += 1
    return out


def _index(fs, te, se, cr) -> int:
    return (((C.FIELD_SIZES_HA.index(fs) * 3 + C.TERRAINS.index(te)) * 4
             + C.SEASONS.index(se)) * 4 + C.CROP_LIST.index(cr))


def abl_subset() -> List[int]:
    out, j = [], 0
    for te in C.TERRAINS:
        for se in C.SEASONS:
            for cr in C.CROP_LIST:
                out.append(_index(C.FIELD_SIZES_HA[j % 4], te, se, cr))
                j += 1
    return out


def sens_subset() -> List[int]:
    out = []
    for a, te in enumerate(C.TERRAINS):
        for b, se in enumerate(C.SEASONS):
            cr = C.CROP_LIST[(a + b) % 4]
            fs = (10, 20)[(a + b) % 2]
            out.append(_index(fs, te, se, cr))
    return out


SEED_SUBSET = lambda: sens_subset()[:8]
CONV_SUBSET = lambda: sens_subset()[::2]
ILLUS_INDEX = lambda: _index(20, "rolling", "pre_monsoon", "wheat")

SENS_SETTINGS = {
    # name: (kind, value)
    "sig_off": ("sigma", -1.0), "sig070": ("sigma", 0.70), "sig080": ("sigma", 0.80),
    "sig085": ("sigma", 0.85),
    "thr080": ("thr", 0.8), "thr120": ("thr", 1.2),
    "noise0": ("noise", 0.0), "noise2": ("noise", 2.0),
    "indep": ("indep", None),
    "mea0": ("meander", 0.0), "mea1": ("meander", 1.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Row extraction
# ─────────────────────────────────────────────────────────────────────────────
def _row(o: PL.PlanOutcome, ctx: PL.Context) -> Dict:
    r = o.result
    d = r.as_row()
    d.update(method=o.method, heading=o.heading, plan_s=o.plan_time_s, n_eval=o.n_eval,
             triggers=o.triggers)
    sp = r.unit_speed[np.isfinite(r.unit_speed)] if r.unit_speed is not None else []
    d["mean_speed"] = float(np.mean(sp)) if len(sp) else float("nan")
    x = o.extra.get("x")
    if x is not None:
        d["comp"] = float(x[PL.IC])
        d["speed_gene"] = float(x[PL.IS])
    if o.front is not None:
        d["front"] = np.round(o.front, 5).tolist()
        d["hv"] = o.extra.get("hv")
    if o.extra.get("replan_s"):
        d["replan_s"] = list(map(float, o.extra["replan_s"]))
    for k in ("energy_per_treated_ha",):
        if not np.isfinite(d[k]):
            d[k] = None
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in d.items()}


def _scen_meta(sc: Scenario, idx: int) -> Dict:
    return dict(idx=idx, field_ha=sc.field_ha, terrain=sc.terrain, season=sc.season,
                crop=sc.crop, seed=sc.seed, correlated=sc.correlated)


def _ctx(sc: Scenario, **kw) -> PL.Context:
    return PL.Context(Atmosphere(sc), **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────────────────────
def task_main(i: int) -> Dict:
    sc = main_scenarios()[i]
    ctx = _ctx(sc)
    rows = [_row(f(ctx), ctx) for f in (PL.run_bst, PL.run_wabst, PL.run_eacpp, PL.run_grd)]
    for m in ("NSGA-C", "T-FSS", "DSS-WAEOP"):
        for o in PL.run_nsga_method(ctx, m, seed=1000 + i, also_no_replan=(m == "DSS-WAEOP")):
            rows.append(_row(o, ctx))
    # environment summary (operating window) for stratified analysis
    k0, k1 = ctx.k0, ctx.k1
    env = ctx.truth
    dyn = DSSModel(env, "dynamic")
    summ = dict(
        mean_wind=float(env.fields["e1"][k0:k1].mean()),
        frac_reg_gate=float(ctx.gate("measured", "wind")[k0:k1].mean()),
        frac_dss_adverse=float(dyn.adverse_zone()[k0:k1].mean()),
        frac_all_adverse=float(DSSModel(env, "all").adverse_zone()[k0:k1].mean()),
        frac_fav_zone=float(dyn.favourable_zone()[k0:k1].mean()),
        active_mean=float(dyn.A[k0:k1].sum(axis=1).mean()),
    )
    return dict(task=f"main_{i:03d}", scenario=_scen_meta(sc, i), env=summ, rows=rows)


def task_abl(i: int) -> Dict:
    sc = main_scenarios()[i]
    ctx = _ctx(sc)
    rows = []
    for m in ("A1", "A3", "A4"):
        for o in PL.run_nsga_method(ctx, m, seed=1000 + i):
            rows.append(_row(o, ctx))
    return dict(task=f"abl_{i:03d}", scenario=_scen_meta(sc, i), rows=rows)


def task_sens(i: int, name: str) -> Dict:
    kind, val = SENS_SETTINGS[name]
    sc = main_scenarios()[i]
    rows = []
    if kind == "sigma":
        ctx = _ctx(sc)
        rows += [_row(o, ctx) for o in PL.run_nsga_method(ctx, "DSS-WAEOP", 1000 + i, sigma_min=val)]
    elif kind == "thr":
        ctx = _ctx(sc, thr_scale=val)
        rows += [_row(o, ctx) for o in PL.run_nsga_method(ctx, "DSS-WAEOP", 1000 + i)]
    elif kind == "noise":
        ctx = _ctx(sc, noise_scale=val)
        rows += [_row(o, ctx) for o in PL.run_nsga_method(ctx, "DSS-WAEOP", 1000 + i)]
    else:
        if kind == "indep":
            sc = Scenario(sc.field_ha, sc.terrain, sc.season, sc.crop, sc.seed, correlated=False)
            ctx = _ctx(sc)
        else:
            ctx = _ctx(sc, meander=val)
        rows.append(_row(PL.run_bst(ctx), ctx))
        for m in ("NSGA-C", "DSS-WAEOP"):
            rows += [_row(o, ctx) for o in PL.run_nsga_method(ctx, m, 1000 + i)]
    return dict(task=f"sens_{name}_{i:03d}", setting=name, scenario=_scen_meta(sc, i), rows=rows)


def task_seed(i: int, s: int) -> Dict:
    sc = main_scenarios()[i]
    ctx = _ctx(sc)
    rows = []
    for m in ("NSGA-C", "DSS-WAEOP"):
        rows += [_row(o, ctx) for o in PL.run_nsga_method(ctx, m, seed=1000 + i + 100_000 * s)]
    return dict(task=f"seed_{i:03d}_{s}", opt_seed=s, scenario=_scen_meta(sc, i), rows=rows)


def task_conv(i: int) -> Dict:
    sc = main_scenarios()[i]
    ctx = _ctx(sc)
    out = {}
    for name, biased in (("DSS-WAEOP", True), ("A3", False)):
        t0 = time.time()
        _, X, F, opt = PL.run_nsga(ctx, "dss:dynamic", "dynamic", biased, True, seed=1000 + i,
                                   gens=60, track_hv=True)
        out[name] = dict(hv=list(map(float, opt.history)), time_s=time.time() - t0)
    return dict(task=f"conv_{i:03d}", scenario=_scen_meta(sc, i), conv=out)


def task_illus() -> Dict:
    i = ILLUS_INDEX()
    sc = main_scenarios()[i]
    ctx = _ctx(sc)
    store = {}
    meta = {}
    for name, fn in (("BST", lambda: PL.run_bst(ctx, keep=True)),
                     ("NSGA-C", lambda: PL.run_nsga_method(ctx, "NSGA-C", 1000 + i, keep=True)[0]),
                     ("DSS-WAEOP", lambda: PL.run_nsga_method(ctx, "DSS-WAEOP", 1000 + i, keep=True)[0])):
        o = fn()
        r = o.result
        ls = ctx.lanes(o.heading)
        store[f"{name}_dose"] = r.dose.astype(np.float32)
        store[f"{name}_mask"] = ls.mask
        store[f"{name}_tstart"] = r.unit_tstart
        store[f"{name}_ustart"] = ls.u_start
        store[f"{name}_uend"] = ls.u_end
        store[f"{name}_ublock"] = ls.u_block
        store[f"{name}_sprayed"] = r.sprayed
        paths = r.sortie_paths or []
        if paths:
            flat = np.array([(si, *p) for si, pth in enumerate(paths) for p in pth])
            store[f"{name}_paths"] = flat
        meta[name] = dict(heading=o.heading, rate=ls.rate, x0=ls.x0, d=ls.d.tolist(),
                          n=ls.n.tolist(), row=_row(o, ctx))
        if o.front is not None:
            store[f"{name}_front"] = o.front
            # re-evaluate every first-front plan on the truth (open-loop execution)
            Ft = []
            for x in o.front_X:
                ls2, order, sp, off = ctx.decode(x)
                spec = PL.SPECS[name]
                pol = PL.SkipPolicy() if spec["skip"] else PL.Policy()
                rr = simulate(ls2, order, sp, ctx.gate("measured", spec["rule"]), ctx.truth,
                              policy=pol, deposit="fast", offsets=off)
                Ft.append(PL._objectives(rr))
            store[f"{name}_front_truth"] = np.array(Ft)
            if o.extra.get("sigma_log"):
                store[f"{name}_sigma_log"] = np.array(o.extra["sigma_log"])
    env = ctx.truth
    store["e_fields"] = np.stack([env.fields[p] for p in C.PARAMS]).astype(np.float32)
    store["dir_from"] = env.dir_from
    store["hours"] = env.hours
    store["terrain_z"] = ctx.atm.z
    for act in ("dynamic", "all"):
        m = DSSModel(env, act)
        store[f"A_{act}"] = m.A
        store[f"Zf_{act}"] = m.favourable_zone()
        store[f"Za_{act}"] = m.adverse_zone()
    m = DSSModel(env, "dynamic")
    store["Phi"] = m.phi().astype(np.float32)
    store["cls"] = np.stack([m.cls[p] for p in C.PARAMS]).astype(np.int8)
    fc = ctx.forecast
    store["fc_e1"] = fc.fields["e1"]
    store["meas_e1"] = ctx.measured.fields["e1"]
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW / "illus.npz", **store)
    meta.update(scenario=_scen_meta(sc, i), geom=dict(L=ctx.geom.length, W=ctx.geom.width,
                                                       nx=ctx.geom.nx, ny=ctx.geom.ny))
    return dict(task="illus", meta=meta)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def all_tasks(quick: bool = False):
    if quick:
        idx = sens_subset()[:4]
        t = [("main", i) for i in idx] + [("abl", i) for i in idx[:2]]
        t += [("sens", i, "sig_off") for i in idx[:2]] + [("seed", idx[0], 1), ("conv", idx[0])]
        return t
    main = list(range(192))
    # interleave large and small fields for load balance
    main = sorted(main, key=lambda i: (-main_scenarios()[i].field_ha, i))
    t = [("illus",)] + [("main", i) for i in main]
    t += [("abl", i) for i in abl_subset()]
    t += [("sens", i, n) for n in SENS_SETTINGS for i in sens_subset()]
    t += [("seed", i, s) for s in (1, 2, 3, 4) for i in SEED_SUBSET()]
    t += [("conv", i) for i in CONV_SUBSET()]
    return t


def task_name(t) -> str:
    if t[0] == "main":
        return f"main_{t[1]:03d}"
    if t[0] == "abl":
        return f"abl_{t[1]:03d}"
    if t[0] == "sens":
        return f"sens_{t[2]}_{t[1]:03d}"
    if t[0] == "seed":
        return f"seed_{t[1]:03d}_{t[2]}"
    if t[0] == "conv":
        return f"conv_{t[1]:03d}"
    return "illus"


def run_task(t):
    name = task_name(t)
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.json"
    if path.exists():
        return name, "cached", 0.0
    t0 = time.time()
    try:
        fn = {"main": task_main, "abl": task_abl, "sens": task_sens, "seed": task_seed,
              "conv": task_conv, "illus": task_illus}[t[0]]
        out = fn(*t[1:])
        out["wall_s"] = time.time() - t0
        tmp = path.with_suffix(".tmp")
        tmp.write_text(encoding="utf-8", data=json.dumps(out, default=_json_default))
        tmp.replace(path)
        return name, "ok", time.time() - t0
    except Exception:
        (RAW / f"{name}.err").write_text(encoding="utf-8", data=traceback.format_exc())
        return name, "error", time.time() - t0


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def run(quick: bool = False, workers: int = 2, only: str = ""):
    RAW.mkdir(parents=True, exist_ok=True)
    tasks = all_tasks(quick)
    if only:
        tasks = [t for t in tasks if t[0] in only.split(",")]
    t0 = time.time()
    done = 0
    with Pool(workers, maxtasksperchild=8) as pool:
        for name, status, dt in pool.imap_unordered(run_task, tasks):
            done += 1
            print(f"[{done:4d}/{len(tasks)}] {name:28s} {status:6s} {dt:7.1f}s  "
                  f"elapsed {(time.time() - t0) / 60:6.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the simulation campaign (normally started through "
                                 "reproduce_all.py --quick or --full).")
    ap.add_argument("--quick", action="store_true", help="10-task smoke test (about 5 minutes)")
    ap.add_argument("--full", action="store_true", help="all 411 tasks (about 2.5 h on 2 cores)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if not (a.quick or a.full or a.only):
        print("dsswaeop.experiments runs the simulation campaign and needs an option:\n"
              "  python dsswaeop/experiments.py --quick   (10 tasks, about 5 minutes)\n"
              "  python dsswaeop/experiments.py --full    (411 tasks, about 2.5 h on 2 cores)\n"
              "The shipped raw results in results/raw are already complete; to regenerate the tables,\n"
              "figures and numbers of the paper from them, run:  python reproduce_all.py")
    else:
        if a.quick:   # keep the smoke test apart from the shipped results (as reproduce_all.py does)
            os.environ["DSSWAEOP_RESULTS"] = str(Path(__file__).resolve().parents[1] / "results_quick")
            ROOT = Path(os.environ["DSSWAEOP_RESULTS"])
            RAW = ROOT / "raw"
        run(a.quick, a.workers, a.only)
