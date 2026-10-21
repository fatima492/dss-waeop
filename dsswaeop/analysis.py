"""
Statistics, LaTeX tables and LaTeX macros from results/raw.

Every number quoted in the manuscript is written to results/tex/macros.tex by this
module (as \\newcommand), and every results table to results/tex/tab_*.tex, so the
paper cannot drift from the data.  CSV copies are written to results/tables/.

Statistics
  * Scenario-level means with 95 % percentile-bootstrap CIs (B = 4000, resampling
    scenarios).
  * Paired comparisons on the same scenarios (common random numbers): mean paired
    difference with bootstrap CI, two-sided Wilcoxon signed-rank test, Holm-Bonferroni
    adjustment within each metric family, matched-pairs rank-biserial correlation r.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

if __package__ in (None, ""):  # also allow `python dsswaeop/<module>.py` (e.g. the IDE "Run" button)
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "dsswaeop"

from . import config as C
from .experiments import ROOT, RAW, abl_subset, sens_subset, SEED_SUBSET, CONV_SUBSET

TEX = ROOT / "tex"
TAB = ROOT / "tables"

MAIN_METHODS = ["BST", "WA-BST", "EA-CPP", "GRD", "NSGA-C", "T-FSS", "DSS-WAEOP"]
ABL_METHODS = ["DSS-WAEOP", "A1", "A2", "A3", "A4"]
CODE = {"BST": "Bst", "WA-BST": "Wabst", "EA-CPP": "Eacpp", "GRD": "Grd", "NSGA-C": "Nsgac",
        "T-FSS": "Tfss", "DSS-WAEOP": "Dss", "A1": "Aone", "A2": "Atwo", "A3": "Athree",
        "A4": "Afour"}
METRICS = {  # column: (label, unit, lower_is_better, decimals)
    "E": ("Energy per treated hectare", "Wh/ha", True, 1),
    "cov": ("Coverage", "%", False, 1),
    "prec": ("Spray precision", "%", False, 1),
    "drift": ("Off-target deposition", "%", True, 2),
    "shift": ("Mean deposit displacement", "cm", True, 0),
    "dur": ("Mission duration", "h", True, 2),
}
B = 4000
TEX_UNIT = {"e1": r"~\si{m.s^{-1}}", "e2": r"~\si{\degree}", "e3": r"~\si{m^2.s^{-2}}", "e4": "",
            "e5": r"~\si{s^{-1}}", "e6": ""}
RNG = np.random.default_rng(20261011)


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def _rows(prefix: str) -> List[dict]:
    out = []
    for p in sorted(RAW.glob(f"{prefix}*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        sc = d["scenario"]
        for r in d["rows"]:
            row = {**sc, **{k: v for k, v in r.items() if k not in ("front",)}}
            row["task"] = d["task"]
            row["setting"] = d.get("setting", "")
            row["opt_seed"] = d.get("opt_seed", 0)
            out.append(row)
    return out


def _frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["E"] = df["energy_per_treated_ha"].astype(float)
    df["cov"] = 100 * df["coverage"]
    df["prec"] = 100 * df["precision"]
    df["drift"] = df["drift_pct"]
    df["shift"] = df["shift_cm"]
    df["dur"] = df["duration_h"]
    return df


def load_main() -> pd.DataFrame:
    return _frame(_rows("main_"))


def load_env() -> pd.DataFrame:
    rows = []
    for p in sorted(RAW.glob("main_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({**d["scenario"], **d["env"]})
    return pd.DataFrame(rows)


def load_fronts(method: str) -> Dict[int, np.ndarray]:
    out = {}
    for p in sorted(RAW.glob("main_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d["rows"]:
            if r["method"] == method and "front" in r:
                out[d["scenario"]["idx"]] = np.array(r["front"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────
def boot_ci(x, stat=np.mean, b=B):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float("nan"),) * 2
    idx = RNG.integers(0, len(x), (b, len(x)))
    s = stat(x[idx], axis=1)
    return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def wilcoxon(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    if len(nz) < 5:
        return float("nan"), float("nan")
    res = stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
    # matched-pairs rank-biserial correlation
    r = stats.rankdata(np.abs(nz))
    rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return float(res.pvalue), float(rb)


def holm(p: List[float]) -> List[float]:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * p[i])
        run = max(run, v)
        adj[i] = run
    return adj.tolist()


def paired(df: pd.DataFrame, a: str, b: str, col: str, rel: bool = False) -> pd.Series:
    """Per-scenario difference a − b (or relative, in %) aligned on scenario index."""
    pa = df[df.method == a].set_index("idx")[col]
    pb = df[df.method == b].set_index("idx")[col]
    j = pa.index.intersection(pb.index)
    if rel:
        return 100 * (pa[j] - pb[j]) / pb[j]
    return pa[j] - pb[j]


def _minus(tex: str) -> str:
    """Typeset negative numbers in LaTeX tables with a true minus sign."""
    import re
    return re.sub(r"(?<![\w$^{}\-])-(?=\d)", "$-$", tex)


def fmt(v, d=1):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    out = f"{v:,.{d}f}".replace(",", "{,}")
    return out[1:] if out.startswith("-") and float(out[1:].replace("{,}", "")) == 0 else out


def fmt_p(p):
    if not math.isfinite(p):
        return "--"
    if p < 1e-4:
        return "$<$0.0001"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# Macro registry
# ─────────────────────────────────────────────────────────────────────────────
class Macros:
    def __init__(self):
        self.d: Dict[str, str] = {}

    def set(self, name: str, value: str):
        assert name.isalpha(), name
        self.d[name] = value

    def setp(self, name: str, p: float):
        """p-value macro (table form) plus a math-mode form with its relation: \\R<name>r."""
        self.set(name, fmt_p(p))
        if math.isfinite(p):
            self.set(name + "r", "{<}\\,0.0001" if p < 1e-4 else "{=}\\," + (f"{p:.4f}" if p < 0.01 else f"{p:.3f}"))

    def write(self, path: Path):
        lines = ["% Auto-generated by dsswaeop.analysis -- do not edit by hand."]
        for k in sorted(self.d):
            v = self.d[k]
            if v.startswith("-") and v[1:2].isdigit():   # true minus sign in text and math mode
                v = "\\ensuremath{-}" + v[1:]
            lines.append(f"\\newcommand{{\\R{k}}}{{{v}}}")
        path.write_text(encoding="utf-8", data="\n".join(lines) + "\n")


def _num(n: float) -> str:
    """Letters-only name fragment for small integers / settings."""
    words = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 10: "Ten",
             20: "Twenty", 40: "Forty"}
    return words[int(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Analyses
# ─────────────────────────────────────────────────────────────────────────────
def main_table(df: pd.DataFrame, M: Macros) -> pd.DataFrame:
    rows = []
    for m in MAIN_METHODS:
        s = df[df.method == m]
        r = {"method": m, "n": len(s)}
        for col, (_, _, _, dec) in METRICS.items():
            mu = float(s[col].mean())
            lo, hi = boot_ci(s[col])
            r[col] = mu; r[col + "_lo"] = lo; r[col + "_hi"] = hi
            M.set(f"{CODE[m]}{col.capitalize()}", fmt(mu, dec))
            M.set(f"{CODE[m]}{col.capitalize()}Lo", fmt(lo, dec))
            M.set(f"{CODE[m]}{col.capitalize()}Hi", fmt(hi, dec))
        r["sorties10"] = float((s["sorties"] / s["field_ha"] * 10).mean())
        r["wait"] = float(s["wait_h"].mean())
        r["plan_s"] = float(s["plan_s"].mean())
        M.set(f"{CODE[m]}Wait", fmt(r["wait"], 2))
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "table_main.csv", index=False)

    # LaTeX
    best = {col: (t[col].min() if METRICS[col][2] else t[col].max()) for col in METRICS}
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Method & $f_1$ (\si{Wh.ha^{-1}}) & $\mathcal{C}$ (\%) & $\mathcal{P}$ (\%) & "
         r"$\mathcal{O}$ (\%) & Displacement (\si{cm}) & Duration (\si{h}) \\", r"\midrule"]
    for _, r in t.iterrows():
        cells = []
        for col, (_, _, _, dec) in METRICS.items():
            s = f"{fmt(r[col], dec)} [{fmt(r[col + '_lo'], dec)}, {fmt(r[col + '_hi'], dec)}]"
            if abs(r[col] - best[col]) < 1e-12:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        name = r"\textbf{DSS-WAEOP}" if r["method"] == "DSS-WAEOP" else r["method"]
        L.append(name + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "tab_main.tex").write_text(encoding="utf-8", data=_minus("\n".join(L) + "\n"))
    return t


def paired_tests(df: pd.DataFrame, M: Macros) -> pd.DataFrame:
    """DSS-WAEOP vs every other method, all metrics; Holm within metric."""
    out = []
    for col, (label, unit, lower, dec) in METRICS.items():
        rel = col == "E"
        ps, recs = [], []
        for m in MAIN_METHODS:
            if m == "DSS-WAEOP":
                continue
            d = paired(df, "DSS-WAEOP", m, col, rel=rel)
            p, rb = wilcoxon(d)
            lo, hi = boot_ci(d)
            better = (d < 0) if lower else (d > 0)
            recs.append(dict(metric=col, versus=m, n=len(d), mean=float(d.mean()), lo=lo, hi=hi,
                             median=float(d.median()), p=p, r_rb=rb,
                             win=float(100 * better.mean())))
            ps.append(p)
        for r, pa in zip(recs, holm(ps)):
            r["p_holm"] = pa
            out.append(r)
            tag = f"Dss{col.capitalize()}Vs{CODE[r['versus']]}"
            M.set(tag, fmt(r["mean"], dec if col != "E" else 1))
            M.set(tag + "Lo", fmt(r["lo"], dec if col != "E" else 1))
            M.set(tag + "Hi", fmt(r["hi"], dec if col != "E" else 1))
            M.setp(tag + "P", r["p_holm"])
            M.set(tag + "Win", fmt(r["win"], 0))
            M.set(tag + "Abs", fmt(abs(r["mean"]), dec if col != "E" else 1))
    t = pd.DataFrame(out)
    t.to_csv(TAB / "table_paired_tests.csv", index=False)

    # relative energy vs BST for every method (reported in text)
    for m in MAIN_METHODS:
        if m == "BST":
            continue
        d = paired(df, m, "BST", "E", rel=True)
        lo, hi = boot_ci(d)
        M.set(f"{CODE[m]}EVsBst", fmt(d.mean(), 1))
        M.set(f"{CODE[m]}EVsBstAbs", fmt(abs(d.mean()), 1))
        M.set(f"{CODE[m]}EVsBstLo", fmt(lo, 1))
        M.set(f"{CODE[m]}EVsBstHi", fmt(hi, 1))
        for col in ("cov", "prec", "drift"):
            dd = paired(df, m, "BST", col)
            M.set(f"{CODE[m]}{col.capitalize()}VsBst", fmt(dd.mean(), 1 if col != "drift" else 2))
            M.set(f"{CODE[m]}{col.capitalize()}VsBstAbs", fmt(abs(dd.mean()), 1 if col != "drift" else 2))
    # NSGA-C vs T-FSS on coverage (the fixed-parameter cost)
    d = paired(df, "DSS-WAEOP", "T-FSS", "cov")
    M.set("DssCovGainTfssMax", fmt(d.max(), 1))

    L = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Metric & DSS-WAEOP vs. & Mean paired difference [95\% CI] & Better in (\%) & "
         r"$r_{\mathrm{rb}}$ & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    for col, (label, unit, _, dec) in METRICS.items():
        if col in ("dur", "shift"):
            continue
        sub = t[t.metric == col]
        first = True
        u = r"\%" if col == "E" else ("pp" if col in ("cov", "prec") else
                                     ("pp" if col == "drift" else unit))
        plab = {"E": r"$f_1$", "cov": r"$\mathcal{C}$", "prec": r"$\mathcal{P}$",
                "drift": r"$\mathcal{O}$"}[col]
        for _, r in sub.iterrows():
            d = dec if col != "E" else 1
            lab = (plab + f" ({u})") if first else ""
            L.append(f"{lab} & {r['versus']} & {fmt(r['mean'], d)} [{fmt(r['lo'], d)}, "
                     f"{fmt(r['hi'], d)}] & {fmt(r['win'], 0)} & {fmt(r['r_rb'], 2)} & "
                     f"{fmt_p(r['p_holm'])} \\\\")
            first = False
        L.append(r"\midrule" if col != "drift" else "")
    L = [x for x in L if x]
    if L[-1] == r"\midrule":
        L = L[:-1]
    L += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "tab_paired.tex").write_text(encoding="utf-8", data=_minus("\n".join(L) + "\n"))
    return t


def stratified(df: pd.DataFrame, M: Macros) -> pd.DataFrame:
    recs = []
    for by in ("season", "terrain", "field_ha", "crop"):
        for lev, s in df.groupby(by):
            for a, b in (("DSS-WAEOP", "BST"), ("DSS-WAEOP", "NSGA-C"), ("NSGA-C", "BST"),
                         ("DSS-WAEOP", "T-FSS")):
                for col in ("E", "cov", "prec", "drift"):
                    d = paired(s, a, b, col, rel=(col == "E"))
                    recs.append(dict(by=by, level=lev, a=a, b=b, metric=col, mean=float(d.mean()),
                                     n=len(d)))
    t = pd.DataFrame(recs)
    t.to_csv(TAB / "table_stratified.csv", index=False)
    # season x terrain grid for heat maps
    grid = []
    for (se, te), s in df.groupby(["season", "terrain"]):
        for a, b in (("DSS-WAEOP", "BST"), ("DSS-WAEOP", "NSGA-C"), ("DSS-WAEOP", "T-FSS")):
            for col in ("E", "cov", "prec", "drift"):
                d = paired(s, a, b, col, rel=(col == "E"))
                grid.append(dict(season=se, terrain=te, a=a, b=b, metric=col, mean=float(d.mean())))
    pd.DataFrame(grid).to_csv(TAB / "table_grid.csv", index=False)
    for se in C.SEASONS:
        s = df[df.season == se]
        for m in ("BST", "NSGA-C", "DSS-WAEOP", "T-FSS"):
            M.set(f"{CODE[m]}Cov{se.replace('_', '').capitalize()}",
                  fmt(s[s.method == m]["cov"].mean(), 1))
    for fs in C.FIELD_SIZES_HA:
        s = df[df.field_ha == fs]
        for m in ("BST", "NSGA-C", "DSS-WAEOP"):
            M.set(f"{CODE[m]}CovHa{_num(fs)}", fmt(s[s.method == m]["cov"].mean(), 1))
            M.set(f"{CODE[m]}EHa{_num(fs)}", fmt(s[s.method == m]["E"].mean(), 0))
        d = paired(s, "DSS-WAEOP", "BST", "E", rel=True)
        M.set(f"DssEVsBstHa{_num(fs)}", fmt(d.mean(), 1))
    return t


def env_summary(env: pd.DataFrame, M: Macros):
    for c in ("frac_reg_gate", "frac_dss_adverse", "frac_all_adverse", "frac_fav_zone"):
        name = "".join(w.capitalize() for w in c.split("_"))
        M.set(name, fmt(100 * env[c].mean(), 1))
    M.set("ActiveMean", fmt(env["active_mean"].mean(), 2))
    M.set("MeanWind", fmt(env["mean_wind"].mean(), 2))
    env.to_csv(TAB / "table_env.csv", index=False)


def ablation(df_main: pd.DataFrame, M: Macros) -> pd.DataFrame:
    abl = _frame(_rows("abl_"))
    if abl.empty:
        return abl
    idx = sorted(set(abl.idx))
    base = df_main[df_main.idx.isin(idx) & df_main.method.isin(["DSS-WAEOP", "A2", "BST",
                                                                  "NSGA-C", "T-FSS"])]
    df = pd.concat([base, abl], ignore_index=True)
    rows = []
    for m in ["DSS-WAEOP", "A1", "A2", "A3", "A4", "NSGA-C", "T-FSS", "BST"]:
        s = df[df.method == m]
        r = dict(method=m, n=len(s))
        for col in ("E", "cov", "prec", "drift", "shift", "dur"):
            r[col] = float(s[col].mean())
            lo, hi = boot_ci(s[col])
            r[col + "_lo"], r[col + "_hi"] = lo, hi
        if m != "DSS-WAEOP":
            for col in ("E", "cov", "prec", "drift"):
                d = paired(df, m, "DSS-WAEOP", col, rel=(col == "E"))
                p, _ = wilcoxon(d)
                r["d_" + col] = float(d.mean())
                r["p_" + col] = p
        rows.append(r)
    t = pd.DataFrame(rows)
    # Holm across the 4 ablations within each metric
    for col in ("E", "cov", "prec", "drift"):
        sub = t[t.method.isin(["A1", "A2", "A3", "A4"])]
        adj = holm(sub["p_" + col].tolist())
        t.loc[sub.index, "ph_" + col] = adj
    t.to_csv(TAB / "table_ablation.csv", index=False)
    M.set("AblN", str(len(idx)))
    d = paired(df, "A4", "T-FSS", "cov")
    M.set("AblAfourCovVsTfss", fmt(d.mean(), 1))
    for _, r in t.iterrows():
        c = CODE[r["method"]]
        M.set(f"Abl{c}E", fmt(r["E"], 1))
        M.set(f"Abl{c}Cov", fmt(r["cov"], 1))
        M.set(f"Abl{c}Prec", fmt(r["prec"], 1))
        M.set(f"Abl{c}Drift", fmt(r["drift"], 2))
        if r["method"] in ("A1", "A2", "A3", "A4"):
            for col in ("E", "cov", "prec", "drift"):
                M.set(f"Abl{c}D{col.capitalize()}", fmt(r["d_" + col], 1 if col != "drift" else 2))
                M.set(f"Abl{c}D{col.capitalize()}Abs", fmt(abs(r["d_" + col]), 1 if col != "drift" else 2))
                M.setp(f"Abl{c}P{col.capitalize()}", r["ph_" + col])

    desc = {"DSS-WAEOP": "Full method", "A1": r"$A_t=\{e_1\}$ (wind speed only)",
            "A2": "No lane skipping or replanning", "A3": "No DSS-biased mutation",
            "A4": r"$A_t=E$ (no dynamic activation)"}
    L = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Variant & Description & $f_1$ (\si{Wh.ha^{-1}}) & $\Delta f_1$ (\%) & "
         r"$\mathcal{C}$ (\%) & $\mathcal{P}$ (\%) & $\mathcal{O}$ (\%) & $p_{\mathrm{Holm}}$ ($f_1$ / $\mathcal{C}$) \\",
         r"\midrule"]
    for m in ["DSS-WAEOP", "A1", "A2", "A3", "A4"]:
        r = t[t.method == m].iloc[0]
        dE = "--" if m == "DSS-WAEOP" else fmt(r["d_E"], 1)
        pp = "--" if m == "DSS-WAEOP" else f"{fmt_p(r['ph_E'])} / {fmt_p(r['ph_cov'])}"
        L.append(f"{m} & {desc[m]} & {fmt(r['E'], 1)} & {dE} & {fmt(r['cov'], 1)} & "
                 f"{fmt(r['prec'], 1)} & {fmt(r['drift'], 2)} & {pp} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "tab_ablation.tex").write_text(encoding="utf-8", data=_minus("\n".join(L) + "\n"))
    return t


def sensitivity(df_main: pd.DataFrame, M: Macros) -> pd.DataFrame:
    sens = _frame(_rows("sens_"))
    if sens.empty:
        return sens
    idx = sorted(set(sens.idx))
    M.set("SensN", str(len(idx)))
    base = df_main[df_main.idx.isin(idx)].copy()
    base["setting"] = "default"
    rows = []
    groups = {
        "sigma": [("sig_off", "never"), ("sig070", "0.70"), ("default", "0.75"),
                  ("sig080", "0.80"), ("sig085", "0.85")],
        "thr": [("thr080", "0.8"), ("default", "1.0"), ("thr120", "1.2")],
        "noise": [("noise0", r"$\times 0$"), ("default", r"$\times 1$"), ("noise2", r"$\times 2$")],
        "indep": [("default", "correlated"), ("indep", "independent")],
        "meander": [("mea0", "0"), ("default", "0.5"), ("mea1", "1")],
    }
    for g, levels in groups.items():
        for key, lab in levels:
            src = base if key == "default" else sens[sens.setting == key]
            for m in ("BST", "NSGA-C", "DSS-WAEOP"):
                s = src[src.method == m]
                if s.empty:
                    continue
                r = dict(group=g, setting=key, label=lab, method=m, n=len(s))
                for col in ("E", "cov", "prec", "drift", "triggers"):
                    r[col] = float(s[col].mean())
                if m == "DSS-WAEOP":
                    r["replans"] = float(s["triggers"].mean())
                rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "table_sensitivity.csv", index=False)

    def get(g, key, m, col):
        s = t[(t.group == g) & (t.setting == key) & (t.method == m)]
        return float(s[col].iloc[0]) if len(s) else float("nan")

    for key, nm in (("sig_off", "SigOff"), ("sig070", "SigSeventy"), ("default", "SigDefault"),
                    ("sig080", "SigEighty"), ("sig085", "SigEightyfive")):
        for col in ("E", "cov", "prec", "drift", "triggers"):
            M.set(f"Sens{nm}{col.capitalize()}", fmt(get("sigma", key, "DSS-WAEOP", col),
                                                     2 if col in ("drift", "triggers") else 1))
    for g, keys in (("thr", (("thr080", "ThrLow"), ("thr120", "ThrHigh"))),
                    ("noise", (("noise0", "NoiseZero"), ("noise2", "NoiseTwo"))),
                    ("meander", (("mea0", "MeaZero"), ("mea1", "MeaOne"))),
                    ("indep", (("indep", "Indep"),))):
        for key, nm in keys:
            for m in ("BST", "NSGA-C", "DSS-WAEOP"):
                for col in ("E", "cov", "prec", "drift"):
                    v = get(g, key, m, col)
                    if math.isfinite(v):
                        M.set(f"Sens{nm}{CODE[m]}{col.capitalize()}",
                              fmt(v, 2 if col == "drift" else 1))
    for m in ("BST", "NSGA-C", "DSS-WAEOP"):
        for col in ("E", "cov", "prec", "drift"):
            M.set(f"SensDef{CODE[m]}{col.capitalize()}",
                  fmt(get("sigma" if m == "DSS-WAEOP" else "indep", "default", m, col),
                      2 if col == "drift" else 1))

    # LaTeX: compact sensitivity table (DSS-WAEOP; for model-structure checks also baselines)
    L = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Factor & Setting & Method & $f_1$ (\si{Wh.ha^{-1}}) & $\mathcal{C}$ (\%) & "
         r"$\mathcal{P}$ (\%) & $\mathcal{O}$ (\%) & Replans \\", r"\midrule"]
    names = {"sigma": r"$\sigma_{\min}$", "thr": "Threshold scale", "noise": "Sensor noise",
             "indep": "OU driving noise", "meander": r"Meander fraction $\omega$"}
    for g, levels in groups.items():
        first_g = True
        for key, lab in levels:
            ms = ["DSS-WAEOP"] if g in ("sigma", "thr", "noise") else ["BST", "NSGA-C", "DSS-WAEOP"]
            first_l = True
            for m in ms:
                s = t[(t.group == g) & (t.setting == key) & (t.method == m)]
                if s.empty:
                    continue
                r = s.iloc[0]
                rep = fmt(r["triggers"], 2) if m == "DSS-WAEOP" else "--"
                L.append(f"{names[g] if first_g else ''} & {lab if first_l else ''} & {m} & "
                         f"{fmt(r['E'], 1)} & {fmt(r['cov'], 1)} & {fmt(r['prec'], 1)} & "
                         f"{fmt(r['drift'], 2)} & {rep} \\\\")
                first_g = first_l = False
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    (TEX / "tab_sensitivity.tex").write_text(encoding="utf-8", data=_minus("\n".join(L) + "\n"))
    return t


def seed_variability(df_main: pd.DataFrame, M: Macros) -> pd.DataFrame:
    sd = _frame(_rows("seed_"))
    if sd.empty:
        return sd
    idx = sorted(set(sd.idx))
    base = df_main[df_main.idx.isin(idx) & df_main.method.isin(["NSGA-C", "DSS-WAEOP"])].copy()
    base["opt_seed"] = 0
    df = pd.concat([base, sd], ignore_index=True)
    rows = []
    for (i, m), s in df.groupby(["idx", "method"]):
        rows.append(dict(idx=i, method=m, n=len(s), E_mean=s.E.mean(), E_cv=100 * s.E.std() / s.E.mean(),
                         cov_sd=s["cov"].std(), prec_sd=s["prec"].std()))
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "table_seed.csv", index=False)
    # between-seed SD of the paired DSS - NSGA-C relative energy difference
    piv = df.pivot_table(index=["idx", "opt_seed"], columns="method", values="E")
    rel = 100 * (piv["DSS-WAEOP"] - piv["NSGA-C"]) / piv["NSGA-C"]
    within = rel.groupby(level=0).std().mean()
    between = rel.groupby(level=0).mean().std()
    M.set("SeedN", str(len(idx)))
    M.set("SeedCvDss", fmt(t[t.method == "DSS-WAEOP"].E_cv.mean(), 1))
    M.set("SeedCvNsgac", fmt(t[t.method == "NSGA-C"].E_cv.mean(), 1))
    M.set("SeedCovSdDss", fmt(t[t.method == "DSS-WAEOP"].cov_sd.mean(), 1))
    M.set("SeedPrecSdDss", fmt(t[t.method == "DSS-WAEOP"].prec_sd.mean(), 1))
    M.set("SeedRelWithin", fmt(within, 1))
    M.set("SeedRelBetween", fmt(between, 1))
    return t


def convergence(M: Macros) -> pd.DataFrame:
    recs = []
    for p in sorted(RAW.glob("conv_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for m, v in d["conv"].items():
            for g, hv in enumerate(v["hv"]):
                recs.append(dict(idx=d["scenario"]["idx"], method=m, gen=g, hv=hv))
    t = pd.DataFrame(recs)
    if t.empty:
        return t
    t.to_csv(TAB / "table_convergence.csv", index=False)
    last = t[t.gen == t.gen.max()].pivot_table(index="idx", columns="method", values="hv")
    g24 = t[t.gen == C.OPT.gens].pivot_table(index="idx", columns="method", values="hv")
    M.set("ConvN", str(len(last)))
    M.set("ConvGensMax", str(int(t.gen.max())))
    rel_last = 100 * (last["DSS-WAEOP"] - last["A3"]) / last["A3"]
    rel24 = 100 * (g24["DSS-WAEOP"] - g24["A3"]) / g24["A3"]
    M.set("ConvHvGainFinal", fmt(rel_last.mean(), 2))
    M.set("ConvHvGainDefault", fmt(rel24.mean(), 2))
    # generations to reach 99 % of own final HV
    reach = []
    for (i, m), s in t.groupby(["idx", "method"]):
        s = s.sort_values("gen")
        fin = s.hv.iloc[-1]
        g = int(s[s.hv >= 0.99 * fin].gen.iloc[0])
        reach.append(dict(idx=i, method=m, g99=g))
    r = pd.DataFrame(reach)
    for m in ("DSS-WAEOP", "A3"):
        M.set(f"ConvGninetynine{CODE[m]}", fmt(r[r.method == m].g99.mean(), 1))
    return t


def compute_perf(df: pd.DataFrame, M: Macros) -> pd.DataFrame:
    rows = []
    for (fs, m), s in df[df.method.isin(["EA-CPP", "NSGA-C", "T-FSS", "DSS-WAEOP"])].groupby(
            ["field_ha", "method"]):
        rs = [x for l in s.get("replan_s", pd.Series(dtype=object)).dropna() for x in l]
        rows.append(dict(field_ha=fs, method=m, plan_s=s.plan_s.mean(), plan_max=s.plan_s.max(),
                         n_eval=s.n_eval.mean(), replans=s.triggers.mean(),
                         replan_s=np.mean(rs) if rs else float("nan"),
                         replan_max=np.max(rs) if rs else float("nan")))
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "table_compute.csv", index=False)
    d = t[t.method == "DSS-WAEOP"]
    for _, r in d.iterrows():
        M.set(f"PlanS{_num(r.field_ha)}", fmt(r.plan_s, 1))
        M.set(f"ReplanS{_num(r.field_ha)}", fmt(r.replan_s, 1))
    allr = [x for l in df[df.method == "DSS-WAEOP"].get("replan_s", pd.Series(dtype=object)).dropna()
            for x in l]
    M.set("ReplanMeanS", fmt(np.mean(allr), 1) if allr else "--")
    M.set("ReplanMaxS", fmt(np.max(allr), 1) if allr else "--")
    M.set("ReplanCount", fmt(df[df.method == "DSS-WAEOP"].triggers.mean(), 2))
    M.set("ReplanFracScen", fmt(100 * (df[df.method == "DSS-WAEOP"].triggers > 0).mean(), 0))
    M.set("NEval", str(int(df[df.method == "DSS-WAEOP"].n_eval.median())))

    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{4}{c}{Planning time (\si{s})} & \multicolumn{2}{c}{DSS-WAEOP replanning} \\",
         r"\cmidrule(lr){2-5}\cmidrule(l){6-7}",
         r"Field (\si{ha}) & EA-CPP & NSGA-C & T-FSS & DSS-WAEOP & per mission & time, mean / max (\si{s}) \\", r"\midrule"]
    for fs in C.FIELD_SIZES_HA:
        g = t[t.field_ha == fs].set_index("method")
        if not {"EA-CPP", "NSGA-C", "T-FSS", "DSS-WAEOP"} <= set(g.index):
            continue
        L.append(f"{fs} & {fmt(g.loc['EA-CPP', 'plan_s'], 1)} & {fmt(g.loc['NSGA-C', 'plan_s'], 1)} & "
                 f"{fmt(g.loc['T-FSS', 'plan_s'], 1)} & {fmt(g.loc['DSS-WAEOP', 'plan_s'], 1)} & "
                 f"{fmt(g.loc['DSS-WAEOP', 'replans'], 2)} & {fmt(g.loc['DSS-WAEOP', 'replan_s'], 1)} / "
                 f"{fmt(g.loc['DSS-WAEOP', 'replan_max'], 1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "tab_compute.tex").write_text(encoding="utf-8", data=_minus("\n".join(L) + "\n"))
    return t


def energy_battery(df: pd.DataFrame, M: Macros):
    s = df[df.method.isin(MAIN_METHODS)]
    M.set("MaxSortieWh", fmt(s.max_sortie_Wh.max(), 0))
    M.set("MeanSortieWh", fmt(s.mean_sortie_Wh.mean(), 0))
    M.set("BatteryViolations", str(int(s.battery_violations.sum())))
    M.set("UsableWh", fmt(C.PLATFORM.usable_Wh, 1))
    M.set("BatteryWh", fmt(C.PLATFORM.battery_Wh, 1))
    M.set("MaxSortieFrac", fmt(100 * s.max_sortie_Wh.max() / C.PLATFORM.battery_Wh, 0))
    b = s[(s.method == "BST") & (s.field_ha == 40)]
    M.set("BstSortiesFortyHa", fmt(b.sorties.mean(), 0))
    for m in MAIN_METHODS:
        t = s[s.method == m]
        tot = t[["e_spray", "e_ferry", "e_turn", "e_tol"]].sum(axis=1)
        M.set(f"{CODE[m]}FerryShare", fmt(100 * (t.e_ferry / tot).mean(), 1))
        M.set(f"{CODE[m]}SpraySpeed", fmt(t.mean_speed.mean(), 2))
    rows = []
    for m in MAIN_METHODS:
        t = s[s.method == m]
        rows.append(dict(method=m, **{c: float((t[c] / t.treated_ha).mean()) for c in
                                      ("e_spray", "e_ferry", "e_turn", "e_tol")},
                         mean_speed=float(t.mean_speed.mean()),
                         sorties10=float((t.sorties / t.field_ha * 10).mean())))
    pd.DataFrame(rows).to_csv(TAB / "table_energy_breakdown.csv", index=False)


def proposition_check(M: Macros):
    """Empirical fraction of cells within one sensor-noise SD of a class boundary,
    over the operating window of the sensitivity scenarios, vs the uniform-marginal
    bound of Proposition 3.1 (renumbered)."""
    from .atmosphere import Atmosphere
    from .dss import boundary_fraction_bound
    from .experiments import main_scenarios
    rows = []
    k0, k1 = C.step_of_hour(C.OP_START_H), C.step_of_hour(C.OP_END_H)
    for i in sens_subset():
        atm = Atmosphere(main_scenarios()[i])
        for p in C.PARAMS:
            lo, fav, adv = C.THRESHOLDS[p]
            v = atm.truth.fields[p][k0:k1]
            eps = C.SENSOR_NOISE[p]
            near = (np.abs(v - fav) <= eps) | (np.abs(v - adv) <= eps)
            rows.append(dict(idx=i, param=p, eps=eps, empirical=float(near.mean()),
                             bound=boundary_fraction_bound(eps, p)))
    t = pd.DataFrame(rows).groupby("param").agg(eps=("eps", "first"), empirical=("empirical", "mean"),
                                                 bound=("bound", "first")).reset_index()
    t.to_csv(TAB / "table_boundary.csv", index=False)
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Attribute & $\bar\eta_k$ (sensor noise s.d.) & Uniform-marginal value, Eq.~\eqref{eq:boundary} (\%) "
         r"& Simulated (\%) \\", r"\midrule"]
    for _, r in t.iterrows():
        L.append(f"$e_{r.param[1]}$ & {r.eps:g}{TEX_UNIT[r.param]} & "
                 f"{fmt(100 * r.bound, 1)} & {fmt(100 * r.empirical, 1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "tab_boundary.tex").write_text(encoding="utf-8", data=_minus("\n".join(L) + "\n"))
    M.set("BoundaryMax", fmt(100 * t.empirical.max(), 1))
    M.set("BoundaryMaxParam", f"$e_{t.loc[t.empirical.idxmax(), 'param'][1]}$")
    return t


def params_table(M: Macros):
    """Table of platform, spray and simulation parameters straight from config."""
    from . import physics as ph
    P, S = C.PLATFORM, C.SPRAY
    hov = float(ph.electric_power(0.0, P.mtow, False))
    hov_e = float(ph.electric_power(0.0, P.m_dry, False))
    v = np.linspace(0, 15, 151)
    pc = ph.power_curve(P.mtow, v)
    M.set("HoverPowerFull", fmt(hov, 0))
    M.set("HoverPowerEmpty", fmt(hov_e, 0))
    M.set("VminPower", fmt(v[np.argmin(pc)], 1))
    M.set("MinPowerFull", fmt(pc.min(), 0))
    M.set("ProfilePower", fmt(ph.P_PROFILE, 0))
    M.set("Mtow", fmt(P.mtow, 0))
    M.set("DiskArea", fmt(P.disk_area, 2))
    for crop, d in C.CROPS.items():
        M.set(f"Vmax{crop.capitalize()}", fmt(min(S.v_spray_max, S.q_max * 1e4 / (60 * d['rate'] * S.swath)), 2))
    rows = [
        ("Platform", r"Mass, dry / maximum take-off ($m_{\mathrm{dry}}$ / $m$)", rf"{P.m_dry:g} / {P.mtow:g}~\si{{kg}}"),
        ("", "Rotors (total disk area $A_d$)", rf"{P.n_rotor} $\times$ {P.rotor_d:g}~\si{{m}} ({P.disk_area:.2f}~\si{{m^2}})"),
        ("", r"Induced-power factor $\kappa$; hover figure of merit", f"{P.kappa_ind:g}; {P.fm_hover:g}"),
        ("", r"Motor and controller efficiency $\eta_{\mathrm{el}}$; $P_{\mathrm{av}}$; $P_{\mathrm{pump}}$", rf"{P.eta_e:g}; {P.p_avionics:g}~\si{{W}}; {P.p_pump:g}~\si{{W}}"),
        ("", r"Drag area $S_D$", rf"{P.cda:g}~\si{{m^2}}"),
        ("", r"Battery $\mathcal{E}_{\mathrm{bat}}$ (usable with 20\% reserve)", rf"{P.battery_Wh:g}~\si{{Wh}} ({P.usable_Wh:.1f}~\si{{Wh}})"),
        ("", "Ferry speed; turn time; service time", rf"{P.v_ferry:g}~\si{{m.s^{{-1}}}}; {P.t_turn:g}~\si{{s}}; {P.t_service:g}~\si{{s}}"),
        ("", "Hover electric power (full / empty tank)", rf"{hov:.0f} / {hov_e:.0f}~\si{{W}}"),
        ("Spray", r"Swath $w_{\mathrm{sw}}$; release height $h_r$; pump capacity $Q_{\max}$", rf"{S.swath:g}~\si{{m}}; {S.height:g}~\si{{m}}; {S.q_max:g}~\si{{L.min^{{-1}}}}"),
        ("", r"Spraying speed range $[v_{\min}, v_{\max}(R^*)]$", rf"[{S.v_spray_min:g}, {S.v_spray_max:g}]~\si{{m.s^{{-1}}}}, flow-limited"),
        ("", r"Calm-air spread $s_0$; downwash factor $c_{\mathrm{dw}}$", rf"{S.sigma0:g}~\si{{m}}; {S.c_downwash:g}"),
        ("", "Target rate $R^*$ (rice / wheat / orchards / vineyards)",
         " / ".join(f"{C.CROPS[c]['rate']:g}" for c in C.CROP_LIST) + r"~\si{L.ha^{-1}}"),
        ("", "Coverage / precision criteria", f"$D\\geq{C.COVERAGE_MIN:g}R^*$ / ${C.PRECISION_BAND[0]:g}R^*\\leq D\\leq{C.PRECISION_BAND[1]:g}R^*$"),
        ("Simulation", "Atmospheric grid; time step; operating day", rf"{C.WIND_CELL:g}~\si{{m}}; {C.DT_MIN:g}~\si{{min}}; {C.OP_START_H:g}:00--{int(C.OP_END_H)}:30"),
        ("", "Dose grid (along $\\times$ across)", rf"{C.DOSE_ALONG:g}~\si{{m}} $\times$ {C.CROSS_BIN:g}~\si{{m}}"),
        ("", "Lane headings; blocks $B$; lane piece length", rf"{len(C.HEADINGS_DEG)} (\SI{{15}}{{\degree}} steps); {C.N_BLOCKS}; 100~\si{{m}}"),
        ("NSGA-II", "Population; generations; crossover probability; mutation probability (offspring / gene); $p_{\\mathrm{bias}}$",
         f"{C.OPT.pop}; {C.OPT.gens}; {C.OPT.p_cx:g}; {2 * C.OPT.p_mut:g} / {max(1 / 19, C.OPT.p_mut / 2):.3g}; {C.OPT.p_bias:g}"),
        ("", "Replanning: $\\sigma_{\\min}$; window; population / generations; minimum gap",
         rf"{C.OPT.sigma_min:g}; {C.OPT.sigma_window * C.DT_MIN:g}~\si{{min}}; {C.OPT.replan_pop} / {C.OPT.replan_gens}; {C.OPT.replan_min_gap_h * 60:g}~\si{{min}}"),
    ]
    L = [r"\begin{tabularx}{\textwidth}{l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{4.6cm}}", r"\toprule", r"Group & Quantity & Value \\", r"\midrule"]
    for g, q, v in rows:
        if g and L[-1] != r"\midrule":
            L.append(r"\midrule")
        L.append(f"{g} & {q} & {v} \\\\")
    L += [r"\bottomrule", r"\end{tabularx}"]
    (TEX / "tab_params.tex").write_text(encoding="utf-8", data=_minus("\n".join(L) + "\n"))


def thresholds_table():
    L = [r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xlccc>{\raggedright\arraybackslash}p{3.3cm}}", r"\toprule",
         r"Attribute & Unit & Favourable ($\alpha$) & Marginal ($\beta$) & Adverse ($\gamma$) & "
         r"Active when ($A_t$ rule) \\", r"\midrule"]
    rule = {"e1": "always", "e2": "always", "e3": "always",
            "e4": r"field-mean $e_1\geq\SI{2}{m.s^{-1}}$",
            "e5": r"no surface heating ($S(h)\leq 0.1$)",
            "e6": r"surface heating ($S(h)>0.1$)"}
    units = {"e1": r"\si{m.s^{-1}}", "e2": r"\si{\degree}", "e3": r"\si{m^2.s^{-2}}", "e4": "--",
             "e5": r"\si{s^{-1}}", "e6": "--"}
    for p in C.PARAMS:
        lo, fav, adv = C.THRESHOLDS[p]
        L.append(f"$e_{p[1]}$ {C.PARAM_LABEL[p].lower().replace('0-10 m', '0--10~m')} & {units[p]} & $\\leq {fav:g}$ & "
                 f"$({fav:g}, {adv:g}]$ & $> {adv:g}$ & {rule[p]} \\\\")
    L += [r"\bottomrule", r"\end{tabularx}"]
    (TEX / "tab_thresholds.tex").write_text(encoding="utf-8", data=_minus("\n".join(L) + "\n"))


def climate_table():
    L = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Attribute & Statistic & Pre-monsoon & Monsoon & Post-monsoon & Winter \\", r"\midrule"]
    for p in C.DRIVEN:
        mu_lab = r"$\bar\mu_k$" + (f" ({TEX_UNIT[p].lstrip('~')})" if TEX_UNIT[p] else "")
        for j, (k, lab) in enumerate((("mean", mu_lab), ("amp", r"$r_k$"),
                                      ("theta", r"$\lambda_k$ (\si{h^{-1}})"),
                                      ("sigma", r"$\varsigma_k$ (unit\,\si{h^{-1/2}})"))):
            vals = " & ".join(f"{C.CLIMATE[p][s][k]:g}" for s in C.SEASONS)
            L.append(f"{'$e_' + p[1] + '$' if j == 0 else ''} & {lab} & {vals} \\\\")
        L.append(r"\midrule")
    L.append(r"Direction & prevailing, from (\si{\degree}) & " +
             " & ".join(f"{C.PREVAILING_FROM[s]:g}" for s in C.SEASONS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TEX / "tab_climate.tex").write_text(encoding="utf-8", data=_minus("\n".join(L) + "\n"))


def adverse_by_param(M: Macros):
    """Share of operating-window field-time in which each ACTIVE parameter is adverse (truth),
    over all main scenarios."""
    from .atmosphere import Atmosphere
    from .dss import DSSModel
    from .experiments import main_scenarios
    k0, k1 = C.step_of_hour(C.OP_START_H), C.step_of_hour(C.OP_END_H)
    rows = []
    for sc in main_scenarios():
        m = DSSModel(Atmosphere(sc).truth, "dynamic")
        r = {p: float(((m.cls[p][k0:k1] == 2) & m.A[k0:k1, j][:, None, None]).mean())
             for j, p in enumerate(C.PARAMS)}
        rows.append(r)
    t = pd.DataFrame(rows).mean() * 100
    t.to_csv(TAB / "table_adverse_by_param.csv")
    names = {"e1": "One", "e2": "Two", "e3": "Three", "e4": "Four", "e5": "Five", "e6": "Six"}
    for p, v in t.items():
        M.set(f"AdvParam{names[p]}", fmt(v, 1))


def illus_macros(M: Macros):
    p = RAW / "illus.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))["meta"]
    sc = d["scenario"]
    M.set("IlluHa", f"{sc['field_ha']:g}")
    M.set("IlluTerrain", sc["terrain"])
    M.set("IlluSeason", C.SEASON_LABEL[sc["season"]].lower())
    M.set("IlluCrop", sc["crop"])
    for m in ("BST", "NSGA-C", "DSS-WAEOP"):
        r = d[m]["row"]
        c = CODE[m]
        M.set(f"Illu{c}E", fmt(r["energy_per_treated_ha"], 0))
        M.set(f"Illu{c}Cov", fmt(100 * r["coverage"], 1))
        M.set(f"Illu{c}Prec", fmt(100 * r["precision"], 1))
        M.set(f"Illu{c}Drift", fmt(r["drift_pct"], 1))
        M.set(f"Illu{c}Sorties", str(r["sorties"]))
        M.set(f"Illu{c}Dur", fmt(r["duration_h"], 1))
        M.set(f"Illu{c}Heading", fmt(d[m]["heading"], 0))
        M.set(f"Illu{c}Trig", str(r.get("triggers", 0)))
    z = np.load(RAW / "illus.npz")
    h = z["hours"]
    win = (h >= C.OP_START_H) & (h <= C.OP_END_H)
    za_d = z["Za_dynamic"].reshape(len(h), -1).mean(axis=1)
    za_a = z["Za_all"].reshape(len(h), -1).mean(axis=1)
    only_fixed = win & (za_a > za_d + 0.5)
    M.set("IlluFixedOnlyH", fmt(only_fixed.sum() * C.DT_MIN / 60, 1))
    M.set("IlluZaDyn", fmt(100 * za_d[win].mean(), 0))
    M.set("IlluZaAll", fmt(100 * za_a[win].mean(), 0))
    for m in ("NSGA-C", "DSS-WAEOP"):
        F, Ft = z[f"{m}_front"], z[f"{m}_front_truth"]
        M.set(f"Illu{CODE[m]}FrontN", str(len(F)))
        M.set(f"Illu{CODE[m]}PredBias", fmt(100 * np.mean(Ft[:, 0] / F[:, 0] - 1), 1))
        M.set(f"Illu{CODE[m]}PrecPredMean", fmt(-100 * F[:, 1].mean(), 1))
        M.set(f"Illu{CODE[m]}PrecTruthMean", fmt(-100 * Ft[:, 1].mean(), 1))


def run():
    TEX.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    M = Macros()
    df = load_main()
    df = df[df.method.isin(MAIN_METHODS + ["A2"])]
    n_sc = df.idx.nunique()
    M.set("NScen", str(n_sc))
    M.set("NFactorial", "192")
    params_table(M)
    thresholds_table()
    climate_table()
    main_table(df[df.method.isin(MAIN_METHODS)], M)
    paired_tests(df, M)
    stratified(df, M)
    env_summary(load_env(), M)
    ablation(df, M)
    sensitivity(df, M)
    seed_variability(df, M)
    convergence(M)
    compute_perf(df, M)
    energy_battery(df, M)
    proposition_check(M)
    adverse_by_param(M)
    illus_macros(M)
    M.write(TEX / "macros.tex")
    print(f"analysis: {n_sc} scenarios, {len(M.d)} macros -> {TEX}")
    return M


if __name__ == "__main__":
    run()
