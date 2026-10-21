"""
All manuscript figures, generated from results/raw (and, for model illustrations,
directly from the model code).  Output: results/figures/figNN_*.png (600 dpi).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

if __package__ in (None, ""):  # also allow `python dsswaeop/<module>.py` (e.g. the IDE "Run" button)
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "dsswaeop"

from . import config as C
from . import physics as ph
from .analysis import (MAIN_METHODS, TAB, load_env, load_main, paired, boot_ci, _frame, _rows)
from .experiments import RAW, ROOT

FIG = ROOT / "figures"

# ── style (validated reference palette; see README) ─────────────────────────
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
METHOD_COLOR = {"BST": "#a8a69f", "WA-BST": "#a8a69f", "EA-CPP": "#a8a69f", "GRD": "#a8a69f",
                "NSGA-C": ORANGE, "T-FSS": VIOLET, "DSS-WAEOP": BLUE,
                "A1": "#86b6ef", "A2": "#86b6ef", "A3": "#86b6ef", "A4": "#86b6ef"}
W_FULL = 7.0
DIV = mcolors.LinearSegmentedColormap.from_list(
    "div", ["#b2182b", "#e34948", "#f4a9a3", "#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])
SEQ = mcolors.LinearSegmentedColormap.from_list("seq", ["#f0efec"] + SEQ_BLUE)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "savefig.dpi": 300,
    "figure.dpi": 150, "lines.linewidth": 1.6, "pdf.fonttype": 42,
    "axes.unicode_minus": True,
})


def _latex_preamble():
    """Same fonts as the manuscript class (cas-sc): STIX if installed, otherwise Latin Modern."""
    stix = False
    if shutil.which("kpsewhich"):
        try:
            stix = subprocess.run(["kpsewhich", "stix.sty"], capture_output=True, text=True,
                                  timeout=30).stdout.strip() != ""
        except Exception:
            stix = False
    fonts = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{stix}" if stix else \
        r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}"
    return r"\usepackage[T1]{fontenc}" + fonts + r"\usepackage{siunitx}\usepackage{bm}"


def _latex_works(preamble):
    """True only if a complete LaTeX tool chain with the required packages is available: a test
    label is actually typeset.  Any failure (no LaTeX, missing package, MiKTeX asking to install
    a package, ...) selects the Matplotlib mathtext fallback instead of failing every figure."""
    if os.environ.get("DSS_NO_USETEX") == "1" or shutil.which("latex") is None:
        return False
    if shutil.which("pdftoppm") is None and shutil.which("dvipng") is None:
        return False   # a PNG cannot be produced from LaTeX text without one of these tools
    import io
    try:
        with plt.rc_context({"text.usetex": True, "text.latex.preamble": preamble}):
            f = plt.figure(figsize=(1, 1))
            f.text(0.5, 0.5, r"$\mathcal{C}$ \SI{5}{m.s^{-1}} \textbf{x}")
            f.savefig(io.BytesIO(), format="pdf")
            plt.close(f)
        return True
    except Exception:
        plt.close("all")
        return False


_PREAMBLE = _latex_preamble() if shutil.which("latex") else ""
USETEX = _latex_works(_PREAMBLE)
if USETEX:
    # all labels are typeset by LaTeX with the manuscript fonts
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": _PREAMBLE})
else:  # fallback without a LaTeX installation: mathtext with Computer Modern
    plt.rcParams.update({"mathtext.fontset": "cm", "font.serif": ["DejaVu Serif"]})
    import matplotlib.text as _mtext
    _orig_set_text = _mtext.Text.set_text

    def _delatex(t):
        if not isinstance(t, str):
            return t
        t = re.sub(r"\\SI\{([^}]*)\}\{([^}]*)\}", lambda m: m.group(1) + " " + _unit(m.group(2)), t)
        t = re.sub(r"\\si\{([^}]*)\}", lambda m: _unit(m.group(1)), t)
        t = t.replace(r"\textbf", "")
        t = re.sub(r"\\le(?![a-z])", r"\\leq", t)
        t = re.sub(r"\\ge(?![a-z])", r"\\geq", t)
        t = t.replace(r"\big[", "[").replace(r"\big]", "]").replace(r"\!", "")
        parts = t.split("$")
        for k in range(0, len(parts), 2):          # text outside math mode
            parts[k] = parts[k].replace(r"\%", "%").replace("--", "\u2013").replace(r"\,", " ")
        return "$".join(parts)

    def _unit(u):
        u = u.replace(r"\degree", r"$^\circ$").replace(r"\micro", r"$\mu$")
        u = re.sub(r"\^\{([^}]*)\}", r"$^{\1}$", u)
        return u.replace(".", " ")

    def _set_text(self, s):
        return _orig_set_text(self, _delatex(s))
    _mtext.Text.set_text = _set_text


def B(s):
    """Bold text that also works with usetex."""
    return r"\textbf{" + s + "}" if USETEX else s


def _save(fig, name, dpi=600):
    """Write results/figures/<name>.png only.  With LaTeX text and poppler available, the figure is
    first written as a temporary vector PDF and rasterised with pdftoppm (exact LaTeX typesetting);
    otherwise Matplotlib writes the PNG directly."""
    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / f"{name}.png"
    if USETEX and shutil.which("pdftoppm"):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / f"{name}.pdf"
            fig.savefig(pdf, bbox_inches="tight")
            subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(pdf), str(png.with_suffix(""))],
                           check=True)
    else:
        fig.savefig(png, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


# ── icons (Material Design Icons, Apache-2.0; see assets/icons/LICENSE) ─────
ICON_DIR = Path(__file__).resolve().parent / "assets" / "icons"


_ICONS = None


def _icon_path(name):
    """Icon outline as a Matplotlib path in a unit box centred at the origin.  The SVG outlines of
    assets/icons/*.svg are stored pre-converted in assets/icons/icons.json, so no SVG parser is
    needed at run time."""
    global _ICONS
    from matplotlib.path import Path as MPath
    if _ICONS is None:
        _ICONS = json.loads((ICON_DIR / "icons.json").read_text(encoding="utf-8"))
    d = _ICONS[name]
    v = np.array(d["v"], dtype=float)
    v[:, 1] = 24 - v[:, 1]          # SVG y-axis points down
    return MPath((v - 12) / 24, np.array(d["c"], dtype=np.uint8))


def icon(ax, name, x, y, size, color=INK, badge=None, badge_ec=None, zorder=6, lw_badge=0.9, bfac=0.78):
    """Draw icon `name` centred at data coords (x, y) with width/height `size` (data units).
    Optional circular badge behind it.  Assumes an equal-aspect axis."""
    from matplotlib.patches import Circle, PathPatch
    from matplotlib.transforms import Affine2D
    if badge is not None:
        ax.add_patch(Circle((x, y), bfac * size, fc=badge, ec=badge_ec or "none", lw=lw_badge,
                            zorder=zorder - 1))
    p = _icon_path(name)
    tr = Affine2D().scale(size).translate(x, y) + ax.transData
    ax.add_patch(PathPatch(p, fc=color, ec="none", transform=tr, zorder=zorder))


def _panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, B(f"({letter})"), transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=INK, va="bottom")


if USETEX:
    import matplotlib.axes as _maxes
    _orig_title = _maxes.Axes.set_title

    def _bold_title(self, label, *a, **k):
        if k.get("fontweight", "bold") != "normal" and label and not label.startswith(r"\textbf"):
            label = "\n".join(B(x) if x else x for x in label.split("\n"))
        return _orig_title(self, label, *a, **k)
    _maxes.Axes.set_title = _bold_title


# ═════════════════════════════════════════════════════════════════════════════
# Schematic helpers (overview and architecture figures)
# ═════════════════════════════════════════════════════════════════════════════
PAL = dict(input="#5f6b7a", dss="#1b6fc6", plan="#7a4fc9", exec="#0f8f6a", replan="#d9822b",
           out="#3a3a3a", field="#6fa84a", soil="#e8dcc4")


def _macros():
    """Numbers quoted in the paper (written by analysis.py)."""
    f = ROOT / "tex" / "macros.tex"
    out = {}
    if f.exists():
        for m in re.finditer(r"\\newcommand\{\\R(\w+)\}\{([^}]*(?:\{,\}[^}]*)*)\}", f.read_text(encoding="utf-8")):
            out[m.group(1)] = m.group(2)
    return out


def _canvas(w_in, h_in, W, H):
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def _card(ax, x, y, w, h, title, color, icon_name=None, subtitle=None, fc="#ffffff", head=6.2,
          tsize=7.4):
    """Rounded card with a coloured header strip, a white icon and a title."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=fc, ec=color, lw=1.0, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - head), w, head, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=color, ec=color, lw=1.0, zorder=3))
    ax.add_patch(Rectangle((x, y + h - head), w, head / 2, fc=color, ec="none", zorder=3))
    tx = x + 2.0
    if icon_name:
        icon(ax, icon_name, x + 0.62 * head, y + h - head / 2, head * 0.84, color="white", zorder=5)
        tx = x + 1.22 * head
    ax.text(tx, y + h - head / 2, B(title), color="white", fontsize=tsize, va="center", ha="left",
            zorder=5)
    if subtitle:
        ax.text(x + w / 2, y + h - head - 2.2, subtitle, color=color, fontsize=6.3, va="center",
                ha="center", zorder=5)


def _item(ax, x, y, name, text, color, size=3.4, fs=6.2, tcolor=INK, badge=True, glyph=None, bfac=0.78):
    """Icon bullet followed by text (one or two lines).  `glyph` (if given) sets the icon size
    directly; the badge radius is then bfac * glyph and the text starts just right of the badge."""
    if glyph is not None:
        icon(ax, name, x, y, glyph, color="white", badge=color, zorder=6, bfac=bfac)
        ax.text(x + bfac * glyph + 1.6, y, text, fontsize=fs, va="center", ha="left", color=tcolor, zorder=6,
                linespacing=1.25)
        return
    if badge:
        icon(ax, name, x, y, size * 0.66, color="white", badge=color, zorder=6)
    else:
        icon(ax, name, x, y, size, color=color, zorder=6)
    ax.text(x + size * 0.85, y, text, fontsize=fs, va="center", ha="left", color=tcolor, zorder=6,
            linespacing=1.25)


def _arrow(ax, p, q, color=INK2, rad=0.0, lw=1.2, ls="-", ms=9, z=4, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z,
                                 shrinkA=0, shrinkB=0))


def _chip(ax, x, y, w, h, text, color, fs=5.9, fc=None, z=5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                fc=fc or mcolors.to_rgba(color, 0.10), ec=color, lw=0.7, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, ha="center", va="center", color=INK, zorder=z + 1,
            linespacing=1.15)


def _hexacopter(ax, cx, cy, s=1.0, spray=True, z=8):
    """Side view of a six-rotor spraying UAV with a tank and a droplet fan (drawn, not an icon)."""
    from matplotlib.patches import Ellipse, Polygon as Poly
    body, arm, rotor = "#2f3640", "#4b5563", "#9aa5b1"
    # spray fans (behind the airframe)
    if spray:
        for dx in (-6.5, 6.5):
            ax.add_patch(Poly([(cx + dx * s, cy - 2.6 * s), (cx + (dx - 3.4) * s, cy - 11 * s),
                               (cx + (dx + 3.4) * s, cy - 11 * s)], closed=True,
                              fc=mcolors.to_rgba("#4ea3e6", 0.22), ec="none", zorder=z - 2))
            rng = np.random.default_rng(int(10 + dx))
            px = cx + (dx + rng.uniform(-3, 3, 26)) * s
            py = cy - rng.uniform(3.2, 10.5, 26) * s
            frac = (cy - 2.6 * s - py) / (8.4 * s)
            ok = np.abs(px - (cx + dx * s)) <= (0.3 + 3.1 * frac) * s
            ax.scatter(px[ok], py[ok], s=1.6 * s, color="#2a78d6", lw=0, zorder=z - 1)
    # arms and rotors
    for dx in (-9.5, -5.0, 5.0, 9.5):
        ax.plot([cx, cx + dx * s], [cy + 0.6 * s, cy + 1.4 * s], color=arm, lw=1.5 * s, zorder=z,
                solid_capstyle="round")
        ax.plot([cx + dx * s] * 2, [cy + 1.4 * s, cy + 2.4 * s], color=body, lw=1.6 * s, zorder=z)
        ax.add_patch(Ellipse((cx + dx * s, cy + 2.6 * s), 6.2 * s, 0.9 * s, fc=mcolors.to_rgba(rotor, 0.55),
                             ec=rotor, lw=0.5, zorder=z + 1))
    # body, tank, legs, boom with nozzles
    ax.add_patch(FancyBboxPatch((cx - 3.4 * s, cy - 0.4 * s), 6.8 * s, 2.4 * s,
                                boxstyle=f"round,pad=0,rounding_size={0.9 * s}", fc=body, ec="none", zorder=z + 1))
    ax.add_patch(FancyBboxPatch((cx - 2.4 * s, cy - 2.5 * s), 4.8 * s, 2.2 * s,
                                boxstyle=f"round,pad=0,rounding_size={0.8 * s}", fc="#f2f2f2", ec=body, lw=0.6,
                                zorder=z + 1))
    ax.add_patch(Rectangle((cx - 2.4 * s, cy - 2.5 * s), 4.8 * s, 1.0 * s, fc="#7cc0f0", ec="none", zorder=z + 1))
    for dx in (-3.0, 3.0):
        ax.plot([cx + dx * s * 0.6, cx + dx * s], [cy - 0.4 * s, cy - 3.6 * s], color=body, lw=1.0 * s, zorder=z)
    ax.plot([cx - 4.2 * s, cx + 4.2 * s], [cy - 3.6 * s, cy - 3.6 * s], color=body, lw=1.0 * s, zorder=z)
    ax.plot([cx - 7.5 * s, cx + 7.5 * s], [cy - 2.6 * s, cy - 2.6 * s], color="#6b7280", lw=0.9 * s, zorder=z)
    for dx in (-6.5, 6.5):
        ax.plot(cx + dx * s, cy - 2.75 * s, marker="v", ms=2.6 * s, color=body, zorder=z + 1)


def _crop_field(ax, x0, y0, w, h, n=7, crop=True):
    """Simple perspective crop field with rows of plants."""
    from matplotlib.patches import Polygon as Poly
    sk = 0.22 * w
    ax.add_patch(Poly([(x0, y0), (x0 + w, y0), (x0 + w - sk, y0 + h), (x0 + sk, y0 + h)], closed=True,
                      fc="#efe4cc", ec="none", zorder=1))
    for i in range(n):
        f = (i + 0.5) / n
        xa, xb = x0 + f * w, x0 + sk + f * (w - 2 * sk)
        ax.plot([xa, xb], [y0, y0 + h], color="#8db86b", lw=2.2, zorder=1.5, solid_capstyle="round")
        if crop:
            for t in np.linspace(0.05, 0.95, 6):
                px, py = xa + t * (xb - xa), y0 + t * h
                r = 0.75 * (1 - 0.45 * t)
                ax.add_patch(plt.Circle((px, py), r, fc="#4f9a3a", ec="none", zorder=2))


def _wind(ax, x0, y0, L, n=3, gap=2.4, color="#5b8fd6"):
    for k in range(n):
        xs = np.linspace(x0 + 0.8 * k, x0 + L + 0.8 * k, 60)
        ys = y0 + k * gap + 0.55 * np.sin((xs - x0) / L * 2 * np.pi)
        ax.plot(xs[:-3], ys[:-3], color=color, lw=1.1, zorder=3, solid_capstyle="round")
        _arrow(ax, (xs[-6], ys[-6]), (xs[-1], ys[-1]), color=color, lw=1.1, ms=6, z=3)


def _sun(ax, x, y, r=2.2):
    ax.add_patch(plt.Circle((x, y), r, fc="#f6b40e", ec="none", zorder=2))
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        ax.plot([x + 1.35 * r * np.cos(a), x + 1.85 * r * np.cos(a)],
                [y + 1.35 * r * np.sin(a), y + 1.85 * r * np.sin(a)], color="#f6b40e", lw=1.0, zorder=2)


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 0  Graphical overview of the study
# ═════════════════════════════════════════════════════════════════════════════
def _elbow(ax, pts, color=INK2, lw=1.4, ls="-", ms=10, z=4):
    """Orthogonal connector through `pts`, with an arrowhead on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw, ls=ls, zorder=z,
            solid_joinstyle="miter", solid_capstyle="butt")
    _arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, ms=ms, z=z + 0.1)


def fig_overview():
    M = _macros()
    g = lambda k, d="?": M.get(k, d).replace("{,}", ",").replace("\\ensuremath{-}", "-")
    W, H = 160, 138
    fig, ax = _canvas(7.0, 7.0 * H / W, W, H)
    cP_, cM, cD, cE, cF = PAL["input"], "#2e7d32", PAL["dss"], PAL["plan"], "#b8452e"
    HD = 10.0                                    # header height of every card
    FB, FS = 7.4, 6.8                            # body and secondary text sizes (pt)
    ax.text(W / 2, H - 3.6, B("Wind-aware mission planning for energy-efficient agricultural UAV spraying"
                              " using dynamic soft sets"), fontsize=9.2, ha="center", va="center", color=INK)
    yB, hB = 2.0, 58.0
    yT, hT = yB + hB + 6.0, 64.0

    def card(x, y, w, h, title, c, ic):
        _card(ax, x, y, w, h, title, c, ic, fc=mcolors.to_rgba(c, 0.035), head=HD, tsize=9.4)

    # 1 Problem ────────────────────────────────────────────────────────────
    x, w = 1, 54
    card(x, yT, w, hT, "1  Problem", cP_, "quadcopter")
    sx, sy, sw, sh = x + 1.4, yT + 1.4, 23.0, hT - HD - 2.8
    ax.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="#e6f0fa", ec="none", zorder=1))
    _crop_field(ax, sx, sy, sw, 15.0)
    _sun(ax, sx + sw - 4.2, sy + sh - 4.4, 1.9)
    _wind(ax, sx + 1.2, sy + sh - 10.5, 8.0, n=3, gap=2.3)
    ax.text(sx + 1.2, sy + sh - 3.0, r"$\mathbf{v}_w(u,t)$", fontsize=FS, color="#2a5fa8", zorder=9)
    _hexacopter(ax, sx + sw / 2 + 0.8, sy + 28.5, s=0.72)
    items = [("weather-windy", "Diurnal wind,\nturbulence,\ngusts"),
             ("battery-high", r"Battery \SI{497}{Wh}," "\n" r"tank \SI{10}{L}," "\n" "refills"),
             ("target", r"Dose $0.8$--$1.2\,R^*$," "\n" "limited drift")]
    for k, (n, t) in enumerate(items):
        _item(ax, x + 30.2, yT + hT - HD - 8.8 - 15.6 * k, n, t, cP_, fs=FB, glyph=7.6, bfac=0.66)

    # 2 Physical models ────────────────────────────────────────────────────
    x, w = 59, 46
    card(x, yT, w, hT, "2  Physical models", cM, "flask-outline")
    tiles = [("weather-windy", "Atmosphere", "OU anomalies,\nrandom field"),
             ("flash", "Flight power", "momentum\n" r"theory, $P_e(V_a,m)$"),
             ("spray", "Deposition", "3 droplet classes,\nrotor downwash"),
             ("map-marker-path", "Mission", "sorties, refills,\nbattery, tank")]
    tw, th = (w - 4.4) / 2, (hT - HD - 1.4 - 1.2 - 1.4) / 2
    for k, (n, t1, t2) in enumerate(tiles):
        tx = x + 1.6 + (k % 2) * (tw + 1.2)
        ty = yT + hT - HD - 1.4 - th - (k // 2) * (th + 1.2)
        ax.add_patch(FancyBboxPatch((tx, ty), tw, th, boxstyle="round,pad=0,rounding_size=1.0",
                                    fc=mcolors.to_rgba(cM, 0.08), ec="none", zorder=3))
        icon(ax, n, tx + tw / 2, ty + th - 6.2, 7.6, color="white", badge=cM, bfac=0.66)
        ax.text(tx + tw / 2, ty + 9.3, B(t1), fontsize=FB + 0.4, ha="center", va="center", color=INK, zorder=6)
        ax.text(tx + tw / 2, ty + 3.6, t2, fontsize=FS, ha="center", va="center", color=INK2, zorder=6,
                linespacing=1.15)

    # 3 DSS-WAEOP ──────────────────────────────────────────────────────────
    x, w = 109, 50
    card(x, yT, w, hT, "3  DSS-WAEOP planner", cD, "hexagon-multiple")
    steps = [("hexagon-multiple", "Dynamic soft set\n" r"$(F_t, A_t)$, $A_t\subseteq E$"),
             ("dna", "NSGA-II: elitist\ngenetic algorithm"),
             ("gate-arrow-right", "Look-ahead deferral,\nlane skipping"),
             ("sync", "Replanning when\n" r"$\hat\sigma<\sigma_{\min}$")]
    gap = 1.3
    chh = (hT - HD - 2.8 - 3 * gap) / 4
    for k, (n, t) in enumerate(steps):
        cy = yT + hT - HD - 1.4 - chh / 2 - k * (chh + gap)
        _chip(ax, x + 1.6, cy - chh / 2, w - 3.2, chh, "", cD, fc=mcolors.to_rgba(cD, 0.08))
        icon(ax, n, x + 7.4, cy, 7.4, color="white", badge=cD, bfac=0.66)
        ax.text(x + 14.0, cy, t, fontsize=FB, va="center", color=INK, zorder=7, linespacing=1.2)

    # 4 Evaluation ─────────────────────────────────────────────────────────
    x, w = 1, 84
    card(x, yB, w, hB, "4  Evaluation", cE, "scale-balance")
    top = yB + hB - HD
    ax.text(x + 25.4, top - 5.0, B(f"{g('NScen')} factorial scenarios"), fontsize=FB + 1.0, ha="center",
            va="center", color=INK)
    fac = [("ruler-square", "4 sizes"), ("terrain", "3 terrains"), ("weather-partly-cloudy", "4 seasons"),
           ("barley", "4 crops")]
    for k, (n, t) in enumerate(fac):
        cx_ = x + 7.4 + 12.0 * k
        icon(ax, n, cx_, top - 19.0, 8.0, color="white", badge=cE, bfac=0.66)
        ax.text(cx_, top - 27.8, t, fontsize=FS, ha="center", va="center", color=INK2)
    ax.text(x + 25.4, yB + 9.0, "all results from simulation;\nopen code and data", fontsize=FS,
            ha="center", va="center", color=INK2, style="italic", linespacing=1.2)
    ax.plot([x + 51.5, x + 51.5], [yB + 3, top - 2], color=mcolors.to_rgba(cE, 0.35), lw=0.7)
    ev = [("routes", "7 planners, incl.\nNSGA-C and T-FSS"),
          ("tune-variant", "4 ablations,\nsensitivity and\nstructure checks"),
          ("chart-bell-curve", "Wilcoxon--Holm\ntests, bootstrap\n" r"95\,\% CIs")]
    for k, (n, t) in enumerate(ev):
        _item(ax, x + 57.0, top - 8.0 - 15.4 * k, n, t, cE, fs=FB, glyph=7.6, bfac=0.66)

    # 5 Findings ───────────────────────────────────────────────────────────
    x, w = 89, 70
    card(x, yB, w, hB, "5  Main findings", cF, "lightbulb-on-outline")
    tiles = [(f"$-${g('NsgacEVsBstAbs')}\\,\\%", "energy per treated ha,\nNSGA-II vs. BST",
              "#1b7f4a", "flash"),
             (f"{g('TfssCov')} $\\rightarrow$ {g('DssCov')}\\,\\%", "coverage: hard gate\nvs. look-ahead DSS",
              PAL["dss"], "grid"),
             (f"$+${g('DssEVsNsgac')}\\,\\%", "energy, DSS vs. NSGA-C;\nno deposition benefit", "#b8452e",
              "alert-octagon"),
             ("Field trials", "needed to test multi-\nattribute deferral", PAL["input"], "flask-outline")]
    tw, th = (w - 4.4) / 2, (hB - HD - 1.3 - 1.2 - 1.3) / 2
    for k, (big, small, c, ic) in enumerate(tiles):
        tx = x + 1.6 + (k % 2) * (tw + 1.2)
        ty = yB + hB - HD - 1.3 - th - (k // 2) * (th + 1.2)
        ax.add_patch(FancyBboxPatch((tx, ty), tw, th, boxstyle="round,pad=0,rounding_size=1.0",
                                    fc=mcolors.to_rgba(c, 0.09), ec=c, lw=0.8, zorder=5))
        icon(ax, ic, tx + 5.6, ty + th - 6.0, 7.0, color="white", badge=c, zorder=7, bfac=0.66)
        ax.text(tx + 11.2 + (tw - 11.2) / 2, ty + th - 6.0, B(big), fontsize=10.0 if k < 3 else 9.0,
                ha="center", va="center", color=c, zorder=6)
        ax.text(tx + tw / 2, ty + 0.38 * th, small, fontsize=FS + 0.2, ha="center", va="center", color=INK,
                zorder=6, linespacing=1.2)

    # flow ────────────────────────────────────────────────────────────────
    yc = yT + hT / 2
    _arrow(ax, (55.3, yc), (58.7, yc), color=INK2, lw=1.4, ms=10)
    _arrow(ax, (105.3, yc), (108.7, yc), color=INK2, lw=1.4, ms=10)
    _arrow(ax, (85.3, yB + hB / 2), (88.7, yB + hB / 2), color=INK2, lw=1.4, ms=10)
    ym = yB + hB + 3.0
    _elbow(ax, [(134.0, yT - 0.2), (134.0, ym), (40, ym), (40, yB + hB + 0.2)], color=INK2, lw=1.4)
    _save(fig, "fig00_overview")


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 1  Framework architecture (DSS layer + NSGA-II planner + execution + replanning)
# ═════════════════════════════════════════════════════════════════════════════
def _node(ax, cx, cy, w, h, title, sub, color, ic=None, fs=6.0, sfs=5.4, dashed=False, fill=None, z=5):
    """Outlined node (white fill, coloured border, coloured icon) — the block-diagram style of Fig. 4."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                fc=fill or "white", ec=color, lw=1.1, ls=(0, (3, 2)) if dashed else "-", zorder=z))
    tx = cx - w / 2 + 2.0
    if ic:
        icon(ax, ic, cx - w / 2 + 3.6, cy, 4.2, color=color, zorder=z + 1)
        tx = cx - w / 2 + 6.6
    if sub:
        ax.text(tx, cy + 1.9, B(title), fontsize=fs, ha="left", va="center", color=INK, zorder=z + 1)
        ax.text(tx, cy - 2.0, sub, fontsize=sfs, ha="left", va="center", color=INK2, zorder=z + 1,
                linespacing=1.15)
    else:
        ax.text(tx, cy, title, fontsize=fs, ha="left", va="center", color=INK, zorder=z + 1, linespacing=1.15)


def _arc(ax, c, r, a0, a1, color, lw=1.3, ms=8, n=60, z=3):
    """Arc from angle a0 to a1 (degrees; clockwise if a1 < a0) with an arrowhead."""
    a = np.deg2rad(np.linspace(a0, a1, n))
    xs, ys = c[0] + r * np.cos(a), c[1] + r * np.sin(a)
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="round")
    _arrow(ax, (xs[-3], ys[-3]), (xs[-1], ys[-1]), color=color, lw=lw, ms=ms, z=z + 0.1)


def fig_architecture():
    """DSS-WAEOP as a block diagram with a closed run-time loop (a), and the same modules acting over
    the illustrative operating day (b)."""
    from matplotlib.patches import RegularPolygon, Patch
    cD, cP, cE, cR, cI = PAL["dss"], PAL["plan"], PAL["exec"], PAL["replan"], PAL["input"]
    fig = plt.figure(figsize=(7.0, 5.62))
    W, Ha = 160, 60
    ax = fig.add_axes([0.0, 0.533, 1.0, 0.467])
    ax.set_xlim(0, W); ax.set_ylim(0, Ha); ax.set_aspect("equal"); ax.axis("off")

    # stage backgrounds
    for (x0, x1, c, lab) in ((0.5, 91.0, cP, r"Plan time: once at 06:00, on the forecast"),
                             (92.5, 159.5, cE, r"Run time: 06:00--18:30, on measurements")):
        ax.add_patch(FancyBboxPatch((x0, 1.0), x1 - x0, Ha - 6.5, boxstyle="round,pad=0,rounding_size=2.4",
                                    fc=mcolors.to_rgba(c, 0.05), ec=mcolors.to_rgba(c, 0.35), lw=0.8, zorder=0))
        ax.text(x0 + 2.0, Ha - 8.1, B(lab), fontsize=6.5, color=c, va="center", ha="left")
    ax.text(0.5, Ha - 1.9, B("(a)"), fontsize=9, va="center", ha="left", color=INK)
    ax.text(6.2, Ha - 1.9, B("Structure of DSS-WAEOP") + ": plan once, then execute in a closed loop",
            fontsize=7.2, va="center", ha="left", color=INK)

    # inputs (dashed: data, not computation)
    inp = [("radio-tower", "06:00 station", r"$e_1$--$e_6$ measured"),
           ("weather-partly-cloudy", "Climatology", "and anomaly"),
           ("map-outline", "Field and crop", r"$R^*$, UAV model")]
    ys_in = [42.0, 28.0, 14.0]
    for (n, t, s_), y in zip(inp, ys_in):
        _node(ax, 12.6, y, 23, 10, t, s_, cI, n, dashed=True, fs=5.9, sfs=5.2)
    ax.plot([24.2, 25.6, 25.6, 24.2], [42.0, 42.0, 14.0, 14.0], color=cI, lw=1.0, zorder=2)
    ax.plot([24.2, 25.6], [28.0, 28.0], color=cI, lw=1.0, zorder=2)

    # M1: hexagon
    hx, hy, hr = 38.0, 28.0, 10.4
    _arrow(ax, (25.6, hy), (hx - hr - 0.2, hy), color=INK2, lw=1.4)
    ax.add_patch(RegularPolygon((hx, hy), 6, radius=hr, orientation=np.pi / 6, fc="white", ec=cD, lw=1.4,
                                zorder=5))
    icon(ax, "hexagon-multiple", hx, hy + 5.8, 4.0, color=cD, zorder=6)
    ax.text(hx, hy + 2.0, B("M1  DSS layer"), fontsize=6.1, ha="center", va="center", color=INK, zorder=6)
    for k, t in enumerate((r"$A_t$, $W_t(e)$", r"$Z_a(t)$, $\Phi(u,t)$", r"$\hat\sigma(t)$")):
        ax.text(hx, hy - 1.6 - 2.5 * k, t, fontsize=5.1, ha="center", va="center", color=INK2, zorder=6)
    bx0, bx1, by0, by1 = 50.6, 89.0, 6.5, 50.0
    _arrow(ax, (hx + hr + 0.2, hy), (bx0 - 0.2, hy), color=INK2, lw=1.4)

    # M2: NSGA-II with its generation loop drawn as a cycle
    ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0, boxstyle="round,pad=0,rounding_size=1.6",
                                fc="white", ec=cP, lw=1.4, zorder=5))
    icon(ax, "dna", bx0 + 3.6, by1 - 3.6, 4.4, color=cP, zorder=6)
    ax.text(bx0 + 6.8, by1 - 3.0, B("M2  NSGA-II planner"), fontsize=6.2, va="center", color=INK, zorder=6)
    ax.text(bx0 + 6.8, by1 - 6.4, "elitist multi-objective GA", fontsize=5.3, va="center", color=cP, zorder=6)
    lc, lr = ((bx0 + bx1) / 2 - 1.4, 27.0), 6.8
    for ang in (90, 0, 270, 180):
        _arc(ax, lc, lr, ang - 16, ang - 74, cP, lw=1.0, ms=6, z=6)
    steps = {90: ("select, SBX", (0, 2.6, "center", "bottom")),
             0: ("mutate\n(DSS-biased)", (2.1, 0, "left", "center")),
             270: ("simulate\n" r"$f_1,\mathcal{P},\mathcal{C}$", (0, -2.4, "center", "top")),
             180: ("sort,\ncrowding", (-2.1, 0, "right", "center"))}
    for ang, (t, off) in steps.items():
        px, py = lc[0] + lr * np.cos(np.deg2rad(ang)), lc[1] + lr * np.sin(np.deg2rad(ang))
        ax.add_patch(plt.Circle((px, py), 1.2, fc=cP, ec="white", lw=0.6, zorder=7))
        ax.text(px + off[0], py + off[1], t, fontsize=4.6, ha=off[2], va=off[3], color=INK, zorder=7,
                linespacing=1.05)
    ax.text(lc[0], lc[1], "24\ngenerations", fontsize=4.8, ha="center", va="center", color=cP, linespacing=1.0,
            zorder=7)
    ax.text(lc[0], by0 + 2.4, r"knee point $x^\star\in\mathcal{F}_1$", fontsize=5.6, ha="center", va="center",
            color=cP, zorder=7)

    # run-time loop: M3 -> refill -> sigma check -> (no) M3 | (yes) M4 -> M3; measurements feed M3
    m3c, m3y, m3w, m3h = 111.0, 42.0, 32.0, 9.5
    m3b = m3y - m3h / 2
    _arrow(ax, (bx1 + 0.2, m3y), (m3c - m3w / 2 - 0.2, m3y), color=cP, lw=1.4)
    ax.text((bx1 + m3c - m3w / 2) / 2, m3y + 1.8, r"$x^\star$", fontsize=6.0, color=cP, ha="center")
    _node(ax, m3c, m3y, m3w, m3h, "M3  execution", r"skip or defer if $G(u_p,t)$", cE, "quadcopter")
    # day end
    _node(ax, 148.5, 45.5, 19.0, 8.0, "Day end", r"$f_1,\mathcal{C},\mathcal{P},\mathcal{O}$", INK2,
          "chart-box-outline", fs=5.8)
    _arrow(ax, (m3c + m3w / 2 + 0.2, 45.5), (138.8, 45.5), color=INK2, lw=1.1, ms=7)
    ax.text(133.0, 42.4, "all done\nor 18:30", fontsize=4.6, color=INK2, ha="center", va="center",
            linespacing=1.0)
    # refill
    rx, ry, rw, rh = 148.0, 26.0, 21.0, 9.0
    _node(ax, rx, ry, rw, rh, "Refill", "swap battery", cE, "battery-charging")
    _elbow(ax, [(m3c + m3w / 2 + 0.2, 38.6), (144.0, 38.6), (144.0, ry + rh / 2 + 0.2)], color=cE, lw=1.3, ms=8)
    ax.text(142.6, 34.6, "tank or battery low", fontsize=5.0, color=cE, ha="right", va="center")
    # decision
    dx, dy, dw, dh = 148.0, 11.0, 15.5, 10.4
    ax.add_patch(Polygon([(dx - dw / 2, dy), (dx, dy + dh / 2), (dx + dw / 2, dy), (dx, dy - dh / 2)], closed=True,
                         fc="white", ec=cR, lw=1.2, zorder=5))
    ax.text(dx, dy, r"$\hat\sigma<\sigma_{\min}$?", fontsize=5.4, ha="center", va="center", color=INK, zorder=6)
    _arrow(ax, (rx, ry - rh / 2 - 0.2), (dx, dy + dh / 2 + 0.2), color=cE, lw=1.3, ms=8)
    # yes -> M4 -> new plan -> M3
    m4x, m4y, m4w, m4h = 122.0, 11.0, 18.0, 9.0
    _arrow(ax, (dx - dw / 2 - 0.2, dy), (m4x + m4w / 2 + 0.2, m4y), color=cR, lw=1.2, ms=7)
    ax.text((dx - dw / 2 + m4x + m4w / 2) / 2, dy + 1.5, "yes", fontsize=5.3, color=cR, ha="center", va="center")
    ax.add_patch(FancyBboxPatch((m4x - m4w / 2, m4y - m4h / 2), m4w, m4h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=mcolors.to_rgba(cR, 0.12), ec=cR, lw=1.1, zorder=5))
    ax.text(m4x, m4y + 1.9, B("M4  replan"), fontsize=5.9, ha="center", va="center", color=INK, zorder=6)
    ax.text(m4x, m4y - 2.0, "M1 on nowcast,\nwarm NSGA-II", fontsize=4.8, ha="center", va="center",
            color=INK2, zorder=6, linespacing=1.1)
    _arrow(ax, (m4x, m4y + m4h / 2 + 0.2), (m4x, m3b - 0.2), color=cR, lw=1.2, ms=7)
    ax.text(m4x + 1.0, 21.0, "new\nplan", fontsize=5.0, color=cR, ha="left", va="center", linespacing=1.0)
    # no -> back to M3 (around the bottom)
    _elbow(ax, [(dx, dy - dh / 2 - 0.2), (dx, 3.2), (97.0, 3.2), (97.0, m3b - 0.2)], color=cE, lw=1.3, ms=8)
    ax.text(135.0, 4.5, "no: continue", fontsize=5.0, color=INK2, ha="center", va="center")
    # measurements feeding M3
    sx_, sy_, sw_, sh_ = 109.6, 24.0, 20.4, 9.0
    _node(ax, sx_, sy_, sw_, sh_, "Measure", r"wind, $\hat Z_a(t)$", cE, "radio-tower", dashed=True)
    _arrow(ax, (sx_, sy_ + sh_ / 2 + 0.2), (sx_, m3b - 0.2), color=cE, lw=1.0, ls=(0, (3, 2)), ms=7)

    # ── (b) the same modules over the illustrative operating day ──────────
    z, meta = _illus()
    p = z["DSS-WAEOP_paths"]
    sid = p[:, 0].astype(int)
    sl = z["DSS-WAEOP_sigma_log"]
    rp, last = [], -9.0
    for t, v in sl:
        if v < C.OPT.sigma_min and t - last >= C.OPT.replan_min_gap_h:
            rp.append(t); last = t
    b = fig.add_axes([0.135, 0.073, 0.835, 0.35])
    lanes = ["M1 + M2", "M3 sorties", r"$\hat\sigma$ at refills", "M4 replans"]
    yl = {k: i for i, k in enumerate(lanes[::-1])}
    t0, t1 = C.OP_START_H - 0.25, C.OP_END_H + 0.1
    b.set_xlim(t0, t1); b.set_ylim(-0.6, len(lanes) - 0.4)
    b.set_yticks(range(len(lanes)), lanes[::-1]); b.grid(axis="y", visible=False)
    b.set_xlabel(r"Local time (\si{h})")
    for k in range(len(lanes)):
        b.axhspan(k - 0.45, k + 0.45, color="#f6f6f3" if k % 2 else "white", lw=0, zorder=0)
    y = yl["M1 + M2"]
    b.add_patch(Rectangle((C.OP_START_H - 0.12, y - 0.22), 0.12, 0.44, fc=cP, ec="none", zorder=3))
    b.text(C.OP_START_H + 0.15, y, r"plan $x^\star$ computed at 06:00 (\SI{" +
           f"{meta['DSS-WAEOP']['row']['plan_s']:.0f}" + r"}{s})", fontsize=6.0, va="center", color=cP)
    y = yl["M3 sorties"]
    ends = []
    for s_ in range(sid.max() + 1):
        q = p[sid == s_]
        th = q[:, 3] / 3600
        for i in range(1, len(q)):
            col = cE if q[i, 4] > 0.5 else "#b9c3cd"
            b.add_patch(Rectangle((th[i - 1], y - 0.25), th[i] - th[i - 1], 0.5, fc=col, ec="none", zorder=3))
        ends.append((th[0], th[-1]))
    for (s0, e0), (s1, _) in zip([(C.OP_START_H, C.OP_START_H)] + ends[:-1], ends):
        if s1 - e0 > 0.06:
            b.add_patch(Rectangle((e0, y - 0.25), s1 - e0, 0.5, fc="none", ec=cR, hatch="////", lw=0.0, zorder=2))
    b.text(ends[-1][1] + 0.15, y, f"all {int(meta['DSS-WAEOP']['row']['units_done'])} pieces done",
           fontsize=6.0, va="center", color=cE)
    y = yl[r"$\hat\sigma$ at refills"]
    sy = lambda v: y - 0.36 + 0.72 * (v - 0.6) / 0.42
    b.plot(sl[:, 0], sy(sl[:, 1]), color=cR, lw=0.9, marker="o", ms=2.0, zorder=4)
    b.axhline(sy(C.OPT.sigma_min), color=cR, lw=0.7, ls="--", zorder=3)
    b.text(t1 - 0.05, sy(C.OPT.sigma_min) + 0.05, r"$\sigma_{\min}$", fontsize=5.8, color=cR, ha="right",
           va="bottom")
    y = yl["M4 replans"]
    for t in rp:
        b.plot([t], [y], marker="*", ms=8, color=cR, zorder=5)
        b.axvline(t, color=cR, lw=0.6, ls=":", zorder=1)
    hl = [Patch(fc=cE, label="spraying"), Patch(fc="#b9c3cd", label="ferry and turns"),
          Patch(fc="none", ec=cR, hatch="////", label="waiting at base (all pieces deferred)"),
          Line2D([], [], color=cR, marker="*", lw=0, ms=7, label="replan triggered")]
    b.legend(handles=hl, ncol=4, fontsize=5.8, loc="lower center", bbox_to_anchor=(0.45, 1.0),
             handlelength=1.6, columnspacing=1.2)
    b.spines["left"].set_visible(False)
    b.tick_params(axis="y", length=0)
    fig.text(0.0045, 0.502, B("(b)"), fontsize=9, va="top", ha="left", color=INK)
    fig.text(0.04, 0.502, B("The modules over the illustrative day") + f" ({meta['scenario']['field_ha']} ha, "
             f"{meta['DSS-WAEOP']['row']['sorties']} sorties)", fontsize=7.2, va="top", ha="left", color=INK)
    _save(fig, "fig01_architecture")


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 2  Field discretisation and decision variables
# ═════════════════════════════════════════════════════════════════════════════
def fig_discretisation():
    from .atmosphere import FieldGeometry
    from .mission import build_lanes
    geom = FieldGeometry.from_area(5)
    ls = build_lanes(geom, 30.0, "wheat")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 3.15), gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.28))
    L, W = geom.length, geom.width
    # (a) spray lanes coloured by block (adjacent blocks always in clearly different hues)
    cols = [BLUE, YELLOW, AQUA, VIOLET, ORANGE, "#4b5d6e", MAGENTA, GREEN]
    for u in range(ls.n_units):
        p, q = ls.u_start[u], ls.u_end[u]
        a.plot([p[0], q[0]], [p[1], q[1]], color=cols[ls.u_block[u] % 8], lw=1.0, zorder=2,
               solid_capstyle="butt")
    # atmospheric (wind) cells of 25 m, drawn on top of the lanes so that the grid is visible
    xg = [min(i * C.WIND_CELL, L) for i in range(geom.nx + 1)]
    yg = [min(j * C.WIND_CELL, W) for j in range(geom.ny + 1)]
    for x_ in xg[1:-1]:
        a.plot([x_, x_], [0, W], color="#3d3d3d", lw=0.45, ls=(0, (2.5, 2)), alpha=0.65, zorder=3)
    for y_ in yg[1:-1]:
        a.plot([0, L], [y_, y_], color="#3d3d3d", lw=0.45, ls=(0, (2.5, 2)), alpha=0.65, zorder=3)
    a.add_patch(Rectangle((0, 0), L, W, fill=False, ec=INK, lw=1.2, zorder=4))
    a.annotate("", (0, W + 6), (C.WIND_CELL, W + 6), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7),
               annotation_clip=False)
    a.text(C.WIND_CELL + 3, W + 6, r"one wind cell (\SI{25}{m})", fontsize=6.4, color=INK, va="center")
    bshow = -1
    for b_, lanes in enumerate(ls.block_lanes):
        us = np.flatnonzero(np.isin(ls.u_lane, lanes))
        c = np.vstack([ls.u_start[us], ls.u_end[us]]).mean(axis=0)
        if b_ == bshow:
            c = c + 40 * ls.d
        a.text(*c, f"$b_{{{b_ + 1}}}$", fontsize=6.8, ha="center", va="center", color=INK,
               bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=cols[b_], lw=0.6, alpha=0.95), zorder=6)
    a.plot(0, 0, marker="s", ms=7, color=INK, zorder=7, clip_on=False)
    a.annotate("base / refill", (0, 0), (10, -24), fontsize=6.6, color=INK2, va="center",
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6), annotation_clip=False)
    # heading: angle psi between the x axis and the lane direction
    ox, oy, r_ = L * 0.6, -24.0, 30.0
    a.plot([ox, ox + r_ + 6], [oy, oy], color=INK, lw=0.6, clip_on=False)
    a.annotate("", (ox + r_ * np.cos(np.deg2rad(30)), oy + r_ * np.sin(np.deg2rad(30))), (ox, oy),
               arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1), annotation_clip=False)
    th = np.deg2rad(np.linspace(0, 30, 30))
    a.plot(ox + 11 * np.cos(th), oy + 11 * np.sin(th), color=INK, lw=0.6, clip_on=False)
    a.text(ox + 14.5, oy + 2.6, r"$\psi$", fontsize=6.4, color=INK, va="center")
    a.text(ox + r_ + 9, oy, r"lane heading $\psi=30^\circ$", fontsize=6.8, color=INK, va="center")
    a.set_aspect("equal"); a.set_xlim(-8, L + 8); a.set_ylim(-34, W + 13)
    a.set_xlabel(r"$x$ (\si{m})"); a.set_ylabel(r"$y$ (\si{m})"); a.grid(False)
    a.set_title(f"5 ha field: ${geom.nx}\\times{geom.ny}$ wind cells, "
                f"{ls.n_lanes} lanes in $B={len(ls.block_lanes)}$ blocks", loc="left")
    _panel(a, "a", x=-0.1)

    # (b) lane-frame detail: swath, offset, dose grid
    b.set_xlim(-1, 21); b.set_ylim(-9, 16.4); b.grid(False)
    for xg in np.arange(0, 21, C.CROSS_BIN):
        b.plot([xg, xg], [-6, 10], color=GRID, lw=0.4)
    for yg in np.arange(-5, 11, C.DOSE_ALONG):
        b.plot([0, 20], [yg, yg], color=GRID, lw=0.4)
    for k, xc in enumerate((2.5, 7.5, 12.5, 17.5)):
        b.add_patch(Rectangle((xc - 2.5, -6), 5, 16, fc=SEQ_BLUE[0] if k % 2 else "#e6effa",
                              ec="none", alpha=0.7, zorder=0))
        b.plot([xc, xc], [-6, 10], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    xc = 7.5; off = -1.6
    b.annotate("", (xc + off, 11), (xc + off, -6), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
    for x_ in (xc + off, xc):
        b.plot([x_, x_], [10.6, 12.6], color=INK, lw=0.7)
    b.plot([xc + off, xc], [11.6, 11.6], color=INK, lw=0.7)
    b.text(xc + 0.5, 11.6, r"upwind offset $c\,\hat\Delta_b$", ha="left", va="center", fontsize=6.5)
    xs = np.linspace(0, 20, 300)
    g = np.exp(-0.5 * ((xs - (xc + off + 1.6)) / C.SPRAY.sigma0) ** 2)
    b.fill_between(xs, -8.8, -8.8 + 2.6 * g, color=BLUE, alpha=0.35, lw=0)
    b.text(0.0, -8.5, "deposit", fontsize=6.3, color=INK2, ha="left")
    b.annotate("", (19, 3), (14, 3), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
    b.text(16.5, 3.8, "crosswind", fontsize=6.5, color=RED, ha="center",
           bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    b.annotate("", (15, 14.2), (20, 14.2), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7))
    b.text(17.5, 15.0, rf"swath $w_{{\mathrm{{sw}}}}=\SI{{{C.SPRAY.swath:g}}}{{m}}$", ha="center", fontsize=6.5)
    b.text(12.6, -8.5, rf"dose grid $\SI{{{C.CROSS_BIN:g}}}{{m}}\times\SI{{{C.DOSE_ALONG:g}}}{{m}}$", fontsize=6.5, color=INK2)
    b.set_xlabel(r"Cross-track position (\si{m})"); b.set_ylabel(r"Along-track position (\si{m})")
    b.set_title("Lane-frame detail", loc="left")
    _panel(b, "b", x=-0.15)
    _save(fig, "fig02_discretisation")


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 3  Physical models: power and deposition
# ═════════════════════════════════════════════════════════════════════════════
def fig_physics():
    P = C.PLATFORM
    fig, axs = plt.subplots(1, 3, figsize=(W_FULL, 2.45))
    v = np.linspace(0, 15, 151)
    for m, lab, c in ((P.m_dry, "empty tank", "#86b6ef"), (P.m_dry + 5, "half tank", BLUE),
                      (P.mtow, "full tank", "#0d366b")):
        p = ph.power_curve(m, v)
        axs[0].plot(v, p / 1000, color=c, label=rf"{lab} ($m=\SI{{{m:g}}}{{kg}}$)")
        i = np.argmin(p)
        axs[0].plot(v[i], p[i] / 1000, "o", ms=4, color=c)
    axs[0].axvspan(C.SPRAY.v_spray_min, C.SPRAY.v_spray_max, color=GRID, alpha=0.6, lw=0)
    axs[0].text(5, 3.3, "spraying\nspeeds", ha="center", va="center", fontsize=6.5, color=INK2)
    axs[0].set_ylim(0.95, 3.5)
    axs[0].set_xlabel(r"Airspeed $V_a$ (\si{m.s^{-1}})"); axs[0].set_ylabel(r"Electric power $P_e$ (\si{kW})")
    axs[0].legend(loc="lower right", fontsize=6.0, handlelength=1.4); axs[0].set_title("Power versus airspeed", loc="left")
    _panel(axs[0], "a", x=-0.2)

    ang = np.linspace(0, 360, 181)
    g = np.array([[5.0, 0.0]])
    for U, c in ((2, "#86b6ef"), (4, BLUE), (6, "#0d366b")):
        wind = np.stack([U * np.cos(np.deg2rad(ang)), U * np.sin(np.deg2rad(ang))], axis=1)
        pw, _, _ = ph.turbulent_power(np.repeat(g, len(ang), 0), wind, np.full(len(ang), 0.6),
                                      np.full(len(ang), P.mtow), True)
        axs[1].plot(ang, pw / 1000, color=c, label=rf"$e_1=\SI{{{U}}}{{m.s^{{-1}}}}$")
    p0 = float(ph.electric_power(5.0, P.mtow, True))
    axs[1].axhline(p0 / 1000, color=MUTED, lw=0.8, ls="--")
    axs[1].set_xticks([0, 90, 180, 270, 360], ["tail", "cross", "head", "cross", "tail"])
    axs[1].set_xlabel("Wind direction relative to track"); axs[1].set_ylabel(r"Electric power $P_e$ (\si{kW})")
    axs[1].set_title(r"Effect of wind, $v_g=\SI{5}{m.s^{-1}}$", loc="left")
    axs[1].set_ylim(None, axs[1].get_ylim()[1] + 0.36)
    axs[1].legend(fontsize=5.8, loc="upper center", ncol=1, handlelength=1.4, labelspacing=0.3)
    _panel(axs[1], "b", x=-0.2)

    U = np.linspace(0, 7, 71)
    classes = C.CROPS["wheat"]["classes"]
    for vg, ls_, in ((3.0, ":"), (5.0, "-"), (7.0, "--")):
        va = np.sqrt(vg ** 2 + U ** 2)
        _, vi, _ = ph.rotor_state(va, np.full_like(U, P.m_dry + 5))
        dp = ph.deposit_params(U, np.tile([[0, 1.0]], (len(U), 1)), np.tile([[0, 1.0]], (len(U), 1)),
                               va, vi, np.full_like(U, 15.0), np.full_like(U, 0.8), np.zeros_like(U),
                               np.full_like(U, 0.5), classes, meander=0.0)
        for k, (d, f) in enumerate(classes):
            if k == 1 or vg == 5.0:
                col = ["#86b6ef", BLUE, "#0d366b"][k]
                axs[2].plot(U, dp.shift[:, k], color=col, ls=ls_, lw=1.3 if k == 1 else 1.0)
    axs[2].set_xlabel(r"Crosswind $v_\perp$ (\si{m.s^{-1}})")
    axs[2].set_ylabel(r"Mean displacement $\Delta_j$ (\si{m})")
    axs[2].set_title("Deposit displacement", loc="left")
    hl = [Line2D([], [], color=c, label=rf"\SI{{{d}}}{{\micro m}}") for c, (d, _) in zip(["#86b6ef", BLUE, "#0d366b"], classes)]
    hl += [Line2D([], [], color=INK2, ls=s, label=rf"$v_g=\SI{{{v:g}}}{{m.s^{{-1}}}}$") for v, s in ((3, ":"), (5, "-"), (7, "--"))]
    axs[2].set_ylim(-0.8, 27)
    axs[2].legend(handles=hl, fontsize=5.8, ncol=2, loc="upper left", columnspacing=0.8, handlelength=1.6)
    _panel(axs[2], "c", x=-0.2)
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig03_physics")


# ═════════════════════════════════════════════════════════════════════════════
# Illustrative scenario helpers
# ═════════════════════════════════════════════════════════════════════════════
def _illus():
    z = np.load(RAW / "illus.npz")
    meta = json.loads((RAW / "illus.json").read_text(encoding="utf-8"))["meta"]
    return z, meta


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 4  Atmosphere and DSS over one day
# ═════════════════════════════════════════════════════════════════════════════
def fig_dss_day():
    z, meta = _illus()
    h = z["hours"]
    E = z["e_fields"]
    A = z["A_dynamic"]
    k0, k1 = C.step_of_hour(C.OP_START_H), C.step_of_hour(C.OP_END_H)
    fig = plt.figure(figsize=(W_FULL, 7.4))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.3, 0.5, 0.55, 1.1], hspace=0.62, wspace=0.12)

    ax = fig.add_subplot(gs[0, :])
    cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET]
    for i, p in enumerate(C.PARAMS):
        lo, fav, adv = C.THRESHOLDS[p]
        y = (E[i].reshape(len(h), -1).mean(axis=1) - lo) / (fav - lo)
        q = np.percentile(E[i].reshape(len(h), -1), [10, 90], axis=1)
        ax.fill_between(h, (q[0] - lo) / (fav - lo), (q[1] - lo) / (fav - lo), color=cols[i], alpha=0.12, lw=0)
        yy = np.where(A[:, i], y, np.nan)
        ax.plot(h, y, color=cols[i], lw=0.8, alpha=0.45)
        ax.plot(h, yy, color=cols[i], lw=1.8, label=f"$e_{i + 1}$ " + C.PARAM_LABEL[p].split(" (")[0].lower())
    ax.axhline(1, color=INK, lw=0.8, ls="--")
    ax.text(h[-1] + 0.1, 1, "favourable\nupper bound", fontsize=6.3, va="center", color=INK2)
    for i, p in enumerate(C.PARAMS):
        lo, fav, adv = C.THRESHOLDS[p]
    ax.set_xlim(h[0], h[-1]); ax.set_ylabel(r"$\varepsilon_e$ (field mean, 10--90\,\% band)")
    ax.set_xlabel(r"Local time (\si{h})")
    ax.set_title(f"Normalised atmospheric attributes, {meta['scenario']['field_ha']} ha "
                 f"{meta['scenario']['terrain']} field, {C.SEASON_LABEL[meta['scenario']['season']].lower()}"
                 " (thick lines: active in $A_t$)", loc="left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.32)
    ax.legend(ncol=3, fontsize=6.3, loc="upper left")
    _panel(ax, "a", x=-0.06)

    ax2 = fig.add_subplot(gs[1, :])
    for i in range(6):
        ax2.fill_between(h, i + 0.1, i + 0.9, where=A[:, i], step="mid", color=cols[i], lw=0)
    ax2.set_yticks(np.arange(6) + 0.5, [f"$e_{i + 1}$" for i in range(6)])
    ax2.set_xlim(h[0], h[-1]); ax2.set_ylim(0, 6); ax2.grid(False); ax2.invert_yaxis()
    ax2.set_title("Activation timeline of $A_t$ (filled: attribute active)", loc="left")
    ax2.set_xticklabels([])
    _panel(ax2, "b", x=-0.06, y=1.08)
    Za = z["Za_dynamic"].reshape(len(h), -1).mean(axis=1)
    Zall = z["Za_all"].reshape(len(h), -1).mean(axis=1)
    ax3 = fig.add_subplot(gs[2, :])
    ax3.fill_between(h, 100 * Za, 100 * Zall, where=Zall > Za + 1e-9, color=VIOLET, alpha=0.25,
                     lw=0, step="mid", label=r"blocked only under fixed $A_t\equiv E$")
    ax3.plot(h, 100 * Zall, color=VIOLET, lw=1.0, ls="--", label=r"fixed $A_t\equiv E$")
    ax3.plot(h, 100 * Za, color=BLUE, lw=1.3, label="dynamic $A_t$")
    ax3.set_xlim(h[0], h[-1]); ax3.set_ylim(0, 150); ax3.set_yticks([0, 50, 100]); ax3.set_ylabel(r"Cells in $Z_a$ (\%)")
    ax3.set_xlabel(r"Local time (\si{h})"); ax3.legend(fontsize=6.3, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    ax3.set_title("Adverse-zone size", loc="left")
    _panel(ax3, "c", x=-0.06, y=1.1)

    times = (7.0, 10.0, 13.0, 16.0)
    maps = []
    e1max = float(np.percentile(E[0][k0:k1], 99.5))
    L, W = meta["geom"]["L"], meta["geom"]["W"]
    for j, t in enumerate(times):
        k = C.step_of_hour(t)
        a = fig.add_subplot(gs[3, j])
        im = a.imshow(E[0][k].T, origin="lower", extent=(0, L, 0, W), cmap=SEQ, vmin=0, vmax=e1max,
                      aspect="equal")
        a.contour(np.linspace(0, L, E[0].shape[1]), np.linspace(0, W, E[0].shape[2]),
                  z["Za_dynamic"][k].T.astype(float), levels=[0.5], colors=[RED], linewidths=1.0)
        to = np.deg2rad((z["dir_from"][k] + 180) % 360)
        a.annotate("", (L * 0.5 + 0.18 * L * np.sin(to), W * 0.5 + 0.18 * L * np.cos(to)),
                   (L * 0.5 - 0.18 * L * np.sin(to), W * 0.5 - 0.18 * L * np.cos(to)),
                   arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=10))
        frac = 100 * z["Za_dynamic"][k].mean()
        a.set_title(rf"{int(t):02d}:00, $|Z_a|/|U|={frac:.0f}\,\%$", fontsize=7.5)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        maps.append(a)
    # panel (d): label and title on one line, aligned with the labels and titles of panels (a)-(c)
    pa, p0, p3 = ax.get_position(), maps[0].get_position(), maps[-1].get_position()
    ytitle = p0.y1 + 0.044
    x_lab = pa.x0 - 0.06 * pa.width
    fig.text(x_lab, ytitle, B("(d)"), fontsize=9, fontweight="bold", color=INK, va="bottom", ha="left")
    fig.text(pa.x0, ytitle, B("Wind-speed maps") + " (arrow: wind direction; red contour: adverse zone $Z_a$)",
             fontsize=8.5, color=INK, va="bottom", ha="left")
    cax = fig.add_axes([p3.x1 + 0.012, p0.y0, 0.012, p0.height])
    cb = fig.colorbar(im, cax=cax); cb.set_label(r"$e_1$ (\si{m.s^{-1}})", fontsize=7); cb.outline.set_visible(False)
    _save(fig, "fig04_dss_day")


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 5  Trajectories and dose maps
# ═════════════════════════════════════════════════════════════════════════════
def _dose_grid(meta_m, geom, dose, mask):
    d, n = np.array(meta_m["d"]), np.array(meta_m["n"])
    L, W = geom["L"], geom["W"]
    corners = np.array([[0, 0], [L, 0], [L, W], [0, W]])
    umin = (corners @ d).min()
    nr, nc = dose.shape
    ue = umin + np.arange(nr + 1) * C.DOSE_ALONG
    xe = meta_m["x0"] + np.arange(nc + 1) * C.CROSS_BIN
    X = ue[:, None] * d[0] + xe[None, :] * n[0]
    Y = ue[:, None] * d[1] + xe[None, :] * n[1]
    D = np.where(mask, dose, np.nan)
    return X, Y, D


def fig_trajectories():
    z, meta = _illus()
    geom = meta["geom"]
    L, W = geom["L"], geom["W"]
    names = ["BST", "NSGA-C", "DSS-WAEOP"]
    fig, axs = plt.subplots(2, 3, figsize=(W_FULL, 5.0), gridspec_kw=dict(hspace=0.35, wspace=0.08))
    tmin, tmax = C.OP_START_H, C.OP_END_H
    tcmap = mcolors.LinearSegmentedColormap.from_list("t", ["#cde2fb", "#3987e5", "#0d366b", "#eb6834"])
    for j, m in enumerate(names):
        a = axs[0, j]
        s, e = z[f"{m}_ustart"], z[f"{m}_uend"]
        t = z[f"{m}_tstart"] / 3600
        sp = z[f"{m}_sprayed"].astype(bool)
        segs = np.stack([s, e], axis=1)
        lc = LineCollection(segs[sp], array=t[sp], cmap=tcmap, linewidths=0.9)
        lc.set_clim(tmin, tmax)
        a.add_collection(lc)
        if (~sp).any():
            a.add_collection(LineCollection(segs[~sp], colors=[RED], linewidths=0.6, linestyles=":"))
        a.add_patch(Rectangle((0, 0), L, W, fill=False, ec=INK, lw=1.0))
        a.set_xlim(-10, L + 10); a.set_ylim(-10, W + 10); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([]); a.grid(False); [sp.set_visible(False) for sp in a.spines.values()]
        r = meta[m]["row"]
        a.set_title(B(m) + rf" ($\psi={meta[m]['heading']:.0f}^\circ$)" + "\n" + f"{r['sorties']} sorties, "
                    rf"\SI{{{r['duration_h']:.1f}}}{{h}}, $f_1=\SI{{{r['energy_per_treated_ha']:.0f}}}{{Wh.ha^{{-1}}}}$",
                    fontsize=7, fontweight="normal")
        b = axs[1, j]
        X, Y, D = _dose_grid(meta[m], geom, z[f"{m}_dose"], z[f"{m}_mask"])
        rate = meta[m]["rate"]
        im = b.pcolormesh(X, Y, D / rate, cmap=DIV, norm=mcolors.TwoSlopeNorm(1.0, 0.0, 2.0),
                          shading="flat", rasterized=True)
        b.add_patch(Rectangle((0, 0), L, W, fill=False, ec=INK, lw=1.0))
        b.set_xlim(-10, L + 10); b.set_ylim(-10, W + 10); b.set_aspect("equal")
        b.set_xticks([]); b.set_yticks([]); b.grid(False); [sp.set_visible(False) for sp in b.spines.values()]
        b.set_title(rf"$\mathcal{{C}}={100 * r['coverage']:.1f}\,\%$, $\mathcal{{P}}={100 * r['precision']:.1f}\,\%$"
                    "\n" rf"$\mathcal{{O}}={r['drift_pct']:.1f}\,\%$", fontsize=7, fontweight="normal")

    cax = fig.add_axes([0.92, 0.56, 0.012, 0.3])
    cb = fig.colorbar(lc, cax=cax); cb.set_label(r"Spraying time (\si{h})", fontsize=7); cb.outline.set_visible(False)
    cax2 = fig.add_axes([0.92, 0.12, 0.012, 0.3])
    cb2 = fig.colorbar(im, cax=cax2); cb2.set_label(r"Dose $D/R^*$", fontsize=7); cb2.outline.set_visible(False)
    fig.text(0.12, 0.93, B("(a) Executed coverage paths (colour: time sprayed; red dotted: not sprayed)"),
             fontsize=8.5, fontweight="bold")
    fig.text(0.12, 0.475, B("(b) Deposited dose relative to the prescription (white: on target)"),
             fontsize=8.5, fontweight="bold")
    _save(fig, "fig05_trajectories")


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 6  Pareto fronts
# ═════════════════════════════════════════════════════════════════════════════
def fig_pareto():
    z, meta = _illus()
    fig, axs = plt.subplots(1, 3, figsize=(W_FULL, 2.6))
    for m, c, mk in (("NSGA-C", ORANGE, "s"), ("DSS-WAEOP", BLUE, "o")):
        F = z[f"{m}_front"]; Ft = z[f"{m}_front_truth"]
        x = np.array(meta[m]["row"].get("front", F)) if False else F
        axs[0].scatter(F[:, 0], -100 * F[:, 1], s=14, color=c, ec="white", lw=0.4, label=m, zorder=3)
        axs[1].scatter(F[:, 0], -100 * F[:, 2], s=14, color=c, ec="white", lw=0.4, zorder=3)
        axs[2].scatter(F[:, 0], Ft[:, 0], s=14, color=c, ec="white", lw=0.4, zorder=3)
        r = meta[m]["row"]
        # selected plan: smallest distance in the knee metric
        E, P_, Cv = F[:, 0], -F[:, 1], -F[:, 2]
        dd = np.sqrt((E / E.min() - 1) ** 2 + (P_.max() - P_) ** 2 + (Cv.max() - Cv) ** 2)
        k = int(np.argmin(dd))
        for a, yy in ((axs[0], -100 * F[k, 1]), (axs[1], -100 * F[k, 2])):
            a.scatter(F[k, 0], yy, s=70, facecolor="none", ec=c, lw=1.6, zorder=4)
    b = meta["BST"]["row"]
    axs[0].scatter([], [], s=70, facecolor="none", ec=INK2, lw=1.6, label="executed (knee) plan")
    axs[0].legend(loc="lower right", fontsize=6.5)
    axs[0].set_xlabel(r"$f_1$ (\si{Wh.ha^{-1}})"); axs[0].set_ylabel(r"Precision $\mathcal{P}$ (\%)")
    axs[1].set_xlabel(r"$f_1$ (\si{Wh.ha^{-1}})"); axs[1].set_ylabel(r"Coverage $\mathcal{C}$ (\%)")
    axs[0].set_title(r"Planned front: $f_1$ and $\mathcal{P}$", loc="left")
    axs[1].set_title(r"Planned front: $f_1$ and $\mathcal{C}$", loc="left")
    lim = [min(axs[2].get_xlim()[0], axs[2].get_ylim()[0]), max(axs[2].get_xlim()[1], axs[2].get_ylim()[1])]
    axs[2].plot(lim, lim, color=MUTED, lw=0.8, ls="--")
    axs[2].set_xlabel(r"Predicted $f_1$ (\si{Wh.ha^{-1}})"); axs[2].set_ylabel(r"Realised $f_1$ (\si{Wh.ha^{-1}})")
    axs[2].set_title("Predicted versus realised energy", loc="left")
    for i, a in enumerate(axs):
        _panel(a, "abc"[i], x=-0.22)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig06_pareto")


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 7  Main results distributions
# ═════════════════════════════════════════════════════════════════════════════
def fig_main_results():
    df = load_main()
    df = df[df.method.isin(MAIN_METHODS)]
    fig, axs = plt.subplots(2, 2, figsize=(W_FULL, 4.8), sharex=True)
    specs = (("E", r"Energy $f_1$ (\si{Wh.ha^{-1}})"), ("cov", r"Coverage $\mathcal{C}$ (\%)"),
             ("prec", r"Precision $\mathcal{P}$ (\%)"), ("drift", r"Off-target share $\mathcal{O}$ (\%)"))
    rng = np.random.default_rng(0)
    ttl = {"E": "Energy per treated hectare", "cov": "Coverage", "prec": "Precision",
           "drift": "Off-target share"}
    for a, (col, lab), letter in zip(axs.ravel(), specs, "abcd"):
        a.set_title(ttl[col], loc="left")
        data = [df[df.method == m][col].dropna().values for m in MAIN_METHODS]
        bp = a.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                       medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=AXIS),
                       capprops=dict(color=AXIS), boxprops=dict(lw=0))
        for patch, m in zip(bp["boxes"], MAIN_METHODS):
            patch.set_facecolor(mcolors.to_rgba(METHOD_COLOR[m], 0.35))
        for i, (m, d) in enumerate(zip(MAIN_METHODS, data)):
            a.scatter(i + 1 + rng.uniform(-0.18, 0.18, len(d)), d, s=3, color=METHOD_COLOR[m],
                      alpha=0.55, lw=0, zorder=3)
            a.scatter(i + 1, np.mean(d), marker="D", s=16, color="white", ec=INK, lw=0.8, zorder=4)
        a.set_ylabel(lab)
        if col == "E":
            cap = 1000.0
            n_out = int(sum((d > cap).sum() for d in data))
            a.set_ylim(150, cap)
            if n_out:
                a.text(0.99, 0.97, f"{n_out} T-FSS values $>{cap:.0f}$ not shown", transform=a.transAxes,
                       ha="right", va="top", fontsize=6.5, color=INK2)
        _panel(a, letter, x=-0.14, y=1.03)
    for a in axs[1]:
        a.set_xticks(range(1, len(MAIN_METHODS) + 1), MAIN_METHODS, rotation=30, ha="right")
    fig.text(0.5, 0.995, rf"{df.idx.nunique()} scenarios (box: interquartile range (IQR); whiskers: $1.5\times$IQR; $\diamond$: mean)",
             ha="center", va="top", fontsize=7.5, color=INK2)
    fig.tight_layout(h_pad=0.8, rect=(0, 0, 1, 0.98))
    _save(fig, "fig07_main_results")


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 8  Paired differences (forest plot)
# ═════════════════════════════════════════════════════════════════════════════
def fig_forest():
    df = load_main()
    others = [m for m in MAIN_METHODS if m != "DSS-WAEOP"]
    specs = (("E", r"$\Delta f_1$ (\%)", True), ("cov", r"$\Delta\mathcal{C}$ (pp)", False),
             ("prec", r"$\Delta\mathcal{P}$ (pp)", False), ("drift", r"$\Delta\mathcal{O}$ (pp)", True))
    fig, axs = plt.subplots(1, 4, figsize=(W_FULL, 2.6), sharey=True)
    for a, (col, lab, lower), letter in zip(axs, specs, "abcd"):
        for i, m in enumerate(others):
            d = paired(df, "DSS-WAEOP", m, col, rel=(col == "E"))
            lo, hi = boot_ci(d)
            mu = d.mean()
            good = (mu < 0) == lower
            sig = not (lo <= 0 <= hi)
            c = (GREEN if good else RED) if sig else MUTED
            a.plot([lo, hi], [i, i], color=c, lw=2.0, solid_capstyle="round")
            a.plot(mu, i, "o", ms=5, color=c, mec="white", mew=0.6)
        a.axvline(0, color=INK, lw=0.8)
        a.set_xlabel(lab); a.grid(axis="y", visible=False)
        name = {"E": "Energy", "cov": "Coverage", "prec": "Precision", "drift": "Off-target share"}[col]
        a.set_title(B(name) + "\n" + r"{\footnotesize " + ("lower is better" if lower else "higher is better") + "}"
                    if USETEX else name + "\n" + ("lower is better" if lower else "higher is better"),
                    fontsize=7.6, color=INK, fontweight="normal", loc="left")
        _panel(a, letter, x=-0.22 if letter != "a" else -0.7, y=1.115)
    axs[0].set_yticks(range(len(others)), [f"vs. {m}" for m in others]); axs[0].invert_yaxis()
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig08_forest")


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 9  Season x terrain heat maps + gating fractions
# ═════════════════════════════════════════════════════════════════════════════
def fig_heatmaps():
    g = pd.read_csv(TAB / "table_grid.csv")
    env = load_env()
    fig = plt.figure(figsize=(W_FULL, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.6, wspace=0.45)
    combos = (("DSS-WAEOP", "BST", "E", r"$\Delta f_1$ vs. BST (\%)"),
              ("DSS-WAEOP", "NSGA-C", "E", r"$\Delta f_1$ vs. NSGA-C (\%)"),
              ("DSS-WAEOP", "NSGA-C", "cov", r"$\Delta\mathcal{C}$ vs. NSGA-C (pp)"),
              ("DSS-WAEOP", "T-FSS", "cov", r"$\Delta\mathcal{C}$ vs. T-FSS (pp)"))
    for j, (a_, b_, col, title) in enumerate(combos):
        ax = fig.add_subplot(gs[0, j])
        s = g[(g.a == a_) & (g.b == b_) & (g.metric == col)]
        M = s.pivot(index="season", columns="terrain", values="mean").reindex(
            index=list(C.SEASONS), columns=list(C.TERRAINS))
        lim = np.nanmax(np.abs(M.values)) or 1
        sign = -1 if col == "E" else 1          # blue = better for DSS-WAEOP in every panel
        im = ax.imshow(sign * M.values, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
        for (ii, jj), v in np.ndenumerate(M.values):
            ax.text(jj, ii, f"${v:+.1f}$", ha="center", va="center", fontsize=6.8,
                    color="white" if abs(v) > 0.6 * lim else INK)
        ax.set_xticks(range(3), [t.capitalize() for t in C.TERRAINS], fontsize=6.8)
        ax.set_yticks(range(4), [C.SEASON_LABEL[s] for s in C.SEASONS] if j == 0 else [], fontsize=6.8)
        ax.grid(False); ax.set_title(title, fontsize=7.5)
        if j == 0:
            _panel(ax, "a", x=-0.75)
    fig.text(0.5, 0.955, B("DSS-WAEOP paired differences by season and terrain (blue: DSS-WAEOP better)"),
             ha="center", fontsize=8.5, fontweight="bold")

    ax = fig.add_subplot(gs[1, :2])
    width = 0.26
    x = np.arange(4)
    for k, (c, lab, col) in enumerate((("frac_reg_gate", r"Regulatory limits only, $G_{\mathrm{reg}}$", "#a8a69f"),
                                       ("frac_dss_adverse", r"DSS adverse zone $Z_a$, dynamic $A_t$", BLUE),
                                       ("frac_all_adverse", r"Adverse in any attribute, $A_t\equiv E$", VIOLET))):
        v = [100 * env[env.season == s][c].mean() for s in C.SEASONS]
        ax.bar(x + (k - 1) * width, v, width * 0.92, color=col, label=lab)
    ax.set_xticks(x, [C.SEASON_LABEL[s] for s in C.SEASONS])
    ax.set_ylabel("Field time not\n" r"sprayable (\%)"); ax.set_ylim(0, 112); ax.set_yticks([0, 20, 40, 60, 80])
    ax.legend(fontsize=6.3, loc="upper right", ncol=1)
    ax.set_title("Field time blocked by each rule", loc="left")
    _panel(ax, "b", x=-0.16)

    df = load_main()
    ax = fig.add_subplot(gs[1, 2:])
    for m in ("BST", "NSGA-C", "T-FSS", "DSS-WAEOP"):
        v = [df[(df.method == m) & (df.field_ha == f)]["cov"].mean() for f in C.FIELD_SIZES_HA]
        ax.plot(C.FIELD_SIZES_HA, v, marker="o", ms=4, color=METHOD_COLOR[m] if m != "BST" else INK2,
                label=m)
    ax.set_xscale("log"); ax.set_xticks(C.FIELD_SIZES_HA, [str(f) for f in C.FIELD_SIZES_HA])
    ax.minorticks_off()
    ax.set_xlabel(r"Field size (\si{ha})"); ax.set_ylabel(r"Coverage $\mathcal{C}$ (\%)")
    ax.set_title("Coverage versus field size", loc="left"); ax.legend(fontsize=6.5)
    _panel(ax, "c", x=-0.16)
    _save(fig, "fig09_heatmaps")


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 10  Ablation + convergence
# ═════════════════════════════════════════════════════════════════════════════
def fig_ablation_convergence():
    t = pd.read_csv(TAB / "table_ablation.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W_FULL, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    order = ["DSS-WAEOP", "A1", "A2", "A3", "A4"]
    lab = ["Full", "A1", "A2", "A3", "A4"]
    for a, col, ylab, letter in ((axs[0], "E", r"$f_1$ (\si{Wh.ha^{-1}})", "a"),
                                 (axs[1], "cov", r"Coverage $\mathcal{C}$ (\%)", "b")):
        for i, m in enumerate(order):
            r = t[t.method == m].iloc[0]
            c_ = BLUE if m == "DSS-WAEOP" else "#86b6ef"
            a.errorbar(i, r[col], yerr=[[r[col] - r[col + "_lo"]], [r[col + "_hi"] - r[col]]],
                       color=c_, lw=1.6, capsize=0, fmt="o", ms=6, mec="white", mew=0.8)
        a.set_xticks(range(5), lab, fontsize=7.5)
        a.set_ylabel(ylab)
        a.set_title({"E": "Energy of the variants", "cov": "Coverage of the variants"}[col], loc="left")
        a.set_xlim(-0.6, 4.6)
        a.grid(axis="x", visible=False)
        _panel(a, letter, x=-0.2)
    conv = pd.read_csv(TAB / "table_convergence.csv")
    for m, c, lab_ in (("DSS-WAEOP", BLUE, "with DSS-biased mutation"), ("A3", MUTED, "without (A3)")):
        s = conv[conv.method == m]
        # normalise each scenario by the best final HV of the two variants
        ref = conv[conv.gen == conv.gen.max()].groupby("idx").hv.max()
        s = s.assign(rel=s.hv / s.idx.map(ref))
        mu = s.groupby("gen").rel.mean()
        q = s.groupby("gen").rel.quantile([0.25, 0.75]).unstack()
        axs[2].plot(mu.index, mu.values, color=c, label=lab_)
        axs[2].fill_between(q.index, q[0.25], q[0.75], color=c, alpha=0.15, lw=0)
    axs[2].axvline(C.OPT.gens, color=INK, lw=0.7, ls=":")
    axs[2].text(C.OPT.gens + 0.8, axs[2].get_ylim()[1] - 0.001, "default budget", fontsize=6.3, color=INK2, va="top")
    axs[2].set_xlabel("Generation"); axs[2].set_ylabel("Relative hypervolume")
    axs[2].legend(fontsize=6.5, loc="lower right")
    axs[2].set_title("Planning-stage convergence", loc="left")
    _panel(axs[2], "c", x=-0.2)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig10_ablation_convergence")


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 11  Sensitivity
# ═════════════════════════════════════════════════════════════════════════════
def fig_sensitivity():
    t = pd.read_csv(TAB / "table_sensitivity.csv")
    fig, axs = plt.subplots(1, 4, figsize=(W_FULL, 2.5))
    s = t[(t.group == "sigma") & (t.method == "DSS-WAEOP")]
    labs = list(s.label)
    x = np.arange(len(s))
    a = axs[0]
    a.plot(x, s.E, marker="o", color=BLUE)
    for xi, (e, tr) in enumerate(zip(s.E, s.triggers)):
        a.annotate(f"{tr:.1f}", (xi, e), textcoords="offset points", xytext=(0, 6), ha="center",
                   fontsize=6.3, color=INK2)
    a.set_xticks(x, labs, rotation=35, ha="right"); a.set_xlabel(r"$\sigma_{\min}$")
    a.set_ylabel(r"$f_1$ (\si{Wh.ha^{-1}})")
    lo_, hi_ = float(s.E.min()), float(s.E.max())
    a.set_ylim(lo_ - 0.3 * (hi_ - lo_) - 2, hi_ + 0.45 * (hi_ - lo_) + 2)
    a.set_title(r"Replanning", loc="left")
    a.text(0.04, 0.04, "labels: replans per mission", transform=a.transAxes, fontsize=6.2, color=INK2)
    _panel(a, "a", x=-0.3)
    for a, g, title, letter in ((axs[1], "thr", "Thresholds", "b"), (axs[2], "noise", "Sensor noise", "c")):
        s = t[(t.group == g) & (t.method == "DSS-WAEOP")]
        x = np.arange(len(s))
        a.plot(x, s["cov"], marker="o", color=BLUE, label=r"$\mathcal{C}$")
        a.plot(x, s["prec"], marker="s", color=AQUA, label=r"$\mathcal{P}$")
        a.set_xticks(x, list(s.label)); a.set_ylabel(r"\%"); a.set_title(title, loc="left")
        a.legend(fontsize=6.3, loc="best"); _panel(a, letter, x=-0.25)
    a = axs[3]
    groups = [("indep", "indep", "OU indep."), ("indep", "default", "OU corr."),
              ("meander", "mea0", r"$\omega=0$"), ("meander", "default", r"$\omega=0.5$"), ("meander", "mea1", r"$\omega=1$")]
    for j, m in enumerate(("BST", "NSGA-C", "DSS-WAEOP")):
        vals = []
        for g, key, _ in groups:
            r = t[(t.group == g) & (t.setting == key) & (t.method == m)]
            vals.append(float(r.E.iloc[0]) if len(r) else np.nan)
        col = METHOD_COLOR[m] if m != "BST" else INK2
        a.plot([0, 1], vals[:2], marker="o", ms=3.5, color=col, label=m)
        a.plot([2, 3, 4], vals[2:], marker="o", ms=3.5, color=col)
    a.axvline(1.5, color=AXIS, lw=0.8)
    a.set_xticks(range(len(groups)), [g[2] for g in groups], rotation=35, ha="right", fontsize=6.5)
    a.set_ylabel(r"$f_1$ (\si{Wh.ha^{-1}})"); a.set_title("Model structure", loc="left")
    a.set_ylim(None, a.get_ylim()[1] + 0.35 * (a.get_ylim()[1] - a.get_ylim()[0]))
    a.legend(fontsize=6.0, loc="upper left", ncol=2); _panel(a, "d", x=-0.25)
    fig.tight_layout(w_pad=1.4)
    _save(fig, "fig11_sensitivity")


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 12  Energy breakdown and battery use
# ═════════════════════════════════════════════════════════════════════════════
def fig_energy_battery():
    t = pd.read_csv(TAB / "table_energy_breakdown.csv")
    df = load_main()
    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 2.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a = axs[0]
    comps = (("e_spray", "spraying", BLUE), ("e_ferry", "ferry to and from base", ORANGE),
             ("e_turn", "headland turns", AQUA), ("e_tol", "take-off and landing", YELLOW))
    left = np.zeros(len(t))
    y = np.arange(len(t))
    for c, lab, col in comps:
        a.barh(y, t[c], left=left, color=col, label=lab, height=0.62, ec="white", lw=0.8)
        left += t[c].values
    a.set_yticks(y, t.method); a.invert_yaxis(); a.set_xlabel(r"Energy per treated hectare $f_1$ (\si{Wh.ha^{-1}})")
    a.legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.2)); a.grid(axis="y", visible=False)
    a.set_title("Where the energy goes (mean over scenarios)", loc="left"); _panel(a, "a", x=-0.22)
    a = axs[1]
    s = df[df.method.isin(MAIN_METHODS)]
    a.hist(s.max_sortie_Wh, bins=40, color=BLUE, alpha=0.75, label="max sortie energy")
    a.hist(s.mean_sortie_Wh, bins=40, color=MUTED, alpha=0.6, label="mean sortie energy")
    a.axvline(C.PLATFORM.usable_Wh, color=RED, lw=1.2)
    a.text(C.PLATFORM.usable_Wh - 5, a.get_ylim()[1] * 0.5, "usable battery\n" r"(20\,\% reserve)",
           ha="right", fontsize=6.5, color=RED)
    a.set_xlabel(r"Energy per sortie $\mathcal{E}_q$ (\si{Wh})"); a.set_ylabel("Missions")
    a.legend(fontsize=6.5, loc="upper left"); a.set_title("Sortie energy and battery capacity", loc="left")
    _panel(a, "b", x=-0.2)
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig12_energy_battery")


# ═════════════════════════════════════════════════════════════════════════════
# Fig. 13  Replanning in the illustrative mission
# ═════════════════════════════════════════════════════════════════════════════
def fig_replanning():
    z, meta = _illus()
    if "DSS-WAEOP_sigma_log" not in z:
        return
    sl = z["DSS-WAEOP_sigma_log"]
    h = z["hours"]
    fig, axs = plt.subplots(2, 1, figsize=(W_FULL, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 0.9]))
    a = axs[0]
    a.plot(sl[:, 0], sl[:, 1], "o-", ms=3, color=BLUE, lw=1.0, label=r"$\hat\sigma$ at each refill")
    a.axhline(C.OPT.sigma_min, color=RED, lw=1.0, ls="--")
    a.text(sl[0, 0], C.OPT.sigma_min - 0.035, r"$\sigma_{\min}$", color=RED, fontsize=7)
    trig = sl[sl[:, 1] < C.OPT.sigma_min]
    a.scatter(trig[:, 0], trig[:, 1], s=40, facecolor="none", ec=RED, lw=1.2, zorder=4,
              label=r"$\hat\sigma<\sigma_{\min}$ (replan if $\geq\SI{30}{min}$ since the last one)")
    a.set_ylabel(r"Stability $\hat\sigma$"); a.legend(fontsize=6.5, loc="upper right")
    a.set_title(r"Stability index $\hat\sigma$ of the measured favourable sets (DSS-WAEOP, illustrative mission)", loc="left")
    _panel(a, "a", x=-0.07)
    a = axs[1]
    m1 = z["meas_e1"].reshape(len(h), -1).mean(axis=1)
    f1 = z["fc_e1"].reshape(len(h), -1).mean(axis=1)
    t1 = z["e_fields"][0].reshape(len(h), -1).mean(axis=1)
    a.plot(h, t1, color=INK, lw=1.2, label="truth (field mean)")
    a.plot(h, m1, color=BLUE, lw=0.9, alpha=0.8, label="measured (three-step median)")
    a.plot(h, f1, color=ORANGE, lw=1.2, ls="--", label="06:00 forecast")
    a.axhline(C.THRESHOLDS["e1"][1], color=MUTED, lw=0.7, ls=":")
    a.axhline(C.THRESHOLDS["e1"][2], color=RED, lw=0.7, ls=":")
    a.set_xlim(C.OP_START_H - 0.2, C.OP_END_H + 0.2)
    a.set_title(r"Field-mean wind speed $e_1$: truth, measurements and 06:00 forecast", loc="left")
    a.set_ylabel(r"$e_1$ (\si{m.s^{-1}})"); a.set_xlabel(r"Local time (\si{h})"); a.legend(fontsize=6.5, ncol=3, loc="upper left")
    _panel(a, "b", x=-0.07)
    fig.tight_layout(h_pad=0.5)
    _save(fig, "fig13_replanning")


ALL = [fig_overview, fig_architecture, fig_discretisation, fig_physics, fig_dss_day, fig_trajectories, fig_pareto,
       fig_main_results, fig_forest, fig_heatmaps, fig_ablation_convergence, fig_sensitivity,
       fig_energy_battery, fig_replanning]


def run(only=None):
    """Draw all figures (or those named in `only`); returns the number of failures."""
    print("figures: text typeset with " + ("LaTeX" if USETEX else
          "Matplotlib mathtext (no working LaTeX found; content is identical)"))
    failed = 0
    for f in ALL:
        if only and f.__name__ not in only:
            continue
        try:
            f()
            print("ok   ", f.__name__)
        except Exception as e:  # report and continue so one missing input does not stop the rest
            import traceback
            failed += 1
            print("FAIL ", f.__name__, repr(e))
            traceback.print_exc()
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if run(sys.argv[1:] or None) else 0)
