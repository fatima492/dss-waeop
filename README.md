# DSS-WAEOP — reproducibility package

Code and data for the manuscript

> **Wind-aware mission planning for energy-efficient agricultural UAV spraying using dynamic soft sets**

Every number, table and figure in the manuscript is produced by this package from the raw
simulation outputs in `results/raw/`. Nothing is typed in by hand: the manuscript reads its
numbers from `results/tex/macros.tex`, which `reproduce_all.py --analysis` writes.

## Quick start

Requirements: Python 3.9 or newer and the packages in `requirements.txt` (NumPy, SciPy, pandas,
Matplotlib; pytest is optional). The package has been checked with Python 3.9 (NumPy 1.24,
pandas 2.0, Matplotlib 3.7), Python 3.10 and Python 3.13 (NumPy 2.5, pandas 3.0, Matplotlib 3.11).
Run all commands from this folder (the one that contains `reproduce_all.py`).

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_all.py
```

Windows (Command Prompt or PowerShell):

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python reproduce_all.py
```

Without options, `reproduce_all.py` runs `--analysis`. All options:

```bash
python reproduce_all.py --tests      # 14 unit tests of the models (seconds; pytest optional)
python reproduce_all.py --analysis   # all tables, figures and in-text numbers from results/raw (~3 min)
python reproduce_all.py --quick      # smoke-test campaign (10 tasks on 4 scenarios) into results_quick/ (~5 min)
python reproduce_all.py --full       # re-run every simulation (411 tasks, ~2.5 h on 2 cores), then analysis
python reproduce_all.py --no-latex   # as --analysis, but figure text drawn without LaTeX
```

`--full` and `--quick` use all CPU cores by default (`--workers N` to change). Completed tasks
are cached in `results/raw/`, so an interrupted run resumes where it stopped. Delete
`results/raw/` to start from scratch. Individual steps can also be run as modules, e.g.
`python -m dsswaeop.analysis` or `python -m dsswaeop.figures fig_overview fig_physics`.

### Running individual files

Every file can also be run on its own, e.g. with the "Run" button of an editor (VS Code, PyCharm,
Spyder) or from a terminal:

| File | What running it does |
|---|---|
| `reproduce_all.py` | full analysis: all tables, figures and in-text numbers (main entry point) |
| `dsswaeop/config.py` | prints the main constants of the model |
| `dsswaeop/physics.py` | prints the electric power of the UAV at several airspeeds |
| `dsswaeop/atmosphere.py` | generates one scenario and prints its field-mean wind speed |
| `dsswaeop/dss.py` | builds the dynamic soft set on a forecast and prints the active attributes |
| `dsswaeop/mission.py` | prints the lane decomposition of a 5 ha field |
| `dsswaeop/nsga2.py` | solves a small two-objective test problem |
| `dsswaeop/planners.py` | flies the boustrophedon baseline on one field and prints its metrics |
| `dsswaeop/experiments.py` | explains its options; `--quick` or `--full` runs the simulation campaign |
| `dsswaeop/analysis.py` | statistics, tables and `macros.tex` from `results/raw` |
| `dsswaeop/figures.py` | all figures from `results/` (names of single figures can be given) |
| `tests/test_models.py` | runs the 14 unit tests |
| `scripts/fit_climatology_era5.py` | without a file: a demonstration on synthetic data; with an ERA5 CSV: the fitted climatology (JSON) |
| `scripts/bench_onboard.py` | timing benchmark (see below) |

`scripts/bench_onboard.py` prints lines such as
`5 ha  DSS 0.001 s  plan 10.5 s  replan 1.0 s`. These are run times on your computer, not
errors: building the dynamic soft set (M1), the NSGA-II planning stage (M2) and one replanning
(M4) for each field size, to be compared with Table 12 of the manuscript. The times depend on the
computer; a JSON summary follows at the end.

### Troubleshooting

* `ModuleNotFoundError: No module named 'numpy'` (or scipy, pandas, matplotlib): the packages are
  not installed in the Python that runs the script. Activate the virtual environment and run
  `python -m pip install -r requirements.txt`.
* LaTeX is optional. If a working LaTeX installation (with `siunitx`, and `stix` or `lmodern`)
  and `pdftoppm` (poppler) or `dvipng` are found, figure text is typeset by LaTeX; otherwise the
  figures are drawn with Matplotlib's own math renderer and the script says so. The numbers and
  the content of the figures are the same in both cases. `--no-latex` forces the second mode.
* The figures are written as 600 dpi PNG files to `results/figures/`.
* Different SciPy versions can change a few unrounded Wilcoxon p-values in the CSV files of
  `results/tables/` in the second or third significant digit (exact versus approximate tie
  handling); every value printed in the manuscript (`results/tex/`) is identical across the
  versions listed above.

## What is simulated

| Component | Module | Manuscript |
|---|---|---|
| Field geometry, lanes, blocks, dose grid | `dsswaeop/mission.py` (`build_lanes`) | Sec. 4.1, Fig. 2 |
| Atmosphere: diurnal climatology + correlated OU anomalies + terrain/canopy modulation + spatially correlated random field; forecast, nowcast and noisy measurements | `dsswaeop/atmosphere.py` | Sec. 4.2, App. A |
| Dynamic soft set: classes, activation rule A_t, favourable/adverse zones, energy potential Φ, stability index σ | `dsswaeop/dss.py` | Sec. 3 |
| Multirotor power (momentum theory, wind-relative airspeed, turbulence averaging) | `dsswaeop/physics.py` | Sec. 4.3, Fig. 3 |
| Droplet-class deposition with downwash, crosswind, turbulence and meander | `dsswaeop/physics.py`, `mission.py` | Sec. 4.4 |
| Mission: sorties, refill, battery projection, spray gate, skip/wait, daylight window | `dsswaeop/mission.py` (`simulate`) | Sec. 4.5 |
| NSGA-II (elitist non-dominated-sorting genetic algorithm), hypervolume | `dsswaeop/nsga2.py` | Sec. 5.2 |
| All planners (BST, WA-BST, EA-CPP, GRD, NSGA-C, T-FSS, DSS-WAEOP, ablations A1–A4) | `dsswaeop/planners.py` | Sec. 5, 6.2 |
| Experiment campaign | `dsswaeop/experiments.py` | Sec. 6 |
| Statistics, LaTeX tables, in-text numbers | `dsswaeop/analysis.py` | Sec. 6.3, 7 |
| Figures, including the study overview (Fig. 1) and the architecture schematic (Fig. 4) | `dsswaeop/figures.py` | all figures |

All physical and algorithmic constants are in `dsswaeop/config.py`, with units and sources;
Table 4 of the manuscript is generated from it.

## Experiments (`dsswaeop/experiments.py`)

| Task | Count | Content |
|---|---|---|
| `illus` | 1 | one scenario stored in detail (wind, DSS sets, paths, dose maps, Pareto fronts) |
| `main_###` | 192 | full factorial: 4 field sizes × 3 terrains × 4 seasons × 4 crops; BST, WA-BST, EA-CPP, GRD, NSGA-C, T-FSS, DSS-WAEOP, A2 |
| `abl_###` | 48 | ablations A1, A3, A4 (stratified subset) |
| `sens_<setting>_###` | 11 × 12 | σ_min, threshold scale, sensor noise, independent OU noise, meander fraction |
| `seed_###_s` | 8 × 4 | additional optimiser seeds |
| `conv_###` | 6 | hypervolume convergence with/without DSS-biased mutation |

Evaluation scenarios use weather seeds 5000–5191. The method was designed on separate
development seeds (1, 101–103), which are not part of the evaluation.

## Outputs

* `results/raw/*.json` — one file per task (scenario metadata, every method's metrics, Pareto fronts)
* `results/raw/illus.npz` — arrays of the illustrative scenario
* `results/tables/*.csv` — every analysis table
* `results/tex/*.tex` — LaTeX tables and `macros.tex` (all in-text numbers)
* `results/figures/*.png` — all figures as 600 dpi PNG. `figNN` file numbers are stable
  identifiers; the manuscript numbers figures in order of appearance
  (fig00 → Fig. 1, fig02 → Fig. 2, fig03 → Fig. 3, fig01 → Fig. 4, fig04 → Fig. 5, fig05 → Fig. 6,
  fig13 → Fig. 7, fig06 → Fig. 8, fig07 → Fig. 9, fig08 → Fig. 10, fig12 → Fig. 11, fig09 → Fig. 12,
  fig10 → Fig. 13, fig11 → Fig. 14).

### Figure typesetting

If a working LaTeX installation is found (`latex` on the PATH, with `siunitx` and either `stix`
or `lmodern`, plus `pdftoppm` from poppler or `dvipng`), all figure text is typeset by LaTeX in
the same fonts as the manuscript class (STIX when installed, otherwise Latin Modern), and each
PNG is rasterised at 600 dpi from a temporary vector PDF. Otherwise the figures are drawn with
Matplotlib's mathtext (`--no-latex` or `DSS_NO_USETEX=1` forces this); content and numbers are
identical, only the fonts differ.

Schematic icons (`dsswaeop/assets/icons/*.svg`, stored as ready-to-draw outlines in
`dsswaeop/assets/icons/icons.json`) are from Material Design Icons by
Pictogrammers, redistributed under the Apache License 2.0
(`dsswaeop/assets/icons/LICENSE-MaterialDesignIcons.txt`).

## Extra scripts

* `scripts/fit_climatology_era5.py` — re-estimates the seasonal climatology and OU parameters
  (`config.CLIMATE`) from ERA5 hourly single-level data for any location. The manuscript uses
  representative values, not a fit; see Appendix A. Usage:
  `python scripts/fit_climatology_era5.py era5_point.csv --utc-offset 5 > climate_fit.json`.
* `scripts/bench_onboard.py` — times the DSS layer, the planning stage and one replan on the
  machine it runs on (e.g. a Jetson companion computer). The manuscript reports cloud-CPU times only.
  Usage: `python scripts/bench_onboard.py --sizes 5 10 --repeats 3`.

## Limitations

This is a simulation study. The power model has not been validated against flight logs of a
specific aircraft, the deposition model is a Gaussian-plume approximation and has not been
calibrated against field drift measurements, and the climatology is representative rather than
fitted. See Section 8.4 of the manuscript.

## License

MIT (see `LICENSE`).
