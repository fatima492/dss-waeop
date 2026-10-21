#!/usr/bin/env python3
"""
Reproduce every number, table and figure of the manuscript.

  python reproduce_all.py                # same as --analysis
  python reproduce_all.py --analysis     # tables, macros and figures from the shipped raw
                                         # results in results/raw (about 3 minutes)
  python reproduce_all.py --tests        # unit tests of the physical models and the DSS layer
  python reproduce_all.py --quick        # smoke test: 4 scenarios of every experiment type
                                         # into results_quick/ (about 5 minutes)
  python reproduce_all.py --full         # the whole campaign (411 tasks; about 2.5 hours on
                                         # 2 cores, scales with --workers), then --analysis
  add --no-latex to draw the figures with Matplotlib's own math renderer instead of LaTeX

All randomness is seeded; re-running --full reproduces results/raw bit-for-bit on the same
NumPy version (and to floating-point tolerance across versions).
"""
import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MIN_PYTHON = (3, 9)


def _environment_report():
    """Print versions and stop with a clear message if a required package is missing."""
    if sys.version_info < MIN_PYTHON:
        sys.exit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required "
                 f"(found {platform.python_version()}).")
    missing, found = [], []
    for name in ("numpy", "scipy", "pandas", "matplotlib"):
        try:
            mod = __import__(name)
            found.append(f"{name} {mod.__version__}")
        except ImportError:
            missing.append(name)
    if missing:
        sys.exit("Missing package(s): " + ", ".join(missing) +
                 "\nInstall them with:  python -m pip install -r requirements.txt")
    print(f"Python {platform.python_version()} | " + " | ".join(found))


def _run_tests():
    """Run the unit tests with pytest if it is installed, otherwise with a minimal built-in runner."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        pytest = None
    if pytest is not None:
        return subprocess.call([sys.executable, "-m", "pytest", "-q", str(HERE / "tests")], cwd=HERE)
    print("pytest is not installed; running the tests with the built-in runner.")
    import importlib.util
    import inspect
    import traceback
    spec = importlib.util.spec_from_file_location("test_models", HERE / "tests" / "test_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    n_fail = n_ok = 0
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            n_ok += 1
        except Exception:
            n_fail += 1
            print(f"FAILED {name}")
            traceback.print_exc()
    print(f"{n_ok} passed, {n_fail} failed")
    return 1 if n_fail else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--analysis", action="store_true", help="statistics, tables and figures from results/raw "
                   "(default)")
    g.add_argument("--quick", action="store_true", help="small smoke-test campaign into results_quick/")
    g.add_argument("--full", action="store_true", help="full campaign, then analysis")
    g.add_argument("--tests", action="store_true", help="run unit tests")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="parallel worker processes for --quick and --full")
    ap.add_argument("--no-latex", action="store_true", help="do not use LaTeX for figure text")
    a = ap.parse_args()
    if not (a.analysis or a.quick or a.full or a.tests):
        a.analysis = True
        print("No option given: running --analysis (see --help for the other options).")

    _environment_report()
    if a.no_latex:
        os.environ["DSS_NO_USETEX"] = "1"
    if a.tests:
        sys.exit(_run_tests())
    if a.quick:
        os.environ["DSSWAEOP_RESULTS"] = str(HERE / "results_quick")
    from dsswaeop import experiments, analysis, figures
    if a.quick or a.full:
        experiments.run(quick=a.quick, workers=a.workers)
    if a.quick:
        print("Quick campaign finished: raw outputs in results_quick/raw "
              "(too few scenarios for the paper's statistics; use --analysis on results/raw).")
        return
    analysis.run()
    failed = figures.run()
    print("Done. Tables: results/tex and results/tables; figures: results/figures; "
          "numbers quoted in the paper: results/tex/macros.tex")
    if failed:
        sys.exit(f"{failed} figure(s) could not be drawn; see the messages above.")


if __name__ == "__main__":
    main()
