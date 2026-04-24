"""
benchmark.py — Prediqt model accuracy benchmark
───────────────────────────────────────────────────────────────────────────
Standalone harness for measuring the model's actual performance. Runs
walk-forward backtests on a fixed basket of tickers across the standard
horizons, then saves results to a timestamped JSON file under `.benchmarks/`.

USE THIS BEFORE AND AFTER any change that could affect accuracy. The before
run becomes your baseline; the after run tells you whether the change
actually helped or hurt.

Usage:
    python benchmark.py                    # run full benchmark (default tickers)
    python benchmark.py --quick            # 2 tickers, 2 horizons (for a fast smoke test)
    python benchmark.py --tickers AAPL TSLA
    python benchmark.py --label "added-oil" # tag this run in the saved JSON

Output:
    • Console summary table
    • .benchmarks/bench_YYYY-MM-DD_HHMM_<label>.json
    • .benchmarks/latest.json (always the most recent run)

To compare two runs:
    python benchmark.py --compare <baseline.json> <new.json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import subprocess
from datetime import datetime
from typing import Optional

# Silence the noisy warnings these libraries emit by default
import warnings
warnings.filterwarnings("ignore")


# ── Benchmark config (keep static for reproducibility) ────────────────────
DEFAULT_TICKERS = ["AAPL", "NVDA", "JPM", "XOM"]          # tech / semi / financial / energy
DEFAULT_HORIZONS = ["1 Week", "1 Month", "1 Quarter"]     # 3D is noisy, 1Y has too few OOS windows
DATA_PERIOD = "5y"                                         # training history per walk-forward window

BENCH_DIR = ".benchmarks"


# ── Small helpers ─────────────────────────────────────────────────────────
def _git_commit() -> str:
    """Short git sha for provenance; 'unknown' if not a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and x != x):  # None or NaN
        return "—"
    return f"{x:.1f}%"


def _fmt_float(x, places: int = 2) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.{places}f}"


# ── The main benchmark run ────────────────────────────────────────────────
def run_benchmark(
    tickers: list[str],
    horizons: list[str],
    label: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Run walk-forward backtests across tickers×horizons and return results."""
    # Lazy imports so we don't pay the startup cost for --compare/--help
    from data_fetcher import (
        fetch_stock_data, fetch_market_context, fetch_fundamentals,
        fetch_earnings_data, fetch_options_data, HORIZONS,
    )
    from backtester import run_backtest

    # Filter horizons to valid ones so we don't fail silently on typos
    invalid = [h for h in horizons if h not in HORIZONS]
    if invalid:
        raise ValueError(
            f"Unknown horizon(s) {invalid}. Valid: {list(HORIZONS.keys())}"
        )

    started_at = datetime.now()
    run_meta = {
        "started_at": started_at.isoformat(),
        "label": label or "",
        "git_commit": _git_commit(),
        "tickers": tickers,
        "horizons": horizons,
        "data_period": DATA_PERIOD,
        "python_version": sys.version.split()[0],
    }

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"Prediqt benchmark — {started_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"  commit: {run_meta['git_commit']}  |  label: {run_meta['label'] or '—'}")
        print(f"  tickers: {', '.join(tickers)}")
        print(f"  horizons: {', '.join(horizons)}")
        print(f"{'=' * 72}\n")

    per_ticker = {}
    total_elapsed = 0.0

    for sym in tickers:
        t0 = time.time()
        if verbose:
            print(f"[{sym}] fetching data…", end=" ", flush=True)

        try:
            df = fetch_stock_data(sym, period=DATA_PERIOD)
            if df is None or df.empty or len(df) < 252:
                raise ValueError(f"Insufficient data for {sym} ({len(df) if df is not None else 0} rows)")

            info_sector = None
            try:
                # Cheap metadata peek to pick the right sector ETF for market_ctx
                import yfinance as yf
                info = yf.Ticker(sym).info
                info_sector = info.get("sector")
            except Exception:
                pass

            market_ctx = fetch_market_context(period=DATA_PERIOD, sector=info_sector)
            fundamentals = fetch_fundamentals(sym)
            earnings = fetch_earnings_data(sym)
            current_price = float(df["Close"].iloc[-1])
            options = {}
            try:
                options = fetch_options_data(sym, current_price) or {}
            except Exception:
                pass

            if verbose:
                print("running walk-forward…", end=" ", flush=True)

            # Only run the horizons we care about — subset the HORIZONS dict
            # We can't subset easily (run_backtest uses all HORIZONS), so we
            # run the full backtest and then filter results. Slightly wasteful
            # but keeps backtester.py unchanged.
            results = run_backtest(
                df,
                market_ctx=market_ctx,
                fundamentals=fundamentals,
                earnings_data=earnings,
                options_data=options,
            )

            # Slim results to target horizons only + drop non-serializable bits
            slim = {}
            for h in horizons:
                r = results.get(h) or {}
                if "error" in r:
                    slim[h] = {"error": r["error"]}
                    continue
                slim[h] = {
                    "dir_accuracy":  r.get("dir_accuracy"),
                    "win_rate":      r.get("win_rate"),
                    "strat_sharpe":  r.get("strat_sharpe"),
                    "bh_sharpe":     r.get("bh_sharpe"),
                    "max_drawdown":  r.get("max_drawdown"),
                    "profit_factor": r.get("profit_factor"),
                    "n_windows":     r.get("n_windows"),
                    "test_start":    str(r.get("test_start", "")),
                    "test_end":      str(r.get("test_end", "")),
                    "regime_accuracy": r.get("regime_accuracy", {}),
                }

            elapsed = time.time() - t0
            total_elapsed += elapsed
            per_ticker[sym] = {
                "status": "ok",
                "elapsed_sec": round(elapsed, 1),
                "horizons": slim,
            }
            if verbose:
                accs = [slim[h].get("dir_accuracy") for h in horizons if "error" not in slim[h]]
                avg = sum(a for a in accs if a is not None) / max(1, len(accs)) if accs else 0
                print(f"done in {elapsed:.1f}s  (avg acc {avg:.1f}%)")

        except Exception as exc:
            elapsed = time.time() - t0
            total_elapsed += elapsed
            per_ticker[sym] = {
                "status": "failed",
                "elapsed_sec": round(elapsed, 1),
                "error": str(exc),
                "traceback": traceback.format_exc(limit=3),
            }
            if verbose:
                print(f"FAILED ({exc})")

    # ── Aggregates across all successful (ticker, horizon) cells ──────────
    cells = []
    for sym, d in per_ticker.items():
        if d.get("status") != "ok":
            continue
        for h, cell in d["horizons"].items():
            if "error" in cell:
                continue
            if cell.get("dir_accuracy") is not None:
                cells.append(cell)

    def _mean(key):
        vals = [c[key] for c in cells if c.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    aggregate = {
        "n_cells": len(cells),
        "avg_dir_accuracy": _mean("dir_accuracy"),
        "avg_win_rate":     _mean("win_rate"),
        "avg_strat_sharpe": _mean("strat_sharpe"),
        "avg_max_drawdown": _mean("max_drawdown"),
        "avg_profit_factor":_mean("profit_factor"),
    }

    finished_at = datetime.now()
    run_meta["finished_at"] = finished_at.isoformat()
    run_meta["total_elapsed_sec"] = round(total_elapsed, 1)

    return {
        "meta": run_meta,
        "aggregate": aggregate,
        "per_ticker": per_ticker,
    }


# ── Pretty-print the summary to the console ───────────────────────────────
def print_summary(result: dict, title: str = "RESULTS") -> None:
    print(f"\n{'─' * 72}")
    print(f"{title}")
    print(f"{'─' * 72}")

    agg = result.get("aggregate", {})
    print(f"  Cells tested:           {agg.get('n_cells', 0)}")
    print(f"  Avg directional acc:    {_fmt_pct(agg.get('avg_dir_accuracy'))}")
    print(f"  Avg win rate:           {_fmt_pct(agg.get('avg_win_rate'))}")
    print(f"  Avg strategy Sharpe:    {_fmt_float(agg.get('avg_strat_sharpe'))}")
    print(f"  Avg max drawdown:       {_fmt_pct(agg.get('avg_max_drawdown'))}")
    print(f"  Avg profit factor:      {_fmt_float(agg.get('avg_profit_factor'))}")
    print()

    # Per-ticker detail
    horizons = result["meta"]["horizons"]
    header = f"  {'Ticker':<8}  " + "  ".join(f"{h:>12}" for h in horizons) + "   avg"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for sym, d in result.get("per_ticker", {}).items():
        if d.get("status") != "ok":
            print(f"  {sym:<8}  FAILED: {d.get('error', '')[:60]}")
            continue
        row_accs = []
        for h in horizons:
            cell = d["horizons"].get(h, {})
            if "error" in cell:
                row_accs.append(None)
            else:
                row_accs.append(cell.get("dir_accuracy"))
        row_vals = "  ".join(
            f"{_fmt_pct(a):>12}" for a in row_accs
        )
        valid_accs = [a for a in row_accs if a is not None]
        avg = sum(valid_accs) / len(valid_accs) if valid_accs else None
        print(f"  {sym:<8}  {row_vals}   {_fmt_pct(avg):>5}")
    print()


# ── Save / compare utilities ──────────────────────────────────────────────
def save_results(result: dict, label: Optional[str] = None) -> str:
    os.makedirs(BENCH_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    tag = f"_{label}" if label else ""
    path = os.path.join(BENCH_DIR, f"bench_{ts}{tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    # Also update latest.json
    with open(os.path.join(BENCH_DIR, "latest.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    return path


def compare(baseline_path: str, new_path: str) -> None:
    """Print a side-by-side diff of two benchmark runs."""
    with open(baseline_path) as f: baseline = json.load(f)
    with open(new_path) as f: new = json.load(f)

    b_agg = baseline.get("aggregate", {})
    n_agg = new.get("aggregate", {})

    print(f"\n{'=' * 72}")
    print("COMPARISON")
    print(f"  baseline: {baseline['meta'].get('label', '—')}  ({baseline['meta'].get('git_commit', '?')})")
    print(f"  new:      {new['meta'].get('label', '—')}  ({new['meta'].get('git_commit', '?')})")
    print(f"{'=' * 72}\n")

    rows = [
        ("Avg directional accuracy", "avg_dir_accuracy", "pct"),
        ("Avg win rate",             "avg_win_rate",     "pct"),
        ("Avg strategy Sharpe",      "avg_strat_sharpe", "float"),
        ("Avg max drawdown",         "avg_max_drawdown", "pct"),
        ("Avg profit factor",        "avg_profit_factor","float"),
    ]
    print(f"  {'Metric':<28} {'Baseline':>12} {'New':>12} {'Δ':>12}")
    print("  " + "─" * 66)
    for label, key, kind in rows:
        b = b_agg.get(key)
        n = n_agg.get(key)
        if b is None or n is None:
            d_str = "—"
        else:
            d = n - b
            sign = "+" if d >= 0 else ""
            d_str = f"{sign}{d:.2f}" + ("pp" if kind == "pct" else "")
        fmt = _fmt_pct if kind == "pct" else lambda x: _fmt_float(x)
        print(f"  {label:<28} {fmt(b):>12} {fmt(n):>12} {d_str:>12}")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Prediqt benchmark harness")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help=f"Override the default ticker basket. Default: {DEFAULT_TICKERS}")
    parser.add_argument("--horizons", nargs="+", default=None,
                        help=f"Override horizons. Default: {DEFAULT_HORIZONS}")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke test: AAPL + NVDA, 1 Week + 1 Month only.")
    parser.add_argument("--label", default=None,
                        help="Tag this run in the saved JSON (e.g. 'baseline', 'after-oil').")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "NEW"),
                        help="Compare two saved benchmark JSON files.")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
        return 0

    tickers = args.tickers or (["AAPL", "NVDA"] if args.quick else DEFAULT_TICKERS)
    horizons = args.horizons or (["1 Week", "1 Month"] if args.quick else DEFAULT_HORIZONS)
    label = args.label or ("quick" if args.quick else "baseline")

    try:
        result = run_benchmark(tickers, horizons, label=label)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        traceback.print_exc()
        return 1

    print_summary(result, title="SUMMARY")
    path = save_results(result, label=label)
    print(f"  → saved to  {path}")
    print(f"  → symlinked {os.path.join(BENCH_DIR, 'latest.json')}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
