#!/usr/bin/env python
"""
Nightly model pre-warm.

Trains and caches the top ~30 most-asked tickers to MODEL_DIR so that
the first user request of the day hits a warm cache (sub-second
predict) instead of a cold cache (5–20 min full retrain).

Runs idempotently — for each ticker:
  • Loads the cached model first
  • Trains fresh only when the cache is missing, stale (>12 h), or has
    a TRAINING_VERSION mismatch
  • Saves to MODEL_DIR for the next predict() request to pick up

Designed for a GitHub Actions cron + Render persistent disk path
(MODEL_DIR=/var/models). Safe to run locally too — it'll write to the
local default (.models/) when MODEL_DIR isn't set.

Exits 0 even when individual tickers fail, so one broken ticker
doesn't fail the whole run. Per-ticker results print to stdout for
the Actions log to surface.

Usage:
    python -m scripts.prewarm_models
    python -m scripts.prewarm_models AAPL MSFT       # subset only
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime


# Top tickers by SEO traffic + asking frequency. Mirrors the
# PRE_RENDER_SYMBOLS list on the /stock/[symbol] page + a handful of
# liquid sector / index tickers that get asked a lot in the live
# flow. Keep this ≤ 30 so a full nightly run finishes inside the
# GitHub Actions 6-hour job budget on Render's standard tier; each
# ticker is ~30–90 seconds of train time post-warm-deps.
TOP_TICKERS = [
    # Mega-caps (always asked)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "BRK-B", "JPM", "AVGO",

    # High-traffic secondaries
    "ORCL", "AMD", "INTC", "NFLX", "DIS", "BA", "COIN", "PLTR",
    "SHOP", "DELL",

    # Sector + index ETFs (common comparisons / context)
    "SPY", "QQQ", "DIA", "XLK", "XLF",

    # Misc high-asking
    "EBAY", "PYPL", "UBER", "ABNB", "RBLX",
]


def _train_one(symbol: str) -> tuple[bool, str]:
    """Train + cache `symbol`. Returns (success, message)."""
    try:
        from data_fetcher import (
            fetch_stock_data,
            fetch_fundamentals,
            fetch_earnings_data,
            fetch_options_data,
            fetch_market_context,
        )
        from model import StockPredictor

        df = fetch_stock_data(symbol, period="10y")
        if df is None or len(df) == 0:
            return False, "no price data"

        market_ctx = None
        try:
            market_ctx = fetch_market_context()
        except Exception as exc:
            print(f"  [warn] market_ctx fetch failed: {exc}")

        fundamentals = None
        earnings = None
        options = None
        try:
            fundamentals = fetch_fundamentals(symbol)
        except Exception as exc:
            print(f"  [warn] fundamentals fetch failed: {exc}")
        try:
            earnings = fetch_earnings_data(symbol)
        except Exception as exc:
            print(f"  [warn] earnings fetch failed: {exc}")
        try:
            current_price = float(df["Close"].iloc[-1])
            options = fetch_options_data(symbol, current_price)
        except Exception as exc:
            print(f"  [warn] options fetch failed: {exc}")

        predictor = StockPredictor(symbol)
        # train() loads-from-cache when fresh enough, trains otherwise —
        # both paths leave a warm cache afterwards so the next
        # user-facing predict() request short-circuits.
        predictor.train(
            df,
            market_ctx=market_ctx,
            fundamentals=fundamentals,
            earnings_data=earnings,
            options_data=options,
        )
        # save_model() runs inside train() on a fresh retrain; if the
        # load-from-cache path fired, the existing file's still on
        # disk and untouched. Either way: warm cache after this.
        return True, "ok"

    except ValueError as exc:
        # Most likely "not enough price history" for recent IPOs.
        # Soft failure — record + continue.
        return False, f"skip: {str(exc)[:140]}"
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return False, f"error: {type(exc).__name__}: {str(exc)[:140]}"


def main(argv: list[str] | None = None) -> int:
    argv = argv or []
    tickers = [t.upper() for t in argv] if argv else TOP_TICKERS

    started = datetime.utcnow().isoformat(timespec="seconds")
    print(f"=== Pre-warm started {started}Z ===")
    print(f"MODEL_DIR = {os.environ.get('MODEL_DIR', '(default .models)')}")
    print(f"Tickers   = {len(tickers)}")
    print()

    summary = {"ok": 0, "fail": 0}
    failures: list[tuple[str, str]] = []

    for i, sym in enumerate(tickers, start=1):
        t0 = time.time()
        ok, msg = _train_one(sym)
        elapsed = time.time() - t0
        marker = "✓" if ok else "✗"
        print(
            f"[{i:>2}/{len(tickers)}] {marker} {sym:<6} "
            f"({elapsed:>6.1f}s)  {msg}"
        )
        if ok:
            summary["ok"] += 1
        else:
            summary["fail"] += 1
            failures.append((sym, msg))

    finished = datetime.utcnow().isoformat(timespec="seconds")
    print()
    print(f"=== Pre-warm finished {finished}Z ===")
    print(f"Trained: {summary['ok']} / {len(tickers)}")
    print(f"Failed:  {summary['fail']}")
    if failures:
        print("\nFailures:")
        for sym, msg in failures:
            print(f"  {sym}: {msg}")

    # Always exit 0 — per-ticker failures shouldn't fail the whole
    # run, since the workflow log already surfaces them.
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
