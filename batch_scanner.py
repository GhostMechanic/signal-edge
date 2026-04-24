"""
batch_scanner.py
───────────────────────────────────────────────────────────────────────────────
Two-tier daily pipeline for rapid model learning across the S&P 500 / NASDAQ / DOW.

TIER 1 — Quick Screen  (~15 min for 600 stocks)
  Fast technical analysis on every stock in the universe.
  No model training. Uses vectorised yfinance batch download.
  Produces a ranked list (momentum + RSI + volume surge).

TIER 2 — Deep ML Scan  (~1-2 hours for top 50)
  Full 16-model ensemble training + prediction logging on the top-ranked stocks.
  These are the data points that feed the auto-retraining loop.

Run via:
  python batch_scanner.py                    # both tiers, top 50
  python batch_scanner.py --quick-only       # tier 1 only
  python batch_scanner.py --deep-n 30        # tier 2 on top 30
  python batch_scanner.py --index dow        # dow 30 only
"""

import os
import sys
import json
import logging
import argparse
import warnings
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf

from universe import get_full_universe, get_dow30_tickers, get_nasdaq100_tickers
from data_fetcher import fetch_stock_data, fetch_market_context, fetch_fundamentals, \
    fetch_earnings_data, fetch_options_data, HORIZONS
from model import StockPredictor
from prediction_logger_v2 import (
    log_prediction_v2, score_all_intervals, get_current_model_version,
)
from auto_retrain import apply_confidence_adjustments
from regime_detector import detect_regime
from watchlist import save_watchlist, load_watchlist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCAN_RESULTS_DIR  = ".predictions"
SCAN_RESULTS_FILE = os.path.join(SCAN_RESULTS_DIR, "scan_results.json")


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 — QUICK SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

def quick_screen_universe(tickers: list, period: str = "6mo",
                           batch_size: int = 100,
                           progress_cb=None) -> list:
    """
    Fast technical screen of all tickers using yfinance batch download.
    Returns list of dicts sorted by composite score (best first).
    Each dict: {symbol, score, signal, rsi, momentum_21d, vol_ratio, price, change_1d}
    """
    results = []
    total   = len(tickers)
    logger.info(f"Quick-screening {total} tickers in batches of {batch_size}…")

    for start in range(0, total, batch_size):
        batch = tickers[start : start + batch_size]
        pct_done = start / total * 100
        if progress_cb:
            progress_cb(pct_done, f"Quick screen {start}/{total}…")

        try:
            raw = yf.download(
                batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as e:
            logger.warning(f"Batch download failed ({start}-{start+batch_size}): {e}")
            continue

        for sym in batch:
            try:
                # Extract this ticker's OHLCV
                if len(batch) == 1:
                    df = raw.copy()
                else:
                    if sym not in raw.columns.get_level_values(0):
                        continue
                    df = raw[sym].copy()

                df = df.dropna(subset=["Close"])
                if len(df) < 30:
                    continue

                close  = df["Close"]
                volume = df["Volume"]

                # RSI-14
                delta  = close.diff()
                gain   = delta.clip(lower=0).rolling(14).mean()
                loss   = (-delta.clip(upper=0)).rolling(14).mean()
                rs     = gain / (loss + 1e-9)
                rsi    = float((100 - 100 / (1 + rs)).iloc[-1])

                # Momentum
                mom_21 = float(close.pct_change(21).iloc[-1]) * 100
                mom_5  = float(close.pct_change(5).iloc[-1])  * 100
                chg_1d = float(close.pct_change(1).iloc[-1])  * 100

                # Volume surge
                vol_ratio = float(
                    volume.iloc[-1] / (volume.rolling(20).mean().iloc[-1] + 1)
                )

                # MA position
                ma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float(close.mean())
                ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())
                price = float(close.iloc[-1])
                ma50_dist  = (price - ma50)  / (ma50  + 1e-9) * 100
                ma200_dist = (price - ma200) / (ma200 + 1e-9) * 100

                # Composite momentum score
                # Higher = more interesting for deep scan
                score = (
                    mom_21 * 0.35 +           # 21-day trend (most predictive)
                    mom_5  * 0.25 +           # short-term momentum
                    (vol_ratio - 1) * 5 +     # volume surge bonus
                    (50 - abs(rsi - 50)) * 0.1  # near extremes = interesting
                )

                # Simple signal
                if rsi < 35 and mom_5 < -2:
                    signal = "OVERSOLD"
                elif rsi > 70 and mom_5 > 2:
                    signal = "OVERBOUGHT"
                elif mom_21 > 5 and ma50_dist > 0:
                    signal = "BULLISH"
                elif mom_21 < -5 and ma50_dist < 0:
                    signal = "BEARISH"
                else:
                    signal = "NEUTRAL"

                results.append({
                    "symbol":      sym,
                    "score":       round(score, 2),
                    "signal":      signal,
                    "rsi":         round(rsi, 1),
                    "momentum_21d": round(mom_21, 2),
                    "momentum_5d":  round(mom_5, 2),
                    "change_1d":    round(chg_1d, 2),
                    "vol_ratio":    round(vol_ratio, 2),
                    "price":        round(price, 2),
                    "ma50_dist":    round(ma50_dist, 2),
                    "ma200_dist":   round(ma200_dist, 2),
                })

            except Exception as e:
                logger.debug(f"{sym} quick-screen error: {e}")

        logger.info(f"  … {min(start + batch_size, total)}/{total} screened, "
                    f"{len(results)} valid so far")

    # Sort by absolute score (most directional either way)
    results.sort(key=lambda x: abs(x["score"]), reverse=True)

    # Save scan results
    os.makedirs(SCAN_RESULTS_DIR, exist_ok=True)
    with open(SCAN_RESULTS_FILE, "w") as f:
        json.dump({
            "scanned_at": datetime.now().isoformat(),
            "total_screened": total,
            "results": results,
        }, f, indent=2)

    logger.info(f"Quick screen done. {len(results)} stocks ranked.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — DEEP ML SCAN
# ═══════════════════════════════════════════════════════════════════════════════

def deep_scan(symbols: list, data_period: str = "5y",
              progress_cb=None) -> dict:
    """
    Full 16-model training + prediction logging for each symbol.
    Logs results to prediction_logger_v2 for model learning.
    Returns summary dict.
    """
    total   = len(symbols)
    logged  = 0
    errors  = []
    skipped = 0

    logger.info(f"Deep ML scan: {total} stocks…")

    # Fetch shared market context once (SPY + VIX + sentiment)
    try:
        market_ctx = fetch_market_context(period=data_period, include_sentiment=True)
        logger.info("Market context loaded (SPY + VIX + sentiment)")
    except Exception:
        market_ctx = None

    mv = get_current_model_version()

    for i, sym in enumerate(symbols):
        pct = (i / total) * 100
        if progress_cb:
            progress_cb(pct, f"[{i+1}/{total}] Deep scan: {sym}")

        logger.info(f"  [{i+1}/{total}] {sym}…")

        try:
            # 1. Fetch price data
            df = fetch_stock_data(sym, period=data_period)
            if df is None or len(df) < 100:
                logger.warning(f"    {sym}: insufficient data, skipping")
                skipped += 1
                continue

            # 2. Fundamentals (non-fatal if unavailable)
            try:
                fundamentals = fetch_fundamentals(sym)
            except Exception:
                fundamentals = None

            try:
                earnings_data = fetch_earnings_data(sym)
            except Exception:
                earnings_data = None

            try:
                current_price = float(df["Close"].iloc[-1])
                options_data  = fetch_options_data(sym, current_price)
            except Exception:
                options_data = None

            # 3. Train model
            predictor = StockPredictor(sym)
            predictor.train(
                df,
                market_ctx=market_ctx,
                fundamentals=fundamentals,
                earnings_data=earnings_data,
                options_data=options_data,
            )

            # 4. Predict
            predictions = predictor.predict(df, market_ctx=market_ctx)
            if not predictions:
                skipped += 1
                continue

            # 5. Get feature importance
            try:
                fi = predictor.get_feature_importance()
            except Exception:
                fi = None

            # 6. Detect regime (pass SPY/VIX for full signal set)
            try:
                _spy = market_ctx.get("spy") if market_ctx else None
                _vix = market_ctx.get("vix") if market_ctx else None
                regime_label = detect_regime(df, _spy, _vix).get("label", "Unknown")
            except Exception:
                regime_label = "Unknown"

            # 7. Apply learned confidence adjustments
            try:
                for h in predictions:
                    base_conf = predictions[h].get("confidence", 50)
                    predictions[h]["confidence"] = apply_confidence_adjustments(
                        h, base_conf, regime_label
                    )
            except Exception:
                pass

            # 8. Log prediction
            pred_id = log_prediction_v2(sym, predictions, fi, mv, regime_label)
            logged += 1
            logger.info(f"    ✓ {sym} logged (id={pred_id}, horizons={list(predictions.keys())})")

        except Exception as e:
            errors.append({"symbol": sym, "error": str(e)[:120]})
            logger.warning(f"    ✗ {sym}: {str(e)[:80]}")

    # Score any pending predictions that are now ready
    try:
        score_result = score_all_intervals()
        logger.info(f"Scoring pass: {score_result.get('scored', 0)} intervals scored")
    except Exception:
        pass

    summary = {
        "completed_at":   datetime.now().isoformat(),
        "total_attempted": total,
        "logged":          logged,
        "skipped":         skipped,
        "errors":          len(errors),
        "error_details":   errors[:20],   # keep first 20
        "data_points_added": logged * len(HORIZONS),
    }

    logger.info(
        f"\n{'='*60}\n"
        f"Deep scan complete.\n"
        f"  Logged:  {logged}/{total} stocks\n"
        f"  Skipped: {skipped}\n"
        f"  Errors:  {len(errors)}\n"
        f"  New data points: {logged * len(HORIZONS)}\n"
        f"{'='*60}"
    )
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_daily_pipeline(
    index: str = "all",        # "all" | "sp500" | "nasdaq" | "dow"
    deep_n: int = 50,          # how many top stocks get full ML scan
    quick_only: bool = False,  # set True to skip Tier 2
    data_period: str = "5y",
    progress_cb=None,
) -> dict:
    """
    Main entry point for the daily batch pipeline.
    1. Fetch universe
    2. Quick-screen all tickers
    3. Deep-scan top N
    Returns full summary dict.
    """
    start_time = datetime.now()
    logger.info(f"\n{'='*60}")
    logger.info(f"Prediqt Daily Pipeline  —  {start_time:%Y-%m-%d %H:%M}")
    logger.info(f"Index: {index} | Deep scan top: {deep_n} | Quick-only: {quick_only}")
    logger.info(f"{'='*60}\n")

    # 1. Fetch universe
    universe = get_full_universe()
    if index == "sp500":
        tickers = universe["sp500"]
    elif index == "nasdaq":
        tickers = universe["nasdaq100"]
    elif index == "dow":
        tickers = universe["dow30"]
    else:   # all
        tickers = universe["all"]

    logger.info(f"Universe: {len(tickers)} tickers ({index})")

    # 2. Quick screen
    if progress_cb:
        progress_cb(0, f"Quick-screening {len(tickers)} stocks…")

    screen_results = quick_screen_universe(tickers, progress_cb=progress_cb)
    logger.info(f"Quick screen: {len(screen_results)} valid results")

    if quick_only or not screen_results:
        return {
            "pipeline": "quick_only",
            "screened": len(screen_results),
            "deep_scanned": 0,
            "elapsed_min": round((datetime.now() - start_time).seconds / 60, 1),
            "top_picks": screen_results[:20],
        }

    # 3. Pick top N for deep scan
    # Balance: take top momentum stocks + some oversold (contrarian) + Dow 30 always
    top_momentum = [r["symbol"] for r in screen_results[:deep_n]]
    oversold     = [r["symbol"] for r in screen_results if r["signal"] == "OVERSOLD"][:10]
    dow_always   = get_dow30_tickers()[:15]   # always scan half the Dow for macro signal

    deep_symbols = list(dict.fromkeys(top_momentum + oversold + dow_always))[:deep_n]
    logger.info(f"Deep scan targets: {len(deep_symbols)} symbols")

    if progress_cb:
        progress_cb(20, f"Deep ML scan: {len(deep_symbols)} stocks…")

    # 4. Deep ML scan
    deep_summary = deep_scan(deep_symbols, data_period=data_period, progress_cb=progress_cb)

    elapsed = round((datetime.now() - start_time).seconds / 60, 1)
    pipeline_summary = {
        "pipeline": "full",
        "run_at":   start_time.isoformat(),
        "elapsed_min": elapsed,
        "universe_size": len(tickers),
        "screened":      len(screen_results),
        "deep_scanned":  deep_summary["logged"],
        "data_points_added": deep_summary["data_points_added"],
        "errors":        deep_summary["errors"],
        "top_picks":     screen_results[:20],
    }

    # Save pipeline summary
    summary_file = os.path.join(SCAN_RESULTS_DIR, "pipeline_summary.json")
    os.makedirs(SCAN_RESULTS_DIR, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(pipeline_summary, f, indent=2)

    logger.info(f"Pipeline complete in {elapsed} min. "
                f"{deep_summary['data_points_added']} new ML data points.")
    return pipeline_summary


def get_last_scan_summary() -> Optional[dict]:
    """Load the most recent pipeline summary."""
    summary_file = os.path.join(SCAN_RESULTS_DIR, "pipeline_summary.json")
    if not os.path.exists(summary_file):
        return None
    try:
        with open(summary_file) as f:
            return json.load(f)
    except Exception:
        return None


def get_last_screen_results(n: int = 50) -> list:
    """Return top N results from the most recent quick screen."""
    if not os.path.exists(SCAN_RESULTS_FILE):
        return []
    try:
        with open(SCAN_RESULTS_FILE) as f:
            data = json.load(f)
        return data.get("results", [])[:n]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediqt Daily Batch Scanner")
    parser.add_argument("--index",      default="all",   choices=["all","sp500","nasdaq","dow"])
    parser.add_argument("--deep-n",     default=50,      type=int, help="How many top stocks get full ML scan")
    parser.add_argument("--quick-only", action="store_true",       help="Run quick screen only, skip ML training")
    parser.add_argument("--data-period",default="5y",              help="Training data period (e.g. 5y, 3y)")
    args = parser.parse_args()

    summary = run_daily_pipeline(
        index=args.index,
        deep_n=args.deep_n,
        quick_only=args.quick_only,
        data_period=args.data_period,
    )

    print(f"\n{'='*50}")
    print(f"Pipeline complete.")
    print(f"  Elapsed:         {summary['elapsed_min']} min")
    print(f"  Stocks scanned:  {summary['screened']}")
    print(f"  ML predictions:  {summary['deep_scanned']}")
    print(f"  New data points: {summary['data_points_added']}")
    print(f"{'='*50}\n")
