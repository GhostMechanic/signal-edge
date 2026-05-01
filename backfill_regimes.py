"""
backfill_regimes.py
===================
One-shot: tag every prediction in the SQLite store with a real regime
label (Bull / Sideways / Bear), computed historically from SPY + VIX as
of each prediction's timestamp.

Why this exists:
  - The 54 entries recovered from prediction_log.json (v1) had no regime
    field, so they all defaulted to "Unknown".
  - The TSM entry logged through the live API on 2026-04-25 also landed
    as "Unknown" because of an isinstance(str) bug on
    `predictor.regime_cache` (it's a dict, not a string). That bug is
    fixed in api/main.py — but the existing rows still need backfilling.
  - With every entry tagged Unknown, the "By Market Backdrop" section
    on Track Record is starved of data: every regime card has zero calls
    and the section either renders empty or with visual artifacts.

How it works:
  1. Pulls SPY + VIX history once (5 years back from today).
  2. For each prediction, slices the history up to its prediction
     timestamp and calls regime_detector.detect_regime(...).
  3. Writes the resulting label back into prediction_store via a single
     UPDATE per row (cheap, indexed by prediction_id).

Idempotent: re-running re-evaluates regimes against the same historical
data and arrives at the same labels — safe to re-run if you tune the
classifier.

Usage (from Stock Bot directory):
    python backfill_regimes.py            # actually write
    python backfill_regimes.py --dry-run  # preview, no writes
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

import prediction_store
from regime_detector import detect_regime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_spy_vix(years_back: int = 5) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Fetch SPY + VIX once at the start. The classifier needs:
      - SPY OHLC dataframe (`spy_df`)
      - SPY adjusted close series (`spy_close`)
      - VIX close series (`vix_close`)
    Slicing each per-prediction is then an O(1) date filter on the cached
    series; no per-row network round-trips.
    """
    print(f"[backfill] fetching SPY + VIX history ({years_back}y) …", flush=True)
    spy = yf.download("SPY", period=f"{years_back}y", auto_adjust=False, progress=False)
    vix = yf.download("^VIX", period=f"{years_back}y", auto_adjust=False, progress=False)

    if spy.empty or vix.empty:
        raise RuntimeError("SPY/VIX download came back empty — aborting backfill.")

    # yfinance can return either ('Close', 'SPY') multiindex columns or
    # just 'Close' depending on how it's called. Normalize to single-level.
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] for c in vix.columns]

    spy_close = spy["Close"]
    vix_close = vix["Close"]
    print(
        f"[backfill] SPY rows={len(spy)}, VIX rows={len(vix)}; "
        f"range {spy.index.min().date()} → {spy.index.max().date()}",
        flush=True,
    )
    return spy, spy_close, vix_close


def classify_at(
    ts_iso: str,
    spy: pd.DataFrame,
    spy_close: pd.Series,
    vix_close: pd.Series,
) -> Optional[str]:
    """
    Classify the regime for a single prediction by slicing the cached
    SPY/VIX history up to ts_iso and asking the detector.

    Returns the regime label (Bull / Sideways / Bear), or None if the
    historical window is too thin to classify (which the detector itself
    decides — we just defer to it).
    """
    try:
        as_of = pd.Timestamp(ts_iso[:10])
    except Exception:
        return None

    spy_slice = spy[spy.index <= as_of]
    spy_close_slice = spy_close[spy_close.index <= as_of]
    vix_close_slice = vix_close[vix_close.index <= as_of]

    if len(spy_slice) < 60:
        # The classifier needs at least ~60 trading days of history to
        # compute the trend / vol bands it gates on. Anything thinner
        # we leave as Unknown.
        return None

    try:
        out = detect_regime(spy_slice, spy_close_slice, vix_close_slice)
    except Exception as e:
        print(f"  detect_regime failed at {ts_iso}: {type(e).__name__}: {e}")
        return None

    if isinstance(out, dict):
        return out.get("label") or None
    if isinstance(out, str):
        return out
    return None


def main(dry_run: bool = False) -> None:
    prediction_store.init_db()
    db_path = prediction_store.DB_PATH

    # Pull all predictions + their current regime tag.
    with sqlite3.connect(db_path) as conn:
        rows = list(
            conn.execute(
                "SELECT prediction_id, symbol, timestamp, regime "
                "FROM predictions ORDER BY timestamp"
            )
        )
    print(f"[backfill] {len(rows)} predictions in DB")

    spy, spy_close, vix_close = fetch_spy_vix(years_back=5)

    plan: List[Dict[str, Any]] = []
    unchanged = 0
    unable = 0
    for pid, symbol, ts_iso, current in rows:
        new_label = classify_at(ts_iso, spy, spy_close, vix_close)
        if new_label is None:
            unable += 1
            continue
        if new_label == current:
            unchanged += 1
            continue
        plan.append(
            {
                "prediction_id": pid,
                "symbol": symbol,
                "timestamp": ts_iso,
                "old": current,
                "new": new_label,
            }
        )

    # Distribution preview
    dist: Dict[str, int] = {}
    for p in plan:
        dist[p["new"]] = dist.get(p["new"], 0) + 1
    print()
    print(f"[backfill] would update:    {len(plan)}")
    print(f"[backfill] already correct: {unchanged}")
    print(f"[backfill] unable to tag:   {unable} (kept as 'Unknown')")
    print()
    print("[backfill] new regime distribution (only changed rows):")
    for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<10} {v}")

    if not plan:
        print("\n[backfill] nothing to do — all rows are up-to-date.")
        return

    print()
    print("[backfill] sample of what will change:")
    for p in plan[:10]:
        print(
            f"  {p['symbol']:<6} {p['timestamp'][:10]} "
            f"{p['old']:<10} → {p['new']}"
        )
    if len(plan) > 10:
        print(f"  …and {len(plan) - 10} more")

    if dry_run:
        print("\n[backfill] dry run — nothing written. Pass without --dry-run to commit.")
        return

    # Single transaction — commit-or-rollback at the end.
    print()
    print(f"[backfill] writing {len(plan)} updates …", flush=True)
    with sqlite3.connect(db_path) as conn:
        try:
            for p in plan:
                conn.execute(
                    "UPDATE predictions SET regime = ? WHERE prediction_id = ?",
                    (p["new"], p["prediction_id"]),
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[backfill] FAILED, rolled back: {type(e).__name__}: {e}")
            sys.exit(1)

    # Verify post-write distribution
    with sqlite3.connect(db_path) as conn:
        post = list(
            conn.execute(
                "SELECT regime, COUNT(*) AS n FROM predictions GROUP BY regime ORDER BY n DESC"
            )
        )
    print()
    print("[backfill] done. Final regime distribution:")
    for label, count in post:
        print(f"  {label:<10} {count}")
    print()
    print(
        "[backfill] next: restart uvicorn so the analytics cache is\n"
        "          dropped, then hard-refresh Track Record."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change but don't write to the DB.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
