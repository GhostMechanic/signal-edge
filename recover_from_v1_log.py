"""
recover_from_v1_log.py
======================
One-time recovery: rebuild prediction_log_v2.json from the legacy
prediction_log.json (v1 format) when the v2 file has been wiped.

The v1 format predates the per-interval scoring scheme, so each v1 entry
only has a single final-score per horizon. We map that to one entry in
the v2 `scores` dict at the horizon's natural interval (1 Week → 7d,
1 Month → 30d, etc.). This restores Expiration Win Rate and gives some
data to Checkpoint Win Rate, but Per-interval Checkpoint detail (1d/3d
intermediate scoring) won't be present — that data was never recorded
in v1 and can't be reconstructed.

Idempotent + safe:
  - Backs up the current v2 file before writing
  - Skips v1 entries whose (symbol, date) collides with an existing v2
    entry, so re-running is a no-op
  - Generates deterministic 8-char UUIDs from (symbol, timestamp) so
    re-running produces the same prediction_ids

Usage (from Stock Bot directory):
    python recover_from_v1_log.py            # actually write
    python recover_from_v1_log.py --dry-run  # preview, no writes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V1_FILE   = os.path.join(SCRIPT_DIR, ".predictions", "prediction_log.json")
V2_FILE   = os.path.join(SCRIPT_DIR, ".predictions", "prediction_log_v2.json")
BACKUP_FMT = os.path.join(
    SCRIPT_DIR, ".predictions", "prediction_log_v2.preretrieve-{ts}.json"
)

# Map a v1 horizon name to the v2 score-interval key it should land at.
# v1 only stored a single final score per horizon, so each horizon
# contributes exactly one entry to its natural interval.
HORIZON_TO_INTERVAL = {
    "3 Day":     "3d",
    "1 Week":    "7d",
    "1 Month":   "30d",
    "1 Quarter": "90d",
    "1 Year":    "365d",
}

HORIZON_TO_DAYS = {
    "3 Day":     3,
    "1 Week":    7,
    "1 Month":   30,
    "1 Quarter": 90,
    "1 Year":    365,
}


def deterministic_id(symbol: str, timestamp_iso: str) -> str:
    """8-char hex digest derived from (symbol, timestamp). Same inputs →
    same id, so re-running this script produces the same prediction_ids
    and the v2 dedupe paths stay consistent."""
    h = hashlib.sha1(f"{symbol}|{timestamp_iso}".encode("utf-8")).hexdigest()
    return h[:8]


def transform_v1_entry(v1: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a single v1 entry into a v2-shaped entry. Returns None if
    the entry is malformed (missing symbol or timestamp)."""
    symbol = (v1.get("symbol", "") or "").upper().strip()
    timestamp = v1.get("timestamp", "")
    if not symbol or not timestamp:
        return None

    pred_date = timestamp[:10]
    try:
        pred_dt = datetime.fromisoformat(timestamp)
    except ValueError:
        pred_dt = None

    v2_horizons: Dict[str, Any] = {}
    for h_name, hdata in (v1.get("horizons") or {}).items():
        if not isinstance(hdata, dict):
            continue
        interval = HORIZON_TO_INTERVAL.get(h_name)
        days = HORIZON_TO_DAYS.get(h_name)
        if not interval or not days:
            continue

        out: Dict[str, Any] = {
            "predicted_return":   hdata.get("predicted_return", 0),
            "predicted_price":    hdata.get("predicted_price", 0),
            "current_price":      hdata.get("current_price", 0),
            "confidence":         hdata.get("confidence", 0),
            "ensemble_agreement": 0,
            "val_dir_accuracy":   0,
            "direction":          hdata.get("direction", "up"),
            # No top features in v1 — leave empty. Populates over time as
            # new predictions land via the API (which now writes them).
            "top_features":       [],
            "scores":             {},
            "final_scored":       False,
            "final_correct":      None,
        }

        scored = bool(hdata.get("scored"))
        if scored:
            actual_price  = hdata.get("actual_price")
            actual_return = hdata.get("actual_return")
            correct       = hdata.get("direction_correct")

            scored_at = (
                (pred_dt + timedelta(days=days)).isoformat()
                if pred_dt is not None else timestamp
            )

            out["scores"][interval] = {
                "actual_price":      actual_price,
                "actual_return":     actual_return,
                "direction_correct": bool(correct) if correct is not None else False,
                "scored_at":         scored_at,
                "days_actual":       days,
            }
            out["final_scored"] = True
            out["final_correct"] = bool(correct) if correct is not None else False

        v2_horizons[h_name] = out

    if not v2_horizons:
        return None

    return {
        "prediction_id":  deterministic_id(symbol, timestamp),
        "symbol":         symbol,
        "timestamp":      timestamp,
        "model_version":  "1.0",  # v1 didn't track this
        "regime":         "Unknown",  # v1 didn't track this
        "horizons":       v2_horizons,
    }


def merge(existing_v2: List[Dict[str, Any]], new_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Combine existing v2 entries with newly-transformed v1 entries.
    Dedupes on (symbol, timestamp[:10]) — matches the dedup logic in
    prediction_logger_v2.log_prediction_v2 — preserving any existing v2
    entry over the v1 version (since v2 has richer data when it overlaps).
    """
    by_key: Dict[tuple, Dict[str, Any]] = {}
    for e in existing_v2:
        sym = (e.get("symbol", "") or "").upper()
        ts = (e.get("timestamp", "") or "")[:10]
        if sym and ts:
            by_key[(sym, ts)] = e

    added = 0
    for e in new_entries:
        sym = (e.get("symbol", "") or "").upper()
        ts = (e.get("timestamp", "") or "")[:10]
        if not sym or not ts:
            continue
        if (sym, ts) in by_key:
            continue  # don't overwrite existing v2 entries
        by_key[(sym, ts)] = e
        added += 1

    merged = list(by_key.values())
    # Sort chronologically so the file stays human-readable when grepping.
    merged.sort(key=lambda x: (x.get("timestamp", ""), x.get("symbol", "")))
    return merged, added


def main(dry_run: bool = False) -> None:
    if not os.path.exists(V1_FILE):
        print(f"V1 log not found: {V1_FILE}")
        sys.exit(1)
    if not os.path.exists(V2_FILE):
        print(f"V2 log not found: {V2_FILE}")
        sys.exit(1)

    with open(V1_FILE, "r") as f:
        v1_entries = json.load(f)
    with open(V2_FILE, "r") as f:
        v2_entries = json.load(f)

    print(f"V1 entries:     {len(v1_entries)}")
    print(f"V2 entries:     {len(v2_entries)}")

    # Transform v1 → v2 shape
    transformed: List[Dict[str, Any]] = []
    skipped = 0
    for v1 in v1_entries:
        out = transform_v1_entry(v1)
        if out is None:
            skipped += 1
            continue
        transformed.append(out)
    print(f"Transformed:    {len(transformed)}")
    if skipped:
        print(f"Skipped (bad):  {skipped}")

    # Merge into existing v2
    merged, added = merge(v2_entries, transformed)
    print(f"Will add:       {added} new entries")
    print(f"Merged total:   {len(merged)}")

    if dry_run:
        print()
        print("Dry run — nothing written. Pass without --dry-run to commit.")
        return

    # Backup the current v2 before mutating
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = BACKUP_FMT.format(ts=ts)
    shutil.copy(V2_FILE, backup_path)
    print(f"Backup written: {backup_path}")

    with open(V2_FILE, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"V2 log written: {V2_FILE}")
    print()
    print(
        "Next steps:\n"
        "  1. Restart the API server so the in-memory analytics cache is\n"
        "     dropped: kill / re-run uvicorn (Ctrl-C + restart).\n"
        "  2. Reload Track Record in the browser. The honesty cards,\n"
        "     per-horizon grid, and prediction log should populate from\n"
        "     the recovered data within a few seconds.\n"
        "  3. Per-interval Checkpoint detail (1d/3d intermediate scores)\n"
        "     will remain blank for v1-recovered entries — that data\n"
        "     wasn't recorded in v1 and can't be reconstructed. New\n"
        "     predictions logged going forward will have full detail."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen but don't write any files.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
