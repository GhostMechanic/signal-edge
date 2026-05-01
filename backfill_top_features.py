"""
backfill_top_features.py
=========================
One-time script: walk every entry in prediction_log_v2.json and patch
horizons whose `top_features` is empty (typically older API-logged
predictions that ran when the API passed `feature_importance=None`).

For each empty horizon we:
  1. Load the symbol's trained model joblib from .models/
  2. Pull `selected_feats[horizon]` — the SHAP-ranked feature list
  3. Build the same {name, importance} shape the live logger writes,
     using rank-based synthetic importance (1.0, 0.9, 0.8, ...) so the
     ordering survives even though we don't have absolute SHAP values
     stored at log time.

Idempotent — already-populated horizons are skipped.
Writes a timestamped backup of the original log before saving.

Usage (from the Stock Bot directory):
    python backfill_top_features.py
    python backfill_top_features.py --dry-run    # print stats, don't write
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib


# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, ".predictions", "prediction_log_v2.json")
MODELS_DIR = os.path.join(SCRIPT_DIR, ".models")
BACKUP_FMT = os.path.join(
    SCRIPT_DIR, ".predictions", "prediction_log_v2.backup-{ts}.json"
)


def load_selected_feats(symbol: str) -> Optional[Dict[str, List[str]]]:
    """
    Load selected_feats from a symbol's model joblib. Returns None if the
    model file is missing or fails to deserialize. Returns {} if the
    model loaded fine but doesn't have a selected_feats attribute (very
    old format) — distinct from None so the caller can tell apart
    "model doesn't exist" vs "model exists but has no feature data."
    """
    path = os.path.join(MODELS_DIR, f"{symbol}_model.joblib")
    if not os.path.exists(path):
        return None
    try:
        data = joblib.load(path)
    except Exception as e:
        print(f"  [load failed] {symbol}: {e}")
        return None
    raw = data.get("selected_feats", None)
    if raw is None:
        return {}
    # Normalize: ensure values are list[str].
    out: Dict[str, List[str]] = {}
    for h, feats in (raw or {}).items():
        if not feats:
            continue
        out[h] = [f for f in list(feats) if isinstance(f, str) and f]
    return out


def synthetic_importance(features: List[str]) -> List[Dict[str, Any]]:
    """
    Build the {name, importance} list the prediction logger expects.
    Importance is rank-based since we no longer have the absolute SHAP
    magnitude — the model only stored the ordered selected_feats list.
    The logger / analytics layer only cares about ordering anyway.
    """
    return [
        {"name": f, "importance": round(max(0.1, 1.0 - i * 0.1), 4)}
        for i, f in enumerate(features[:10])
    ]


def backfill(dry_run: bool = False) -> None:
    if not os.path.exists(LOG_FILE):
        print(f"Log file not found: {LOG_FILE}")
        sys.exit(1)

    with open(LOG_FILE, "r") as f:
        entries = json.load(f)

    # ── Backup before mutating ────────────────────────────────────────────
    if not dry_run:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = BACKUP_FMT.format(ts=ts)
        shutil.copy(LOG_FILE, backup_path)
        print(f"Backup written to {backup_path}")

    # Per-symbol cache so we don't reload the same joblib for every horizon.
    feats_cache: Dict[str, Optional[Dict[str, List[str]]]] = {}

    stats = {
        "entries":                   len(entries),
        "horizons_total":            0,
        "horizons_already_filled":   0,
        "horizons_patched":          0,
        "horizons_no_model":         0,
        "horizons_model_empty":      0,
        "horizons_no_horizon_match": 0,
    }

    # Symbols whose model is missing — collect for the summary so the user
    # knows how much can't be backfilled.
    missing_symbols: set = set()

    for entry in entries:
        symbol = (entry.get("symbol", "") or "").upper().strip()
        if not symbol:
            continue

        for horizon, hdata in (entry.get("horizons") or {}).items():
            stats["horizons_total"] += 1

            existing = hdata.get("top_features") or []
            if isinstance(existing, list) and len(existing) > 0:
                stats["horizons_already_filled"] += 1
                continue

            # Lazy-load the model on first need.
            if symbol not in feats_cache:
                feats_cache[symbol] = load_selected_feats(symbol)
            selected = feats_cache[symbol]

            if selected is None:
                stats["horizons_no_model"] += 1
                missing_symbols.add(symbol)
                continue
            if not selected:
                stats["horizons_model_empty"] += 1
                continue

            feats = selected.get(horizon, [])
            if not feats:
                stats["horizons_no_horizon_match"] += 1
                continue

            hdata["top_features"] = synthetic_importance(feats)
            stats["horizons_patched"] += 1

    # ── Save ──────────────────────────────────────────────────────────────
    if not dry_run:
        with open(LOG_FILE, "w") as f:
            json.dump(entries, f, indent=2, default=str)

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("Backfill complete." + (" (dry run — no writes)" if dry_run else ""))
    print(f"  Entries scanned:                    {stats['entries']}")
    print(f"  Horizons total:                     {stats['horizons_total']}")
    print(f"  Already populated (skipped):        {stats['horizons_already_filled']}")
    print(f"  Patched:                            {stats['horizons_patched']}")
    print(f"  No model on disk:                   {stats['horizons_no_model']}")
    print(f"  Model exists but has no feats:      {stats['horizons_model_empty']}")
    print(f"  Model has no entry for horizon:     {stats['horizons_no_horizon_match']}")
    if missing_symbols:
        sample = ", ".join(sorted(missing_symbols)[:10])
        more = (
            f" (+{len(missing_symbols) - 10} more)"
            if len(missing_symbols) > 10 else ""
        )
        print(f"  Symbols with no model on disk:      {sample}{more}")
    print()
    print(
        "Tip: regenerate analytics by calling /api/analytics/per-regime "
        "after restart — the in-memory analytics cache is busted by the "
        "next live prediction OR by restarting the API server."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats but don't write changes to the log file.",
    )
    args = parser.parse_args()
    backfill(dry_run=args.dry_run)
