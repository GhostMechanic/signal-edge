"""
etl_sqlite_to_supabase.py
─────────────────────────
One-shot ETL that migrates the legacy SQLite predictions into Supabase.

Per data-model anchor § 8:
  • Each SQLite asking-event becomes ONE Supabase predictions row, keyed
    on the canonical horizon (1 Month preferred, fallback per priority).
  • user_id = NULL (legacy era — no accounts existed at insert time).
  • traded = false. We do NOT retroactively claim the model would have
    traded these — that would invalidate the no-cherry-picking guarantee.
  • verdict mapped from final_correct: 1→HIT, 0→MISSED, NULL→OPEN.
    PARTIAL is unavailable historically (we don't have intra-window OHLC
    traces to reconstruct it), so all settled rows go HIT or MISSED only.
  • Fresh UUIDs are minted; the legacy 8-char SQLite IDs are dropped
    (they don't fit the Supabase uuid type).
  • is_public_ledger applied per the same Python-side rules as new
    inserts (canonical universe + confidence ≥ 55).

Idempotency: this script issues plain INSERT (not UPSERT). Running it
twice will produce duplicate rows. Run once. If you need to re-run,
truncate the predictions rows where user_id IS NULL first.

Usage:

    USE_SUPABASE=true \\
    SUPABASE_URL=https://… \\
    SUPABASE_SERVICE_ROLE_KEY=… \\
    python scripts/etl_sqlite_to_supabase.py [--dry-run]

The --dry-run flag prints what would be inserted without hitting
Supabase. Use it once before the real run to confirm row counts and
field mappings look right.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

# Add the project root to sys.path so this script works whether you run
# it from the project root or from scripts/.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned inserts; don't write to Supabase."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Migrate only the first N predictions (smoke testing)."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Load .env before reading USE_SUPABASE — db.py also loads it on its
    # imports, but we check the env var before importing db, so we have
    # to load it ourselves first or the check fails even when .env is set
    # correctly.
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

    # Sanity checks before anything else.
    if not args.dry_run and os.getenv("USE_SUPABASE", "").lower() not in ("1", "true", "yes", "on"):
        print("USE_SUPABASE=true is required for non-dry-run mode.", file=sys.stderr)
        return 2

    # Imports happen here (not at module top) so --dry-run works even if
    # supabase-py is missing.
    import prediction_store
    from db import (
        _HORIZON_NAME_TO_CODE,
        _HORIZON_CODE_TO_DAYS,
        _pick_canonical_horizon,
        _is_public_ledger,
    )

    # Pull the SQLite predictions.
    sqlite_rows = prediction_store.get_all_predictions()
    if args.limit:
        sqlite_rows = sqlite_rows[: args.limit]

    print(f"SQLite rows to migrate: {len(sqlite_rows)}")
    if not sqlite_rows:
        print("Nothing to migrate. Exiting.")
        return 0

    rows_to_insert: list[dict] = []
    skipped: list[dict] = []

    for entry in sqlite_rows:
        try:
            row = _build_supabase_row(entry, _HORIZON_NAME_TO_CODE,
                                      _HORIZON_CODE_TO_DAYS,
                                      _pick_canonical_horizon,
                                      _is_public_ledger,
                                      dry_run=args.dry_run)
            if row is None:
                skipped.append({
                    "prediction_id": entry.get("prediction_id"),
                    "reason":        "no usable canonical horizon",
                })
                continue
            rows_to_insert.append(row)
        except Exception as exc:
            logger.exception("ETL failed for prediction %s",
                             entry.get("prediction_id"))
            skipped.append({
                "prediction_id": entry.get("prediction_id"),
                "reason":        f"{type(exc).__name__}: {exc}",
            })

    print(f"Built {len(rows_to_insert)} rows. Skipped {len(skipped)}.")
    if skipped:
        for s in skipped[:5]:
            print(f"  skip: {s}")
        if len(skipped) > 5:
            print(f"  …and {len(skipped) - 5} more")

    # Verdict / traded summary for the operator.
    verdicts = {}
    for r in rows_to_insert:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    print(f"Verdict distribution: {verdicts}")
    print(f"Public ledger eligible: "
          f"{sum(1 for r in rows_to_insert if r.get('is_public_ledger'))} / {len(rows_to_insert)}")

    if args.dry_run:
        print("\n--dry-run: not writing to Supabase. Sample row:")
        if rows_to_insert:
            sample = dict(rows_to_insert[0])
            for k, v in list(sample.items()):
                if isinstance(v, str) and len(v) > 60:
                    sample[k] = v[:60] + "…"
            for k in sorted(sample.keys()):
                print(f"  {k:>22}: {sample[k]!r}")
        return 0

    # Real run.
    from db import _service_client
    client = _service_client()

    inserted = 0
    failures: list[dict] = []
    for row in rows_to_insert:
        try:
            client.table("predictions").insert(row).execute()
            inserted += 1
        except Exception as exc:
            failures.append({
                "prediction_id": row.get("id"),
                "symbol":        row.get("symbol"),
                "error":         f"{type(exc).__name__}: {str(exc)[:120]}",
            })
            logger.exception("Insert failed for %s", row.get("id"))

    print(f"\nInserted: {inserted}")
    print(f"Failed:   {len(failures)}")
    for f in failures[:5]:
        print(f"  fail: {f}")
    return 0 if not failures else 1


def _build_supabase_row(
    entry: dict,
    horizon_name_to_code: dict,
    horizon_code_to_days: dict,
    pick_canonical_horizon,
    is_public_ledger_fn,
    *,
    dry_run: bool,
) -> Optional[dict]:
    """Map one SQLite predictions row to one Supabase predictions row.
    Returns None if the entry has no usable canonical horizon."""
    canonical = pick_canonical_horizon(entry)
    if canonical is None:
        return None

    h = entry["horizons"][canonical]
    horizon_code = horizon_name_to_code[canonical]

    # Direction from sign of predicted_return.
    pred_return = float(h.get("predicted_return", 0) or 0)
    if pred_return > 0:
        direction = "Bullish"
    elif pred_return < 0:
        direction = "Bearish"
    else:
        direction = "Neutral"

    confidence = float(h.get("confidence", 0) or 0)

    # Verdict from final_scored + final_correct (lossy — see § 8 of the
    # data-model anchor).
    final_scored  = bool(h.get("final_scored"))
    final_correct = h.get("final_correct")
    if not final_scored or final_correct is None:
        verdict           = "OPEN"
        rating_target     = "pending"
        rating_checkpoint = "pending"
        rating_expiration = "pending"
    elif int(final_correct) == 1:
        verdict           = "HIT"
        rating_target     = "hit"
        rating_checkpoint = "hit"
        rating_expiration = "hit"
    else:
        verdict           = "MISSED"
        rating_target     = "miss"
        rating_checkpoint = "miss"
        rating_expiration = "miss"

    # Horizon endpoints. starts_at = the SQLite timestamp; ends_at =
    # starts_at + the horizon's standard interval.
    timestamp_iso = entry.get("timestamp")
    try:
        starts_at = datetime.fromisoformat(str(timestamp_iso).replace("Z", "+00:00"))
        if starts_at.tzinfo is None:
            starts_at = starts_at.replace(tzinfo=timezone.utc)
    except Exception:
        starts_at = datetime.now(timezone.utc)
    ends_at = starts_at + timedelta(days=horizon_code_to_days[horizon_code])

    # is_public_ledger — same Python-side rules as live inserts, except
    # we skip the dedupe DB query in dry-run mode (no client available).
    if dry_run:
        # Approximate: confidence floor + canonical universe.
        try:
            from universe import canonical_universe
            in_universe = entry["symbol"].strip().upper() in canonical_universe()
        except Exception:
            in_universe = False
        is_public = (confidence >= 55) and in_universe
    else:
        # Real run: legacy rows have user_id=NULL, so the dedupe-by-user
        # query in _is_public_ledger would always pass. The function takes
        # a user_id arg though, so we hand it a sentinel UUID that won't
        # match anything.
        is_public = is_public_ledger_fn(
            symbol       = entry["symbol"],
            confidence   = confidence,
            horizon_code = horizon_code,
            user_id      = "00000000-0000-0000-0000-000000000000",
        )

    row = {
        "id":                 str(uuid.uuid4()),
        "user_id":            None,                       # legacy era
        "symbol":             entry["symbol"],
        "horizon":            horizon_code,
        "predicted_return":   round(pred_return * 100, 4),
        "predicted_price":    round(float(h.get("predicted_price", 0) or 0), 4),
        "confidence":         round(confidence, 2),
        "direction":          direction,
        "model_version":      entry.get("model_version") or "1.0",
        "regime":             entry.get("regime") or "Unknown",
        "is_public_ledger":   is_public,
        "horizon_starts_at":  starts_at.isoformat(),
        "horizon_ends_at":    ends_at.isoformat(),
        "created_at":         starts_at.isoformat(),     # backdate to original
        "verdict":            verdict,
        "rating_target":      rating_target,
        "rating_checkpoint":  rating_checkpoint,
        "rating_expiration":  rating_expiration,
        "traded":             False,                     # § 8: never retroactive
        "entry_price":        None,
        "stop_price":         None,
        "target_price":       None,
    }
    # Drop None values so Postgres defaults apply (notably created_at —
    # we override it above to backdate, but if parsing failed we want
    # the default now()).
    return {k: v for k, v in row.items() if v is not None}


if __name__ == "__main__":
    sys.exit(main())
