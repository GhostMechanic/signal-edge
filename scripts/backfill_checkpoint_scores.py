"""
backfill_checkpoint_scores.py
─────────────────────────────
One-shot historical backfill for the new checkpoint scoring scheme
(see methodology § 4.3 post-migration 0017).

What this does
──────────────
Walks every existing prediction in the `predictions` table and, for
each one, computes per-interval directional scores under the new
scheme (close > entry for Bullish, close < entry for Bearish, at each
scheduled interval since prediction issue). Writes the results to
prediction_checkpoint_scores and updates the parent prediction's
rating_checkpoint to reflect the latest fired interval.

Intervals are the same schedule scoring_worker.HORIZON_INTERVALS uses
in steady state — this script just runs the scoring logic in a single
pass across all historical predictions instead of waiting for the cron
to catch up over weeks.

Idempotency
───────────
Safe to re-run. The prediction_checkpoint_scores table has
UNIQUE (prediction_id, interval_days), so re-attempted inserts no-op
at the DB layer. A second run only ever touches predictions that
gained new fire-able intervals since the last run (rare in practice
once the steady-state cron is live).

Usage
─────
    python -m scripts.backfill_checkpoint_scores            # run for real
    python -m scripts.backfill_checkpoint_scores --dry-run  # show counts only
    python -m scripts.backfill_checkpoint_scores --limit 50 # cap row count

Output is a per-prediction tally written to stdout plus a summary at
the end. yfinance is rate-limited — expect ~1s per prediction.

Prerequisites
─────────────
USE_SUPABASE=true plus SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY in the
environment (same setup as the scoring worker). The service-role key
is required to insert into prediction_checkpoint_scores (RLS allows
only service_role writes, per migration 0017).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

# Reuse the steady-state scoring logic so backfill semantics and live
# semantics can never drift apart — if HORIZON_INTERVALS or the
# directional check formula changes, both paths inherit the change.
from scoring_worker import (
    HORIZON_INTERVALS,
    _checkpoint_intervals_for_horizon,
    _close_at_or_after,
    _parse_iso,
)

logger = logging.getLogger(__name__)


def backfill(
    *,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> dict:
    """Walk every prediction, compute checkpoint scores for all elapsed
    intervals, write to prediction_checkpoint_scores, and update each
    parent's rating_checkpoint.

    Returns a summary dict for logs.
    """
    from db import _service_client

    client = _service_client()
    now = datetime.now(timezone.utc)

    # Pull every non-Neutral prediction. Neutral predictions skip the
    # checkpoint axis entirely (no directional check defined).
    query = (
        client.table("predictions")
              .select(
                  "id, symbol, direction, entry_price, "
                  "created_at, horizon"
              )
              .neq("direction", "Neutral")
    )
    if limit is not None:
        query = query.limit(limit)
    preds = query.execute().data or []

    summary = {
        "checked":           len(preds),
        "predictions_with_new_scores": 0,
        "scores_inserted":   0,
        "rating_updated":    0,
        "skipped_no_entry":  0,
        "skipped_no_data":   0,
        "errors":            [],
    }

    for i, pred in enumerate(preds, 1):
        pred_id = pred["id"]
        try:
            symbol = pred["symbol"]
            direction = (pred.get("direction") or "").strip()
            entry_price = pred.get("entry_price")

            if not entry_price or float(entry_price) <= 0:
                summary["skipped_no_entry"] += 1
                continue
            entry_price = float(entry_price)
            issued_at = _parse_iso(pred["created_at"])
            intervals = _checkpoint_intervals_for_horizon(pred.get("horizon", ""))

            # Elapsed intervals only — future intervals will fire via
            # tick_checkpoints in steady state.
            from datetime import timedelta
            due = [
                d for d in intervals
                if issued_at + timedelta(days=d) <= now
            ]
            if not due:
                continue

            # What's already in the table for this prediction?
            existing = (
                client.table("prediction_checkpoint_scores")
                      .select("interval_days")
                      .eq("prediction_id", pred_id)
                      .execute()
            ).data or []
            already_scored = {row["interval_days"] for row in existing}

            to_fire = [d for d in due if d not in already_scored]
            if not to_fire:
                continue

            fired_this_pred = 0
            for interval_days in to_fire:
                target_dt = issued_at + timedelta(days=interval_days)
                close_price = _close_at_or_after(symbol, target_dt)

                if close_price is None:
                    summary["skipped_no_data"] += 1
                    continue

                if direction == "Bullish":
                    status = "hit" if close_price > entry_price else "miss"
                elif direction == "Bearish":
                    status = "hit" if close_price < entry_price else "miss"
                else:
                    continue  # defensive — Neutral filtered upstream

                if dry_run:
                    fired_this_pred += 1
                    continue

                client.table("prediction_checkpoint_scores").insert({
                    "prediction_id": pred_id,
                    "interval_days": interval_days,
                    "scored_at":     now.isoformat(),
                    "status":        status,
                    "close_price":   close_price,
                }).execute()
                fired_this_pred += 1

            if fired_this_pred > 0:
                summary["predictions_with_new_scores"] += 1
                summary["scores_inserted"] += fired_this_pred

                if not dry_run:
                    latest = (
                        client.table("prediction_checkpoint_scores")
                              .select("status")
                              .eq("prediction_id", pred_id)
                              .order("interval_days", desc=True)
                              .limit(1)
                              .execute()
                    ).data
                    if latest:
                        client.table("predictions").update({
                            "rating_checkpoint": latest[0]["status"],
                        }).eq("id", pred_id).execute()
                        summary["rating_updated"] += 1

            # Per-prediction progress log every 25 rows — keeps the
            # backfill watchable without spamming stdout.
            if i % 25 == 0:
                print(
                    f"  ...processed {i}/{len(preds)} predictions; "
                    f"{summary['scores_inserted']} scores so far"
                )

        except Exception as exc:
            logger.exception("backfill failed for %s: %s", pred_id, exc)
            summary["errors"].append({
                "prediction_id": pred_id,
                "error":         str(exc)[:200],
            })

    return summary


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill historical predictions to the new "
                    "checkpoint scoring scheme (migration 0017)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute scores and report counts; don't write to Supabase.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N predictions. Useful for spot-"
             "checking the backfill output before running the full pass.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"backfill_checkpoint_scores — {mode}")
    print(f"horizon intervals in scope: {HORIZON_INTERVALS}")
    if args.limit is not None:
        print(f"limit: {args.limit} predictions")
    print()

    result = backfill(dry_run=args.dry_run, limit=args.limit)

    print()
    print("─── summary ─────────────────────────────")
    print(f"  predictions checked            : {result['checked']}")
    print(f"  predictions with new scores    : {result['predictions_with_new_scores']}")
    print(f"  total score rows inserted      : {result['scores_inserted']}")
    print(f"  predictions w/ rating updated  : {result['rating_updated']}")
    print(f"  skipped (no entry price)       : {result['skipped_no_entry']}")
    print(f"  skipped (no yfinance data)     : {result['skipped_no_data']}")
    print(f"  errors                         : {len(result['errors'])}")
    if result["errors"]:
        print()
        print("first 5 errors:")
        for err in result["errors"][:5]:
            print(f"  • {err['prediction_id']}: {err['error']}")
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    sys.exit(_cli())
