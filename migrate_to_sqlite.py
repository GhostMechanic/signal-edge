"""
migrate_to_sqlite.py
====================
One-time migration: copy every entry from prediction_log_v2.json into
the SQLite-backed prediction store. Idempotent — re-running won't
duplicate entries because insert_prediction's upsert-by-(symbol, date)
matches the legacy JSON dedup behavior.

Usage (from Stock Bot directory):
    python migrate_to_sqlite.py            # actually write
    python migrate_to_sqlite.py --dry-run  # just count, don't write

Safe by design:
  - Doesn't touch the JSON file. After migration the JSON file remains
    on disk as a historical reference.
  - Each entry is inserted in its own transaction; a failure mid-loop
    leaves successfully-migrated rows intact.
  - Reports any rows that couldn't be migrated (with reason) at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import prediction_store


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, ".predictions", "prediction_log_v2.json")


def main(dry_run: bool = False) -> None:
    if not os.path.exists(JSON_PATH):
        print(f"JSON log not found: {JSON_PATH}")
        sys.exit(1)

    with open(JSON_PATH, "r") as f:
        entries = json.load(f)
    print(f"Loaded {len(entries)} entries from JSON.")

    prediction_store.init_db()
    pre_count = prediction_store.count_predictions()
    print(f"SQLite already contains {pre_count} predictions.")

    if dry_run:
        print()
        print("Dry run — would migrate up to {} entries.".format(len(entries)))
        print("Pass without --dry-run to commit.")
        return

    inserted = 0
    skipped: list = []
    for i, entry in enumerate(entries):
        try:
            if not isinstance(entry, dict) or not entry.get("prediction_id"):
                skipped.append((i, "missing prediction_id"))
                continue
            prediction_store.insert_prediction(entry)
            inserted += 1
        except Exception as e:
            skipped.append((i, f"{type(e).__name__}: {e}"))

    post_count = prediction_store.count_predictions()
    delta = post_count - pre_count

    print()
    print("Migration complete.")
    print(f"  Loaded from JSON:     {len(entries)}")
    print(f"  Insert calls succeeded: {inserted}")
    print(f"  Skipped/errored:      {len(skipped)}")
    print(f"  SQLite count before:  {pre_count}")
    print(f"  SQLite count after:   {post_count}")
    print(f"  Net new in DB:        {delta}")

    if skipped:
        print()
        print("Skipped entries:")
        for idx, reason in skipped[:20]:
            print(f"  [{idx}] {reason}")
        if len(skipped) > 20:
            print(f"  …and {len(skipped) - 20} more")

    health = prediction_store.health_check()
    print()
    print(f"DB path:       {health['db_path']}")
    print(f"Predictions:   {health['predictions']}")
    print(f"Horizons:      {health['horizons']}")
    print(f"Scores:        {health['scores']}")
    print(f"Earliest:      {health['earliest']}")
    print(f"Latest:        {health['latest']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write anything; just report what the migration would do.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
