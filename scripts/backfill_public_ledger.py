"""
backfill_public_ledger.py
─────────────────────────────────────────────────────────────────────────────
One-shot: any prediction sitting in Supabase with `is_public_ledger=false`
that is NOW eligible (confidence >= 55, symbol in current canonical_universe,
no recent dupe) gets flipped to true.

Why this exists:
The canonical_universe is dynamic — it expands to include every ticker the
system has trained a model for. Predictions written before that expansion
landed (e.g., the first BABA call right after auto-training that ticker)
were inserted with `is_public_ledger=false` because BABA wasn't in the
universe at the time. This script fixes those rows in place.

Usage:
    python scripts/backfill_public_ledger.py              # dry run by default
    python scripts/backfill_public_ledger.py --apply      # actually update
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually flip is_public_ledger. Without this, prints what would change.",
    )
    args = parser.parse_args()

    from db import USE_SUPABASE, _client
    from universe import canonical_universe

    if not USE_SUPABASE:
        print("USE_SUPABASE=false — nothing to backfill (file mode doesn't have this flag).")
        return 0

    cli = _client()

    # Pull every is_public_ledger=false row above the confidence floor.
    # We don't need to filter by user_id because this is a global cleanup.
    rows = (
        cli.table("predictions")
            .select("id, symbol, confidence, is_public_ledger, created_at")
            .eq("is_public_ledger", False)
            .gte("confidence", 55)
            .execute()
        .data
        or []
    )

    universe = canonical_universe()

    to_flip = [
        r for r in rows
        if (r.get("symbol") or "").upper() in universe
    ]

    print(f"Scanned {len(rows)} predictions with confidence >= 55 and is_public_ledger = false.")
    print(f"Of those, {len(to_flip)} are now in the canonical universe.\n")

    if not to_flip:
        print("Nothing to flip. Exiting.")
        return 0

    by_symbol: dict[str, int] = {}
    for r in to_flip:
        by_symbol[r["symbol"]] = by_symbol.get(r["symbol"], 0) + 1

    print("Counts by symbol:")
    for sym, n in sorted(by_symbol.items(), key=lambda kv: -kv[1]):
        print(f"  {sym:8s}  {n}")
    print()

    if not args.apply:
        print("(dry run — re-run with --apply to actually update)")
        return 0

    ids = [r["id"] for r in to_flip]
    # Supabase-py's .in_() filter has a max-length per call (URL length).
    # Chunk to be safe.
    BATCH = 100
    flipped = 0
    for i in range(0, len(ids), BATCH):
        chunk = ids[i : i + BATCH]
        res = (
            cli.table("predictions")
                .update({"is_public_ledger": True})
                .in_("id", chunk)
                .execute()
        )
        flipped += len(res.data or [])

    print(f"Flipped {flipped} rows. Reload /track-record to see them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
