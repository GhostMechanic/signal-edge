"""
debug_baba.py
─────────────────────────────────────────────────────────────────────────────
Pulls every BABA prediction from Supabase and prints the fields that matter
for whether it shows up on the public ledger:
  • confidence (must be >= 55)
  • is_public_ledger (must be true)
  • symbol in canonical_universe()
  • verdict / created_at / horizon for context

Usage:  python scripts/debug_baba.py [SYMBOL]
        SYMBOL defaults to BABA.
"""

import sys


def main() -> int:
    sym = (sys.argv[1] if len(sys.argv) > 1 else "BABA").upper()

    from db import USE_SUPABASE, _client
    from universe import canonical_universe

    if not USE_SUPABASE:
        print("USE_SUPABASE=false — backend isn't reading from Supabase yet.")
        return 1

    universe = canonical_universe()
    in_universe = sym in universe
    print(f"Symbol:                   {sym}")
    print(f"In canonical_universe():  {in_universe}")
    print(f"Universe size:            {len(universe)}")
    print()

    cli = _client()
    rows = (
        cli.table("predictions")
            .select(
                "id, symbol, confidence, direction, horizon, verdict, "
                "is_public_ledger, traded, created_at, user_id"
            )
            .eq("symbol", sym)
            .order("created_at", desc=True)
            .execute()
        .data
        or []
    )

    if not rows:
        print(f"No predictions in Supabase for {sym}.")
        print("  → That means the predict endpoint never wrote a row for it.")
        print("    Likely cause: every horizon was suppressed (no vision).")
        print("    The backend logs `[predict {sym}] all N horizons suppressed`")
        print("    when this happens — check uvicorn output.")
        return 0

    print(f"Found {len(rows)} {sym} prediction row(s):\n")
    for r in rows:
        flag = "✓" if r["is_public_ledger"] else "✗"
        ledger_floor = "✓" if (r["confidence"] or 0) >= 55 else "✗"
        print(
            f"  [{r['id'][:8]}…]  "
            f"{r['horizon']:3s}  "
            f"conf={r['confidence']:5.1f}  "
            f"dir={r['direction']:8s}  "
            f"verdict={r['verdict']:7s}  "
            f"traded={'Y' if r['traded'] else 'N'}  "
            f"public_ledger={flag}  "
            f"≥55={ledger_floor}  "
            f"{r['created_at']}"
        )

    print()
    eligible = [
        r for r in rows
        if (r["confidence"] or 0) >= 55 and not r["is_public_ledger"] and in_universe
    ]
    if eligible:
        print(f"{len(eligible)} row(s) are eligible for backfill (conf>=55, in universe,")
        print("  but is_public_ledger=false). Run:")
        print("    python scripts/backfill_public_ledger.py --apply")

    on_ledger = [r for r in rows if r["is_public_ledger"]]
    if on_ledger:
        print(f"{len(on_ledger)} row(s) are already on the public ledger.")
        print("  → If you don't see them in the UI, hard-refresh /track-record")
        print("    (Cmd+Shift+R). React Query caches the ledger for 5 minutes.")

    sub_55 = [r for r in rows if (r["confidence"] or 0) < 55]
    if sub_55:
        print(f"{len(sub_55)} row(s) have confidence below 55 — those will never")
        print("  reach the public ledger by design (anchor § 6.2).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
