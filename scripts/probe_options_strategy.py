"""
probe_options_strategy.py
-------------------------
Pull a single prediction's stored options_strategy JSON from Supabase and
check whether the persisted numeric block reconciles. Use this to decide
whether a card's drifting cost/max_profit/breakeven values come from the
backend or the frontend.

Usage
-----
    cd "Stock Bot"
    python3 scripts/probe_options_strategy.py 9162b8c5...811c

The argument can be:
    • a full uuid                   → exact match
    • a UUID prefix                 → ilike '9162b8c5%'
    • a 'prefix...suffix' shape     → ilike '9162b8c5%811c'  (matches the
      receipt format shown on the public ledger card)

If you don't pass anything, the script picks the most recent prediction
across the whole ledger.

Reads SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from .env.

What you see
------------
1. The prediction row's key fields (symbol, horizon, predicted return,
   trade plan).
2. The options_strategy column dumped as raw JSON.
3. An invariant check on the persisted numeric block. If the backend
   stored a consistent (cost, max_profit, breakeven) triple, the bug is
   in the frontend (prediqt-web is reading or recomputing one of those
   from the wrong field). If the persisted block is already drifted,
   the bug is in the backend pipeline before persistence — paste the
   output back into Claude and we'll narrow it down.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

# Make sure we can import options_analyzer regardless of cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import requests
except ImportError:
    sys.exit("requests not installed — pip install requests")


def _load_env() -> Dict[str, str]:
    """Tiny stdlib-only .env reader. Handles KEY=value, optional quotes,
    blank lines, and `#` comments. No dotenv dependency required."""
    env: Dict[str, str] = {}
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        sys.exit(f".env not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip a single layer of matching quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            env[key] = value
    if not env.get("SUPABASE_URL"):
        sys.exit("SUPABASE_URL missing in .env")
    if not env.get("SUPABASE_SERVICE_ROLE_KEY"):
        sys.exit("SUPABASE_SERVICE_ROLE_KEY missing in .env")
    return env


def _matches_receipt(row_id: str, receipt: str) -> bool:
    """Receipt forms: full uuid (exact), prefix... (startswith),
    prefix...suffix (startswith + endswith)."""
    rid = (row_id or "").lower()
    rcpt = receipt.lower()
    if "..." in rcpt:
        prefix, suffix = rcpt.split("...", 1)
        return rid.startswith(prefix) and (not suffix or rid.endswith(suffix))
    if "-" in rcpt and len(rcpt) == 36:
        return rid == rcpt
    return rid.startswith(rcpt)


def _fetch(env: Dict[str, str], receipt: Optional[str]) -> Optional[Dict[str, Any]]:
    """`id` is a uuid column — PostgREST won't ilike it directly. We fetch
    the most recent N rows ordered by created_at and filter client-side.
    Cheap (200 rows) and avoids the cast-operator dance."""
    base = env["SUPABASE_URL"].rstrip("/")
    headers = {
        "apikey":        env["SUPABASE_SERVICE_ROLE_KEY"],
        "Authorization": f"Bearer {env['SUPABASE_SERVICE_ROLE_KEY']}",
        "Accept":        "application/json",
    }
    cols = (
        "id,symbol,horizon,direction,confidence,predicted_return,"
        "predicted_price,entry_price,stop_price,target_price,traded,"
        "horizon_starts_at,horizon_ends_at,options_strategy,model_version,"
        "regime,is_public_ledger,created_at"
    )

    # No receipt → just the latest.
    if not receipt:
        url = f"{base}/rest/v1/predictions?select={cols}&order=created_at.desc&limit=1"
        print(f"→ GET {url}")
        r = requests.get(url, headers=headers, timeout=15)
        print(f"← HTTP {r.status_code}")
        r.raise_for_status()
        rows = r.json()
        return rows[0] if rows else None

    # Full uuid → exact lookup is cheap and accurate.
    if "-" in receipt and len(receipt) == 36 and "..." not in receipt:
        url = f"{base}/rest/v1/predictions?select={cols}&id=eq.{receipt}"
        print(f"→ GET {url}")
        r = requests.get(url, headers=headers, timeout=15)
        print(f"← HTTP {r.status_code}")
        r.raise_for_status()
        rows = r.json()
        return rows[0] if rows else None

    # Prefix or prefix...suffix → scan recent rows client-side.
    url = (
        f"{base}/rest/v1/predictions?select={cols}"
        "&order=created_at.desc&limit=200"
    )
    print(f"→ GET (scan) {url}")
    r = requests.get(url, headers=headers, timeout=15)
    print(f"← HTTP {r.status_code}")
    r.raise_for_status()
    rows = r.json()
    print(f"  scanning {len(rows)} most-recent rows for receipt {receipt!r}")
    matches = [row for row in rows if _matches_receipt(row.get("id", ""), receipt)]
    if not matches:
        print(f"  no match in last {len(rows)} rows. Rerun with the full UUID:")
        print("  python3 scripts/probe_options_strategy.py <full-uuid>")
        return None
    if len(matches) > 1:
        print(f"  multiple matches ({len(matches)}); using the most recent")
    return matches[0]


def _check_numeric(strat: Dict[str, Any]) -> None:
    from options_analyzer import _check_spread_invariants

    st = strat.get("strategy")
    legs = strat.get("legs") or []
    numeric = dict(strat.get("numeric") or {})

    print(f"\nstrategy = {st!r}, {len(legs)} legs")
    if not legs or not numeric:
        print("(no legs or numeric block — nothing to check)")
        return

    # Vertical debit spreads carry a single breakeven.
    if st in ("Bear Put Spread", "Bull Call Spread") and len(legs) == 2:
        buy = next((l for l in legs if (l.get("action") or "").upper() == "BUY"), None)
        sell = next((l for l in legs if (l.get("action") or "").upper() == "SELL"), None)
        if not (buy and sell):
            print("missing BUY or SELL leg")
            return
        K_buy, K_sell = float(buy["strike"]), float(sell["strike"])
        p_buy, p_sell = float(buy.get("premium") or 0), float(sell.get("premium") or 0)
        if st == "Bear Put Spread":
            width = round((K_buy - K_sell) * 100, 2)
            expected_be = round(K_buy - float(numeric.get("cost_per_contract") or 0) / 100.0, 2)
            implied_be_from_legs = round(K_buy - p_buy + p_sell, 2)
        else:
            width = round((K_sell - K_buy) * 100, 2)
            expected_be = round(K_buy + float(numeric.get("cost_per_contract") or 0) / 100.0, 2)
            implied_be_from_legs = round(K_buy + p_buy - p_sell, 2)
        implied_cost_from_legs = round((p_buy - p_sell) * 100, 2)
        implied_max_from_legs  = round(width - implied_cost_from_legs, 2)

        checked = _check_spread_invariants(
            st, numeric,
            width_dollars_per_contract=width,
            expected_breakeven=expected_be,
        )
        print(f"  K_buy={K_buy}  K_sell={K_sell}  width=${width}")
        print(f"  legs: BUY @ ${p_buy:.4f}  /  SELL @ ${p_sell:.4f}")
        print(f"  numeric.cost           = ${numeric.get('cost_per_contract')}")
        print(f"  numeric.max_profit     = ${numeric.get('max_profit_per_contract')}")
        print(f"  numeric.max_loss       = ${numeric.get('max_loss_per_contract')}")
        print(f"  numeric.breakevens[0]  = ${(numeric.get('breakevens') or [None])[0]}")
        print()
        print(f"  ➔ legs imply: cost=${implied_cost_from_legs}, "
              f"max_profit=${implied_max_from_legs}, "
              f"breakeven=${implied_be_from_legs}")
        legs_match_numeric = (
            abs(implied_cost_from_legs - float(numeric.get("cost_per_contract") or 0)) < 0.05
            and abs(implied_max_from_legs - float(numeric.get("max_profit_per_contract") or 0)) < 0.05
            and abs(implied_be_from_legs - float((numeric.get("breakevens") or [0])[0] or 0)) < 0.05
        )
        print(f"  ➔ legs match numeric:   {legs_match_numeric}")
        print(f"  ➔ invariant_violation:  {checked['invariant_violation']}")
        if checked.get("invariant_violation"):
            print(f"     reason: {checked['invariant_violation_reason']}")
        if legs_match_numeric and not checked["invariant_violation"]:
            print("\n   verdict: BACKEND IS CONSISTENT. Bug is on the frontend.")
        else:
            print("\n   verdict: backend produced drifted numbers — paste this output back "
                  "to Claude and we'll trace the path that wrote them.")
    else:
        print("(non-vertical strategy — generic invariant check coming soon)")


def main(argv: list) -> int:
    receipt = argv[1] if len(argv) > 1 else None
    env = _load_env()
    row = _fetch(env, receipt)
    if not row:
        print(f"no row matched {receipt!r}")
        return 1

    print("\n── prediction row ─────────────────────────────────────────────")
    for k in (
        "id", "symbol", "horizon", "direction", "confidence",
        "predicted_return", "predicted_price", "entry_price",
        "stop_price", "target_price", "traded", "regime",
        "model_version", "horizon_starts_at", "horizon_ends_at",
        "is_public_ledger", "created_at",
    ):
        if k in row:
            print(f"  {k:24s} {row.get(k)!r}")

    strat = row.get("options_strategy")
    if not strat:
        print("\n(no options_strategy persisted on this row)")
        return 0

    print("\n── options_strategy (raw JSON) ────────────────────────────────")
    print(json.dumps(strat, indent=2, default=str))

    print("\n── invariant check on stored numeric ─────────────────────────")
    try:
        _check_numeric(strat)
    except Exception as exc:
        print(f"check failed: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
