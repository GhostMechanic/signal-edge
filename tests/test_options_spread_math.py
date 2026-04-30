"""
test_options_spread_math.py
---------------------------
Invariant tests for options strategy numerics.

Motivation
----------
A live MSFT prediction surfaced this on the public ledger:

    legs:       BUY 410 PUT @ $17.23 / SELL 380 PUT @ $6.17
    cost:       $1,181  -> implies $11.81 net debit / share
    max profit: $1,719  -> implies $12.81 net debit / share
    breakeven:  $396.19 -> implies $13.81 net debit / share

Three different implied debits in one card. Whatever path produced those
three numbers, they didn't all come out of the same (p_buy, p_sell) tuple.

The right invariants for a vertical debit spread are simple:

    cost  +  max_profit  ==  (K_buy − K_sell) * 100      (bear put)
                          ==  (K_sell − K_buy) * 100      (bull call)
    breakeven * 100        ==  K_buy * 100  ∓  cost      (∓ = -/+)
    max_loss             ==  cost                         (debit spread)

If those hold, the three derived numbers cannot drift apart from each other
because they're all algebraic rearrangements of the same `(p_buy, p_sell)`.

These tests pin those invariants down on `recompute_strategy_numerics` —
the single function every persistence + display path runs through after
strikes/premiums get snapped to the live chain. Any future change that
introduces a stale-read or partial-recompute regression will trip here.

Run
---
    cd "Stock Bot"
    python3 -m unittest tests.test_options_spread_math -v

(stdlib only — no pytest, scipy, or yfinance required.)
"""

from __future__ import annotations

import os
import sys
import unittest

# Allow running this file directly from inside ./tests as well as via
# `python3 -m unittest tests.test_options_spread_math` from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from options_analyzer import recompute_strategy_numerics


# ─── Leg builders ─────────────────────────────────────────────────────────────

def _leg(action: str, opt_type: str, strike: float, premium: float) -> dict:
    """Match the leg shape options_analyzer / options_pricer pass through."""
    return {
        "action":  action,
        "type":    opt_type,
        "strike":  float(strike),
        "premium": float(premium),
    }


# Pinned reference numbers from the MSFT screenshot (Apr 30, 2026).
# We don't know the live mids the backend used; we know the strikes the
# user saw and we know the formulas. The fixtures reproduce the *legal*
# spread shape and assert the invariants hold for any valid premium pair.
MSFT_BUY_K  = 410.0
MSFT_SELL_K = 380.0
MSFT_SPREAD_WIDTH = MSFT_BUY_K - MSFT_SELL_K   # $30 per share = $3,000 / contract
MSFT_BUY_PREMIUM  = 17.23
MSFT_SELL_PREMIUM = 6.17


# ─── Bear put spread ──────────────────────────────────────────────────────────

class BearPutSpreadInvariants(unittest.TestCase):
    """recompute_strategy_numerics('Bear Put Spread', legs) must obey:
        cost + max_profit == (K_buy - K_sell) * 100
        breakeven         == K_buy - cost / 100
        max_loss          == cost
        is_credit         == False
    for every legal (p_buy, p_sell) pair where p_buy >= p_sell >= 0."""

    def _build(self, p_buy: float, p_sell: float,
               K_buy: float = MSFT_BUY_K,
               K_sell: float = MSFT_SELL_K) -> dict:
        legs = [
            _leg("BUY",  "PUT", K_buy,  p_buy),
            _leg("SELL", "PUT", K_sell, p_sell),
        ]
        return recompute_strategy_numerics("Bear Put Spread", legs)

    def assertSpreadConsistent(self, num: dict, K_buy: float, K_sell: float):
        cost       = num["cost_per_contract"]
        max_profit = num["max_profit_per_contract"]
        max_loss   = num["max_loss_per_contract"]
        breakevens = num["breakevens"]

        # Width invariant — cost + max_profit fills the whole spread width.
        width = round((K_buy - K_sell) * 100, 2)
        self.assertAlmostEqual(
            cost + max_profit, width, places=2,
            msg=(
                f"cost ({cost}) + max_profit ({max_profit}) must equal "
                f"spread width ({width}). Difference = "
                f"{round(cost + max_profit - width, 2)}."
            ),
        )

        # Breakeven invariant — for a bear put spread, BE = K_buy - cost/100.
        expected_be = round(K_buy - cost / 100.0, 2)
        self.assertEqual(len(breakevens), 1, "bear put spread should have one BE")
        self.assertAlmostEqual(
            breakevens[0], expected_be, places=2,
            msg=(
                f"breakeven ({breakevens[0]}) doesn't match K_buy - cost/100 "
                f"({expected_be}). cost={cost}, K_buy={K_buy}."
            ),
        )

        # Debit spread → max_loss == cost.
        self.assertEqual(
            max_loss, cost,
            f"max_loss ({max_loss}) should equal cost ({cost}) on a debit spread",
        )
        self.assertFalse(num["is_credit"], "bear put spread is a debit spread")

    # ── The exact MSFT screenshot fixture ─────────────────────────────────────

    def test_msft_410_380_at_screenshot_premiums(self):
        """The Apr 30 MSFT card showed BUY 410P @ 17.23 / SELL 380P @ 6.17.
        Those exact premiums must produce a single, internally-consistent
        cost/max_profit/breakeven triple."""
        num = self._build(MSFT_BUY_PREMIUM, MSFT_SELL_PREMIUM)
        self.assertSpreadConsistent(num, MSFT_BUY_K, MSFT_SELL_K)

        # Pin the exact expected numbers so we can compare against what the
        # frontend showed. If the displayed cost/max_profit/breakeven on a
        # future card don't match these for the same legs, the bug isn't
        # in this function.
        expected_cost = round((MSFT_BUY_PREMIUM - MSFT_SELL_PREMIUM) * 100, 2)
        expected_max  = round(MSFT_SPREAD_WIDTH * 100 - expected_cost, 2)
        expected_be   = round(MSFT_BUY_K - (MSFT_BUY_PREMIUM - MSFT_SELL_PREMIUM), 2)

        self.assertEqual(num["cost_per_contract"],       expected_cost)
        self.assertEqual(num["max_profit_per_contract"], expected_max)
        self.assertEqual(num["breakevens"][0],           expected_be)
        self.assertEqual(num["max_loss_per_contract"],   expected_cost)

    def test_screenshot_displayed_numbers_are_internally_inconsistent(self):
        """Sanity check: the MSFT card's three displayed values
        (cost=$1181, max_profit=$1719, breakeven=$396.19) cannot all come
        from one legal (p_buy, p_sell) pair. This test pins that fact down
        as documentation — if it ever STARTS passing, somebody has 'fixed'
        the bug by making the formulas produce drifting outputs, which
        would be worse, not better."""
        screenshot_cost       = 1181.0
        screenshot_max_profit = 1719.0
        screenshot_breakeven  = 396.19

        # Implied debits per share, recovered from each displayed number.
        d_from_cost       = screenshot_cost / 100.0                            # 11.81
        d_from_max_profit = MSFT_SPREAD_WIDTH - (screenshot_max_profit / 100)  # 12.81
        d_from_breakeven  = MSFT_BUY_K - screenshot_breakeven                  # 13.81

        debits = {
            "cost":       round(d_from_cost,       2),
            "max_profit": round(d_from_max_profit, 2),
            "breakeven":  round(d_from_breakeven,  2),
        }
        # Each implied debit must be DIFFERENT — that's the whole point of
        # this regression test.
        self.assertEqual(len(set(debits.values())), 3, debits)

        # And none of them match the displayed leg premiums' raw debit.
        legal_debit = round(MSFT_BUY_PREMIUM - MSFT_SELL_PREMIUM, 2)  # 11.06
        for name, d in debits.items():
            self.assertNotAlmostEqual(
                d, legal_debit, places=2,
                msg=f"{name} debit {d} unexpectedly matches legal debit {legal_debit}",
            )

    # ── Coverage across the fee/IV/strike space ────────────────────────────────

    def test_invariants_hold_for_a_grid_of_premium_pairs(self):
        """Sweep premium pairs covering deep-ITM, ATM, OTM, and tight/wide
        spreads. If any combination drifts the three derived numbers apart,
        we want to know."""
        cases = []
        # (p_buy, p_sell) — p_buy must be >= p_sell for a debit bear put
        for p_buy in (0.10, 1.50, 5.00, 11.06, 13.81, 17.23, 25.00):
            for p_sell in (0.00, 0.05, 0.50, 2.00, 4.00, 6.17, 9.00):
                if p_buy < p_sell:
                    continue
                cases.append((p_buy, p_sell))

        for p_buy, p_sell in cases:
            with self.subTest(p_buy=p_buy, p_sell=p_sell):
                num = self._build(p_buy, p_sell)
                self.assertSpreadConsistent(num, MSFT_BUY_K, MSFT_SELL_K)

    def test_invariants_hold_for_different_strike_widths(self):
        """Same invariants must hold whether the spread is $5 wide or $50."""
        for K_buy, K_sell in [
            (100.0, 95.0),   # tight, $5 wide
            (200.0, 190.0),  # $10 wide
            (410.0, 380.0),  # screenshot — $30 wide
            (500.0, 450.0),  # wide, $50
        ]:
            with self.subTest(K_buy=K_buy, K_sell=K_sell):
                # Mid-of-spread premium pair, scaled to the strikes
                num = self._build(p_buy=(K_buy - K_sell) * 0.4, p_sell=0.50,
                                  K_buy=K_buy, K_sell=K_sell)
                self.assertSpreadConsistent(num, K_buy, K_sell)

    def test_premium_rounding_does_not_break_invariants(self):
        """Live mids land at 4-decimal precision (see options_pricer.py:400);
        recompute rounds to 2 decimals. Make sure that rounding chain still
        leaves cost/max_profit/breakeven consistent with each other."""
        # 4-decimal premiums similar to what `_build_leg_quote` writes
        num = self._build(p_buy=14.8125, p_sell=1.0050)
        self.assertSpreadConsistent(num, MSFT_BUY_K, MSFT_SELL_K)


# ─── Bull call spread (mirror) ────────────────────────────────────────────────

class BullCallSpreadInvariants(unittest.TestCase):
    """Bull call spread: cost + max_profit == (K_sell - K_buy) * 100,
    breakeven == K_buy + cost/100. Same shape, opposite direction."""

    def _build(self, p_buy: float, p_sell: float,
               K_buy: float = 100.0, K_sell: float = 110.0) -> dict:
        legs = [
            _leg("BUY",  "CALL", K_buy,  p_buy),
            _leg("SELL", "CALL", K_sell, p_sell),
        ]
        return recompute_strategy_numerics("Bull Call Spread", legs)

    def test_invariants_basic(self):
        num = self._build(p_buy=4.50, p_sell=1.20)
        K_buy, K_sell = 100.0, 110.0
        cost       = num["cost_per_contract"]
        max_profit = num["max_profit_per_contract"]
        width      = round((K_sell - K_buy) * 100, 2)

        self.assertAlmostEqual(cost + max_profit, width, places=2)
        self.assertAlmostEqual(num["breakevens"][0],
                               round(K_buy + cost / 100.0, 2),
                               places=2)
        self.assertEqual(num["max_loss_per_contract"], cost)
        self.assertFalse(num["is_credit"])

    def test_grid(self):
        for p_buy in (0.50, 2.00, 4.50, 8.00):
            for p_sell in (0.10, 1.20, 3.00):
                if p_buy < p_sell:
                    continue
                with self.subTest(p_buy=p_buy, p_sell=p_sell):
                    num = self._build(p_buy=p_buy, p_sell=p_sell)
                    cost = num["cost_per_contract"]
                    self.assertAlmostEqual(
                        cost + num["max_profit_per_contract"],
                        round((110.0 - 100.0) * 100, 2),
                        places=2,
                    )
                    self.assertAlmostEqual(
                        num["breakevens"][0],
                        round(100.0 + cost / 100.0, 2),
                        places=2,
                    )


# ─── Long single-leg sanity (cost == max_loss, BE = K ± premium) ─────────────

class LongOptionInvariants(unittest.TestCase):
    """Long call/put: cost == max_loss, breakeven = K ± premium.
    Long call: max_profit unbounded (None). Long put: max_profit = (K-p)*100."""

    def test_long_call(self):
        num = recompute_strategy_numerics(
            "Long Call",
            [_leg("BUY", "CALL", 100.0, 4.50)],
        )
        self.assertEqual(num["cost_per_contract"], 450.0)
        self.assertEqual(num["max_loss_per_contract"], 450.0)
        self.assertIsNone(num["max_profit_per_contract"])  # unbounded
        self.assertAlmostEqual(num["breakevens"][0], 104.50, places=2)

    def test_long_put(self):
        num = recompute_strategy_numerics(
            "Long Put",
            [_leg("BUY", "PUT", 100.0, 3.20)],
        )
        self.assertEqual(num["cost_per_contract"], 320.0)
        self.assertEqual(num["max_loss_per_contract"], 320.0)
        # K=$100, premium=$3.20 → max profit if S→0 is (100-3.20)*100 = $9,680
        self.assertAlmostEqual(num["max_profit_per_contract"], 9680.0, places=2)
        self.assertAlmostEqual(num["breakevens"][0], 96.80, places=2)


# ─── Iron condor sanity (credit + bounded loss + two breakevens) ─────────────

class IronCondorInvariants(unittest.TestCase):
    """Iron condor: net_credit + max_loss == (call_sell - put_sell) * 100,
    two breakevens straddling the body."""

    def test_basic(self):
        legs = [
            _leg("BUY",  "PUT",  90.0, 0.50),
            _leg("SELL", "PUT",  95.0, 1.50),
            _leg("SELL", "CALL", 105.0, 1.40),
            _leg("BUY",  "CALL", 110.0, 0.45),
        ]
        num = recompute_strategy_numerics("Iron Condor", legs)

        # Credit strategy → cost stored as negative-of-credit by convention
        net_credit = num["max_profit_per_contract"]
        self.assertEqual(num["cost_per_contract"], -net_credit)
        self.assertTrue(num["is_credit"])

        # Width invariant — credit + max_loss fills the wider wing.
        body_width = round((105.0 - 95.0) * 100, 2)
        self.assertAlmostEqual(net_credit + num["max_loss_per_contract"],
                               body_width, places=2)

        # Two breakevens, ordered low-then-high.
        self.assertEqual(len(num["breakevens"]), 2)
        be_lo, be_hi = num["breakevens"]
        self.assertLess(be_lo, be_hi)
        # be_lo = K_put_sell - net_credit/100, be_hi = K_call_sell + net_credit/100
        self.assertAlmostEqual(be_lo, 95.0 - net_credit / 100.0, places=2)
        self.assertAlmostEqual(be_hi, 105.0 + net_credit / 100.0, places=2)


# ─── Invariant self-check (the bug-shouting layer) ────────────────────────────

class InvariantSelfCheck(unittest.TestCase):
    """recompute_strategy_numerics now stamps `invariant_violation` on the
    output. A clean recompute should set it False; a deliberately mangled
    numeric block should set it True with a human-readable reason."""

    def test_bear_put_spread_clean_run_sets_invariant_false(self):
        legs = [
            _leg("BUY",  "PUT", MSFT_BUY_K,  MSFT_BUY_PREMIUM),
            _leg("SELL", "PUT", MSFT_SELL_K, MSFT_SELL_PREMIUM),
        ]
        num = recompute_strategy_numerics("Bear Put Spread", legs)
        self.assertIn("invariant_violation", num)
        self.assertFalse(num["invariant_violation"], num.get("invariant_violation_reason"))

    def test_bull_call_spread_clean_run_sets_invariant_false(self):
        legs = [
            _leg("BUY",  "CALL", 100.0, 4.50),
            _leg("SELL", "CALL", 110.0, 1.20),
        ]
        num = recompute_strategy_numerics("Bull Call Spread", legs)
        self.assertFalse(num["invariant_violation"], num.get("invariant_violation_reason"))

    def test_iron_condor_clean_run_sets_invariant_false(self):
        legs = [
            _leg("BUY",  "PUT",  90.0, 0.50),
            _leg("SELL", "PUT",  95.0, 1.50),
            _leg("SELL", "CALL", 105.0, 1.40),
            _leg("BUY",  "CALL", 110.0, 0.45),
        ]
        num = recompute_strategy_numerics("Iron Condor", legs)
        self.assertFalse(num["invariant_violation"], num.get("invariant_violation_reason"))

    def test_invariant_check_catches_synthetic_corruption(self):
        """Directly call the private check with the screenshot's drifting
        numbers. It must set `invariant_violation=True` and explain why."""
        from options_analyzer import _check_spread_invariants
        # Build a numeric block that mirrors what the MSFT card displayed.
        bad_numeric = {
            "cost_per_contract":       1181.0,
            "max_profit_per_contract": 1719.0,
            "max_loss_per_contract":   1181.0,
            "breakevens":              [396.19],
            "is_credit":               False,
            "tradable":                True,
        }
        out = _check_spread_invariants(
            "Bear Put Spread",
            bad_numeric,
            width_dollars_per_contract=3000.0,           # $30 wide x 100
            expected_breakeven=410.0 - 1181.0 / 100.0,    # 398.19
        )
        self.assertTrue(out["invariant_violation"])
        # Reason should mention both the width and breakeven failures.
        reason = out["invariant_violation_reason"]
        self.assertIn("width invariant", reason)
        self.assertIn("breakeven invariant", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
