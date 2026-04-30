"""
test_risk_reward_floor.py
-------------------------
Pin down the R:R floor introduced after the MSFT 1-month bear put spread
landed on the public ledger Apr 30 2026 with R:R 0.76:1. The methodology's
gate (§ 4.4) is now: if the ATR-clamped stop/target produce reward/risk
below RISK_REWARD_FLOOR (1.0:1), the equity trade is suppressed. The
prediction itself still ships — only the equity trade card is hidden.

Run
---
    cd "Stock Bot"
    python3 -m unittest tests.test_risk_reward_floor -v
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass
from typing import Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from prediqt_open_trade import _plan_risk_reward, compute_trade_attachment
from trade_decision import PassReason, RISK_REWARD_FLOOR
from trade_plan import TradePlan


# Fake portfolio state — the R:R gate fires before any portfolio check, so
# values here don't matter for that path. Kept generous so the gate is the
# only reason a test could fail.
@dataclass
class _FakePortfolioState:
    cash: float = 10_000.0
    open_positions_count: int = 0
    open_symbols: frozenset = frozenset()
    starting_capital: float = 10_000.0
    max_open_positions: int = 25


# Minimal OHLC stub that atr_from_ohlc_df can chew through for the
# compute_trade_attachment path. We pre-stamp ATR via direct plan
# construction in most tests, but compute_trade_attachment also runs
# through indicators.atr_from_ohlc_df which requires a DataFrame. Those
# integration tests live below; the unit tests use _plan_risk_reward
# directly and skip the OHLC requirement.

UNIVERSE = frozenset({"MSFT", "AAPL", "NVDA"})


class PlanRiskRewardUnit(unittest.TestCase):
    """_plan_risk_reward must agree with the formula the frontend uses."""

    def test_long_returns_reward_over_risk(self):
        # LONG: target above entry, stop below entry
        plan = TradePlan(
            entry_price=100.0, stop_price=95.0, target_price=110.0,
            atr_distance=5, floor_distance=1.5, ceiling_distance=8, stop_distance=5,
        )
        self.assertAlmostEqual(_plan_risk_reward(plan, "Bullish"), 2.0, places=2)

    def test_short_returns_reward_over_risk(self):
        # SHORT: stop above entry, target below entry
        plan = TradePlan(
            entry_price=100.0, stop_price=105.0, target_price=90.0,
            atr_distance=5, floor_distance=1.5, ceiling_distance=8, stop_distance=5,
        )
        self.assertAlmostEqual(_plan_risk_reward(plan, "Bearish"), 2.0, places=2)

    def test_msft_screenshot_case(self):
        """The Apr 30 MSFT card: entry $407.78, stop $430.4837, target $390.62.
        Reward $17.16, risk $22.7037 → R:R ≈ 0.756. That's exactly what the
        frontend showed (rounded to 0.76:1)."""
        plan = TradePlan(
            entry_price=407.78, stop_price=430.4837, target_price=390.62,
            atr_distance=22.70, floor_distance=6.12, ceiling_distance=32.62,
            stop_distance=22.70,
        )
        rr = _plan_risk_reward(plan, "Bearish")
        self.assertAlmostEqual(rr, 0.7560, places=3)
        self.assertLess(rr, RISK_REWARD_FLOOR)

    def test_neutral_returns_none(self):
        plan = TradePlan(
            entry_price=100, stop_price=100, target_price=100,
            atr_distance=0, floor_distance=0, ceiling_distance=0, stop_distance=0,
        )
        self.assertIsNone(_plan_risk_reward(plan, "Neutral"))

    def test_zero_risk_returns_none(self):
        """Defensive: a degenerate plan where entry == stop (shouldn't
        happen given the floor in trade_plan.py, but if it did, R:R is
        undefined and we don't want to leak inf into downstream math)."""
        plan = TradePlan(
            entry_price=100, stop_price=100, target_price=110,
            atr_distance=0, floor_distance=0, ceiling_distance=0, stop_distance=0,
        )
        self.assertIsNone(_plan_risk_reward(plan, "Bullish"))


class ComputeTradeAttachmentEnforcesFloor(unittest.TestCase):
    """compute_trade_attachment must short-circuit with
    PassReason.POOR_RISK_REWARD when the geometry comes out below the
    floor — BEFORE decide_trade gets a chance to weigh in on confidence."""

    def _ohlc(self, atr_dollars: float, last_close: float = 100.0):
        """Synthesize an OHLC frame whose 14-period ATR equals
        approximately `atr_dollars`. We use a constant true range so the
        ATR calc lands on it directly regardless of the smoothing variant."""
        import pandas as pd
        n = 30
        # Constant H-L = atr_dollars, constant close = last_close; this
        # makes True Range == atr_dollars on every bar, so any ATR variant
        # converges to atr_dollars after warmup.
        rows = []
        for i in range(n):
            rows.append({
                "Open":  last_close,
                "High":  last_close + atr_dollars / 2,
                "Low":   last_close - atr_dollars / 2,
                "Close": last_close,
                "Volume": 1_000_000,
            })
        return pd.DataFrame(rows)

    def test_below_floor_pass_reason_is_poor_rr(self):
        """Build a Bearish trade plan whose target is just barely below
        entry (small reward) while ATR is large (wide stop). R:R will be
        well below 1.0 and the gate must fire."""
        # Entry $100, predicted $99 (small reward $1). ATR $4 → 2× = $8 stop.
        # R:R = 1 / 8 = 0.125. Floor is 1.0 — must trip.
        ohlc = self._ohlc(atr_dollars=4.0, last_close=100.0)
        result = compute_trade_attachment(
            symbol="MSFT", direction="Bearish", confidence=80,
            entry_price=100.0, predicted_price=99.0,
            ohlc_for_atr=ohlc,
            canonical_universe=UNIVERSE,
            portfolio_state=_FakePortfolioState(),
        )
        self.assertFalse(result.decision.traded)
        self.assertEqual(result.decision.reason, PassReason.POOR_RISK_REWARD)
        self.assertIn("R:R", result.decision.note)
        # Plan still computed — audit trail intact.
        self.assertEqual(result.plan.entry_price, 100.0)

    def test_above_floor_decision_runs_normally(self):
        """Plan with R:R well above 1 should fall through to decide_trade,
        which approves a high-confidence Bullish call on a known symbol."""
        # Entry $100, predicted $110 (reward $10). ATR $1 → 2× = $2, but
        # stop floor is 1.5% of $100 = $1.50, so stop_distance = $2 (ATR).
        # R:R = 10 / 2 = 5.
        ohlc = self._ohlc(atr_dollars=1.0, last_close=100.0)
        result = compute_trade_attachment(
            symbol="MSFT", direction="Bullish", confidence=80,
            entry_price=100.0, predicted_price=110.0,
            ohlc_for_atr=ohlc,
            canonical_universe=UNIVERSE,
            portfolio_state=_FakePortfolioState(),
        )
        self.assertTrue(result.decision.traded, result.decision.note)
        self.assertIsNone(result.decision.reason)

    def test_neutral_skips_floor_check(self):
        """Neutral predictions return a flat synthetic plan with R:R
        undefined — the floor check must be silent there. The existing
        NEUTRAL_DIRECTION reason should win."""
        ohlc = self._ohlc(atr_dollars=1.0, last_close=100.0)
        result = compute_trade_attachment(
            symbol="MSFT", direction="Neutral", confidence=99,
            entry_price=100.0, predicted_price=100.0,
            ohlc_for_atr=ohlc,
            canonical_universe=UNIVERSE,
            portfolio_state=_FakePortfolioState(),
        )
        self.assertFalse(result.decision.traded)
        self.assertEqual(result.decision.reason, PassReason.NEUTRAL_DIRECTION)

    def test_floor_takes_priority_over_low_confidence(self):
        """When BOTH R:R and confidence are bad, the floor reason wins.
        That's intentional — geometry is more fundamental than confidence
        and gives the user a more actionable signal ('the math doesn't
        work' is louder than 'we're not 65% sure')."""
        # Same R:R-bad setup as test_below_floor_pass_reason_is_poor_rr,
        # but with confidence below TRADE_CONFIDENCE_THRESHOLD too.
        ohlc = self._ohlc(atr_dollars=4.0, last_close=100.0)
        result = compute_trade_attachment(
            symbol="MSFT", direction="Bearish", confidence=40,
            entry_price=100.0, predicted_price=99.0,
            ohlc_for_atr=ohlc,
            canonical_universe=UNIVERSE,
            portfolio_state=_FakePortfolioState(),
        )
        self.assertEqual(result.decision.reason, PassReason.POOR_RISK_REWARD)


if __name__ == "__main__":
    unittest.main(verbosity=2)
