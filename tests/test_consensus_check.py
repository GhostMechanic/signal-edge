"""
test_consensus_check.py
-----------------------
Pin the contrarian guardrail (methodology § 4.3). The MSFT 1m card on
Apr 30 surfaced a "31.6% below analyst consensus" signal that was
apples-to-oranges (1-month model vs. 12-month consensus). The new
consensus_check module:
  • horizon-adjusts consensus by de-annualising before comparing
  • flags strong-contrarian directionally-opposed calls
  • applies a confidence haircut so the displayed conviction reflects
    the bigger bet implied by disagreeing with the room

Run
---
    cd "Stock Bot"
    python3 -m unittest tests.test_consensus_check -v
"""

from __future__ import annotations

import math
import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from consensus_check import (
    ANALYST_TARGET_HORIZON_DAYS,
    CONTRARIAN_CONFIDENCE_HAIRCUT,
    STRONG_DIVERGENCE_PP,
    deannualise_return,
    directional_opposition,
    evaluate,
    haircut_confidence,
)


class DeannualiseMath(unittest.TestCase):

    def test_round_trip_one_year(self):
        """A 12-month return de-annualised over 365 days is unchanged."""
        self.assertAlmostEqual(
            deannualise_return(0.20, ANALYST_TARGET_HORIZON_DAYS),
            0.20, places=4,
        )

    def test_compounding_reduces_proportionally(self):
        """30 days of a 39.96% annual rate should compound to ~2.84%."""
        # MSFT consensus implied annual: $570.72 / $407.78 - 1 ≈ 0.3996
        result = deannualise_return(0.3996, 30)
        # (1.3996) ** (30/365) - 1 ≈ 0.0280
        self.assertAlmostEqual(result, 0.0280, places=3)

    def test_three_day_horizon_tiny(self):
        # Same 40% annual, 3 days → ~0.281%
        self.assertAlmostEqual(deannualise_return(0.40, 3), 0.00277, places=3)

    def test_negative_annual_compounds_correctly(self):
        # -20% annual over 30 days → -1.83%
        self.assertAlmostEqual(deannualise_return(-0.20, 30), -0.01831, places=3)

    def test_zero_horizon_is_zero(self):
        self.assertEqual(deannualise_return(0.50, 0), 0.0)
        self.assertEqual(deannualise_return(0.50, -10), 0.0)

    def test_nan_returns_zero(self):
        self.assertEqual(deannualise_return(float("nan"), 30), 0.0)

    def test_total_loss_clipped_to_minus_one(self):
        """Defensive: a sub-(-100%) return is mathematically nonsense
        but we should not raise on fractional powers of negatives."""
        self.assertEqual(deannualise_return(-1.5, 30), -1.0)


class DirectionalOpposition(unittest.TestCase):

    def test_bull_vs_bear_is_opposed(self):
        self.assertTrue(directional_opposition(0.05, -0.03))
        self.assertTrue(directional_opposition(-0.05, 0.03))

    def test_same_sign_not_opposed(self):
        self.assertFalse(directional_opposition(0.05, 0.10))
        self.assertFalse(directional_opposition(-0.05, -0.10))

    def test_one_inside_deadband_not_opposed(self):
        # Consensus within deadband → has no view → can't be opposed
        self.assertFalse(directional_opposition(0.05, 0.0005))
        # Model within deadband → has no view → can't be opposed
        self.assertFalse(directional_opposition(0.0005, 0.05))

    def test_both_inside_deadband_not_opposed(self):
        self.assertFalse(directional_opposition(0.0005, -0.0005))

    def test_nan_inputs_safe(self):
        self.assertFalse(directional_opposition(float("nan"), 0.05))


class EvaluateMSFTCase(unittest.TestCase):
    """Pin the exact MSFT screenshot scenario from Apr 30, 2026."""

    def test_msft_1m_is_strong_contrarian_with_haircut(self):
        check = evaluate(
            consensus_target_price=570.72,
            current_price=407.78,
            predicted_price=390.62,
            horizon_days=30,
        )
        self.assertTrue(check.has_consensus)
        # Consensus implies ~+39.96% annual, ~+2.80% over a month.
        self.assertAlmostEqual(check.consensus_implied_annual, 0.3996, places=3)
        self.assertAlmostEqual(check.consensus_implied_horizon, 0.0280, places=3)
        # Model says -4.21% over a month.
        self.assertAlmostEqual(check.model_return_horizon, -0.0421, places=3)
        # Divergence ≈ -4.21 - 2.80 = -7.01 pp
        self.assertAlmostEqual(check.divergence_pp, -7.01, places=1)
        self.assertTrue(check.is_directionally_opposed)
        self.assertTrue(check.is_strong_contrarian)
        self.assertTrue(check.apply_haircut)

    def test_haircut_drops_confidence_below_trade_threshold(self):
        """70.1% × 0.85 = 59.585%. That sits below the 65% TRADE threshold —
        so the contrarian haircut alone is enough to flip a borderline
        trade decision from TRADE to PASS."""
        check = evaluate(
            consensus_target_price=570.72,
            current_price=407.78,
            predicted_price=390.62,
            horizon_days=30,
        )
        self.assertAlmostEqual(haircut_confidence(70.1, check), 59.585, places=2)
        self.assertLess(haircut_confidence(70.1, check), 65.0)


class EvaluateAlignmentCases(unittest.TestCase):
    """When model and consensus agree, no haircut should fire even on
    a large gap. Wide-but-aligned is not contrarian."""

    def test_aligned_bullish_no_haircut_even_with_big_gap(self):
        # Consensus +40% annual (~+2.8%/month), model says +10%/month.
        # Both Bullish — directionally aligned. Divergence is huge but
        # not contrarian; no haircut.
        check = evaluate(
            consensus_target_price=140.0,
            current_price=100.0,
            predicted_price=110.0,
            horizon_days=30,
        )
        self.assertTrue(check.has_consensus)
        self.assertFalse(check.is_directionally_opposed)
        # Divergence pp: model ~+10%, consensus ~+2.84% → +7.16 pp,
        # over the strong threshold but aligned → no haircut.
        self.assertGreater(check.divergence_pp, STRONG_DIVERGENCE_PP)
        self.assertTrue(check.is_strong_contrarian)
        self.assertFalse(check.apply_haircut)
        # Confidence passes through unchanged.
        self.assertEqual(haircut_confidence(80.0, check), 80.0)


class EvaluateMissingInputs(unittest.TestCase):
    """When any input is missing, has_consensus=False and no haircut.
    These are the common cases — illiquid names with no analyst coverage,
    fresh listings, racey first-fetches."""

    def test_no_consensus_target(self):
        check = evaluate(
            consensus_target_price=None,
            current_price=100.0,
            predicted_price=105.0,
            horizon_days=30,
        )
        self.assertFalse(check.has_consensus)
        self.assertFalse(check.apply_haircut)
        self.assertIsNone(check.divergence_pp)

    def test_zero_current_price(self):
        check = evaluate(
            consensus_target_price=120.0,
            current_price=0.0,
            predicted_price=110.0,
            horizon_days=30,
        )
        self.assertFalse(check.has_consensus)

    def test_nan_predicted_price(self):
        check = evaluate(
            consensus_target_price=120.0,
            current_price=100.0,
            predicted_price=float("nan"),
            horizon_days=30,
        )
        self.assertFalse(check.has_consensus)

    def test_negative_horizon_days(self):
        check = evaluate(
            consensus_target_price=120.0,
            current_price=100.0,
            predicted_price=110.0,
            horizon_days=-5,
        )
        self.assertFalse(check.has_consensus)


class HaircutNoOps(unittest.TestCase):
    """haircut_confidence should pass through cleanly when not warranted
    or when given garbage."""

    def test_no_haircut_returns_unchanged(self):
        check = evaluate(
            consensus_target_price=140.0,
            current_price=100.0,
            predicted_price=110.0,
            horizon_days=30,
        )  # aligned bull, no haircut
        self.assertEqual(haircut_confidence(75.0, check), 75.0)

    def test_haircut_constant_documented(self):
        """If somebody nudges the haircut, this test fails so they have
        to think about whether the methodology doc needs an update."""
        self.assertEqual(CONTRARIAN_CONFIDENCE_HAIRCUT, 0.85)
        self.assertEqual(STRONG_DIVERGENCE_PP, 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
