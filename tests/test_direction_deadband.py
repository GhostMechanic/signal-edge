"""
test_direction_deadband.py
--------------------------
Pin the methodology § 4.1 direction-from-return deadband. The MSFT 1m card
on Apr 30 2026 surfaced a +0.1% predicted return labeled "Bullish" with
70% confidence — a high-confidence "nothing happens" call dressed up as
directional. The deadband fixes that: |predicted_return| ≤ 0.5% → Neutral.

Run
---
    cd "Stock Bot"
    python3 -m unittest tests.test_direction_deadband -v
"""

from __future__ import annotations

import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from trade_decision import (
    NEUTRAL_RETURN_DEADBAND,
    direction_from_return,
)


class DeadbandClassification(unittest.TestCase):

    def test_msft_screenshot_case_is_neutral_not_bullish(self):
        """+0.1% predicted return — the exact MSFT 1m bug — must classify
        as Neutral, not Bullish. This is the regression test for the bug
        the user reported."""
        self.assertEqual(direction_from_return(0.001), "Neutral")

    def test_clear_bullish_above_deadband(self):
        # 1% is comfortably outside the 0.5% deadband
        self.assertEqual(direction_from_return(0.01), "Bullish")
        self.assertEqual(direction_from_return(0.05), "Bullish")
        self.assertEqual(direction_from_return(0.30), "Bullish")

    def test_clear_bearish_below_negative_deadband(self):
        self.assertEqual(direction_from_return(-0.01), "Bearish")
        self.assertEqual(direction_from_return(-0.0421), "Bearish")  # MSFT bear case
        self.assertEqual(direction_from_return(-0.30), "Bearish")

    def test_zero_is_neutral(self):
        self.assertEqual(direction_from_return(0.0), "Neutral")
        self.assertEqual(direction_from_return(-0.0), "Neutral")

    def test_boundary_inside_deadband_is_neutral(self):
        """Returns at exactly the deadband (and just inside) classify as
        Neutral — abs(r) ≤ deadband is the rule, not abs(r) < deadband."""
        self.assertEqual(direction_from_return(NEUTRAL_RETURN_DEADBAND), "Neutral")
        self.assertEqual(direction_from_return(-NEUTRAL_RETURN_DEADBAND), "Neutral")
        self.assertEqual(direction_from_return(NEUTRAL_RETURN_DEADBAND - 1e-9), "Neutral")

    def test_just_outside_deadband_classifies_directionally(self):
        eps = 1e-4
        self.assertEqual(
            direction_from_return(NEUTRAL_RETURN_DEADBAND + eps), "Bullish",
        )
        self.assertEqual(
            direction_from_return(-(NEUTRAL_RETURN_DEADBAND + eps)), "Bearish",
        )

    def test_custom_deadband_overrides_default(self):
        # 1% deadband — 0.6% becomes Neutral (was Bullish at 0.5% default)
        self.assertEqual(
            direction_from_return(0.006, deadband=0.01), "Neutral",
        )
        # And 1.5% is still Bullish even with the wider deadband
        self.assertEqual(
            direction_from_return(0.015, deadband=0.01), "Bullish",
        )

    def test_handles_garbage_input_safely(self):
        """Defensive: NaN / non-finite / non-numeric should fall back to
        Neutral rather than raise. The path that calls this gets its
        predicted_return from JSON or pandas — both can produce surprises."""
        self.assertEqual(direction_from_return(float("nan")), "Neutral")
        self.assertEqual(direction_from_return(None), "Neutral")  # type: ignore[arg-type]
        self.assertEqual(direction_from_return("not a number"), "Neutral")  # type: ignore[arg-type]

    def test_deadband_constant_is_documented_value(self):
        """The methodology page says 0.5%. If somebody nudges the constant,
        this test fails so they have to think about whether the methodology
        doc needs an update too."""
        self.assertEqual(NEUTRAL_RETURN_DEADBAND, 0.005)


if __name__ == "__main__":
    unittest.main(verbosity=2)
