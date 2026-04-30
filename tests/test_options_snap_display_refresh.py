"""
test_options_snap_display_refresh.py
------------------------------------
Regression tests for the legacy-display-string staleness bug found on
the MSFT 410/380 bear-put-spread card (Apr 30, 2026, prediction
9162b8c5-...-811c).

Bug
---
options_pricer.snap_strategy_to_real_chain returned the snapped strategy
with a `**strategy` spread, replacing only `legs` and `numeric`. The
legacy display-string fields (`estimated_cost`, `max_profit`, `max_loss`,
`breakeven`) carried over from the original Black-Scholes theoretical
build at the pre-snap strikes — so the public-ledger card showed BS
numbers from strikes 408/379 ($1181 / $1719 / $396.19) while legs and
numeric reflected the live-chain snap to 410/380 ($1105 / $1895 / $398.95).

Fix
---
snap_strategy_to_real_chain now also calls _format_display_strings on
the live numeric block and merges those refreshed strings before the
return. This file pins down that behavior so a future refactor can't
silently regress.

Run
---
    cd "Stock Bot"
    python3 -m unittest tests.test_options_snap_display_refresh -v
"""

from __future__ import annotations

import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from options_pricer import _format_display_strings


class FormatDisplayStringsBearPut(unittest.TestCase):
    """Bear put spread display strings must come from the live numeric
    block, not from any pre-snap theoretical numbers."""

    def test_reproduces_msft_screenshot_bug_when_called_with_live_block(self):
        """Direct repro of the MSFT case. Pre-snap strategy was for
        strikes 408/379 with BS premiums 21.57/9.76 (cost $1181, max
        $1719, BE $396.19). Post-snap, strikes are 410/380 with live
        mids 17.225/6.175 (cost $1105, max $1895, BE $398.95). The
        refreshed display strings MUST reflect the post-snap numbers."""
        live_legs = [
            {"action": "BUY",  "type": "PUT", "strike": 410.0, "premium": 17.225},
            {"action": "SELL", "type": "PUT", "strike": 380.0, "premium": 6.175},
        ]
        live_numeric = {
            "cost_per_contract":       1105.0,
            "max_profit_per_contract": 1895.0,
            "max_loss_per_contract":   1105.0,
            "breakevens":              [398.95],
            "is_credit":               False,
            "tradable":                True,
        }

        out = _format_display_strings("Bear Put Spread", live_legs, live_numeric)

        # All four strings must reflect the LIVE numeric block.
        self.assertIn("$1105.00", out["estimated_cost"])
        self.assertIn("$1895.00", out["max_profit"])
        self.assertIn("$1105.00", out["max_loss"])
        self.assertEqual(out["breakeven"], "$398.95")

        # And specifically must NOT contain the stale Black-Scholes numbers.
        for k in ("estimated_cost", "max_profit", "max_loss", "breakeven"):
            self.assertNotIn("1181", out[k], k)
            self.assertNotIn("1719", out[k], k)
            self.assertNotIn("396.19", out[k], k)

    def test_label_format_matches_constructor(self):
        """The string format must match what _bear_put_spread originally
        wrote, so the frontend doesn't see a different shape after snap."""
        live_legs = [
            {"action": "BUY",  "type": "PUT", "strike": 100.0, "premium": 5.00},
            {"action": "SELL", "type": "PUT", "strike": 90.0,  "premium": 1.00},
        ]
        live_numeric = {
            "cost_per_contract":       400.0,
            "max_profit_per_contract": 600.0,
            "max_loss_per_contract":   400.0,
            "breakevens":              [96.00],
            "is_credit":               False,
            "tradable":                True,
        }
        out = _format_display_strings("Bear Put Spread", live_legs, live_numeric)
        self.assertEqual(out["estimated_cost"], "$400.00 net debit / contract")
        self.assertEqual(out["max_profit"],     "$600.00 / contract")
        self.assertEqual(out["max_loss"],       "$400.00 / contract")
        self.assertEqual(out["breakeven"],      "$96.00")


class FormatDisplayStringsBullCall(unittest.TestCase):
    """Bull call spread strings include the SELL strike in the max-profit tail."""

    def test_max_profit_includes_sell_strike(self):
        live_legs = [
            {"action": "BUY",  "type": "CALL", "strike": 100.0, "premium": 4.50},
            {"action": "SELL", "type": "CALL", "strike": 110.0, "premium": 1.20},
        ]
        live_numeric = {
            "cost_per_contract":       330.0,
            "max_profit_per_contract": 670.0,
            "max_loss_per_contract":   330.0,
            "breakevens":              [103.30],
            "is_credit":               False,
            "tradable":                True,
        }
        out = _format_display_strings("Bull Call Spread", live_legs, live_numeric)
        self.assertEqual(out["estimated_cost"], "$330.00 net debit / contract")
        self.assertIn("$670.00", out["max_profit"])
        self.assertIn("$110.00+", out["max_profit"])  # SELL strike tail
        self.assertEqual(out["breakeven"], "$103.30")


class FormatDisplayStringsIronCondor(unittest.TestCase):
    """Iron condor strings include both breakevens and the body strikes."""

    def test_two_breakevens_and_body_strikes(self):
        live_legs = [
            {"action": "BUY",  "type": "PUT",  "strike": 90.0,  "premium": 0.50},
            {"action": "SELL", "type": "PUT",  "strike": 95.0,  "premium": 1.50},
            {"action": "SELL", "type": "CALL", "strike": 105.0, "premium": 1.40},
            {"action": "BUY",  "type": "CALL", "strike": 110.0, "premium": 0.45},
        ]
        live_numeric = {
            "cost_per_contract":              -195.0,  # -credit
            "max_profit_per_contract":         195.0,
            "max_loss_per_contract":           305.0,
            "breakevens":                      [93.05, 106.95],
            "is_credit":                       True,
            "tradable":                        True,
        }
        out = _format_display_strings("Iron Condor", live_legs, live_numeric)
        self.assertIn("$195.00", out["estimated_cost"])
        self.assertIn("net credit", out["estimated_cost"])
        self.assertIn("$95", out["max_profit"])
        self.assertIn("$105", out["max_profit"])
        self.assertIn("$93.05 (down)", out["breakeven"])
        self.assertIn("$106.95 (up)", out["breakeven"])
        self.assertIn("$305.00", out["max_loss"])


class FormatDisplayStringsLongOptions(unittest.TestCase):

    def test_long_call_max_profit_unbounded_string(self):
        legs = [{"action": "BUY", "type": "CALL", "strike": 100.0, "premium": 4.50}]
        numeric = {
            "cost_per_contract":       450.0,
            "max_profit_per_contract": None,
            "max_loss_per_contract":   450.0,
            "breakevens":              [104.50],
            "is_credit":               False,
            "tradable":                True,
        }
        out = _format_display_strings("Long Call", legs, numeric)
        self.assertEqual(out["estimated_cost"], "$450.00 / contract")
        self.assertIn("Unlimited", out["max_profit"])
        self.assertEqual(out["max_loss"], "$450.00 per contract (premium paid)")
        self.assertEqual(out["breakeven"], "$104.50")

    def test_long_put_max_profit_bounded_at_strike(self):
        legs = [{"action": "BUY", "type": "PUT", "strike": 100.0, "premium": 3.20}]
        numeric = {
            "cost_per_contract":       320.0,
            "max_profit_per_contract": 9680.0,
            "max_loss_per_contract":   320.0,
            "breakevens":              [96.80],
            "is_credit":               False,
            "tradable":                True,
        }
        out = _format_display_strings("Long Put", legs, numeric)
        self.assertEqual(out["estimated_cost"], "$320.00 / contract")
        self.assertIn("$9680.00", out["max_profit"])
        self.assertIn("stock → $0", out["max_profit"])
        self.assertEqual(out["breakeven"], "$96.80")


class UnknownStrategyLeavesStringsAlone(unittest.TestCase):
    """If the strategy_type isn't one we know how to refresh, return an
    empty dict so the snap path's `**strategy` spread carries the original
    strings through unchanged."""

    def test_returns_empty_dict_for_unknown(self):
        out = _format_display_strings("Mystery Strategy", [], {"cost_per_contract": 100})
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
