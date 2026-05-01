"""
test_haircut_ordering.py
------------------------
Pin the ordering of the methodology § 4.3 contrarian haircut relative
to log_prediction_v2 in api/main.py's predict_symbol flow.

The bug this guards against: from 2026-04-29 through 2026-04-30 the
consensus_check loop ran AFTER log_prediction_v2, which meant any
contrarian call landed in Supabase with its pre-haircut confidence
while the live response carried the post-haircut value. Receipt page
and live page silently disagreed by exactly the 0.85× factor on every
contrarian flagged prediction.

This test reads the api/main.py source as text and verifies that the
`hdata["confidence"] = _consensus_haircut(...)` line appears BEFORE
the `log_prediction_v2(` call. Source-text-level check is intentional
— the actual flow is too tangled to mock cleanly, but the textual
ordering is what matters and is unambiguous.

Run
---
    cd "Stock Bot"
    python3 -m unittest tests.test_haircut_ordering -v
"""

from __future__ import annotations

import os
import re
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_API_MAIN = os.path.join(_REPO_ROOT, "api", "main.py")


class HaircutRunsBeforePersist(unittest.TestCase):

    def test_haircut_application_precedes_log_prediction(self):
        """The line that mutates ``hdata["confidence"]`` via
        ``_consensus_haircut`` must appear *earlier* in api/main.py than
        the call to ``log_prediction_v2(``. If the order ever flips
        again, contrarian calls will silently persist with the wrong
        confidence and the receipt page will disagree with the live
        page on every directionally-opposed prediction."""
        with open(_API_MAIN, "r", encoding="utf-8") as fh:
            src = fh.read()

        # Find the haircut application — flexible enough to survive
        # whitespace tweaks but not a real refactor that changes intent.
        haircut_re = re.compile(
            r'hdata\["confidence"\]\s*=\s*_consensus_haircut\(',
        )
        # Match the actual function CALL, not parenthetical mentions in
        # docstrings/comments (api/main.py has a "log_prediction_v2 (when
        # we get there)" parenthetical near the top of predict_symbol
        # that would otherwise match before the real call site).
        log_re = re.compile(r"=\s*log_prediction_v2\s*\(")

        haircut_match = haircut_re.search(src)
        log_match = log_re.search(src)

        self.assertIsNotNone(
            haircut_match,
            "Could not locate the consensus haircut application "
            "(`hdata[\"confidence\"] = _consensus_haircut(...)`) inside "
            "api/main.py. If this assertion fires, the haircut path was "
            "renamed or removed — investigate before relaxing the test.",
        )
        self.assertIsNotNone(
            log_match,
            "Could not locate the log_prediction_v2 call inside "
            "api/main.py. If this assertion fires, the persistence path "
            "was renamed — investigate before relaxing the test.",
        )

        self.assertLess(
            haircut_match.start(),
            log_match.start(),
            "Contrarian haircut runs AFTER log_prediction_v2 in "
            "api/main.py. That means saved Supabase rows carry the "
            "pre-haircut confidence while the live response carries "
            "the post-haircut value — the very bug this test exists to "
            "prevent. Move the consensus_check loop above the "
            "log_prediction_v2 call.",
        )

    def test_consensus_check_stamping_precedes_log_prediction(self):
        """The consensus_check dict that gets stamped on each horizon
        also needs to be present BEFORE the log call, so the saved
        rows can carry the divergence_pp / apply_haircut metadata for
        the ledger UI's "strong contrarian" badge."""
        with open(_API_MAIN, "r", encoding="utf-8") as fh:
            src = fh.read()

        stamp_re = re.compile(r'hdata\["consensus_check"\]\s*=\s*\{')
        # Match the actual call site, not the parenthetical mention in
        # the predict_symbol docstring (see comment in the prior test).
        log_re = re.compile(r"=\s*log_prediction_v2\s*\(")

        stamp_match = stamp_re.search(src)
        log_match = log_re.search(src)

        self.assertIsNotNone(stamp_match)
        self.assertIsNotNone(log_match)
        self.assertLess(
            stamp_match.start(),
            log_match.start(),
            "consensus_check stamping runs AFTER log_prediction_v2 — "
            "saved rows won't carry the contrarian metadata.",
        )


if __name__ == "__main__":
    unittest.main()
