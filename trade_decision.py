"""
trade_decision.py
-----------------
The TRADE/PASS rule from methodology anchor § 2.1. Pure function — no DB,
no I/O. The caller hands in the prediction's confidence, the symbol, and a
snapshot of the model's portfolio state; this module returns a verdict and
a structured reason.

The five rules in § 2.1 (all must hold for TRADE):

  1. confidence ≥ TRADE_CONFIDENCE_THRESHOLD
  2. symbol in canonical universe
  3. portfolio cash ≥ MIN_POSITION_SIZE
  4. open positions < MAX_OPEN_POSITIONS
  5. no existing OPEN position for this symbol

Plus an implicit zero-th rule from § 4.1: Neutral direction → no trade.

If a parameter changes (e.g. you raise the threshold to 70 after the
200-prediction recalibration), write a `methodology_changes` audit row
before deploying. Every constant in this file is on the public methodology
page.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import AbstractSet


# ─── Parameters from methodology § 2.1 / § 2.3 (locked v1) ────────────────────

TRADE_CONFIDENCE_THRESHOLD: float = 65.0   # § 2.3 — recalibrate at 200 predictions
MIN_POSITION_SIZE:          float = 400.0  # § 3 — $10k / 25 slots
MAX_OPEN_POSITIONS:         int   = 25     # § 2.1 #4

# Minimum reward/risk ratio for an equity trade plan to ship. Below this,
# even a high-confidence directional call is asymmetric in the wrong way:
# the user would have to be right >57% of the time at a 0.76:1 R:R just to
# break even. The methodology suppresses these so the public ledger never
# carries trade plans whose geometry forces the user into a high-accuracy
# corner. The PREDICTION still ships — only the equity trade card is hidden.
# (Methodology § 4.4. Locked at 1.0:1.)
RISK_REWARD_FLOOR:          float = 1.0

# Predicted-return deadband around 0 below which the prediction is
# classified Neutral instead of Bullish/Bearish. The MSFT 1m card on
# Apr 30 surfaced a +0.1% predicted return labeled "Bullish" with 70%
# confidence — that's a high-confidence "nothing happens" call dressed
# up as a directional one, which misleads the user. Anything inside
# ±DEADBAND is now Neutral; the old strict zero-tolerance rule
# (return > 0 → Bullish) is gone.
#
# 0.005 (0.5%) is small enough not to suppress real directional calls
# (a 1% predicted move comfortably clears it) but large enough to catch
# the tiny-magnitude calls that look directional only by sign-of-noise.
# Methodology § 4.1.
NEUTRAL_RETURN_DEADBAND:    float = 0.005


def direction_from_return(
    predicted_return: float,
    *,
    deadband: float = NEUTRAL_RETURN_DEADBAND,
) -> str:
    """Map a predicted return into Bullish / Bearish / Neutral with the
    methodology's deadband applied.

    `predicted_return` is decimal (e.g. 0.05 = 5%). Returns one of the
    three direction strings the rest of the system uses.

    The deadband is symmetric around zero — abs(predicted_return) ≤
    deadband → Neutral. Strictly less than (not ≤) the deadband would
    classify the boundary case as Neutral too, which is what we want
    given that 0.5% predicted is functionally indistinguishable from 0.
    """
    import math
    try:
        r = float(predicted_return)
    except (TypeError, ValueError):
        return _NEUTRAL
    # NaN / inf both fall back to Neutral rather than producing bogus
    # directional labels. abs(nan) <= deadband is False and nan > 0 is
    # False, which would otherwise misclassify NaN as Bearish.
    if not math.isfinite(r):
        return _NEUTRAL
    if abs(r) <= deadband:
        return _NEUTRAL
    return _BULLISH if r > 0 else _BEARISH

# Direction enum kept tiny here on purpose — `trade_plan.py` and downstream
# callers all agree on these strings so we don't drift.
_BULLISH, _BEARISH, _NEUTRAL = "Bullish", "Bearish", "Neutral"


class PassReason(str, Enum):
    """Structured reasons a prediction is PASSED. Logged on the prediction
    row for the audit trail and used to debug 'why didn't the model trade
    this one?' questions on the Track Record page."""
    NEUTRAL_DIRECTION       = "neutral_direction"
    LOW_CONFIDENCE          = "low_confidence"
    SYMBOL_NOT_IN_UNIVERSE  = "symbol_not_in_universe"
    INSUFFICIENT_CASH       = "insufficient_cash"
    BOOK_FULL               = "book_full"
    SYMBOL_ALREADY_OPEN     = "symbol_already_open"
    # The trade-plan geometry (entry/stop/target) gave a reward/risk
    # ratio below RISK_REWARD_FLOOR. Set by the orchestrator
    # (prediqt_open_trade.compute_trade_attachment) before this module's
    # decide_trade gets a chance to weigh in — geometry beats every
    # other check because it's a property of the call itself.
    POOR_RISK_REWARD        = "poor_risk_reward"


@dataclass(frozen=True)
class TradeDecision:
    """The verdict + a human-readable note. `traded=True` means open the
    position; `traded=False` means write the prediction with traded=false
    and skip the model_paper_trades insert."""
    traded: bool
    reason: PassReason | None       # None when traded=True
    note:   str                      # one-line explanation suitable for logs


def decide_trade(
    *,
    confidence:            float,
    symbol:                str,
    direction:             str,
    canonical_universe:    AbstractSet[str],
    portfolio_cash:        float,
    open_positions_count:  int,
    open_symbols:          AbstractSet[str],
    confidence_threshold:  float = TRADE_CONFIDENCE_THRESHOLD,
    min_position_size:     float = MIN_POSITION_SIZE,
    max_open_positions:    int   = MAX_OPEN_POSITIONS,
) -> TradeDecision:
    """
    Decide whether the model commits paper money to this prediction.

    The order of checks here is intentional — fastest/cheapest first, so
    a deluge of low-confidence requests doesn't thrash on universe-set
    membership lookups before bailing on confidence. It also produces
    the most informative `reason` for partial failures (we report the
    first failing rule).

    All set-membership comparisons are uppercase to avoid drift between
    'aapl' / 'AAPL' / ' AAPL ' inputs.
    """
    sym = symbol.strip().upper()

    if direction == _NEUTRAL:
        return TradeDecision(
            traded=False,
            reason=PassReason.NEUTRAL_DIRECTION,
            note="neutral predictions never trade (§ 4.1)",
        )

    if confidence < confidence_threshold:
        return TradeDecision(
            traded=False,
            reason=PassReason.LOW_CONFIDENCE,
            note=f"confidence {confidence:.1f} < {confidence_threshold}",
        )

    if sym not in canonical_universe:
        return TradeDecision(
            traded=False,
            reason=PassReason.SYMBOL_NOT_IN_UNIVERSE,
            note=f"{sym} not in canonical universe",
        )

    if portfolio_cash < min_position_size:
        return TradeDecision(
            traded=False,
            reason=PassReason.INSUFFICIENT_CASH,
            note=f"cash ${portfolio_cash:.2f} < ${min_position_size:.2f}",
        )

    if open_positions_count >= max_open_positions:
        return TradeDecision(
            traded=False,
            reason=PassReason.BOOK_FULL,
            note=f"book full ({open_positions_count}/{max_open_positions})",
        )

    if sym in {s.strip().upper() for s in open_symbols}:
        return TradeDecision(
            traded=False,
            reason=PassReason.SYMBOL_ALREADY_OPEN,
            note=f"open position already exists for {sym}",
        )

    return TradeDecision(
        traded=True,
        reason=None,
        note=f"all rules pass (confidence={confidence:.1f})",
    )


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    UNIVERSE = frozenset({"AAPL", "MSFT", "NVDA", "SPY", "QQQ"})

    base_kwargs = dict(
        canonical_universe   = UNIVERSE,
        portfolio_cash       = 1000.0,
        open_positions_count = 5,
        open_symbols         = frozenset({"AAPL"}),
    )

    # Happy path: trades.
    d = decide_trade(confidence=72, symbol="NVDA", direction="Bullish", **base_kwargs)
    assert d.traded is True
    print(f"OK trade: {d.note}")

    # Neutral direction: pass.
    d = decide_trade(confidence=99, symbol="NVDA", direction="Neutral", **base_kwargs)
    assert d.traded is False and d.reason == PassReason.NEUTRAL_DIRECTION
    print(f"OK neutral pass: {d.note}")

    # Low confidence: pass.
    d = decide_trade(confidence=64, symbol="NVDA", direction="Bullish", **base_kwargs)
    assert d.traded is False and d.reason == PassReason.LOW_CONFIDENCE
    print(f"OK low-conf pass: {d.note}")

    # Symbol off-universe: pass.
    d = decide_trade(confidence=80, symbol="CRWV", direction="Bullish", **base_kwargs)
    assert d.traded is False and d.reason == PassReason.SYMBOL_NOT_IN_UNIVERSE
    print(f"OK off-universe pass: {d.note}")

    # Insufficient cash.
    d = decide_trade(
        confidence=80, symbol="MSFT", direction="Bullish",
        canonical_universe=UNIVERSE,
        portfolio_cash=399.99,
        open_positions_count=5,
        open_symbols=frozenset(),
    )
    assert d.traded is False and d.reason == PassReason.INSUFFICIENT_CASH
    print(f"OK low-cash pass: {d.note}")

    # Book full.
    d = decide_trade(
        confidence=80, symbol="MSFT", direction="Bullish",
        canonical_universe=UNIVERSE,
        portfolio_cash=1000,
        open_positions_count=25,
        open_symbols=frozenset(),
    )
    assert d.traded is False and d.reason == PassReason.BOOK_FULL
    print(f"OK book-full pass: {d.note}")

    # Symbol already open.
    d = decide_trade(confidence=80, symbol="AAPL", direction="Bullish", **base_kwargs)
    assert d.traded is False and d.reason == PassReason.SYMBOL_ALREADY_OPEN
    print(f"OK already-open pass: {d.note}")

    # Case-insensitive universe match.
    d = decide_trade(confidence=80, symbol="nvda", direction="Bullish", **base_kwargs)
    assert d.traded is True
    print(f"OK case-insensitive trade: {d.note}")

    # Case-insensitive open-symbols match (lowercase entry shouldn't bypass guard).
    d = decide_trade(
        confidence=80, symbol="AAPL", direction="Bullish",
        canonical_universe=UNIVERSE,
        portfolio_cash=1000,
        open_positions_count=5,
        open_symbols=frozenset({"aapl"}),
    )
    assert d.traded is False and d.reason == PassReason.SYMBOL_ALREADY_OPEN
    print(f"OK case-insensitive guard: {d.note}")

    print("trade_decision.py self-test OK")
