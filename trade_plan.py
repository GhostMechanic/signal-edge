"""
trade_plan.py
-------------
Builds the (entry, stop, target) trade plan for a single prediction
following methodology anchor § 4.4. Pure function — given the inputs, the
output is deterministic and reproducible from public OHLC data.

Stop placement is volatility-aware (2× ATR₁₄), bounded between 1.5% and
8% of entry. Target is the model's predicted price, unmodified.

The dataclass returned here is what gets written into the `predictions`
row's (entry_price, stop_price, target_price) columns and into the
matching `model_paper_trades` row.

If you change the parameter values below, write a `methodology_changes`
audit row before deploying — every knob in this file is on the public
methodology page.
"""

from __future__ import annotations

from dataclasses import dataclass


# ─── Parameters from methodology § 4.4 (locked v1) ────────────────────────────

# Stop = clamp(ATR_MULTIPLIER × ATR₁₄, STOP_FLOOR_PCT × entry, STOP_CEILING_PCT × entry)
ATR_MULTIPLIER:    float = 2.0     # k — Wilder's standard
STOP_FLOOR_PCT:    float = 0.015   # 1.5% — protects against intraday whipsaw
STOP_CEILING_PCT:  float = 0.080   # 8.0% — caps single-trade damage

# Direction strings the model emits. Anything else raises.
_BULLISH = "Bullish"
_BEARISH = "Bearish"
_NEUTRAL = "Neutral"


@dataclass(frozen=True)
class TradePlan:
    """The three locked-at-open levels that drive the trade for its lifetime."""
    entry_price:  float
    stop_price:   float
    target_price: float
    # The components used to build the stop, kept on the plan for the
    # audit story the Track Record page tells. Not strictly needed for
    # trading but useful for "show me the math" on a single prediction.
    atr_distance:     float
    floor_distance:   float
    ceiling_distance: float
    stop_distance:    float


def build_trade_plan(
    *,
    entry_price:     float,
    predicted_price: float,
    atr_14:          float,
    direction:       str,
    atr_multiplier:    float = ATR_MULTIPLIER,
    stop_floor_pct:    float = STOP_FLOOR_PCT,
    stop_ceiling_pct:  float = STOP_CEILING_PCT,
) -> TradePlan:
    """
    Compute (entry, stop, target) for a single prediction.

    Parameters
    ----------
    entry_price     : last known price at the moment of prediction.
    predicted_price : the model's own forecasted price for the horizon.
    atr_14          : Wilder's 14-period ATR for the symbol, in price units.
    direction       : "Bullish" → LONG, "Bearish" → SHORT.
                      "Neutral" raises because the TRADE rule (§ 2.1) says
                      neutral predictions never generate a trade.

    Optional knobs (defaults match methodology § 4.4):
    atr_multiplier  : k in (k × ATR). Default 2.0.
    stop_floor_pct  : minimum stop distance as fraction of entry. Default 1.5%.
    stop_ceiling_pct: maximum stop distance as fraction of entry. Default 8%.

    Returns
    -------
    TradePlan with entry / stop / target rounded to 4 decimal places to
    match the schema's numeric(14, 4).

    Raises
    ------
    ValueError if any input is non-positive or the direction is unsupported.
    """
    if direction == _NEUTRAL:
        raise ValueError(
            "Neutral predictions cannot have a trade plan — "
            "the TRADE rule (§ 2.1) skips them."
        )
    if direction not in (_BULLISH, _BEARISH):
        raise ValueError(f"Unsupported direction: {direction!r}")

    if entry_price <= 0:
        raise ValueError(f"entry_price must be positive; got {entry_price}")
    if predicted_price <= 0:
        raise ValueError(f"predicted_price must be positive; got {predicted_price}")
    if atr_14 < 0:
        raise ValueError(f"atr_14 must be non-negative; got {atr_14}")

    # Stop distance — the canonical clamp.
    atr_distance     = atr_multiplier * atr_14
    floor_distance   = stop_floor_pct * entry_price
    ceiling_distance = stop_ceiling_pct * entry_price
    stop_distance    = max(floor_distance, min(atr_distance, ceiling_distance))

    if direction == _BULLISH:
        stop_price = entry_price - stop_distance
    else:  # Bearish
        stop_price = entry_price + stop_distance

    # Coerce to plain Python float — `atr_14` may arrive as numpy.float64
    # which infects every downstream computation. JSON serializers (Supabase
    # REST, fastapi) and SQLite parameter binding both prefer plain floats.
    return TradePlan(
        entry_price      = float(round(entry_price,      4)),
        stop_price       = float(round(stop_price,       4)),
        target_price     = float(round(predicted_price,  4)),
        atr_distance     = float(round(atr_distance,     4)),
        floor_distance   = float(round(floor_distance,   4)),
        ceiling_distance = float(round(ceiling_distance, 4)),
        stop_distance    = float(round(stop_distance,    4)),
    )


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Case 1 — ATR-bound stop (the typical case).
    # Entry $100, ATR $1.50 → 2×ATR = $3.00. Floor = $1.50 (1.5%). Ceiling = $8.00 (8%).
    # Stop distance = clamp($3.00, $1.50, $8.00) = $3.00. Stop = $97.00 for LONG.
    plan = build_trade_plan(
        entry_price     = 100.0,
        predicted_price = 110.0,
        atr_14          = 1.5,
        direction       = "Bullish",
    )
    assert plan.entry_price  == 100.0,  plan
    assert plan.target_price == 110.0,  plan
    assert plan.stop_price   ==  97.0,  plan
    assert plan.stop_distance ==  3.0,  plan
    print(f"Case 1 (ATR-bound, LONG): stop={plan.stop_price}, target={plan.target_price}")

    # Case 2 — Floor-bound (very low volatility name).
    # Entry $100, ATR $0.20 → 2×ATR = $0.40. Floor = $1.50. Stop = $1.50.
    plan = build_trade_plan(
        entry_price     = 100.0,
        predicted_price = 105.0,
        atr_14          = 0.20,
        direction       = "Bullish",
    )
    assert plan.stop_distance == 1.5,  plan
    assert plan.stop_price    == 98.5, plan
    print(f"Case 2 (floor-bound, LONG): stop_distance={plan.stop_distance}")

    # Case 3 — Ceiling-bound (high-vol name).
    # Entry $100, ATR $5.0 → 2×ATR = $10.00. Ceiling = $8.00. Stop = $8.00.
    plan = build_trade_plan(
        entry_price     = 100.0,
        predicted_price = 120.0,
        atr_14          = 5.0,
        direction       = "Bullish",
    )
    assert plan.stop_distance == 8.0, plan
    assert plan.stop_price   == 92.0, plan
    print(f"Case 3 (ceiling-bound, LONG): stop_distance={plan.stop_distance}")

    # Case 4 — SHORT: stop is above entry.
    # Entry $100, ATR $1.5, Bearish. Stop distance $3.00, stop at $103.00.
    plan = build_trade_plan(
        entry_price     = 100.0,
        predicted_price = 90.0,
        atr_14          = 1.5,
        direction       = "Bearish",
    )
    assert plan.stop_price == 103.0,   plan
    assert plan.target_price == 90.0, plan
    print(f"Case 4 (SHORT): stop={plan.stop_price}, target={plan.target_price}")

    # Case 5 — Neutral direction must raise.
    try:
        build_trade_plan(
            entry_price=100, predicted_price=100, atr_14=1, direction="Neutral",
        )
    except ValueError as e:
        print(f"Neutral guard: {e}")
    else:
        raise AssertionError("expected ValueError for Neutral direction")

    # Case 6 — Bad inputs.
    for bad in [
        dict(entry_price=-1.0, predicted_price=10, atr_14=1, direction="Bullish"),
        dict(entry_price=10, predicted_price=0, atr_14=1, direction="Bullish"),
        dict(entry_price=10, predicted_price=10, atr_14=-1, direction="Bullish"),
        dict(entry_price=10, predicted_price=10, atr_14=1, direction="Sideways"),
    ]:
        try:
            build_trade_plan(**bad)
        except ValueError as e:
            print(f"Bad-input guard ({list(bad.values())[-1]}): {e}")
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")

    print("trade_plan.py self-test OK")
