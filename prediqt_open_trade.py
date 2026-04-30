"""
prediqt_open_trade.py
---------------------
The orchestrator that wires the three pure modules (indicators, trade_plan,
trade_decision) and the portfolio repository (model_portfolio) into the
prediction-insertion path.

Two-phase API. The prediction-insertion path needs to land the prediction
row BEFORE the model_paper_trades row (so the FK is satisfied), so we
expose two distinct functions:

  • compute_trade_attachment(...) — pure logic. Computes the trade plan
    and the TRADE/PASS decision based on a portfolio snapshot. No DB
    writes. Output drives the fields that go on the prediction row.

  • open_model_trade_for_prediction(...) — runs only when the decision
    was TRADE. Calls the portfolio repo to atomically insert the
    model_paper_trades row + decrement cash. Must be called AFTER the
    prediction row exists.

  • attach_trade_metadata(...) — backward-compat wrapper that does both
    in one call (used by tests + the in-memory path where there's no
    real FK to satisfy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Optional

from indicators       import atr_from_ohlc_df
from trade_plan       import TradePlan, build_trade_plan
from trade_decision   import (
    TradeDecision, PassReason, decide_trade, RISK_REWARD_FLOOR,
)
from model_portfolio  import (
    ModelPortfolioRepo, OpenTradeRequest, OpenTradeResult, PortfolioState,
)


def _plan_risk_reward(plan: TradePlan, direction: str) -> Optional[float]:
    """Return reward/risk for the plan, or None when the direction isn't
    tradeable (Neutral) or the geometry is degenerate (zero risk).

    For a LONG: reward = target − entry, risk = entry − stop.
    For a SHORT: reward = entry − target, risk = stop − entry.
    Both reduce to abs(target − entry) / abs(entry − stop), which is
    direction-agnostic and matches the R:R the frontend already shows.
    """
    if direction not in ("Bullish", "Bearish"):
        return None
    risk = abs(plan.entry_price - plan.stop_price)
    reward = abs(plan.target_price - plan.entry_price)
    if risk <= 0:
        return None
    return reward / risk


@dataclass(frozen=True)
class TradeAttachResult:
    """Backward-compat aggregate — what the in-memory test paths return."""
    plan:        TradePlan
    decision:    TradeDecision
    open_result: Optional[OpenTradeResult]   # None if PASSED


@dataclass(frozen=True)
class TradeComputeResult:
    """Phase-1 output: everything we know without doing any DB writes.
    The plan is always populated; the decision tells the caller whether
    Phase 2 (open_model_trade_for_prediction) should run."""
    plan:     TradePlan
    decision: TradeDecision


# ─── Phase 1: pure logic ──────────────────────────────────────────────────────

def compute_trade_attachment(
    *,
    symbol:             str,
    direction:          str,           # 'Bullish' | 'Bearish' | 'Neutral'
    confidence:         float,         # 0..100
    entry_price:        float,
    predicted_price:    float,
    ohlc_for_atr,                      # pd.DataFrame with High/Low/Close cols
    canonical_universe: AbstractSet[str],
    portfolio_state:    PortfolioState,
) -> TradeComputeResult:
    """
    Compute the trade plan and the TRADE/PASS decision. No DB writes.

    The portfolio_state is a snapshot — the caller is responsible for
    pulling it from the repo (or mocking it for tests). Because this
    function is pure, you can run it without a real Supabase connection.
    """
    # 1. Trade plan. Neutral predictions get a flat synthetic plan so the
    #    prediction row's not-null guarantees aren't surprised.
    if direction == "Neutral":
        plan = TradePlan(
            entry_price      = round(entry_price, 4),
            stop_price       = round(entry_price, 4),
            target_price     = round(entry_price, 4),
            atr_distance     = 0.0,
            floor_distance   = 0.0,
            ceiling_distance = 0.0,
            stop_distance    = 0.0,
        )
    else:
        atr_14 = atr_from_ohlc_df(ohlc_for_atr, period=14)
        plan = build_trade_plan(
            entry_price     = entry_price,
            predicted_price = predicted_price,
            atr_14          = atr_14,
            direction       = direction,
        )

    # 2. R:R floor — geometry gate (methodology § 4.4).
    #
    # If the ATR-clamped stop/target produce reward/risk below the floor,
    # the equity trade plan is suppressed BEFORE decide_trade gets a chance
    # to weigh in. This is order-sensitive: a low-R:R call could also be
    # low-confidence or off-universe, but the geometry reason is the most
    # informative one to surface — fix the floor and the rest become moot.
    rr = _plan_risk_reward(plan, direction)
    if rr is not None and rr < RISK_REWARD_FLOOR:
        return TradeComputeResult(
            plan=plan,
            decision=TradeDecision(
                traded=False,
                reason=PassReason.POOR_RISK_REWARD,
                note=(
                    f"R:R {rr:.2f}:1 below floor of "
                    f"{RISK_REWARD_FLOOR:.2f}:1 — equity trade suppressed"
                ),
            ),
        )

    # 3. TRADE/PASS decision.
    decision = decide_trade(
        confidence           = confidence,
        symbol               = symbol,
        direction            = direction,
        canonical_universe   = canonical_universe,
        portfolio_cash       = portfolio_state.cash,
        open_positions_count = portfolio_state.open_positions_count,
        open_symbols         = portfolio_state.open_symbols,
    )

    return TradeComputeResult(plan=plan, decision=decision)


# ─── Phase 2: DB-side, only when decision says TRADE ──────────────────────────

def open_model_trade_for_prediction(
    *,
    prediction_id:    str,
    symbol:           str,
    direction:        str,                  # 'Bullish' | 'Bearish'
    plan:             TradePlan,
    portfolio_repo:   ModelPortfolioRepo,
    portfolio_state:  PortfolioState,       # the same snapshot used in Phase 1
) -> OpenTradeResult:
    """
    Open the model paper trade for a prediction that has already been
    persisted. Expects the caller to have inserted the predictions row
    first so the model_paper_trades.prediction_id FK is satisfied.

    Equal-weight sizing per methodology § 3:
        notional = starting_capital / max_open_positions
        qty      = notional / entry_price

    Raises whatever the portfolio repo raises on race-condition failures.
    """
    if direction not in ("Bullish", "Bearish"):
        raise ValueError(f"open_model_trade_for_prediction: bad direction {direction!r}")

    notional = round(portfolio_state.starting_capital / portfolio_state.max_open_positions, 2)
    qty      = round(notional / plan.entry_price, 4)
    direction_long_short = "LONG" if direction == "Bullish" else "SHORT"

    return portfolio_repo.open_trade(OpenTradeRequest(
        prediction_id = prediction_id,
        symbol        = symbol.strip().upper(),
        direction     = direction_long_short,
        entry_price   = plan.entry_price,
        target_price  = plan.target_price,
        stop_price    = plan.stop_price,
        qty           = qty,
        notional      = notional,
    ))


# ─── Backward-compat one-shot wrapper ─────────────────────────────────────────

def attach_trade_metadata(
    *,
    symbol:             str,
    direction:          str,
    confidence:         float,
    entry_price:        float,
    predicted_price:    float,
    ohlc_for_atr,
    canonical_universe: AbstractSet[str],
    portfolio_repo:     ModelPortfolioRepo,
    prediction_id:      str,
) -> TradeAttachResult:
    """
    Backward-compat wrapper: do both phases in one call. Use this when
    there's no real FK constraint between the prediction row and the
    model_paper_trades row (in-memory tests, single-process scripts).

    Production paths should call compute_trade_attachment() and
    open_model_trade_for_prediction() separately so the prediction row
    can land between them and satisfy the FK.
    """
    state = portfolio_repo.get_state()
    compute = compute_trade_attachment(
        symbol             = symbol,
        direction          = direction,
        confidence         = confidence,
        entry_price        = entry_price,
        predicted_price    = predicted_price,
        ohlc_for_atr       = ohlc_for_atr,
        canonical_universe = canonical_universe,
        portfolio_state    = state,
    )

    if not compute.decision.traded:
        return TradeAttachResult(
            plan=compute.plan, decision=compute.decision, open_result=None
        )

    open_result = open_model_trade_for_prediction(
        prediction_id   = prediction_id,
        symbol          = symbol,
        direction       = direction,
        plan            = compute.plan,
        portfolio_repo  = portfolio_repo,
        portfolio_state = state,
    )

    return TradeAttachResult(
        plan=compute.plan, decision=compute.decision, open_result=open_result
    )


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from model_portfolio import InMemoryPortfolioRepo

    UNIVERSE = frozenset({"NVDA", "AAPL", "MSFT", "SPY"})

    # Build a tiny OHLC frame — 20 bars, NVDA-flavored. ATR will be ~$2.
    bars = []
    base = 800.0
    for i in range(20):
        h = base + 2.0 + (i % 3)
        l = base - 1.5 - ((i + 1) % 2)
        c = base + 0.5
        bars.append({"High": h, "Low": l, "Close": c})
        base += 0.5
    ohlc = pd.DataFrame(bars)

    repo = InMemoryPortfolioRepo()

    # 1) Confident bullish call → TRADE.
    res = attach_trade_metadata(
        symbol="NVDA", direction="Bullish", confidence=72,
        entry_price=810.0, predicted_price=880.0,
        ohlc_for_atr=ohlc,
        canonical_universe=UNIVERSE,
        portfolio_repo=repo,
        prediction_id="pred-A",
    )
    assert res.decision.traded is True, res.decision
    assert res.open_result is not None
    assert res.open_result.new_cash == 9_600.00
    print(f"TRADE: plan stop={res.plan.stop_price}, target={res.plan.target_price}, "
          f"new_cash={res.open_result.new_cash}")

    # 2) Same NVDA again — passes on "symbol_already_open".
    res = attach_trade_metadata(
        symbol="NVDA", direction="Bullish", confidence=72,
        entry_price=815.0, predicted_price=890.0,
        ohlc_for_atr=ohlc,
        canonical_universe=UNIVERSE,
        portfolio_repo=repo,
        prediction_id="pred-B",
    )
    assert res.decision.traded is False
    assert res.decision.reason == PassReason.SYMBOL_ALREADY_OPEN
    # Plan still computed for the paid-tier view.
    assert res.plan.stop_price < res.plan.entry_price
    print(f"PASS (already open): note={res.decision.note}; "
          f"plan still rendered: entry={res.plan.entry_price}, stop={res.plan.stop_price}")

    # 3) Low confidence — passes, plan still rendered.
    res = attach_trade_metadata(
        symbol="MSFT", direction="Bullish", confidence=58,
        entry_price=420.0, predicted_price=440.0,
        ohlc_for_atr=ohlc,
        canonical_universe=UNIVERSE,
        portfolio_repo=repo,
        prediction_id="pred-C",
    )
    assert res.decision.traded is False
    assert res.decision.reason == PassReason.LOW_CONFIDENCE
    print(f"PASS (low conf): note={res.decision.note}")

    # 4) Neutral direction — passes, synthetic plan.
    res = attach_trade_metadata(
        symbol="AAPL", direction="Neutral", confidence=85,
        entry_price=200.0, predicted_price=200.0,
        ohlc_for_atr=ohlc,
        canonical_universe=UNIVERSE,
        portfolio_repo=repo,
        prediction_id="pred-D",
    )
    assert res.decision.traded is False
    assert res.decision.reason == PassReason.NEUTRAL_DIRECTION
    assert res.plan.entry_price == res.plan.stop_price == res.plan.target_price
    print(f"PASS (neutral): note={res.decision.note}; flat plan as expected")

    # 5) Two-phase API directly — exercising the new path that the Supabase
    #    write flow uses (compute → persist prediction → open trade).
    repo2 = InMemoryPortfolioRepo()
    state = repo2.get_state()
    compute = compute_trade_attachment(
        symbol="SPY", direction="Bullish", confidence=72,
        entry_price=500.0, predicted_price=520.0,
        ohlc_for_atr=ohlc,
        canonical_universe=UNIVERSE | {"SPY"},
        portfolio_state=state,
    )
    assert compute.decision.traded is True
    # In production, the prediction would be persisted here, between phases.
    open_res = open_model_trade_for_prediction(
        prediction_id="pred-E",
        symbol="SPY", direction="Bullish",
        plan=compute.plan,
        portfolio_repo=repo2,
        portfolio_state=state,
    )
    assert open_res.new_cash == 9_600.00
    print(f"Two-phase: plan={compute.plan.target_price}, opened={open_res.trade_id[:8]}…, "
          f"new_cash={open_res.new_cash}")

    print("prediqt_open_trade.py self-test OK")
