-- 0006_options_strategy_on_predictions
--
-- Phase B.1 — Persist the options strategy that was recommended for
-- each prediction at the time it was made.
--
-- Why: the asking flow at /prediqt computes a per-horizon options
-- strategy (Bull Call Spread, Bear Put Spread, Iron Condor, etc.) but
-- never persists it. When a user later opens the prediction's detail
-- page on Track Record and clicks "Take this play," there's nothing to
-- paper-trade against — strikes, premium, breakeven all live only in
-- the live response and disappear once the user navigates away.
--
-- This migration adds an `options_strategy` JSONB column on
-- `predictions` that stores the strategy recommended for THAT row's
-- horizon. Shape mirrors options_analyzer.py's strategy dict, with the
-- new `numeric` block (cost_per_contract, max_profit_per_contract,
-- max_loss_per_contract, breakevens, is_credit, tradable, optional
-- margin_required_per_contract).
--
-- Backward compatibility: NULL on existing rows. Inserts going forward
-- attach the strategy at write time. The OptionsPlayCard on the
-- detail page falls back to its existing illustrative empty state when
-- options_strategy IS NULL.

ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS options_strategy jsonb;

COMMENT ON COLUMN predictions.options_strategy IS
    'Per-horizon options strategy recommendation captured at predict-time. '
    'Shape: { strategy, direction, complexity, expiry_days, legs[], '
    'estimated_cost (formatted), max_profit (formatted), max_loss (formatted), '
    'breakeven (formatted), iv_used (formatted), rationale, risk_note, '
    'numeric: { cost_per_contract, max_profit_per_contract, max_loss_per_contract, '
    'breakevens[], is_credit, tradable, margin_required_per_contract? } }. '
    'See options_analyzer.py for the canonical generator.';
