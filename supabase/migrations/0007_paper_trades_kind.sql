-- 0007_paper_trades_kind
--
-- Phase B.2 — Extend paper_trades to carry both equity and option
-- positions in a single table.
--
-- Why one table:
--   - Cash math is unified (one paper_portfolios.cash for both kinds)
--   - Position cap (anchor § 7) applies to total open count
--   - Closed-trade ledger reads stay simple (no UNION across tables)
--
-- Shape:
--   `kind` — 'equity' (default for legacy rows) | 'option'
--   `instrument_data` — JSONB with kind-specific fields. NULL for equity.
--
-- For options, instrument_data carries:
--   {
--     "strategy_type": "Bull Call Spread",
--     "direction": "Bullish",
--     "expiry_days": 45,
--     "expiry_date": "2026-06-12",       -- ISO YYYY-MM-DD
--     "legs": [...],                      -- copied from options_strategy
--     "premium_per_contract": 185.42,    -- positive=debit, negative=credit
--     "max_profit_per_contract": 314.58, -- null = unbounded
--     "max_loss_per_contract": 185.42,
--     "breakevens": [243.42],
--     "is_credit": false,
--     "margin_required_per_contract": 0  -- spreads with credit lock margin
--   }
--
-- For equity trades, qty = number of shares. For options, qty = number
-- of contracts (each contract represents 100 shares of the underlying
-- per CBOE spec). This is the same `qty` column reused — the unit
-- depends on `kind`.

ALTER TABLE paper_trades
    ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'equity'
        CHECK (kind IN ('equity', 'option'));

ALTER TABLE paper_trades
    ADD COLUMN IF NOT EXISTS instrument_data jsonb;

COMMENT ON COLUMN paper_trades.kind IS
    'Trade kind: equity (default — entry_price = per-share fill, qty = shares) '
    'or option (entry_price = premium per contract in dollars, qty = contracts; '
    'see instrument_data for legs/expiry/strikes).';

COMMENT ON COLUMN paper_trades.instrument_data IS
    'Kind-specific extra data. NULL for equity. For option: { strategy_type, '
    'direction, expiry_days, expiry_date, legs[], premium_per_contract, '
    'max_profit_per_contract, max_loss_per_contract, breakevens[], is_credit, '
    'margin_required_per_contract }.';

-- Index for filtering open option trades on read (rare today, but the
-- portfolio enrichment loop benefits from a kind-aware fast path).
CREATE INDEX IF NOT EXISTS paper_trades_kind_status_idx
    ON paper_trades (kind, status);
