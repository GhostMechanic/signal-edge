-- 0011_symbol_snapshots
--
-- Per-symbol nightly model snapshots — feeds the SEO-driven /stock/[symbol]
-- pages. Separate from `predictions` for several reasons:
--
--   1. The `predictions` table is the user-asked / public-ledger surface.
--      Mixing in 600 nightly batch rows per day would obliterate the
--      "honest live record" narrative — users would see thousands of model
--      calls per week and the public ledger would feel like noise.
--
--   2. Snapshot rows are intentionally NOT scored against the methodology
--      track record. They're inference-only artefacts: today's model
--      view, persisted for tomorrow's page render. We don't pretend the
--      model "committed paper money" on every S&P 500 ticker every night.
--
--   3. Different write cadence + ownership: predictions are user-authored
--      (one user_id per row), snapshots are system-authored (no user_id).
--      Different RLS posture too — public READ for everyone (these power
--      anonymous-visitor pages), service-role-only WRITE.
--
--   4. Different read pattern: the page only ever wants the LATEST row
--      per (symbol, horizon) combo. A small dedicated table with a tight
--      composite index makes that lookup ~free; the same query against
--      predictions would have to filter past snapshots-as-noise out.
--
-- Shape mirrors `predictions` so the page components can render either
-- snapshot rows or real user predictions without branching.
--
-- Methodology compliance: § 5 ("the public ledger") only counts user
-- predictions. Snapshots live outside the methodology surface.

CREATE TABLE IF NOT EXISTS symbol_snapshots (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol          text         NOT NULL,
    snapshot_date   date         NOT NULL,
    horizon         text         NOT NULL,           -- '3d' | '1w' | '1m' | '1q' | '1y'

    -- Model output — same fields the predictions table carries.
    direction        text,                            -- 'Bullish' | 'Bearish' | 'Neutral'
    confidence       double precision,
    predicted_price  double precision,
    predicted_return double precision,
    current_price    double precision,                -- close on snapshot_date
    horizon_end      date,
    horizon_days     int,

    -- Trade plan (entry/stop/target) for completeness — useful when a
    -- visitor lands and we want to render the "if the model traded today"
    -- block without hitting predict() again.
    entry_price   double precision,
    stop_price    double precision,
    target_price  double precision,
    traded        boolean,
    trade_pass_reason text,

    -- Per-horizon options strategy snapshot (mirrors predictions.options_strategy).
    options_strategy jsonb,

    -- Metadata for cache invalidation + audit.
    model_version text,
    regime        text,
    signals_summary jsonb,                            -- compact list of named signals + values

    created_at timestamptz NOT NULL DEFAULT now(),

    -- One row per (symbol, snapshot_date, horizon). The nightly batch is
    -- idempotent — re-running it on the same day overwrites in place
    -- rather than appending duplicates.
    UNIQUE (symbol, snapshot_date, horizon)
);

-- Hot path: "give me the latest snapshot for AAPL across all horizons"
-- (the page loader). Index supports `WHERE symbol = $1 ORDER BY
-- snapshot_date DESC LIMIT 5` cheaply.
CREATE INDEX IF NOT EXISTS idx_symbol_snapshots_symbol_date
    ON symbol_snapshots (symbol, snapshot_date DESC);

-- Secondary: "list all snapshots since X" for the model's running
-- per-symbol track-record components.
CREATE INDEX IF NOT EXISTS idx_symbol_snapshots_date
    ON symbol_snapshots (snapshot_date DESC);

-- ── RLS: public read, service-role write only ─────────────────────────────
ALTER TABLE symbol_snapshots ENABLE ROW LEVEL SECURITY;

-- Anyone (anon, authed, service) can SELECT — these power public pages.
DROP POLICY IF EXISTS symbol_snapshots_public_read ON symbol_snapshots;
CREATE POLICY symbol_snapshots_public_read
    ON symbol_snapshots
    FOR SELECT
    USING (true);

-- Service role only writes. The nightly batch runs from GitHub Actions
-- with the service-role key; no user-side path should ever insert here.
-- (Service role bypasses RLS by design — this stays explicit so anyone
-- reading the migration knows writes are not user-driven.)
DROP POLICY IF EXISTS symbol_snapshots_service_write ON symbol_snapshots;
CREATE POLICY symbol_snapshots_service_write
    ON symbol_snapshots
    FOR INSERT
    WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS symbol_snapshots_service_update ON symbol_snapshots;
CREATE POLICY symbol_snapshots_service_update
    ON symbol_snapshots
    FOR UPDATE
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

COMMENT ON TABLE symbol_snapshots IS
    'Nightly per-symbol model snapshots powering the /stock/[symbol] SEO '
    'pages. NOT counted toward methodology § 5 track-record (those use the '
    'user-authored `predictions` table only). Public read, service-role '
    'write only. One row per (symbol, snapshot_date, horizon).';
