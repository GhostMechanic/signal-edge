-- 0017_checkpoint_scores
--
-- Per-interval directional-accuracy scores for predictions, introduced
-- as part of the Checkpoint scoring migration (see methodology § 4.3
-- after this migration lands).
--
-- The OLD checkpoint definition was "did the price touch the midpoint
-- between entry and target during the window?" — a single binary hit
-- per prediction, computed inline in scoring_worker._evaluate_trade()
-- and _evaluate_open_prediction(). It was a magnitude check, in the
-- same family as Target.
--
-- The NEW checkpoint definition is "at each scheduled interval since
-- issue, was the close price on the correct side of entry?" — multiple
-- binary observations per prediction, one row per interval. It's a
-- *directional persistence* check, an axis distinct from Target
-- (magnitude) and Expiration (terminal direction).
--
-- Per-horizon schedule (mirrors HORIZON_INTERVALS in scoring_worker.py):
--
--    3 Day     →  1d
--    1 Week    →  1d, 3d
--    1 Month   →  1d, 7d, 14d
--    1 Quarter →  1d, 7d, 30d, 60d
--    1 Year    →  1d, 7d, 30d, 90d, 180d, 365d
--
-- Per-call binary verdict (the value written back to
-- predictions.rating_checkpoint after this migration): the status of
-- the most recently fired interval. "Most recent stands" matches the
-- existing ThreeReads / PerformanceSummary copy on the frontend and
-- gives users a sensible "where is the call right now?" read.
--
-- Detailed per-interval history is preserved here so the receipt's
-- Checkpoints tab can show the trajectory (hit at 1d, miss at 7d, etc.)
-- and analytics surfaces can recompute aggregations under different
-- policies without re-scanning yfinance.
--
-- Intervals continue firing even after the linked model_paper_trade
-- closes on target/stop. Checkpoint measures directional accuracy of
-- the *call* across its full horizon, independent of trade outcome —
-- a winning trade that exited on day 5 of a 1-year horizon can still
-- demonstrate the model's call was *consistently* right over the
-- following 360 days. (Or that it wasn't — equally informative.)
--
-- Neutral predictions are skipped: there's no directional check to make
-- against entry. Their rating_checkpoint stays NULL.

CREATE TABLE IF NOT EXISTS prediction_checkpoint_scores (
    id              bigserial   PRIMARY KEY,
    prediction_id   uuid        NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    interval_days   int         NOT NULL,
    scored_at       timestamptz NOT NULL DEFAULT now(),
    status          text        NOT NULL CHECK (status IN ('hit', 'miss')),
    close_price     numeric     NOT NULL,

    -- Hard idempotency guarantee: a single (prediction, interval) pair
    -- can only be scored once. Re-runs of tick_checkpoints() that
    -- attempt to re-score an already-scored interval will hit this
    -- constraint and be no-ops by design.
    UNIQUE (prediction_id, interval_days)
);

-- Hot path: "give me all checkpoint scores for this prediction in
-- interval order" — used by the receipt's Checkpoints tab and by the
-- rating_checkpoint update in tick_checkpoints (which needs the latest
-- interval). The composite index covers both ORDER BY interval_days
-- and equality on prediction_id.
CREATE INDEX IF NOT EXISTS idx_checkpoint_scores_prediction_interval
    ON prediction_checkpoint_scores (prediction_id, interval_days DESC);

-- Secondary: "how many scores fired today" / analytics over time. The
-- scored_at index supports the worker's per-run telemetry without
-- scanning the whole table.
CREATE INDEX IF NOT EXISTS idx_checkpoint_scores_scored_at
    ON prediction_checkpoint_scores (scored_at DESC);

-- ── RLS: public read, service-role write only ─────────────────────────────
-- Same posture as predictions (public ledger reads are open, writes go
-- through the worker with service_role). Frontend reads the trajectory
-- without auth; the receipt's Checkpoints tab uses the anon key.
ALTER TABLE prediction_checkpoint_scores ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS checkpoint_scores_public_read ON prediction_checkpoint_scores;
CREATE POLICY checkpoint_scores_public_read
    ON prediction_checkpoint_scores
    FOR SELECT
    USING (true);

DROP POLICY IF EXISTS checkpoint_scores_service_write ON prediction_checkpoint_scores;
CREATE POLICY checkpoint_scores_service_write
    ON prediction_checkpoint_scores
    FOR INSERT
    WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS checkpoint_scores_service_update ON prediction_checkpoint_scores;
CREATE POLICY checkpoint_scores_service_update
    ON prediction_checkpoint_scores
    FOR UPDATE
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- ── Enable realtime ──────────────────────────────────────────────────────
-- The receipt's Checkpoints tab subscribes to per-row inserts so the
-- trajectory updates live as new intervals fire. Mirrors the realtime
-- setup added in 0016_enable_realtime_publication for predictions /
-- paper_trades.
ALTER PUBLICATION supabase_realtime ADD TABLE prediction_checkpoint_scores;

COMMENT ON TABLE prediction_checkpoint_scores IS
    'Per-interval directional-accuracy scores for predictions. One row per '
    '(prediction_id, interval_days). rating_checkpoint on the predictions '
    'table is derived from the latest interval here ("most recent stands"). '
    'See methodology § 4.3 and scoring_worker.tick_checkpoints().';
