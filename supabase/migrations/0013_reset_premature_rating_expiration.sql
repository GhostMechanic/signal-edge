-- 0013_reset_premature_rating_expiration.sql
--
-- One-shot: revert rating_expiration to 'pending' on TRADED predictions
-- whose horizon hasn't elapsed yet but which got rating_expiration
-- stamped at trade-close time.
--
-- Background
-- ──────────
-- Until commit d1929c2 + this migration, scoring_worker._evaluate_trade
-- wrote rating_expiration='hit' (on target-close) or rating_expiration=
-- 'miss' (on stop-close) at the moment the trade closed. That's wrong
-- when the trade closes early: closing 350 days before the horizon date
-- tells us nothing about where the price will be at horizon_ends_at.
-- The fix is to leave rating_expiration='pending' on early-close and
-- have score_expiration_for_traded() resolve it once the horizon
-- arrives.
--
-- Visible symptom (the NVDA case that surfaced this): a 1-year Bullish
-- NVDA prediction asked 2026-04-27, stopped out on day 4, ledger showed
-- "EXPIRATION: Missed" with 350 days remaining until the actual horizon
-- end (2027-04-27). User correctly flagged that the expiration verdict
-- couldn't possibly be known yet.
--
-- What this script does
-- ─────────────────────
-- For every prediction that:
--   • has a linked model_paper_trade (i.e. was a backed/traded call), AND
--   • has horizon_ends_at > now() (horizon hasn't arrived), AND
--   • has rating_expiration in ('hit', 'miss')
-- set rating_expiration back to 'pending'.
--
-- Going forward, the trade-close path only stamps 'pending' for
-- rating_expiration on early-close, and score_expiration_for_traded()
-- resolves it on the EOD tick after horizon_ends_at elapses.
--
-- Idempotent: the WHERE clause excludes rows whose rating_expiration
-- is already 'pending', so re-running this is a no-op.

UPDATE predictions AS p
SET rating_expiration = 'pending'
WHERE p.horizon_ends_at > now()
  AND p.rating_expiration IN ('hit', 'miss')
  AND EXISTS (
    SELECT 1 FROM model_paper_trades t
    WHERE t.prediction_id = p.id
  );
