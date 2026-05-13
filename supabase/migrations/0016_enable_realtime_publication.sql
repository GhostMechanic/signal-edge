-- 0016_enable_realtime_publication.sql
--
-- Enable Supabase Realtime change-data-capture on the predictions and
-- model_paper_trades tables so the front-end can receive sub-second
-- push notifications when rows are inserted or updated.
--
-- Without this membership the supabase_realtime publication doesn't
-- replicate changes from these tables, and the lib/queries.ts
-- useRealtimePredictions() subscription silently degrades to no-op:
-- the channel stays open but no events fire. The polling fallback in
-- usePublicLedger / useUserPaperPortfolio keeps things working, but
-- the UX is "wait up to 5-15s for the next poll" instead of "see it
-- instantly".
--
-- Two tables included:
--   • predictions       — drives the public ledger + /me + receipt
--   • model_paper_trades — drives YourBook + portfolio mark-to-market
--
-- Idempotent guard: ALTER PUBLICATION ADD TABLE errors if the table
-- is already a member. Wrap in a DO block that catches duplicate_object.
-- Re-runs of this migration become no-ops.

DO $$
BEGIN
  BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE predictions;
  EXCEPTION WHEN duplicate_object THEN
    -- Already a publication member — nothing to do.
    NULL;
  END;
END $$;

DO $$
BEGIN
  BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE model_paper_trades;
  EXCEPTION WHEN duplicate_object THEN
    NULL;
  END;
END $$;

-- Replica identity FULL on predictions so UPDATE payloads include the
-- previous-value column data, not just the primary key. Lets the
-- front-end show diffs in the future (e.g. "rating_target just
-- changed from pending to hit") without re-fetching the row.
-- Skip for model_paper_trades for now — payload size matters more
-- than rich diffs for the high-volume close path.
ALTER TABLE predictions REPLICA IDENTITY FULL;
