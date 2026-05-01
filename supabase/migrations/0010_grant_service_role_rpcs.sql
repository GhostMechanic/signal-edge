-- 0010_grant_service_role_rpcs.sql
-- ---------------------------------------------------------------------------
-- Re-issue EXECUTE grants on every model-side RPC the scoring worker
-- (and api/main.py) call against PostgREST under the service_role JWT.
--
-- Background: the GitHub Actions cron's first run on 2026-05-01 surfaced
-- Postgres error 42501 ("permission denied for function close_model_trade")
-- on every attempt to close a matured trade. Migrations 0003/0004/0008 each
-- grant EXECUTE to service_role inline — but the production Supabase
-- project ended up without those grants in place (most likely cause: the
-- functions were re-created via the SQL editor at some point during the
-- new key-system rotation, and `create or replace function` reset
-- privileges to the Postgres default of EXECUTE-to-PUBLIC, which the
-- accompanying `revoke all from public, anon, authenticated` then stripped
-- — leaving no role with EXECUTE).
--
-- This migration is idempotent: re-issuing GRANT EXECUTE on a function
-- where the role already has it is a no-op. Safe to run multiple times.
--
-- Methodology § 4.3: the scoring worker is the only writer to
-- model_paper_trades and model_paper_portfolio. Without these grants,
-- every settled prediction stays at verdict=OPEN and the public Track
-- Record is frozen.
-- ---------------------------------------------------------------------------

-- 0003 — open_model_trade: called by api/main.py when a prediction passes
-- the TRADE rule (methodology § 2.1).
revoke all on function public.open_model_trade(
    uuid, text, text, numeric, numeric, numeric, numeric, numeric
) from public, anon, authenticated;
grant execute on function public.open_model_trade(
    uuid, text, text, numeric, numeric, numeric, numeric, numeric
) to service_role;

-- 0004 — close_model_trade: called by scoring_worker.tick() when a trade's
-- target/stop/expiry triggers (methodology § 4.2).
revoke all on function public.close_model_trade(
    uuid, numeric, text, numeric, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.close_model_trade(
    uuid, numeric, text, numeric, text, text, text, text
) to service_role;

-- 0004 — append_equity_curve_point: called by scoring_worker.eod_pass()
-- once per US trading day at 4:15 PM ET (methodology § 7).
revoke all on function public.append_equity_curve_point(date, numeric)
    from public, anon, authenticated;
grant execute on function public.append_equity_curve_point(date, numeric)
    to service_role;

-- 0008 — append_user_equity_curve_point: called by scoring_worker.user_eod_pass()
-- once per user per US trading day at 4:15 PM ET. Same idempotency story
-- as the model-side equity curve.
revoke all on function public.append_user_equity_curve_point(uuid, date, numeric)
    from public, anon, authenticated;
grant execute on function public.append_user_equity_curve_point(uuid, date, numeric)
    to service_role;

-- End of 0010_grant_service_role_rpcs.sql
