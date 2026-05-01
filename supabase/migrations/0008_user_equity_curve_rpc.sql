-- ─────────────────────────────────────────────────────────────────────────────
-- 0008 — append_user_equity_curve_point
--
-- Per-user mirror of append_equity_curve_point (0004). The 4:15 PM ET EOD
-- pass walks every paper_portfolios row, marks open positions to market,
-- and writes one {date, equity} point per user per trading day so the
-- YourBook back-face has a curve to render against.
--
-- Why upsert-by-date instead of plain append (the model-side function does
-- the latter): with potentially many users, the EOD pass may need a partial
-- retry if it errors mid-walk. Plain append would write duplicate points
-- for whichever users got hit twice. Replace-by-date keeps the curve clean.
-- ─────────────────────────────────────────────────────────────────────────────


create or replace function public.append_user_equity_curve_point(
    p_user_id uuid,
    p_date    date,
    p_equity  numeric
)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
    update public.paper_portfolios
       set equity_curve = coalesce((
               select jsonb_agg(elem order by (elem->>'date'))
                 from jsonb_array_elements(coalesce(equity_curve, '[]'::jsonb)) elem
                where elem->>'date' <> p_date::text
           ), '[]'::jsonb)
           || jsonb_build_array(
                  jsonb_build_object('date', p_date, 'equity', p_equity)
              ),
           updated_at = now()
     where user_id = p_user_id;
end;
$$;

comment on function public.append_user_equity_curve_point is
    'EOD pass: append (or replace-by-date) one {date, equity} point on the '
    'user''s paper_portfolios.equity_curve. Idempotent on same-day re-runs. '
    'Service role only — drives the YourBook back-face curve.';

revoke all on function public.append_user_equity_curve_point(uuid, date, numeric)
    from public, anon, authenticated;
grant execute on function public.append_user_equity_curve_point(uuid, date, numeric)
    to service_role;


-- ─────────────────────────────────────────────────────────────────────────────
-- End of 0008_user_equity_curve_rpc.sql
-- ─────────────────────────────────────────────────────────────────────────────
