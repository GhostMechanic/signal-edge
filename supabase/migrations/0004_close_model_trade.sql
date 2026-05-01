-- ─────────────────────────────────────────────────────────────────────────────
-- Prediqt — Phase 1 atomicity helper (migration 0004)
--
-- Adds the `close_model_trade()` RPC: closes an open model paper trade and
-- recomputes the linked prediction's verdict + rating_* in one transaction.
--
-- The matching open path is `open_model_trade` from 0003. This function is
-- the close counterpart called by the scoring worker (see scoring_worker.py)
-- whenever target / stop / expiry triggers.
--
-- Inputs are pre-computed by the Python worker:
--   • exit_price       — the price the trade closed at
--   • close_status     — 'closed_target' | 'closed_stop' | 'closed_expiry'
--   • realised_pnl     — already computed: (exit - entry) * qty for LONG, etc.
--   • verdict          — 'HIT' | 'PARTIAL' | 'MISSED'
--   • rating_*         — 'hit' | 'miss'
--
-- The RPC trusts what it's given. Verdict-from-price logic stays in the
-- worker (Python) where intraday OHLC scanning happens; SQL just persists.
-- ─────────────────────────────────────────────────────────────────────────────


create or replace function public.close_model_trade(
    p_trade_id           uuid,
    p_exit_price         numeric,
    p_close_status       text,
    p_realised_pnl       numeric,
    p_verdict            text,
    p_rating_target      text,
    p_rating_checkpoint  text,
    p_rating_expiration  text
)
returns table (
    new_cash       numeric,
    prediction_id  uuid
)
language plpgsql
security definer
set search_path = public
as $$
declare
    v_trade        record;
    v_new_cash     numeric;
begin
    -- Validate inputs.
    if p_close_status not in ('closed_target', 'closed_stop', 'closed_expiry') then
        raise exception 'invalid close_status: %', p_close_status;
    end if;
    if p_verdict not in ('HIT', 'PARTIAL', 'MISSED') then
        raise exception 'invalid verdict: %', p_verdict;
    end if;
    if p_rating_target not in ('hit', 'miss')
       or p_rating_checkpoint not in ('hit', 'miss')
       or p_rating_expiration not in ('hit', 'miss') then
        raise exception 'invalid rating values';
    end if;

    -- Pull and lock the trade row. Bail if not open (idempotent — running
    -- close on an already-closed trade is a no-op-with-error).
    select * into v_trade
      from public.model_paper_trades
     where id = p_trade_id
     for update;

    if not found then
        raise exception 'model_paper_trades row % not found', p_trade_id;
    end if;
    if v_trade.status != 'open' then
        raise exception 'model_paper_trade % is already closed (status=%)',
                        p_trade_id, v_trade.status;
    end if;

    -- Lock the singleton portfolio row to serialize the cash credit against
    -- any concurrent open_model_trade calls.
    perform 1 from public.model_paper_portfolio where id = 1 for update;

    -- 1. Close the trade.
    update public.model_paper_trades
       set status        = p_close_status,
           exit_price    = p_exit_price,
           closed_at     = now(),
           realised_pnl  = p_realised_pnl,
           updated_at    = now()
     where id = p_trade_id;

    -- 2. Credit the portfolio. Cash gets back the notional plus the P&L.
    --    For a LONG that hit target: notional + positive_pnl.
    --    For a LONG that hit stop:   notional + negative_pnl (still > 0).
    --    For a SHORT that hit target: notional + positive_pnl.
    update public.model_paper_portfolio
       set cash       = cash + v_trade.notional + p_realised_pnl,
           updated_at = now()
     where id = 1
    returning cash into v_new_cash;

    -- 3. Recompute verdict + ratings on the linked prediction.
    --    Per methodology § 5: verdict is the public projection;
    --    rating_* is the member-tier per-horizon detail.
    update public.predictions
       set verdict           = p_verdict,
           rating_target     = p_rating_target,
           rating_checkpoint = p_rating_checkpoint,
           rating_expiration = p_rating_expiration,
           actual_price      = p_exit_price,
           actual_return     = case
               when entry_price is not null and entry_price > 0
                 then ((p_exit_price / entry_price) - 1) * 100
               else null
           end,
           scored_at         = now()
     where id = v_trade.prediction_id;

    return query select v_new_cash, v_trade.prediction_id;
end;
$$;


comment on function public.close_model_trade is
    'Atomic close: locks portfolio + trade rows, updates trade status, '
    'credits cash, writes verdict/rating_*/actual_price/actual_return on '
    'the linked prediction. Called by Python scoring_worker.py via PostgREST '
    'RPC. Verdict logic stays in the worker; SQL just persists.';


-- Service-role only — like open_model_trade, this mutates the universal
-- portfolio and must never be callable by user JWTs.
revoke all on function public.close_model_trade(
    uuid, numeric, text, numeric, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.close_model_trade(
    uuid, numeric, text, numeric, text, text, text, text
) to service_role;


-- ─────────────────────────────────────────────────────────────────────────────
-- Helper for the EOD mark-to-market pass.
--
-- The scoring worker's 4:15 PM EOD run appends one point to the portfolio's
-- equity_curve: { date, equity = cash + Σ(qty × close_price) }. The
-- mark-to-market values come from the worker (it has the price feed); this
-- function just appends them atomically.
-- ─────────────────────────────────────────────────────────────────────────────

create or replace function public.append_equity_curve_point(
    p_date    date,
    p_equity  numeric
)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
    update public.model_paper_portfolio
       set equity_curve = coalesce(equity_curve, '[]'::jsonb)
                          || jsonb_build_array(
                              jsonb_build_object('date', p_date, 'equity', p_equity)
                             ),
           updated_at = now()
     where id = 1;
end;
$$;

comment on function public.append_equity_curve_point is
    'EOD pass: append one {date, equity} point to model_paper_portfolio.equity_curve. '
    'Called once per US trading day at 4:15 PM ET by the scoring worker.';

revoke all on function public.append_equity_curve_point(date, numeric)
    from public, anon, authenticated;
grant execute on function public.append_equity_curve_point(date, numeric)
    to service_role;


-- ─────────────────────────────────────────────────────────────────────────────
-- End of 0004_close_model_trade.sql
-- ─────────────────────────────────────────────────────────────────────────────
