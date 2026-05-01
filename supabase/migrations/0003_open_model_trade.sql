-- ─────────────────────────────────────────────────────────────────────────────
-- Prediqt — Phase 1 atomicity helper (migration 0003)
--
-- Adds the `open_model_trade()` RPC: a single Postgres function that opens a
-- model paper trade and decrements the model's cash in one transaction.
--
-- Why an RPC: the open_trade workflow has three steps that MUST be atomic:
--   1. Read current cash + check constraints.
--   2. Insert into model_paper_trades.
--   3. Decrement model_paper_portfolio.cash.
--
-- Doing these as three separate Supabase REST calls from Python opens a race
-- where two concurrent predictions both pass the cash check, both insert, and
-- the model's portfolio over-spends. SELECT FOR UPDATE on the singleton
-- portfolio row serializes the path; the partial unique index on
-- model_paper_trades(symbol) WHERE status='open' provides a backstop.
--
-- The Python repo calls this via .rpc('open_model_trade', {...}). The
-- decision module already pre-checks each rule for a clean PASS reason; this
-- function re-checks as defense-in-depth and raises specific errors so the
-- repo can report which guardrail tripped if the race ever fires.
-- ─────────────────────────────────────────────────────────────────────────────


create or replace function public.open_model_trade(
    p_prediction_id uuid,
    p_symbol        text,
    p_direction     text,
    p_entry_price   numeric,
    p_target_price  numeric,
    p_stop_price    numeric,
    p_qty           numeric,
    p_notional      numeric
)
returns table (
    trade_id  uuid,
    opened_at timestamptz,
    new_cash  numeric
)
language plpgsql
security definer
set search_path = public
as $$
declare
    v_trade_id     uuid;
    v_opened_at    timestamptz;
    v_new_cash     numeric;
    v_current_cash numeric;
    v_max_open     integer;
    v_open_count   integer;
    v_sym          text := upper(trim(p_symbol));
begin
    -- Validate inputs early (cheap checks, before any locking).
    if p_direction not in ('LONG', 'SHORT') then
        raise exception 'invalid direction: %', p_direction;
    end if;
    if p_entry_price <= 0 or p_target_price <= 0 or p_stop_price <= 0 then
        raise exception 'prices must be positive';
    end if;
    if p_qty <= 0 or p_notional <= 0 then
        raise exception 'qty and notional must be positive';
    end if;

    -- Lock the singleton portfolio row. SELECT FOR UPDATE serializes
    -- concurrent open_model_trade calls so the cash check below is reliable.
    select cash, max_open_positions
      into v_current_cash, v_max_open
      from public.model_paper_portfolio
     where id = 1
     for update;

    if not found then
        raise exception 'model_paper_portfolio singleton row missing — run 0002';
    end if;

    -- Defense-in-depth checks. The decision module pre-screens these and the
    -- partial unique index will block double-opens at write time, but raising
    -- a specific error here gives the repo a useful failure message.
    if v_current_cash < p_notional then
        raise exception 'insufficient_cash: % available, % needed',
                        v_current_cash, p_notional;
    end if;

    select count(*)
      into v_open_count
      from public.model_paper_trades
     where status = 'open';

    if v_open_count >= v_max_open then
        raise exception 'book_full: %/% positions open', v_open_count, v_max_open;
    end if;

    if exists (
        select 1
          from public.model_paper_trades
         where status = 'open'
           and upper(trim(symbol)) = v_sym
    ) then
        raise exception 'symbol_already_open: % has an open trade', v_sym;
    end if;

    -- Insert the trade. The partial unique index provides the final
    -- race-condition backstop; if a concurrent transaction opened the same
    -- symbol between our check and our insert, the unique violation surfaces
    -- here.
    insert into public.model_paper_trades (
        prediction_id, symbol, direction,
        entry_price, target_price, stop_price,
        qty, notional
    )
    values (
        p_prediction_id, v_sym, p_direction,
        p_entry_price, p_target_price, p_stop_price,
        p_qty, p_notional
    )
    returning id, opened_at
        into v_trade_id, v_opened_at;

    -- Decrement cash.
    update public.model_paper_portfolio
       set cash       = cash - p_notional,
           updated_at = now()
     where id = 1
    returning cash into v_new_cash;

    -- Return the result row.
    return query select v_trade_id, v_opened_at, v_new_cash;
end;
$$;


comment on function public.open_model_trade is
    'Atomic: validate constraints, insert model_paper_trades row, decrement '
    'model_paper_portfolio.cash. Called by Python SupabaseModelPortfolioRepo '
    'via PostgREST RPC. SELECT FOR UPDATE on the portfolio row serializes '
    'concurrent opens.';


-- Grant execute to the service_role only — this function mutates the
-- universal model portfolio and should never be callable by user JWTs.
revoke all on function public.open_model_trade(
    uuid, text, text, numeric, numeric, numeric, numeric, numeric
) from public, anon, authenticated;
grant execute on function public.open_model_trade(
    uuid, text, text, numeric, numeric, numeric, numeric, numeric
) to service_role;


-- ─────────────────────────────────────────────────────────────────────────────
-- End of 0003_open_model_trade.sql
-- ─────────────────────────────────────────────────────────────────────────────
