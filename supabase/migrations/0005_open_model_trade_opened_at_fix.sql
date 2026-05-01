-- ─────────────────────────────────────────────────────────────────────────────
-- Prediqt — fix `opened_at` ambiguity in open_model_trade() (migration 0005)
--
-- 0003 declared the function's RETURN TABLE with a column named `opened_at`.
-- Inside PL/pgSQL that declaration becomes an implicit local variable. The
-- function body's `INSERT ... RETURNING id, opened_at INTO v_trade_id, v_opened_at`
-- then has two candidates for the bare `opened_at` identifier:
--   • the INSERT's table column model_paper_trades.opened_at, or
--   • the OUT variable from the RETURNS TABLE clause.
--
-- Postgres' default plpgsql.variable_conflict = error refuses to guess and
-- raises 42702 "column reference 'opened_at' is ambiguous. It could refer to
-- either a PL/pgSQL variable or a table column."
--
-- Symptom in production (2026-04-27 logs):
--   open_model_trade RPC failed for NVDA (prediction_id=...): {'message':
--   'column reference "opened_at" is ambiguous', 'code': '42702', ...}
--   → predictions persist with traded=true but no model_paper_trades row,
--     so the singleton model portfolio drifts out of sync.
--
-- Fix: qualify the INSERT's RETURNING clause with the table name so Postgres
-- knows we mean the column, not the OUT variable. Function signature is
-- unchanged so the existing GRANT/REVOKE in 0003 still applies and Python
-- callers don't need to change.
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
    --
    -- NOTE the qualified `model_paper_trades.opened_at` in RETURNING — the
    -- bare `opened_at` would collide with the OUT variable of the same name
    -- declared in this function's RETURNS TABLE clause (migration 0005 fix).
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
    returning model_paper_trades.id, model_paper_trades.opened_at
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
    'concurrent opens. Migration 0005 fixed the opened_at column/variable '
    'ambiguity inside the INSERT...RETURNING clause.';


-- Signature is unchanged from 0003, so the GRANT/REVOKE there still applies.
-- No need to re-grant.


-- ─────────────────────────────────────────────────────────────────────────────
-- End of 0005_open_model_trade_opened_at_fix.sql
-- ─────────────────────────────────────────────────────────────────────────────
