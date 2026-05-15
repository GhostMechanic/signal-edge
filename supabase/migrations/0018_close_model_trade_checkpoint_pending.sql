-- 0018_close_model_trade_checkpoint_pending
--
-- Companion migration to 0017 (the checkpoint scoring scheme migration).
--
-- Background: the close_model_trade RPC from migration 0004 validates
-- that all three rating values are in ('hit', 'miss'). Post-0017,
-- rating_checkpoint is owned by scoring_worker.tick_checkpoints() and
-- the trade-close path passes rating_checkpoint='pending' — meaning
-- "don't touch this field, the interval scorer owns it." Without this
-- migration, every paper-trade close would fail with the existing
-- 'invalid rating values' exception.
--
-- This migration replaces close_model_trade so:
--   • p_rating_checkpoint = 'pending' is a legal input value (in
--     addition to 'hit' and 'miss').
--   • When p_rating_checkpoint = 'pending', the function DOES NOT
--     overwrite the predictions.rating_checkpoint column — the
--     existing value (whatever tick_checkpoints has written) is
--     preserved.
--
-- All other behaviour (cash credit, verdict / target / expiration
-- writes, idempotency, RLS posture) is unchanged from 0004.

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
    -- rating_target and rating_expiration still require terminal values.
    -- rating_checkpoint additionally accepts 'pending' (the post-0017
    -- "don't touch — tick_checkpoints owns it" signal).
    if p_rating_target not in ('hit', 'miss') then
        raise exception 'invalid rating_target: %', p_rating_target;
    end if;
    if p_rating_checkpoint not in ('hit', 'miss', 'pending') then
        raise exception 'invalid rating_checkpoint: %', p_rating_checkpoint;
    end if;
    if p_rating_expiration not in ('hit', 'miss', 'pending') then
        -- 'pending' for rating_expiration was already implicitly allowed
        -- by 0013 (reset_premature_rating_expiration) — this migration
        -- makes the validation match.
        raise exception 'invalid rating_expiration: %', p_rating_expiration;
    end if;

    -- Pull and lock the trade row. Bail if not open (idempotent —
    -- running close on an already-closed trade is a no-op-with-error).
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
    update public.model_paper_portfolio
       set cash       = cash + v_trade.notional + p_realised_pnl,
           updated_at = now()
     where id = 1
    returning cash into v_new_cash;

    -- 3. Recompute verdict + ratings on the linked prediction.
    --    rating_checkpoint is only written when explicitly hit/miss —
    --    'pending' is the "don't touch this column" signal post-0017.
    if p_rating_checkpoint = 'pending' then
        -- Skip rating_checkpoint write — tick_checkpoints owns it.
        update public.predictions
           set verdict           = p_verdict,
               rating_target     = p_rating_target,
               rating_expiration = p_rating_expiration,
               actual_price      = p_exit_price,
               actual_return     = case
                   when entry_price is not null and entry_price > 0
                     then ((p_exit_price / entry_price) - 1) * 100
                   else null
               end,
               scored_at         = now()
         where id = v_trade.prediction_id;
    else
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
    end if;

    return query select v_new_cash, v_trade.prediction_id;
end;
$$;


comment on function public.close_model_trade is
    'Atomic close: locks portfolio + trade rows, updates trade status, '
    'credits cash, writes verdict/rating_target/rating_expiration on the '
    'linked prediction. rating_checkpoint is only written when the input '
    'is hit/miss; pending is the "tick_checkpoints owns this" signal '
    'introduced in migration 0017 and the field is left untouched in '
    'that case.';


-- Permissions: same posture as 0004 — service_role only, no anon/auth
-- exposure. Re-declaring after CREATE OR REPLACE to be explicit (the
-- function signature is identical so the existing grants carry over,
-- but it's clearer to make the intent visible in this migration too).
revoke all on function public.close_model_trade(
    uuid, numeric, text, numeric, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.close_model_trade(
    uuid, numeric, text, numeric, text, text, text, text
) to service_role;
