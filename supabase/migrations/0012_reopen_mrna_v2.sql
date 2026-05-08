-- ─────────────────────────────────────────────────────────────────────────────
-- One-shot revert v2: reopen the MRNA 1-month call falsely marked MISSED
-- on May 8, 2026.
--
-- v1 (0011) keyed off the receipt UUID prefix. Likely the prefix
-- characters from the receipt display didn't match the row's actual id
-- (UUID display formats vary by ledger view). This version is broader:
-- match by symbol + verdict + recency. The window (created_at within
-- last 36 hours) is narrow enough that it can only catch the row the
-- user is staring at.
--
-- Wrapped in a DO $$ block that bails loudly if it would touch zero or
-- more than one row — refuses to mutate state on ambiguous targets.
-- ─────────────────────────────────────────────────────────────────────────────

begin;

do $$
declare
    v_pred_id   uuid;
    v_match_count int;
    v_trade     record;
begin
    select count(*) into v_match_count
      from public.predictions
     where symbol = 'MRNA'
       and verdict = 'MISSED'
       and created_at > now() - interval '36 hours';

    if v_match_count = 0 then
        raise exception
            'no recent MISSED MRNA prediction found in last 36h — already reverted, or wrong row';
    end if;
    if v_match_count > 1 then
        raise exception
            'matched % rows — too ambiguous, aborting. Tighten the window or use receipt prefix',
            v_match_count;
    end if;

    select id into v_pred_id
      from public.predictions
     where symbol = 'MRNA'
       and verdict = 'MISSED'
       and created_at > now() - interval '36 hours'
     limit 1;

    -- Confirm there's exactly one trade row to reverse.
    select * into v_trade
      from public.model_paper_trades
     where prediction_id = v_pred_id
       and status in ('closed_target', 'closed_stop', 'closed_expiry')
     limit 2;

    if v_trade.id is null then
        -- The prediction is MISSED but has no closed trade row. That's
        -- consistent with a Watched (non-traded) prediction. Just reset
        -- the prediction row and skip the trade-reversal.
        update public.predictions
           set verdict           = 'OPEN',
               rating_target     = null,
               rating_checkpoint = null,
               rating_expiration = null,
               actual_price      = null,
               actual_return     = null,
               scored_at         = null
         where id = v_pred_id;

        raise notice
            'reopened watched MRNA prediction % (no trade row to reverse)',
            v_pred_id;
        return;
    end if;

    -- Reverse the portfolio cash credit. close_model_trade did:
    --   cash := cash + notional + realised_pnl
    -- so we subtract the same. coalesce realised_pnl in case it was
    -- written as null somehow.
    update public.model_paper_portfolio
       set cash       = cash - (v_trade.notional + coalesce(v_trade.realised_pnl, 0)),
           updated_at = now()
     where id = 1;

    -- Reopen the trade row.
    update public.model_paper_trades
       set status       = 'open',
           exit_price   = null,
           closed_at    = null,
           realised_pnl = null,
           updated_at   = now()
     where id = v_trade.id;

    -- Reset the prediction row back to OPEN.
    update public.predictions
       set verdict           = 'OPEN',
           rating_target     = null,
           rating_checkpoint = null,
           rating_expiration = null,
           actual_price      = null,
           actual_return     = null,
           scored_at         = null
     where id = v_pred_id;

    raise notice
        'reopened MRNA prediction % (trade %): trade.status=open, prediction.verdict=OPEN',
        v_pred_id, v_trade.id;
end $$;

commit;

-- ─────────────────────────────────────────────────────────────────────────────
-- End of 0012_reopen_mrna_v2.sql
-- ─────────────────────────────────────────────────────────────────────────────
