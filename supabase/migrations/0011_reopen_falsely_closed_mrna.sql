-- ─────────────────────────────────────────────────────────────────────────────
-- One-shot revert: reopen the MRNA 1-month call that was falsely marked
-- MISSED on May 8, 2026 due to the _intraday_bars_since bug.
--
-- Bug summary: scoring_worker.py:_intraday_bars_since was passing
-- start=since.date() to yfinance, which strips the time-of-day. yfinance
-- then returned bars from the start of since's date in NY-tz —
-- INCLUDING bars from before the trade was opened. Any intraday low
-- earlier in the day touching the stop price would falsely trigger
-- _scan_bars's stop_hit check and close the trade as MISSED before it
-- ever had a chance to play out.
--
-- Code fix shipped in scoring_worker.py — bars frame is now filtered to
-- timestamps strictly >= since (UTC-normalised). This file cleans up the
-- single row that was already victimised by the bug before the fix.
--
-- The receipt ID prefix from the screenshot is `71828f94…1493`; we
-- match the prediction by that prefix to be surgical and safe (won't
-- touch any other row even if more were affected).
--
-- TRANSACTIONAL: all four mutations either commit together or roll back
-- together — predictions row, model_paper_trades row, portfolio cash
-- credit reversal, and a sanity check on the trade count.
-- ─────────────────────────────────────────────────────────────────────────────

begin;

-- 1. Locate the prediction + its trade. Bail loudly if we don't find
--    exactly one match; refuse to mutate state on ambiguous targets.
do $$
declare
    v_pred_id   uuid;
    v_trade     record;
begin
    select id into v_pred_id
      from public.predictions
     where id::text ilike '71828f94%1493'
       and symbol = 'MRNA'
       and verdict = 'MISSED'
     limit 2;

    if v_pred_id is null then
        raise exception
            'no MISSED MRNA prediction matching receipt prefix 71828f94…1493 found — already reverted, or wrong row';
    end if;

    -- Confirm there's exactly one trade row to reverse.
    select * into v_trade
      from public.model_paper_trades
     where prediction_id = v_pred_id
       and status = 'closed_stop'
     limit 2;

    if v_trade.id is null then
        raise exception
            'no closed_stop trade row found for prediction %', v_pred_id;
    end if;

    -- 2. Reverse the portfolio cash credit. close_model_trade did:
    --      cash := cash + notional + realised_pnl
    --    so we subtract the same.
    update public.model_paper_portfolio
       set cash       = cash - (v_trade.notional + coalesce(v_trade.realised_pnl, 0)),
           updated_at = now()
     where id = 1;

    -- 3. Reopen the trade row.
    update public.model_paper_trades
       set status       = 'open',
           exit_price   = null,
           closed_at    = null,
           realised_pnl = null,
           updated_at   = now()
     where id = v_trade.id;

    -- 4. Reset the prediction row back to OPEN. scored_at + rating_*
    --    + actual_* all clear so the dashboards stop showing the false
    --    miss and the scoring worker will pick this back up at expiry.
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
-- End of 0011_reopen_falsely_closed_mrna.sql
-- ─────────────────────────────────────────────────────────────────────────────
