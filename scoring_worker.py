"""
scoring_worker.py
-----------------
The model's portfolio scoring worker. Two entry points, both idempotent:

  • tick()           — 5-min hit/stop/expiry check on the model's open trades.
                       Closes any whose price has touched its target, stop,
                       or whose horizon has elapsed.
  • eod_pass()       — 4:15 PM ET pass. Marks the model's open positions to
                       market at EOD close prices and appends one point to
                       model_paper_portfolio.equity_curve.
  • user_eod_pass()  — 4:15 PM ET pass for every user's paper_portfolios.
                       MTMs equity positions by direction, options by live
                       mid-price (or cash_locked fallback), and writes a
                       point to each user's equity_curve. Idempotent on
                       same-day re-runs (the RPC replaces by date).
  • refresh_model_adjustments() — runs ModelImprover.analyze_prediction_outcomes()
                       to refresh feature_boosts / regime_adjustments /
                       confidence_bias / horizon_weights in
                       .predictions/model_adjustments.json. Cheap; no
                       model training, just analytics over the
                       prediction log. The model reads this at the
                       start of every train(), so refreshing daily
                       keeps the next retrain anchored to the latest
                       scored evidence.
  • retrain_pending_stocks() — walks every stock above the per-stock
                       retrain threshold and force-rebuilds its
                       predictor. Manual / weekly cron. Methodology
                       § 2.3 documents the loop.

Schedule (per methodology § 4.3):

    Every 5 min during US market hours (9:30 AM – 4:00 PM ET) → tick()
    Once at 4:15 PM ET                                          → tick() + eod_pass() + user_eod_pass() + refresh_model_adjustments()
    After-hours / overnight / weekends                          → no runs

Deployment is intentionally not opinionated: a host crontab, GitHub
Actions, or any cloud cron can call the CLI entry point at the bottom of
this file. The function is pure-side-effects-on-Supabase, so it doesn't
care where it runs as long as USE_SUPABASE=true and the service-role key
is configured.

Verdict logic (anchor § 5):

    HIT      ← price touched target_price (predicted direction) before expiry
    PARTIAL  ← checkpoint touched but target not, before expiry
    MISSED   ← stop hit, OR expiry passed without target/checkpoint
    OPEN     ← otherwise (still in window, no trigger)

The checkpoint is the midpoint between entry and target in log-return
space (see anchor § 5.1):

    checkpoint_return = predicted_return / 2
    checkpoint_price  = entry_price * exp(checkpoint_return)

For LONG: hit_target = high >= target; hit_checkpoint = high >= checkpoint;
            stop = low <= stop_price.
For SHORT: hit_target = low <= target; hit_checkpoint = low <= checkpoint;
            stop = high >= stop_price.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────

def tick(now: Optional[datetime] = None) -> dict:
    """
    Run the 5-min check. Pulls every open model paper trade, fetches the
    latest price + intraday OHLC, and closes any trade whose price has
    touched target/stop or whose horizon has elapsed.

    Returns a summary dict for logs:
        { "checked": N, "closed_target": x, "closed_stop": y, "closed_expiry": z,
          "errors": [{"trade_id": ..., "error": ...}, ...] }
    """
    from db import _service_client

    now = now or datetime.now(timezone.utc)
    client = _service_client()

    open_trades = (
        client.table("model_paper_trades")
              .select("*")
              .eq("status", "open")
              .execute()
    ).data or []

    summary = {
        "checked":       len(open_trades),
        "closed_target": 0,
        "closed_stop":   0,
        "closed_expiry": 0,
        "errors":        [],
    }

    for trade in open_trades:
        try:
            outcome = _evaluate_trade(trade, now)
            if outcome is None:
                continue   # still open
            _close_trade(client, trade, outcome)
            summary[f"closed_{outcome.close_status.split('_')[1]}"] += 1
        except Exception as exc:
            logger.exception("scoring tick failed for trade %s: %s",
                             trade.get("id"), exc)
            summary["errors"].append({
                "trade_id": trade.get("id"),
                "error":    str(exc)[:200],
            })

    return summary


def eod_pass(today: Optional[date] = None) -> dict:
    """
    EOD mark-to-market. Captures the day's portfolio value (cash + open
    positions valued at EOD close) and appends to equity_curve.

    Run once at 4:15 PM ET on US trading days, after the day's last
    tick().

    Returns: { "date": ..., "equity": ..., "open_count": ... }
    """
    from db import _service_client

    client = _service_client()
    today  = today or datetime.now(timezone.utc).date()

    # Pull cash + open trades.
    portfolio = (
        client.table("model_paper_portfolio")
              .select("cash")
              .eq("id", 1)
              .single()
              .execute()
    ).data
    if not portfolio:
        raise RuntimeError("model_paper_portfolio singleton row missing")

    cash = float(portfolio["cash"])

    open_trades = (
        client.table("model_paper_trades")
              .select("symbol, qty, entry_price")
              .eq("status", "open")
              .execute()
    ).data or []

    # Mark to market. Use yfinance's last close for each open symbol.
    mtm_value = 0.0
    for trade in open_trades:
        try:
            close_price = _last_close(trade["symbol"])
            mtm_value += float(trade["qty"]) * close_price
        except Exception as exc:
            logger.exception("EOD MTM failed for %s; using entry_price: %s",
                             trade["symbol"], exc)
            mtm_value += float(trade["qty"]) * float(trade["entry_price"])

    equity = round(cash + mtm_value, 2)

    # Append the point.
    client.rpc("append_equity_curve_point", {
        "p_date":   today.isoformat(),
        "p_equity": equity,
    }).execute()

    return {
        "date":       today.isoformat(),
        "equity":     equity,
        "open_count": len(open_trades),
        "cash":       cash,
    }


def user_eod_pass(today: Optional[date] = None) -> dict:
    """
    EOD mark-to-market for every user's paper_portfolios. Mirrors
    eod_pass() but per-user: walks `paper_portfolios`, computes each
    user's equity = cash + Σ(MTM contribution per open trade), writes
    one {date, equity} point via append_user_equity_curve_point.

    MTM rules:
      • equity LONG  → contribution = qty × eod_close
      • equity SHORT → contribution = qty × (2 × entry_price − eod_close)
                       (cost basis qty × entry, plus (entry − close) × qty
                        of unrealised P&L)
      • option (live)     → contribution = pricer.cash_back  (= cash_locked + pnl_total)
                            so cash + cash_back = original_cash + pnl, which is
                            what total user equity should reflect.
      • option (no quote) → contribution = cash_locked  (zero-P&L assumption,
                            the most defensible no-info fallback)

    Last-close per symbol is cached across users so we don't re-fetch the
    same yfinance ticker N times in a multi-user pass. Same for the
    spot used by the option pricer's intrinsic fallback.

    Returns a per-pass summary:
        {
            "date":     "...",
            "users":    int,
            "written":  int,
            "errors":   [{"user_id": ..., "error": ...}, ...],
        }
    """
    from db import _service_client

    client = _service_client()
    today = today or datetime.now(timezone.utc).date()
    today_iso = today.isoformat()

    portfolios = (
        client.table("paper_portfolios")
              .select("user_id, cash, starting_capital")
              .execute()
    ).data or []

    summary = {"date": today_iso, "users": len(portfolios), "written": 0, "errors": []}
    if not portfolios:
        return summary

    close_cache: dict[str, Optional[float]] = {}

    def cached_close(symbol: str) -> Optional[float]:
        s = (symbol or "").upper().strip()
        if not s:
            return None
        if s in close_cache:
            return close_cache[s]
        try:
            v = _last_close(s)
        except Exception as exc:  # noqa: BLE001
            logger.warning("user_eod_pass: _last_close failed for %s: %s", s, exc)
            v = None
        close_cache[s] = v
        return v

    # Late import — keep yfinance/pandas pulls out of module load when the
    # worker isn't actually running.
    try:
        from options_pricer import value_option_position
    except Exception as exc:  # noqa: BLE001
        logger.warning("options_pricer import failed; option MTM will fall back: %s", exc)
        value_option_position = None  # type: ignore[assignment]

    for portfolio in portfolios:
        user_id = portfolio.get("user_id")
        if not user_id:
            continue
        try:
            cash = float(portfolio.get("cash") or 0)

            # Defensive SELECT — pre-0007 portfolios won't have kind/instrument_data.
            try:
                rows = (
                    client.table("paper_trades")
                          .select(
                              "id, symbol, direction, entry_price, qty, "
                              "kind, instrument_data"
                          )
                          .eq("user_id", user_id)
                          .eq("status", "open")
                          .execute()
                ).data or []
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "user_eod_pass: full SELECT failed (%s); retrying without kind", exc,
                )
                rows = (
                    client.table("paper_trades")
                          .select("id, symbol, direction, entry_price, qty")
                          .eq("user_id", user_id)
                          .eq("status", "open")
                          .execute()
                ).data or []

            mtm_value = 0.0
            for tr in rows:
                kind = (tr.get("kind") or "equity").lower()
                qty = float(tr.get("qty") or 0)
                entry = float(tr.get("entry_price") or 0)
                symbol = (tr.get("symbol") or "").upper().strip()

                if kind == "option":
                    ins = tr.get("instrument_data") or {}
                    cash_locked = float(ins.get("cash_locked") or 0)
                    if value_option_position is None:
                        mtm_value += cash_locked
                        continue
                    spot = cached_close(symbol)
                    val = value_option_position(tr, spot_override=spot)
                    if val.get("ok"):
                        mtm_value += float(val.get("cash_back") or 0)
                    else:
                        mtm_value += cash_locked
                    continue

                # Equity (LONG / SHORT).
                close_price = cached_close(symbol)
                if close_price is None:
                    # No close — keep at cost basis so we don't poison the curve.
                    mtm_value += qty * entry
                    continue
                direction = (tr.get("direction") or "LONG").upper()
                if direction == "SHORT":
                    mtm_value += qty * (2.0 * entry - close_price)
                else:
                    mtm_value += qty * close_price

            equity = round(cash + mtm_value, 2)
            client.rpc("append_user_equity_curve_point", {
                "p_user_id": user_id,
                "p_date":    today_iso,
                "p_equity":  equity,
            }).execute()
            summary["written"] += 1

        except Exception as exc:  # noqa: BLE001
            logger.exception("user_eod_pass: failed for user %s: %s", user_id, exc)
            summary["errors"].append({
                "user_id": user_id,
                "error":   str(exc)[:200],
            })

    return summary


def refresh_model_adjustments() -> dict:
    """
    Run the global learning analysis: feature importance, calibration
    error, regime accuracy, horizon weights → write to
    .predictions/model_adjustments.json. The model picks these up at
    the start of every train() via _load_learned_adjustments().

    Cheap (no model training, just analytics over the prediction log).
    Wired into the daily `both` command so the model's adjustment knobs
    stay current with the latest scored predictions.

    Returns a small summary the CLI prints.
    """
    try:
        from model_improvement import ModelImprover
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"ModelImprover import failed: {exc}"}

    try:
        improver = ModelImprover()
        findings = improver.analyze_prediction_outcomes()
    except Exception as exc:  # noqa: BLE001
        logger.exception("refresh_model_adjustments failed")
        return {"ok": False, "error": str(exc)[:200]}

    return {
        "ok": True,
        "scored":           int(findings.get("total_scored") or 0),
        "overall_accuracy": findings.get("overall_accuracy"),
        "calibration_err":  findings.get("confidence_calibration_error"),
        "n_horizon_insights": len(findings.get("horizon_insights") or {}),
        "n_recommendations":  len(findings.get("recommendations") or []),
    }


def retrain_pending_stocks(*, dry_run: bool = False) -> dict:
    """
    Walk learning_engine.stocks_pending_retrain() and force-retrain each
    in turn. Used as a manual / weekly background sweep so users don't
    pay the retrain latency on their next predict() call.

    Each retrain triggers model.train() with the current symbol's
    history, applies the latest model_adjustments.json, and resets the
    per-stock retrain marker on success.

    `dry_run=True` reports which symbols would retrain without doing it.

    Returns a per-pass summary:
        {
            "checked":     int,
            "retrained":   [symbols],
            "failed":      [{symbol, error}],
            "skipped":     [symbols not retrained, with reason],
        }
    """
    try:
        from learning_engine import stocks_pending_retrain
        pending = stocks_pending_retrain()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"stocks_pending_retrain failed: {exc}"}

    summary = {
        "checked":   len(pending),
        "retrained": [],
        "failed":    [],
        "skipped":   [],
    }

    if not pending:
        return summary

    if dry_run:
        summary["retrained"] = list(pending)  # would-have-retrained list
        return summary

    # Late imports — keep heavy ML deps out of module load when the CLI
    # is invoked for unrelated commands.
    try:
        from data_fetcher import fetch_stock_data, fetch_market_context
        from model import StockPredictor
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"model import failed: {exc}"}

    for symbol in pending:
        try:
            df = fetch_stock_data(symbol, period="10y")
            if df is None or df.empty or len(df) < 500:
                summary["skipped"].append({
                    "symbol": symbol,
                    "reason": f"insufficient data: {0 if df is None else len(df)} rows",
                })
                continue
            ctx = fetch_market_context()
            predictor = StockPredictor(symbol)
            # train() honors the per-stock retrain marker; passing the
            # symbol via the constructor is enough to trigger force-rebuild.
            predictor.train(df, market_ctx=ctx)
            summary["retrained"].append(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.exception("retrain_pending_stocks: %s failed", symbol)
            summary["failed"].append({"symbol": symbol, "error": str(exc)[:200]})

    return summary


# ─── Trade evaluation ─────────────────────────────────────────────────────────

@dataclass
class CloseOutcome:
    close_status:      str    # 'closed_target' | 'closed_stop' | 'closed_expiry'
    exit_price:        float
    realised_pnl:      float
    verdict:           str    # 'HIT' | 'PARTIAL' | 'MISSED'
    rating_target:     str    # 'hit' | 'miss'
    rating_checkpoint: str    # 'hit' | 'miss'
    rating_expiration: str    # 'hit' | 'miss'


def _evaluate_trade(trade: dict, now: datetime) -> Optional[CloseOutcome]:
    """
    Decide whether to close this trade and at what state. Returns None if
    the trade is still open.

    Order of checks matters. Target before stop before expiry — methodology
    § 4.2: "the model cannot close a position early for any other reason."
    Once a level is touched, the close is immediate.
    """
    direction    = trade["direction"]                # 'LONG' | 'SHORT'
    symbol       = trade["symbol"]
    entry_price  = float(trade["entry_price"])
    target_price = float(trade["target_price"]) if trade.get("target_price") else None
    stop_price   = float(trade["stop_price"])   if trade.get("stop_price")   else None
    qty          = float(trade["qty"])

    # The window we need to scan is opened_at → now. For a per-5-min tick,
    # we pull a small intraday OHLC chunk that covers the recent activity.
    # For simplicity and to be robust to multi-day predictions, just pull
    # the last few days of intraday bars and filter by opened_at.
    opened_at = _parse_iso(trade["opened_at"])

    bars = _intraday_bars_since(symbol, opened_at)
    if bars is None or bars.empty:
        # No bars available — can't safely evaluate. Leave open and try again.
        return None

    last_close = float(bars["Close"].iloc[-1])

    # Compute checkpoint per anchor § 5.1. Direction-aware.
    if target_price and entry_price:
        target_return    = (target_price / entry_price) - 1
        checkpoint_return = target_return / 2
        checkpoint_price = entry_price * (1 + checkpoint_return)
    else:
        checkpoint_price = None

    # Did target hit during the window?
    target_hit_bool, checkpoint_hit_bool, stop_hit_bool = _scan_bars(
        bars, direction, target_price, checkpoint_price, stop_price,
    )

    # NOTE on rating_checkpoint: post-migration 0017 the checkpoint axis
    # is owned exclusively by tick_checkpoints() — it measures directional
    # persistence at scheduled intervals, independent of trade outcome.
    # The trade-close paths below pass rating_checkpoint="pending" so the
    # close doesn't overwrite the checkpoint state the interval scorer
    # is responsible for. The legacy midpoint-touch (`checkpoint_hit_bool`)
    # is no longer used for rating_checkpoint; we still compute it for
    # the PARTIAL verdict assignment at expiry, since PARTIAL is a
    # trade-outcome concept (not the same as rating_checkpoint).

    # 1. TARGET hit → HIT.
    if target_hit_bool:
        # Use target_price as exit (the level we said we'd take profit at).
        exit_price = float(target_price)
        # rating_expiration is INDEPENDENT of trade outcome — it
        # measures whether the predicted direction was correct AT
        # horizon_ends_at. Closing the trade early (because we touched
        # our take-profit) tells us nothing about where the price will
        # be on the horizon date, so this rating must stay 'pending'
        # until the real horizon arrives. score_open_predictions()
        # picks it up at that point.
        return _build_close(
            trade, direction, qty, entry_price, exit_price,
            close_status="closed_target",
            verdict="HIT",
            rating_target="hit", rating_checkpoint="pending",
            rating_expiration="pending",
        )

    # 2. STOP hit → MISSED (stop = the call didn't pay off).
    if stop_hit_bool:
        exit_price = float(stop_price)
        # Same independence argument as the target-hit case above: we
        # closed early, so we don't know yet what the price will be at
        # the horizon date. The trade verdict is MISSED (we took the
        # loss), but rating_expiration is its own measurement and
        # stays pending until horizon_ends_at.
        return _build_close(
            trade, direction, qty, entry_price, exit_price,
            close_status="closed_stop",
            verdict="MISSED",
            rating_target="miss", rating_checkpoint="pending",
            rating_expiration="pending",
        )

    # 3. Expiry?
    horizon_ends_at = _parse_iso(trade.get("opened_at"))   # fallback (won't expire)
    # We need the prediction's horizon_ends_at — pulled separately because
    # it's not on the trade row directly.
    horizon_ends_at = _prediction_horizon_ends(trade["prediction_id"])
    if horizon_ends_at is not None and now >= horizon_ends_at:
        # Window over. Use last close as exit price. PARTIAL stays a
        # midpoint-touch concept at the TRADE layer — it's the verdict
        # axis that distinguishes "stopped out" from "drifted past
        # halfway then didn't make it." rating_checkpoint, the analytics
        # axis, is governed separately by tick_checkpoints().
        exit_price = last_close
        if checkpoint_hit_bool:
            verdict = "PARTIAL"
        else:
            verdict = "MISSED"
        # rating_expiration NOW resolves to a definite value: was the
        # endpoint direction correct? For LONG, last_close > entry =
        # right side; for SHORT, last_close < entry = right side.
        # Pre-fix this hardcoded 'miss' regardless of endpoint, which
        # made strictness analytics double-count any non-target hit
        # as a direction failure even when the price closed correctly.
        if direction == "LONG":
            endpoint_correct = last_close > entry_price
        else:  # SHORT
            endpoint_correct = last_close < entry_price
        return _build_close(
            trade, direction, qty, entry_price, exit_price,
            close_status="closed_expiry",
            verdict=verdict,
            rating_target="miss",
            rating_checkpoint="pending",
            rating_expiration="hit" if endpoint_correct else "miss",
        )

    # Still open.
    return None


def _scan_bars(
    bars: pd.DataFrame,
    direction: str,
    target_price: Optional[float],
    checkpoint_price: Optional[float],
    stop_price: Optional[float],
) -> tuple[bool, bool, bool]:
    """Scan an OHLC frame for whether each level was touched in the window.
    Direction-aware — LONG hits a level by going up to it, SHORT by going
    down to it."""
    if bars is None or bars.empty:
        return (False, False, False)

    highs = bars["High"].astype(float)
    lows  = bars["Low"].astype(float)

    if direction == "LONG":
        target_hit     = bool((target_price     is not None) and (highs >= target_price).any())
        checkpoint_hit = bool((checkpoint_price is not None) and (highs >= checkpoint_price).any())
        stop_hit       = bool((stop_price       is not None) and (lows  <= stop_price).any())
    else:  # SHORT
        target_hit     = bool((target_price     is not None) and (lows  <= target_price).any())
        checkpoint_hit = bool((checkpoint_price is not None) and (lows  <= checkpoint_price).any())
        stop_hit       = bool((stop_price       is not None) and (highs >= stop_price).any())

    return (target_hit, checkpoint_hit, stop_hit)


def _build_close(
    trade: dict,
    direction: str,
    qty: float,
    entry_price: float,
    exit_price: float,
    *,
    close_status: str,
    verdict: str,
    rating_target: str,
    rating_checkpoint: str,
    rating_expiration: str,
) -> CloseOutcome:
    """Compute realised P&L and pack a CloseOutcome.

    LONG:  pnl = (exit - entry) × qty
    SHORT: pnl = (entry - exit) × qty
    """
    if direction == "LONG":
        pnl = (exit_price - entry_price) * qty
    else:
        pnl = (entry_price - exit_price) * qty

    return CloseOutcome(
        close_status      = close_status,
        exit_price        = round(float(exit_price), 4),
        realised_pnl      = round(float(pnl), 2),
        verdict           = verdict,
        rating_target     = rating_target,
        rating_checkpoint = rating_checkpoint,
        rating_expiration = rating_expiration,
    )


def _close_trade(client, trade: dict, outcome: CloseOutcome) -> None:
    """Call the close_model_trade RPC from migration 0004."""
    client.rpc("close_model_trade", {
        "p_trade_id":           trade["id"],
        "p_exit_price":         outcome.exit_price,
        "p_close_status":       outcome.close_status,
        "p_realised_pnl":       outcome.realised_pnl,
        "p_verdict":            outcome.verdict,
        "p_rating_target":      outcome.rating_target,
        "p_rating_checkpoint":  outcome.rating_checkpoint,
        "p_rating_expiration":  outcome.rating_expiration,
    }).execute()


# ─── Price helpers ────────────────────────────────────────────────────────────

def _last_close(symbol: str) -> float:
    """Latest available daily close. Falls back through period sizes if the
    short period yields nothing."""
    for period in ("1d", "5d", "1mo"):
        try:
            hist = yf.Ticker(symbol).history(period=period, interval="1d")
            if hist is not None and not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            continue
    raise RuntimeError(f"_last_close: no data for {symbol}")


def _intraday_bars_since(symbol: str, since: datetime) -> Optional[pd.DataFrame]:
    """Pull intraday 5-min bars from `since` to now. Falls back to daily if
    intraday is unavailable. Returns a DataFrame with High/Low/Close columns.

    BUGFIX: yfinance's `start=` parameter only takes a date — it returns
    bars from the START of that day (9:30 ET for US equities) regardless
    of the time-of-day in `since`. Pre-fix, a trade opened at 12:23 PM
    would scan against bars from 9:30–12:23 too, and any intraday low
    earlier in the day would falsely trip the stop check and close the
    trade as MISSED before it ever had time to play out. Now we filter
    the returned frame to rows whose timestamp is strictly >= `since`."""
    try:
        # Intraday bars only available for ~60 days back.
        days_back = max(1, (datetime.now(timezone.utc) - since).days + 1)
        if days_back <= 7:
            hist = yf.Ticker(symbol).history(
                start=since.date().isoformat(), interval="5m"
            )
        else:
            hist = yf.Ticker(symbol).history(
                start=since.date().isoformat(), interval="1d"
            )
        if hist is None or hist.empty:
            return None

        # Filter to bars strictly after `since`. yfinance's index is a
        # tz-aware DatetimeIndex (exchange-local for US equities).
        # Normalise both sides to UTC for the comparison so the filter
        # is robust regardless of which tz the index carries.
        try:
            idx = hist.index
            if getattr(idx, "tz", None) is None:
                # Naive index — treat as UTC.
                idx_utc = idx.tz_localize("UTC")
            else:
                idx_utc = idx.tz_convert("UTC")
            since_utc = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
            mask = idx_utc >= since_utc
            hist = hist[mask]
        except Exception as filter_exc:
            # If filtering fails for any reason, log and fall through
            # with the unfiltered frame — better to score with extra bars
            # than to skip scoring entirely.
            logger.warning("bars filter failed for %s: %s", symbol, filter_exc)

        if hist is None or hist.empty:
            return None
        return hist
    except Exception as exc:
        logger.exception("intraday fetch failed for %s: %s", symbol, exc)
        return None


def _prediction_horizon_ends(prediction_id: str) -> Optional[datetime]:
    """Look up horizon_ends_at on the linked prediction row."""
    try:
        from db import _service_client
        res = (
            _service_client().table("predictions")
                .select("horizon_ends_at")
                .eq("id", prediction_id)
                .single()
                .execute()
        )
        if res.data and res.data.get("horizon_ends_at"):
            return _parse_iso(res.data["horizon_ends_at"])
    except Exception as exc:
        logger.exception("horizon_ends_at fetch failed for %s: %s", prediction_id, exc)
    return None


def _parse_iso(s: str) -> datetime:
    """Parse a Postgres-style ISO timestamp into a tz-aware datetime.

    Python 3.9's datetime.fromisoformat is strict about fractional seconds —
    it only accepts exactly 3 or 6 digits, and raises 'Invalid isoformat
    string' on Postgres serializations like '.92735+00:00' (5 digits).
    Python 3.11+ handles this natively. We normalize the fractional part
    to exactly 6 digits before parsing so the worker keeps running on
    older Pythons without forcing a runtime upgrade.
    """
    import re
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    s = str(s).replace("Z", "+00:00")
    # Pad/truncate the .digits group to exactly 6 chars. The lookahead
    # stops the match at the timezone suffix (+, -, or end of string).
    s = re.sub(
        r"\.(\d+)(?=[+\-]|$)",
        lambda m: "." + m.group(1).ljust(6, "0")[:6],
        s,
    )
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ─── Watched-prediction scoring ───────────────────────────────────────────────
# tick() handles predictions where the model committed paper money — those have
# a model_paper_trades row whose close path updates predictions.verdict via the
# close_model_trade RPC. Watched predictions (traded=false) don't have a trade
# row, so without this function their verdict stays OPEN forever even after
# horizon_ends_at passes. Methodology § 5 says verdict applies to ALL settled
# predictions, so this fills that gap.
#
# Direction handling (per § 5 + § 4.1):
#   LONG  → HIT if highs reached target; PARTIAL if highs reached checkpoint;
#           else MISSED.
#   SHORT → HIT if lows reached target; PARTIAL if lows reached checkpoint;
#           else MISSED.
#   NEUT  → no trade plan, no target / checkpoint. We score on whether the
#           actual return at horizon end stayed inside the deadband
#           (±NEUTRAL_RETURN_DEADBAND, default ±0.5%): HIT inside, MISSED
#           outside. There's no PARTIAL case for Neutral — the call is either
#           "stayed flat" or it didn't.

NEUTRAL_RETURN_DEADBAND = 0.005   # ±0.5% — matches the asking-flow constant.


def score_open_predictions(now: Optional[datetime] = None) -> dict:
    """
    Update verdict + rating_* fields for every prediction whose horizon has
    elapsed and that isn't being scored elsewhere (no model_paper_trade row).

    Idempotent: only touches rows where verdict='OPEN' and
    horizon_ends_at <= now. Re-running on the same tick is safe.

    Writes via service_role (bypasses RLS). Same trust model as the
    insert path — the API authenticates user, the worker enforces
    methodology, RLS is not the integrity boundary for backend writes.

    Returns a per-run summary:
        { "checked": N, "scored_hit": x, "scored_partial": y,
          "scored_missed": z, "skipped_traded": k, "errors": [...] }
    """
    from db import _service_client

    now = now or datetime.now(timezone.utc)
    client = _service_client()

    open_preds = (
        client.table("predictions")
              .select("*")
              .eq("verdict", "OPEN")
              .lte("horizon_ends_at", now.isoformat())
              .execute()
    ).data or []

    summary = {
        "checked":        len(open_preds),
        "scored_hit":     0,
        "scored_partial": 0,
        "scored_missed":  0,
        "skipped_traded": 0,
        "skipped_nodata": 0,
        "errors":         [],
    }

    for pred in open_preds:
        pred_id = pred["id"]
        try:
            # Skip predictions that have a linked model_paper_trade — those
            # are scored via tick() → close_model_trade. We only fill the
            # gap for Watched (traded=false) predictions.
            trade_check = (
                client.table("model_paper_trades")
                      .select("id", count="exact")
                      .eq("prediction_id", pred_id)
                      .execute()
            )
            if (trade_check.count or 0) > 0:
                summary["skipped_traded"] += 1
                continue

            outcome = _evaluate_open_prediction(pred, now)
            if outcome is None:
                summary["skipped_nodata"] += 1
                continue

            # Post-migration 0017: rating_checkpoint is owned by
            # tick_checkpoints. The `rc` slot from the tuple is
            # intentionally not written — leaving it to whatever
            # tick_checkpoints has already set (or NULL for Neutral
            # predictions, which generate no checkpoint scores).
            verdict, rt, _rc_unused, re_, actual_price, actual_return = outcome

            update_fields: dict = {
                "verdict":           verdict,
                "rating_target":     rt,
                "rating_expiration": re_,
                "scored_at":         now.isoformat(),
            }
            if actual_price is not None:
                update_fields["actual_price"] = round(float(actual_price), 4)
            if actual_return is not None:
                update_fields["actual_return"] = round(float(actual_return) * 100, 4)

            client.table("predictions").update(update_fields).eq(
                "id", pred_id
            ).execute()

            key = f"scored_{verdict.lower()}"
            if key in summary:
                summary[key] += 1
        except Exception as exc:
            logger.exception("score_open_predictions failed for %s: %s",
                             pred_id, exc)
            summary["errors"].append({
                "prediction_id": pred_id,
                "error":         str(exc)[:200],
            })

    return summary


def _evaluate_open_prediction(
    pred: dict,
    now: datetime,
) -> Optional[tuple]:
    """
    Compute the verdict for a single Watched prediction whose horizon has
    elapsed. Returns (verdict, rating_target, rating_checkpoint,
    rating_expiration, actual_price, actual_return) or None when bars aren't
    available.

    Mirrors the trade-side _evaluate_trade for LONG/SHORT (target +
    checkpoint scan over the prediction window). Adds the Neutral path
    described above.
    """
    direction_raw = (pred.get("direction") or "").strip()
    # Normalise: predictions table stores "Bullish" / "Bearish" / "Neutral".
    if direction_raw == "Bullish":
        direction = "LONG"
    elif direction_raw == "Bearish":
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    symbol = pred["symbol"]
    starts_at = _parse_iso(pred.get("horizon_starts_at") or pred.get("created_at"))

    bars = _intraday_bars_since(symbol, starts_at)
    if bars is None or bars.empty:
        return None

    last_close = float(bars["Close"].iloc[-1])
    entry_price = float(pred.get("entry_price") or 0)

    # Compute actual_return for the row — same shape as the close_model_trade
    # writes for traded calls. Useful for analytics and the receipt copy.
    if entry_price > 0:
        try:
            import math
            actual_return = math.log(last_close / entry_price)
        except Exception:
            actual_return = None
    else:
        actual_return = None

    # ── Neutral case ──────────────────────────────────────────────────────
    if direction == "NEUTRAL":
        # No target / checkpoint to scan against. Score on whether the
        # actual log-return at horizon end stayed inside the deadband.
        if actual_return is None:
            # No entry to compute against — can't honestly score.
            return None
        if abs(actual_return) <= NEUTRAL_RETURN_DEADBAND:
            return ("HIT", "hit", "hit", "hit", last_close, actual_return)
        return ("MISSED", "miss", "miss", "miss", last_close, actual_return)

    # ── LONG / SHORT ──────────────────────────────────────────────────────
    target_price = float(pred["target_price"]) if pred.get("target_price") else None
    if target_price and entry_price > 0:
        target_return    = (target_price / entry_price) - 1
        checkpoint_return = target_return / 2
        checkpoint_price = entry_price * (1 + checkpoint_return)
    else:
        checkpoint_price = None

    target_hit_bool, checkpoint_hit_bool, _stop_hit_bool = _scan_bars(
        bars, direction, target_price, checkpoint_price, stop_price=None,
    )

    # Methodology § 5: the three rating fields are independent reads.
    #   • rating_target     — was target_price touched (in the predicted
    #                          direction) at any point during the window?
    #   • rating_checkpoint — was the §5.1 midpoint touched at any point?
    #   • rating_expiration — was direction correct AT horizon_ends_at,
    #                          i.e. is the closing price on the right side
    #                          of entry_price?
    #
    # Pre-fix this code hardcoded rating_expiration='hit' whenever target
    # was hit — locking target_hit_rate and expiration_win_rate together
    # mathematically. A stock that touched target mid-window then
    # reversed to close below entry should register as
    # rating_expiration='miss' under the strict reading; previously it
    # registered 'hit'. Fixing makes Generous and Strict actually
    # measure different things.
    if actual_return is None:
        # Without an actual_return we can't honestly score the strict
        # reading. Fall through to the conservative interpretation:
        # endpoint_correct = whether last_close is on the right side
        # of entry_price even if return is unknown.
        if direction == "LONG":
            endpoint_correct = last_close > entry_price
        else:  # SHORT
            endpoint_correct = last_close < entry_price
    else:
        if direction == "LONG":
            endpoint_correct = actual_return > 0
        else:  # SHORT
            endpoint_correct = actual_return < 0
    rating_expiration = "hit" if endpoint_correct else "miss"

    if target_hit_bool:
        return ("HIT", "hit", "hit", rating_expiration, last_close, actual_return)
    if checkpoint_hit_bool:
        return ("PARTIAL", "miss", "hit", rating_expiration, last_close, actual_return)
    return ("MISSED", "miss", "miss", rating_expiration, last_close, actual_return)


def score_expiration_for_traded(now: Optional[datetime] = None) -> dict:
    """
    Resolve rating_expiration='pending' for TRADED predictions whose
    horizon has elapsed.

    Why this exists
    ───────────────
    When a paper trade closes early at target or stop, `_evaluate_trade`
    now leaves rating_expiration='pending' (because closing early says
    nothing about where the price will be on the horizon date). This
    function picks those rows up once the horizon actually arrives and
    decides rating_expiration based on whether the predicted direction
    matches the price at horizon_ends_at.

    score_open_predictions() handles untraded ("Watched") predictions.
    This is its mirror for traded predictions — it only touches
    rating_expiration, leaving verdict + rating_target +
    rating_checkpoint alone (the trade-close path already wrote those).

    Idempotent: only updates rows still pending. Re-runs are safe.
    """
    from db import _service_client

    now = now or datetime.now(timezone.utc)
    client = _service_client()

    # Pull traded predictions whose horizon has elapsed AND whose
    # expiration rating is still pending. Verdict will already be
    # HIT/MISSED/PARTIAL from the trade-close path.
    rows = (
        client.table("predictions")
              .select("*")
              .lte("horizon_ends_at", now.isoformat())
              .eq("rating_expiration", "pending")
              .execute()
    ).data or []

    summary = {
        "checked":           len(rows),
        "marked_hit":        0,
        "marked_miss":       0,
        "skipped_untraded":  0,
        "skipped_nodata":    0,
        "errors":            [],
    }

    for pred in rows:
        pred_id = pred["id"]
        try:
            # Must be a traded prediction. Untraded ones are handled by
            # score_open_predictions; running both functions is safe
            # because each only writes to its own rows.
            trade_check = (
                client.table("model_paper_trades")
                      .select("id", count="exact")
                      .eq("prediction_id", pred_id)
                      .execute()
            )
            if (trade_check.count or 0) == 0:
                summary["skipped_untraded"] += 1
                continue

            direction_raw = (pred.get("direction") or "").strip()
            symbol = pred["symbol"]
            entry_price = float(pred.get("entry_price") or 0)
            if entry_price <= 0:
                continue

            # Fetch a price slice that ends at/after horizon_ends_at so we
            # can read the actual close on the horizon date.
            horizon_end = _parse_iso(pred["horizon_ends_at"])
            bars = _intraday_bars_since(symbol, horizon_end - timedelta(days=3))
            if bars is None or bars.empty:
                summary["skipped_nodata"] += 1
                continue

            # Last available close at or after horizon_end. Fall back to
            # the very latest close if we can't find a bar right on the
            # horizon (weekends, holidays, late data).
            try:
                bars_at_or_after = bars[bars.index >= horizon_end]
                if len(bars_at_or_after) > 0:
                    endpoint_close = float(bars_at_or_after["Close"].iloc[0])
                else:
                    endpoint_close = float(bars["Close"].iloc[-1])
            except Exception:
                endpoint_close = float(bars["Close"].iloc[-1])

            if direction_raw == "Bullish":
                endpoint_correct = endpoint_close > entry_price
            elif direction_raw == "Bearish":
                endpoint_correct = endpoint_close < entry_price
            elif direction_raw == "Neutral":
                # Neutral: endpoint correct if return stayed inside deadband.
                ret = (endpoint_close / entry_price) - 1
                endpoint_correct = abs(ret) <= NEUTRAL_RETURN_DEADBAND
            else:
                continue

            new_rating = "hit" if endpoint_correct else "miss"

            client.table("predictions").update({
                "rating_expiration": new_rating,
            }).eq("id", pred_id).execute()

            summary[f"marked_{new_rating}"] += 1
        except Exception as exc:
            logger.exception(
                "score_expiration_for_traded failed for %s: %s", pred_id, exc
            )
            summary["errors"].append({
                "prediction_id": pred_id,
                "error":         str(exc)[:200],
            })

    return summary


def score_inflight_predictions(now: Optional[datetime] = None) -> dict:
    """
    Update rating_target MID-FLIGHT — i.e. for predictions whose
    horizon hasn't elapsed yet but whose target has already been
    touched.

    Why this exists
    ───────────────
    `score_open_predictions()` only scores predictions whose
    horizon_ends_at has elapsed. For a 1-month AAPL call asked
    May 1 with horizon ending May 31, the rating fields stay
    'pending' for the full month even if price crossed the target
    on day 3. The ledger UI surfaces `rating_target` directly, so
    the user sees "TARGET: Pending" for the entire in-flight window
    despite the chart clearly showing price above the target line.

    What it touches
    ───────────────
    rating_target ONLY. Post-migration 0017, rating_checkpoint is owned
    by tick_checkpoints() — directional persistence at scheduled
    intervals, not a midpoint-touch check. rating_expiration stays
    'pending' here because the endpoint reading isn't decided until
    the horizon expires (price can reverse before the close). verdict
    also stays 'OPEN' for the same reason.

    Idempotency
    ───────────
    Only writes when the value would change (pending → hit). Never
    flips an existing 'hit' or 'miss' back to 'pending', and skips
    any prediction that already has a linked model_paper_trade
    (those are scored via close_model_trade and own their rating
    lifecycle).

    Returns a per-run summary for logs:
        { "checked": N, "marked_target": x,
          "skipped_traded": k, "skipped_nodata": z, "errors": [...] }
    """
    from db import _service_client

    now = now or datetime.now(timezone.utc)
    client = _service_client()

    # Pull open predictions still inside their window. We DON'T filter
    # on rating_target = 'pending' at the DB level because legacy rows
    # may have NULL for these columns; we handle both cases in-Python.
    inflight = (
        client.table("predictions")
              .select("*")
              .eq("verdict", "OPEN")
              .gt("horizon_ends_at", now.isoformat())
              .execute()
    ).data or []

    summary = {
        "checked":           len(inflight),
        "marked_target":     0,
        "skipped_traded":    0,
        "skipped_nodata":    0,
        "errors":            [],
    }

    for pred in inflight:
        pred_id = pred["id"]
        try:
            # If rating_target is already in a terminal state (hit or
            # miss), nothing left to do here — short-circuit to avoid
            # the yfinance round-trip. 'pending' is the only state that
            # invites another look. rating_checkpoint is no longer
            # touched in this function (post-migration 0017 it lives
            # on tick_checkpoints).
            rt = (pred.get("rating_target") or "pending").lower()
            if rt in ("hit", "miss"):
                continue

            # Skip traded predictions — model_paper_trades owns those
            # rating fields via close_model_trade.
            trade_check = (
                client.table("model_paper_trades")
                      .select("id", count="exact")
                      .eq("prediction_id", pred_id)
                      .execute()
            )
            if (trade_check.count or 0) > 0:
                summary["skipped_traded"] += 1
                continue

            direction_raw = (pred.get("direction") or "").strip()
            if direction_raw == "Bullish":
                direction = "LONG"
            elif direction_raw == "Bearish":
                direction = "SHORT"
            elif direction_raw == "Neutral":
                direction = "NEUTRAL"
            else:
                continue

            symbol = pred["symbol"]
            starts_at = _parse_iso(
                pred.get("horizon_starts_at") or pred.get("created_at")
            )

            bars = _intraday_bars_since(symbol, starts_at)
            if bars is None or bars.empty:
                summary["skipped_nodata"] += 1
                continue

            entry_price = float(pred.get("entry_price") or 0)

            update_fields: dict = {}

            if direction == "NEUTRAL":
                # Neutral prediction = "price stays within deadband".
                # In-flight semantic: if price has EXCEEDED 1× deadband,
                # rating_target flips to 'miss' (the tight band broke).
                # Post-migration 0017 we no longer touch rating_checkpoint
                # here — Neutral predictions don't generate checkpoint
                # scores at all (directional check is undefined for
                # Neutral).
                if entry_price <= 0:
                    # Can't compute deadband bounds without an entry.
                    continue
                upper_t = entry_price * (1 + NEUTRAL_RETURN_DEADBAND)
                lower_t = entry_price * (1 - NEUTRAL_RETURN_DEADBAND)
                highs = bars["High"].astype(float)
                lows = bars["Low"].astype(float)
                target_breached = bool(
                    (highs > upper_t).any() or (lows < lower_t).any()
                )
                if target_breached and rt != "miss":
                    update_fields["rating_target"] = "miss"

            else:
                # LONG / SHORT — scan for target touched in the predicted
                # direction. The checkpoint touch (midpoint between entry
                # and target) is no longer used to set rating_checkpoint —
                # that axis is owned by tick_checkpoints and measures
                # directional persistence, not midpoint touches.
                target_price = (
                    float(pred["target_price"])
                    if pred.get("target_price") else None
                )

                target_hit_bool, _, _ = _scan_bars(
                    bars, direction, target_price, checkpoint_price=None,
                    stop_price=None,
                )

                if target_hit_bool and rt != "hit":
                    update_fields["rating_target"] = "hit"

            if not update_fields:
                continue

            client.table("predictions").update(update_fields).eq(
                "id", pred_id
            ).execute()

            if "rating_target" in update_fields:
                summary["marked_target"] += 1
        except Exception as exc:
            logger.exception(
                "score_inflight_predictions failed for %s: %s", pred_id, exc
            )
            summary["errors"].append({
                "prediction_id": pred_id,
                "error":         str(exc)[:200],
            })

    return summary


# ─── Checkpoint scoring (new scheme, methodology § 4.3 post-migration 0017) ─
#
# The CHECKPOINT axis used to be "did the price touch the midpoint between
# entry and target during the window?" — a single binary hit per prediction,
# the same family as Target (magnitude check). That conflated three axes into
# two-and-a-half: Target and Checkpoint were essentially "did the path reach
# X% of the way" measurements that differed only in X.
#
# The new scheme makes Checkpoint a *directional persistence* axis,
# distinct from Target (magnitude) and Expiration (terminal direction):
# at each scheduled interval since the call was issued, was the close
# price on the correct side of entry? Each interval is one binary
# observation. The per-call rating_checkpoint surfaces "most recent
# interval stands" — a sensible glanceable "where is the call right now?"
# read — while the full trajectory is preserved in the new
# prediction_checkpoint_scores table for the receipt's Checkpoints tab.
#
# Intervals continue firing even after the linked paper trade closes on
# target / stop. A winning trade that exited on day 5 of a 1-year call
# can still demonstrate (or fail to demonstrate) that the call was
# *consistently* right over the following 360 days. Decoupling
# checkpoint from trade state is the whole point of the refactor.
#
# Neutral predictions are skipped — there's no directional check to make
# against entry. Their rating_checkpoint stays NULL.

# Per-horizon checkpoint schedule. Square-root-ish spacing — heavier
# sampling on longer horizons where directional drift accumulates,
# minimal sampling on short horizons where noise dominates. Add a new
# horizon by extending the dict; the schedule is the schedule.
HORIZON_INTERVALS: dict[str, list[int]] = {
    "3 Day":     [1],
    "1 Week":    [1, 3],
    "1 Month":   [1, 7, 14],
    "1 Quarter": [1, 7, 30, 60],
    "1 Year":    [1, 7, 30, 90, 180, 365],
}


def _checkpoint_intervals_for_horizon(horizon: str) -> list[int]:
    """Return the per-horizon checkpoint interval schedule.

    Unknown horizons fall back to a minimal schedule (1d, 7d) rather
    than raising — keeps tick_checkpoints resilient to legacy rows
    with horizon labels that pre-date the canonical set.
    """
    return HORIZON_INTERVALS.get((horizon or "").strip(), [1, 7])


def _close_at_or_after(symbol: str, target_dt: datetime) -> Optional[float]:
    """Return the daily close on `target_dt` (or the next trading day
    if it lands on a weekend/holiday). None when no bars are available.

    Uses daily bars (interval='1d'). The lookup window is target_dt to
    target_dt + 10 days — accommodates long weekends without pulling a
    full year of history per call.
    """
    try:
        end_dt = (target_dt + timedelta(days=10))
        hist = yf.Ticker(symbol).history(
            start=target_dt.date().isoformat(),
            end=end_dt.date().isoformat(),
            interval="1d",
        )
        if hist is None or hist.empty:
            return None
        # First available close at or after target_dt.
        first = hist.iloc[0]
        close = float(first["Close"])
        if not (close > 0):
            return None
        return close
    except Exception as exc:
        logger.exception(
            "_close_at_or_after fetch failed for %s @ %s: %s",
            symbol, target_dt.isoformat(), exc,
        )
        return None


def tick_checkpoints(now: Optional[datetime] = None) -> dict:
    """
    Fire any scheduled checkpoint intervals that have come due and
    aren't yet scored. Writes a row to prediction_checkpoint_scores
    per (prediction_id, interval_days) pair and updates the parent
    predictions.rating_checkpoint to the latest interval's status.

    Idempotency
    ───────────
    The new table's UNIQUE (prediction_id, interval_days) constraint
    prevents double-scoring at the DB layer. This function also
    short-circuits if every scheduled interval for a prediction is
    already represented — a one-liner that keeps the yfinance fetch
    out of the path on no-op runs.

    Cadence
    ───────
    Designed to run on the same cron as tick() — every 5 min during
    market hours plus the 4:15 PM EOD pass. Interval boundaries are
    measured in calendar days from prediction issue, so weekend
    intervals shift to the next trading day inside _close_at_or_after.

    Returns a per-run summary:
        { "checked": N, "fired": x, "skipped_neutral": k,
          "skipped_no_data": z, "errors": [...] }
    """
    from db import _service_client

    now = now or datetime.now(timezone.utc)
    client = _service_client()

    # Pull every non-Neutral prediction. We don't filter by horizon end
    # at the DB layer — intervals up to and including horizon_ends_at
    # are valid, and per-prediction "every interval scored" short-
    # circuiting handles the past-window case efficiently in-Python.
    preds = (
        client.table("predictions")
              .select(
                  "id, symbol, direction, entry_price, "
                  "created_at, horizon"
              )
              .neq("direction", "Neutral")
              .execute()
    ).data or []

    summary = {
        "checked":         len(preds),
        "fired":           0,
        "skipped_neutral": 0,  # filtered at the query, retained for symmetry
        "skipped_no_data": 0,
        "errors":          [],
    }

    for pred in preds:
        pred_id = pred["id"]
        try:
            symbol = pred["symbol"]
            direction = (pred.get("direction") or "").strip()
            entry_price = pred.get("entry_price")

            if not entry_price or float(entry_price) <= 0:
                # Can't direction-check against a missing entry.
                continue

            entry_price = float(entry_price)
            issued_at = _parse_iso(pred["created_at"])
            intervals = _checkpoint_intervals_for_horizon(pred.get("horizon", ""))

            # Which intervals are due (issued_at + N days <= now)?
            due = [
                i for i in intervals
                if issued_at + timedelta(days=i) <= now
            ]
            if not due:
                continue

            # Which are already scored? Pull once per prediction, not
            # per interval, to keep the query count linear in
            # predictions rather than quadratic.
            existing = (
                client.table("prediction_checkpoint_scores")
                      .select("interval_days")
                      .eq("prediction_id", pred_id)
                      .execute()
            ).data or []
            already_scored = {row["interval_days"] for row in existing}

            to_fire = [i for i in due if i not in already_scored]
            if not to_fire:
                continue

            # Fire each due interval. We iterate one-by-one rather than
            # batch-inserting so an individual yfinance failure doesn't
            # take down the whole prediction's pending intervals.
            for interval_days in to_fire:
                target_dt = issued_at + timedelta(days=interval_days)
                close_price = _close_at_or_after(symbol, target_dt)

                if close_price is None:
                    summary["skipped_no_data"] += 1
                    continue

                # Directional check against entry.
                if direction == "Bullish":
                    status = "hit" if close_price > entry_price else "miss"
                elif direction == "Bearish":
                    status = "hit" if close_price < entry_price else "miss"
                else:
                    # Defensive — the SQL query already excludes Neutral,
                    # but if anything sneaks through, skip cleanly.
                    summary["skipped_neutral"] += 1
                    continue

                client.table("prediction_checkpoint_scores").insert({
                    "prediction_id": pred_id,
                    "interval_days": interval_days,
                    "scored_at":     now.isoformat(),
                    "status":        status,
                    "close_price":   close_price,
                }).execute()
                summary["fired"] += 1

            # Update parent prediction's rating_checkpoint to reflect the
            # latest scored interval. We re-fetch (rather than relying on
            # in-Python state) so the value is always derived from the
            # canonical table — handles races with parallel runs cleanly.
            latest = (
                client.table("prediction_checkpoint_scores")
                      .select("status")
                      .eq("prediction_id", pred_id)
                      .order("interval_days", desc=True)
                      .limit(1)
                      .execute()
            ).data
            if latest:
                client.table("predictions").update({
                    "rating_checkpoint": latest[0]["status"],
                }).eq("id", pred_id).execute()

        except Exception as exc:
            logger.exception("tick_checkpoints failed for %s: %s", pred_id, exc)
            summary["errors"].append({
                "prediction_id": pred_id,
                "error":         str(exc)[:200],
            })

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> int:
    """
    CLI entry point. Examples:

        python -m scoring_worker tick    # run one 5-min check
        python -m scoring_worker eod     # run the 4:15 PM EOD pass
        python -m scoring_worker both    # tick then eod (use at the 4:15 mark)
    """
    args = sys.argv[1:]
    cmd = args[0] if args else "tick"

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if cmd == "tick":
        result = tick()
        # Mid-flight rating update for non-traded ("Watched") predictions.
        # Lives alongside tick() because the GitHub Actions cron runs
        # "tick" every 5 min during market hours, and we want target
        # ratings to flip from pending → hit on the same cadence —
        # not wait until the next 20:15 "both" pass.
        inflight = score_inflight_predictions()
        # Expiration resolver for traded predictions whose horizon has
        # just elapsed. Pairs with the close-at-target/stop path which
        # leaves rating_expiration='pending' until the horizon arrives.
        expiration = score_expiration_for_traded()
        # NEW: directional-persistence checkpoint scoring. Fires any
        # interval that's come due on any non-Neutral prediction and
        # writes a row to prediction_checkpoint_scores; rating_checkpoint
        # on the parent prediction is updated to the latest interval's
        # status. Runs on the same cadence as tick() so the receipt /
        # ledger surfaces fire on a sensible timeline.
        checkpoints = tick_checkpoints()
        print(f"tick             : {result}")
        print(f"score_inflight   : {inflight}")
        print(f"score_expiration : {expiration}")
        print(f"tick_checkpoints : {checkpoints}")
        return 0
    if cmd == "eod":
        result = eod_pass()
        print(f"eod : {result}")
        return 0
    if cmd == "users":
        result = user_eod_pass()
        print(f"users: {result}")
        return 0
    if cmd in ("score_open", "watched", "score_watched"):
        # Score Watched (non-traded) predictions whose horizon has elapsed.
        # tick() handles traded predictions via close_model_trade RPC; this
        # fills the gap so verdict ≠ "OPEN" forever for Watched calls.
        result = score_open_predictions()
        print(f"score_open: {result}")
        return 0
    if cmd in ("score_inflight", "inflight"):
        # Mid-flight rating update: scan still-open predictions whose
        # target or checkpoint has already been touched and mark
        # rating_target / rating_checkpoint = 'hit'. Leaves verdict
        # and rating_expiration alone (those decide at horizon end).
        result = score_inflight_predictions()
        print(f"score_inflight: {result}")
        return 0
    if cmd in ("score_expiration", "expiration"):
        # Resolve pending rating_expiration for traded predictions
        # whose horizon has now elapsed. Mirror of score_open but
        # for the trade-side close path that leaves rating_expiration
        # pending when closing early at target/stop.
        result = score_expiration_for_traded()
        print(f"score_expiration: {result}")
        return 0
    if cmd in ("tick_checkpoints", "checkpoints"):
        # Standalone interval-firing pass. Useful for debugging the
        # new checkpoint scheme without running the full tick chain,
        # and for the historical backfill (which is a one-shot full
        # sweep at much longer per-prediction cost than the steady-
        # state cron).
        result = tick_checkpoints()
        print(f"tick_checkpoints: {result}")
        return 0
    if cmd == "both":
        t = tick()
        e = eod_pass()
        u = user_eod_pass()
        s = score_open_predictions()
        # Inflight scoring lives alongside score_open so the ledger
        # surfaces target-hit immediately for any in-flight call,
        # not only after the horizon expires.
        inf = score_inflight_predictions()
        # Expiration resolver for traded predictions whose horizon
        # just elapsed (closed-early-via-target-or-stop case).
        exp = score_expiration_for_traded()
        # NEW: checkpoint interval scoring — fires any due intervals
        # and updates rating_checkpoint on the parent prediction.
        cp = tick_checkpoints()
        r = refresh_model_adjustments()
        print(f"tick             : {t}")
        print(f"eod              : {e}")
        print(f"users            : {u}")
        print(f"score_open       : {s}")
        print(f"score_inflight   : {inf}")
        print(f"score_expiration : {exp}")
        print(f"tick_checkpoints : {cp}")
        print(f"refresh          : {r}")
        return 0
    if cmd in ("backfill_option_expiries", "backfill"):
        from db import backfill_option_expiry_dates
        dry = "--dry-run" in args
        result = backfill_option_expiry_dates(dry_run=dry)
        print(f"backfill{'  (dry-run)' if dry else ''}: {result}")
        return 0
    if cmd == "refresh":
        result = refresh_model_adjustments()
        print(f"refresh: {result}")
        return 0
    if cmd in ("retrain_pending", "retrain"):
        result = retrain_pending_stocks()
        print(f"retrain_pending: {result}")
        return 0

    print(f"unknown command: {cmd!r}; expected tick|eod|users|score_open|"
          f"score_inflight|score_expiration|tick_checkpoints|"
          f"both|backfill|refresh|retrain_pending")
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
