"""Nightly per-symbol model snapshot pass.

Walks the canonical universe (S&P 500 + NASDAQ-100 + Dow 30, deduped),
runs StockPredictor.predict() for each ticker, and writes one row per
(symbol, snapshot_date, horizon) into the `symbol_snapshots` Supabase
table.

Powers the model-analysis card on every /stock/[symbol] SEO page. Runs
nightly via GitHub Actions cron after the EOD scoring pass — see
.github/workflows/score.yml.

Design notes:
  • Bypasses the FastAPI route deliberately. The route does a lot of
    extra enrichment (options strategy generation, narrative analysis,
    trade-plan computation, log_prediction_v2 writes) that we don't want
    polluting the public ledger. Calling StockPredictor.predict()
    directly skips all of it.
  • One ticker failure does NOT break the batch. We catch + log per
    symbol so a missing yfinance row, an unloadable model, or a
    transient network blip doesn't take down the other 599.
  • Idempotent. The migration's UNIQUE (symbol, snapshot_date, horizon)
    constraint plus our upsert on conflict means re-running the batch
    on the same day overwrites in place. Safe to retry.
  • Methodology compliance: snapshots are NOT counted toward § 5
    track-record. They live in their own table, are flagged "system-
    authored" by absence of user_id, and the page components treat
    them as inference artifacts not paper-money commitments.

Usage:
    python -m nightly_symbol_pass            # full canonical universe
    python -m nightly_symbol_pass AAPL MSFT  # specific tickers (debug)
    python -m nightly_symbol_pass --limit 5  # first 5 tickers (smoke test)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Map model.py horizon names → short codes the DB / frontend uses.
HORIZON_NAME_TO_CODE = {
    "3 Day":     "3d",
    "1 Week":    "1w",
    "1 Month":   "1m",
    "1 Quarter": "1q",
    "1 Year":    "1y",
}
HORIZON_DAYS = {
    "3d": 3,
    "1w": 7,
    "1m": 30,
    "1q": 90,
    "1y": 365,
}

# Direction deadband — match methodology § 4.1 exactly. Anything inside
# ±0.5% predicted return collapses to Neutral so the snapshots agree
# with the user-asked predictions on the same definition.
NEUTRAL_RETURN_DEADBAND = 0.005


def _direction(pred_return: Optional[float]) -> Optional[str]:
    """Bullish / Bearish / Neutral from a fractional return."""
    if pred_return is None:
        return None
    if abs(pred_return) < NEUTRAL_RETURN_DEADBAND:
        return "Neutral"
    return "Bullish" if pred_return > 0 else "Bearish"


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _build_snapshot_rows(
    symbol: str,
    predictions: dict,
    snapshot_date: str,
    model_version: Optional[str],
    regime: Optional[str],
) -> list[dict]:
    """Map a predict() response into one row per horizon for Supabase upsert."""
    rows: list[dict] = []
    for horizon_name, h in (predictions or {}).items():
        if not isinstance(h, dict):
            continue
        # Skip suppressed / not-trained horizons — pages can render the
        # "model has no view at this horizon" empty state when the row
        # is missing, which is more honest than persisting noise.
        if h.get("suppress") or h.get("skipped"):
            continue

        code = HORIZON_NAME_TO_CODE.get(horizon_name)
        if not code:
            continue

        pred_return  = _safe_float(h.get("predicted_return"))
        pred_price   = _safe_float(h.get("predicted_price"))
        cur_price    = _safe_float(h.get("current_price"))
        confidence   = _safe_float(h.get("confidence"))

        horizon_days = HORIZON_DAYS.get(code)
        horizon_end = None
        if horizon_days:
            try:
                snap_dt = date.fromisoformat(snapshot_date)
                horizon_end = (snap_dt + timedelta(days=horizon_days)).isoformat()
            except ValueError:
                pass

        # Trim signals_summary to a small JSON blob — keep the model's
        # internal calculations readable from the page without bloating
        # the row. Skip numpy arrays and large nested objects.
        signals_summary = {
            "interval_low":      _safe_float(h.get("interval_low")),
            "interval_high":     _safe_float(h.get("interval_high")),
            "ensemble_agreement": _safe_float(h.get("ensemble_agreement")),
            "val_dir_accuracy":  _safe_float(h.get("val_dir_accuracy")),
            "n_models":          h.get("n_models"),
            "n_dropped":         h.get("n_dropped"),
            "avg_member_acc":    _safe_float(h.get("avg_member_acc")),
            "is_extreme":        bool(h.get("is_extreme")) if h.get("is_extreme") is not None else None,
            "confidence_label":  h.get("confidence_label"),
            "suppress_reason":   h.get("suppress_reason"),
        }

        rows.append({
            "symbol":           symbol.upper(),
            "snapshot_date":    snapshot_date,
            "horizon":          code,
            "direction":        _direction(pred_return),
            "confidence":       confidence,
            "predicted_price":  pred_price,
            "predicted_return": pred_return,
            "current_price":    cur_price,
            "horizon_end":      horizon_end,
            "horizon_days":     horizon_days,
            # Trade plan + options strategy intentionally left null for
            # snapshots — they're inference artifacts, not paper trades.
            # Pages that want a snapshot-flavored "if the model traded
            # today" block can compute it on the fly from entry/target/stop
            # heuristics or skip it. See migration 0011 docstring.
            "entry_price":        None,
            "stop_price":         None,
            "target_price":       None,
            "traded":             None,
            "trade_pass_reason":  None,
            "options_strategy":   None,
            "model_version":      model_version,
            "regime":             regime,
            "signals_summary":    signals_summary,
        })
    return rows


def _process_symbol(
    symbol: str,
    snapshot_date: str,
    market_ctx: Any,
) -> tuple[bool, str]:
    """Run the model on one symbol and persist its snapshot rows.

    Returns (ok, status_message). Failures are caught and logged here
    so the outer loop never crashes mid-batch.
    """
    try:
        from data_fetcher import fetch_stock_data
        from model import StockPredictor
        from db import upsert_symbol_snapshot
    except Exception as exc:
        return False, f"import failed: {exc}"

    try:
        df = fetch_stock_data(symbol, period="2y")
    except Exception as exc:
        return False, f"fetch_stock_data failed: {exc}"
    if df is None or len(df) < 60:
        return False, "insufficient history (<60 bars)"

    try:
        predictor = StockPredictor(symbol)
        # Permissive max_age — nightly batch shouldn't trigger retraining
        # on a model that's a few months old. Long-aged models still
        # produce inference; if a model genuinely needs retraining the
        # auto-retrain pipeline catches it on the next user ask.
        predictor.load_model(max_age_hours=24 * 365)
    except Exception as exc:
        return False, f"load_model failed: {exc}"

    if not getattr(predictor, "is_trained", False):
        return False, "model not trained"

    try:
        predictions = predictor.predict(df, market_ctx=market_ctx)
    except Exception as exc:
        return False, f"predict failed: {exc}"

    if not predictions:
        return False, "no predictions returned"

    model_version = getattr(predictor, "model_version", None) or getattr(predictor, "version", None)
    regime = getattr(predictor, "current_regime", None) or getattr(predictor, "regime", None)

    rows = _build_snapshot_rows(
        symbol=symbol,
        predictions=predictions,
        snapshot_date=snapshot_date,
        model_version=model_version,
        regime=regime,
    )

    if not rows:
        return False, "no horizon rows produced"

    persisted = 0
    for row in rows:
        try:
            upsert_symbol_snapshot(row)
            persisted += 1
        except Exception as exc:
            logger.warning(
                "upsert failed for %s/%s/%s: %s",
                row["symbol"], row["snapshot_date"], row["horizon"], exc,
            )

    return True, f"persisted {persisted}/{len(rows)} horizons"


def _resolve_universe(symbols: list[str], limit: Optional[int]) -> list[str]:
    """Return the deduped, uppercased ticker list to process."""
    if symbols:
        return [s.strip().upper() for s in symbols if s.strip()]
    try:
        from universe import get_full_universe
    except Exception as exc:
        logger.error("universe import failed: %s", exc)
        return []
    u = get_full_universe()
    canon: set[str] = set()
    for key in ("sp500", "nasdaq100", "dow30"):
        for t in (u.get(key) or []):
            canon.add(str(t).strip().upper())
    out = sorted(canon)
    if limit and limit > 0:
        out = out[:limit]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Nightly per-symbol snapshot pass.")
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Specific tickers to process. Empty = full canonical universe.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the universe size (debug / smoke test).",
    )
    parser.add_argument(
        "--snapshot-date",
        type=str,
        default=None,
        help="Override snapshot_date (YYYY-MM-DD). Defaults to today UTC.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG-level logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if os.environ.get("USE_SUPABASE", "").lower() != "true":
        logger.error(
            "USE_SUPABASE is not 'true'. The nightly batch writes to "
            "Supabase only — bailing rather than no-op'ing silently.",
        )
        return 1

    snapshot_date = args.snapshot_date or datetime.now(timezone.utc).date().isoformat()
    logger.info("nightly_symbol_pass starting — snapshot_date=%s", snapshot_date)

    tickers = _resolve_universe(args.symbols, args.limit or None)
    if not tickers:
        logger.error("no tickers resolved — universe load probably failed.")
        return 1
    logger.info("processing %d tickers", len(tickers))

    # Fetch market context once at the top of the batch — same context
    # is shared across all 600 tickers, no need to refetch per-symbol.
    try:
        from data_fetcher import fetch_market_context
        market_ctx = fetch_market_context()
    except Exception as exc:
        logger.warning("fetch_market_context failed (continuing without): %s", exc)
        market_ctx = None

    ok = 0
    failed = 0
    failures: list[tuple[str, str]] = []
    started = time.time()

    for i, sym in enumerate(tickers, start=1):
        t0 = time.time()
        try:
            success, msg = _process_symbol(sym, snapshot_date, market_ctx)
        except Exception as exc:  # noqa: BLE001
            success, msg = False, f"unhandled: {exc}\n{traceback.format_exc()[:300]}"

        elapsed = time.time() - t0
        if success:
            ok += 1
            logger.info("[%4d/%d] %-7s OK  %s (%.1fs)", i, len(tickers), sym, msg, elapsed)
        else:
            failed += 1
            failures.append((sym, msg))
            logger.warning("[%4d/%d] %-7s FAIL %s (%.1fs)", i, len(tickers), sym, msg, elapsed)

    total = time.time() - started
    logger.info(
        "nightly_symbol_pass done — %d ok, %d failed, %.1fs total (avg %.2fs/ticker)",
        ok, failed, total, total / max(1, len(tickers)),
    )
    if failures:
        logger.info("failure summary (first 20):")
        for sym, msg in failures[:20]:
            logger.info("  %-7s  %s", sym, msg[:120])

    # Exit code: success unless EVERYTHING failed (in which case
    # something systemic is wrong and the cron job should fire an alert).
    return 0 if ok > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
