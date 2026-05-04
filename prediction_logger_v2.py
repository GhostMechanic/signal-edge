"""
prediction_logger_v2.py
========================
Production-grade prediction tracking system with:
  - Progressive scoring (1d → 3d → 7d → 30d → 90d → 365d)
  - Accelerated backtest scoring (instant historical accuracy)
  - Feature importance tracking per prediction
  - Model version recording
  - Confidence calibration data
  - CSV/Excel export

Usage:
    log_prediction_v2(symbol, predictions, feature_importance, model_version)
    quick_score_all()               # runs 1-day quick checks
    score_all_intervals()           # runs all interval scoring
    backtest_accuracy(symbol, df)   # synthetic backtest scoring
    get_full_analytics()            # returns complete analytics dict
    export_predictions_csv(path)    # export to CSV
"""

import json
import logging
import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List

import yfinance as yf

# SQLite-backed durable store. Replaces the JSON file as the source of
# truth — see prediction_store.py for the schema and rationale.
import prediction_store

logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".predictions")
LOG_FILE = os.path.join(LOG_DIR, "prediction_log_v2.json")  # legacy; no longer written
IMPORTANCE_FILE = os.path.join(LOG_DIR, "feature_importance.json")
MODEL_VERSION_FILE = os.path.join(LOG_DIR, "model_versions.json")

# ─── Scoring intervals (days) ───────────────────────────────────────────────
SCORE_INTERVALS = {
    "3 Day":     [1, 3],               # scores fully in 3 trading days
    "1 Week":    [1, 3, 7],
    "1 Month":   [1, 3, 7, 14, 30],
    "1 Quarter": [1, 7, 14, 30, 60, 90],
    "1 Year":    [1, 7, 30, 90, 180, 365],
}

# Final scoring interval per horizon
FINAL_INTERVALS = {"3 Day": 3, "1 Week": 7, "1 Month": 30, "1 Quarter": 90, "1 Year": 365}


def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA I/O
# ═══════════════════════════════════════════════════════════════════════════════

class LogCorruptedError(Exception):
    """
    Raised when the prediction log file exists on disk but can't be
    parsed as a JSON array. We bubble this up rather than silently
    treating the log as empty, because a silent fallback to [] meant a
    single subsequent write would atomically replace the entire history
    with one entry — exactly the failure mode that wiped 1,062
    predictions in late April 2026.
    """


def _load_log() -> list:
    """
    Return all predictions in the legacy list-of-dicts shape, sourced
    from the SQLite store.

    Kept around as a compat shim so all the analytics code in this
    module (get_full_analytics, score_all_intervals, deduplicate_log,
    etc.) keeps working unchanged. Direct SQLite access via
    prediction_store is preferred for new code.
    """
    return prediction_store.get_all_predictions()


def _save_log(entries: list):
    """
    Compatibility shim — older code paths in this module loaded the
    full log with _load_log(), mutated entries in-place, then saved
    the entire list back. With SQLite as the source of truth that's
    no longer the right model: we now upsert each entry individually.

    Shrink-guard: refuse to operate if `entries` is materially smaller
    than what's currently stored. The legitimate callers (log_prediction,
    score_all_intervals, backtest_accuracy) only ever modify entries
    in-place or append; they never bulk-delete. A shrink here is a bug.
    """
    current_count = prediction_store.count_predictions()
    if len(entries) < current_count - 1:
        raise RuntimeError(
            f"Refusing to shrink prediction log from "
            f"{current_count} → {len(entries)} entries via _save_log. "
            f"This would destroy data. If this is intentional "
            f"(e.g., one-time pruning), call _save_log_unsafe() "
            f"explicitly."
        )

    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("prediction_id"):
            continue
        try:
            prediction_store.insert_prediction(entry)
        except Exception as e:
            print(f"[_save_log] failed to upsert {entry.get('symbol')} {entry.get('prediction_id')}: {e}")


def _save_log_unsafe(entries: list):
    """
    Bypass the shrink-guard. Used by the deduplicate_log() one-shot
    cleanup, which intentionally collapses duplicate (symbol, date)
    rows. Still goes through the SQLite upsert path — no JSON write.
    """
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("prediction_id"):
            continue
        try:
            prediction_store.insert_prediction(entry)
        except Exception as e:
            print(f"[_save_log_unsafe] failed to upsert {entry.get('symbol')} {entry.get('prediction_id')}: {e}")


def _load_importance() -> dict:
    _ensure_dir()
    if os.path.exists(IMPORTANCE_FILE):
        try:
            with open(IMPORTANCE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"features": {}, "update_count": 0}
    return {"features": {}, "update_count": 0}


def _save_importance(data: dict):
    _ensure_dir()
    with open(IMPORTANCE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_model_versions() -> dict:
    _ensure_dir()
    if os.path.exists(MODEL_VERSION_FILE):
        try:
            with open(MODEL_VERSION_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"current_version": "1.0", "versions": [], "retrain_count": 0}
    return {"current_version": "1.0", "versions": [], "retrain_count": 0}


def _save_model_versions(data: dict):
    _ensure_dir()
    with open(MODEL_VERSION_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def log_prediction_v2(
    symbol: str,
    predictions: dict,
    feature_importance: Optional[dict] = None,
    model_version: str = "1.0",
    regime: str = "Unknown",
    *,
    ohlc_df: Optional[pd.DataFrame] = None,
    user_id: Optional[str] = None,
    user_horizon_code: Optional[str] = None,
    options_strategies: Optional[dict] = None,
    out_meta: Optional[dict] = None,
) -> str:
    """
    Log a new prediction with enhanced tracking.

    Returns the prediction_id (a full UUID), or "" if the underlying store
    rejected the write. Routes through `db.insert_prediction()` so file
    mode and Supabase mode are transparent to the caller.

    Phase ordering (per data-model anchor § 5 — model_paper_trades.prediction_id
    is a FK to predictions.id, so the prediction row must land first):
      1. Generate UUID.
      2. Compute trade plan + TRADE/PASS decision (no DB writes).
      3. Stamp plan + decision onto the record.
      4. Insert prediction row (db.insert_prediction handles the
         file-vs-Supabase routing).
      5. If decision.traded: open the model paper trade — references the
         now-existing prediction by FK.

    `ohlc_df` is optional but strongly preferred. When omitted, the
    prediction persists with `traded=False` and no plan; callers that
    don't yet pass OHLC keep working without re-architecture.

    Failures in steps 2, 3, or 5 are logged but never block step 4. The
    user always gets a persisted prediction; the model's portfolio just
    doesn't reflect predictions that broke trade attachment.
    """
    # Full UUID (was 8-char prefix). Required for Supabase predictions.id
    # which is `uuid` type. Legacy 8-char IDs in existing SQLite rows are
    # untouched — the column is TEXT so it tolerates either width.
    pred_id = str(uuid.uuid4())

    # ── Anchor § 8.1 same-day dedupe ──────────────────────────────────────
    # If this user already has a prediction for (symbol, user_horizon_code)
    # within the current UTC day, return that prediction's id and skip all
    # downstream work — no fresh insert, no trade-plan compute, no
    # model_paper_trades open. Prevents duplicate ledger rows and the
    # scoring imbalance that would follow from two scored outcomes against
    # what the user means as one prediction.
    #
    # Granularity:
    #   • Supabase mode:  (user_id, symbol, horizon_code, today_utc)
    #   • File-mode:      (symbol, today_utc) — legacy SQLite has no
    #                     user_id and rolls all five horizons under one
    #                     prediction_id, so the dedupe is per-symbol.
    try:
        from db import find_todays_prediction
        existing_pred_id = find_todays_prediction(
            symbol=symbol,
            horizon_code=user_horizon_code,
            user_id=user_id,
        )
        if existing_pred_id:
            print(
                f"[log_prediction_v2 {symbol}] dedupe hit — re-ask of "
                f"existing prediction {existing_pred_id} (user_horizon="
                f"{user_horizon_code}); skipping insert + trade-open per "
                f"anchor § 8.1"
            )
            return existing_pred_id
    except Exception as exc:
        # Lookup failure is non-fatal: fall through to the normal write
        # path. Worst case we get a duplicate row (the existing behavior
        # before this change) — the dedupe is a guarantee, not a critical
        # path.
        logger.exception(
            "find_todays_prediction failed for %s; proceeding with fresh "
            "write: %s", symbol, exc,
        )

    record = {
        "prediction_id":     pred_id,
        "symbol":            symbol.upper(),
        "timestamp":         datetime.now().isoformat(),
        "model_version":     model_version,
        "regime":            regime,
        "horizons":          {},
        # Anchor-doc §§ 2.1, 4.4 fields. Filled by _compute_trade_attachment
        # below when ohlc_df is provided. Persisted by the Supabase write
        # path; ignored by the legacy SQLite store until those columns
        # land there too.
        "traded":            False,
        "entry_price":       None,
        "stop_price":        None,
        "target_price":      None,
        "trade_pass_reason": None,
        "model_trade_id":    None,
        # User's selected horizon (3d|1w|1m|1q|1y). When set, the Supabase
        # writer uses this as the canonical row's horizon — so the public
        # ledger reflects what the user *asked*, not what the priority-
        # based fallback would pick. None means "use default fallback."
        "user_horizon_code": user_horizon_code,
        # Per-horizon options strategies recommended at predict-time. The
        # Supabase writer pulls the canonical horizon's strategy out and
        # persists it on the row's `options_strategy` JSONB column, so
        # users can paper-trade the play later from the detail page.
        # File-mode SQLite ignores this field.
        "options_strategies": options_strategies,
    }

    for horizon, data in predictions.items():
        pred_return = data.get("predicted_return", 0)
        record["horizons"][horizon] = {
            "predicted_return":   pred_return,
            "predicted_price":    data.get("predicted_price", 0),
            "current_price":      data.get("current_price", 0),
            "confidence":         data.get("confidence", 0),
            "ensemble_agreement": data.get("ensemble_agreement", 0),
            "val_dir_accuracy":   data.get("val_dir_accuracy", 0),
            "direction":          "up" if pred_return > 0 else "down",
            "top_features":       _top_features(feature_importance, horizon) if feature_importance else [],
            "scores":             {},
            "final_scored":       False,
            "final_correct":      None,
        }

    # ── Phase 1+2: compute the trade plan + TRADE/PASS decision ──────────────
    # Pure logic, no DB writes. Stamps plan + decision onto `record`.
    pending_open: Optional[dict] = None
    if ohlc_df is not None:
        try:
            pending_open = _compute_trade_attachment(record, predictions, ohlc_df, symbol)
        except Exception as exc:
            logger.exception(
                "trade attachment compute failed for %s/%s; prediction will "
                "persist with traded=False: %s", symbol, pred_id, exc
            )

    # ── Phase 3: persist the prediction row (db handles routing) ─────────────
    try:
        from db import insert_prediction as _db_insert_prediction
        persisted_id = _db_insert_prediction(record, user_id=user_id) or pred_id
    except Exception as e:
        # When USE_SUPABASE=true and the Supabase write fails, falling
        # back to the legacy SQLite store is a correctness hazard — the
        # prediction lands somewhere the rest of the app (ledger, model
        # paper trades, scoring worker) can't see. Better to fail loudly
        # so the caller can surface a clear error to the user and they
        # can retry, than to silently produce a phantom row.
        try:
            from db import USE_SUPABASE as _USE_SUPABASE
        except Exception:
            _USE_SUPABASE = False
        if _USE_SUPABASE:
            logger.exception(
                "db.insert_prediction FAILED in Supabase mode for %s; "
                "refusing legacy SQLite fallback to keep ledger/portfolio "
                "in sync. Surfacing the failure to the caller.", symbol
            )
            return ""
        # Legacy / dev mode (USE_SUPABASE=false): SQLite IS the canonical
        # store, so fall back as before.
        logger.exception(
            "db.insert_prediction failed in legacy mode; falling back to "
            "prediction_store: %s", e
        )
        try:
            prediction_store.insert_prediction(record)
            persisted_id = pred_id
        except Exception as e2:
            print(f"[log_prediction_v2] CRITICAL: store insert failed: {e2}")
            return ""

    # ── Phase 4: open the model paper trade (only when decision said TRADE) ──
    # Now safe to open: the prediction row exists and the FK can resolve.
    if pending_open is not None:
        try:
            _open_pending_model_trade(record, persisted_id, pending_open)
        except Exception as exc:
            # Trade-open failure leaves the prediction in a slightly
            # inconsistent state (traded=true on the row, no
            # model_paper_trades row). Log loudly so the audit job can
            # surface it; don't roll back the prediction itself.
            logger.exception(
                "model trade open FAILED for %s/%s — prediction is "
                "persisted with traded=true but no model_paper_trades row "
                "exists. Audit follow-up required: %s",
                symbol, persisted_id, exc
            )

    # Side-channel metadata for callers (api/main.py) that want the
    # trade-attachment outcome surfaced on the predict response without
    # changing log_prediction_v2's return type. Populated only when an
    # `out_meta` dict was supplied; non-blocking if not.
    if out_meta is not None and isinstance(out_meta, dict):
        out_meta["traded"]            = bool(record.get("traded", False))
        out_meta["trade_pass_reason"] = record.get("trade_pass_reason")
        out_meta["entry_price"]       = record.get("entry_price")
        out_meta["stop_price"]        = record.get("stop_price")
        out_meta["target_price"]      = record.get("target_price")
        # Compute the actual R:R the geometry produced — useful both for
        # the frontend's "no trade — risk geometry" copy and for any
        # downstream telemetry that wants to histogram R:R distribution.
        try:
            entry  = record.get("entry_price")
            stop   = record.get("stop_price")
            target = record.get("target_price")
            if entry is not None and stop is not None and target is not None:
                risk = abs(float(entry) - float(stop))
                rew  = abs(float(target) - float(entry))
                out_meta["risk_reward"] = round(rew / risk, 4) if risk > 0 else None
            else:
                out_meta["risk_reward"] = None
        except Exception:  # noqa: BLE001
            out_meta["risk_reward"] = None

    return persisted_id


# ─── Trade attachment (anchor docs §§ 2.1, 4.4) ───────────────────────────────

# Lazy-import the orchestrator + repo so this module still works on hosts
# where the new files haven't been deployed yet. Resolved on first use.
_compute_trade_attachment_fn = None
_open_model_trade_fn         = None
_get_portfolio_repo_fn       = None
_canonical_universe_fn       = None


def _resolve_trade_imports():
    """Lazy-import the trade-attachment dependencies. Caches resolution
    on the module so we only pay the import cost once."""
    global _compute_trade_attachment_fn, _open_model_trade_fn
    global _get_portfolio_repo_fn, _canonical_universe_fn
    if _compute_trade_attachment_fn is None:
        from prediqt_open_trade import (
            compute_trade_attachment,
            open_model_trade_for_prediction,
        )
        from model_portfolio import get_portfolio_repo
        try:
            from universe import canonical_universe as _uni
        except (ImportError, AttributeError):
            _uni = lambda: frozenset()
        _compute_trade_attachment_fn = compute_trade_attachment
        _open_model_trade_fn         = open_model_trade_for_prediction
        _get_portfolio_repo_fn       = get_portfolio_repo
        _canonical_universe_fn       = _uni
    return (
        _compute_trade_attachment_fn,
        _open_model_trade_fn,
        _get_portfolio_repo_fn,
        _canonical_universe_fn,
    )


def _compute_trade_attachment(
    record: dict,
    predictions: dict,
    ohlc_df: pd.DataFrame,
    symbol: str,
) -> Optional[dict]:
    """Phase 1: compute plan + decision, stamp onto `record`.

    Returns a `pending_open` payload (dict with the data needed to open
    the trade in Phase 4) when the decision is TRADE; None when PASS.
    """
    compute_fn, _open_fn, get_repo, get_universe = _resolve_trade_imports()

    # Same long-form horizon name lookup db._pick_canonical_horizon uses.
    # When the user picked a specific horizon at /api/predict (e.g. "1y"),
    # the trade attachment honors it — so target_price = predicted_price
    # for the *user's* horizon, not always for the 1-Month default.
    HORIZON_CODE_TO_NAME = {
        "3d": "3 Day", "1w": "1 Week", "1m": "1 Month",
        "1q": "1 Quarter", "1y": "1 Year",
    }
    HORIZON_PRIORITY = ["1 Month", "1 Quarter", "1 Year", "1 Week", "3 Day"]

    user_code = record.get("user_horizon_code")
    horizon = None
    if user_code:
        candidate = HORIZON_CODE_TO_NAME.get(user_code)
        if candidate and candidate in predictions:
            horizon = candidate
    # Fallback to priority-list pick when the user didn't supply a horizon
    # (CLI batch scans, legacy callers) or the picked horizon was suppressed.
    if horizon is None:
        horizon = next((h for h in HORIZON_PRIORITY if h in predictions), None)
    if horizon is None:
        return None

    h = predictions[horizon]
    confidence      = float(h.get("confidence", 0) or 0)
    entry_price     = float(h.get("current_price", 0) or 0)
    predicted_price = float(h.get("predicted_price", 0) or 0)
    pred_return     = float(h.get("predicted_return", 0) or 0)

    if entry_price <= 0 or predicted_price <= 0:
        return None

    # Direction with the methodology § 4.1 deadband — see trade_decision.
    from trade_decision import direction_from_return
    direction = direction_from_return(pred_return)

    repo  = get_repo()
    state = repo.get_state()
    res   = compute_fn(
        symbol             = symbol,
        direction          = direction,
        confidence         = confidence,
        entry_price        = entry_price,
        predicted_price    = predicted_price,
        ohlc_for_atr       = ohlc_df,
        canonical_universe = get_universe(),
        portfolio_state    = state,
    )

    record["entry_price"]       = res.plan.entry_price
    record["stop_price"]        = res.plan.stop_price
    record["target_price"]      = res.plan.target_price
    record["traded"]            = res.decision.traded
    record["trade_pass_reason"] = (
        res.decision.reason.value if res.decision.reason is not None else None
    )

    if res.decision.traded:
        return {
            "symbol":          symbol,
            "direction":       direction,
            "plan":            res.plan,
            "portfolio_repo":  repo,
            "portfolio_state": state,
        }
    return None


def _open_pending_model_trade(record: dict, persisted_id: str, pending: dict) -> None:
    """Phase 4: open the model paper trade. Called only after the
    prediction row has been inserted (so the FK on prediction_id resolves)."""
    _compute, open_fn, _repo_factory, _uni = _resolve_trade_imports()
    open_result = open_fn(
        prediction_id   = persisted_id,
        symbol          = pending["symbol"],
        direction       = pending["direction"],
        plan            = pending["plan"],
        portfolio_repo  = pending["portfolio_repo"],
        portfolio_state = pending["portfolio_state"],
    )
    record["model_trade_id"] = open_result.trade_id


def deduplicate_log() -> dict:
    """
    One-time cleanup: remove duplicate entries for same symbol+date.
    Keeps the entry with the most scores (best data), merges scores from others.
    Returns summary of what was removed.
    """
    entries = _load_log()
    original_count = len(entries)

    # Group by symbol + date
    groups: dict = {}
    for entry in entries:
        key = (entry["symbol"], entry["timestamp"][:10])
        groups.setdefault(key, []).append(entry)

    cleaned = []
    for (sym, date), group in groups.items():
        if len(group) == 1:
            cleaned.append(group[0])
            continue

        # Pick the entry with the most total scores across all horizons
        def _score_count(e):
            return sum(len(h.get("scores", {})) for h in e.get("horizons", {}).values())

        group.sort(key=_score_count, reverse=True)
        best = group[0]

        # Merge scores + final flags from all duplicates into the best one
        for dup in group[1:]:
            for horizon, dup_hdata in dup.get("horizons", {}).items():
                best_hdata = best.get("horizons", {}).get(horizon)
                if not best_hdata:
                    continue
                for score_key, score_val in dup_hdata.get("scores", {}).items():
                    if score_key not in best_hdata["scores"]:
                        best_hdata["scores"][score_key] = score_val
                if dup_hdata.get("final_scored") and not best_hdata.get("final_scored"):
                    best_hdata["final_scored"] = True
                    best_hdata["final_correct"] = dup_hdata.get("final_correct")

        cleaned.append(best)

    # Sort by timestamp
    cleaned.sort(key=lambda e: e["timestamp"])
    # deduplicate_log is the one legitimate caller that shrinks the
    # entry count, so it bypasses the shrink-guard explicitly.
    _save_log_unsafe(cleaned)

    return {"original": original_count, "cleaned": len(cleaned), "removed": original_count - len(cleaned)}


def _top_features(fi: dict, horizon: str) -> list:
    """Extract top 10 features for a horizon."""
    if not fi:
        return []
    h_fi = fi.get(horizon, fi.get("global", {}))
    if isinstance(h_fi, dict):
        sorted_feats = sorted(h_fi.items(), key=lambda x: abs(x[1]), reverse=True)
        return [{"name": n, "importance": round(float(v), 4)} for n, v in sorted_feats[:10]]
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESSIVE SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def score_all_intervals() -> dict:
    """
    Check all predictions across all scoring intervals.
    Returns summary of what was scored.
    """
    entries = _load_log()
    updated = False
    scored_count = 0
    errors = []

    for entry in entries:
        ts = datetime.fromisoformat(entry["timestamp"])
        symbol = entry["symbol"]
        days_elapsed = (datetime.now() - ts).days

        if days_elapsed < 1:
            continue  # Too fresh

        for horizon, hdata in entry["horizons"].items():
            intervals = SCORE_INTERVALS.get(horizon, [7, 30])
            current_price = hdata["current_price"]

            if current_price <= 0:
                continue

            # ── Backfill final_scored from existing scores ──
            # If the final interval was already scored but final_scored wasn't set
            # (e.g. from a previous version), fix it now.
            if not hdata.get("final_scored"):
                final_days = FINAL_INTERVALS.get(horizon, 30)
                final_key = f"{final_days}d"
                if final_key in hdata.get("scores", {}):
                    hdata["final_scored"] = True
                    hdata["final_correct"] = hdata["scores"][final_key]["direction_correct"]
                    updated = True

            for interval_days in intervals:
                interval_key = f"{interval_days}d"

                # Skip if already scored at this interval
                if interval_key in hdata["scores"]:
                    continue

                # Skip if not enough time has passed
                if days_elapsed < interval_days:
                    continue

                # Fetch actual price at this interval
                try:
                    score_date = ts + timedelta(days=interval_days)
                    actual_price = _fetch_price_at(symbol, score_date)

                    if actual_price is None or actual_price <= 0:
                        errors.append(f"{symbol}/{horizon}/{interval_key}: price fetch returned None for {score_date.strftime('%Y-%m-%d')}")
                        continue

                    actual_return = float(np.log(actual_price / current_price))  # log return for consistency with training targets
                    predicted_dir = hdata["direction"]
                    actual_dir = "up" if actual_return > 0 else "down"
                    direction_correct = (predicted_dir == actual_dir)

                    hdata["scores"][interval_key] = {
                        "actual_price":     round(actual_price, 2),
                        "actual_return":    round(actual_return, 4),
                        "direction_correct": direction_correct,
                        "scored_at":        datetime.now().isoformat(),
                        "days_actual":      interval_days,
                    }
                    scored_count += 1
                    updated = True

                    # Mark final score
                    final_days = FINAL_INTERVALS.get(horizon, 30)
                    if interval_days >= final_days:
                        hdata["final_scored"] = True
                        hdata["final_correct"] = direction_correct

                except Exception as e:
                    errors.append(f"{symbol}/{horizon}/{interval_key}: {str(e)}")

    if updated:
        _save_log(entries)
        # Update feature importance based on new scores
        _update_feature_importance_from_scores(entries)

    return {"scored": scored_count, "errors": len(errors), "error_details": errors}


def repair_missing_final_scores() -> dict:
    """
    Targeted repair — fills in the final-horizon checkpoint for any prediction
    whose scheduled expiration has passed but wasn't successfully scored at its
    final interval. Much faster than score_all_intervals() because it only
    touches matured predictions that are missing exactly one price point.

    Returns: {
        "repaired":  int  — number of predictions repaired
        "backfilled": int — number that had scores but missing final_scored flag
        "fetched":    int — number where we actually fetched a new price
        "failed":     int — number where fetch attempted but failed
        "failed_details": list of (symbol, horizon, date) strings
    }
    """
    entries = _load_log()
    now = datetime.now()
    repaired = 0
    backfilled = 0
    fetched = 0
    failed = []

    for entry in entries:
        symbol = entry.get("symbol")
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
        except Exception:
            continue
        days_elapsed = (now - ts).days

        for horizon, hdata in entry.get("horizons", {}).items():
            final_days = FINAL_INTERVALS.get(horizon)
            if not final_days:
                continue
            if days_elapsed < final_days:
                continue  # still in flight — nothing to repair
            if hdata.get("final_scored"):
                continue  # already good

            final_key = f"{final_days}d"
            scores = hdata.setdefault("scores", {})

            if final_key in scores:
                # Final interval already scored, just the flag is missing
                hdata["final_scored"] = True
                hdata["final_correct"] = scores[final_key]["direction_correct"]
                repaired += 1
                backfilled += 1
                continue

            # Final interval genuinely missing — try to fetch
            score_date = ts + timedelta(days=final_days)
            price = _fetch_price_at(symbol, score_date)
            entry_price = hdata.get("current_price", 0) or 0

            if price and price > 0 and entry_price > 0:
                actual_return = float(np.log(price / entry_price))
                predicted_dir = hdata.get("direction", "")
                actual_dir = "up" if actual_return > 0 else "down"
                direction_correct = (predicted_dir == actual_dir)
                scores[final_key] = {
                    "actual_price":      round(price, 2),
                    "actual_return":     round(actual_return, 4),
                    "direction_correct": direction_correct,
                    "scored_at":         now.isoformat(),
                    "days_actual":       final_days,
                    "repaired":          True,
                }
                hdata["final_scored"] = True
                hdata["final_correct"] = direction_correct
                repaired += 1
                fetched += 1
            else:
                failed.append(f"{symbol} {horizon} ({ts.strftime('%Y-%m-%d')})")

    if repaired > 0:
        _save_log(entries)

    return {
        "repaired":       repaired,
        "backfilled":     backfilled,
        "fetched":        fetched,
        "failed":         len(failed),
        "failed_details": failed,
    }


def quick_score_predictions() -> dict:
    """
    Quick-check: score 1-day results for all recent predictions.
    Much faster than waiting for full intervals.
    """
    entries = _load_log()
    updated = False
    scored = 0

    for entry in entries:
        ts = datetime.fromisoformat(entry["timestamp"])
        days_elapsed = (datetime.now() - ts).days
        symbol = entry["symbol"]

        if days_elapsed < 1:
            continue

        for horizon, hdata in entry["horizons"].items():
            if "1d" in hdata["scores"]:
                continue  # Already quick-scored

            current_price = hdata["current_price"]
            if current_price <= 0:
                continue

            try:
                score_date = ts + timedelta(days=1)
                actual_price = _fetch_price_at(symbol, score_date)

                if actual_price and actual_price > 0:
                    actual_return = float(np.log(actual_price / current_price))  # log return for consistency with training targets
                    predicted_dir = hdata["direction"]
                    actual_dir = "up" if actual_return > 0 else "down"

                    hdata["scores"]["1d"] = {
                        "actual_price":     round(actual_price, 2),
                        "actual_return":    round(actual_return, 4),
                        "direction_correct": (predicted_dir == actual_dir),
                        "scored_at":        datetime.now().isoformat(),
                        "days_actual":      1,
                        "is_quick_check":   True,
                    }
                    scored += 1
                    updated = True
            except Exception:
                pass

    if updated:
        _save_log(entries)

    return {"quick_scored": scored}


def _fetch_price_at(symbol: str, target_date: datetime) -> Optional[float]:
    """Fetch closing price closest to target_date."""
    try:
        tk = yf.Ticker(symbol)

        # Method 1: date-range query
        start = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")
        end   = (target_date + timedelta(days=7)).strftime("%Y-%m-%d")
        hist  = tk.history(start=start, end=end)

        # Method 2: fallback to period-based if date range returned empty
        if hist.empty:
            hist = tk.history(period="1mo")

        if hist.empty:
            return None

        # Find closest date to target
        target_ts = pd.Timestamp(target_date).tz_localize(None)
        hist.index = hist.index.tz_localize(None)
        idx = hist.index.get_indexer([target_ts], method="nearest")[0]
        if idx < 0 or idx >= len(hist):
            return None
        return float(hist["Close"].iloc[idx])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST SYNTHETIC SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_accuracy(
    symbol: str,
    df: pd.DataFrame,
    predictions: dict,
    n_samples: int = 20,
) -> dict:
    """
    Use historical data to simulate past predictions and score them.
    Returns synthetic accuracy scores per horizon.

    This gives INSTANT accuracy feedback instead of waiting weeks.

    Algorithm:
      For each of n_samples random historical points:
        1. Take price at that point
        2. Compare with actual price N days later
        3. Check if model's current predicted direction matches
    """
    results = {}
    close = df["Close"].values
    dates = df.index

    if len(close) < 100:
        return {}

    for horizon, hdata in predictions.items():
        horizon_days = {"1 Week": 5, "1 Month": 21, "1 Quarter": 63, "1 Year": 252}.get(horizon, 21)
        pred_direction = hdata.get("predicted_return", 0) > 0  # True = up

        # Sample random historical points (leave room for horizon)
        max_start = len(close) - horizon_days - 1
        if max_start < 50:
            continue

        sample_indices = np.random.choice(range(50, max_start), size=min(n_samples, max_start - 50), replace=False)

        correct = 0
        total = 0
        details = []

        for idx in sample_indices:
            start_price = close[idx]
            end_price = close[idx + horizon_days]
            actual_up = end_price > start_price
            actual_return = (end_price - start_price) / start_price

            # Does the model's current bias match what actually happened?
            was_correct = (pred_direction == actual_up)
            correct += int(was_correct)
            total += 1

            details.append({
                "date": str(dates[idx].date()) if hasattr(dates[idx], 'date') else str(dates[idx])[:10],
                "start_price": round(float(start_price), 2),
                "end_price": round(float(end_price), 2),
                "actual_return": round(float(actual_return) * 100, 2),
                "correct": was_correct,
            })

        if total > 0:
            results[horizon] = {
                "accuracy": round(correct / total * 100, 1),
                "correct": correct,
                "total": total,
                "details": sorted(details, key=lambda x: x["date"], reverse=True),
            }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

def _update_feature_importance_from_scores(entries: list):
    """
    Update cumulative feature importance based on scored predictions.
    Features in correct predictions get boosted; wrong ones get penalized.
    """
    imp_data = _load_importance()
    features = imp_data.get("features", {})

    for entry in entries:
        for horizon, hdata in entry["horizons"].items():
            top_feats = hdata.get("top_features", [])
            if not top_feats:
                continue

            # Check most recent score
            scores = hdata.get("scores", {})
            if not scores:
                continue

            # Use the latest score
            latest_key = max(scores.keys(), key=lambda k: int(k.replace("d", "")))
            latest_score = scores[latest_key]
            was_correct = latest_score.get("direction_correct", None)

            if was_correct is None:
                continue

            # Update feature importance
            for feat in top_feats:
                name = feat["name"]
                orig_importance = feat["importance"]

                if name not in features:
                    features[name] = {
                        "cumulative_score": 0,
                        "correct_count": 0,
                        "wrong_count": 0,
                        "total_appearances": 0,
                    }

                f = features[name]
                f["total_appearances"] += 1

                if was_correct:
                    f["cumulative_score"] += orig_importance * 0.1
                    f["correct_count"] += 1
                else:
                    f["cumulative_score"] -= orig_importance * 0.05
                    f["wrong_count"] += 1

    imp_data["features"] = features
    imp_data["update_count"] = imp_data.get("update_count", 0) + 1
    imp_data["last_updated"] = datetime.now().isoformat()
    _save_importance(imp_data)


def get_feature_importance_ranking() -> list:
    """Return features ranked by cumulative importance score."""
    imp_data = _load_importance()
    features = imp_data.get("features", {})

    ranked = []
    for name, data in features.items():
        total = data["total_appearances"]
        if total == 0:
            continue
        accuracy = data["correct_count"] / total * 100
        ranked.append({
            "feature": name,
            "cumulative_score": round(data["cumulative_score"], 4),
            "accuracy_when_top": round(accuracy, 1),
            "appearances": total,
            "correct": data["correct_count"],
            "wrong": data["wrong_count"],
        })

    return sorted(ranked, key=lambda x: x["cumulative_score"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FULL ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def get_full_analytics() -> dict:
    """
    Comprehensive analytics for the prediction dashboard.
    """
    entries = _load_log()

    analytics = {
        "total_predictions": 0,
        "total_analyses": len(entries),
        "scored_final": 0,
        "scored_quick": 0,
        "scored_any": 0,
        "direction_correct_final": 0,
        "direction_correct_quick": 0,
        "direction_correct_any": 0,
        "live_accuracy": 0.0,
        "quick_accuracy": 0.0,
        "per_horizon": {},
        "per_symbol": {},
        "confidence_calibration": [],
        "accuracy_over_time": [],
        "predictions_table": [],
        "model_versions": {},
        "regime_accuracy": {},
    }

    all_scored = []

    for entry in entries:
        symbol = entry["symbol"]
        ts = entry["timestamp"][:10]
        model_ver = entry.get("model_version", "1.0")
        regime = entry.get("regime", "Unknown")

        for horizon, hdata in entry["horizons"].items():
            analytics["total_predictions"] += 1

            scores = hdata.get("scores", {})
            confidence = hdata.get("confidence", 50)
            predicted_return = hdata.get("predicted_return", 0)

            # Build table row
            row = {
                "prediction_id": entry.get("prediction_id", ""),
                "symbol": symbol,
                "date": ts,
                "horizon": horizon,
                "predicted_return": round(predicted_return * 100, 2),
                "predicted_price": hdata.get("predicted_price", 0),
                "current_price": hdata.get("current_price", 0),
                "confidence": round(confidence, 1),
                "direction": hdata.get("direction", ""),
                "model_version": model_ver,
                "regime": regime,
                "top_features": [f["name"] for f in hdata.get("top_features", [])[:3]],
            }

            # Add score columns
            for interval in ["1d", "3d", "7d", "14d", "30d", "60d", "90d", "180d", "365d"]:
                if interval in scores:
                    s = scores[interval]
                    row[f"score_{interval}"] = "✓" if s["direction_correct"] else "✗"
                    row[f"price_{interval}"] = s["actual_price"]
                    row[f"return_{interval}"] = round(s["actual_return"] * 100, 2)
                else:
                    row[f"score_{interval}"] = "—"
                    row[f"price_{interval}"] = None
                    row[f"return_{interval}"] = None

            # Final score
            if hdata.get("final_scored"):
                row["final_result"] = "✓" if hdata["final_correct"] else "✗"
                analytics["scored_final"] += 1
                if hdata["final_correct"]:
                    analytics["direction_correct_final"] += 1
            else:
                row["final_result"] = "pending"

            # Quick score (1d)
            if "1d" in scores:
                analytics["scored_quick"] += 1
                if scores["1d"]["direction_correct"]:
                    analytics["direction_correct_quick"] += 1

            # Any score at all
            if scores:
                analytics["scored_any"] += 1
                # Use latest available score
                latest = max(scores.keys(), key=lambda k: int(k.replace("d", "")))
                if scores[latest]["direction_correct"]:
                    analytics["direction_correct_any"] += 1

                all_scored.append({
                    "date": ts,
                    "correct": scores[latest]["direction_correct"],
                    "confidence": confidence,
                    "horizon": horizon,
                    "symbol": symbol,
                    "regime": regime,
                    "model_version": model_ver,
                })

            analytics["predictions_table"].append(row)

            # ── Per-horizon stats ──
            if horizon not in analytics["per_horizon"]:
                analytics["per_horizon"][horizon] = {
                    "total": 0, "scored": 0, "correct": 0,
                    "accuracy": 0, "avg_confidence": [],
                    "quick_scored": 0, "quick_correct": 0,
                }
            ph = analytics["per_horizon"][horizon]
            ph["total"] += 1
            ph["avg_confidence"].append(confidence)

            if hdata.get("final_scored"):
                ph["scored"] += 1
                if hdata["final_correct"]:
                    ph["correct"] += 1

            if "1d" in scores:
                ph["quick_scored"] += 1
                if scores["1d"]["direction_correct"]:
                    ph["quick_correct"] += 1

            # ── Per-symbol stats ──
            if symbol not in analytics["per_symbol"]:
                analytics["per_symbol"][symbol] = {"total": 0, "scored": 0, "correct": 0}
            ps = analytics["per_symbol"][symbol]
            ps["total"] += 1
            if scores:
                latest = max(scores.keys(), key=lambda k: int(k.replace("d", "")))
                ps["scored"] += 1
                if scores[latest]["direction_correct"]:
                    ps["correct"] += 1

            # ── Per-regime stats ──
            if regime not in analytics["regime_accuracy"]:
                analytics["regime_accuracy"][regime] = {"total": 0, "scored": 0, "correct": 0}
            rg = analytics["regime_accuracy"][regime]
            rg["total"] += 1
            if scores:
                latest = max(scores.keys(), key=lambda k: int(k.replace("d", "")))
                rg["scored"] += 1
                if scores[latest]["direction_correct"]:
                    rg["correct"] += 1

    # ── Compute summary accuracy ──
    if analytics["scored_final"] > 0:
        analytics["live_accuracy"] = round(
            analytics["direction_correct_final"] / analytics["scored_final"] * 100, 1)
    if analytics["scored_quick"] > 0:
        analytics["quick_accuracy"] = round(
            analytics["direction_correct_quick"] / analytics["scored_quick"] * 100, 1)

    # ── Per-horizon accuracy ──
    for h, ph in analytics["per_horizon"].items():
        if ph["scored"] > 0:
            ph["accuracy"] = round(ph["correct"] / ph["scored"] * 100, 1)
        if ph["quick_scored"] > 0:
            ph["quick_accuracy"] = round(ph["quick_correct"] / ph["quick_scored"] * 100, 1)
        else:
            ph["quick_accuracy"] = 0
        ph["avg_confidence"] = round(np.mean(ph["avg_confidence"]), 1) if ph["avg_confidence"] else 0

    # ── Per-symbol accuracy ──
    for sym, ps in analytics["per_symbol"].items():
        if ps["scored"] > 0:
            ps["accuracy"] = round(ps["correct"] / ps["scored"] * 100, 1)
        else:
            ps["accuracy"] = 0

    # ── Per-regime accuracy ──
    for reg, rg in analytics["regime_accuracy"].items():
        if rg["scored"] > 0:
            rg["accuracy"] = round(rg["correct"] / rg["scored"] * 100, 1)
        else:
            rg["accuracy"] = 0

    # ── Confidence calibration data ──
    # Bin predictions by confidence level and check actual accuracy
    if all_scored:
        bins = [(30, 45), (45, 55), (55, 65), (65, 75), (75, 85), (85, 96)]
        for lo, hi in bins:
            in_bin = [s for s in all_scored if lo <= s["confidence"] < hi]
            if in_bin:
                correct_in_bin = sum(1 for s in in_bin if s["correct"])
                analytics["confidence_calibration"].append({
                    "confidence_range": f"{lo}-{hi}%",
                    "predicted_confidence": (lo + hi) / 2,
                    "actual_accuracy": round(correct_in_bin / len(in_bin) * 100, 1),
                    "count": len(in_bin),
                })

    # ── Accuracy over time (rolling 10-prediction window) ──
    if len(all_scored) >= 5:
        sorted_scored = sorted(all_scored, key=lambda x: x["date"])
        window = 10
        for i in range(window, len(sorted_scored) + 1):
            chunk = sorted_scored[max(0, i - window):i]
            acc = sum(1 for c in chunk if c["correct"]) / len(chunk) * 100
            analytics["accuracy_over_time"].append({
                "index": i,
                "date": chunk[-1]["date"],
                "rolling_accuracy": round(acc, 1),
                "window_size": len(chunk),
            })

    # ── Model version performance ──
    for entry_s in all_scored:
        ver = entry_s.get("model_version", "1.0")
        if ver not in analytics["model_versions"]:
            analytics["model_versions"][ver] = {"total": 0, "correct": 0}
        mv = analytics["model_versions"][ver]
        mv["total"] += 1
        if entry_s["correct"]:
            mv["correct"] += 1
    for ver, mv in analytics["model_versions"].items():
        if mv["total"] > 0:
            mv["accuracy"] = round(mv["correct"] / mv["total"] * 100, 1)

    # Sort table by date (newest first)
    analytics["predictions_table"].sort(key=lambda x: x["date"], reverse=True)

    return analytics


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL VERSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_current_model_version() -> str:
    """Return current model version string."""
    data = _load_model_versions()
    return data.get("current_version", "1.0")


def should_retrain() -> bool:
    """Check if model should be retrained (every 20 new scored predictions)."""
    data = _load_model_versions()
    analytics = get_full_analytics()

    scored = analytics["scored_any"]
    last_retrain_scored = data.get("last_retrain_scored", 0)

    return (scored - last_retrain_scored) >= 20


def record_retrain(accuracy_before: float, accuracy_after: float, changes: dict):
    """Record a model retrain event."""
    data = _load_model_versions()
    analytics = get_full_analytics()

    old_ver = data["current_version"]
    parts = old_ver.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0

    # Major version bump if accuracy improved > 3%, otherwise minor
    if accuracy_after - accuracy_before > 3:
        new_ver = f"{major + 1}.0"
    else:
        new_ver = f"{major}.{minor + 1}"

    data["versions"].append({
        "version": new_ver,
        "timestamp": datetime.now().isoformat(),
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "scored_predictions": analytics["scored_any"],
        "changes": changes,
    })
    data["current_version"] = new_ver
    data["retrain_count"] = data.get("retrain_count", 0) + 1
    data["last_retrain_scored"] = analytics["scored_any"]

    _save_model_versions(data)
    return new_ver


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_predictions_csv(path: Optional[str] = None) -> str:
    """Export all predictions to CSV."""
    analytics = get_full_analytics()
    table = analytics["predictions_table"]
    if not table:
        return ""

    df = pd.DataFrame(table)
    if path is None:
        path = os.path.join(LOG_DIR, "predictions_export.csv")
    df.to_csv(path, index=False)
    return path


def export_predictions_dataframe() -> pd.DataFrame:
    """Return predictions as a pandas DataFrame for display."""
    analytics = get_full_analytics()
    table = analytics["predictions_table"]
    if not table:
        return pd.DataFrame()
    return pd.DataFrame(table)


# ═══════════════════════════════════════════════════════════════════════════════
# MIGRATION: Import from v1 log
# ═══════════════════════════════════════════════════════════════════════════════

def migrate_from_v1():
    """Import predictions from the old prediction_log.json into v2 format."""
    old_file = os.path.join(LOG_DIR, "prediction_log.json")
    if not os.path.exists(old_file):
        return 0

    try:
        with open(old_file, "r") as f:
            old_entries = json.load(f)
    except Exception:
        return 0

    migrated = 0
    new_entries = _load_log()
    existing_ids = {e.get("prediction_id") for e in new_entries}

    for old in old_entries:
        # Generate a deterministic ID from old data
        pred_id = f"v1_{old['symbol']}_{old['timestamp'][:10]}"
        if pred_id in existing_ids:
            continue

        record = {
            "prediction_id":  pred_id,
            "symbol":         old["symbol"],
            "timestamp":      old["timestamp"],
            "model_version":  "1.0",
            "regime":         "Unknown",
            "horizons":       {},
        }

        for horizon, hdata in old.get("horizons", {}).items():
            record["horizons"][horizon] = {
                "predicted_return":  hdata.get("predicted_return", 0),
                "predicted_price":   hdata.get("predicted_price", 0),
                "current_price":     hdata.get("current_price", 0),
                "confidence":        hdata.get("confidence", 0),
                "ensemble_agreement": 0,
                "val_dir_accuracy":  0,
                "direction":         hdata.get("direction", "up"),
                "top_features":      [],
                "scores":            {},
                "final_scored":      hdata.get("scored", False),
                "final_correct":     hdata.get("direction_correct", None),
            }

            # Migrate old scored data
            if hdata.get("scored") and hdata.get("actual_price"):
                final_days = FINAL_INTERVALS.get(horizon, 30)
                record["horizons"][horizon]["scores"][f"{final_days}d"] = {
                    "actual_price": hdata["actual_price"],
                    "actual_return": hdata.get("actual_return", 0),
                    "direction_correct": hdata.get("direction_correct", False),
                    "scored_at": old["timestamp"],
                    "days_actual": final_days,
                }

        new_entries.append(record)
        migrated += 1

    if migrated > 0:
        _save_log(new_entries)

    return migrated
