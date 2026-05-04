"""
target_hit_analyzer.py
───────────────────────
For each prediction, determine whether the predicted price target was reached
at ANY point during the horizon using full daily OHLC data (not just checkpoint
close prices).

A BUY/LONG prediction hits target when the daily HIGH reaches/exceeds target.
A SELL/SHORT prediction hits target when the daily LOW reaches/crosses target.

Also computes Peak Favorable Move — the furthest the stock moved in the
predicted direction across the horizon (as a % from entry).

Batch-fetches yfinance data by symbol and caches in-process so repeated
analyses across a single API request are instant. (The legacy version used
Streamlit's @st.cache_data, but Streamlit was deleted from the repo when
the FastAPI rewrite landed; this module is now imported by api/main.py and
db.py only, so a plain functools.lru_cache is sufficient.)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf


# Horizon name → days
HORIZON_DAYS: Dict[str, int] = {
    "3 Day":     3,
    "1 Week":    7,
    "1 Month":   30,
    "1 Quarter": 90,
    "1 Year":    365,
}


# Cache size: 256 distinct (symbol, start, end) windows is plenty for one
# request's worth of analysis. The cache lives for the life of the worker
# process; under uvicorn's default single-worker setup this matches the
# old Streamlit ttl=3600 effectively (workers get recycled long before
# 1 hour of idle), with the bonus that we don't depend on Streamlit's
# session machinery.
@lru_cache(maxsize=256)
def _fetch_daily_history(
    symbol: str, start_str: str, end_str: str
) -> Optional[pd.DataFrame]:
    """
    Cached per-symbol daily OHLC fetch. Returns tz-naive DataFrame indexed by
    date with at minimum High / Low / Close columns.
    """
    try:
        tkr = yf.Ticker(symbol)
        hist = tkr.history(start=start_str, end=end_str, auto_adjust=False)
        if hist is None or hist.empty:
            return None
        # Normalize tz
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        return hist
    except Exception:
        return None


def _empty_result(horizon_days: int, pred_dt: datetime,
                  now: datetime) -> Dict[str, Any]:
    """When we can't analyze (no data / bad input), return neutral None values."""
    expiration = pred_dt + timedelta(days=horizon_days)
    return {
        "target_hit":               None,
        "day_target_hit":           None,
        "target_hit_date":          None,
        "target_hit_price":         None,
        "peak_favorable_move_pct":  None,
        "peak_fav_day":             None,
        "peak_fav_price":           None,
        "horizon_expired":          expiration <= now,
    }


def analyze_prediction_target(
    prediction_date: str,
    entry_price: float,
    target_price: float,
    horizon_days: int,
    daily_history: Optional[pd.DataFrame],
    predicted_return: float = 0.0,
    as_of: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Analyze a single prediction against daily OHLC history.

    Returns a dict with:
        target_hit               : True / False / None (None = unknown)
        day_target_hit           : int (days from prediction to first hit)
        target_hit_date          : str YYYY-MM-DD
        target_hit_price         : float (the High/Low that first crossed target)
        peak_favorable_move_pct  : float  (max % move in predicted direction)
        peak_fav_day             : int (days from prediction to peak)
        peak_fav_price           : float (the High/Low at peak)
        horizon_expired          : bool (has the scheduled horizon passed?)
    """
    if as_of is None:
        as_of = datetime.now()

    try:
        pred_dt = datetime.strptime(prediction_date, "%Y-%m-%d")
    except Exception:
        return _empty_result(horizon_days, as_of, as_of)

    if daily_history is None or daily_history.empty:
        return _empty_result(horizon_days, pred_dt, as_of)

    if not entry_price or not target_price or horizon_days <= 0:
        return _empty_result(horizon_days, pred_dt, as_of)

    expiration = pred_dt + timedelta(days=horizon_days)
    # We can only scan up to min(expiration, today). Beyond that we don't
    # know what the stock did.
    end_dt = min(expiration, as_of)

    # Slice the history to the relevant window
    try:
        mask = (daily_history.index >= pred_dt) & (daily_history.index <= end_dt)
        window = daily_history[mask]
    except Exception:
        return _empty_result(horizon_days, pred_dt, as_of)

    if len(window) == 0:
        return _empty_result(horizon_days, pred_dt, as_of)

    # Determine direction — use predicted_return sign if present, fall back to
    # target vs entry. A LONG call expects price to go UP.
    if predicted_return != 0:
        is_long = predicted_return > 0
    else:
        is_long = target_price >= entry_price

    if is_long:
        # Target hit when daily HIGH crosses target
        hit_series = window["High"] >= target_price
        peak_price = float(window["High"].max())
        peak_day_idx = window["High"].idxmax()
        peak_move_pct = (peak_price - entry_price) / entry_price * 100.0
    else:
        # SHORT: target hit when daily LOW crosses target
        hit_series = window["Low"] <= target_price
        peak_price = float(window["Low"].min())
        peak_day_idx = window["Low"].idxmin()
        # For shorts, "favorable" means price went DOWN — we express as a
        # positive % that represents gain-if-shorted
        peak_move_pct = (entry_price - peak_price) / entry_price * 100.0

    # First hit day
    if bool(hit_series.any()):
        first_hit_idx = hit_series[hit_series].index[0]
        target_hit = True
        target_hit_date = first_hit_idx.strftime("%Y-%m-%d")
        day_target_hit = int((first_hit_idx - pred_dt).days)
        # The exact price that crossed it
        if is_long:
            target_hit_price = float(window.loc[first_hit_idx, "High"])
        else:
            target_hit_price = float(window.loc[first_hit_idx, "Low"])
    else:
        target_hit = False
        target_hit_date = None
        day_target_hit = None
        target_hit_price = None

    peak_day = int((peak_day_idx - pred_dt).days)

    return {
        "target_hit":               target_hit,
        "day_target_hit":           day_target_hit,
        "target_hit_date":          target_hit_date,
        "target_hit_price":         target_hit_price,
        "peak_favorable_move_pct":  round(peak_move_pct, 2),
        "peak_fav_day":             peak_day,
        "peak_fav_price":           peak_price,
        "horizon_expired":          expiration <= as_of,
    }


def enrich_predictions_with_target_hit(
    predictions_table: List[Dict[str, Any]],
    horizon_days_map: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Mutates (and returns) predictions_table, adding target_hit / peak-favorable
    fields to every row.

    Batch-fetches daily OHLC history per symbol using the widest date range
    needed, then slices per-prediction in memory.
    """
    if not predictions_table:
        return predictions_table

    hz_map = horizon_days_map or HORIZON_DAYS
    now = datetime.now()

    # Group by symbol
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for pred in predictions_table:
        sym = pred.get("symbol", "")
        if sym:
            by_symbol.setdefault(sym, []).append(pred)

    for sym, preds in by_symbol.items():
        # Widest date window — earliest prediction through today (or earliest
        # expiration date still in the future, whichever is later).
        try:
            earliest_str = min(p.get("date", "") for p in preds if p.get("date"))
        except ValueError:
            continue
        if not earliest_str:
            continue

        # End: today + buffer (to cover same-day fetches)
        end_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        hist = _fetch_daily_history(sym, earliest_str, end_str)

        for pred in preds:
            horizon_name = pred.get("horizon", "")
            horizon_days = hz_map.get(horizon_name, 0)

            result = analyze_prediction_target(
                prediction_date=pred.get("date", ""),
                entry_price=pred.get("current_price", 0) or 0,
                target_price=pred.get("predicted_price", 0) or 0,
                horizon_days=horizon_days,
                daily_history=hist,
                predicted_return=pred.get("predicted_return", 0) or 0,
                as_of=now,
            )
            pred.update(result)

    return predictions_table


def compute_target_hit_aggregates(
    predictions_table: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Roll up per-prediction target hit data into scoreboard-level metrics.

    A prediction contributes to the aggregate only when its answer is
    definitive — either target was hit, OR the horizon has expired without
    a hit. Pending-and-not-yet-hit predictions are excluded.
    """
    hit_count = 0
    definitive_count = 0

    for p in predictions_table:
        th = p.get("target_hit")
        if th is True:
            hit_count += 1
            definitive_count += 1
        elif th is False and p.get("horizon_expired"):
            definitive_count += 1
        # else: None or still-maturing — not definitive

    rate = (hit_count / definitive_count * 100.0) if definitive_count > 0 else 0.0

    # Per-horizon rollup
    per_horizon: Dict[str, Dict[str, Any]] = {}
    for p in predictions_table:
        hz = p.get("horizon", "")
        d = per_horizon.setdefault(
            hz, {"hit": 0, "definitive": 0, "total": 0}
        )
        d["total"] += 1
        th = p.get("target_hit")
        if th is True:
            d["hit"] += 1
            d["definitive"] += 1
        elif th is False and p.get("horizon_expired"):
            d["definitive"] += 1

    for hz, d in per_horizon.items():
        d["rate"] = (d["hit"] / d["definitive"] * 100.0) if d["definitive"] > 0 else 0.0

    return {
        "target_hit_count":      hit_count,
        "target_definitive":     definitive_count,
        "target_hit_rate":       round(rate, 1),
        "target_per_horizon":    per_horizon,
    }
