"""
api/main.py
──────────────────────────────────────────────────────────────
FastAPI wrapper over the Prediqt Python stack.

Keeps ALL existing ML code untouched — this file just exposes
Python functions (get_full_analytics, get_quick_signal,
target_hit_analyzer, etc.) as HTTP endpoints so the Next.js
frontend at prediqt-web can fetch real data instead of the
hardcoded placeholders in the React components.

Run locally:
    source .venv/bin/activate
    uvicorn api.main:app --reload --port 8000

Then open:
    http://localhost:8000/docs                        → interactive API explorer
    http://localhost:8000/api/health
    http://localhost:8000/api/analytics/summary       → three honesty reads
    http://localhost:8000/api/analytics/per-horizon   → 5 horizon cards
    http://localhost:8000/api/analytics/simulated-portfolio
    http://localhost:8000/api/predictions?filter=scored&days=30&sort=newest&limit=20
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ─── App instance ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Prediqt API",
    description="HTTP wrapper over the Prediqt ML stack.",
    version="0.2.0",
)


# ─── CORS ────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://prediqt-web.vercel.app",
]
ALLOW_ORIGIN_REGEX = r"https://prediqt-web-[a-z0-9-]+\.vercel\.app"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Shared cache ────────────────────────────────────────────────────────────
# Enriching analytics (target-hit daily OHLC scan) is slow on a cold run.
# Every endpoint that touches analytics shares this cache so a single page
# load of Track Record (which hits 3–4 endpoints) only pays the cost once.
_ANALYTICS_CACHE: Dict[str, Any] = {}
_ANALYTICS_TTL_SECONDS = 300  # 5 minutes


def _get_enriched_analytics() -> Dict[str, Any]:
    """
    Returns the full analytics dict with target-hit enrichment applied,
    cached for 5 minutes. All analytics endpoints share this one source
    of truth so they never disagree with each other.
    """
    cached_data = _ANALYTICS_CACHE.get("data")
    cached_ts = _ANALYTICS_CACHE.get("timestamp")

    if cached_data is not None and cached_ts is not None:
        age = (datetime.utcnow() - cached_ts).total_seconds()
        if age < _ANALYTICS_TTL_SECONDS:
            return cached_data

    # Cache miss — recompute
    from prediction_logger_v2 import get_full_analytics
    from target_hit_analyzer import (
        enrich_predictions_with_target_hit,
        compute_target_hit_aggregates,
    )

    pa = get_full_analytics()
    try:
        pa["predictions_table"] = enrich_predictions_with_target_hit(
            pa.get("predictions_table", [])
        )
        pa.update(compute_target_hit_aggregates(pa["predictions_table"]))
    except Exception:
        pa.setdefault("target_hit_rate", 0)
        pa.setdefault("target_hit_count", 0)
        pa.setdefault("target_definitive", 0)
        pa.setdefault("target_per_horizon", {})

    _ANALYTICS_CACHE["data"] = pa
    _ANALYTICS_CACHE["timestamp"] = datetime.utcnow()
    return pa


# ─── Health check ────────────────────────────────────────────────────────────
@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Simple check that the API is alive. Used by the frontend + Render."""
    return {
        "status": "ok",
        "service": "prediqt-api",
        "version": "0.2.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ─── Analytics: summary ──────────────────────────────────────────────────────
@app.get("/api/analytics/summary")
def analytics_summary() -> Dict[str, Any]:
    """
    Top-line scoreboard: three honesty reads + sample sizes.
    Fuels the landing-page ScoreboardTeaser component.
    """
    from prediction_logger_v2 import get_current_model_version

    pa = _get_enriched_analytics()
    model_version = get_current_model_version()

    scored_any = pa.get("scored_any", 0)
    wins = pa.get("direction_correct_any", 0)
    checkpoint_win_rate = (wins / scored_any * 100.0) if scored_any > 0 else 0

    return {
        "model_version":      model_version,
        "total_predictions":  pa.get("total_predictions", 0),
        "scored_any":         scored_any,
        "scored_final":       pa.get("scored_final", 0),

        "target_hit_rate":    round(pa.get("target_hit_rate", 0), 1),
        "target_hit_count":   pa.get("target_hit_count", 0),
        "target_definitive":  pa.get("target_definitive", 0),

        "checkpoint_win_rate": round(checkpoint_win_rate, 1),
        "checkpoint_wins":     wins,
        "checkpoint_losses":   scored_any - wins,

        "expiration_win_rate": round(pa.get("live_accuracy", 0), 1),
        "expiration_matured":  pa.get("scored_final", 0),
    }


# ─── Analytics: per-horizon ──────────────────────────────────────────────────
HORIZON_ORDER = ["3 Day", "1 Week", "1 Month", "1 Quarter", "1 Year"]


@app.get("/api/analytics/per-horizon")
def analytics_per_horizon() -> Dict[str, Any]:
    """
    Breakdown of the three honesty reads across each prediction horizon.
    Fuels the 5 horizon cards on the Track Record page.
    """
    pa = _get_enriched_analytics()
    per_horizon = pa.get("per_horizon", {})
    target_per_horizon = pa.get("target_per_horizon", {})

    # Checkpoint win rate per horizon — walk preds_table, find the latest
    # scored interval per prediction, tally wins/losses per horizon.
    checkpoint_by_hz: Dict[str, Dict[str, int]] = {}
    for p in pa.get("predictions_table", []):
        hz = p.get("horizon", "")
        d = checkpoint_by_hz.setdefault(hz, {"scored": 0, "correct": 0})
        for interval in ["365d", "180d", "90d", "60d", "30d", "14d", "7d", "3d", "1d"]:
            s = p.get(f"score_{interval}")
            if s in ("✓", "✗"):
                d["scored"] += 1
                if s == "✓":
                    d["correct"] += 1
                break

    horizons = []
    for name in HORIZON_ORDER:
        h_data = per_horizon.get(name, {})
        t_data = target_per_horizon.get(name, {})
        c_data = checkpoint_by_hz.get(name, {"scored": 0, "correct": 0})

        total = h_data.get("total", 0)
        scored_final = h_data.get("scored", 0)
        correct_final = h_data.get("correct", 0)
        acc_final = h_data.get("accuracy", 0)

        c_scored = c_data["scored"]
        c_correct = c_data["correct"]
        c_rate = (c_correct / c_scored * 100) if c_scored > 0 else 0

        t_hit = t_data.get("hit", 0)
        t_def = t_data.get("definitive", 0)
        t_rate = t_data.get("rate", 0)

        horizons.append({
            "name":  name,
            "total": total,
            "target": {
                "rate":           round(t_rate, 1),
                "hit_count":      t_hit,
                "resolved_count": t_def,
                "has_data":       t_def > 0,
            },
            "checkpoint": {
                "rate":          round(c_rate, 1),
                "correct_count": c_correct,
                "scored_count":  c_scored,
                "has_data":      c_scored > 0,
            },
            "expiration": {
                "rate":          round(acc_final, 1),
                "correct_count": correct_final,
                "matured_count": scored_final,
                "has_data":      scored_final > 0,
            },
        })

    return {"horizons": horizons}


# ─── Analytics: simulated portfolio ──────────────────────────────────────────
_INTERVAL_DAYS = {
    "1d": 1, "3d": 3, "7d": 7, "14d": 14, "30d": 30,
    "60d": 60, "90d": 90, "180d": 180, "365d": 365,
}
_STARTING_CAPITAL = 10_000.0


def _build_curve_and_stats(trades):
    """
    Walk a list of trades (with exit_date + pnl_pct) and produce an equity
    curve + summary stats. Each trade is sized at 5% of CURRENT equity.
    """
    if not trades:
        return {
            "equity_curve":       [],
            "final_value":        _STARTING_CAPITAL,
            "total_return_pct":   0.0,
            "n_trades":           0,
            "wins":               0,
            "losses":             0,
            "max_drawdown_pct":   0.0,
            "targets_hit":        0,
        }

    equity = _STARTING_CAPITAL
    curve = [{"date": trades[0]["date"], "equity": round(equity, 2)}]
    wins = losses = targets_hit = 0
    running_max = equity
    max_dd = 0.0

    for t in trades:
        position = equity * 0.05
        pnl = position * t["pnl_pct"]
        equity += pnl
        curve.append({"date": t["date"], "equity": round(equity, 2)})
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
        running_max = max(running_max, equity)
        dd = ((equity - running_max) / running_max * 100) if running_max > 0 else 0
        max_dd = min(max_dd, dd)
        if t.get("realized") == "target":
            targets_hit += 1

    total_return = (equity / _STARTING_CAPITAL - 1) * 100
    return {
        "equity_curve":       curve,
        "final_value":        round(equity, 2),
        "total_return_pct":   round(total_return, 2),
        "n_trades":           len(trades),
        "wins":               wins,
        "losses":             losses,
        "max_drawdown_pct":   round(max_dd, 2),
        "targets_hit":        targets_hit,
    }


@app.get("/api/analytics/simulated-portfolio")
def simulated_portfolio() -> Dict[str, Any]:
    """
    Build TWO parallel equity curves from scored predictions:
      - hold:        every trade held to its scheduled checkpoint
      - take_profit: exit at target_hit_date when target was hit,
                     else fall back to hold behavior

    Used by the Track Record "If you followed every signal" chart.
    """
    pa = _get_enriched_analytics()
    scored_preds = [
        p for p in pa.get("predictions_table", [])
        if p.get("final_result") in ("✓", "✗")
    ]

    if len(scored_preds) < 3:
        return {
            "starting_capital": _STARTING_CAPITAL,
            "hold":             _build_curve_and_stats([]),
            "take_profit":      _build_curve_and_stats([]),
            "note":             "Not enough matured predictions yet — at least 3 required.",
        }

    trades_hold = []
    trades_tp = []

    for pred in scored_preds:
        actual_ret = None
        scored_interval = None
        for interval in ["365d", "180d", "90d", "60d", "30d", "14d", "7d", "3d", "1d"]:
            r = pred.get(f"return_{interval}")
            if r is not None:
                actual_ret = r / 100.0
                scored_interval = interval
                break
        if actual_ret is None:
            continue

        try:
            pred_date = datetime.strptime(pred["date"], "%Y-%m-%d")
        except Exception:
            continue
        exit_date = pred_date + timedelta(days=_INTERVAL_DAYS[scored_interval])

        pred_ret = pred.get("predicted_return", 0) or 0
        dir_sign = 1 if pred_ret >= 0 else -1
        hold_pnl_pct = actual_ret * dir_sign

        trades_hold.append({
            "date":    exit_date.strftime("%Y-%m-%d"),
            "pnl_pct": hold_pnl_pct,
        })

        # Take-profit: exit at target-hit date if target was hit, else hold
        th = pred.get("target_hit")
        th_date_str = pred.get("target_hit_date")
        if th is True and th_date_str:
            try:
                tp_exit = datetime.strptime(th_date_str, "%Y-%m-%d")
            except Exception:
                tp_exit = exit_date
            tp_pnl_pct = abs(pred_ret / 100.0)
            realized = "target"
        else:
            tp_exit = exit_date
            tp_pnl_pct = hold_pnl_pct
            realized = "expiration"

        trades_tp.append({
            "date":     tp_exit.strftime("%Y-%m-%d"),
            "pnl_pct":  tp_pnl_pct,
            "realized": realized,
        })

    trades_hold.sort(key=lambda t: t["date"])
    trades_tp.sort(key=lambda t: t["date"])

    return {
        "starting_capital": _STARTING_CAPITAL,
        "hold":             _build_curve_and_stats(trades_hold),
        "take_profit":      _build_curve_and_stats(trades_tp),
    }


# ─── Predictions log (paginated, filtered, sorted) ──────────────────────────
@app.get("/api/predictions")
def predictions_log(
    filter: str = "all",            # "all" | "scored" | "pending"
    days: Optional[int] = None,     # last N days (None = no time filter)
    sort: str = "newest",           # "newest" | "oldest"
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Paginated prediction log with status / time / sort filters.
    Each row has enough context for the Track Record expandable rows
    (target hit, peak favorable move, top features, per-interval scores).
    """
    pa = _get_enriched_analytics()
    preds = list(pa.get("predictions_table", []))

    # Status filter
    if filter == "scored":
        preds = [p for p in preds if p.get("final_result") in ("✓", "✗")]
    elif filter == "pending":
        preds = [p for p in preds if p.get("final_result") not in ("✓", "✗")]

    # Time-range filter (by prediction date)
    if days is not None and days > 0:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        preds = [p for p in preds if p.get("date", "") >= cutoff]

    # Sort
    preds.sort(key=lambda p: p.get("date", ""), reverse=(sort == "newest"))

    total_filtered = len(preds)
    page = preds[offset:offset + limit]

    # Shape each row for the frontend — only what the UI needs
    INTERVALS = ["1d", "3d", "7d", "14d", "30d", "60d", "90d", "180d", "365d"]
    output = []
    for p in page:
        scores = {}
        returns = {}
        for iv in INTERVALS:
            s = p.get(f"score_{iv}")
            r = p.get(f"return_{iv}")
            if s in ("✓", "✗"):
                scores[iv] = s
            if r is not None:
                returns[iv] = r
        output.append({
            "prediction_id":           p.get("prediction_id", ""),
            "symbol":                  p.get("symbol", ""),
            "date":                    p.get("date", ""),
            "horizon":                 p.get("horizon", ""),
            "predicted_return_pct":    p.get("predicted_return", 0),
            "predicted_price":         p.get("predicted_price", 0),
            "current_price":           p.get("current_price", 0),
            "confidence":              p.get("confidence", 0),
            "direction":               p.get("direction", ""),
            "model_version":           p.get("model_version", ""),
            "regime":                  p.get("regime", ""),
            "final_result":            p.get("final_result", "pending"),
            "top_features":            p.get("top_features", []),

            # Target-hit fields (from target_hit_analyzer)
            "target_hit":              p.get("target_hit"),
            "day_target_hit":          p.get("day_target_hit"),
            "target_hit_date":         p.get("target_hit_date"),
            "target_hit_price":        p.get("target_hit_price"),
            "peak_favorable_move_pct": p.get("peak_favorable_move_pct"),
            "peak_fav_day":            p.get("peak_fav_day"),
            "peak_fav_price":          p.get("peak_fav_price"),
            "horizon_expired":         p.get("horizon_expired", False),

            # Per-interval scores & returns
            "scores":  scores,
            "returns": returns,
        })

    return {
        "total":       pa.get("total_predictions", 0),
        "filtered":    total_filtered,
        "offset":      offset,
        "limit":       limit,
        "predictions": output,
    }
