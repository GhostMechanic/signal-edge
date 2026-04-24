"""
learning_engine.py  –  Self-Improving Prediction Intelligence
──────────────────────────────────────────────────────────────
Continuous learning system that analyzes scored predictions and feeds
corrections back into the model. Four subsystems:

1. Feedback Loop: Tracks error patterns per stock/regime/horizon and
   generates correction factors for the next prediction cycle.

2. Per-Stock Adaptive Confidence: Rolling accuracy per symbol →
   multiplier that scales confidence up/down based on track record.

3. Regime-Conditional Weighting: Tracks which L1 models perform best
   in each market regime → adjusts L2 weights at prediction time.

4. Quality Gate: Suppresses predictions where the model has no edge
   (low accuracy + high disagreement → "No Signal").

All state persisted to .predictions/learning_state.json.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# ─── Paths ──────────────────────────────────────────────────────────────────
STATE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".predictions")
STATE_FILE = os.path.join(STATE_DIR, "learning_state.json")
LOG_FILE   = os.path.join(STATE_DIR, "prediction_log_v2.json")

# ─── Constants ──────────────────────────────────────────────────────────────
MIN_SCORED_FOR_ADJUSTMENT = 5   # need this many scored predictions before adjusting
QUALITY_GATE_MIN_ACC = 0.52     # below this accuracy for stock+regime, suppress signal
QUALITY_GATE_MIN_AGREEMENT = 0.55  # ensemble agreement threshold
ROLLING_WINDOW = 20             # rolling window for per-stock accuracy
CONFIDENCE_FLOOR = 25.0         # never adjust confidence below this
CONFIDENCE_CEILING = 95.0       # never adjust above this


# ═══════════════════════════════════════════════════════════════════════════════
# STATE I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _load_state() -> dict:
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return _default_state()


def _default_state() -> dict:
    return {
        "per_stock": {},          # symbol → {accuracy, n_scored, corrections}
        "per_regime": {},         # regime → {accuracy, model_weights, n_scored}
        "per_stock_regime": {},   # "SYMBOL|regime" → {accuracy, n_scored}
        "per_horizon": {},        # horizon → {accuracy, bias, n_scored}
        "model_performance": {},  # regime → {model_name → accuracy}
        "last_updated": None,
        "total_analyzed": 0,
    }


def _save_state(state: dict):
    os.makedirs(STATE_DIR, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _load_predictions() -> list:
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONTINUOUS LEARNING FEEDBACK LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_scored_predictions() -> dict:
    """
    Analyze all scored predictions and build correction tables.
    Called after scoring runs. Returns the updated learning state.
    """
    predictions = _load_predictions()
    state = _load_state()

    # Reset counters
    stock_stats = {}      # symbol → list of (correct, confidence, horizon, regime)
    regime_stats = {}     # regime → list of (correct, confidence, horizon)
    sr_stats = {}         # "SYMBOL|regime" → list of (correct,)
    horizon_stats = {}    # horizon → list of (correct, predicted_return, actual_direction)

    for entry in predictions:
        symbol = entry.get("symbol", "")
        regime = entry.get("regime", "Unknown")
        horizons = entry.get("horizons", {})

        for horizon, hdata in horizons.items():
            # Check if this prediction has been scored
            if not hdata.get("final_scored") and not hdata.get("scores"):
                continue

            # Get the best available score
            correct = None
            pred_return = hdata.get("predicted_return", 0)
            confidence = hdata.get("confidence", 50)
            agreement = hdata.get("ensemble_agreement", 0.5)

            # Check final score first
            if hdata.get("final_scored"):
                correct = hdata.get("final_correct", None)
            else:
                # Use the latest interval score
                scores = hdata.get("scores", {})
                for interval in ["30d", "14d", "7d", "3d", "1d"]:
                    if interval in scores:
                        correct = scores[interval].get("direction_correct")
                        break

            if correct is None:
                continue

            correct_bool = bool(correct)

            # Accumulate stats
            if symbol not in stock_stats:
                stock_stats[symbol] = []
            stock_stats[symbol].append({
                "correct": correct_bool,
                "confidence": confidence,
                "horizon": horizon,
                "regime": regime,
                "agreement": agreement,
                "predicted_return": pred_return,
            })

            if regime not in regime_stats:
                regime_stats[regime] = []
            regime_stats[regime].append({
                "correct": correct_bool,
                "confidence": confidence,
                "horizon": horizon,
            })

            sr_key = f"{symbol}|{regime}"
            if sr_key not in sr_stats:
                sr_stats[sr_key] = []
            sr_stats[sr_key].append(correct_bool)

            if horizon not in horizon_stats:
                horizon_stats[horizon] = []
            horizon_stats[horizon].append({
                "correct": correct_bool,
                "predicted_return": pred_return,
                "confidence": confidence,
            })

    # ── Build per-stock accuracy and corrections ──────────────────────────
    for symbol, stats_list in stock_stats.items():
        n = len(stats_list)
        recent = stats_list[-ROLLING_WINDOW:]  # rolling window
        accuracy = sum(1 for s in recent if s["correct"]) / len(recent) if recent else 0.5
        all_acc = sum(1 for s in stats_list if s["correct"]) / n if n else 0.5

        # Confidence bias: are we over- or under-confident for this stock?
        avg_confidence = np.mean([s["confidence"] for s in stats_list])
        actual_accuracy_pct = all_acc * 100
        conf_bias = actual_accuracy_pct - avg_confidence  # positive = under-confident

        # Direction bias: does the model lean bullish or bearish incorrectly?
        bull_preds = [s for s in stats_list if s["predicted_return"] > 0]
        bear_preds = [s for s in stats_list if s["predicted_return"] <= 0]
        bull_acc = sum(1 for s in bull_preds if s["correct"]) / len(bull_preds) if bull_preds else 0.5
        bear_acc = sum(1 for s in bear_preds if s["correct"]) / len(bear_preds) if bear_preds else 0.5

        state["per_stock"][symbol] = {
            "rolling_accuracy": round(accuracy, 4),
            "all_time_accuracy": round(all_acc, 4),
            "n_scored": n,
            "confidence_bias": round(conf_bias, 2),
            "bull_accuracy": round(bull_acc, 4),
            "bear_accuracy": round(bear_acc, 4),
            # Confidence multiplier: scale confidence by how accurate we've been
            # 60% accuracy → 1.1x multiplier, 40% accuracy → 0.8x
            "confidence_multiplier": round(
                np.clip(0.5 + accuracy, 0.70, 1.30), 3
            ),
        }

    # ── Build per-regime stats ────────────────────────────────────────────
    for regime, stats_list in regime_stats.items():
        n = len(stats_list)
        accuracy = sum(1 for s in stats_list if s["correct"]) / n if n else 0.5
        state["per_regime"][regime] = {
            "accuracy": round(accuracy, 4),
            "n_scored": n,
        }

    # ── Build stock+regime combo stats (for quality gate) ─────────────────
    for sr_key, correct_list in sr_stats.items():
        n = len(correct_list)
        accuracy = sum(correct_list) / n if n else 0.5
        state["per_stock_regime"][sr_key] = {
            "accuracy": round(accuracy, 4),
            "n_scored": n,
        }

    # ── Build per-horizon stats ───────────────────────────────────────────
    for horizon, stats_list in horizon_stats.items():
        n = len(stats_list)
        accuracy = sum(1 for s in stats_list if s["correct"]) / n if n else 0.5

        # Magnitude bias: does the model over/under-predict returns?
        avg_pred = np.mean([abs(s["predicted_return"]) for s in stats_list])
        correct_preds = [s for s in stats_list if s["correct"]]
        wrong_preds = [s for s in stats_list if not s["correct"]]

        # When wrong, how much were we off?
        state["per_horizon"][horizon] = {
            "accuracy": round(accuracy, 4),
            "n_scored": n,
            "avg_predicted_magnitude": round(avg_pred, 4),
            "avg_confidence_when_correct": round(
                np.mean([s["confidence"] for s in correct_preds]), 2
            ) if correct_preds else 50,
            "avg_confidence_when_wrong": round(
                np.mean([s["confidence"] for s in wrong_preds]), 2
            ) if wrong_preds else 50,
        }

    state["total_analyzed"] = sum(len(v) for v in stock_stats.values())
    _save_state(state)
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PER-STOCK ADAPTIVE CONFIDENCE
# ═══════════════════════════════════════════════════════════════════════════════

def get_stock_confidence_multiplier(symbol: str) -> float:
    """
    Returns a multiplier (0.7 - 1.3) based on the model's historical
    accuracy for this specific stock. 1.0 = no adjustment.
    """
    state = _load_state()
    stock_data = state.get("per_stock", {}).get(symbol.upper())
    if not stock_data or stock_data.get("n_scored", 0) < MIN_SCORED_FOR_ADJUSTMENT:
        return 1.0  # not enough data yet
    return stock_data.get("confidence_multiplier", 1.0)


def get_stock_confidence_bias(symbol: str) -> float:
    """
    Returns a confidence bias correction in percentage points.
    Positive = model is under-confident, negative = over-confident.
    """
    state = _load_state()
    stock_data = state.get("per_stock", {}).get(symbol.upper())
    if not stock_data or stock_data.get("n_scored", 0) < MIN_SCORED_FOR_ADJUSTMENT:
        return 0.0
    return stock_data.get("confidence_bias", 0.0)


def get_direction_bias(symbol: str) -> dict:
    """
    Returns bull/bear accuracy for this stock so the model can
    adjust confidence for bullish vs bearish predictions.
    """
    state = _load_state()
    stock_data = state.get("per_stock", {}).get(symbol.upper())
    if not stock_data or stock_data.get("n_scored", 0) < MIN_SCORED_FOR_ADJUSTMENT:
        return {"bull_accuracy": 0.5, "bear_accuracy": 0.5}
    return {
        "bull_accuracy": stock_data.get("bull_accuracy", 0.5),
        "bear_accuracy": stock_data.get("bear_accuracy", 0.5),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. REGIME-CONDITIONAL MODEL WEIGHTING
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_regime_model_performance() -> dict:
    """
    Analyze which L1 models perform best in each regime.
    Looks at predictions where we recorded ensemble_agreement
    and model-level accuracy data.

    Returns dict of regime → weight adjustments.
    """
    state = _load_state()
    regime_data = state.get("per_regime", {})

    adjustments = {}
    for regime, rdata in regime_data.items():
        acc = rdata.get("accuracy", 0.5)
        n = rdata.get("n_scored", 0)
        if n < MIN_SCORED_FOR_ADJUSTMENT:
            adjustments[regime] = {"confidence_scale": 1.0, "n_scored": n}
            continue

        # Scale confidence based on regime accuracy
        # Bull regimes with 65% accuracy → 1.15x
        # Bear regimes with 40% accuracy → 0.85x
        scale = np.clip(0.5 + acc, 0.75, 1.25)
        adjustments[regime] = {
            "confidence_scale": round(float(scale), 3),
            "accuracy": round(acc, 4),
            "n_scored": n,
        }

    state["regime_weights"] = adjustments
    _save_state(state)
    return adjustments


def get_regime_confidence_scale(regime: str) -> float:
    """
    Returns a confidence scaling factor for the given regime.
    Based on historical accuracy in that regime.
    """
    state = _load_state()
    weights = state.get("regime_weights", {})
    if regime in weights:
        return weights[regime].get("confidence_scale", 1.0)
    return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PREDICTION QUALITY GATE
# ═══════════════════════════════════════════════════════════════════════════════

def should_suppress_prediction(
    symbol: str,
    regime: str,
    horizon: str,
    confidence: float,
    ensemble_agreement: float,
) -> Tuple[bool, str]:
    """
    Decides whether to suppress a prediction because the model has no edge.

    Returns (should_suppress, reason).
    """
    state = _load_state()

    # Check stock+regime combo accuracy
    sr_key = f"{symbol.upper()}|{regime}"
    sr_data = state.get("per_stock_regime", {}).get(sr_key)

    # Check per-stock accuracy
    stock_data = state.get("per_stock", {}).get(symbol.upper())

    # Check per-horizon accuracy
    hz_data = state.get("per_horizon", {}).get(horizon)

    # Condition 1: Stock+regime combo has poor accuracy with enough data
    if sr_data and sr_data.get("n_scored", 0) >= MIN_SCORED_FOR_ADJUSTMENT:
        if sr_data["accuracy"] < QUALITY_GATE_MIN_ACC:
            return True, (
                f"Model accuracy for {symbol} in {regime} regime is "
                f"{sr_data['accuracy']*100:.0f}% ({sr_data['n_scored']} predictions) — "
                f"below {QUALITY_GATE_MIN_ACC*100:.0f}% threshold"
            )

    # Condition 2: Stock has poor accuracy AND high ensemble disagreement
    if stock_data and stock_data.get("n_scored", 0) >= MIN_SCORED_FOR_ADJUSTMENT:
        stock_acc = stock_data.get("rolling_accuracy", 0.5)
        if stock_acc < QUALITY_GATE_MIN_ACC and ensemble_agreement < QUALITY_GATE_MIN_AGREEMENT:
            return True, (
                f"Low accuracy ({stock_acc*100:.0f}%) + high disagreement "
                f"({ensemble_agreement*100:.0f}%) for {symbol}"
            )

    # Condition 3: Very low confidence + poor horizon accuracy
    if hz_data and hz_data.get("n_scored", 0) >= MIN_SCORED_FOR_ADJUSTMENT:
        if hz_data["accuracy"] < 0.48 and confidence < 40:
            return True, (
                f"{horizon} horizon has {hz_data['accuracy']*100:.0f}% accuracy "
                f"and confidence is only {confidence:.0f}%"
            )

    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER: APPLY ALL LEARNING TO A PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def apply_learning(
    symbol: str,
    regime: str,
    horizon: str,
    raw_confidence: float,
    predicted_return: float,
    ensemble_agreement: float,
) -> dict:
    """
    Apply all four learning systems to a single prediction.
    Returns dict with:
      - adjusted_confidence: final confidence after all adjustments
      - suppress: whether to suppress this prediction
      - suppress_reason: why it was suppressed
      - adjustments: breakdown of what was applied
    """
    confidence = raw_confidence

    # 1. Per-stock adaptive confidence
    stock_mult = get_stock_confidence_multiplier(symbol)
    confidence *= stock_mult

    # 2. Direction bias: if model is bad at bullish predictions for this stock,
    #    reduce confidence on bullish predictions (and vice versa)
    dir_bias = get_direction_bias(symbol)
    if predicted_return > 0:
        dir_acc = dir_bias["bull_accuracy"]
    else:
        dir_acc = dir_bias["bear_accuracy"]
    # Only apply if we have enough data (n_scored check is inside get_direction_bias)
    if dir_acc != 0.5:  # 0.5 = default/no data
        dir_scale = np.clip(0.5 + dir_acc, 0.75, 1.20)
        confidence *= dir_scale

    # 3. Regime-conditional scaling
    regime_scale = get_regime_confidence_scale(regime)
    confidence *= regime_scale

    # 4. Per-stock confidence bias correction
    conf_bias = get_stock_confidence_bias(symbol)
    # Apply half the bias (conservative correction)
    confidence += conf_bias * 0.5

    # Clamp
    confidence = float(np.clip(confidence, CONFIDENCE_FLOOR, CONFIDENCE_CEILING))

    # 5. Quality gate
    suppress, reason = should_suppress_prediction(
        symbol, regime, horizon, confidence, ensemble_agreement
    )

    # Track suppression count
    if suppress:
        try:
            state = _load_state()
            state["predictions_suppressed"] = state.get("predictions_suppressed", 0) + 1
            _save_state(state)
        except Exception:
            pass

    return {
        "adjusted_confidence": round(confidence, 1),
        "suppress": suppress,
        "suppress_reason": reason,
        "adjustments": {
            "stock_multiplier": round(stock_mult, 3),
            "direction_accuracy": round(dir_acc, 3),
            "regime_scale": round(regime_scale, 3),
            "confidence_bias": round(conf_bias, 2),
            "raw_confidence": round(raw_confidence, 1),
        },
    }


def get_learning_summary() -> dict:
    """Return a summary of what the learning engine knows."""
    state = _load_state()
    per_stock = state.get("per_stock", {})
    total_scored = sum(d.get("n_scored", 0) for d in per_stock.values())
    multipliers = [d.get("confidence_multiplier", 1.0) for d in per_stock.values() if d.get("n_scored", 0) > 0]
    avg_adj = (np.mean(multipliers) - 1.0) * 100 if multipliers else 0  # as pct
    suppressed = state.get("predictions_suppressed", 0)
    return {
        "total_analyzed": state.get("total_analyzed", 0),
        "total_scored": total_scored,
        "stocks_tracked": len(per_stock),
        "regimes_tracked": len(state.get("per_regime", {})),
        "predictions_suppressed": suppressed,
        "avg_confidence_adjustment": round(avg_adj, 1),
        "last_updated": state.get("last_updated"),
        "per_stock": {
            sym: {
                "accuracy": d.get("rolling_accuracy", 0),
                "confidence_multiplier": d.get("confidence_multiplier", 1.0),
                "n_scored": d.get("n_scored", 0),
            }
            for sym, d in per_stock.items()
        },
        "per_regime": state.get("per_regime", {}),
        "per_horizon": state.get("per_horizon", {}),
    }
