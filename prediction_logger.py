"""
prediction_logger.py
--------------------
Logs ML predictions and scores them after 30 days against actual prices.
Stores predictions in a JSON file on disk for persistence across sessions.

Usage:
    log_prediction(symbol, predictions_dict)   # after each analysis
    get_track_record()                          # returns scored past predictions
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf

LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".predictions")
LOG_FILE = os.path.join(LOG_DIR, "prediction_log.json")

# How many days to wait before scoring a prediction per horizon
SCORE_DELAYS = {
    "1 Week":    7,
    "1 Month":   30,
    "1 Quarter": 90,
    "1 Year":    365,
}


def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _load_log() -> list:
    _ensure_dir()
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_log(entries: list):
    _ensure_dir()
    with open(LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def log_prediction(symbol: str, predictions: dict):
    """
    Save current predictions to the log.
    Each entry contains: symbol, timestamp, and per-horizon predictions.
    """
    entries = _load_log()

    record = {
        "symbol":    symbol.upper(),
        "timestamp": datetime.now().isoformat(),
        "horizons":  {},
    }

    for horizon, data in predictions.items():
        record["horizons"][horizon] = {
            "predicted_return":  data.get("predicted_return", 0),
            "predicted_price":   data.get("predicted_price", 0),
            "current_price":     data.get("current_price", 0),
            "confidence":        data.get("confidence", 0),
            "direction":         "up" if data.get("predicted_return", 0) > 0 else "down",
            "scored":            False,
            "actual_price":      None,
            "actual_return":     None,
            "direction_correct": None,
        }

    entries.append(record)

    # Keep last 500 entries max
    if len(entries) > 500:
        entries = entries[-500:]

    _save_log(entries)


def score_predictions() -> list:
    """
    Check all unscored predictions to see if enough time has passed.
    If so, fetch the actual price and score them.
    Returns list of all entries (scored and unscored).
    """
    entries = _load_log()
    updated = False

    for entry in entries:
        ts = datetime.fromisoformat(entry["timestamp"])
        symbol = entry["symbol"]

        for horizon, hdata in entry["horizons"].items():
            if hdata.get("scored", False):
                continue

            delay_days = SCORE_DELAYS.get(horizon, 30)
            score_date = ts + timedelta(days=delay_days)

            if datetime.now() < score_date:
                continue  # Not time yet

            # Time to score — fetch actual price at score_date
            try:
                tk = yf.Ticker(symbol)
                # Fetch data around the score date
                start = (score_date - timedelta(days=5)).strftime("%Y-%m-%d")
                end   = (score_date + timedelta(days=5)).strftime("%Y-%m-%d")
                hist  = tk.history(start=start, end=end)

                if not hist.empty:
                    actual_price = float(hist["Close"].iloc[-1])
                    current_price = hdata["current_price"]

                    if current_price > 0:
                        actual_return = (actual_price - current_price) / current_price
                        predicted_dir = hdata["direction"]
                        actual_dir    = "up" if actual_return > 0 else "down"

                        hdata["actual_price"]      = round(actual_price, 2)
                        hdata["actual_return"]     = round(actual_return, 4)
                        hdata["direction_correct"] = (predicted_dir == actual_dir)
                        hdata["scored"]            = True
                        updated = True
            except Exception:
                pass

    if updated:
        _save_log(entries)

    return entries


def get_track_record() -> dict:
    """
    Returns summary statistics of prediction accuracy.
    """
    entries = score_predictions()

    stats = {
        "total_predictions": 0,
        "scored_predictions": 0,
        "direction_correct": 0,
        "direction_accuracy": 0.0,
        "avg_predicted_return": 0.0,
        "avg_actual_return": 0.0,
        "per_horizon": {},
        "recent": [],  # last 10 scored predictions
    }

    all_scored = []

    for entry in entries:
        for horizon, hdata in entry["horizons"].items():
            stats["total_predictions"] += 1

            if not hdata.get("scored", False):
                continue

            stats["scored_predictions"] += 1
            if hdata.get("direction_correct"):
                stats["direction_correct"] += 1

            # Per-horizon stats
            if horizon not in stats["per_horizon"]:
                stats["per_horizon"][horizon] = {
                    "total": 0, "correct": 0, "accuracy": 0.0,
                    "avg_pred_ret": 0.0, "avg_actual_ret": 0.0,
                    "pred_returns": [], "actual_returns": [],
                }
            ph = stats["per_horizon"][horizon]
            ph["total"] += 1
            if hdata.get("direction_correct"):
                ph["correct"] += 1
            ph["pred_returns"].append(hdata["predicted_return"])
            ph["actual_returns"].append(hdata["actual_return"])

            all_scored.append({
                "symbol": entry["symbol"],
                "date":   entry["timestamp"][:10],
                "horizon": horizon,
                "predicted_return": hdata["predicted_return"],
                "actual_return":    hdata["actual_return"],
                "direction_correct": hdata["direction_correct"],
                "confidence":       hdata["confidence"],
            })

    # Compute averages
    if stats["scored_predictions"] > 0:
        stats["direction_accuracy"] = round(
            stats["direction_correct"] / stats["scored_predictions"] * 100, 1)

    for horizon, ph in stats["per_horizon"].items():
        if ph["total"] > 0:
            ph["accuracy"] = round(ph["correct"] / ph["total"] * 100, 1)
            ph["avg_pred_ret"] = round(float(np.mean(ph["pred_returns"])) * 100, 2)
            ph["avg_actual_ret"] = round(float(np.mean(ph["actual_returns"])) * 100, 2)
        # Clean up internal lists
        del ph["pred_returns"]
        del ph["actual_returns"]

    # Most recent scored predictions
    stats["recent"] = sorted(all_scored, key=lambda x: x["date"], reverse=True)[:10]

    return stats
