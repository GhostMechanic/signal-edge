"""
daily_briefing.py — Daily Intelligence Briefing
─────────────────────────────────────────────────
Generates and caches a daily summary of:
  - Yesterday's results (what scored, W/L)
  - Today's top signals
  - Portfolio status
  - Market conditions

Designed to be called once per session and cached.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any

BRIEFING_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".predictions", "daily_briefing.json",
)


def _load_briefing() -> dict:
    """Load the most recent briefing from disk."""
    if os.path.exists(BRIEFING_FILE):
        try:
            with open(BRIEFING_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_briefing(briefing: dict):
    """Save briefing to disk."""
    os.makedirs(os.path.dirname(BRIEFING_FILE), exist_ok=True)
    with open(BRIEFING_FILE, "w") as f:
        json.dump(briefing, f, indent=2, default=str)


def generate_briefing() -> dict:
    """
    Generate today's briefing. Called once per session.
    Returns dict with sections for the dashboard.
    """
    from prediction_logger_v2 import get_full_analytics

    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Check if we already have today's briefing
    existing = _load_briefing()
    if existing.get("date") == today and existing.get("generated"):
        return existing

    briefing = {
        "date": today,
        "generated": datetime.now().isoformat(),
        "yesterday_results": [],
        "summary_text": "",
    }

    try:
        pa = get_full_analytics()
        preds = pa.get("predictions_table", [])

        # Yesterday's scored results
        scored = [p for p in preds if p.get("final_result") in ("✓", "✗")]
        yesterday_scored = []
        for p in scored:
            ts = p.get("timestamp", "")
            # Check if this prediction's scoring happened recently
            yesterday_scored.append(p)

        recent = scored[:10]
        wins = sum(1 for p in recent if p.get("final_result") == "✓")
        losses = len(recent) - wins

        # Build summary
        total_scored = pa.get("scored_any", 0)
        accuracy = max(pa.get("live_accuracy", 0), pa.get("quick_accuracy", 0))

        if total_scored > 0:
            briefing["summary_text"] = (
                f"Prediqt has scored {total_scored} predictions with {accuracy:.0f}% accuracy. "
                f"Last 10 results: {wins}W / {losses}L."
            )
        else:
            briefing["summary_text"] = "No predictions scored yet. Run a Deep Scan to generate signals."

        briefing["recent_wins"] = wins
        briefing["recent_losses"] = losses
        briefing["total_scored"] = total_scored
        briefing["accuracy"] = accuracy

    except Exception as e:
        briefing["summary_text"] = "Briefing data unavailable."
        briefing["error"] = str(e)

    _save_briefing(briefing)
    return briefing


def get_briefing() -> dict:
    """Get today's briefing, generating if needed."""
    return generate_briefing()
