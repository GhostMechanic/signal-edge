"""
model_improvement.py
=====================
Continuous model improvement system that:
  - Tracks which features predict well vs poorly
  - Auto-adjusts ensemble weights based on prediction outcomes
  - Triggers model retraining after enough data
  - Manages model versions and A/B comparisons
  - Provides feature selection recommendations

Usage:
    improver = ModelImprover()
    improver.analyze_prediction_outcomes()      # update from scored predictions
    improver.get_feature_recommendations()      # what features to boost/drop
    improver.get_retraining_adjustments()       # parameters for next retrain
    improver.should_retrain()                   # check if retrain is needed
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Optional

from prediction_logger_v2 import (
    get_full_analytics,
    get_feature_importance_ranking,
    get_current_model_version,
    should_retrain as _should_retrain,
    record_retrain,
    _load_importance,
    _save_importance,
)

# ─── Paths ───────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".predictions")
ADJUSTMENTS_FILE = os.path.join(LOG_DIR, "model_adjustments.json")


def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _load_adjustments() -> dict:
    _ensure_dir()
    if os.path.exists(ADJUSTMENTS_FILE):
        try:
            with open(ADJUSTMENTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "confidence_bias": 0.0,
        "horizon_weights": {},
        "feature_boosts": {},
        "feature_penalties": {},
        "regime_adjustments": {},
        "ensemble_weight_overrides": {},
        "last_analysis": None,
        "analysis_count": 0,
    }


def _save_adjustments(data: dict):
    _ensure_dir()
    with open(ADJUSTMENTS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


class ModelImprover:
    """
    Analyzes prediction outcomes and generates recommendations
    for model improvement.
    """

    def __init__(self):
        self.analytics = None
        self.adjustments = _load_adjustments()
        self.feature_ranking = []

    def refresh(self):
        """Refresh analytics data."""
        self.analytics = get_full_analytics()
        self.feature_ranking = get_feature_importance_ranking()

    # ═══════════════════════════════════════════════════════════════════════════
    # OUTCOME ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════

    def analyze_prediction_outcomes(self) -> dict:
        """
        Analyze all scored predictions and update model adjustments.
        Returns summary of findings.
        """
        self.refresh()
        a = self.analytics

        findings = {
            "total_scored": a["scored_any"],
            "overall_accuracy": 0,
            "horizon_insights": {},
            "confidence_calibration_error": 0,
            "regime_insights": {},
            "feature_insights": [],
            "recommendations": [],
        }

        if a["scored_any"] == 0:
            findings["recommendations"].append("Run more analyses to generate prediction data.")
            return findings

        # ── Overall accuracy ──
        findings["overall_accuracy"] = round(
            a["direction_correct_any"] / a["scored_any"] * 100, 1)

        # ── Horizon-specific insights ──
        for horizon, ph in a["per_horizon"].items():
            scored = ph.get("scored", 0) + ph.get("quick_scored", 0)
            if scored == 0:
                continue

            # Use quick accuracy if no final scores yet
            acc = ph.get("accuracy", 0)
            if acc == 0 and ph.get("quick_accuracy", 0) > 0:
                acc = ph["quick_accuracy"]

            insight = {
                "accuracy": acc,
                "avg_confidence": ph.get("avg_confidence", 50),
                "calibration_error": abs(acc - ph.get("avg_confidence", 50)),
                "recommendation": "",
            }

            # Check if confidence is well-calibrated
            if insight["calibration_error"] > 15:
                if acc < ph["avg_confidence"]:
                    insight["recommendation"] = f"Overconfident: reduce {horizon} confidence by ~{insight['calibration_error']:.0f}%"
                    self.adjustments["horizon_weights"][horizon] = -insight["calibration_error"] / 100
                else:
                    insight["recommendation"] = f"Underconfident: boost {horizon} confidence by ~{insight['calibration_error']:.0f}%"
                    self.adjustments["horizon_weights"][horizon] = insight["calibration_error"] / 100
            else:
                insight["recommendation"] = f"Well-calibrated for {horizon}"
                self.adjustments["horizon_weights"][horizon] = 0

            findings["horizon_insights"][horizon] = insight

        # ── Confidence calibration ──
        cal_data = a.get("confidence_calibration", [])
        if cal_data:
            errors = [abs(c["predicted_confidence"] - c["actual_accuracy"]) for c in cal_data]
            findings["confidence_calibration_error"] = round(np.mean(errors), 1)

            if findings["confidence_calibration_error"] > 10:
                # Calculate global confidence bias
                biases = [c["actual_accuracy"] - c["predicted_confidence"] for c in cal_data]
                avg_bias = np.mean(biases)
                self.adjustments["confidence_bias"] = round(float(avg_bias), 2)
                findings["recommendations"].append(
                    f"Confidence bias: {avg_bias:+.1f}%. "
                    f"{'Reduce' if avg_bias < 0 else 'Increase'} confidence scores by ~{abs(avg_bias):.0f}%."
                )

        # ── Regime insights ──
        for regime, rg in a.get("regime_accuracy", {}).items():
            if rg["scored"] > 0:
                acc = rg["accuracy"]
                findings["regime_insights"][regime] = {
                    "accuracy": acc,
                    "total": rg["total"],
                    "scored": rg["scored"],
                }
                # Adjust regime confidence multiplier
                if acc < 50 and regime != "Bull":
                    self.adjustments["regime_adjustments"][regime] = -0.15
                    findings["recommendations"].append(
                        f"Poor {regime} accuracy ({acc}%). Reducing confidence in {regime} markets."
                    )
                elif acc > 60:
                    self.adjustments["regime_adjustments"][regime] = 0.05
                    findings["recommendations"].append(
                        f"Strong {regime} accuracy ({acc}%). Model works well in {regime} markets."
                    )

        # ── Feature insights ──
        if self.feature_ranking:
            top_5 = self.feature_ranking[:5]
            bottom_5 = self.feature_ranking[-5:] if len(self.feature_ranking) > 5 else []

            findings["feature_insights"] = {
                "top_features": top_5,
                "weak_features": bottom_5,
            }

            # Record feature boosts/penalties
            for f in top_5:
                self.adjustments["feature_boosts"][f["feature"]] = round(
                    f["cumulative_score"], 4)

            for f in bottom_5:
                if f["cumulative_score"] < 0:
                    self.adjustments["feature_penalties"][f["feature"]] = round(
                        f["cumulative_score"], 4)
                    findings["recommendations"].append(
                        f"Consider dropping feature '{f['feature']}' "
                        f"(negative score: {f['cumulative_score']:.3f})"
                    )

        # ── Retrain recommendation ──
        if _should_retrain():
            findings["recommendations"].append(
                "Model ready for retraining. 20+ new scored predictions since last update."
            )

        # Save adjustments
        self.adjustments["last_analysis"] = datetime.now().isoformat()
        self.adjustments["analysis_count"] = self.adjustments.get("analysis_count", 0) + 1
        _save_adjustments(self.adjustments)

        return findings

    # ═══════════════════════════════════════════════════════════════════════════
    # RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def get_feature_recommendations(self) -> dict:
        """
        Return feature boost/drop recommendations.
        """
        if not self.feature_ranking:
            self.refresh()

        if not self.feature_ranking:
            return {"boost": [], "drop": [], "neutral": []}

        boost = [f for f in self.feature_ranking if f["cumulative_score"] > 0.01]
        drop = [f for f in self.feature_ranking if f["cumulative_score"] < -0.01]
        neutral = [f for f in self.feature_ranking if -0.01 <= f["cumulative_score"] <= 0.01]

        return {
            "boost": boost[:10],
            "drop": drop[:10],
            "neutral": neutral[:10],
        }

    def get_retraining_adjustments(self) -> dict:
        """
        Return the full set of adjustments to apply during next retrain.
        """
        return {
            "confidence_bias": self.adjustments.get("confidence_bias", 0),
            "horizon_weights": self.adjustments.get("horizon_weights", {}),
            "feature_boosts": self.adjustments.get("feature_boosts", {}),
            "feature_penalties": self.adjustments.get("feature_penalties", {}),
            "regime_adjustments": self.adjustments.get("regime_adjustments", {}),
            "model_version": get_current_model_version(),
            "should_retrain": _should_retrain(),
        }

    def should_retrain(self) -> bool:
        """Check if model should be retrained."""
        return _should_retrain()

    # ═══════════════════════════════════════════════════════════════════════════
    # AUTO-RETRAIN
    # ═══════════════════════════════════════════════════════════════════════════

    def get_confidence_adjustment(self, horizon: str, base_confidence: float, regime: str = "Unknown") -> float:
        """
        Apply learned adjustments to confidence score.
        Called during prediction to improve confidence calibration.
        """
        adjusted = base_confidence

        # Global confidence bias correction
        bias = self.adjustments.get("confidence_bias", 0)
        adjusted += bias

        # Horizon-specific adjustment
        h_adj = self.adjustments.get("horizon_weights", {}).get(horizon, 0)
        adjusted += h_adj * 100  # Convert to percentage points

        # Regime adjustment
        r_adj = self.adjustments.get("regime_adjustments", {}).get(regime, 0)
        adjusted *= (1 + r_adj)

        return max(30.0, min(95.0, adjusted))

    def get_improvement_summary(self) -> dict:
        """
        Return a user-friendly summary of model improvement state.
        """
        self.refresh()
        a = self.analytics

        summary = {
            "model_version": get_current_model_version(),
            "total_predictions": a["total_predictions"],
            "scored_predictions": a["scored_any"],
            "live_accuracy": 0,
            "quick_accuracy": a.get("quick_accuracy", 0),
            "improvement_trend": "not enough data",
            "next_retrain_in": 0,
            "top_features": [],
            "weak_features": [],
            "confidence_calibration": "not enough data",
            "best_horizon": "",
            "worst_horizon": "",
        }

        if a["scored_any"] > 0:
            summary["live_accuracy"] = round(
                a["direction_correct_any"] / a["scored_any"] * 100, 1)

        # Trend analysis
        trend = a.get("accuracy_over_time", [])
        if len(trend) >= 2:
            recent = np.mean([t["rolling_accuracy"] for t in trend[-5:]])
            older = np.mean([t["rolling_accuracy"] for t in trend[:5]])
            if recent > older + 2:
                summary["improvement_trend"] = f"improving (+{recent-older:.1f}%)"
            elif recent < older - 2:
                summary["improvement_trend"] = f"declining ({recent-older:.1f}%)"
            else:
                summary["improvement_trend"] = "stable"

        # Next retrain countdown
        from prediction_logger_v2 import _load_model_versions
        mv_data = _load_model_versions()
        last_scored = mv_data.get("last_retrain_scored", 0)
        summary["next_retrain_in"] = max(0, 20 - (a["scored_any"] - last_scored))

        # Best/worst horizon
        best_acc, worst_acc = -1, 101
        for h, ph in a["per_horizon"].items():
            acc = ph.get("accuracy", ph.get("quick_accuracy", 0))
            if acc > best_acc and (ph.get("scored", 0) + ph.get("quick_scored", 0)) > 0:
                best_acc = acc
                summary["best_horizon"] = f"{h} ({acc}%)"
            if acc < worst_acc and (ph.get("scored", 0) + ph.get("quick_scored", 0)) > 0:
                worst_acc = acc
                summary["worst_horizon"] = f"{h} ({acc}%)"

        # Top/weak features
        if self.feature_ranking:
            summary["top_features"] = [
                f["feature"] for f in self.feature_ranking[:5]
                if f["cumulative_score"] > 0
            ]
            summary["weak_features"] = [
                f["feature"] for f in self.feature_ranking
                if f["cumulative_score"] < -0.01
            ][:5]

        # Calibration quality
        cal = a.get("confidence_calibration", [])
        if cal:
            avg_error = np.mean([abs(c["predicted_confidence"] - c["actual_accuracy"]) for c in cal])
            if avg_error < 5:
                summary["confidence_calibration"] = "excellent"
            elif avg_error < 10:
                summary["confidence_calibration"] = "good"
            elif avg_error < 20:
                summary["confidence_calibration"] = "fair"
            else:
                summary["confidence_calibration"] = "poor — needs retraining"

        return summary
