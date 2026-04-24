"""
model.py  –  Stacked Ensemble Prediction Engine  (v5)
------------------------------------------------------
Architecture:
  Level 1:  XGBoost (5) + LightGBM (3) + RandomForest (2)
            + XGBClassifier (2 direction) + RF Classifier (2 direction)
            + Risk-Adjusted regressors (2) = up to 16 L1 members
            Each with feature bagging (random 75% of features) for diversity.
            Time-weighted sample weighting (recent data matters more).
  Level 2:  Ridge meta-learner on L1 out-of-fold predictions
            with adaptive weighting based on recent performance.
  Multi-timeframe: shorter-horizon predictions feed as features into
            longer-horizon models (information cascade).

Accuracy improvements in v5:
  • All v4 improvements (exponential weighting, feature bagging, purged CV, etc.)
  • Fundamental data as features (valuation, profitability, analyst targets)
  • Risk-adjusted targets (return/vol) for Sharpe-focused regressors
  • Direction classifiers with dead-zone thresholding
  • RF classifiers for additional diversity
  • Multi-timeframe stacking (1W preds → 1M features, etc.)
  • Adaptive L2 weighting: underperforming L1 models get lower weight
  • Walk-forward model selection: drop L1 members below accuracy threshold
  • Cross-validated isotonic calibration for honest confidence
  • Venn-ABERS inspired confidence bounding

Python 3.9 compatible throughout.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Optional, List, Dict, Any

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.isotonic import IsotonicRegression

from xgboost import XGBRegressor, XGBClassifier

from data_fetcher import engineer_features, create_targets, HORIZONS, fetch_market_context
from regime_detector import regime_features, detect_regime
from learning_engine import apply_learning

# ── Optional heavy deps ──────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".model_cache")
TRAIN_RATIO      = 0.80
MIN_TRAIN        = 500
N_OPTUNA_TRIALS  = 20
SHAP_SAMPLE      = 300
TOP_K_FEATURES   = 28           # reduced from 40 — less overlap, better diversity
FEATURE_BAG_FRAC = 0.85         # increased from 0.75 — each model sees more of the selected set
PURGE_DAYS       = 5
WEIGHT_DECAY     = 3.0
MODEL_PERF_THRESHOLD = 0.50     # L1 models below this dir accuracy get dropped (was 0.48)
MIN_L1_MODELS    = 5            # always keep at least this many L1 models (was 4)
MAX_L1_MODELS    = 12           # cap ensemble size to prevent weak tail
TRAINING_VERSION = 3            # bump to invalidate cache after accuracy improvements


# ─── Platt (sigmoid) calibrator for confidence when isotonic has few points ──
class PlattCalibrator:
    """Wraps LogisticRegression to have .predict() like IsotonicRegression."""
    def __init__(self, lr_model):
        self._lr = lr_model
    def predict(self, x):
        x = np.asarray(x).reshape(-1, 1)
        return np.clip(self._lr.predict_proba(x)[:, 1], 0.30, 0.95)


# ─── Level-1 model specs ─────────────────────────────────────────────────────
XGB_VARIANTS = [
    dict(n_estimators=500, max_depth=4, learning_rate=0.03,
         subsample=0.80, colsample_bytree=0.80,
         reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
         tree_method="hist", random_state=42),
    dict(n_estimators=400, max_depth=5, learning_rate=0.04,
         subsample=0.70, colsample_bytree=0.75,
         reg_alpha=0.2, reg_lambda=0.8, min_child_weight=4,
         tree_method="hist", random_state=7),
    dict(n_estimators=350, max_depth=3, learning_rate=0.05,
         subsample=0.85, colsample_bytree=0.70,
         reg_alpha=0.05, reg_lambda=1.5, min_child_weight=6,
         tree_method="hist", random_state=13),
    dict(n_estimators=500, max_depth=4, learning_rate=0.03,
         subsample=0.75, colsample_bytree=0.85,
         reg_alpha=0.15, reg_lambda=1.0, min_child_weight=5,
         tree_method="hist", random_state=21),
    dict(n_estimators=400, max_depth=4, learning_rate=0.04,
         subsample=0.65, colsample_bytree=0.80,
         reg_alpha=0.1, reg_lambda=1.2, min_child_weight=4,
         tree_method="hist", random_state=99),
]

LGB_VARIANTS = [
    dict(n_estimators=500, max_depth=4, learning_rate=0.03,
         num_leaves=31, subsample=0.80, colsample_bytree=0.80,
         reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
         random_state=42, verbosity=-1),
    dict(n_estimators=400, max_depth=5, learning_rate=0.04,
         num_leaves=50, subsample=0.70, colsample_bytree=0.75,
         reg_alpha=0.2, reg_lambda=0.8, min_child_samples=15,
         random_state=7,  verbosity=-1),
    dict(n_estimators=350, max_depth=3, learning_rate=0.05,
         num_leaves=20, subsample=0.85, colsample_bytree=0.70,
         reg_alpha=0.05, reg_lambda=1.5, min_child_samples=25,
         random_state=13, verbosity=-1),
]

RF_VARIANTS = [
    dict(n_estimators=250, max_depth=7, min_samples_leaf=8,
         max_features=0.5, random_state=42),
    dict(n_estimators=200, max_depth=5, min_samples_leaf=12,
         max_features=0.6, random_state=7),
]

# Direction classifiers
XGB_CLS_VARIANTS = [
    dict(n_estimators=400, max_depth=4, learning_rate=0.04,
         subsample=0.80, colsample_bytree=0.80,
         reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
         tree_method="hist", random_state=55, use_label_encoder=False,
         eval_metric="logloss"),
    dict(n_estimators=350, max_depth=3, learning_rate=0.05,
         subsample=0.75, colsample_bytree=0.75,
         reg_alpha=0.15, reg_lambda=1.2, min_child_weight=6,
         tree_method="hist", random_state=88, use_label_encoder=False,
         eval_metric="logloss"),
]

# RF classifiers for diversity
RF_CLS_VARIANTS = [
    dict(n_estimators=250, max_depth=6, min_samples_leaf=10,
         max_features=0.5, random_state=33),
    dict(n_estimators=200, max_depth=5, min_samples_leaf=15,
         max_features=0.6, random_state=66),
]

# Risk-adjusted target regressors (trained on return/vol target)
XGB_RADJ_VARIANTS = [
    dict(n_estimators=400, max_depth=4, learning_rate=0.04,
         subsample=0.75, colsample_bytree=0.80,
         reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
         tree_method="hist", random_state=77),
    dict(n_estimators=350, max_depth=3, learning_rate=0.05,
         subsample=0.80, colsample_bytree=0.75,
         reg_alpha=0.15, reg_lambda=1.2, min_child_weight=6,
         tree_method="hist", random_state=44),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _conf_label(conf: float) -> str:
    if conf >= 70: return "High"
    if conf >= 55: return "Medium"
    return "Low"


def _make_sample_weights(n: int, decay: float = WEIGHT_DECAY) -> np.ndarray:
    w = np.exp(np.linspace(0, np.log(decay), n))
    return w / w.mean()


def _feature_bag(feature_names: List[str], frac: float, seed: int,
                  feature_weights: Optional[Dict[str, float]] = None) -> List[str]:
    """
    Select features for bagging with optional learned weights.

    Args:
        feature_names: List of feature names to sample from
        frac: Fraction of features to select
        seed: Random seed for reproducibility
        feature_weights: Optional dict of {feature_name: weight} for biasing selection
                        Features with positive weights are MORE likely to be selected

    Returns:
        List of selected feature names
    """
    rng = np.random.default_rng(seed)
    k   = max(5, int(len(feature_names) * frac))

    # If no weights provided, use uniform distribution
    if not feature_weights:
        idx = rng.choice(len(feature_names), k, replace=False)
        return [feature_names[i] for i in sorted(idx)]

    # Build probability weights: features with high boosts get higher probability
    probs = np.ones(len(feature_names), dtype=float)
    for i, fname in enumerate(feature_names):
        boost = feature_weights.get(fname, 0.0)
        # Convert boost score to probability multiplier: exp(boost) ranges from ~0.37 to ~2.7
        probs[i] = np.exp(max(-1.0, min(1.0, boost)))  # Clip boost to [-1, 1] to prevent extreme values

    # Normalize to probabilities
    probs = probs / probs.sum()

    # Select features with weighted probability
    idx = rng.choice(len(feature_names), k, replace=False, p=probs)
    return [feature_names[i] for i in sorted(idx)]


def _purged_split(X: np.ndarray, y: np.ndarray, train_ratio: float, purge: int):
    n = len(X)
    split = int(n * train_ratio)
    # Cap purge so validation always has at least 30 samples
    max_purge = max(0, n - split - 30)
    actual_purge = min(purge, max_purge)
    val_start = min(split + actual_purge, n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[val_start:], y[val_start:]
    return X_tr, y_tr, X_va, y_va


# ── Horizon-aware purge: target looks N days forward, so purge must be >= N ──
HORIZON_PURGE = {
    "3 Day":     5,
    "1 Week":    7,
    "1 Month":   25,
    "1 Quarter": 70,
    "1 Year":    260,
}


def _walk_forward_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 4,
    purge: int = 5,
    min_train: int = 300,
):
    """
    Expanding-window walk-forward cross-validation.
    Each fold:
      train = data[0 : split_point]
      val   = data[split_point + purge : next_split_point]
    The training window expands with each fold.
    Returns list of (X_tr, y_tr, X_va, y_va) tuples.
    """
    n = len(X)
    # Adjust min_train if dataset is small
    min_train = min(min_train, int(n * 0.5))

    total_val_size = n - min_train
    if total_val_size < n_splits * 50:
        # Not enough data for walk-forward, fall back to single split
        return [_purged_split(X, y, 0.80, purge)]

    fold_val_size = total_val_size // (n_splits + 1)  # +1 so last fold has room
    folds = []
    for i in range(n_splits):
        split_point = min_train + i * fold_val_size
        # Cap purge so validation has room
        actual_purge = min(purge, max(0, n - split_point - 30))
        val_start = min(split_point + actual_purge, n)
        val_end = min(val_start + fold_val_size, n)
        if val_end - val_start < 20:
            continue
        folds.append((
            X[:split_point], y[:split_point],
            X[val_start:val_end], y[val_start:val_end],
        ))
    if not folds:
        return [_purged_split(X, y, 0.80, purge)]
    return folds


def _train_l1_model(model, X_tr, y_tr, X_va, y_va, scaler, weights_tr=None):
    """Fit a single L1 model; returns fitted model + val preds."""
    # Guard: ensure train and val have identical feature counts
    if X_tr.shape[1] != X_va.shape[1]:
        min_cols = min(X_tr.shape[1], X_va.shape[1])
        X_tr = X_tr[:, :min_cols]
        X_va = X_va[:, :min_cols]
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_va = scaler.transform(X_va)

    if isinstance(model, XGBRegressor):
        model.fit(Xs_tr, y_tr, eval_set=[(Xs_va, y_va)],
                  sample_weight=weights_tr, verbose=False)
    elif isinstance(model, XGBClassifier):
        model.fit(Xs_tr, y_tr, eval_set=[(Xs_va, y_va)],
                  sample_weight=weights_tr, verbose=False)
    elif HAS_LGB and isinstance(model, lgb.LGBMRegressor):
        callbacks = [lgb.early_stopping(30, verbose=False),
                     lgb.log_evaluation(period=-1)]
        model.fit(Xs_tr, y_tr,
                  eval_set=[(Xs_va, y_va)],
                  sample_weight=weights_tr,
                  callbacks=callbacks)
    elif isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
        if weights_tr is not None:
            model.fit(Xs_tr, y_tr, sample_weight=weights_tr)
        else:
            model.fit(Xs_tr, y_tr)
    else:
        model.fit(Xs_tr, y_tr)

    # Prediction: classifiers output probability, regressors output value
    if isinstance(model, (XGBClassifier, RandomForestClassifier)):
        proba = model.predict_proba(Xs_va)
        val_preds = proba[:, 1] * 2 - 1  # Map P(up) to [-1, +1]
    else:
        val_preds = model.predict(Xs_va)

    return model, val_preds


# ─── Optuna tuning ────────────────────────────────────────────────────────────

def _optuna_tune_xgb(X_tr, y_tr, X_va, y_va, weights_tr=None,
                     n_trials: int = N_OPTUNA_TRIALS) -> dict:
    if not HAS_OPTUNA:
        return {}

    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 200, 600),
            max_depth        = trial.suggest_int("max_depth", 3, 6),
            learning_rate    = trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
            subsample        = trial.suggest_float("subsample", 0.60, 0.90),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.60, 0.90),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.01, 0.5,  log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 3.0),
            min_child_weight = trial.suggest_int("min_child_weight", 3, 10),
            tree_method      = "hist",
            verbosity        = 0,
            eval_metric      = "rmse",
            early_stopping_rounds = 20,
            random_state     = 42,
        )
        scaler  = RobustScaler()
        Xs_tr   = scaler.fit_transform(X_tr)
        Xs_va   = scaler.transform(X_va)
        model   = XGBRegressor(**params)
        model.fit(Xs_tr, y_tr, eval_set=[(Xs_va, y_va)],
                  sample_weight=weights_tr, verbose=False)
        preds   = model.predict(Xs_va)
        dir_acc = float(np.mean(np.sign(preds) == np.sign(y_va)))
        return -dir_acc

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ─── SHAP feature selection ──────────────────────────────────────────────────

def _shap_top_features(
    model, scaler: RobustScaler,
    X_sample: np.ndarray,
    feature_names: List[str],
    top_k: int = TOP_K_FEATURES,
) -> List[str]:
    if not HAS_SHAP:
        return feature_names
    try:
        Xs = scaler.transform(X_sample)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xs)
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:top_k]
        return [feature_names[i] for i in sorted(top_idx)]
    except Exception:
        return feature_names


# ─── StockPredictor ──────────────────────────────────────────────────────────

class StockPredictor:
    """
    Two-level stacked ensemble predictor (v5).

    New in v5:
      - Fundamental data features (valuation, profitability, analyst targets)
      - Risk-adjusted target regressors
      - RF direction classifiers for diversity
      - Multi-timeframe stacking (shorter → longer horizon cascade)
      - Adaptive L2 weighting (downweight underperforming L1 members)
      - Walk-forward model selection (drop weak L1 members)
      - Cross-validated isotonic calibration
    """

    def __init__(self, symbol: str):
        self.symbol         = symbol.upper()
        self.l1_members     = {}    # horizon → list of {model, scaler, name, feat_idx, ...}
        self.l2_models      = {}    # horizon → Ridge
        self.calibration    = {}    # horizon → calibration stats
        self.conf_calibrator = {}   # horizon → IsotonicRegression
        self.feature_names  = []
        self.selected_feats = {}    # horizon → selected feature names
        self.regime_cache   = None
        self.horizon_preds  = {}    # for multi-timeframe stacking
        self.model_perf     = {}    # horizon → per-model dir accuracy on val
        self.is_trained     = False
        self.fundamentals   = None  # stored for prediction phase
        self.earnings_data  = None  # stored for earnings proximity checks
        self.options_data   = None  # stored for options features
        self._model_dir     = os.path.join(os.path.dirname(__file__), ".models")

    # ── Model Persistence ───────────────────────────────────────────────────

    def _model_path(self) -> str:
        os.makedirs(self._model_dir, exist_ok=True)
        return os.path.join(self._model_dir, f"{self.symbol}_model.joblib")

    def _meta_path(self) -> str:
        os.makedirs(self._model_dir, exist_ok=True)
        return os.path.join(self._model_dir, f"{self.symbol}_meta.json")

    def save_model(self):
        """Save trained model to disk for reuse."""
        if not self.is_trained:
            return
        # Save heavy objects (sklearn models, scalers) via joblib
        model_data = {
            "l1_members":      self.l1_members,
            "l2_models":       self.l2_models,
            "conf_calibrator": self.conf_calibrator,
            "selected_feats":  self.selected_feats,
            "model_perf":      self.model_perf,
        }
        joblib.dump(model_data, self._model_path(), compress=3)

        # Save lightweight metadata as JSON
        meta = {
            "symbol":        self.symbol,
            "trained_at":    pd.Timestamp.now().isoformat(),
            "feature_names": self.feature_names,
            "calibration":   {h: {k: v for k, v in cal.items()
                                   if k != "calibration_pairs"}
                              for h, cal in self.calibration.items()},
            "horizons":      list(self.l1_members.keys()),
            "training_version": TRAINING_VERSION,
        }
        with open(self._meta_path(), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def load_model(self, max_age_hours: int = 24) -> bool:
        """
        Load a previously trained model from disk.
        Returns True if a fresh-enough model was loaded, False otherwise.
        """
        mp = self._model_path()
        meta_p = self._meta_path()
        if not os.path.exists(mp) or not os.path.exists(meta_p):
            return False

        try:
            with open(meta_p) as f:
                meta = json.load(f)

            # Invalidate cache if training logic has changed
            if meta.get("training_version", 0) != TRAINING_VERSION:
                return False  # incompatible training version

            trained_at = pd.Timestamp(meta["trained_at"])
            age_hours = (pd.Timestamp.now() - trained_at).total_seconds() / 3600
            if age_hours > max_age_hours:
                return False  # model is stale

            model_data = joblib.load(mp)
            self.l1_members      = model_data["l1_members"]
            self.l2_models       = model_data["l2_models"]
            self.conf_calibrator = model_data["conf_calibrator"]
            self.selected_feats  = model_data["selected_feats"]
            self.model_perf      = model_data.get("model_perf", {})
            self.feature_names   = meta.get("feature_names", [])
            # Restore calibration (without pairs to save memory)
            self.calibration     = meta.get("calibration", {})
            self.is_trained      = True
            return True
        except Exception:
            return False

    # ── Feature preparation ──────────────────────────────────────────────────

    def _build_features(
        self,
        df: pd.DataFrame,
        market_ctx: Optional[dict] = None,
        fundamentals: Optional[dict] = None,
        earnings_data: Optional[dict] = None,
        options_data: Optional[dict] = None,
    ) -> pd.DataFrame:
        spy = market_ctx.get("spy") if market_ctx else None
        vix = market_ctx.get("vix") if market_ctx else None
        sec = market_ctx.get("sector") if market_ctx else None
        sentiment = market_ctx.get("sentiment") if market_ctx else None

        base  = engineer_features(df, spy_close=spy, vix_close=vix,
                                   sector_close=sec, fundamentals=fundamentals,
                                   earnings_data=earnings_data, options_data=options_data,
                                   sentiment_ctx=sentiment)
        reg   = regime_features(df, spy_close=spy, vix_close=vix)
        combined = pd.concat([base, reg], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined

    def _prep(self, features, targets, horizon, selected=None, target_col=None):
        col    = target_col or f"target_{horizon}"
        cols   = selected if selected else list(features.columns)
        merged = pd.concat([features[cols], targets[col]], axis=1).dropna()
        X      = merged[cols].values
        y      = merged[col].values
        return X, y, merged.index, cols

    def _load_learned_adjustments(self) -> dict:
        """
        Load learned adjustments from model_adjustments.json.
        These adjustments are generated by analyzing prediction outcomes.
        """
        try:
            adjustments_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                ".predictions", "model_adjustments.json"
            )
            if os.path.exists(adjustments_path):
                with open(adjustments_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            pass
        return {
            "feature_boosts": {},
            "feature_penalties": {},
            "regime_adjustments": {},
            "sample_weights": None,
        }

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        market_ctx: Optional[dict] = None,
        fundamentals: Optional[dict] = None,
        earnings_data: Optional[dict] = None,
        options_data: Optional[dict] = None,
        use_shap:   bool = False,
        use_optuna: bool = False,
        progress_callback = None,
    ) -> "StockPredictor":

        if len(df) < MIN_TRAIN:
            raise ValueError(f"Need at least {MIN_TRAIN} trading days (got {len(df)}).")

        # Try loading a cached model first (skip retraining if fresh enough)
        if self.load_model(max_age_hours=12):
            self.fundamentals = fundamentals
            self.earnings_data = earnings_data
            self.options_data = options_data
            return self

        self.fundamentals = fundamentals
        self.earnings_data = earnings_data
        self.options_data = options_data
        features_all = self._build_features(df, market_ctx, fundamentals, earnings_data, options_data)
        targets      = create_targets(df)
        self.feature_names = list(features_all.columns)

        # ── Load learned adjustments from model improvements ──────────────────
        learned_adjustments = self._load_learned_adjustments()
        feature_boosts = learned_adjustments.get("feature_boosts", {})
        regime_adjustments = learned_adjustments.get("regime_adjustments", {})
        sample_weights_override = learned_adjustments.get("sample_weights", None)

        # Count total training steps for progress bar
        n_cls = len(XGB_CLS_VARIANTS) + len(RF_CLS_VARIANTS)
        n_radj = len(XGB_RADJ_VARIANTS)
        n_l1_per_horizon = (len(XGB_VARIANTS)
                            + (len(LGB_VARIANTS) if HAS_LGB else 0)
                            + len(RF_VARIANTS)
                            + n_cls + n_radj)
        total_steps = len(HORIZONS) * (n_l1_per_horizon + 3)  # +3 for L2 + calibration + selection
        step        = 0

        def _prog(msg: str):
            nonlocal step
            if progress_callback:
                progress_callback(min(step / total_steps, 0.99), msg)
            step += 1

        # Train horizons in order (short → long) for multi-timeframe stacking
        horizon_order = sorted(HORIZONS.keys(), key=lambda h: HORIZONS[h])

        for horizon in horizon_order:

            # Horizon-aware purge gap: must be >= target look-ahead days
            h_purge = HORIZON_PURGE.get(horizon, PURGE_DAYS)

            # ── Optional Optuna tuning ───────────────────────────────────────
            tuned_xgb_params = {}
            if use_optuna and HAS_OPTUNA:
                _prog(f"[{horizon}] Optuna tuning…")
                X_all, y_all, _, _ = self._prep(features_all, targets, horizon)
                X_tr_o, y_tr_o, X_va_o, y_va_o = _purged_split(
                    X_all, y_all, TRAIN_RATIO, h_purge)
                w_tr_o = _make_sample_weights(len(X_tr_o))
                best   = _optuna_tune_xgb(
                    X_tr_o, y_tr_o, X_va_o, y_va_o,
                    weights_tr=w_tr_o, n_trials=N_OPTUNA_TRIALS,
                )
                tuned_xgb_params = {k: v for k, v in best.items()
                                    if k in ("max_depth", "learning_rate",
                                             "subsample", "colsample_bytree",
                                             "reg_alpha", "reg_lambda",
                                             "min_child_weight", "n_estimators")}

            # Multi-timeframe: add shorter-horizon calibration signals as scalar features
            feat_with_mtf = features_all.copy()
            for prev_h, prev_cal in self.horizon_preds.items():
                if HORIZONS[prev_h] < HORIZONS[horizon]:
                    # Use dir_acc and avg_member_acc from shorter horizon as constant features
                    mtf_col = f"mtf_dacc_{prev_h.replace(' ', '_')}"
                    feat_with_mtf[mtf_col] = prev_cal.get("dir_acc", 0.5)
                    mtf_col2 = f"mtf_macc_{prev_h.replace(' ', '_')}"
                    feat_with_mtf[mtf_col2] = prev_cal.get("avg_member_acc", 0.5)

            X_all, y_all, _, all_cols = self._prep(feat_with_mtf, targets, horizon)

            # ── Optional SHAP feature selection ──────────────────────────────
            selected = list(all_cols)
            if use_shap and HAS_SHAP and len(X_all) > SHAP_SAMPLE:
                _prog(f"[{horizon}] SHAP feature selection…")
                X_tr_s, y_tr_s, X_va_s, y_va_s = _purged_split(
                    X_all, y_all, TRAIN_RATIO, h_purge)
                sc_shap = RobustScaler()
                Xs_tr   = sc_shap.fit_transform(X_tr_s)
                Xs_va   = sc_shap.transform(X_va_s)
                shap_mdl = XGBRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    tree_method="hist", verbosity=0,
                    eval_metric="rmse", early_stopping_rounds=20,
                    random_state=42,
                )
                shap_mdl.fit(Xs_tr, y_tr_s, eval_set=[(Xs_va, y_va_s)], verbose=False)
                idx_sample = np.random.default_rng(42).choice(
                    len(X_tr_s), min(SHAP_SAMPLE, len(X_tr_s)), replace=False)
                selected = _shap_top_features(
                    shap_mdl, sc_shap, X_tr_s[idx_sample],
                    all_cols, top_k=TOP_K_FEATURES,
                )
                X_all, y_all, _, all_cols = self._prep(
                    feat_with_mtf, targets, horizon, selected=selected)

            # ── Lightweight importance-based pruning (fallback when SHAP skipped) ─
            if len(selected) > TOP_K_FEATURES and not (use_shap and HAS_SHAP and len(X_all) > SHAP_SAMPLE):
                _prog(f"[{horizon}] Fast feature pruning…")
                try:
                    X_tr_p, y_tr_p, X_va_p, y_va_p = _purged_split(
                        X_all, y_all, TRAIN_RATIO, h_purge)
                    sc_p = RobustScaler()
                    Xp_tr = sc_p.fit_transform(X_tr_p)
                    Xp_va = sc_p.transform(X_va_p)
                    prune_mdl = XGBRegressor(
                        n_estimators=150, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", verbosity=0,
                        eval_metric="rmse", early_stopping_rounds=15,
                        random_state=42,
                    )
                    prune_mdl.fit(Xp_tr, y_tr_p, eval_set=[(Xp_va, y_va_p)], verbose=False)
                    importances = prune_mdl.feature_importances_
                    # Keep top-K features by importance (at least TOP_K_FEATURES)
                    keep_n = max(TOP_K_FEATURES, len(selected) // 2)
                    top_idx = np.argsort(importances)[::-1][:keep_n]
                    selected = [all_cols[i] for i in sorted(top_idx)]
                    X_all, y_all, _, all_cols = self._prep(
                        feat_with_mtf, targets, horizon, selected=selected)
                except Exception:
                    pass  # fall back to full feature set

            self.selected_feats[horizon] = selected

            # ── Purged train/val split (horizon-aware purge) ────────────────
            X_tr, y_tr, X_va, y_va = _purged_split(
                X_all, y_all, TRAIN_RATIO, h_purge)

            # Skip horizon if not enough data for train or val
            if len(X_tr) < 50 or len(X_va) < 10:
                _prog(f"[{horizon}] Skipped — not enough data ({len(X_tr)} train, {len(X_va)} val)")
                continue

            weights_tr = _make_sample_weights(len(X_tr))

            # Walk-forward folds for more honest OOF collection
            wf_folds = _walk_forward_splits(X_all, y_all, n_splits=4, purge=h_purge)

            # Direction labels — volatility-aware threshold (matches target construction)
            days = HORIZONS[horizon]
            # Compute from training data: half the 75th percentile of recent returns
            _daily_rets = np.diff(np.log(df["Close"].values[-len(X_all)-63:]))
            _vol_p75 = np.percentile(np.abs(_daily_rets[-63:]), 75) if len(_daily_rets) >= 63 else 0.01
            dir_threshold = max(0.003, min(_vol_p75 * np.sqrt(days) * 0.5, 0.05))
            y_tr_dir = (y_tr > dir_threshold).astype(int)
            y_va_dir = (y_va > dir_threshold).astype(int)

            # Risk-adjusted targets — aligned to same rows as main target
            y_radj_col = f"target_radj_{horizon}"
            if y_radj_col in targets.columns:
                # Use the same merged index as the main target to ensure alignment
                col_main = f"target_{horizon}"
                cols_sel = selected if selected else list(feat_with_mtf.columns)
                merged_radj = pd.concat([feat_with_mtf[cols_sel],
                                          targets[[col_main, y_radj_col]]], axis=1).dropna()
                y_radj_aligned = merged_radj[y_radj_col].values
                # Split with same ratio/purge as main
                split_r = int(len(y_radj_aligned) * TRAIN_RATIO)
                max_purge_r = max(0, len(y_radj_aligned) - split_r - 10)
                val_start_r = min(split_r + min(h_purge, max_purge_r), len(y_radj_aligned))
                y_radj_tr = np.clip(y_radj_aligned[:split_r], -5, 5)
                y_radj_va = np.clip(y_radj_aligned[val_start_r:], -5, 5)
            else:
                y_radj_tr = y_tr
                y_radj_va = y_va

            # ── Level 1 training ─────────────────────────────────────────────
            members       = []
            oof_val_preds = []
            member_dir_acc = []  # track per-model directional accuracy

            # — XGBoost regressors —
            for i, variant in enumerate(XGB_VARIANTS):
                params = {**variant, **tuned_xgb_params}
                if tuned_xgb_params:
                    params["random_state"] = variant["random_state"]
                _prog(f"[{horizon}] XGB reg {i+1}/{len(XGB_VARIANTS)}…")
                try:
                    bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"],
                                             feature_weights=feature_boosts)
                    feat_idx  = [all_cols.index(f) for f in bag_feats]
                    X_tr_bag  = X_tr[:, feat_idx]
                    X_va_bag  = X_va[:, feat_idx]

                    sc  = RobustScaler()
                    mdl = XGBRegressor(
                        eval_metric="rmse", early_stopping_rounds=30,
                        verbosity=0, **params)
                    mdl, vp = _train_l1_model(mdl, X_tr_bag, y_tr, X_va_bag, y_va, sc, weights_tr)
                    da = float(np.mean(np.sign(vp) == np.sign(y_va))) if len(y_va) > 0 else 0.5
                    members.append({"model": mdl, "scaler": sc, "name": f"xgb_{i}",
                                    "feat_idx": feat_idx, "feat_names": bag_feats,
                                    "is_cls": False, "dir_acc": da})
                    oof_val_preds.append(vp)
                    member_dir_acc.append(da)
                except Exception:
                    pass

            # — LightGBM regressors —
            if HAS_LGB:
                for i, variant in enumerate(LGB_VARIANTS):
                    _prog(f"[{horizon}] LGB reg {i+1}/{len(LGB_VARIANTS)}…")
                    try:
                        bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"]+100,
                                                 feature_weights=feature_boosts)
                        feat_idx  = [all_cols.index(f) for f in bag_feats]
                        X_tr_bag  = X_tr[:, feat_idx]
                        X_va_bag  = X_va[:, feat_idx]

                        sc  = RobustScaler()
                        mdl = lgb.LGBMRegressor(**variant)
                        mdl, vp = _train_l1_model(mdl, X_tr_bag, y_tr, X_va_bag, y_va, sc, weights_tr)
                        da = float(np.mean(np.sign(vp) == np.sign(y_va))) if len(y_va) > 0 else 0.5
                        members.append({"model": mdl, "scaler": sc, "name": f"lgb_{i}",
                                        "feat_idx": feat_idx, "feat_names": bag_feats,
                                        "is_cls": False, "dir_acc": da})
                        oof_val_preds.append(vp)
                        member_dir_acc.append(da)
                    except Exception:
                        pass

            # — Random Forest regressors —
            for i, variant in enumerate(RF_VARIANTS):
                _prog(f"[{horizon}] RF reg {i+1}/{len(RF_VARIANTS)}…")
                try:
                    bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"]+200,
                                             feature_weights=feature_boosts)
                    feat_idx  = [all_cols.index(f) for f in bag_feats]
                    X_tr_bag  = X_tr[:, feat_idx]
                    X_va_bag  = X_va[:, feat_idx]

                    sc  = RobustScaler()
                    mdl = RandomForestRegressor(n_jobs=-1, **variant)
                    mdl, vp = _train_l1_model(mdl, X_tr_bag, y_tr, X_va_bag, y_va, sc, weights_tr)
                    da = float(np.mean(np.sign(vp) == np.sign(y_va))) if len(y_va) > 0 else 0.5
                    members.append({"model": mdl, "scaler": sc, "name": f"rf_{i}",
                                    "feat_idx": feat_idx, "feat_names": bag_feats,
                                    "is_cls": False, "dir_acc": da})
                    oof_val_preds.append(vp)
                    member_dir_acc.append(da)
                except Exception:
                    pass

            # — XGBoost direction classifiers —
            for i, variant in enumerate(XGB_CLS_VARIANTS):
                _prog(f"[{horizon}] XGB cls {i+1}/{len(XGB_CLS_VARIANTS)}…")
                try:
                    bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"]+300,
                                             feature_weights=feature_boosts)
                    feat_idx  = [all_cols.index(f) for f in bag_feats]
                    X_tr_bag  = X_tr[:, feat_idx]
                    X_va_bag  = X_va[:, feat_idx]

                    sc  = RobustScaler()
                    cls_params = {k: v for k, v in variant.items()
                                  if k not in ("eval_metric", "use_label_encoder")}
                    mdl = XGBClassifier(
                        eval_metric="logloss", use_label_encoder=False,
                        early_stopping_rounds=30, verbosity=0,
                        **cls_params)
                    mdl, vp = _train_l1_model(mdl, X_tr_bag, y_tr_dir, X_va_bag, y_va_dir, sc, weights_tr)
                    da = float(np.mean(np.sign(vp) == np.sign(y_va))) if len(y_va) > 0 else 0.5
                    members.append({"model": mdl, "scaler": sc, "name": f"cls_{i}",
                                    "feat_idx": feat_idx, "feat_names": bag_feats,
                                    "is_cls": True, "dir_acc": da})
                    oof_val_preds.append(vp)
                    member_dir_acc.append(da)
                except Exception:
                    pass

            # — Random Forest classifiers —
            for i, variant in enumerate(RF_CLS_VARIANTS):
                _prog(f"[{horizon}] RF cls {i+1}/{len(RF_CLS_VARIANTS)}…")
                try:
                    bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"]+400,
                                             feature_weights=feature_boosts)
                    feat_idx  = [all_cols.index(f) for f in bag_feats]
                    X_tr_bag  = X_tr[:, feat_idx]
                    X_va_bag  = X_va[:, feat_idx]

                    sc  = RobustScaler()
                    mdl = RandomForestClassifier(n_jobs=-1, **variant)
                    mdl, vp = _train_l1_model(mdl, X_tr_bag, y_tr_dir, X_va_bag, y_va_dir, sc, weights_tr)
                    da = float(np.mean(np.sign(vp) == np.sign(y_va))) if len(y_va) > 0 else 0.5
                    members.append({"model": mdl, "scaler": sc, "name": f"rfcls_{i}",
                                    "feat_idx": feat_idx, "feat_names": bag_feats,
                                    "is_cls": True, "dir_acc": da})
                    oof_val_preds.append(vp)
                    member_dir_acc.append(da)
                except Exception:
                    pass

            # — Risk-adjusted regressors —
            # Only use risk-adjusted targets if they align with X_tr/X_va
            use_radj = (len(y_radj_tr) == len(y_tr) and len(y_radj_va) == len(y_va))
            radj_y_tr = y_radj_tr if use_radj else y_tr
            radj_y_va = y_radj_va if use_radj else y_va

            for i, variant in enumerate(XGB_RADJ_VARIANTS):
                _prog(f"[{horizon}] XGB radj {i+1}/{len(XGB_RADJ_VARIANTS)}…")
                try:
                    bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"]+500,
                                             feature_weights=feature_boosts)
                    feat_idx  = [all_cols.index(f) for f in bag_feats]
                    X_tr_bag  = X_tr[:, feat_idx]
                    X_va_bag  = X_va[:, feat_idx]

                    sc  = RobustScaler()
                    mdl = XGBRegressor(
                        eval_metric="rmse", early_stopping_rounds=30,
                        verbosity=0, **variant)
                    mdl, vp = _train_l1_model(mdl, X_tr_bag, radj_y_tr, X_va_bag, radj_y_va, sc, weights_tr)
                    # Convert risk-adjusted prediction to directional signal for dir_acc
                    da = float(np.mean(np.sign(vp) == np.sign(y_va))) if len(vp) == len(y_va) and len(y_va) > 0 else 0.5
                    members.append({"model": mdl, "scaler": sc, "name": f"radj_{i}",
                                    "feat_idx": feat_idx, "feat_names": bag_feats,
                                    "is_cls": False, "is_radj": True, "dir_acc": da})
                    oof_val_preds.append(vp)
                    member_dir_acc.append(da)
                except Exception:
                    pass

            # Guard: if ALL models failed somehow, skip this horizon
            if not members:
                continue

            # ── Walk-forward model selection (drop weak members) ─────────────
            _prog(f"[{horizon}] Model selection…")
            member_dir_acc = np.array(member_dir_acc)
            keep_mask = member_dir_acc >= MODEL_PERF_THRESHOLD

            # Always keep at least MIN_L1_MODELS
            if keep_mask.sum() < MIN_L1_MODELS:
                top_idx = np.argsort(member_dir_acc)[::-1][:MIN_L1_MODELS]
                keep_mask = np.zeros(len(members), dtype=bool)
                keep_mask[top_idx] = True

            # Cap at MAX_L1_MODELS to prevent weak tail from diluting ensemble
            if keep_mask.sum() > MAX_L1_MODELS:
                kept_indices = np.where(keep_mask)[0]
                accs_of_kept = member_dir_acc[kept_indices]
                best_of_kept = kept_indices[np.argsort(accs_of_kept)[::-1][:MAX_L1_MODELS]]
                keep_mask = np.zeros(len(members), dtype=bool)
                keep_mask[best_of_kept] = True

            kept_members   = [m for m, k in zip(members, keep_mask) if k]
            kept_oof       = [p for p, k in zip(oof_val_preds, keep_mask) if k]
            kept_dir_acc   = member_dir_acc[keep_mask]

            self.model_perf[horizon] = {
                m["name"]: float(da) for m, da in zip(members, member_dir_acc)
            }

            # ── Level 2 meta-learner with walk-forward OOF ────────────────
            _prog(f"[{horizon}] Training meta-learner (walk-forward)…")
            meta_X = np.column_stack(kept_oof)

            # Adaptive: weight each L1 model's column by its dir accuracy
            model_weights = np.clip(kept_dir_acc - 0.45, 0.05, 1.0)
            model_weights = model_weights / model_weights.mean()
            meta_X_weighted = meta_X * model_weights[np.newaxis, :]

            # ── Walk-forward L2: train on expanding window, collect OOS preds ──
            # This gives truly out-of-sample predictions for calibration.
            _prog(f"[{horizon}] Walk-forward validation…")
            wf_oos_preds = []     # out-of-sample L2 predictions
            wf_oos_actuals = []   # corresponding actual returns
            wf_oos_meta_X = []    # corresponding L1 prediction vectors

            if len(wf_folds) >= 2:
                for fi, (wf_X_tr, wf_y_tr, wf_X_va, wf_y_va) in enumerate(wf_folds):
                    try:
                        # Get L1 predictions for this fold's val set using the
                        # already-trained L1 models (they saw the full training set,
                        # but the walk-forward val sets are temporally later)
                        fold_l1_preds = []
                        for m in kept_members:
                            sc_f = m["scaler"]
                            mdl_f = m["model"]
                            fi_idx = m["feat_idx"]
                            wf_va_bag = wf_X_va[:, fi_idx] if max(fi_idx) < wf_X_va.shape[1] else wf_X_va[:, :len(fi_idx)]
                            try:
                                wf_va_scaled = sc_f.transform(wf_va_bag)
                                if m["is_cls"]:
                                    fp = mdl_f.predict_proba(wf_va_scaled)[:, 1] * 2 - 1
                                else:
                                    fp = mdl_f.predict(wf_va_scaled)
                                fold_l1_preds.append(fp)
                            except Exception:
                                fold_l1_preds.append(np.zeros(len(wf_X_va)))

                        if fold_l1_preds:
                            fold_meta = np.column_stack(fold_l1_preds) * model_weights[np.newaxis, :]
                            # Train a quick L2 on first half of meta_X to prevent overfitting
                            n_tr_equiv = max(30, len(meta_X_weighted) // 2)
                            l2_fold = Ridge(alpha=1.0)
                            l2_fold.fit(meta_X_weighted[:n_tr_equiv], y_va[:n_tr_equiv])
                            fold_preds = l2_fold.predict(fold_meta)
                            wf_oos_preds.extend(fold_preds.tolist())
                            wf_oos_actuals.extend(wf_y_va.tolist())
                            wf_oos_meta_X.extend(fold_meta.tolist())
                    except Exception:
                        continue

            # Compute honest walk-forward accuracy
            if len(wf_oos_preds) >= 30:
                wf_preds_arr = np.array(wf_oos_preds)
                wf_actuals_arr = np.array(wf_oos_actuals)
                val_dir_acc = float(np.mean(
                    np.sign(wf_preds_arr) == np.sign(wf_actuals_arr)
                ))
            else:
                # Fallback: split-half on main validation
                val_mid = len(y_va) // 2
                if val_mid > 20:
                    l2_probe = Ridge(alpha=1.0)
                    l2_probe.fit(meta_X_weighted[:val_mid], y_va[:val_mid])
                    probe_preds = l2_probe.predict(meta_X_weighted[val_mid:])
                    val_dir_acc = float(np.mean(
                        np.sign(probe_preds) == np.sign(y_va[val_mid:])
                    ))
                else:
                    l1_avg_preds = np.mean(meta_X, axis=1)
                    val_dir_acc = float(np.mean(
                        np.sign(l1_avg_preds) == np.sign(y_va)
                    ))

            # Train final L2 on FULL validation for actual predictions
            l2_model = Ridge(alpha=1.0)
            l2_model.fit(meta_X_weighted, y_va)

            # ── Calibration (on walk-forward OOS data when available) ───────
            _prog(f"[{horizon}] Calibrating confidence…")

            # Use walk-forward OOS for calibration if we have enough
            if len(wf_oos_preds) >= 50:
                cal_preds = np.array(wf_oos_preds)
                cal_actuals = np.array(wf_oos_actuals)
                cal_meta_raw = np.array(wf_oos_meta_X)
            else:
                # Fallback to in-sample (less honest but better than nothing)
                cal_preds = l2_model.predict(meta_X_weighted)
                cal_actuals = y_va
                cal_meta_raw = meta_X

            errors       = np.abs(cal_preds - cal_actuals)
            per_pred_std = np.std(cal_meta_raw, axis=1) if cal_meta_raw.ndim == 2 else np.zeros(len(cal_preds))

            dir_correct = (np.sign(cal_preds) == np.sign(cal_actuals)).astype(float)

            std_norm   = np.clip(per_pred_std / (np.percentile(per_pred_std, 90) + 1e-9), 0, 1)
            agree_frac = np.array([
                float(np.mean(np.sign(cal_meta_raw[j]) == np.sign(cal_preds[j])))
                if cal_meta_raw.ndim == 2 else 0.5
                for j in range(len(cal_preds))
            ])

            avg_member_acc = float(kept_dir_acc.mean())
            raw_conf = (0.45 * agree_frac
                       + 0.30 * (1 - std_norm)
                       + 0.25 * avg_member_acc)
            blended = 0.5 * raw_conf + 0.5 * val_dir_acc

            # ── Calibration with isotonic + Platt scaling fallback ──────────
            try:
                n_cal = len(blended)
                mid   = n_cal // 2

                if mid > 20:
                    # Cross-validated isotonic calibration
                    iso1 = IsotonicRegression(y_min=0.30, y_max=0.95, out_of_bounds="clip")
                    iso2 = IsotonicRegression(y_min=0.30, y_max=0.95, out_of_bounds="clip")
                    iso1.fit(blended[:mid], dir_correct[:mid])
                    iso2.fit(blended[mid:], dir_correct[mid:])
                    cal1 = iso1.predict(blended)
                    cal2 = iso2.predict(blended)
                    avg_cal = (cal1 + cal2) / 2
                    iso_final = IsotonicRegression(y_min=0.30, y_max=0.95, out_of_bounds="clip")
                    iso_final.fit(blended, avg_cal)
                    self.conf_calibrator[horizon] = iso_final
                elif mid > 5:
                    # Too few points for cross-validated isotonic — use Platt scaling
                    # Platt scaling: fit logistic regression (sigmoid) to map
                    # raw confidence → calibrated probability
                    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
                    platt.fit(blended.reshape(-1, 1), dir_correct.astype(int))
                    self.conf_calibrator[horizon] = PlattCalibrator(platt)
                else:
                    self.conf_calibrator[horizon] = None
            except Exception:
                self.conf_calibrator[horizon] = None

            # ── Training target distribution (for extreme prediction shrinkage) ─
            y_all_returns = np.concatenate([y_tr, y_va])
            # Store per-sample (confidence, correct) pairs for reliability diagram
            # Use the iso-calibrated confidence when available, else blended.
            try:
                iso_for_cal = self.conf_calibrator.get(horizon)
                if iso_for_cal is not None:
                    cal_confidences = iso_for_cal.predict(blended)
                else:
                    cal_confidences = blended
            except Exception:
                cal_confidences = blended
            calibration_pairs = [
                {"confidence": float(c), "correct": bool(d)}
                for c, d in zip(cal_confidences, dir_correct)
            ]
            self.calibration[horizon] = {
                "mean_abs_err":    float(np.mean(errors)),
                "std_abs_err":     float(np.std(errors)),
                "dir_acc":         val_dir_acc,     # honest OOS accuracy
                "pred_std_p50":    float(np.percentile(per_pred_std, 50)),
                "pred_std_p90":    float(np.percentile(per_pred_std, 90)),
                "n_models":        len(kept_members),
                "n_dropped":       int((~keep_mask).sum()),
                "avg_member_acc":  avg_member_acc,
                # Empirical return distribution for sanity-checking predictions
                "target_mean":     float(np.mean(y_all_returns)),
                "target_std":      float(np.std(y_all_returns)),
                "target_p05":      float(np.percentile(y_all_returns, 5)),
                "target_p95":      float(np.percentile(y_all_returns, 95)),
                "target_max_abs":  float(np.max(np.abs(y_all_returns))),
                # Reliability-diagram raw data
                "calibration_pairs": calibration_pairs,
            }

            self.l1_members[horizon]  = kept_members
            self.l2_models[horizon]   = l2_model
            # Store model weights for prediction
            for i, m in enumerate(kept_members):
                m["l2_weight"] = float(model_weights[i])

            # ── Store calibration stats for multi-timeframe cascade ──────────
            # Must be after calibration block so val_dir_acc & avg_member_acc exist
            self.horizon_preds[horizon] = {
                "dir_acc":        val_dir_acc,
                "avg_member_acc": avg_member_acc,
            }

        # Mark trained if at least one horizon was successfully trained
        if self.l1_members:
            self.is_trained = True
        else:
            raise ValueError("Training failed: no horizons had enough data.")

        # Persist to disk for fast reload
        try:
            self.save_model()
        except Exception:
            pass  # Non-fatal: model works in memory even if save fails

        if progress_callback:
            progress_callback(1.0, "Training complete.")
        return self

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(
        self,
        df: pd.DataFrame,
        market_ctx: Optional[dict] = None,
    ) -> dict:
        if not self.is_trained:
            raise RuntimeError("Call .train() first.")

        features_all = self._build_features(df, market_ctx, self.fundamentals,
                                              self.earnings_data, self.options_data)
        results      = {}

        spy = market_ctx.get("spy") if market_ctx else None
        vix = market_ctx.get("vix") if market_ctx else None
        self.regime_cache = detect_regime(df, spy, vix)

        current_price = float(df["Close"].iloc[-1])

        # Predict in horizon order (short → long) for multi-timeframe
        horizon_order = sorted(HORIZONS.keys(), key=lambda h: HORIZONS[h])

        for horizon in horizon_order:
            # Skip if this horizon wasn't trained (e.g., all L1 models failed)
            if horizon not in self.l1_members:
                continue

            members = self.l1_members[horizon]

            # Reconstruct MTF scalar columns exactly as done during training
            # so that member scalers (which were fit on bags including MTF cols)
            # receive the same feature dimensionality at predict time.
            feat_with_mtf = features_all.copy()
            for prev_h, prev_cal in self.horizon_preds.items():
                if HORIZONS.get(prev_h, 0) < HORIZONS[horizon]:
                    mtf_col  = f"mtf_dacc_{prev_h.replace(' ', '_')}"
                    mtf_col2 = f"mtf_macc_{prev_h.replace(' ', '_')}"
                    feat_with_mtf[mtf_col]  = prev_cal.get("dir_acc", 0.5)
                    feat_with_mtf[mtf_col2] = prev_cal.get("avg_member_acc", 0.5)

            selected = self.selected_feats.get(horizon, list(feat_with_mtf.columns))
            selected = [f for f in selected if f in feat_with_mtf.columns]

            feat_clean = feat_with_mtf[selected].dropna()
            if feat_clean.empty:
                continue

            last_row_full = feat_clean.iloc[-1].values
            # Build a name→index map for the current feature set
            feat_name_to_idx = {name: i for i, name in enumerate(selected)}

            # L1 predictions
            l1_preds = []
            model_weights = []
            for m in members:
                # Always resolve indices by feature NAME to handle MTF columns
                # that exist at train time but not at prediction time
                feat_names_m = m.get("feat_names", [])
                if feat_names_m:
                    # Keep only names that exist in current feature set
                    valid = [(name, feat_name_to_idx[name])
                             for name in feat_names_m
                             if name in feat_name_to_idx]
                    if not valid:
                        # fallback: use all features
                        last_row_bag = last_row_full.reshape(1, -1)
                    else:
                        valid_names, valid_idx = zip(*valid)
                        last_row_bag = last_row_full[list(valid_idx)].reshape(1, -1)
                else:
                    # Legacy fallback using stored feat_idx
                    feat_idx = m["feat_idx"]
                    safe_idx = [i for i in feat_idx if i < len(last_row_full)]
                    last_row_bag = last_row_full[safe_idx].reshape(1, -1)

                try:
                    Xs = m["scaler"].transform(last_row_bag)
                    if m["is_cls"]:
                        proba = m["model"].predict_proba(Xs)
                        pred  = float(proba[0, 1] * 2 - 1)
                    else:
                        pred = float(m["model"].predict(Xs)[0])
                    l1_preds.append(pred)
                    model_weights.append(m.get("l2_weight", 1.0))
                except Exception:
                    pass

            if not l1_preds:
                continue

            l1_preds = np.array(l1_preds)
            model_weights = np.array(model_weights)

            # L2 meta-learner with adaptive weighting — pad/trim to match trained L2 shape
            l2 = self.l2_models[horizon]
            n_l2_feats = l2.coef_.shape[0] if hasattr(l2, "coef_") else len(l1_preds)
            if len(l1_preds) < n_l2_feats:
                # Pad with zeros if fewer models survived prediction than trained
                pad = np.zeros(n_l2_feats - len(l1_preds))
                l1_preds     = np.concatenate([l1_preds, pad])
                model_weights = np.concatenate([model_weights,
                                                np.ones(n_l2_feats - len(model_weights))])
            elif len(l1_preds) > n_l2_feats:
                l1_preds      = l1_preds[:n_l2_feats]
                model_weights = model_weights[:n_l2_feats]

            meta_in = (l1_preds * model_weights).reshape(1, -1)
            stacked_ret_raw = float(l2.predict(meta_in)[0])

            # ── Extreme-prediction shrinkage ────────────────────────────────
            # Pull predictions that escape the training distribution back
            # toward a plausible range, but DON'T hard-cap — the model should
            # be allowed to predict big moves on volatile stocks.
            #
            # Strategy:
            #   1. Soft log-compression when prediction exceeds training max
            #   2. Volatility-scaled cap: max return = 3× annualized vol × sqrt(horizon_years)
            #      This lets WULF (120% vol) predict ±360% on 1Y but stops
            #      JNJ (15% vol) from predicting more than ±45% on 1Y.
            #   3. No arbitrary hard caps — accuracy > caution.

            HORIZON_YEARS = {
                "3 Day":     3 / 252,
                "1 Week":    5 / 252,
                "1 Month":   21 / 252,
                "1 Quarter": 63 / 252,
                "1 Year":    1.0,
            }
            h_years = HORIZON_YEARS.get(horizon, 0.25)

            # Compute annualized volatility from recent data
            try:
                log_rets = np.log(df["Close"] / df["Close"].shift(1)).dropna()
                ann_vol = float(log_rets.iloc[-63:].std() * np.sqrt(252))
            except Exception:
                ann_vol = 0.30  # fallback ~30% vol

            # Volatility-scaled cap: 3× vol scaled by sqrt(horizon)
            # This reflects that longer horizons have wider plausible ranges
            vol_cap = max(0.10, 3.0 * ann_vol * np.sqrt(h_years))

            cal_shrink = self.calibration.get(horizon, {})
            tgt_max = cal_shrink.get("target_max_abs", None)
            stacked_ret = stacked_ret_raw
            if tgt_max is not None and tgt_max > 0:
                # Soft shrinkage: log-compress beyond training max
                upper_sane =  tgt_max * 1.2  # allow 20% extrapolation headroom
                lower_sane = -tgt_max * 1.2
                if stacked_ret_raw > upper_sane:
                    overshoot = stacked_ret_raw - upper_sane
                    stacked_ret = upper_sane + 0.25 * np.log1p(overshoot / upper_sane) * upper_sane
                elif stacked_ret_raw < lower_sane:
                    overshoot = lower_sane - stacked_ret_raw
                    stacked_ret = lower_sane - 0.25 * np.log1p(overshoot / abs(lower_sane)) * abs(lower_sane)
                stacked_ret = float(stacked_ret)

            # Volatility-scaled cap: only clips truly impossible predictions
            stacked_ret = float(np.clip(stacked_ret, -vol_cap, vol_cap))

            # Convert from log-return to percentage return for display/output
            # Model trains on log-returns, but users see pct returns and prices
            stacked_ret_log = stacked_ret  # keep log version for internal use
            stacked_ret = float(np.exp(stacked_ret) - 1.0)  # convert: pct = exp(log) - 1
            stacked_ret_raw = float(np.exp(stacked_ret_raw) - 1.0)

            std_preds  = float(np.std(l1_preds))
            agree_frac = float(np.mean(np.sign(l1_preds) == np.sign(stacked_ret)))

            # ── Confidence score (regime-aware) ─────────────────────────────
            cal       = self.calibration[horizon]
            std_norm  = min(std_preds / (cal["pred_std_p90"] + 1e-9), 1.0)
            avg_ma    = cal.get("avg_member_acc", cal["dir_acc"])
            raw_conf  = 0.45 * agree_frac + 0.30 * (1.0 - std_norm) + 0.25 * avg_ma
            blended   = 0.5 * raw_conf + 0.5 * cal["dir_acc"]

            iso = self.conf_calibrator.get(horizon)
            if iso is not None:
                try:
                    confidence = float(iso.predict([blended])[0]) * 100
                except Exception:
                    confidence = blended * 100
            else:
                confidence = blended * 100

            # Regime-aware adjustment: reduce confidence in bear or
            # sideways regimes since model was likely trained on more bull data
            if self.regime_cache:
                regime = self.regime_cache.get("label", "Sideways")
                score  = self.regime_cache.get("score_norm", 0.5)
                if regime == "Bear":
                    confidence *= 0.80    # 20% confidence penalty in bear markets
                elif regime == "Sideways":
                    confidence *= 0.90    # 10% penalty in sideways (noisy)
                # Additional penalty if regime score is extreme (very bearish)
                if score < 0.25:
                    confidence *= 0.85    # extra 15% penalty if deeply bearish

            # ── Earnings proximity: suppress confidence near earnings ────────
            # Earnings events inject unpredictable jumps; the model can't know
            # what the report will say, so reduce confidence within ±7 days.
            if self.earnings_data and isinstance(self.earnings_data, dict):
                dte = self.earnings_data.get("days_to_next_earnings")
                if dte is not None and not (isinstance(dte, float) and np.isnan(dte)):
                    abs_dte = abs(float(dte))
                    if abs_dte <= 3:
                        confidence *= 0.65   # very close to earnings — high uncertainty
                    elif abs_dte <= 7:
                        confidence *= 0.80   # within a week of earnings

            # Apply learned confidence adjustments from model improvement analysis
            try:
                adjustments = self._load_learned_adjustments()
                confidence_bias = adjustments.get("confidence_bias", 0.0)
                if confidence_bias != 0:
                    confidence += confidence_bias

                horizon_weights = adjustments.get("horizon_weights", {})
                if horizon in horizon_weights:
                    h_adj = horizon_weights[horizon]
                    confidence += h_adj * 100  # Convert to percentage points

                regime_adjustments = adjustments.get("regime_adjustments", {})
                if self.regime_cache and regime in regime_adjustments:
                    r_adj = regime_adjustments[regime]
                    confidence *= (1 + r_adj)
            except Exception:
                pass  # If adjustments fail to load, continue with unadjusted confidence

            # ── Extremity check: distinguish EXTRAPOLATION from LARGE moves ──
            # A "large" prediction that's within what training saw for this
            # stock is legitimate — don't punish the model for correctly
            # flagging a volatile asset. Only penalize predictions that go
            # BEYOND the training distribution (true extrapolation), since
            # those are where the model is guessing blind.
            abs_ret = abs(stacked_ret)
            tgt_max_abs = cal.get("target_max_abs", 1.0)
            tgt_std     = cal.get("target_std", 0.1)

            # How far outside training are we? 1.0 = at max, >1.0 = extrapolation
            extrapolation_ratio = abs_ret / (tgt_max_abs + 1e-9)

            if extrapolation_ratio > 1.0:
                # Beyond anything training ever saw — model is extrapolating blindly
                over = extrapolation_ratio - 1.0
                confidence *= max(0.40, 1.0 - 0.5 * over)
            elif extrapolation_ratio > 0.75:
                # Approaching the edge — mild penalty since model has few samples
                confidence *= 0.90

            # Shrinkage already handled upstream; also catch when member
            # agreement is low on big predictions (disagreement = uncertainty)
            if abs_ret > 2 * tgt_std and agree_frac < 0.60:
                confidence *= 0.80

            # Only flag as "extreme" (UI warning) for true extrapolation
            is_extreme = extrapolation_ratio > 1.0

            # ── Horizon-aware confidence ceiling ────────────────────────────
            # Long horizons have fewer independent validation samples, so
            # isotonic calibration is less reliable at the top. We cap the
            # reported confidence to reflect genuine calibration uncertainty.
            horizon_ceiling = {
                "3 Day":     93,
                "1 Week":    90,
                "1 Month":   87,
                "1 Quarter": 84,
                "1 Year":    80,
            }.get(horizon, 88)
            confidence = min(confidence, horizon_ceiling)

            confidence = max(20.0, min(95.0, round(confidence, 1)))

            # ── Learning Engine: apply all continuous learning adjustments ───
            # Per-stock adaptive confidence, regime weighting, direction bias,
            # and quality gate — all based on historical scored predictions.
            _regime_label = "Unknown"
            if self.regime_cache:
                _regime_label = self.regime_cache.get("label", "Unknown")
            suppress_signal = False
            suppress_reason = ""
            learning_adj = {}
            try:
                learning_result = apply_learning(
                    symbol=self.symbol,
                    regime=_regime_label,
                    horizon=horizon,
                    raw_confidence=confidence,
                    predicted_return=stacked_ret,
                    ensemble_agreement=agree_frac,
                )
                confidence = learning_result["adjusted_confidence"]
                suppress_signal = learning_result["suppress"]
                suppress_reason = learning_result["suppress_reason"]
                learning_adj = learning_result["adjustments"]
            except Exception:
                pass  # learning engine failure is non-fatal

            # Prediction interval
            reg_preds = np.array([l1_preds[i] for i, m in enumerate(members)
                                  if not m["is_cls"] and not m.get("is_radj", False)])
            p10_ret = float(np.percentile(reg_preds, 10)) if len(reg_preds) > 0 else stacked_ret * 0.5
            p90_ret = float(np.percentile(reg_preds, 90)) if len(reg_preds) > 0 else stacked_ret * 1.5

            results[horizon] = {
                "predicted_return":   round(stacked_ret, 4),
                "predicted_return_raw": round(stacked_ret_raw, 4),
                "predicted_price":    round(current_price * (1 + stacked_ret), 2),
                "current_price":      round(current_price, 2),
                "confidence":         confidence,
                "confidence_label":   _conf_label(confidence),
                "ensemble_agreement": round(agree_frac * 100, 1),
                "interval_low":       round(current_price * (1 + p10_ret), 2),
                "interval_high":      round(current_price * (1 + p90_ret), 2),
                "val_dir_accuracy":   round(cal["dir_acc"] * 100, 1),
                "all_preds":          l1_preds,
                "n_models":           len(members),
                "n_dropped":          cal.get("n_dropped", 0),
                "avg_member_acc":     round(cal.get("avg_member_acc", 0.5) * 100, 1),
                "is_extreme":         is_extreme,
                "suppress":           suppress_signal,
                "suppress_reason":    suppress_reason,
                "learning_adjustments": learning_adj,
            }

        return results

    # ── Feature importance ───────────────────────────────────────────────────

    def feature_importance(self, horizon: str = "1 Month") -> pd.DataFrame:
        if horizon not in self.l1_members:
            return pd.DataFrame()

        xgb_members = [m for m in self.l1_members[horizon]
                       if "xgb" in m["name"] and not m["is_cls"]]
        if not xgb_members:
            return pd.DataFrame()

        selected = self.selected_feats.get(horizon, self.feature_names)

        feat_imp = {}
        for m in xgb_members:
            fi = m["model"].feature_importances_
            bag_names = [selected[i] for i in m["feat_idx"] if i < len(selected)]
            for name, imp in zip(bag_names, fi):
                if name not in feat_imp:
                    feat_imp[name] = []
                feat_imp[name].append(imp)

        rows = [(name, np.mean(imps)) for name, imps in feat_imp.items()]
        return (
            pd.DataFrame(rows, columns=["feature", "importance"])
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
            .head(20)
        )

    def get_feature_importance(self) -> dict:
        """
        Return feature importance as dict for prediction logging.
        Returns {horizon: {feature_name: importance_score, ...}, ...}
        """
        result = {}
        for horizon in self.l1_members:
            try:
                fi_df = self.feature_importance(horizon)
                if not fi_df.empty:
                    result[horizon] = dict(zip(fi_df["feature"], fi_df["importance"]))
            except Exception:
                pass
        return result

    def shap_importance(self, df: pd.DataFrame, market_ctx: Optional[dict],
                        horizon: str = "1 Month") -> pd.DataFrame:
        if not HAS_SHAP or horizon not in self.l1_members:
            return pd.DataFrame()

        features_all = self._build_features(df, market_ctx, self.fundamentals)
        selected     = self.selected_feats.get(horizon, list(features_all.columns))
        selected     = [f for f in selected if f in features_all.columns]
        feat_clean   = features_all[selected].dropna()

        if len(feat_clean) < 10:
            return pd.DataFrame()

        xgb_m = next((m for m in self.l1_members[horizon]
                       if "xgb" in m["name"] and not m["is_cls"]), None)
        if xgb_m is None:
            return pd.DataFrame()

        try:
            n_sample = min(SHAP_SAMPLE, len(feat_clean))
            sample   = feat_clean.iloc[-n_sample:].values[:, xgb_m["feat_idx"]]
            Xs       = xgb_m["scaler"].transform(sample)
            explainer = shap.TreeExplainer(xgb_m["model"])
            sv        = explainer.shap_values(Xs)
            mean_abs  = np.abs(sv).mean(axis=0)
            bag_names = [selected[i] for i in xgb_m["feat_idx"] if i < len(selected)]
            return (
                pd.DataFrame({"feature": bag_names[:len(mean_abs)],
                               "shap_importance": mean_abs})
                .sort_values("shap_importance", ascending=False)
                .reset_index(drop=True)
                .head(20)
            )
        except Exception:
            return pd.DataFrame()

    # ── Calibration report ───────────────────────────────────────────────────

    def calibration_report(self) -> pd.DataFrame:
        rows = []
        for h, cal in self.calibration.items():
            rows.append({
                "Horizon":          h,
                "Val Dir. Acc.":    f"{cal['dir_acc']*100:.1f}%",
                "Mean Abs. Error":  f"{cal['mean_abs_err']*100:.2f}%",
                "Ensemble Std P50": f"{cal['pred_std_p50']*100:.2f}%",
                "L1 Models":        cal.get("n_models", "?"),
                "Dropped":          cal.get("n_dropped", 0),
                "Avg Member Acc":   f"{cal.get('avg_member_acc', 0)*100:.1f}%",
            })
        return pd.DataFrame(rows)

    # ── Model performance report ─────────────────────────────────────────────

    def model_performance_report(self, horizon: str = "1 Month") -> pd.DataFrame:
        perf = self.model_perf.get(horizon, {})
        if not perf:
            return pd.DataFrame()
        rows = [{"Model": name, "Dir Accuracy": f"{acc*100:.1f}%",
                 "Status": "Active" if acc >= MODEL_PERF_THRESHOLD else "Dropped"}
                for name, acc in sorted(perf.items(), key=lambda x: -x[1])]
        return pd.DataFrame(rows)


# ─── Persist / reload ────────────────────────────────────────────────────────

def save_predictor(predictor: StockPredictor, symbol: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(predictor, os.path.join(MODEL_DIR, f"{symbol.upper()}.pkl"))


def load_predictor(symbol: str) -> Optional[StockPredictor]:
    path = os.path.join(MODEL_DIR, f"{symbol.upper()}.pkl")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None
