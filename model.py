"""
model.py  –  Stacked Ensemble Prediction Engine  (v4)
------------------------------------------------------
Architecture:
  Level 1:  XGBoost (5 variants) + LightGBM (3 models) + RandomForest (2 models)
            + XGBClassifier (2 direction models) = up to 12 L1 members
            Each with feature bagging (random 75% of features) for diversity.
            Time-weighted sample weighting (recent data matters more).
  Level 2:  Ridge meta-learner trained on L1 out-of-fold predictions.

Accuracy improvements in v4:
  • Exponential sample weighting (recent data weighted 3x more than oldest)
  • Feature bagging: each L1 model trained on random 75% of features
  • Direction classifiers as extra L1 members (probability → return signal)
  • Purged validation: 5-day gap between train and val to prevent leakage
  • Isotonic calibration of confidence scores using validation data
  • SHAP-based feature selection  (select top_k by mean |SHAP|)
  • Optuna hyperparameter tuning  (Bayesian optimisation, ~20 trials)

Confidence scoring:
  1. Collect all L1 model predictions for the latest row
  2. Low std → high ensemble agreement → high confidence
  3. Calibrate against validation directional accuracy (honest)
  4. Isotonic calibration for realism
  5. Clamp to [30, 95] to avoid extreme overconfidence

Python 3.9 compatible throughout.
"""

import warnings
warnings.filterwarnings("ignore")

import os
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
TOP_K_FEATURES   = 35           # increased from 30
FEATURE_BAG_FRAC = 0.75         # each L1 model sees 75% of features
PURGE_DAYS       = 5            # gap between train and val to prevent leakage
WEIGHT_DECAY     = 3.0          # recent data weighted 3x more than oldest


# ─── Level-1 model specs ──────────────────────────────────────────────────────
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

# Direction classifiers (XGBClassifier for direction prediction)
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _conf_label(conf: float) -> str:
    if conf >= 70: return "High"
    if conf >= 55: return "Medium"
    return "Low"


def _make_sample_weights(n: int, decay: float = WEIGHT_DECAY) -> np.ndarray:
    """Exponential sample weights: recent data weighted `decay`x more than oldest."""
    w = np.exp(np.linspace(0, np.log(decay), n))
    return w / w.mean()  # normalise so mean weight = 1


def _feature_bag(feature_names: List[str], frac: float, seed: int) -> List[str]:
    """Random subset of feature names for feature bagging."""
    rng = np.random.default_rng(seed)
    k   = max(5, int(len(feature_names) * frac))
    idx = rng.choice(len(feature_names), k, replace=False)
    return [feature_names[i] for i in sorted(idx)]


def _purged_split(X: np.ndarray, y: np.ndarray, train_ratio: float, purge: int):
    """Split with purge gap between train and validation."""
    split = int(len(X) * train_ratio)
    X_tr, y_tr = X[:split], y[:split]
    val_start  = min(split + purge, len(X))
    X_va, y_va = X[val_start:], y[val_start:]
    return X_tr, y_tr, X_va, y_va


def _train_l1_model(model, X_tr, y_tr, X_va, y_va, scaler, weights_tr=None):
    """Fit a single L1 model; returns fitted model + val preds."""
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
    else:
        # RandomForest — supports sample_weight in fit
        if weights_tr is not None:
            model.fit(Xs_tr, y_tr, sample_weight=weights_tr)
        else:
            model.fit(Xs_tr, y_tr)

    # Prediction: classifiers output probability, regressors output value
    if isinstance(model, XGBClassifier):
        proba = model.predict_proba(Xs_va)
        # Convert probability to pseudo-return: P(up) mapped to [-1, +1]
        val_preds = proba[:, 1] * 2 - 1
    else:
        val_preds = model.predict(Xs_va)

    return model, val_preds


# ─── Optuna tuning ────────────────────────────────────────────────────────────

def _optuna_tune_xgb(X_tr, y_tr, X_va, y_va, weights_tr=None,
                     n_trials: int = N_OPTUNA_TRIALS) -> dict:
    """Return best XGBoost params found by Optuna."""
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


# ─── SHAP feature selection ───────────────────────────────────────────────────

def _shap_top_features(
    model, scaler: RobustScaler,
    X_sample: np.ndarray,
    feature_names: List[str],
    top_k: int = TOP_K_FEATURES,
) -> List[str]:
    """Return top-k feature names by mean |SHAP| value."""
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


# ─── StockPredictor ───────────────────────────────────────────────────────────

class StockPredictor:
    """
    Two-level stacked ensemble predictor (v4).

    Level 1:  XGBoost (5) + LightGBM (3 if available) + RandomForest (2)
              + XGBClassifier (2 direction models)
              Each with feature bagging + exponential sample weighting.
    Level 2:  Ridge meta-learner on L1 out-of-fold predictions.
    """

    def __init__(self, symbol: str):
        self.symbol         = symbol.upper()
        self.l1_members     = {}    # horizon → list of {model, scaler, name, feat_idx}
        self.l2_models      = {}    # horizon → Ridge
        self.calibration    = {}    # horizon → calibration stats
        self.conf_calibrator = {}   # horizon → IsotonicRegression
        self.feature_names  = []
        self.selected_feats = {}    # horizon → selected feature names (post-SHAP)
        self.regime_cache   = None
        self.is_trained     = False

    # ── Feature preparation ───────────────────────────────────────────────────

    def _build_features(
        self,
        df: pd.DataFrame,
        market_ctx: Optional[dict] = None,
    ) -> pd.DataFrame:
        spy = market_ctx.get("spy") if market_ctx else None
        vix = market_ctx.get("vix") if market_ctx else None
        sec = market_ctx.get("sector") if market_ctx else None

        base  = engineer_features(df, spy_close=spy, vix_close=vix, sector_close=sec)
        reg   = regime_features(df, spy_close=spy, vix_close=vix)
        combined = pd.concat([base, reg], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined

    def _prep(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        horizon: str,
        selected: Optional[List[str]] = None,
    ):
        col    = f"target_{horizon}"
        cols   = selected if selected else list(features.columns)
        merged = pd.concat([features[cols], targets[col]], axis=1).dropna()
        X      = merged[cols].values
        y      = merged[col].values
        return X, y, merged.index, cols

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        market_ctx: Optional[dict] = None,
        use_shap:   bool = False,
        use_optuna: bool = False,
        progress_callback = None,
    ) -> "StockPredictor":

        if len(df) < MIN_TRAIN:
            raise ValueError(f"Need at least {MIN_TRAIN} trading days (got {len(df)}).")

        features_all = self._build_features(df, market_ctx)
        targets      = create_targets(df)
        self.feature_names = list(features_all.columns)

        # Count total training steps for progress bar
        n_cls = len(XGB_CLS_VARIANTS)
        n_l1_per_horizon = (len(XGB_VARIANTS)
                            + (len(LGB_VARIANTS) if HAS_LGB else 0)
                            + len(RF_VARIANTS)
                            + n_cls)
        total_steps = len(HORIZONS) * (n_l1_per_horizon + 2)  # +2 for L2 + calibration
        step        = 0

        def _prog(msg: str):
            nonlocal step
            if progress_callback:
                progress_callback(min(step / total_steps, 0.99), msg)
            step += 1

        for horizon in HORIZONS.keys():

            # ── Optional Optuna tuning ────────────────────────────────────────
            tuned_xgb_params = {}
            if use_optuna and HAS_OPTUNA:
                _prog(f"[{horizon}] Optuna tuning…")
                X_all, y_all, _, _ = self._prep(features_all, targets, horizon)
                X_tr_o, y_tr_o, X_va_o, y_va_o = _purged_split(
                    X_all, y_all, TRAIN_RATIO, PURGE_DAYS)
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

            X_all, y_all, _, all_cols = self._prep(features_all, targets, horizon)

            # ── Optional SHAP feature selection ───────────────────────────────
            selected = list(all_cols)
            if use_shap and HAS_SHAP and len(X_all) > SHAP_SAMPLE:
                _prog(f"[{horizon}] SHAP feature selection…")
                X_tr_s, y_tr_s, X_va_s, y_va_s = _purged_split(
                    X_all, y_all, TRAIN_RATIO, PURGE_DAYS)
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
                selected   = _shap_top_features(
                    shap_mdl, sc_shap, X_tr_s[idx_sample],
                    all_cols, top_k=TOP_K_FEATURES,
                )
                # Re-prep with selected features
                X_all, y_all, _, all_cols = self._prep(
                    features_all, targets, horizon, selected=selected)

            self.selected_feats[horizon] = selected

            # ── Purged train/val split ────────────────────────────────────────
            X_tr, y_tr, X_va, y_va = _purged_split(
                X_all, y_all, TRAIN_RATIO, PURGE_DAYS)

            # ── Sample weights (exponential decay) ────────────────────────────
            weights_tr = _make_sample_weights(len(X_tr))

            # ── Direction labels for classifiers ──────────────────────────────
            y_tr_dir = (y_tr > 0).astype(int)
            y_va_dir = (y_va > 0).astype(int)

            # ── Level 1 training ─────────────────────────────────────────────
            members        = []
            oof_val_preds  = []

            model_seed = 0

            # — XGBoost regressors —
            for i, variant in enumerate(XGB_VARIANTS):
                params = {**variant, **tuned_xgb_params}
                if tuned_xgb_params:
                    params["random_state"] = variant["random_state"]
                _prog(f"[{horizon}] XGB reg {i+1}/{len(XGB_VARIANTS)}…")

                # Feature bagging
                bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"])
                feat_idx  = [all_cols.index(f) for f in bag_feats]
                X_tr_bag  = X_tr[:, feat_idx]
                X_va_bag  = X_va[:, feat_idx]

                sc  = RobustScaler()
                mdl = XGBRegressor(
                    eval_metric="rmse", early_stopping_rounds=30,
                    verbosity=0, **params)
                mdl, vp = _train_l1_model(mdl, X_tr_bag, y_tr, X_va_bag, y_va, sc, weights_tr)
                members.append({"model": mdl, "scaler": sc, "name": f"xgb_{i}",
                                "feat_idx": feat_idx, "is_cls": False})
                oof_val_preds.append(vp)

            # — LightGBM regressors —
            if HAS_LGB:
                for i, variant in enumerate(LGB_VARIANTS):
                    _prog(f"[{horizon}] LGB reg {i+1}/{len(LGB_VARIANTS)}…")
                    bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"]+100)
                    feat_idx  = [all_cols.index(f) for f in bag_feats]
                    X_tr_bag  = X_tr[:, feat_idx]
                    X_va_bag  = X_va[:, feat_idx]

                    sc  = RobustScaler()
                    mdl = lgb.LGBMRegressor(**variant)
                    mdl, vp = _train_l1_model(mdl, X_tr_bag, y_tr, X_va_bag, y_va, sc, weights_tr)
                    members.append({"model": mdl, "scaler": sc, "name": f"lgb_{i}",
                                    "feat_idx": feat_idx, "is_cls": False})
                    oof_val_preds.append(vp)

            # — Random Forest regressors —
            for i, variant in enumerate(RF_VARIANTS):
                _prog(f"[{horizon}] RF reg {i+1}/{len(RF_VARIANTS)}…")
                bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"]+200)
                feat_idx  = [all_cols.index(f) for f in bag_feats]
                X_tr_bag  = X_tr[:, feat_idx]
                X_va_bag  = X_va[:, feat_idx]

                sc  = RobustScaler()
                mdl = RandomForestRegressor(n_jobs=-1, **variant)
                mdl, vp = _train_l1_model(mdl, X_tr_bag, y_tr, X_va_bag, y_va, sc, weights_tr)
                members.append({"model": mdl, "scaler": sc, "name": f"rf_{i}",
                                "feat_idx": feat_idx, "is_cls": False})
                oof_val_preds.append(vp)

            # — XGBoost direction classifiers —
            for i, variant in enumerate(XGB_CLS_VARIANTS):
                _prog(f"[{horizon}] XGB cls {i+1}/{len(XGB_CLS_VARIANTS)}…")
                bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"]+300)
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
                members.append({"model": mdl, "scaler": sc, "name": f"cls_{i}",
                                "feat_idx": feat_idx, "is_cls": True})
                oof_val_preds.append(vp)

            # ── Level 2 meta-learner ──────────────────────────────────────────
            _prog(f"[{horizon}] Training meta-learner…")
            meta_X   = np.column_stack(oof_val_preds)
            l2_model = Ridge(alpha=1.0)
            l2_model.fit(meta_X, y_va)

            # ── Calibration ──────────────────────────────────────────────────
            _prog(f"[{horizon}] Calibrating confidence…")
            meta_preds   = l2_model.predict(meta_X)
            errors       = np.abs(meta_preds - y_va)
            all_preds_va = np.column_stack(oof_val_preds)
            per_pred_std = np.std(all_preds_va, axis=1)

            dir_correct = (np.sign(meta_preds) == np.sign(y_va)).astype(float)
            val_dir_acc = float(dir_correct.mean())

            # Isotonic calibration: map raw confidence → actual accuracy
            std_norm   = np.clip(per_pred_std / (np.percentile(per_pred_std, 90) + 1e-9), 0, 1)
            agree_frac = np.array([
                float(np.mean(np.sign(all_preds_va[j]) == np.sign(meta_preds[j])))
                for j in range(len(meta_preds))
            ])
            raw_conf   = 0.6 * agree_frac + 0.4 * (1 - std_norm)
            blended    = 0.5 * raw_conf + 0.5 * val_dir_acc

            try:
                iso = IsotonicRegression(y_min=0.3, y_max=0.95, out_of_bounds="clip")
                iso.fit(blended, dir_correct)
                self.conf_calibrator[horizon] = iso
            except Exception:
                self.conf_calibrator[horizon] = None

            self.calibration[horizon] = {
                "mean_abs_err":  float(np.mean(errors)),
                "std_abs_err":   float(np.std(errors)),
                "dir_acc":       val_dir_acc,
                "pred_std_p50":  float(np.percentile(per_pred_std, 50)),
                "pred_std_p90":  float(np.percentile(per_pred_std, 90)),
                "n_models":      len(members),
            }

            self.l1_members[horizon] = members
            self.l2_models[horizon]  = l2_model

        self.is_trained = True
        if progress_callback:
            progress_callback(1.0, "Training complete.")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        df: pd.DataFrame,
        market_ctx: Optional[dict] = None,
    ) -> dict:
        if not self.is_trained:
            raise RuntimeError("Call .train() first.")

        features_all = self._build_features(df, market_ctx)
        results      = {}

        spy = market_ctx.get("spy") if market_ctx else None
        vix = market_ctx.get("vix") if market_ctx else None
        self.regime_cache = detect_regime(df, spy, vix)

        current_price = float(df["Close"].iloc[-1])

        for horizon, members in self.l1_members.items():
            selected = self.selected_feats.get(horizon, list(features_all.columns))
            selected = [f for f in selected if f in features_all.columns]

            feat_clean = features_all[selected].dropna()
            if feat_clean.empty:
                continue

            last_row_full = feat_clean.iloc[-1].values

            # L1 predictions (each model uses its own feature bag)
            l1_preds = []
            for m in members:
                feat_idx = m["feat_idx"]
                # Map feat_idx from all_cols to selected indices
                last_row_bag = last_row_full[feat_idx] if max(feat_idx) < len(last_row_full) else last_row_full
                last_row_bag = last_row_bag.reshape(1, -1)
                Xs = m["scaler"].transform(last_row_bag)

                if m["is_cls"]:
                    proba = m["model"].predict_proba(Xs)
                    pred  = float(proba[0, 1] * 2 - 1)
                else:
                    pred = float(m["model"].predict(Xs)[0])
                l1_preds.append(pred)

            l1_preds = np.array(l1_preds)

            # L2 meta-learner prediction
            l2        = self.l2_models[horizon]
            meta_in   = l1_preds.reshape(1, -1)
            stacked_ret = float(l2.predict(meta_in)[0])

            std_preds  = float(np.std(l1_preds))
            agree_frac = float(np.mean(np.sign(l1_preds) == np.sign(stacked_ret)))

            # ── Confidence score ─────────────────────────────────────────────
            cal       = self.calibration[horizon]
            std_norm  = min(std_preds / (cal["pred_std_p90"] + 1e-9), 1.0)
            raw_conf  = 0.6 * agree_frac + 0.4 * (1.0 - std_norm)
            blended   = 0.5 * raw_conf + 0.5 * cal["dir_acc"]

            # Apply isotonic calibration if available
            iso = self.conf_calibrator.get(horizon)
            if iso is not None:
                try:
                    confidence = float(iso.predict([blended])[0]) * 100
                except Exception:
                    confidence = blended * 100
            else:
                confidence = blended * 100

            confidence = max(30.0, min(95.0, round(confidence, 1)))

            # ── Prediction interval ──────────────────────────────────────────
            # Use only regressor preds for interval (classifiers are on different scale)
            reg_preds = np.array([l1_preds[i] for i, m in enumerate(members) if not m["is_cls"]])
            p10_ret = float(np.percentile(reg_preds, 10)) if len(reg_preds) > 0 else stacked_ret * 0.5
            p90_ret = float(np.percentile(reg_preds, 90)) if len(reg_preds) > 0 else stacked_ret * 1.5

            results[horizon] = {
                "predicted_return":   round(stacked_ret, 4),
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
            }

        return results

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self, horizon: str = "1 Month") -> pd.DataFrame:
        """Average feature importance across XGBoost L1 regressor members."""
        if horizon not in self.l1_members:
            return pd.DataFrame()

        xgb_members = [m for m in self.l1_members[horizon]
                       if "xgb" in m["name"] and not m["is_cls"]]
        if not xgb_members:
            return pd.DataFrame()

        selected = self.selected_feats.get(horizon, self.feature_names)

        # Each model may have different feature bags; aggregate by feature name
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

    def shap_importance(self, df: pd.DataFrame, market_ctx: Optional[dict],
                        horizon: str = "1 Month") -> pd.DataFrame:
        """Return SHAP-based feature importance for a horizon."""
        if not HAS_SHAP or horizon not in self.l1_members:
            return pd.DataFrame()

        features_all = self._build_features(df, market_ctx)
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

    # ── Calibration report ────────────────────────────────────────────────────

    def calibration_report(self) -> pd.DataFrame:
        rows = []
        for h, cal in self.calibration.items():
            rows.append({
                "Horizon":          h,
                "Val Dir. Acc.":    f"{cal['dir_acc']*100:.1f}%",
                "Mean Abs. Error":  f"{cal['mean_abs_err']*100:.2f}%",
                "Ensemble Std P50": f"{cal['pred_std_p50']*100:.2f}%",
                "L1 Models":        cal.get("n_models", "?"),
            })
        return pd.DataFrame(rows)


# ─── Persist / reload ─────────────────────────────────────────────────────────

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
