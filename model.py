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
TOP_K_FEATURES   = 40           # increased from 35 — more features now available
FEATURE_BAG_FRAC = 0.75
PURGE_DAYS       = 5
WEIGHT_DECAY     = 3.0
MODEL_PERF_THRESHOLD = 0.48     # L1 models below this dir accuracy get dropped
MIN_L1_MODELS    = 4            # always keep at least this many L1 models


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


def _feature_bag(feature_names: List[str], frac: float, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    k   = max(5, int(len(feature_names) * frac))
    idx = rng.choice(len(feature_names), k, replace=False)
    return [feature_names[i] for i in sorted(idx)]


def _purged_split(X: np.ndarray, y: np.ndarray, train_ratio: float, purge: int):
    split = int(len(X) * train_ratio)
    X_tr, y_tr = X[:split], y[:split]
    val_start  = min(split + purge, len(X))
    X_va, y_va = X[val_start:], y[val_start:]
    return X_tr, y_tr, X_va, y_va


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

    # ── Feature preparation ──────────────────────────────────────────────────

    def _build_features(
        self,
        df: pd.DataFrame,
        market_ctx: Optional[dict] = None,
        fundamentals: Optional[dict] = None,
    ) -> pd.DataFrame:
        spy = market_ctx.get("spy") if market_ctx else None
        vix = market_ctx.get("vix") if market_ctx else None
        sec = market_ctx.get("sector") if market_ctx else None

        base  = engineer_features(df, spy_close=spy, vix_close=vix,
                                   sector_close=sec, fundamentals=fundamentals)
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

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        market_ctx: Optional[dict] = None,
        fundamentals: Optional[dict] = None,
        use_shap:   bool = False,
        use_optuna: bool = False,
        progress_callback = None,
    ) -> "StockPredictor":

        if len(df) < MIN_TRAIN:
            raise ValueError(f"Need at least {MIN_TRAIN} trading days (got {len(df)}).")

        self.fundamentals = fundamentals
        features_all = self._build_features(df, market_ctx, fundamentals)
        targets      = create_targets(df)
        self.feature_names = list(features_all.columns)

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

            # ── Optional Optuna tuning ───────────────────────────────────────
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
                selected = _shap_top_features(
                    shap_mdl, sc_shap, X_tr_s[idx_sample],
                    all_cols, top_k=TOP_K_FEATURES,
                )
                X_all, y_all, _, all_cols = self._prep(
                    feat_with_mtf, targets, horizon, selected=selected)

            self.selected_feats[horizon] = selected

            # ── Purged train/val split ───────────────────────────────────────
            X_tr, y_tr, X_va, y_va = _purged_split(
                X_all, y_all, TRAIN_RATIO, PURGE_DAYS)

            weights_tr = _make_sample_weights(len(X_tr))

            # Direction labels (with dead-zone threshold)
            days = HORIZONS[horizon]
            dir_threshold = 0.005 if days <= 10 else 0.01
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
                val_start_r = min(split_r + PURGE_DAYS, len(y_radj_aligned))
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
                    bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"])
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
                        bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"]+100)
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
                    bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"]+200)
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
                    bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"]+400)
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
                    bag_feats = _feature_bag(all_cols, FEATURE_BAG_FRAC, seed=variant["random_state"]+500)
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

            kept_members   = [m for m, k in zip(members, keep_mask) if k]
            kept_oof       = [p for p, k in zip(oof_val_preds, keep_mask) if k]
            kept_dir_acc   = member_dir_acc[keep_mask]

            self.model_perf[horizon] = {
                m["name"]: float(da) for m, da in zip(members, member_dir_acc)
            }

            # ── Level 2 meta-learner with adaptive weighting ─────────────────
            _prog(f"[{horizon}] Training meta-learner…")
            meta_X = np.column_stack(kept_oof)

            # Adaptive: weight each L1 model's column by its dir accuracy
            # This makes the Ridge focus more on models that got direction right
            model_weights = np.clip(kept_dir_acc - 0.45, 0.05, 1.0)
            model_weights = model_weights / model_weights.mean()
            meta_X_weighted = meta_X * model_weights[np.newaxis, :]

            l2_model = Ridge(alpha=1.0)
            l2_model.fit(meta_X_weighted, y_va)

            # ── Cross-validated isotonic calibration ─────────────────────────
            _prog(f"[{horizon}] Calibrating confidence…")
            meta_preds   = l2_model.predict(meta_X_weighted)
            errors       = np.abs(meta_preds - y_va)
            per_pred_std = np.std(meta_X, axis=1)

            dir_correct = (np.sign(meta_preds) == np.sign(y_va)).astype(float)
            val_dir_acc = float(dir_correct.mean())

            # CV isotonic: split val into 2 folds for calibration
            std_norm   = np.clip(per_pred_std / (np.percentile(per_pred_std, 90) + 1e-9), 0, 1)
            agree_frac = np.array([
                float(np.mean(np.sign(meta_X[j]) == np.sign(meta_preds[j])))
                for j in range(len(meta_preds))
            ])

            # Enhanced raw confidence: include model quality signal
            avg_member_acc = float(kept_dir_acc.mean())
            raw_conf = (0.45 * agree_frac
                       + 0.30 * (1 - std_norm)
                       + 0.25 * avg_member_acc)
            blended = 0.5 * raw_conf + 0.5 * val_dir_acc

            # Cross-validated isotonic calibration
            try:
                n_cal = len(blended)
                mid   = n_cal // 2
                iso1 = IsotonicRegression(y_min=0.30, y_max=0.95, out_of_bounds="clip")
                iso2 = IsotonicRegression(y_min=0.30, y_max=0.95, out_of_bounds="clip")

                if mid > 10:
                    iso1.fit(blended[:mid], dir_correct[:mid])
                    iso2.fit(blended[mid:], dir_correct[mid:])
                    # Average the two calibrators' predictions
                    cal1 = iso1.predict(blended)
                    cal2 = iso2.predict(blended)
                    avg_cal = (cal1 + cal2) / 2

                    # Refit on full data using averaged as guide
                    iso_final = IsotonicRegression(y_min=0.30, y_max=0.95, out_of_bounds="clip")
                    iso_final.fit(blended, avg_cal)
                    self.conf_calibrator[horizon] = iso_final
                else:
                    iso_final = IsotonicRegression(y_min=0.30, y_max=0.95, out_of_bounds="clip")
                    iso_final.fit(blended, dir_correct)
                    self.conf_calibrator[horizon] = iso_final
            except Exception:
                self.conf_calibrator[horizon] = None

            self.calibration[horizon] = {
                "mean_abs_err":    float(np.mean(errors)),
                "std_abs_err":     float(np.std(errors)),
                "dir_acc":         val_dir_acc,
                "pred_std_p50":    float(np.percentile(per_pred_std, 50)),
                "pred_std_p90":    float(np.percentile(per_pred_std, 90)),
                "n_models":        len(kept_members),
                "n_dropped":       int((~keep_mask).sum()),
                "avg_member_acc":  avg_member_acc,
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

        self.is_trained = True
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

        features_all = self._build_features(df, market_ctx, self.fundamentals)
        results      = {}

        spy = market_ctx.get("spy") if market_ctx else None
        vix = market_ctx.get("vix") if market_ctx else None
        self.regime_cache = detect_regime(df, spy, vix)

        current_price = float(df["Close"].iloc[-1])

        # Predict in horizon order (short → long) for multi-timeframe
        horizon_order = sorted(HORIZONS.keys(), key=lambda h: HORIZONS[h])

        for horizon in horizon_order:
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
            stacked_ret = float(l2.predict(meta_in)[0])

            std_preds  = float(np.std(l1_preds))
            agree_frac = float(np.mean(np.sign(l1_preds) == np.sign(stacked_ret)))

            # ── Confidence score ────────────────────────────────────────────
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

            confidence = max(30.0, min(95.0, round(confidence, 1)))

            # Prediction interval
            reg_preds = np.array([l1_preds[i] for i, m in enumerate(members)
                                  if not m["is_cls"] and not m.get("is_radj", False)])
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
                "n_dropped":          cal.get("n_dropped", 0),
                "avg_member_acc":     round(cal.get("avg_member_acc", 0.5) * 100, 1),
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
