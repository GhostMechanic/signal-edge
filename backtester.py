"""
backtester.py  –  Walk-Forward Backtesting Engine  (v3)
---------------------------------------------------------
Uses rolling train/test windows to simulate true out-of-sample performance.

Methodology:
  • TRAIN window  : 2 years (504 trading days)  — model learns patterns
  • TEST  window  : 3 months (63 trading days)  — model is evaluated
  • STEP  size    : 1 month  (21 trading days)  — window advances each step
  • Scaler is fit ONLY on training data in each window (no data leakage)
  • Level-1 ensemble: XGBoost (fast variant) + LightGBM (if available)
  • Level-2 meta: Ridge trained on val-set OOF predictions
  • Regime features included for consistent architecture with main model
  • Results aggregated across all windows → honest directional accuracy
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Optional

from data_fetcher import engineer_features, create_targets, HORIZONS
from regime_detector import regime_features

# Optional LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ─── Walk-forward parameters ──────────────────────────────────────────────────
TRAIN_DAYS = 504   # 2 years of training data per window
TEST_DAYS  = 63    # 3-month test window
STEP_DAYS  = 21    # advance 1 month per step

# Fast/light params for backtesting (fewer estimators = reasonable speed)
XGB_BT_VARIANTS = [
    dict(n_estimators=200, max_depth=4, learning_rate=0.06,
         subsample=0.80, colsample_bytree=0.80, reg_alpha=0.1,
         reg_lambda=1.0, min_child_weight=5, tree_method="hist",
         early_stopping_rounds=20, eval_metric="rmse",
         verbosity=0, random_state=42),
    dict(n_estimators=180, max_depth=4, learning_rate=0.07,
         subsample=0.70, colsample_bytree=0.75, reg_alpha=0.2,
         reg_lambda=0.8, min_child_weight=4, tree_method="hist",
         early_stopping_rounds=20, eval_metric="rmse",
         verbosity=0, random_state=7),
]

LGB_BT_VARIANTS = [
    dict(n_estimators=200, max_depth=4, learning_rate=0.06,
         num_leaves=31, subsample=0.80, colsample_bytree=0.80,
         reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
         verbosity=-1, random_state=42),
] if HAS_LGB else []

# Direction classifier for backtesting
from xgboost import XGBClassifier

XGB_CLS_BT = dict(
    n_estimators=200, max_depth=4, learning_rate=0.06,
    subsample=0.80, colsample_bytree=0.80,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
    tree_method="hist", random_state=55,
    use_label_encoder=False, eval_metric="logloss",
    early_stopping_rounds=20, verbosity=0,
)

WEIGHT_DECAY_BT = 2.5   # sample weight decay for backtesting

def _make_weights_bt(n: int) -> np.ndarray:
    w = np.exp(np.linspace(0, np.log(WEIGHT_DECAY_BT), n))
    return w / w.mean()


# ─── Walk-forward engine ──────────────────────────────────────────────────────

def _walk_forward_horizon(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    horizon: str,
    progress_callback=None,
    cb_offset: float = 0.0,
    cb_span: float = 1.0,
) -> dict:
    """
    Run walk-forward backtest for a single horizon using mini stacked ensemble.
    Returns aggregated metrics + per-window detail.
    """
    col    = f"target_{horizon}"
    merged = pd.concat([features, targets[col]], axis=1).dropna()
    n      = len(merged)

    if n < TRAIN_DAYS + TEST_DAYS:
        return {"error": f"Need at least {TRAIN_DAYS + TEST_DAYS} rows for walk-forward."}

    feature_cols = list(features.columns)
    windows      = []
    start        = TRAIN_DAYS

    # Count windows for progress reporting
    total_windows = max(1, (n - TRAIN_DAYS - TEST_DAYS) // STEP_DAYS + 1)

    while start + TEST_DAYS <= n:
        tr_sl  = merged.iloc[start - TRAIN_DAYS : start]
        te_sl  = merged.iloc[start : start + TEST_DAYS]

        X_tr   = tr_sl[feature_cols].values
        y_tr   = tr_sl[col].values
        X_te   = te_sl[feature_cols].values
        y_te   = te_sl[col].values
        d_te   = te_sl.index

        # Val split inside training window (last 15%) with 5-day purge gap
        val_split  = int(len(X_tr) * 0.85)
        purge_end  = min(val_split + 5, len(X_tr))

        X_tr_real = X_tr[:val_split]
        y_tr_real = y_tr[:val_split]
        X_va_real = X_tr[purge_end:]
        y_va_real = y_tr[purge_end:]

        if len(X_va_real) < 5:
            X_va_real = X_tr[val_split:]
            y_va_real = y_tr[val_split:]

        # Sample weights (exponential decay)
        w_tr = _make_weights_bt(len(X_tr_real))

        # Direction labels
        y_tr_dir = (y_tr_real > 0).astype(int)
        y_va_dir = (y_va_real > 0).astype(int)

        # ── Level-1 ensemble ──────────────────────────────────────────────────
        l1_val_preds = []
        l1_te_preds  = []

        for v in XGB_BT_VARIANTS:
            sc = RobustScaler()
            Xs_tr = sc.fit_transform(X_tr_real)
            Xs_va = sc.transform(X_va_real)
            Xs_te = sc.transform(X_te)
            mdl = XGBRegressor(**v)
            mdl.fit(Xs_tr, y_tr_real,
                    eval_set=[(Xs_va, y_va_real)],
                    sample_weight=w_tr,
                    verbose=False)
            l1_val_preds.append(mdl.predict(Xs_va))
            l1_te_preds.append(mdl.predict(Xs_te))

        if HAS_LGB:
            for v in LGB_BT_VARIANTS:
                sc = RobustScaler()
                Xs_tr = sc.fit_transform(X_tr_real)
                Xs_va = sc.transform(X_va_real)
                Xs_te = sc.transform(X_te)
                mdl = lgb.LGBMRegressor(**v)
                cb = [lgb.early_stopping(20, verbose=False),
                      lgb.log_evaluation(period=-1)]
                mdl.fit(Xs_tr, y_tr_real,
                        eval_set=[(Xs_va, y_va_real)],
                        sample_weight=w_tr,
                        callbacks=cb)
                l1_val_preds.append(mdl.predict(Xs_va))
                l1_te_preds.append(mdl.predict(Xs_te))

        # Direction classifier as extra L1 member
        try:
            sc_cls = RobustScaler()
            Xs_tr_c = sc_cls.fit_transform(X_tr_real)
            Xs_va_c = sc_cls.transform(X_va_real)
            Xs_te_c = sc_cls.transform(X_te)
            cls_p = {k: v for k, v in XGB_CLS_BT.items()
                     if k not in ("eval_metric","use_label_encoder","early_stopping_rounds","verbosity")}
            cls = XGBClassifier(eval_metric="logloss", use_label_encoder=False,
                                early_stopping_rounds=20, verbosity=0, **cls_p)
            cls.fit(Xs_tr_c, y_tr_dir,
                    eval_set=[(Xs_va_c, y_va_dir)],
                    sample_weight=w_tr,
                    verbose=False)
            # Convert P(up) to pseudo-return scale for stacking
            l1_val_preds.append(cls.predict_proba(Xs_va_c)[:, 1] * 2 - 1)
            l1_te_preds.append(cls.predict_proba(Xs_te_c)[:, 1] * 2 - 1)
        except Exception:
            pass

        # ── Level-2 meta-learner ──────────────────────────────────────────────
        meta_va_X = np.column_stack(l1_val_preds)
        meta_te_X = np.column_stack(l1_te_preds)
        l2 = Ridge(alpha=1.0)
        l2.fit(meta_va_X, y_va_real)

        preds   = l2.predict(meta_te_X)
        dir_acc = float(np.mean(np.sign(preds) == np.sign(y_te)))

        # Detect market regime in this test window
        regime = _detect_regime(te_sl[col])

        windows.append({
            "start":      str(d_te[0].date()),
            "end":        str(d_te[-1].date()),
            "regime":     regime,
            "dir_acc":    dir_acc,
            "preds":      preds,
            "actuals":    y_te,
            "dates":      d_te,
            "n_trades":   int(np.sum(preds > 0)),
        })

        if progress_callback:
            done = len(windows) / total_windows
            progress_callback(cb_offset + done * cb_span, f"Backtesting {horizon}… ({len(windows)}/{total_windows} windows)")

        start += STEP_DAYS

    if not windows:
        return {"error": "No windows completed."}

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    all_preds   = np.concatenate([w["preds"]   for w in windows])
    all_actuals = np.concatenate([w["actuals"] for w in windows])
    all_dates   = pd.DatetimeIndex(np.concatenate([w["dates"] for w in windows]))

    dir_acc_overall = float(np.mean(np.sign(all_preds) == np.sign(all_actuals)))

    # Simulated strategy: go long when model predicts positive return
    strategy_rets = np.where(all_preds > 0, all_actuals, 0.0)
    bh_rets       = all_actuals

    days = HORIZONS[horizon]
    ann  = 252 / days

    def sharpe(r):
        r = np.array(r)
        return float(np.mean(r) / (r.std() + 1e-9) * np.sqrt(ann))

    strat_sharpe = sharpe(strategy_rets)
    bh_sharpe    = sharpe(bh_rets)

    long_mask  = all_preds > 0
    win_rate   = float(np.mean(all_actuals[long_mask] > 0)) if long_mask.sum() > 0 else 0.0

    strat_cum = (1 + pd.Series(strategy_rets, index=all_dates)).cumprod()
    bh_cum    = (1 + pd.Series(bh_rets,       index=all_dates)).cumprod()

    roll_max  = strat_cum.cummax()
    max_dd    = float(((strat_cum - roll_max) / roll_max).min())

    gw = strategy_rets[strategy_rets > 0].sum()
    gl = abs(strategy_rets[strategy_rets < 0].sum())
    pf = gw / gl if gl > 0 else float("inf")

    # Per-regime accuracy
    regime_acc = {}
    for reg in ["Bull", "Bear", "Sideways"]:
        w_reg = [w for w in windows if w["regime"] == reg]
        if w_reg:
            regime_acc[reg] = round(float(np.mean([w["dir_acc"] for w in w_reg])) * 100, 1)

    # Accuracy trend (rolling 5-window average)
    window_accs = [w["dir_acc"] for w in windows]
    acc_trend   = pd.Series(window_accs).rolling(min(5, len(window_accs))).mean().tolist()

    return {
        "horizon":         horizon,
        "n_windows":       len(windows),
        "test_start":      windows[0]["start"],
        "test_end":        windows[-1]["end"],
        "dir_accuracy":    round(dir_acc_overall * 100, 1),
        "win_rate":        round(win_rate * 100, 1),
        "strat_sharpe":    round(strat_sharpe, 2),
        "bh_sharpe":       round(bh_sharpe, 2),
        "max_drawdown":    round(max_dd * 100, 1),
        "profit_factor":   round(min(pf, 99.9), 2),
        "regime_accuracy": regime_acc,
        "window_details":  windows,
        "acc_trend":       acc_trend,
        "strategy_cum":    strat_cum,
        "bh_cum":          bh_cum,
        "all_preds":       all_preds,
        "all_actuals":     all_actuals,
        "all_dates":       all_dates,
    }


def _detect_regime(returns: pd.Series) -> str:
    """Classify the test window as Bull, Bear, or Sideways."""
    cum = float((1 + returns).prod() - 1)
    if cum >  0.03: return "Bull"
    if cum < -0.03: return "Bear"
    return "Sideways"


# ─── Public API ───────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    market_ctx: Optional[dict] = None,
    progress_callback=None,
) -> dict:
    """Run walk-forward backtest for all horizons with regime features."""
    spy = market_ctx.get("spy") if market_ctx else None
    vix = market_ctx.get("vix") if market_ctx else None
    sec = market_ctx.get("sector") if market_ctx else None

    base_feats = engineer_features(df, spy_close=spy, vix_close=vix, sector_close=sec)
    reg_feats  = regime_features(df, spy_close=spy, vix_close=vix)
    features   = pd.concat([base_feats, reg_feats], axis=1)
    features   = features.loc[:, ~features.columns.duplicated()]
    targets    = create_targets(df)
    results  = {}
    n        = len(HORIZONS)

    for i, horizon in enumerate(HORIZONS.keys()):
        results[horizon] = _walk_forward_horizon(
            features, targets, horizon,
            progress_callback=progress_callback,
            cb_offset=i / n,
            cb_span=1 / n,
        )

    if progress_callback:
        progress_callback(1.0, "Backtest complete.")

    return results


def backtest_summary_df(results: dict) -> pd.DataFrame:
    rows = []
    for horizon, r in results.items():
        if "error" in r:
            continue
        regime_str = "  |  ".join(
            f"{k}: {v}%" for k, v in r.get("regime_accuracy", {}).items()
        )
        rows.append({
            "Horizon":              horizon,
            "Windows Tested":       r["n_windows"],
            "Test Period":          f"{r['test_start']} → {r['test_end']}",
            "Directional Acc.":     f"{r['dir_accuracy']}%",
            "Win Rate":             f"{r['win_rate']}%",
            "Sharpe (Strategy)":    r["strat_sharpe"],
            "Sharpe (B&H)":         r["bh_sharpe"],
            "Max Drawdown":         f"{r['max_drawdown']}%",
            "Profit Factor":        r["profit_factor"],
            "Accuracy by Regime":   regime_str,
        })
    return pd.DataFrame(rows)
