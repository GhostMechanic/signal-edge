"""
diag_importance.py — diagnose where the model is getting its edge (or isn't)
per ticker × horizon.

Trains a fresh model on a single (ticker, horizon) cell and reports:
  • Directional accuracy on a held-out 20% of data
  • Top 15 features by XGBoost importance
  • Accuracy WITHOUT a named feature group (ablation) — e.g. "what does
    accuracy look like if we remove all macro_* features?"

Useful for questions like "is the AAPL 1M regression after adding event
calendar features because those features are hurting or helping?"

Usage:
    python diag_importance.py AAPL "1 Month"
    python diag_importance.py AAPL "1 Month" --ablate macro
    python diag_importance.py AAPL "1 Month" --ablate sent_oil sent_gold
"""

from __future__ import annotations

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def _load_data(symbol: str):
    from data_fetcher import (
        fetch_stock_data, fetch_market_context, fetch_fundamentals,
        fetch_earnings_data, fetch_options_data, engineer_features,
        create_targets,
    )
    from backtester import regime_features

    df = fetch_stock_data(symbol, period="5y")
    if df is None or df.empty or len(df) < 252:
        raise ValueError(f"Insufficient data for {symbol}")

    import yfinance as yf
    sector = None
    try:
        sector = yf.Ticker(symbol).info.get("sector")
    except Exception:
        pass

    market_ctx = fetch_market_context(period="5y", sector=sector)
    fundamentals = fetch_fundamentals(symbol)
    earnings = fetch_earnings_data(symbol)
    current_price = float(df["Close"].iloc[-1])
    options = {}
    try:
        options = fetch_options_data(symbol, current_price) or {}
    except Exception:
        pass

    base_feats = engineer_features(
        df,
        spy_close=market_ctx.get("spy"),
        vix_close=market_ctx.get("vix"),
        sector_close=market_ctx.get("sector"),
        fundamentals=fundamentals,
        earnings_data=earnings,
        options_data=options,
        sentiment_ctx=market_ctx.get("sentiment"),
    )
    reg_feats = regime_features(df, spy_close=market_ctx.get("spy"), vix_close=market_ctx.get("vix"))
    features = pd.concat([base_feats, reg_feats], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    targets = create_targets(df)
    return features, targets, df


def _ablate(features: pd.DataFrame, patterns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Drop any column whose name CONTAINS any of the given substrings.

    Previously used startswith, but our feature naming has inconsistent
    prefixes (e.g. macro_days_to_nfp vs macro_nfp_week). Substring match
    lets you ablate by event name reliably:  `--ablate nfp` drops both.
    """
    to_drop = []
    for c in features.columns:
        for p in patterns:
            if p in c:
                to_drop.append(c)
                break
    kept = features.drop(columns=to_drop)
    return kept, to_drop


def diagnose(symbol: str, horizon: str, ablate: list[str] | None = None, n_top: int = 15) -> dict:
    from data_fetcher import HORIZONS

    if horizon not in HORIZONS:
        raise ValueError(f"Horizon '{horizon}' unknown. Valid: {list(HORIZONS.keys())}")

    print(f"\nLoading {symbol} / {horizon}…")
    features, targets, df = _load_data(symbol)

    target_col = f"target_{horizon}"
    if target_col not in targets.columns:
        raise ValueError(f"No target column '{target_col}'")

    # Align features + target, drop NaN target rows (future unknown)
    y = targets[target_col]
    mask = y.notna()
    X = features.loc[mask].copy()
    y = y.loc[mask].copy()

    ablated_cols = []
    if ablate:
        X, ablated_cols = _ablate(X, ablate)
        print(f"Ablated {len(ablated_cols)} cols matching {ablate}")

    # Convert classification target: direction only
    y_dir = (y > 0).astype(int)

    # 80/20 train/test chronological split
    n = len(X)
    split = int(n * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    yd_tr, yd_te = y_dir.iloc[:split], y_dir.iloc[split:]

    # Fill NaN, replace inf
    X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_te = X_te.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Lazy xgboost import
    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost not installed; can't diagnose.")
        return {}

    print(f"Training XGB on {len(X_tr)} rows × {X_tr.shape[1]} features…")
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, random_state=42,
        tree_method="hist", verbosity=0,
    )
    clf.fit(X_tr, yd_tr)
    preds = clf.predict(X_te)
    acc = (preds == yd_te.values).mean() * 100

    # Feature importance
    imp = pd.Series(clf.feature_importances_, index=X_tr.columns)
    imp = imp[imp > 0].sort_values(ascending=False)

    return {
        "symbol": symbol,
        "horizon": horizon,
        "n_features": X_tr.shape[1],
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "accuracy_pct": round(acc, 2),
        "top_importances": imp.head(n_top).to_dict(),
        "ablated_cols": ablated_cols,
    }


def _print_result(result: dict) -> None:
    print("\n" + "=" * 66)
    print(f"  {result['symbol']}  ·  {result['horizon']}")
    print("=" * 66)
    print(f"  features:      {result['n_features']}"
          + (f"  (ablated {len(result['ablated_cols'])})"
             if result["ablated_cols"] else ""))
    print(f"  train rows:    {result['n_train']}")
    print(f"  test rows:     {result['n_test']}")
    print(f"  directional accuracy (XGB single-model): {result['accuracy_pct']}%")
    print()
    print("  Top features by gain importance:")
    for i, (name, imp) in enumerate(result["top_importances"].items(), 1):
        bar = "█" * int(imp * 200)
        print(f"    {i:>2}. {name:<40} {imp:.4f}  {bar}")
    print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("symbol")
    p.add_argument("horizon", help="e.g. '1 Week', '1 Month', '1 Quarter'")
    p.add_argument("--ablate", nargs="*", default=None,
                   help="Column-name prefixes to drop (e.g. macro, sent_oil)")
    p.add_argument("--n-top", type=int, default=15)
    args = p.parse_args()

    try:
        result = diagnose(args.symbol, args.horizon, ablate=args.ablate, n_top=args.n_top)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return 1

    _print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
