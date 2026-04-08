"""
regime_detector.py
------------------
Classifies the current market regime as Bull / Bear / Sideways using a
multi-signal scoring approach.  Returns a normalised score and label that
can be used to:
  1. Add regime one-hot features to the ML feature matrix
  2. Route the input to a regime-specific sub-model (if available)
  3. Display a contextual badge in the UI

Signals used
------------
  • Stock price vs 50-day MA   (trend)
  • Stock price vs 200-day MA  (long-term trend)
  • 63-day momentum (close / close_63d_ago - 1)
  • 21-day momentum
  • Realised volatility regime  (current vol vs 90-day average)
  • SPY price vs 200-day MA     (market context, optional)
  • VIX level                  (fear gauge, optional)

Scoring
-------
Each signal contributes +1 (bullish), -1 (bearish) or 0 (neutral) to a raw
score.  The score is then converted to:
  • Bull   if score ≥  2
  • Bear   if score ≤ -2
  • Sideways otherwise

One-hot encoding for model features: [is_bull, is_bear, is_sideways]
"""

import numpy as np
import pandas as pd
from typing import Optional


# ─── Constants ────────────────────────────────────────────────────────────────

REGIME_LABELS = ["Bull", "Bear", "Sideways"]


# ─── Core detector ────────────────────────────────────────────────────────────

def detect_regime(
    df: pd.DataFrame,
    spy_close: Optional[pd.Series] = None,
    vix_close: Optional[pd.Series] = None,
) -> dict:
    """
    Detect the current market regime for a stock.

    Parameters
    ----------
    df         : OHLCV DataFrame (Close required)
    spy_close  : SPY close series aligned to df.index (optional)
    vix_close  : VIX close series aligned to df.index (optional)

    Returns
    -------
    dict with keys:
        label       : "Bull" | "Bear" | "Sideways"
        score       : raw signal score (int, typically -7 to +7)
        score_norm  : normalised 0-1 score (1=strong bull, 0=strong bear)
        is_bull     : bool
        is_bear     : bool
        is_sideways : bool
        signals     : list of (signal_name, value, vote) tuples for display
    """
    close = df["Close"]
    n     = len(close)
    score = 0
    signals = []

    def _add(name: str, val, vote: int, fmt: str = ".2f"):
        nonlocal score
        score += vote
        label = "↑ Bullish" if vote > 0 else ("↓ Bearish" if vote < 0 else "→ Neutral")
        signals.append((name, val, vote, label))

    # ── Price vs 50-day MA ───────────────────────────────────────────────────
    if n >= 50:
        ma50  = float(close.rolling(50).mean().iloc[-1])
        cur   = float(close.iloc[-1])
        ratio = cur / (ma50 + 1e-9) - 1
        vote  = 1 if ratio > 0.01 else (-1 if ratio < -0.01 else 0)
        _add("Price vs 50-MA", ratio, vote)

    # ── Price vs 200-day MA ──────────────────────────────────────────────────
    if n >= 200:
        ma200 = float(close.rolling(200).mean().iloc[-1])
        cur   = float(close.iloc[-1])
        ratio = cur / (ma200 + 1e-9) - 1
        vote  = 1 if ratio > 0.02 else (-1 if ratio < -0.02 else 0)
        _add("Price vs 200-MA", ratio, vote)

    # ── 63-day momentum ──────────────────────────────────────────────────────
    if n >= 64:
        mom63 = float(close.pct_change(63).iloc[-1])
        vote  = 1 if mom63 > 0.05 else (-1 if mom63 < -0.05 else 0)
        _add("63d Momentum", mom63, vote)

    # ── 21-day momentum ──────────────────────────────────────────────────────
    if n >= 22:
        mom21 = float(close.pct_change(21).iloc[-1])
        vote  = 1 if mom21 > 0.02 else (-1 if mom21 < -0.02 else 0)
        _add("21d Momentum", mom21, vote)

    # ── Volatility regime ────────────────────────────────────────────────────
    if n >= 90:
        log_ret  = np.log(close / close.shift(1)).dropna()
        vol_21   = float(log_ret.iloc[-21:].std() * np.sqrt(252))
        vol_90   = float(log_ret.iloc[-90:].std() * np.sqrt(252))
        vol_ratio = vol_21 / (vol_90 + 1e-9)
        # High vol spike = bearish; low vol = bullish calm
        vote = -1 if vol_ratio > 1.4 else (1 if vol_ratio < 0.8 else 0)
        _add("Vol Regime (21d/90d)", vol_ratio, vote)

    # ── SPY context ──────────────────────────────────────────────────────────
    if spy_close is not None and len(spy_close) >= 200:
        spy = spy_close.reindex(df.index).ffill()
        if len(spy.dropna()) >= 200:
            spy_ma200 = float(spy.rolling(200).mean().iloc[-1])
            spy_cur   = float(spy.iloc[-1])
            ratio = spy_cur / (spy_ma200 + 1e-9) - 1
            vote  = 1 if ratio > 0.02 else (-1 if ratio < -0.02 else 0)
            _add("SPY vs 200-MA", ratio, vote)

    # ── VIX level ────────────────────────────────────────────────────────────
    if vix_close is not None and len(vix_close) > 0:
        vix = vix_close.reindex(df.index).ffill()
        vix_val = float(vix.iloc[-1]) if not vix.empty else None
        if vix_val and not np.isnan(vix_val):
            # VIX > 30 = fear (bearish context), VIX < 15 = complacent (bullish)
            vote  = -1 if vix_val > 30 else (1 if vix_val < 15 else 0)
            _add("VIX Level", vix_val, vote)

    # ── Classify ─────────────────────────────────────────────────────────────
    max_possible = len(signals)
    label     = "Bull" if score >= 2 else ("Bear" if score <= -2 else "Sideways")
    score_norm = (score + max_possible) / (2 * max_possible + 1e-9) if max_possible else 0.5

    return {
        "label":       label,
        "score":       score,
        "score_norm":  round(float(np.clip(score_norm, 0.0, 1.0)), 3),
        "is_bull":     label == "Bull",
        "is_bear":     label == "Bear",
        "is_sideways": label == "Sideways",
        "signals":     signals,
    }


def regime_features(
    df: pd.DataFrame,
    spy_close: Optional[pd.Series] = None,
    vix_close: Optional[pd.Series] = None,
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute rolling regime one-hot features for the full history.
    Returns DataFrame with columns: regime_bull, regime_bear, regime_sideways,
    regime_score_norm.

    This is used inside engineer_features to add regime context to each row.
    For speed, regime is computed with a rolling window (re-evaluated every
    `window` rows is too slow; instead we use rolling moving-average signals
    directly as continuous features rather than hard classification).
    """
    close = df["Close"]
    n     = len(close)
    feats = pd.DataFrame(index=df.index)

    # ── Rolling trend features (proxy for regime) ────────────────────────────
    # Price vs rolling MAs
    feats["regime_vs_50ma"]  = close / (close.rolling(50).mean()  + 1e-9) - 1
    feats["regime_vs_200ma"] = close / (close.rolling(200).mean() + 1e-9) - 1

    # Rolling momentum
    feats["regime_mom63"] = close.pct_change(63)
    feats["regime_mom21"] = close.pct_change(21)

    # Volatility ratio
    log_ret = np.log(close / close.shift(1))
    vol21   = log_ret.rolling(21).std()
    vol90   = log_ret.rolling(90).std()
    feats["regime_vol_ratio"] = vol21 / (vol90 + 1e-9)

    # SPY context (rolling)
    if spy_close is not None and len(spy_close) >= 200:
        spy = spy_close.reindex(df.index).ffill()
        feats["regime_spy_vs_200ma"] = spy / (spy.rolling(200).mean() + 1e-9) - 1
        feats["regime_spy_mom63"]    = spy.pct_change(63)

    # VIX normalised
    if vix_close is not None and len(vix_close) > 0:
        vix = vix_close.reindex(df.index).ffill()
        feats["regime_vix_norm"] = vix / 20.0 - 1.0   # 20 = neutral

    # ── One-hot snap at current end (scalar broadcast) ────────────────────────
    # These are fixed scalars for the most recent regime, broadcast to whole df.
    # Useful for "what regime are we in now" context during final-row prediction.
    # During training they just repeat the current classification (low signal,
    # but the rolling features above capture the time-varying signal properly).
    snap = detect_regime(df, spy_close, vix_close)
    feats["snap_is_bull"]     = float(snap["is_bull"])
    feats["snap_is_bear"]     = float(snap["is_bear"])
    feats["snap_score_norm"]  = snap["score_norm"]

    return feats
