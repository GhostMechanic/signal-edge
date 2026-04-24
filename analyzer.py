"""
analyzer.py
-----------
Converts raw ML predictions into actionable trading signals:
  • Buy / Sell / Hold recommendation with confidence tier
  • Price targets for each horizon
  • Suggested entry zone and stop-loss
  • Plain-English technical and fundamental analysis summary
"""

import numpy as np
import pandas as pd
from data_fetcher import find_support_resistance, HORIZONS


# ─── Recommendation thresholds (based on predicted 1-month return) ────────────

THRESHOLDS = {
    "Strong Buy":  0.07,   # ≥ +7 %
    "Buy":         0.03,   # ≥ +3 %
    "Hold":       -0.03,   # between -3 % and +3 %
    "Sell":       -0.07,   # between -7 % and -3 %
    "Strong Sell": None,   # below -7 %
}

RECOMMENDATION_COLORS = {
    "Strong Buy":  "#06d6a0",   # brand cyan
    "Buy":         "#4be3c0",   # lighter cyan
    "Hold":        "#f5b841",   # warm amber (readable on dark)
    "Sell":        "#f27d5c",   # soft coral
    "Strong Sell": "#e04a4a",   # muted red (matches CHART_DOWN)
}


def _classify_return(ret: float) -> str:
    if ret >= THRESHOLDS["Strong Buy"]:
        return "Strong Buy"
    elif ret >= THRESHOLDS["Buy"]:
        return "Buy"
    elif ret >= THRESHOLDS["Hold"]:
        return "Hold"
    elif ret >= THRESHOLDS["Sell"]:
        return "Sell"
    else:
        return "Strong Sell"


def _confidence(ret: float) -> str:
    """Rough confidence tier based on magnitude of predicted move."""
    abs_ret = abs(ret)
    if abs_ret >= 0.10:
        return "High"
    elif abs_ret >= 0.05:
        return "Medium"
    else:
        return "Low"


# ─── Core analysis function ───────────────────────────────────────────────────

def generate_analysis(
    df: pd.DataFrame,
    predictions: dict,
    stock_info: dict,
) -> dict:
    """
    Build a complete analysis bundle from raw predictions and OHLCV history.

    Parameters
    ----------
    df          : full OHLCV DataFrame
    predictions : dict from StockPredictor.predict()
    stock_info  : dict from fetch_stock_info()

    Returns
    -------
    dict with keys: recommendation, targets, entry_zone, technicals,
                    signals, narrative
    """

    current_price = float(df["Close"].iloc[-1])

    # ── Primary signal: 1-month prediction ───────────────────────────────────
    # Try 1 Month first, then fall back to any available horizon
    one_month_ret = 0.0
    recommendation = "Hold"
    confidence = "Low"

    for key in ["1 Month", "1 Week", "3 Day", "1 Quarter", "1 Year"]:
        if key in predictions:
            one_month_ret = predictions[key]["predicted_return"]
            recommendation = _classify_return(one_month_ret)
            confidence = _confidence(one_month_ret)
            break

    # ── Price targets ─────────────────────────────────────────────────────────
    targets = {}
    for horizon, data in predictions.items():
        targets[horizon] = {
            "price":  data["predicted_price"],
            "return": round(data["predicted_return"] * 100, 1),
        }

    # ── Support / Resistance ──────────────────────────────────────────────────
    sr = find_support_resistance(df)

    # ── Volatility-based stop-loss & target calculation ──────────────────────
    # Calculate ATR (Average True Range) for dynamic stop-loss placement
    lookback = min(20, len(df) - 1)
    high = df["High"].iloc[-lookback:].values
    low = df["Low"].iloc[-lookback:].values
    close = df["Close"].iloc[-lookback:].values
    prev_close = df["Close"].iloc[-lookback-1:-1].values if len(df) > lookback else close
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = float(np.mean(tr)) if len(tr) > 0 else 0.02 * current_price
    atr_pct = atr / current_price

    # Use 1-month prediction to set target
    one_month_pred = predictions.get("1 Month", {})
    pred_return = one_month_pred.get("predicted_return", 0.0)
    pred_price = one_month_pred.get("predicted_price", current_price)

    # Entry zone: support level to current price for bullish, current to resistance for bearish
    entry_low  = sr["supports"][0] if sr["supports"] else round(current_price * (1 - 2*atr_pct), 2)
    entry_high = round(current_price * 1.005, 2)  # very tight at current

    # Stop-loss: 1 ATR below the lower support (risk per share = ATR + buffer)
    if len(sr["supports"]) >= 2:
        stop_loss = round(sr["supports"][1] - atr * 0.5, 2)
    else:
        stop_loss = round(entry_low - atr, 2)

    # Target: use ML prediction if available, otherwise use resistance + ATR multiple
    if pred_return > 0.02:  # If model predicts positive
        # Target = predicted price OR first resistance + buffer, whichever is higher
        target_from_pred = pred_price
        target_from_sr   = sr["resistances"][0] if sr["resistances"] else round(current_price * (1 + 3*atr_pct), 2)
        first_target = max(target_from_pred, target_from_sr)
    else:
        # Conservative: use first resistance or 1.5x ATR move
        first_target = sr["resistances"][0] if sr["resistances"] else round(current_price + atr * 1.5, 2)

    # Calculate risk/reward ratio (reward / risk)
    risk = entry_high - stop_loss
    reward = first_target - entry_low  # Use best entry for target calculation
    if risk > 0:
        rr = round(reward / risk, 2)
    else:
        rr = 0

    entry_zone = {
        "entry_low":     entry_low,
        "entry_high":    entry_high,
        "stop_loss":     stop_loss,
        "first_target":  first_target,
        "risk_reward":   rr,
        "atr":           round(atr, 2),
        "atr_pct":       round(atr_pct * 100, 1),
        "supports":      sr["supports"],
        "resistances":   sr["resistances"],
    }

    # ── Technical indicator signals ───────────────────────────────────────────
    technicals = _compute_technicals(df)

    # ── Composite signal score (−100 → +100) ──────────────────────────────────
    signals = _build_signals(df, technicals, predictions)

    # ── Plain-English narrative ───────────────────────────────────────────────
    narrative = _build_narrative(
        stock_info, current_price, recommendation,
        confidence, targets, technicals, signals, entry_zone
    )

    return {
        "recommendation":  recommendation,
        "confidence":      confidence,
        "color":           RECOMMENDATION_COLORS[recommendation],
        "current_price":   current_price,
        "targets":         targets,
        "entry_zone":      entry_zone,
        "technicals":      technicals,
        "signals":         signals,
        "narrative":       narrative,
        "sr":              sr,
    }


# ─── Technical snapshot ───────────────────────────────────────────────────────

def _compute_technicals(df: pd.DataFrame) -> dict:
    close  = df["Close"]
    volume = df["Volume"]

    def rsi(s, p=14):
        d = s.diff()
        g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
        l = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
        return float((100 - 100 / (1 + g/l.replace(0, np.nan))).iloc[-1])

    def ma(n):
        return float(close.rolling(n).mean().iloc[-1])

    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    sig    = macd.ewm(span=9, adjust=False).mean()

    sma20  = close.rolling(20).mean()
    std20  = close.rolling(20).std()
    bb_up  = sma20 + 2 * std20
    bb_lo  = sma20 - 2 * std20
    pct_b  = float(((close - bb_lo) / (bb_up - bb_lo + 1e-9)).iloc[-1])

    cur    = float(close.iloc[-1])

    vol_ratio = float(
        (volume / volume.rolling(20).mean()).iloc[-1]
    )

    return {
        "price":       cur,
        "ma20":        round(ma(20), 2),
        "ma50":        round(ma(50), 2),
        "ma200":       round(ma(200), 2),
        "rsi":         round(rsi(close), 1),
        "macd":        round(float(macd.iloc[-1]), 4),
        "macd_signal": round(float(sig.iloc[-1]), 4),
        "macd_cross":  "Bullish" if float(macd.iloc[-1]) > float(sig.iloc[-1]) else "Bearish",
        "bb_pct":      round(pct_b * 100, 1),
        "bb_pos":      _bb_label(pct_b),
        "vol_ratio":   round(vol_ratio, 2),
        "above_ma20":  cur > ma(20),
        "above_ma50":  cur > ma(50),
        "above_ma200": cur > ma(200),
        "trend":       _trend_label(cur, ma(20), ma(50), ma(200)),
    }


def _bb_label(pct_b: float) -> str:
    if pct_b > 1.0:   return "Above Upper Band"
    if pct_b > 0.8:   return "Near Upper Band"
    if pct_b < 0.0:   return "Below Lower Band"
    if pct_b < 0.2:   return "Near Lower Band"
    return "Middle of Band"


def _trend_label(price, ma20, ma50, ma200) -> str:
    if price > ma20 > ma50 > ma200:
        return "Strong Uptrend"
    if price > ma50 > ma200:
        return "Uptrend"
    if price < ma20 < ma50 < ma200:
        return "Strong Downtrend"
    if price < ma50 < ma200:
        return "Downtrend"
    return "Sideways / Mixed"


# ─── Signal scorecard ─────────────────────────────────────────────────────────

def _build_signals(df: pd.DataFrame, tech: dict, predictions: dict) -> list:
    """Return a list of (name, direction, description) signal tuples."""
    signals = []
    cur = tech["price"]

    # Trend signals
    if cur > tech["ma200"]:
        signals.append(("Price > 200 MA", "Bullish", "Price is above the long-term moving average"))
    else:
        signals.append(("Price < 200 MA", "Bearish", "Price is below the long-term moving average"))

    if tech["above_ma50"] and tech["above_ma20"]:
        signals.append(("MA Stack", "Bullish", "Price above both 20 and 50-day MAs"))
    elif not tech["above_ma50"] and not tech["above_ma20"]:
        signals.append(("MA Stack", "Bearish", "Price below both 20 and 50-day MAs"))
    else:
        signals.append(("MA Stack", "Neutral", "Price between key moving averages"))

    # Momentum
    rsi = tech["rsi"]
    if rsi > 70:
        signals.append(("RSI", "Bearish", f"RSI at {rsi} — overbought territory"))
    elif rsi < 30:
        signals.append(("RSI", "Bullish", f"RSI at {rsi} — oversold territory"))
    elif rsi > 50:
        signals.append(("RSI", "Bullish", f"RSI at {rsi} — bullish momentum"))
    else:
        signals.append(("RSI", "Bearish", f"RSI at {rsi} — bearish momentum"))

    # MACD
    signals.append(("MACD", tech["macd_cross"],
                    f"MACD is {tech['macd_cross'].lower()} (MACD: {tech['macd']:.4f})"))

    # Bollinger Bands
    bb = tech["bb_pos"]
    if "Upper" in bb:
        signals.append(("Bollinger Bands", "Bearish", f"Price {bb.lower()} — potential resistance"))
    elif "Lower" in bb:
        signals.append(("Bollinger Bands", "Bullish", f"Price {bb.lower()} — potential support"))
    else:
        signals.append(("Bollinger Bands", "Neutral", f"Price in {bb.lower()}"))

    # Volume
    vr = tech["vol_ratio"]
    if vr > 1.5:
        signals.append(("Volume", "Bullish" if tech["above_ma20"] else "Bearish",
                        f"Volume {vr:.1f}× average — strong conviction"))
    else:
        signals.append(("Volume", "Neutral", f"Volume {vr:.1f}× average — normal activity"))

    # ML model (short + long-term agreement)
    # Safely extract returns from available horizons
    w_ret  = predictions.get("1 Week", {}).get("predicted_return", 0)
    m_ret  = predictions.get("1 Month", {}).get("predicted_return", 0)
    q_ret  = predictions.get("1 Quarter", {}).get("predicted_return", 0)

    if w_ret > 0 and m_ret > 0 and q_ret > 0:
        signals.append(("ML Forecast", "Bullish",
                        "Short, medium & long-term models all predict positive returns"))
    elif w_ret < 0 and m_ret < 0 and q_ret < 0:
        signals.append(("ML Forecast", "Bearish",
                        "Short, medium & long-term models all predict negative returns"))
    else:
        signals.append(("ML Forecast", "Neutral", "Mixed signals across time horizons"))

    return signals


# ─── Narrative ────────────────────────────────────────────────────────────────

def _build_narrative(
    info, price, rec, conf, targets, tech, signals, entry_zone
) -> str:
    name   = info.get("name", "This stock")
    sector = info.get("sector", "N/A")
    beta   = info.get("beta", None)

    bull = sum(1 for s in signals if s[1] == "Bullish")
    bear = sum(1 for s in signals if s[1] == "Bearish")

    lines = []

    lines.append(f"**Overview:** {name} is trading at ${price:.2f}. "
                 f"Our ML model issues a **{rec}** recommendation with **{conf} Confidence**.")

    lines.append(f"\n**Technical Trend:** The stock is in a **{tech['trend']}**. "
                 f"It is {'above' if tech['above_ma20'] else 'below'} its 20-day MA (${tech['ma20']:.2f}), "
                 f"{'above' if tech['above_ma50'] else 'below'} its 50-day MA (${tech['ma50']:.2f}), "
                 f"and {'above' if tech['above_ma200'] else 'below'} its 200-day MA (${tech['ma200']:.2f}).")

    lines.append(f"\n**Momentum:** RSI is {tech['rsi']} "
                 f"({'overbought' if tech['rsi'] > 70 else 'oversold' if tech['rsi'] < 30 else 'neutral range'}). "
                 f"MACD is {tech['macd_cross'].lower()}, suggesting "
                 f"{'increasing' if tech['macd_cross'] == 'Bullish' else 'weakening'} momentum.")

    lines.append(f"\n**Signal Summary:** {bull} bullish vs {bear} bearish signals across technical indicators.")

    if entry_zone["risk_reward"] and entry_zone["risk_reward"] > 0:
        atr_info = f" (ATR: ${entry_zone['atr']:.2f} / {entry_zone['atr_pct']:.1f}%)" if "atr" in entry_zone else ""
        lines.append(f"\n**Entry Zone:** ${entry_zone['entry_low']:.2f} – ${entry_zone['entry_high']:.2f} "
                     f"with stop-loss at ${entry_zone['stop_loss']:.2f} "
                     f"and target at ${entry_zone['first_target']:.2f} "
                     f"(ratio: {entry_zone['risk_reward']:.1f}×){atr_info}.")

    if beta:
        lines.append(f"\n**Risk:** Beta is {beta:.2f}, meaning this stock is "
                     f"{'more volatile' if beta > 1 else 'less volatile'} than the market.")

    lines.append(f"\n**Disclaimer:** This analysis is generated by a machine learning model "
                 f"for informational purposes only. Past performance does not guarantee future results. "
                 f"Always do your own research before making any investment decisions.")

    return "\n".join(lines)


