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
    "Strong Buy":  "#00c853",
    "Buy":         "#69f0ae",
    "Hold":        "#ffd740",
    "Sell":        "#ff6d00",
    "Strong Sell": "#d50000",
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
    one_month_ret = predictions["1 Month"]["predicted_return"]
    recommendation = _classify_return(one_month_ret)
    confidence     = _confidence(one_month_ret)

    # ── Price targets ─────────────────────────────────────────────────────────
    targets = {}
    for horizon, data in predictions.items():
        targets[horizon] = {
            "price":  data["predicted_price"],
            "return": round(data["predicted_return"] * 100, 1),
        }

    # ── Support / Resistance ──────────────────────────────────────────────────
    sr = find_support_resistance(df)

    # Entry zone: between closest support and current price (±0.5 %)
    entry_low  = sr["supports"][0] if sr["supports"] else round(current_price * 0.97, 2)
    entry_high = round(current_price * 1.005, 2)

    # Stop-loss: just below the second support level (or 5 % below entry)
    if len(sr["supports"]) >= 2:
        stop_loss = round(sr["supports"][1] * 0.995, 2)
    else:
        stop_loss = round(current_price * 0.95, 2)

    # First resistance target
    first_target = sr["resistances"][0] if sr["resistances"] else round(current_price * 1.05, 2)

    entry_zone = {
        "entry_low":     entry_low,
        "entry_high":    entry_high,
        "stop_loss":     stop_loss,
        "first_target":  first_target,
        "risk_reward":   _risk_reward(entry_high, stop_loss, first_target),
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
    w_ret  = predictions["1 Week"]["predicted_return"]
    m_ret  = predictions["1 Month"]["predicted_return"]
    q_ret  = predictions["1 Quarter"]["predicted_return"]

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
        lines.append(f"\n**Entry Zone:** ${entry_zone['entry_low']:.2f} – ${entry_zone['entry_high']:.2f} "
                     f"with a stop-loss at ${entry_zone['stop_loss']:.2f} "
                     f"and initial target at ${entry_zone['first_target']:.2f} "
                     f"(risk/reward ratio: {entry_zone['risk_reward']:.1f}×).")

    if beta:
        lines.append(f"\n**Risk:** Beta is {beta:.2f}, meaning this stock is "
                     f"{'more volatile' if beta > 1 else 'less volatile'} than the market.")

    lines.append(f"\n**Disclaimer:** This analysis is generated by a machine learning model "
                 f"for informational purposes only. Past performance does not guarantee future results. "
                 f"Always do your own research before making any investment decisions.")

    return "\n".join(lines)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _risk_reward(entry, stop, target):
    risk   = entry - stop
    reward = target - entry
    if risk <= 0:
        return None
    return round(reward / risk, 2)
