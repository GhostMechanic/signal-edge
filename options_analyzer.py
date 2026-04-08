"""
options_analyzer.py  –  Options Strategy Recommendation Engine
--------------------------------------------------------------
Given ML predictions (direction, magnitude, confidence) and historical
volatility data, recommends the optimal options strategy for each horizon.

NOTE: Uses historical realised volatility as an IV proxy since we don't have
live options chain data. Real trading should verify actual IV from a broker.

Strategy selection logic:
  ┌──────────────────────┬──────────────┬──────────────────────────────────┐
  │ Predicted Move       │ Confidence   │ Recommended Strategy             │
  ├──────────────────────┼──────────────┼──────────────────────────────────┤
  │ Strong Bull (>7%)    │ High         │ Long Call (ATM)                  │
  │ Strong Bull (>7%)    │ Med/Low      │ Bull Call Spread                 │
  │ Moderate Bull (3-7%) │ Any          │ Bull Call Spread                 │
  │ Weak Bull (1-3%)     │ High         │ Covered Call / Cash-Secured Put  │
  │ Neutral (-1% to 1%)  │ Any          │ Iron Condor                      │
  │ Weak Bear (-3% to-1%)│ High         │ Protective Put / Bear Put Spread │
  │ Moderate Bear(-7%-3%)│ Any          │ Bear Put Spread                  │
  │ Strong Bear (<-7%)   │ High         │ Long Put (ATM)                   │
  │ Strong Bear (<-7%)   │ Med/Low      │ Bear Put Spread                  │
  │ High vol expected    │ Any          │ Long Straddle / Strangle         │
  └──────────────────────┴──────────────┴──────────────────────────────────┘
"""

import numpy as np
import pandas as pd
from typing import Optional

# ─── Thresholds ───────────────────────────────────────────────────────────────
STRONG_BULL  =  0.07
MOD_BULL     =  0.03
WEAK_BULL    =  0.01
WEAK_BEAR    = -0.01
MOD_BEAR     = -0.03
STRONG_BEAR  = -0.07

HIGH_CONF    = 70    # confidence score threshold
MED_CONF     = 55

HIGH_VOL_PERCENTILE = 70   # current vol vs historical: above this = "elevated"


# ─── Black-Scholes helpers (simplified, for illustration) ────────────────────

def _bs_call_price(S, K, T, sigma, r=0.05):
    """Approximate Black-Scholes call price. T in years."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    from math import log, sqrt, exp
    try:
        from scipy.stats import norm
        d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        return S * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)
    except ImportError:
        # Rough approximation without scipy
        intrinsic = max(S - K, 0)
        time_val  = S * sigma * (T ** 0.5) * 0.4
        return intrinsic + time_val


def _bs_put_price(S, K, T, sigma, r=0.05):
    """Approximate Black-Scholes put price via put-call parity."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    from math import exp
    call = _bs_call_price(S, K, T, sigma, r)
    return call - S + K * (1 / (1 + r*T))


def _estimate_iv(df: pd.DataFrame, window: int = 21) -> float:
    """
    Estimate implied volatility from historical realised vol (annualised).
    Uses a fixed window ending at the LAST row so result is deterministic
    for a given DataFrame (independent of when the function is called).
    Adds a 20% premium to approximate typical IV > RV spread.
    """
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    # Use only the last `window` complete observations — fully deterministic
    rv = float(log_ret.iloc[-window:].std() * np.sqrt(252))
    return round(rv * 1.20, 4)


def _vol_percentile(df: pd.DataFrame) -> float:
    """Where does current 21-day vol sit relative to 1-year history? Deterministic."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    rv_21d  = log_ret.rolling(21).std() * np.sqrt(252)
    rv_21d  = rv_21d.dropna()
    cur_rv  = float(rv_21d.iloc[-1]) if len(rv_21d) > 0 else 0.20
    pct     = float((rv_21d < cur_rv).mean() * 100)
    return round(pct, 1)


# ─── Strategy builder ─────────────────────────────────────────────────────────

def _expiry_days(horizon: str) -> int:
    """Map horizon to a practical options expiry (slightly longer than hold period)."""
    return {
        "1 Week":    14,
        "1 Month":   45,
        "1 Quarter": 90,
        "1 Year":   365,
    }.get(horizon, 45)


def _strike_choices(S: float, direction: str, iv: float, T_years: float) -> dict:
    """
    Calculate ATM, OTM, and spread strikes.
    One-sigma move approximation for OTM strikes.
    """
    one_sigma = S * iv * (T_years ** 0.5)
    if direction == "bullish":
        return {
            "atm":    round(S, 0),
            "otm":    round(S + 0.5 * one_sigma, 0),
            "far_otm":round(S + one_sigma, 0),
            "spread_buy": round(S * 0.995, 0),         # slightly ITM
            "spread_sell":round(S + 0.5 * one_sigma, 0),
        }
    else:
        return {
            "atm":    round(S, 0),
            "otm":    round(S - 0.5 * one_sigma, 0),
            "far_otm":round(S - one_sigma, 0),
            "spread_buy": round(S * 1.005, 0),          # slightly ITM
            "spread_sell":round(S - 0.5 * one_sigma, 0),
        }


def build_strategy(
    horizon: str,
    pred: dict,
    df: pd.DataFrame,
) -> dict:
    """
    Build a full options strategy recommendation for one horizon.

    Parameters
    ----------
    horizon : e.g. "1 Month"
    pred    : dict from StockPredictor.predict()[horizon]
    df      : OHLCV DataFrame

    Returns
    -------
    dict with strategy name, rationale, strikes, estimated cost, max gain/loss
    """
    ret        = pred["predicted_return"]
    conf       = pred["confidence"]
    S          = pred["current_price"]
    T_days     = _expiry_days(horizon)
    T_years    = T_days / 365.0
    iv         = _estimate_iv(df)
    vol_pct    = _vol_percentile(df)
    high_vol   = vol_pct >= HIGH_VOL_PERCENTILE

    # ── Special case: very high vol → straddle makes sense ──────────────────
    if high_vol and abs(ret) < MOD_BULL and conf < HIGH_CONF:
        return _straddle(S, T_years, T_days, iv, vol_pct, horizon, pred)

    # ── Direction-based selection ────────────────────────────────────────────
    if ret >= STRONG_BULL:
        if conf >= HIGH_CONF:
            return _long_call(S, T_years, T_days, iv, horizon, pred)
        else:
            return _bull_call_spread(S, T_years, T_days, iv, horizon, pred)

    elif ret >= MOD_BULL:
        return _bull_call_spread(S, T_years, T_days, iv, horizon, pred)

    elif ret >= WEAK_BULL:
        if conf >= HIGH_CONF:
            return _cash_secured_put(S, T_years, T_days, iv, horizon, pred)
        else:
            return _bull_call_spread(S, T_years, T_days, iv, horizon, pred)

    elif ret >= WEAK_BEAR:
        return _iron_condor(S, T_years, T_days, iv, horizon, pred)

    elif ret >= MOD_BEAR:
        return _bear_put_spread(S, T_years, T_days, iv, horizon, pred)

    elif ret >= STRONG_BEAR:
        return _bear_put_spread(S, T_years, T_days, iv, horizon, pred)

    else:  # strong bear
        if conf >= HIGH_CONF:
            return _long_put(S, T_years, T_days, iv, horizon, pred)
        else:
            return _bear_put_spread(S, T_years, T_days, iv, horizon, pred)


# ─── Individual strategy constructors ────────────────────────────────────────

def _long_call(S, T, T_days, iv, horizon, pred):
    strikes = _strike_choices(S, "bullish", iv, T)
    K       = strikes["atm"]
    premium = round(_bs_call_price(S, K, T, iv), 2)
    cost    = round(premium * 100, 2)  # per contract (100 shares)
    breakeven = round(K + premium, 2)
    max_gain   = "Unlimited (stock price - breakeven)"
    max_loss   = f"${cost:.2f} per contract (premium paid)"

    return {
        "strategy":      "Long Call",
        "emoji":         "📈",
        "direction":     "Bullish",
        "complexity":    "Beginner",
        "horizon":       horizon,
        "expiry_days":   T_days,
        "legs": [
            {"action": "BUY", "type": "CALL", "strike": K,
             "expiry": f"~{T_days}d out", "premium": premium}
        ],
        "estimated_cost":    f"${cost:.2f} / contract",
        "max_profit":        max_gain,
        "max_loss":          max_loss,
        "breakeven":         f"${breakeven:.2f}",
        "iv_used":           f"{iv*100:.1f}%",
        "rationale": (
            f"Strong bullish signal ({pred['predicted_return']*100:.1f}% predicted return, "
            f"{pred['confidence']:.0f}% confidence). Long ATM call gives maximum upside "
            f"exposure with limited downside (capped at premium paid)."
        ),
        "risk_note": "Full premium at risk if stock stays below breakeven at expiry.",
    }


def _bull_call_spread(S, T, T_days, iv, horizon, pred):
    strikes  = _strike_choices(S, "bullish", iv, T)
    K_buy    = strikes["atm"]
    K_sell   = strikes["spread_sell"]
    p_buy    = round(_bs_call_price(S, K_buy,  T, iv), 2)
    p_sell   = round(_bs_call_price(S, K_sell, T, iv), 2)
    net_debit= round((p_buy - p_sell) * 100, 2)
    max_gain = round((K_sell - K_buy - p_buy + p_sell) * 100, 2)
    breakeven= round(K_buy + p_buy - p_sell, 2)

    return {
        "strategy":      "Bull Call Spread",
        "emoji":         "📊",
        "direction":     "Bullish",
        "complexity":    "Intermediate",
        "horizon":       horizon,
        "expiry_days":   T_days,
        "legs": [
            {"action": "BUY",  "type": "CALL", "strike": K_buy,  "expiry": f"~{T_days}d", "premium": p_buy},
            {"action": "SELL", "type": "CALL", "strike": K_sell, "expiry": f"~{T_days}d", "premium": p_sell},
        ],
        "estimated_cost": f"${net_debit:.2f} net debit / contract",
        "max_profit":     f"${max_gain:.2f} / contract (at ${K_sell:.2f}+)",
        "max_loss":       f"${net_debit:.2f} / contract (net premium paid)",
        "breakeven":      f"${breakeven:.2f}",
        "iv_used":        f"{iv*100:.1f}%",
        "rationale": (
            f"Bullish signal ({pred['predicted_return']*100:.1f}% predicted). "
            f"Spread reduces cost vs outright call while still capturing upside to ${K_sell:.2f}. "
            f"Ideal when confidence is moderate or IV is elevated."
        ),
        "risk_note": "Max loss is the net premium paid. Profit capped at upper strike.",
    }


def _long_put(S, T, T_days, iv, horizon, pred):
    strikes  = _strike_choices(S, "bearish", iv, T)
    K        = strikes["atm"]
    premium  = round(_bs_put_price(S, K, T, iv), 2)
    cost     = round(premium * 100, 2)
    breakeven= round(K - premium, 2)

    return {
        "strategy":      "Long Put",
        "emoji":         "📉",
        "direction":     "Bearish",
        "complexity":    "Beginner",
        "horizon":       horizon,
        "expiry_days":   T_days,
        "legs": [
            {"action": "BUY", "type": "PUT", "strike": K,
             "expiry": f"~{T_days}d", "premium": premium}
        ],
        "estimated_cost": f"${cost:.2f} / contract",
        "max_profit":     f"${round((K - premium)*100,2):.2f} / contract (stock → $0)",
        "max_loss":       f"${cost:.2f} / contract (premium paid)",
        "breakeven":      f"${breakeven:.2f}",
        "iv_used":        f"{iv*100:.1f}%",
        "rationale": (
            f"Strong bearish signal ({pred['predicted_return']*100:.1f}% predicted return, "
            f"{pred['confidence']:.0f}% confidence). Long ATM put profits as stock falls."
        ),
        "risk_note": "Full premium at risk if stock stays above breakeven at expiry.",
    }


def _bear_put_spread(S, T, T_days, iv, horizon, pred):
    strikes   = _strike_choices(S, "bearish", iv, T)
    K_buy     = strikes["atm"]
    K_sell    = strikes["spread_sell"]
    p_buy     = round(_bs_put_price(S, K_buy,  T, iv), 2)
    p_sell    = round(_bs_put_price(S, K_sell, T, iv), 2)
    net_debit = round((p_buy - p_sell) * 100, 2)
    max_gain  = round((K_buy - K_sell - p_buy + p_sell) * 100, 2)
    breakeven = round(K_buy - p_buy + p_sell, 2)

    return {
        "strategy":       "Bear Put Spread",
        "emoji":          "🐻",
        "direction":      "Bearish",
        "complexity":     "Intermediate",
        "horizon":        horizon,
        "expiry_days":    T_days,
        "legs": [
            {"action": "BUY",  "type": "PUT", "strike": K_buy,  "expiry": f"~{T_days}d", "premium": p_buy},
            {"action": "SELL", "type": "PUT", "strike": K_sell, "expiry": f"~{T_days}d", "premium": p_sell},
        ],
        "estimated_cost": f"${net_debit:.2f} net debit / contract",
        "max_profit":     f"${max_gain:.2f} / contract",
        "max_loss":       f"${net_debit:.2f} / contract",
        "breakeven":      f"${breakeven:.2f}",
        "iv_used":        f"{iv*100:.1f}%",
        "rationale": (
            f"Bearish signal ({pred['predicted_return']*100:.1f}% predicted). "
            f"Put spread costs less than outright put and profits down to ${K_sell:.2f}."
        ),
        "risk_note": "Profit is capped at lower strike. Lower risk/reward than long put.",
    }


def _iron_condor(S, T, T_days, iv, horizon, pred):
    one_sigma  = S * iv * (T**0.5)
    K_put_sell = round(S - 0.5 * one_sigma, 0)
    K_put_buy  = round(S - 1.0 * one_sigma, 0)
    K_call_sell= round(S + 0.5 * one_sigma, 0)
    K_call_buy = round(S + 1.0 * one_sigma, 0)

    p_ps = round(_bs_put_price( S, K_put_sell,  T, iv), 2)
    p_pb = round(_bs_put_price( S, K_put_buy,   T, iv), 2)
    p_cs = round(_bs_call_price(S, K_call_sell, T, iv), 2)
    p_cb = round(_bs_call_price(S, K_call_buy,  T, iv), 2)

    net_credit = round((p_ps - p_pb + p_cs - p_cb) * 100, 2)
    wing_width = round((K_call_sell - K_put_sell) / 2, 2)
    max_loss   = round(((K_call_sell - K_put_sell) - (p_ps - p_pb + p_cs - p_cb)) * 100, 2)

    return {
        "strategy":       "Iron Condor",
        "emoji":          "🦅",
        "direction":      "Neutral",
        "complexity":     "Advanced",
        "horizon":        horizon,
        "expiry_days":    T_days,
        "legs": [
            {"action": "BUY",  "type": "PUT",  "strike": K_put_buy,   "expiry": f"~{T_days}d", "premium": p_pb},
            {"action": "SELL", "type": "PUT",  "strike": K_put_sell,  "expiry": f"~{T_days}d", "premium": p_ps},
            {"action": "SELL", "type": "CALL", "strike": K_call_sell, "expiry": f"~{T_days}d", "premium": p_cs},
            {"action": "BUY",  "type": "CALL", "strike": K_call_buy,  "expiry": f"~{T_days}d", "premium": p_cb},
        ],
        "estimated_cost": f"${net_credit:.2f} net credit / contract",
        "max_profit":     f"${net_credit:.2f} / contract (stock stays between ${K_put_sell}–${K_call_sell})",
        "max_loss":       f"${max_loss:.2f} / contract (stock breaks outside wings)",
        "breakeven":      f"${K_put_sell - net_credit/100:.2f} (down) / ${K_call_sell + net_credit/100:.2f} (up)",
        "iv_used":        f"{iv*100:.1f}%",
        "rationale": (
            f"Neutral/range-bound prediction ({pred['predicted_return']*100:.1f}%). "
            f"Iron condor profits if stock stays between ${K_put_sell} and ${K_call_sell}. "
            f"Best in high-IV environments where you're selling overpriced premium."
        ),
        "risk_note": "Four-legged strategy. Manage early if stock approaches short strikes.",
    }


def _straddle(S, T, T_days, iv, vol_pct, horizon, pred):
    K        = round(S, 0)
    p_call   = round(_bs_call_price(S, K, T, iv), 2)
    p_put    = round(_bs_put_price( S, K, T, iv), 2)
    total    = round((p_call + p_put) * 100, 2)
    be_up    = round(K + p_call + p_put, 2)
    be_down  = round(K - p_call - p_put, 2)

    return {
        "strategy":       "Long Straddle",
        "emoji":          "⚡",
        "direction":      "Neutral (Volatility Play)",
        "complexity":     "Intermediate",
        "horizon":        horizon,
        "expiry_days":    T_days,
        "legs": [
            {"action": "BUY", "type": "CALL", "strike": K, "expiry": f"~{T_days}d", "premium": p_call},
            {"action": "BUY", "type": "PUT",  "strike": K, "expiry": f"~{T_days}d", "premium": p_put},
        ],
        "estimated_cost": f"${total:.2f} / contract",
        "max_profit":     "Unlimited (large move in either direction)",
        "max_loss":       f"${total:.2f} / contract (no movement at expiry)",
        "breakeven":      f"${be_down:.2f} (down) / ${be_up:.2f} (up)",
        "iv_used":        f"{iv*100:.1f}%",
        "rationale": (
            f"Volatility is elevated (top {100-vol_pct:.0f}% historically) but direction is unclear. "
            f"Straddle profits from a large move either way. "
            f"NOTE: Works best when IV is low — current elevated IV increases cost."
        ),
        "risk_note": "Requires significant price movement to profit. Time decay (theta) works against you.",
    }


def _cash_secured_put(S, T, T_days, iv, horizon, pred):
    K        = round(S * 0.97, 0)   # slightly OTM
    premium  = round(_bs_put_price(S, K, T, iv), 2)
    credit   = round(premium * 100, 2)
    breakeven= round(K - premium, 2)

    return {
        "strategy":       "Cash-Secured Put",
        "emoji":          "💰",
        "direction":      "Mildly Bullish",
        "complexity":     "Intermediate",
        "horizon":        horizon,
        "expiry_days":    T_days,
        "legs": [
            {"action": "SELL", "type": "PUT", "strike": K,
             "expiry": f"~{T_days}d", "premium": premium}
        ],
        "estimated_cost": f"${round(K*100,2):,.2f} cash required (to buy 100 shares if assigned)",
        "max_profit":     f"${credit:.2f} / contract (keep full premium if stock > ${K})",
        "max_loss":       f"${round((K - premium)*100,2):,.2f} / contract (stock → $0)",
        "breakeven":      f"${breakeven:.2f}",
        "iv_used":        f"{iv*100:.1f}%",
        "rationale": (
            f"Mild bullish bias ({pred['predicted_return']*100:.1f}% predicted) with high confidence. "
            f"Sell a slightly OTM put to collect premium. If assigned, you buy stock at an effective "
            f"price of ${breakeven:.2f} — below today's price."
        ),
        "risk_note": "You must have cash to buy 100 shares if assigned. Avoid in strong downtrends.",
    }


# ─── Full options report ──────────────────────────────────────────────────────

def generate_options_report(
    predictions: dict,
    df: pd.DataFrame,
    focus_horizon: str = "1 Month",
) -> dict:
    """
    Build options strategy recommendations for all horizons.
    Returns dict keyed by horizon.
    """
    report = {}
    for horizon, pred in predictions.items():
        try:
            report[horizon] = build_strategy(horizon, pred, df)
        except Exception as e:
            report[horizon] = {"error": str(e), "strategy": "N/A"}

    # Mark the focus horizon
    if focus_horizon in report:
        report[focus_horizon]["is_primary"] = True

    return report
