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
from typing import List, Optional

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
        # Numeric block — drives paper-trade math. cost_per_contract is
        # the dollar premium debit (positive = paid out). max_profit_per_contract
        # is null when unbounded (long call has uncapped upside).
        "numeric": {
            "cost_per_contract":       cost,
            "max_profit_per_contract": None,
            "max_loss_per_contract":   cost,
            "breakevens":              [breakeven],
            "is_credit":               False,
            "tradable":                True,
        },
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
        "numeric": {
            "cost_per_contract":       net_debit,
            "max_profit_per_contract": max_gain,
            "max_loss_per_contract":   net_debit,
            "breakevens":              [breakeven],
            "is_credit":               False,
            "tradable":                True,
        },
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
        "numeric": {
            "cost_per_contract":       cost,
            # Bounded at K (stock can't go below 0); intrinsic value is K-S.
            "max_profit_per_contract": round((K - premium) * 100, 2),
            "max_loss_per_contract":   cost,
            "breakevens":              [breakeven],
            "is_credit":               False,
            "tradable":                True,
        },
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
        "numeric": {
            "cost_per_contract":       net_debit,
            "max_profit_per_contract": max_gain,
            "max_loss_per_contract":   net_debit,
            "breakevens":              [breakeven],
            "is_credit":               False,
            "tradable":                True,
        },
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

    be_down = round(K_put_sell - net_credit / 100, 2)
    be_up   = round(K_call_sell + net_credit / 100, 2)
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
        "breakeven":      f"${be_down:.2f} (down) / ${be_up:.2f} (up)",
        "iv_used":        f"{iv*100:.1f}%",
        # Iron condor receives net credit on open. For paper trading we
        # model this as: cash credited at open, max-loss notional locked
        # against margin-equivalent cash (defined-risk spread). Lock
        # value = max_loss per contract.
        "numeric": {
            # Negative cost = credit received (cash lands in account)
            "cost_per_contract":       -net_credit,
            "max_profit_per_contract": net_credit,
            "max_loss_per_contract":   max_loss,
            "breakevens":              [be_down, be_up],
            "is_credit":               True,
            # Locked cash for the defined-risk margin requirement.
            "margin_required_per_contract": max_loss,
            "tradable":                True,
        },
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
        "numeric": {
            "cost_per_contract":       total,
            "max_profit_per_contract": None,  # unbounded on the upside
            "max_loss_per_contract":   total,
            "breakevens":              [be_down, be_up],
            "is_credit":               False,
            "tradable":                True,
        },
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

    cash_required = round(K * 100, 2)
    max_loss_csp  = round((K - premium) * 100, 2)
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
        "estimated_cost": f"${cash_required:,.2f} cash required (to buy 100 shares if assigned)",
        "max_profit":     f"${credit:.2f} / contract (keep full premium if stock > ${K})",
        "max_loss":       f"${max_loss_csp:,.2f} / contract (stock → $0)",
        "breakeven":      f"${breakeven:.2f}",
        "iv_used":        f"{iv*100:.1f}%",
        # CSP is fully tradable in paper: locks K*100 cash as collateral
        # (margin_required), receives premium credit. At expiry: if S ≥ K
        # the put expires worthless and the user keeps the premium and
        # the cash. If S < K the user is "assigned" — for our paper
        # model, we resolve as a realised loss of (K - S)*100 + premium.
        "numeric": {
            "cost_per_contract":       -credit,
            "max_profit_per_contract": credit,
            "max_loss_per_contract":   max_loss_csp,
            "breakevens":              [breakeven],
            "is_credit":               True,
            "margin_required_per_contract": cash_required,
            "tradable":                True,
        },
        "rationale": (
            f"Mild bullish bias ({pred['predicted_return']*100:.1f}% predicted) with high confidence. "
            f"Sell a slightly OTM put to collect premium. If assigned, you buy stock at an effective "
            f"price of ${breakeven:.2f} — below today's price."
        ),
        "risk_note": "You must have cash to buy 100 shares if assigned. Avoid in strong downtrends.",
    }


# ─── Recompute numerics from live legs ───────────────────────────────────────
#
# At trade-open time the open path replaces each leg's asking-flow Black-
# Scholes premium with a live mid (or last) from the options chain, then
# calls in here to reshape the strategy's numeric block around the real
# numbers. The strategy_type + leg structure (action / type / strike)
# stays fixed; only the dollar-derived fields move.
#
# Closed-form formulas per strategy. Each leg dict carries
# `premium` (per-share dollars). All output dollar amounts are
# per-contract (× 100).
#
# Methodology § 4.5: persisting estimates as if they were fills is the
# same class of bug as keeping `today + N` as an expiry. Recomputing
# here is what fixes it for the cost-basis side.


def _find_leg(legs, action, leg_type):
    """First leg matching (action, type). Returns None if not present."""
    target_action = action.upper()
    target_type   = leg_type.upper()
    for leg in legs or []:
        if (
            (leg.get("action") or "").upper() == target_action
            and (leg.get("type") or "").upper() == target_type
        ):
            return leg
    return None


def _leg_premium(leg) -> float:
    return float(leg.get("premium") or 0)


def _leg_strike(leg) -> float:
    return float(leg.get("strike") or 0)


# ─── Cross-field invariant check ─────────────────────────────────────────────
#
# Hardening added after a MSFT bear-put-spread card on the public ledger
# showed three internally-inconsistent debit numbers (cost $1,181, max
# profit $1,719, breakeven $396.19) — each implied a different per-share
# debit. The bug wasn't here in `recompute_strategy_numerics`, but the
# only way a future regression — a partial recompute, a stale cached
# breakeven, a frontend that re-derives one field from a different
# source — can break out is if we don't shout when it does. So every
# call to `recompute_strategy_numerics` self-checks the width and
# breakeven invariants and stamps `numeric.invariant_violation` plus a
# human-readable reason on the output. The open path (db.py) and any
# display surface can refuse to use a numeric block that flags itself
# inconsistent.
#
# Tests live in tests/test_options_spread_math.py.

# Tolerance for the equality checks. Premium round-trips are 4 decimals,
# numerics round to 2 — a $0.01 / contract slop is the worst legitimate
# rounding error. A nickel of slack swallows that; anything beyond is a
# real divergence.
_INVARIANT_TOLERANCE_DOLLARS = 0.05


def _check_spread_invariants(
    strategy_type: str,
    numeric: dict,
    *,
    width_dollars_per_contract: float,
    expected_breakeven: Optional[float] = None,
    breakeven_lo: Optional[float] = None,
    breakeven_hi: Optional[float] = None,
    is_credit: bool = False,
) -> dict:
    """Verify cost/max_profit/breakeven came from one consistent (p_buy,
    p_sell) snapshot, and that max_loss matches cost on debit spreads.

    Returns the same `numeric` dict, with `invariant_violation` and an
    `invariant_violation_reason` key set if any check failed. Doesn't
    raise — the open path is responsible for refusing to fill on a
    flagged numeric block; the display path is responsible for showing
    a 'numbers don't reconcile' state instead of silently rendering them.
    """
    cost       = float(numeric.get("cost_per_contract") or 0)
    max_profit = numeric.get("max_profit_per_contract")
    max_loss   = float(numeric.get("max_loss_per_contract") or 0)
    breakevens = numeric.get("breakevens") or []

    violations: List[str] = []

    # Width check: cost + max_profit fills the whole spread width on a
    # bounded vertical / iron condor. (Long call has unbounded max_profit
    # and is skipped by the caller.)
    if max_profit is not None:
        # Credit strategies store cost as -credit; the width is always
        # |cost| + max_loss for an iron condor, |cost| + max_profit for
        # a vertical debit spread.
        if is_credit:
            actual_sum = round(float(max_profit) + max_loss, 2)
        else:
            actual_sum = round(cost + float(max_profit), 2)
        if abs(actual_sum - width_dollars_per_contract) > _INVARIANT_TOLERANCE_DOLLARS:
            violations.append(
                f"width invariant broke: "
                f"{'max_profit + max_loss' if is_credit else 'cost + max_profit'} "
                f"= {actual_sum:.2f}, expected {width_dollars_per_contract:.2f}"
            )

    # Debit spreads have max_loss == cost. Credit strategies don't, by
    # convention (cost stores -credit, max_loss stores the real risk).
    if not is_credit and abs(max_loss - cost) > _INVARIANT_TOLERANCE_DOLLARS:
        violations.append(
            f"max_loss invariant broke: "
            f"max_loss={max_loss:.2f} != cost={cost:.2f} on a debit strategy"
        )

    # Breakeven check — single-BE strategies (verticals).
    if expected_breakeven is not None:
        if not breakevens:
            violations.append("breakeven invariant broke: no breakevens reported")
        elif abs(breakevens[0] - expected_breakeven) > _INVARIANT_TOLERANCE_DOLLARS:
            violations.append(
                f"breakeven invariant broke: "
                f"breakeven={breakevens[0]:.2f}, expected {expected_breakeven:.2f}"
            )

    # Two-BE strategies (iron condor).
    if breakeven_lo is not None and breakeven_hi is not None:
        if len(breakevens) < 2:
            violations.append(
                f"breakeven invariant broke: expected 2 breakevens, "
                f"got {len(breakevens)}"
            )
        else:
            be_lo, be_hi = breakevens[0], breakevens[1]
            if abs(be_lo - breakeven_lo) > _INVARIANT_TOLERANCE_DOLLARS:
                violations.append(
                    f"breakeven_lo invariant broke: {be_lo:.2f} != "
                    f"{breakeven_lo:.2f}"
                )
            if abs(be_hi - breakeven_hi) > _INVARIANT_TOLERANCE_DOLLARS:
                violations.append(
                    f"breakeven_hi invariant broke: {be_hi:.2f} != "
                    f"{breakeven_hi:.2f}"
                )

    if violations:
        numeric["invariant_violation"]        = True
        numeric["invariant_violation_reason"] = (
            f"{strategy_type}: " + "; ".join(violations)
        )
        # Don't auto-flip `tradable` here — the caller (db.py open path)
        # decides how to handle a violation. Surfacing the flag is enough.
    else:
        numeric["invariant_violation"] = False

    return numeric


def recompute_strategy_numerics(strategy_type: str, legs: list) -> dict:
    """
    Recompute the `numeric` block for a strategy from live per-leg
    premiums. Each leg dict must carry `action`, `type`, `strike`, and
    `premium` (per-share dollars; the live mid or last from the chain).

    Returns the same shape options_analyzer's individual constructors
    write into `numeric`:
        cost_per_contract        — positive for debit, negative for credit
        max_profit_per_contract  — None when unbounded
        max_loss_per_contract
        breakevens               — list of price levels
        is_credit                — bool
        margin_required_per_contract — only set for credit strategies
        tradable                 — always True; gating happens elsewhere

    Raises ValueError when the strategy_type isn't supported or the
    legs don't match its expected shape.
    """
    st = (strategy_type or "").strip()

    if st == "Long Call":
        leg = _find_leg(legs, "BUY", "CALL")
        if leg is None:
            raise ValueError("Long Call: missing BUY CALL leg")
        p = _leg_premium(leg)
        K = _leg_strike(leg)
        cost = round(p * 100, 2)
        return {
            "cost_per_contract":       cost,
            "max_profit_per_contract": None,
            "max_loss_per_contract":   cost,
            "breakevens":              [round(K + p, 2)],
            "is_credit":               False,
            "tradable":                True,
        }

    if st == "Long Put":
        leg = _find_leg(legs, "BUY", "PUT")
        if leg is None:
            raise ValueError("Long Put: missing BUY PUT leg")
        p = _leg_premium(leg)
        K = _leg_strike(leg)
        cost = round(p * 100, 2)
        return {
            "cost_per_contract":       cost,
            "max_profit_per_contract": round((K - p) * 100, 2),
            "max_loss_per_contract":   cost,
            "breakevens":              [round(K - p, 2)],
            "is_credit":               False,
            "tradable":                True,
        }

    if st == "Bull Call Spread":
        buy  = _find_leg(legs, "BUY",  "CALL")
        sell = _find_leg(legs, "SELL", "CALL")
        if buy is None or sell is None:
            raise ValueError("Bull Call Spread: needs BUY CALL + SELL CALL")
        p_buy, p_sell   = _leg_premium(buy),  _leg_premium(sell)
        K_buy, K_sell   = _leg_strike(buy),   _leg_strike(sell)
        net_debit = round((p_buy - p_sell) * 100, 2)
        max_gain  = round((K_sell - K_buy - p_buy + p_sell) * 100, 2)
        breakeven = round(K_buy + p_buy - p_sell, 2)
        numeric = {
            "cost_per_contract":       net_debit,
            "max_profit_per_contract": max_gain,
            "max_loss_per_contract":   net_debit,
            "breakevens":              [breakeven],
            "is_credit":               False,
            "tradable":                True,
        }
        return _check_spread_invariants(
            st, numeric,
            width_dollars_per_contract=round((K_sell - K_buy) * 100, 2),
            expected_breakeven=round(K_buy + net_debit / 100.0, 2),
        )

    if st == "Bear Put Spread":
        buy  = _find_leg(legs, "BUY",  "PUT")
        sell = _find_leg(legs, "SELL", "PUT")
        if buy is None or sell is None:
            raise ValueError("Bear Put Spread: needs BUY PUT + SELL PUT")
        p_buy, p_sell = _leg_premium(buy), _leg_premium(sell)
        K_buy, K_sell = _leg_strike(buy),  _leg_strike(sell)
        net_debit = round((p_buy - p_sell) * 100, 2)
        max_gain  = round((K_buy - K_sell - p_buy + p_sell) * 100, 2)
        breakeven = round(K_buy - p_buy + p_sell, 2)
        numeric = {
            "cost_per_contract":       net_debit,
            "max_profit_per_contract": max_gain,
            "max_loss_per_contract":   net_debit,
            "breakevens":              [breakeven],
            "is_credit":               False,
            "tradable":                True,
        }
        return _check_spread_invariants(
            st, numeric,
            width_dollars_per_contract=round((K_buy - K_sell) * 100, 2),
            expected_breakeven=round(K_buy - net_debit / 100.0, 2),
        )

    if st == "Iron Condor":
        buy_put   = _find_leg(legs, "BUY",  "PUT")
        sell_put  = _find_leg(legs, "SELL", "PUT")
        sell_call = _find_leg(legs, "SELL", "CALL")
        buy_call  = _find_leg(legs, "BUY",  "CALL")
        if not all([buy_put, sell_put, sell_call, buy_call]):
            raise ValueError("Iron Condor: needs BUY/SELL PUT + SELL/BUY CALL")
        p_pb, p_ps = _leg_premium(buy_put),  _leg_premium(sell_put)
        p_cs, p_cb = _leg_premium(sell_call), _leg_premium(buy_call)
        K_put_sell  = _leg_strike(sell_put)
        K_call_sell = _leg_strike(sell_call)
        net_credit = round((p_ps - p_pb + p_cs - p_cb) * 100, 2)
        max_loss   = round(((K_call_sell - K_put_sell)
                            - (p_ps - p_pb + p_cs - p_cb)) * 100, 2)
        be_down = round(K_put_sell  - net_credit / 100, 2)
        be_up   = round(K_call_sell + net_credit / 100, 2)
        numeric = {
            "cost_per_contract":              -net_credit,
            "max_profit_per_contract":         net_credit,
            "max_loss_per_contract":           max_loss,
            "breakevens":                      [be_down, be_up],
            "is_credit":                       True,
            "margin_required_per_contract":    max_loss,
            "tradable":                        True,
        }
        return _check_spread_invariants(
            st, numeric,
            width_dollars_per_contract=round((K_call_sell - K_put_sell) * 100, 2),
            breakeven_lo=round(K_put_sell  - net_credit / 100.0, 2),
            breakeven_hi=round(K_call_sell + net_credit / 100.0, 2),
            is_credit=True,
        )

    if st == "Long Straddle":
        buy_call = _find_leg(legs, "BUY", "CALL")
        buy_put  = _find_leg(legs, "BUY", "PUT")
        if buy_call is None or buy_put is None:
            raise ValueError("Long Straddle: needs BUY CALL + BUY PUT")
        p_call, p_put = _leg_premium(buy_call), _leg_premium(buy_put)
        K = _leg_strike(buy_call)  # straddles are ATM — both legs same strike
        total = round((p_call + p_put) * 100, 2)
        return {
            "cost_per_contract":       total,
            "max_profit_per_contract": None,
            "max_loss_per_contract":   total,
            "breakevens":              [
                round(K - p_call - p_put, 2),
                round(K + p_call + p_put, 2),
            ],
            "is_credit":               False,
            "tradable":                True,
        }

    if st == "Cash-Secured Put":
        leg = _find_leg(legs, "SELL", "PUT")
        if leg is None:
            raise ValueError("Cash-Secured Put: missing SELL PUT leg")
        p = _leg_premium(leg)
        K = _leg_strike(leg)
        credit       = round(p * 100, 2)
        max_loss_csp = round((K - p) * 100, 2)
        cash_required = round(K * 100, 2)
        return {
            "cost_per_contract":              -credit,
            "max_profit_per_contract":         credit,
            "max_loss_per_contract":           max_loss_csp,
            "breakevens":                      [round(K - p, 2)],
            "is_credit":                       True,
            "margin_required_per_contract":    cash_required,
            "tradable":                        True,
        }

    raise ValueError(f"unsupported strategy_type for recompute: {strategy_type!r}")


# ─── Full options report ──────────────────────────────────────────────────────

def generate_options_report(
    predictions: dict,
    df: pd.DataFrame,
    focus_horizon: str = "1 Month",
    *,
    symbol: Optional[str] = None,
) -> dict:
    """
    Build options strategy recommendations for all horizons.
    Returns dict keyed by horizon.

    When `symbol` is supplied, each strategy is reshaped against the
    live options chain via `options_pricer.snap_strategy_to_real_chain`
    — expiry snaps to nearest chain date, strikes snap to nearest real
    strikes, premiums replaced with live mids, numerics recomputed.
    The result: what the user sees on /prediqt is exactly what they'll
    fill at when they take the play. No drift between quote and fill.

    `symbol` is keyword-only and optional so legacy call sites that
    don't have it (or want to skip the live-chain pass for tests)
    keep working with the theoretical Black-Scholes output.
    """
    report = {}
    for horizon, pred in predictions.items():
        try:
            report[horizon] = build_strategy(horizon, pred, df)
        except Exception as e:
            report[horizon] = {"error": str(e), "strategy": "N/A"}

    # Live-chain pass. Only run when we have a symbol and the
    # entry's actually a strategy (not an error sentinel).
    if symbol:
        try:
            from options_pricer import snap_strategy_to_real_chain
        except Exception:  # noqa: BLE001
            snap_strategy_to_real_chain = None  # type: ignore[assignment]

        if snap_strategy_to_real_chain is not None:
            for horizon, strategy in list(report.items()):
                if not isinstance(strategy, dict):
                    continue
                if "error" in strategy or strategy.get("strategy") == "N/A":
                    continue
                try:
                    report[horizon] = snap_strategy_to_real_chain(strategy, symbol)
                except Exception as exc:  # noqa: BLE001
                    # Snap failed — keep theoretical strategy but flag
                    # untradeable so the open path refuses cleanly
                    # instead of trying to fill a fake.
                    numeric = dict(strategy.get("numeric") or {})
                    numeric["tradable"] = False
                    numeric["untradeable_reason"] = (
                        f"chain snap failed at asking-flow time: {exc}"
                    )
                    report[horizon] = {**strategy, "numeric": numeric}

    # Mark the focus horizon
    if focus_horizon in report:
        report[focus_horizon]["is_primary"] = True

    return report
