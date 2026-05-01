"""
options_greeks.py
-----------------
Black-Scholes greeks for the UI's display panels.

Methodology § 4.5 is explicit about scope: the system trades on live
mid (or last) per leg. It does NOT use greek-derived estimates to
fill, mark to market, or close positions. These functions exist so the
panel can give the user useful context — "this option moves about
$0.40 per $1 in spot, and decays $0.05/day right now" — without
introducing a second pricing source the trade math could disagree with.

Pure-python: no scipy or numpy dependency. The standard normal CDF
is computed via `math.erf`, which is in the stdlib.

Convention:
  S = underlying spot price
  K = strike
  σ = annualized implied volatility (decimal — 0.30, not 30)
  T = time to expiry in years (e.g. 30 days = 30/365)
  r = risk-free rate (decimal — 0.045 ≈ current 3-month T-bill)

Greeks returned per share. Per-contract values multiply by 100 (the
standard equity option multiplier).
"""

from __future__ import annotations

import math
from typing import Optional


# 3-month T-bill yield ballpark for 2026. Updated when the rate environment
# moves materially. Greeks aren't sensitive enough to this to require live
# rate fetching; a stable constant keeps the display deterministic.
DEFAULT_RISK_FREE_RATE = 0.045


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _safe_inputs(
    spot: Optional[float],
    strike: Optional[float],
    iv: Optional[float],
    days_to_expiry: Optional[int],
    rate: float,
) -> Optional[tuple]:
    """
    Common input validation. Returns (S, K, sigma, T, r) or None if
    any input is missing or pathological. Greeks aren't defined when
    sigma or T is zero — the function returns None and callers render
    a dash rather than a wrong number.
    """
    if spot is None or strike is None or iv is None or days_to_expiry is None:
        return None
    try:
        S = float(spot)
        K = float(strike)
        sigma = float(iv)
        dte = int(days_to_expiry)
    except (TypeError, ValueError):
        return None
    if S <= 0 or K <= 0 or sigma <= 0 or dte <= 0:
        return None
    T = dte / 365.0
    return S, K, sigma, T, float(rate)


def _bs_d1(S: float, K: float, sigma: float, T: float, r: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def bs_delta(
    spot: Optional[float],
    strike: Optional[float],
    iv: Optional[float],
    days_to_expiry: Optional[int],
    leg_type: str,
    rate: float = DEFAULT_RISK_FREE_RATE,
) -> Optional[float]:
    """
    Per-share delta. Roughly: how much the option's value changes per
    $1 move in the underlying. Calls: 0 (deep OTM) → 1 (deep ITM).
    Puts: 0 (deep OTM) → −1 (deep ITM). At-the-money is near ±0.5.

    Returns None when inputs are missing or volatility/time is zero.
    """
    inp = _safe_inputs(spot, strike, iv, days_to_expiry, rate)
    if inp is None:
        return None
    S, K, sigma, T, _ = inp
    d1 = _bs_d1(S, K, sigma, T, rate)
    if (leg_type or "").upper() == "CALL":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def bs_theta(
    spot: Optional[float],
    strike: Optional[float],
    iv: Optional[float],
    days_to_expiry: Optional[int],
    leg_type: str,
    rate: float = DEFAULT_RISK_FREE_RATE,
) -> Optional[float]:
    """
    Per-day theta — dollar premium decay per share per calendar day.
    Almost always negative for long positions (you're paying for time);
    positive for short positions (you're collecting).

    Standard Black-Scholes theta is per-year; we divide by 365 to give
    the user a daily number that matches how they think about decay.
    """
    inp = _safe_inputs(spot, strike, iv, days_to_expiry, rate)
    if inp is None:
        return None
    S, K, sigma, T, _ = inp
    d1 = _bs_d1(S, K, sigma, T, rate)
    d2 = d1 - sigma * math.sqrt(T)

    common = -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    discount = rate * K * math.exp(-rate * T)

    if (leg_type or "").upper() == "CALL":
        annual_theta = common - discount * _norm_cdf(d2)
    else:
        annual_theta = common + discount * _norm_cdf(-d2)

    return annual_theta / 365.0


def bs_iv_from_chain(iv_from_chain: Optional[float]) -> Optional[float]:
    """
    Pass-through that drops obviously bad IV readings (negative,
    inf, > 5.0). yfinance occasionally returns sentinel values like
    1e-5 or huge numbers for illiquid strikes; we just refuse those
    rather than computing greeks against garbage.
    """
    if iv_from_chain is None:
        return None
    try:
        v = float(iv_from_chain)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0.001 or v > 5.0:
        return None
    return v
