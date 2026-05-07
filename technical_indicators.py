"""Standard-issue technical indicators computed from a daily-close series.

Used by the /api/stock/{symbol} aggregator to surface RSI, MACD, SMA,
and 30-day historical volatility on the SEO pages without paying for
a third-party indicator feed. Inputs are plain Python lists/floats —
no pandas dependency at this boundary, even though everything else in
the project leans on pandas (keeps this importable from contexts that
haven't paid the pandas-import cost yet).

All functions return None when there isn't enough history to compute
the indicator (e.g. RSI(14) needs ≥15 closes). The page renders the
indicator's section empty rather than zero-fill, since "RSI = 0"
visually reads as "extremely oversold" and would be misleading.
"""
from __future__ import annotations

import math
from typing import Optional


def sma(closes: list[float], period: int) -> Optional[float]:
    """Simple moving average of the last `period` closes."""
    if not closes or len(closes) < period or period <= 0:
        return None
    window = closes[-period:]
    return round(sum(window) / period, 4)


def rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Wilder's RSI on `closes`. Returns the latest value in 0–100.

    Uses the modified RSI smoother (exponential, not simple average)
    to match what most charting platforms ship by default — picking
    a different definition would create a "your RSI doesn't match
    TradingView" complaint we don't need.
    """
    if not closes or len(closes) < period + 1:
        return None

    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(0.0, delta))
        losses.append(max(0.0, -delta))

    # Initial average gain/loss = simple mean of first `period` deltas.
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder smoothing through the rest of the series.
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def ema(closes: list[float], period: int) -> Optional[list[float]]:
    """Exponential moving average series (full length). Returns None when
    there isn't enough history. Used internally by macd()."""
    if not closes or len(closes) < period or period <= 0:
        return None
    k = 2.0 / (period + 1.0)
    out: list[float] = []
    # Seed with SMA of first `period` values (matches TA-Lib convention).
    seed = sum(closes[:period]) / period
    out.extend([float("nan")] * (period - 1))
    out.append(seed)
    prev = seed
    for c in closes[period:]:
        cur = c * k + prev * (1.0 - k)
        out.append(cur)
        prev = cur
    return out


def macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[dict]:
    """Standard MACD(12, 26, 9). Returns the latest tuple as a dict:

        { "macd": float, "signal": float, "hist": float }

    Where macd = EMA(fast) − EMA(slow), signal = EMA(macd_series, signal_period),
    and hist = macd − signal. Requires ~slow + signal closes to settle.
    """
    if not closes or len(closes) < slow + signal:
        return None
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    if ema_fast is None or ema_slow is None:
        return None

    # MACD line = EMA(fast) − EMA(slow), aligned to the slow start.
    macd_line: list[float] = []
    for i in range(len(closes)):
        if math.isnan(ema_fast[i]) or math.isnan(ema_slow[i]):
            macd_line.append(float("nan"))
        else:
            macd_line.append(ema_fast[i] - ema_slow[i])

    # Signal line — EMA(macd_line, signal_period). EMA() needs a clean
    # tail without NaNs, so seed only after the slow EMA has settled.
    valid_start = next(
        (i for i, v in enumerate(macd_line) if not math.isnan(v)),
        None,
    )
    if valid_start is None:
        return None
    macd_tail = macd_line[valid_start:]
    if len(macd_tail) < signal:
        return None
    signal_series = ema(macd_tail, signal)
    if signal_series is None or math.isnan(signal_series[-1]):
        return None

    macd_now   = macd_line[-1]
    signal_now = signal_series[-1]
    if math.isnan(macd_now) or math.isnan(signal_now):
        return None
    hist_now = macd_now - signal_now

    return {
        "macd":   round(macd_now,   4),
        "signal": round(signal_now, 4),
        "hist":   round(hist_now,   4),
    }


def historical_volatility(closes: list[float], window: int = 30) -> Optional[float]:
    """Sample standard deviation of daily log returns over the last
    `window` days. Returned as a plain decimal (e.g. 0.018 = 1.8% daily
    σ), unannualised — page-side code can ×√252 if it wants annual.
    """
    if not closes or len(closes) < window + 1:
        return None
    tail = closes[-(window + 1):]
    log_returns: list[float] = []
    for i in range(1, len(tail)):
        prev = tail[i - 1]
        cur = tail[i]
        if prev <= 0 or cur <= 0:
            return None
        log_returns.append(math.log(cur / prev))
    if not log_returns:
        return None
    mean = sum(log_returns) / len(log_returns)
    var = sum((r - mean) ** 2 for r in log_returns) / max(1, len(log_returns) - 1)
    return round(math.sqrt(var), 6)


def compute_all(closes: list[float]) -> dict:
    """Compute the full technical bundle in one shot. Each field is
    None when there isn't enough history.

    Returns:
        {
            "rsi_14":       float | None,
            "sma_50":       float | None,
            "sma_200":      float | None,
            "macd":         {macd, signal, hist} | None,
            "volatility_30d": float | None,
        }
    """
    return {
        "rsi_14":         rsi(closes, 14),
        "sma_50":         sma(closes, 50),
        "sma_200":        sma(closes, 200),
        "macd":           macd(closes),
        "volatility_30d": historical_volatility(closes, 30),
    }


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Trivial sanity check — synthetic random walk should give RSI around 50.
    import random
    random.seed(42)
    closes = [100.0]
    for _ in range(300):
        closes.append(closes[-1] * (1 + random.gauss(0, 0.015)))
    print(compute_all(closes))
