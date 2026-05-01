"""
indicators.py
-------------
Canonical implementations of technical indicators used by the trade-plan
and decision modules. Pure functions; no DB, no I/O, no global state.

Existing inline ATR calculations (in `analyzer.py` and `api/main.py`) use
a 20-period simple moving average — fine for general analysis, but the
methodology anchor (§ 4.4) specifies **Wilder's 14-period ATR** for stop
placement. This module is the single source of truth for that number, so
auditors recomputing a historical stop arrive at the same value the
worker recorded.

Reference: J. Welles Wilder Jr., "New Concepts in Technical Trading
Systems" (1978). The ATR formula here matches the canonical Wilder
definition that TA-Lib, TradingView, and most charting platforms publish
as "ATR(14)".
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pandas as pd

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


# ─── True Range ───────────────────────────────────────────────────────────────

def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Per-bar True Range. The first bar uses high-low only (no prior close)."""
    n = len(high)
    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        prev_close = close[i - 1]
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - prev_close),
            abs(low[i] - prev_close),
        )
    return tr


# ─── ATR (Wilder's smoothing) ─────────────────────────────────────────────────

def atr(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 14,
) -> float:
    """
    Compute the latest Wilder's ATR(period) value.

    The series must be chronologically ordered (oldest first) and contain at
    least `period + 1` bars — one prior close to seed the True Range, plus
    `period` TR values to seed the SMA used as the initial ATR.

    For ATR(14) that means 15 bars minimum.

    Wilder's update rule (after the SMA seed):
        ATR_t = ((period - 1) * ATR_{t-1} + TR_t) / period

    Equivalent to an EMA with smoothing factor α = 1/period.

    Parameters
    ----------
    high, low, close : sequences of float, same length, chronologically
                       ordered (oldest first).
    period           : ATR period. Defaults to 14 per methodology § 4.4.

    Returns
    -------
    float — the most recent ATR value.

    Raises
    ------
    ValueError if inputs are misaligned or there aren't enough bars.
    """
    h = np.asarray(high,  dtype=float)
    l = np.asarray(low,   dtype=float)
    c = np.asarray(close, dtype=float)

    if not (len(h) == len(l) == len(c)):
        raise ValueError(
            f"high/low/close must be equal length; got {len(h)}/{len(l)}/{len(c)}"
        )
    if period < 1:
        raise ValueError(f"period must be >= 1; got {period}")
    if len(h) < period + 1:
        raise ValueError(
            f"ATR({period}) requires at least {period + 1} bars; got {len(h)}"
        )

    tr = _true_range(h, l, c)

    # Seed: SMA of the first `period` TR values.
    atr_val = float(tr[:period].mean())

    # Wilder's recursive smoothing for every bar past the seed window.
    for i in range(period, len(tr)):
        atr_val = ((period - 1) * atr_val + tr[i]) / period

    return atr_val


def atr_from_ohlc_df(
    df: pd.DataFrame,
    period: int = 14,
    *,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> float:
    """
    Convenience wrapper for the project's standard OHLC DataFrame shape
    (columns 'High', 'Low', 'Close', chronologically indexed). Matches what
    `data_fetcher.py` produces.
    """
    return atr(
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        period=period,
    )


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Hand-verified example — small, deterministic, easy to recompute by hand
    # if the formula ever drifts. Bars are H/L/C only.
    bars = [
        # (high, low, close)
        (10.0, 9.0,  9.5),    # bar  0  TR = 1.0  (H-L; no prior close)
        (11.0, 9.5,  10.5),   # bar  1  TR = max(1.5, 1.5, 0.0) = 1.5
        (12.0, 10.0, 11.5),   # bar  2  TR = max(2.0, 1.5, 0.5) = 2.0
        (11.5, 10.0, 10.5),   # bar  3  TR = max(1.5, 0.0, 1.5) = 1.5
        (12.5, 10.5, 12.0),   # bar  4  TR = max(2.0, 2.0, 0.0) = 2.0
        (13.0, 11.5, 12.5),   # bar  5  TR = max(1.5, 1.0, 0.5) = 1.5
        (13.5, 12.0, 13.0),   # bar  6  TR = max(1.5, 1.0, 0.5) = 1.5
        (14.0, 12.5, 13.5),   # bar  7  TR = max(1.5, 1.0, 0.5) = 1.5
        (14.5, 13.0, 14.0),   # bar  8  TR = max(1.5, 1.0, 0.5) = 1.5
        (15.0, 13.5, 14.5),   # bar  9  TR = max(1.5, 1.0, 0.5) = 1.5
        (15.5, 14.0, 15.0),   # bar 10  TR = max(1.5, 1.0, 0.5) = 1.5
        (16.0, 14.5, 15.5),   # bar 11  TR = max(1.5, 1.0, 0.5) = 1.5
        (16.5, 15.0, 16.0),   # bar 12  TR = max(1.5, 1.0, 0.5) = 1.5
        (17.0, 15.5, 16.5),   # bar 13  TR = max(1.5, 1.0, 0.5) = 1.5
        (17.5, 16.0, 17.0),   # bar 14  TR = max(1.5, 1.0, 0.5) = 1.5
    ]
    h, l, c = zip(*bars)

    # Expected ATR(14):
    #   First 14 TRs (bars 0..13) = [1.0, 1.5, 2.0, 1.5, 2.0, 1.5, 1.5, 1.5,
    #                                1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    #     • bars 0..4 contribute: 1.0 + 1.5 + 2.0 + 1.5 + 2.0 = 8.0
    #     • bars 5..13 contribute: 9 × 1.5 = 13.5
    #     • sum = 21.5; SMA seed at bar 13 = 21.5 / 14 ≈ 1.535714
    #   Bar 14 TR = max(1.5, 1.0, 0.5) = 1.5
    #   ATR(14) = (13 * 1.535714 + 1.5) / 14
    #           = (19.964286 + 1.5) / 14
    #           = 21.464286 / 14
    #           ≈ 1.533163
    expected = ((13 * (21.5 / 14)) + 1.5) / 14
    got = atr(h, l, c, period=14)

    print(f"ATR(14) hand-computed: {expected:.6f}")
    print(f"ATR(14) module result: {got:.6f}")
    assert abs(got - expected) < 1e-9, f"ATR mismatch: {got} vs {expected}"

    # Edge case: too few bars.
    try:
        atr([1.0] * 10, [1.0] * 10, [1.0] * 10, period=14)
    except ValueError as e:
        print(f"Insufficient-bars guard: {e}")
    else:
        raise AssertionError("expected ValueError for insufficient bars")

    # Edge case: misaligned inputs.
    try:
        atr([1.0, 2.0], [1.0], [1.0, 2.0])
    except ValueError as e:
        print(f"Misaligned-inputs guard: {e}")
    else:
        raise AssertionError("expected ValueError for misaligned inputs")

    print("indicators.py self-test OK")
