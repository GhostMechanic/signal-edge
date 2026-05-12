"""
watchlist.py
───────────────────────────────────────────────────────────────────────────────
Persistent watchlist management and quick signal generation for Prediqt.
Stores user's tracked symbols and provides fast technical analysis for
watchlist cards without running full model training.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from data_fetcher import fetch_stock_data


# ─── Configuration ────────────────────────────────────────────────────────────
WATCHLIST_FILE = os.path.join(".predictions", "watchlist.json")


# ─── Watchlist Persistence ────────────────────────────────────────────────────

def _ensure_watchlist_dir():
    """Ensure .predictions directory exists."""
    os.makedirs(".predictions", exist_ok=True)


def load_watchlist() -> list[str]:
    """
    Load watchlist from persistent storage.
    Returns: List of uppercase symbols, or empty list if file missing/corrupt.
    """
    try:
        _ensure_watchlist_dir()
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [s.upper().strip() for s in data if s]
        return []
    except Exception:
        return []


def save_watchlist(symbols: list[str]) -> list[str]:
    """
    Persist watchlist to disk.
    Args:
        symbols: List of stock symbols to save
    Returns: The cleaned, deduplicated list that was saved
    """
    try:
        _ensure_watchlist_dir()
        clean = [s.upper().strip() for s in symbols if s]
        clean = list(dict.fromkeys(clean))  # Remove duplicates, preserve order
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(clean, f)
        return clean
    except Exception:
        return list(symbols)


def add_to_watchlist(symbol: str) -> list[str]:
    """
    Add a symbol to watchlist and persist.
    Returns: Updated watchlist
    """
    watchlist = load_watchlist()
    sym = symbol.upper().strip()
    if sym and sym not in watchlist:
        watchlist.append(sym)
    save_watchlist(watchlist)
    return watchlist


def remove_from_watchlist(symbol: str) -> list[str]:
    """
    Remove a symbol from watchlist and persist.
    Returns: Updated watchlist
    """
    watchlist = load_watchlist()
    sym = symbol.upper().strip()
    watchlist = [s for s in watchlist if s != sym]
    save_watchlist(watchlist)
    return watchlist


# ─── Quick Signal Generation ──────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_quick_signal(symbol: str) -> Dict[str, Any]:
    """
    Fast technical analysis signal for watchlist display.
    Does NOT run full model training—just basic technicals.

    Returns dict with:
        - symbol, price, change_1d/5d/21d (%)
        - rsi, ma50_dist, ma200_dist (%), vol_ratio
        - signal (BUY/SELL/HOLD), confidence (0-100)
        - error (None or str)
    """
    try:
        sym = symbol.upper().strip()

        # Fetch 1 year of data for technicals
        df = fetch_stock_data(sym, period="1y")

        if df.empty or len(df) < 50:
            return {
                "symbol": sym,
                "price": None,
                "change_1d": None,
                "change_5d": None,
                "change_21d": None,
                "rsi": None,
                "ma50_dist": None,
                "ma200_dist": None,
                "vol_ratio": None,
                "signal": "HOLD",
                "confidence": 0,
                "error": "Insufficient data",
            }

        # Current price and changes
        current_price = df["Close"].iloc[-1]
        change_1d = ((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100) if len(df) > 1 else 0
        change_5d = ((df["Close"].iloc[-1] - df["Close"].iloc[-5]) / df["Close"].iloc[-5] * 100) if len(df) > 5 else 0
        change_21d = ((df["Close"].iloc[-1] - df["Close"].iloc[-21]) / df["Close"].iloc[-21] * 100) if len(df) > 21 else 0

        # Moving averages
        ma50 = df["Close"].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
        ma200 = df["Close"].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None

        ma50_dist = ((current_price - ma50) / ma50 * 100) if ma50 and not pd.isna(ma50) else 0
        ma200_dist = ((current_price - ma200) / ma200 * 100) if ma200 and not pd.isna(ma200) else 0

        # RSI(14)
        rsi = _calculate_rsi(df["Close"], period=14)

        # Volume ratio
        vol_ma_20 = df["Volume"].rolling(window=20).mean().iloc[-1]
        current_vol = df["Volume"].iloc[-1]
        vol_ratio = (current_vol / vol_ma_20) if vol_ma_20 > 0 else 1.0

        # ── Multi-factor scoring system ────────────────────────────────
        # Score from -100 (strong sell) to +100 (strong buy) across 5 factors
        score = 0.0
        factors = 0

        # Factor 1: RSI momentum (oversold = bullish, overbought = bearish)
        if rsi is not None:
            if rsi < 30:
                score += 30
            elif rsi < 40:
                score += 15
            elif rsi < 50:
                score += 5
            elif rsi > 80:
                score -= 30
            elif rsi > 70:
                score -= 15
            elif rsi > 60:
                score -= 5
            factors += 1

        # Factor 2: MA50 trend (above = bullish, below = bearish)
        if ma50_dist is not None:
            _ma50_contrib = np.clip(ma50_dist * 2, -25, 25)
            score += _ma50_contrib
            factors += 1

        # Factor 3: MA200 trend (long-term positioning)
        if ma200_dist is not None:
            _ma200_contrib = np.clip(ma200_dist, -20, 20)
            score += _ma200_contrib
            factors += 1

        # Factor 4: Short-term momentum (5d change)
        if change_5d:
            _mom_contrib = np.clip(change_5d * 3, -20, 20)
            score += _mom_contrib
            factors += 1

        # Factor 5: Volume confirmation (high volume on up move = bullish)
        if vol_ratio > 1.3 and change_1d > 0:
            score += 10
        elif vol_ratio > 1.3 and change_1d < 0:
            score -= 10
        factors += 1

        # Normalize to signal + confidence
        if factors > 0:
            score = score / factors * (factors / 5)  # scale by factor coverage

        if score > 8:
            signal = "BUY"
        elif score < -8:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Confidence: magnitude of score mapped to 20-90 range
        confidence = int(min(90, max(20, 30 + abs(score) * 1.5)))

        return {
            "symbol": sym,
            "price": round(current_price, 2),
            "change_1d": round(change_1d, 2),
            "change_5d": round(change_5d, 2),
            "change_21d": round(change_21d, 2),
            "rsi": round(rsi, 1) if rsi is not None else None,
            "ma50_dist": round(ma50_dist, 2) if ma50_dist is not None else None,
            "ma200_dist": round(ma200_dist, 2) if ma200_dist is not None else None,
            "vol_ratio": round(vol_ratio, 2),
            "signal": signal,
            "confidence": max(0, min(100, confidence)),
            "error": None,
        }

    except Exception as e:
        return {
            "symbol": symbol.upper().strip(),
            "price": None,
            "change_1d": None,
            "change_5d": None,
            "change_21d": None,
            "rsi": None,
            "ma50_dist": None,
            "ma200_dist": None,
            "vol_ratio": None,
            "signal": "HOLD",
            "confidence": 0,
            "error": str(e),
        }


def _calculate_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    """Calculate RSI(period) for a price series."""
    try:
        if len(prices) < period + 1:
            return None

        deltas = prices.diff()
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = 100.0 - 100.0 / (1.0 + rs)

        for i in range(period + 1, len(prices)):
            delta = deltas.iloc[i]
            if delta > 0:
                up = (up * (period - 1) + delta) / period
                down = (down * (period - 1) + 0) / period
            else:
                up = (up * (period - 1) + 0) / period
                down = (down * (period - 1) - delta) / period

            rs = up / down if down != 0 else 0
            rsi = 100.0 - 100.0 / (1.0 + rs)

        return rsi
    except Exception:
        return None
