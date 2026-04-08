"""
data_fetcher.py
---------------
Downloads historical OHLCV data + market context (SPY, VIX) and engineers
an extended set of features for the ensemble ML models.

New in v2:
  • SPY relative strength (alpha vs market)
  • VIX-based market regime (high fear / normal / complacent)
  • Rolling beta to SPY
  • Sector ETF relative strength (optional, best-effort)
  • Distance to recent earnings (proxy via 90-day cycle)
  • Enhanced volatility regime features
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional
from datetime import datetime

# ─── Horizons ─────────────────────────────────────────────────────────────────
HORIZONS = {
    "1 Week":    5,
    "1 Month":  21,
    "1 Quarter": 63,
    "1 Year":  252,
}
HORIZON_KEYS = list(HORIZONS.keys())

# ─── Sector ETF map ───────────────────────────────────────────────────────────
SECTOR_ETFS = {
    "Technology":            "XLK",
    "Health Care":           "XLV",
    "Financials":            "XLF",
    "Consumer Discretionary":"XLY",
    "Consumer Staples":      "XLP",
    "Industrials":           "XLI",
    "Energy":                "XLE",
    "Utilities":             "XLU",
    "Real Estate":           "XLRE",
    "Materials":             "XLB",
    "Communication Services":"XLC",
}


# ─── Data Download ────────────────────────────────────────────────────────────

def fetch_stock_data(symbol: str, period: str = "7y") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for '{symbol}'. Check the ticker symbol.")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


def fetch_stock_info(symbol: str) -> dict:
    try:
        info = yf.Ticker(symbol).info
        return {
            "name":        info.get("longName", symbol),
            "sector":      info.get("sector", "N/A"),
            "industry":    info.get("industry", "N/A"),
            "market_cap":  info.get("marketCap"),
            "pe_ratio":    info.get("trailingPE"),
            "52w_high":    info.get("fiftyTwoWeekHigh"),
            "52w_low":     info.get("fiftyTwoWeekLow"),
            "avg_volume":  info.get("averageVolume"),
            "beta":        info.get("beta"),
            "description": info.get("longBusinessSummary", ""),
        }
    except Exception:
        return {"name": symbol, "sector": "N/A", "industry": "N/A",
                "market_cap": None, "pe_ratio": None, "52w_high": None,
                "52w_low": None, "avg_volume": None, "beta": None, "description": ""}


def fetch_fundamentals(symbol: str) -> dict:
    """
    Fetch fundamental data from yfinance for use as ML features.
    Returns a dict of scalar values that apply to the most recent period.
    Gracefully returns empty dict on failure.
    """
    try:
        tk = yf.Ticker(symbol)
        info = tk.info or {}

        fundamentals = {}

        # Valuation
        fundamentals["pe_ratio"]       = info.get("trailingPE")
        fundamentals["forward_pe"]     = info.get("forwardPE")
        fundamentals["peg_ratio"]      = info.get("pegRatio")
        fundamentals["pb_ratio"]       = info.get("priceToBook")
        fundamentals["ps_ratio"]       = info.get("priceToSalesTrailing12Months")
        fundamentals["ev_ebitda"]      = info.get("enterpriseToEbitda")

        # Profitability
        fundamentals["profit_margin"]  = info.get("profitMargins")
        fundamentals["oper_margin"]    = info.get("operatingMargins")
        fundamentals["roe"]            = info.get("returnOnEquity")
        fundamentals["roa"]            = info.get("returnOnAssets")
        fundamentals["gross_margin"]   = info.get("grossMargins")

        # Growth
        fundamentals["rev_growth"]     = info.get("revenueGrowth")
        fundamentals["earn_growth"]    = info.get("earningsGrowth")
        fundamentals["earn_qtr_growth"] = info.get("earningsQuarterlyGrowth")

        # Financial health
        fundamentals["debt_to_equity"] = info.get("debtToEquity")
        fundamentals["current_ratio"]  = info.get("currentRatio")
        fundamentals["quick_ratio"]    = info.get("quickRatio")

        # Analyst sentiment
        fundamentals["target_mean"]    = info.get("targetMeanPrice")
        fundamentals["target_low"]     = info.get("targetLowPrice")
        fundamentals["target_high"]    = info.get("targetHighPrice")
        fundamentals["recommend_score"] = info.get("recommendationMean")  # 1=strong buy, 5=sell
        fundamentals["n_analysts"]     = info.get("numberOfAnalystOpinions")

        # Short interest
        fundamentals["short_pct"]      = info.get("shortPercentOfFloat")

        # Dividend
        fundamentals["div_yield"]      = info.get("dividendYield")

        # Clean: replace None with NaN
        for k, v in fundamentals.items():
            if v is None:
                fundamentals[k] = float("nan")

        return fundamentals

    except Exception:
        return {}


def _fetch_series(symbol: str, period: str = "7y") -> pd.Series:
    """Fetch a single close-price series; return empty Series on failure."""
    try:
        t  = yf.Ticker(symbol)
        df = t.history(period=period, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df["Close"].rename(symbol)
    except Exception:
        return pd.Series(dtype=float)


# ─── Technical Indicators ─────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series):
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal


def _bollinger(series: pd.Series, period: int = 20):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
    bw    = (upper - lower) / (sma + 1e-9)
    return upper, lower, pct_b, bw


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, pc = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period-1, min_periods=period).mean()


def _stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    lo  = df["Low"].rolling(k).min()
    hi  = df["High"].rolling(k).max()
    sk  = 100 * (df["Close"] - lo) / (hi - lo + 1e-9)
    return sk, sk.rolling(d).mean()


def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi = df["High"].rolling(period).max()
    lo = df["Low"].rolling(period).min()
    return -100 * (hi - df["Close"]) / (hi - lo + 1e-9)


def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-9)


def _rolling_beta(stock_rets: pd.Series, market_rets: pd.Series, window: int = 63) -> pd.Series:
    """Rolling beta of stock vs market."""
    cov = stock_rets.rolling(window).cov(market_rets)
    var = market_rets.rolling(window).var()
    return cov / (var + 1e-9)


# ─── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    spy_close: Optional[pd.Series] = None,
    vix_close: Optional[pd.Series] = None,
    sector_close: Optional[pd.Series] = None,
    fundamentals: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build feature matrix. spy_close / vix_close / sector_close are optional
    enrichment series aligned to df's index.
    fundamentals: optional dict of scalar fundamental values to broadcast.
    """
    close  = df["Close"]
    volume = df["Volume"]
    feats  = pd.DataFrame(index=df.index)

    # ── Lagged returns ──────────────────────────────────────────────────────
    for p in [1, 2, 3, 5, 10, 21, 63]:
        feats[f"ret_{p}d"] = close.pct_change(p)

    # ── Moving-average ratios ───────────────────────────────────────────────
    for ma in [5, 10, 20, 50, 100, 200]:
        feats[f"ma{ma}_ratio"] = close / (close.rolling(ma).mean() + 1e-9) - 1

    # ── MA crossovers ───────────────────────────────────────────────────────
    feats["ma5_20_xover"]   = close.rolling(5).mean()  / (close.rolling(20).mean()  + 1e-9) - 1
    feats["ma20_50_xover"]  = close.rolling(20).mean() / (close.rolling(50).mean()  + 1e-9) - 1
    feats["ma50_200_xover"] = close.rolling(50).mean() / (close.rolling(200).mean() + 1e-9) - 1

    # ── RSI ─────────────────────────────────────────────────────────────────
    feats["rsi_14"] = _rsi(close, 14)
    feats["rsi_28"] = _rsi(close, 28)

    # ── MACD ────────────────────────────────────────────────────────────────
    macd, sig, hist = _macd(close)
    feats["macd_norm"]  = macd  / (close + 1e-9)
    feats["macd_sig"]   = sig   / (close + 1e-9)
    feats["macd_hist"]  = hist  / (close + 1e-9)

    # ── Bollinger Bands ─────────────────────────────────────────────────────
    _, _, pct_b, bw = _bollinger(close)
    feats["bb_pct_b"] = pct_b
    feats["bb_bw"]    = bw

    # ── ATR ─────────────────────────────────────────────────────────────────
    feats["atr_ratio"] = _atr(df) / (close + 1e-9)

    # ── Stochastic ──────────────────────────────────────────────────────────
    sk, sd = _stochastic(df)
    feats["stoch_k"]  = sk / 100
    feats["stoch_d"]  = sd / 100
    feats["stoch_kd"] = (sk - sd) / 100

    # ── Williams %R ─────────────────────────────────────────────────────────
    feats["williams_r"] = _williams_r(df) / 100

    # ── CCI ─────────────────────────────────────────────────────────────────
    feats["cci"] = _cci(df) / 200

    # ── Volume ──────────────────────────────────────────────────────────────
    feats["vol_ratio_5"]  = volume / (volume.rolling(5).mean()  + 1)
    feats["vol_ratio_20"] = volume / (volume.rolling(20).mean() + 1)
    feats["vol_trend"]    = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1) - 1

    # ── 52-week position ────────────────────────────────────────────────────
    feats["pos_52w_high"] = close / (close.rolling(252).max() + 1e-9) - 1
    feats["pos_52w_low"]  = close / (close.rolling(252).min() + 1e-9) - 1

    # ── Realised volatility + regime ────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    feats["vol_5d"]  = log_ret.rolling(5).std()  * np.sqrt(252)
    feats["vol_21d"] = log_ret.rolling(21).std() * np.sqrt(252)
    feats["vol_63d"] = log_ret.rolling(63).std() * np.sqrt(252)
    # Volatility regime: is current vol high vs recent average?
    feats["vol_regime"] = feats["vol_21d"] / (feats["vol_63d"] + 1e-9) - 1

    # ── Momentum acceleration (2nd derivative) ─────────────────────────────
    mom_5  = close.pct_change(5)
    mom_21 = close.pct_change(21)
    feats["mom_accel_5"]  = mom_5  - mom_5.shift(5)    # momentum of momentum
    feats["mom_accel_21"] = mom_21 - mom_21.shift(21)

    # ── Rate of Change (ROC) ──────────────────────────────────────────────
    for p in [10, 30, 60]:
        feats[f"roc_{p}d"] = close.pct_change(p)

    # ── Relative Volume (RVOL) ────────────────────────────────────────────
    feats["rvol_10"] = volume / (volume.rolling(10).mean() + 1)
    feats["rvol_50"] = volume / (volume.rolling(50).mean() + 1)

    # ── On-Balance Volume trend ───────────────────────────────────────────
    obv = (np.sign(close.diff()) * volume).cumsum()
    obv_ma = obv.rolling(20).mean()
    feats["obv_trend"] = (obv - obv_ma) / (obv_ma.abs() + 1e-9)

    # ── Volume-price trend ────────────────────────────────────────────────
    vpt = (close.pct_change() * volume).cumsum()
    vpt_ma = vpt.rolling(20).mean()
    feats["vpt_trend"] = (vpt - vpt_ma) / (vpt_ma.abs() + 1e-9)

    # ── Ichimoku Cloud signals ────────────────────────────────────────────
    hi9  = df["High"].rolling(9).max()
    lo9  = df["Low"].rolling(9).min()
    hi26 = df["High"].rolling(26).max()
    lo26 = df["Low"].rolling(26).min()
    hi52 = df["High"].rolling(52).max()
    lo52 = df["Low"].rolling(52).min()
    tenkan  = (hi9 + lo9)   / 2
    kijun   = (hi26 + lo26) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (hi52 + lo52)   / 2
    feats["ichi_tk_ratio"]    = (tenkan - kijun) / (close + 1e-9)
    feats["ichi_cloud_width"] = (senkou_a - senkou_b) / (close + 1e-9)
    feats["ichi_price_vs_cloud"] = (close - (senkou_a + senkou_b)/2) / (close + 1e-9)

    # ── Pivot Point distance ──────────────────────────────────────────────
    pivot = (df["High"] + df["Low"] + close) / 3
    feats["pivot_dist"] = (close - pivot) / (close + 1e-9)

    # ── Candle body and wick features ─────────────────────────────────────
    body   = (close - df["Open"]).abs()
    hl     = df["High"] - df["Low"]
    feats["candle_body_ratio"]  = body / (hl + 1e-9)
    feats["upper_wick_ratio"]   = (df["High"] - pd.concat([close, df["Open"]], axis=1).max(axis=1)) / (hl + 1e-9)
    feats["lower_wick_ratio"]   = (pd.concat([close, df["Open"]], axis=1).min(axis=1) - df["Low"]) / (hl + 1e-9)

    # ── Price range (High-Low) normalised ─────────────────────────────────
    feats["hl_range_norm"] = hl / (close + 1e-9)
    feats["hl_range_5d_avg"] = (hl / (close + 1e-9)).rolling(5).mean()

    # ── Gap features ──────────────────────────────────────────────────────
    feats["gap_up"]   = (df["Open"] / close.shift(1) - 1).clip(lower=0)
    feats["gap_down"] = (df["Open"] / close.shift(1) - 1).clip(upper=0).abs()

    # ── Hurst exponent proxy (mean-reversion vs trending) ────────────────
    # Simplified via variance ratio test: VR(q) = Var(q-day ret) / (q * Var(1-day ret))
    # VR > 1 = trending, VR < 1 = mean-reverting
    daily_ret = close.pct_change()
    for q in [10, 21]:
        var_1d = daily_ret.rolling(63).var()
        var_qd = close.pct_change(q).rolling(63).var()
        feats[f"var_ratio_{q}d"] = var_qd / (q * var_1d + 1e-9)

    # ── Autocorrelation features ──────────────────────────────────────────
    for lag in [1, 5, 10]:
        feats[f"autocorr_{lag}d"] = daily_ret.rolling(63).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else 0, raw=False
        )

    # ── Distance from N-day high/low (mean reversion signal) ─────────────
    for p in [20, 50]:
        feats[f"dist_high_{p}d"] = close / (df["High"].rolling(p).max() + 1e-9) - 1
        feats[f"dist_low_{p}d"]  = close / (df["Low"].rolling(p).min()  + 1e-9) - 1

    # ── Calendar ────────────────────────────────────────────────────────────
    feats["day_of_week"]  = df.index.dayofweek / 4.0
    feats["month"]        = df.index.month     / 12.0
    feats["quarter"]      = df.index.quarter   / 4.0
    feats["month_sin"]    = np.sin(2 * np.pi * df.index.month / 12)
    feats["month_cos"]    = np.cos(2 * np.pi * df.index.month / 12)
    feats["dow_sin"]      = np.sin(2 * np.pi * df.index.dayofweek / 5)
    feats["dow_cos"]      = np.cos(2 * np.pi * df.index.dayofweek / 5)

    # ── Earnings proximity proxy ─────────────────────────────────────────────
    day_num = pd.Series(range(len(df)), index=df.index)
    feats["earnings_proxy"]     = np.sin(2 * np.pi * day_num / 63)
    feats["earnings_proxy_cos"] = np.cos(2 * np.pi * day_num / 63)

    # ── SPY relative features ────────────────────────────────────────────────
    if spy_close is not None and len(spy_close) > 0:
        spy = spy_close.reindex(df.index).ffill()
        spy_ret = spy.pct_change()
        stk_ret = close.pct_change()

        # Alpha (excess return vs SPY) over various windows
        for p in [5, 21, 63]:
            feats[f"alpha_{p}d"] = (
                close.pct_change(p) - spy.pct_change(p)
            )

        # Rolling beta
        feats["beta_63d"] = _rolling_beta(stk_ret, spy_ret, 63)

        # Relative strength ratio
        feats["rs_spy_20"] = (
            (close / close.shift(20)) / (spy / spy.shift(20) + 1e-9) - 1
        )

        # SPY trend context
        for ma in [20, 50, 200]:
            feats[f"spy_ma{ma}_ratio"] = spy / (spy.rolling(ma).mean() + 1e-9) - 1

    # ── VIX / fear regime ────────────────────────────────────────────────────
    if vix_close is not None and len(vix_close) > 0:
        vix = vix_close.reindex(df.index).ffill()
        feats["vix_level"]    = vix / 20.0 - 1          # normalised (20 = neutral)
        feats["vix_ma20_dev"] = vix / (vix.rolling(20).mean() + 1e-9) - 1
        feats["vix_ret_5d"]   = vix.pct_change(5)
        # Fear regime flag: high fear = potential contrarian buy
        feats["high_fear"]    = (vix > 30).astype(float)
        feats["low_fear"]     = (vix < 15).astype(float)

    # ── Sector relative strength ─────────────────────────────────────────────
    if sector_close is not None and len(sector_close) > 0:
        sec = sector_close.reindex(df.index).ffill()
        feats["sector_rs_21d"] = (
            (close / close.shift(21)) / (sec / sec.shift(21) + 1e-9) - 1
        )
        feats["sector_rs_63d"] = (
            (close / close.shift(63)) / (sec / sec.shift(63) + 1e-9) - 1
        )

    # ── Fundamental features (scalar, broadcast across all rows) ─────────
    if fundamentals and isinstance(fundamentals, dict):
        cur = float(close.iloc[-1])

        # Valuation ratios
        for key in ["pe_ratio", "forward_pe", "peg_ratio", "pb_ratio",
                    "ps_ratio", "ev_ebitda"]:
            val = fundamentals.get(key, float("nan"))
            if val is not None and not np.isnan(val):
                feats[f"fund_{key}"] = val
            else:
                feats[f"fund_{key}"] = 0.0

        # Profitability
        for key in ["profit_margin", "oper_margin", "roe", "roa", "gross_margin"]:
            val = fundamentals.get(key, float("nan"))
            if val is not None and not np.isnan(val):
                feats[f"fund_{key}"] = val
            else:
                feats[f"fund_{key}"] = 0.0

        # Growth signals
        for key in ["rev_growth", "earn_growth", "earn_qtr_growth"]:
            val = fundamentals.get(key, float("nan"))
            if val is not None and not np.isnan(val):
                feats[f"fund_{key}"] = val
            else:
                feats[f"fund_{key}"] = 0.0

        # Financial health
        for key in ["debt_to_equity", "current_ratio"]:
            val = fundamentals.get(key, float("nan"))
            if val is not None and not np.isnan(val):
                feats[f"fund_{key}"] = val
            else:
                feats[f"fund_{key}"] = 0.0

        # Analyst target vs current price (upside/downside potential)
        tgt_mean = fundamentals.get("target_mean", float("nan"))
        if tgt_mean and not np.isnan(tgt_mean) and cur > 0:
            feats["analyst_upside"] = tgt_mean / cur - 1
        else:
            feats["analyst_upside"] = 0.0

        tgt_high = fundamentals.get("target_high", float("nan"))
        tgt_low  = fundamentals.get("target_low", float("nan"))
        if tgt_high and tgt_low and not np.isnan(tgt_high) and not np.isnan(tgt_low) and cur > 0:
            feats["analyst_range_pct"] = (tgt_high - tgt_low) / cur
        else:
            feats["analyst_range_pct"] = 0.0

        # Analyst recommendation score (1=strong buy, 5=sell → invert so higher=bullish)
        rec_score = fundamentals.get("recommend_score", float("nan"))
        if rec_score and not np.isnan(rec_score):
            feats["analyst_sentiment"] = (5.0 - rec_score) / 4.0  # normalize to 0-1
        else:
            feats["analyst_sentiment"] = 0.5

        # Short interest
        short_pct = fundamentals.get("short_pct", float("nan"))
        if short_pct and not np.isnan(short_pct):
            feats["short_interest"] = short_pct
        else:
            feats["short_interest"] = 0.0

    return feats




# ─── Market context fetcher ───────────────────────────────────────────────────

def fetch_market_context(period: str = "7y", sector: str = None) -> dict:
    """
    Download SPY and VIX data for enriched feature engineering.
    Returns dict with 'spy' and 'vix' Series (may be empty on failure).
    """
    spy = _fetch_series("SPY", period)
    vix = _fetch_series("^VIX", period)

    sector_series = None
    if sector and sector in SECTOR_ETFS:
        sector_series = _fetch_series(SECTOR_ETFS[sector], period)

    return {"spy": spy, "vix": vix, "sector": sector_series}


# ─── Target Variables ─────────────────────────────────────────────────────────

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    close   = df["Close"]
    targets = pd.DataFrame(index=df.index)
    for name, days in HORIZONS.items():
        raw_ret = close.shift(-days) / close - 1
        targets[f"target_{name}"] = raw_ret

        # Risk-adjusted target: return / realised volatility over horizon
        # This helps the model focus on Sharpe-like quality, not just magnitude
        rv = close.pct_change().rolling(days).std() * np.sqrt(252)
        targets[f"target_radj_{name}"] = raw_ret / (rv + 1e-9)

        # Direction classification target: 1 if up > threshold, 0 otherwise
        # Use a small dead-zone threshold to reduce noise near zero
        threshold = 0.005 if days <= 10 else 0.01
        targets[f"target_dir_{name}"] = (raw_ret > threshold).astype(int)

    return targets


# ─── Support / Resistance ─────────────────────────────────────────────────────

def find_support_resistance(df: pd.DataFrame, lookback: int = 90) -> dict:
    recent  = df.tail(lookback)
    close   = recent["Close"]
    high    = recent["High"]
    low     = recent["Low"]
    current = float(close.iloc[-1])

    pivot = float((high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3)
    r1    = 2 * pivot - float(low.iloc[-1])
    r2    = pivot + float(high.iloc[-1] - low.iloc[-1])
    s1    = 2 * pivot - float(high.iloc[-1])
    s2    = pivot - float(high.iloc[-1] - low.iloc[-1])

    lows_20   = float(low.rolling(20).min().iloc[-1])
    highs_20  = float(high.rolling(20).max().iloc[-1])
    lows_50   = float(low.rolling(min(50, lookback)).min().iloc[-1])
    highs_50  = float(high.rolling(min(50, lookback)).max().iloc[-1])

    resistances = sorted(set([r for r in [r1, r2, highs_20, highs_50] if r > current]))
    supports    = sorted(set([s for s in [s1, s2, lows_20, lows_50]   if s < current]), reverse=True)

    return {
        "current":     current,
        "pivot":       round(pivot, 2),
        "resistances": [round(r, 2) for r in resistances[:4]],
        "supports":    [round(s, 2) for s in supports[:4]],
    }
