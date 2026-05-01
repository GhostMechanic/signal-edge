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

import time
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional
from datetime import datetime


# ─── Browser-impersonating session (anti-rate-limit) ──────────────────────────
# Yahoo Finance has been actively blocking and rate-limiting requests
# from cloud-datacenter IP ranges (Fly.io, Render, AWS, GCP) since late
# 2024. The plain `requests` library yfinance uses by default produces
# a TLS handshake that's trivially fingerprintable as "Python script,"
# and Yahoo just returns empty payloads.
#
# curl_cffi is a `requests`-compatible session that uses libcurl with
# Chrome's actual TLS fingerprint, so the request looks like a browser
# from the wire. Combined with cookie-jar reuse across calls, this
# dramatically reduces the rate of empty fetches.
#
# yfinance natively accepts a `session=` kwarg on Ticker / download.
# We build the session lazily so unit tests that don't need it (and
# environments where curl_cffi isn't installed) still work — the
# fallback is a regular requests.Session.
_YF_SESSION: Optional[object] = None
_YF_SESSION_FAILED: bool = False


def _yf_session():
    """Lazy-build a Chrome-impersonating session for yfinance. Falls
    back to None (default yfinance behaviour) if curl_cffi isn't
    available, so this never breaks tests or environments where the
    extra dependency isn't installed yet."""
    global _YF_SESSION, _YF_SESSION_FAILED
    if _YF_SESSION is not None or _YF_SESSION_FAILED:
        return _YF_SESSION
    try:
        # curl_cffi exposes a `Session` that's a drop-in replacement for
        # requests.Session but uses libcurl with browser TLS impersonation.
        from curl_cffi import requests as curl_requests  # type: ignore
        _YF_SESSION = curl_requests.Session(impersonate="chrome")
        return _YF_SESSION
    except Exception:
        # Mark as failed so we don't retry the import on every fetch
        # call — that would be a measurable perf hit on busy endpoints.
        _YF_SESSION_FAILED = True
        return None


# ─── Typed fetch errors ───────────────────────────────────────────────────────
# These let callers (api/main.py) tell apart the two reasons a price
# fetch can come back empty:
#   • TickerNotFoundError       — Yahoo doesn't recognise the symbol at
#                                 all. Show the user "check the spelling."
#   • TickerDataUnavailableError — Yahoo recognises the ticker but its
#                                 chart endpoint returned nothing right
#                                 now (rate-limit, transient API hiccup,
#                                 brief outage). The right UX is "try
#                                 again in a moment," not "your ticker
#                                 is wrong."
# Both subclass ValueError so existing `except Exception` paths
# elsewhere in the codebase keep working unchanged.

class TickerNotFoundError(ValueError):
    """Yahoo doesn't know this symbol (or it's delisted with no live
    price feed). Carries an optional ``suggestions`` list — tickers
    Yahoo's search endpoint thinks the user might have meant. The API
    layer surfaces these so the UI can render 'did you mean CROX?'
    chips instead of the same static suggestion list every time."""

    def __init__(self, message: str, suggestions: Optional[list] = None):
        super().__init__(message)
        self.suggestions: list = list(suggestions or [])


class TickerDataUnavailableError(ValueError):
    """Yahoo knows the symbol but can't serve price data right now."""
    pass

# ─── Horizons ─────────────────────────────────────────────────────────────────
HORIZONS = {
    "3 Day":      3,   # fastest-scoring horizon — ground truth in 3 trading days
    "1 Week":     5,
    "1 Month":   21,
    "1 Quarter":  63,
    "1 Year":   252,
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

def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Common post-processing on the raw yfinance frame: tz-strip the
    index, keep just the OHLCV columns, drop NaN rows. Centralised so
    the multiple fetch paths (Ticker.history vs yf.download vs Stooq)
    produce identical-shaped output."""
    # yf.download with multiple symbols can return a MultiIndex on
    # columns; with a single symbol that should not happen, but newer
    # yfinance versions sometimes still wrap. Flatten defensively.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Some sources (Stooq) emit naive indexes already; the tz_localize
    # step would no-op there, but guard against tz-aware indexes from
    # yfinance which always come UTC-stamped.
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[cols].copy()
    df.dropna(inplace=True)
    return df


def _period_to_trading_days(period: str) -> Optional[int]:
    """Rough mapping from a yfinance-style period string ('2y', '6mo',
    '60d', 'max') to an approximate trading-day count. Used to trim
    Stooq's "all history" response back down to roughly what the caller
    asked for, so feature engineering doesn't run on years of extra
    rows. None means "no trim — keep what you have"."""
    p = (period or "").lower().strip()
    if not p or p == "max":
        return None
    try:
        if p.endswith("y"):
            return int(float(p[:-1]) * 252)
        if p.endswith("mo"):
            return int(float(p[:-2]) * 21)
        if p.endswith("d"):
            return int(float(p[:-1]))
    except ValueError:
        return None
    return None


def _fetch_via_stooq(symbol: str, period: str) -> Optional[pd.DataFrame]:
    """Free OHLCV fallback when Yahoo's chart endpoint is throttling us.

    Stooq publishes daily CSV at ``https://stooq.com/q/d/l/?s=<sym>&i=d``
    with no auth, no key, no advertised rate-limit. Critically: it's
    reliable from cloud-hosted IPs where Yahoo's chart endpoint has
    been silently returning empty payloads since late 2024. US tickers
    take a ``.us`` suffix.

    Caveat: Stooq's prices are NOT split-adjusted. For US equities with
    no recent splits this matches yfinance's ``auto_adjust=True`` output
    closely. For split-affected tickers there will be drift. The
    tradeoff is worth it: a slightly-less-perfect prediction beats no
    prediction at all.

    Returns None on any failure path so the caller can continue to its
    classification step. Callers that want to know whether Stooq served
    the data should look at ``df.attrs.get('data_source')``.
    """
    try:
        import requests
        from io import StringIO
    except Exception:
        return None

    sym_lower = symbol.lower()
    # Stooq quirk: most US equities resolve as ``<sym>.us``. Try the
    # bare symbol as a backup for tickers that don't follow that
    # convention (rare, but cheap to attempt).
    candidates = [f"{sym_lower}.us", sym_lower]

    for stooq_sym in candidates:
        try:
            url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
            r = requests.get(
                url,
                timeout=10,
                headers={
                    # Plain Mozilla UA is enough — Stooq doesn't
                    # fingerprint the way Yahoo does.
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            )
            if r.status_code != 200:
                continue
            text = r.text.strip()
            # Two failure modes: an HTML error page (starts with `<`)
            # or a literal "No data" body. Both are non-CSV.
            if not text or text.startswith("<") or "No data" in text:
                continue

            df = pd.read_csv(StringIO(text))
            if df.empty or "Date" not in df.columns:
                continue

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            df.index.name = None

            # Stooq returns ALL history by default. Trim to roughly
            # the period the caller asked for plus 50% headroom (so
            # rolling features that look 200 days back still have
            # warmup). Memory matters: scoring_worker calls this
            # constantly.
            n_days = _period_to_trading_days(period)
            if n_days is not None and len(df) > int(n_days * 1.5):
                df = df.tail(int(n_days * 1.5))

            # Tag the source so downstream callers / logs can tell
            # which path served the data. df.attrs survives most
            # pandas operations.
            df.attrs["data_source"] = "stooq"
            return df
        except Exception:
            continue
    return None


def _ticker_exists(symbol: str) -> bool:
    """Best-effort check: does Yahoo recognise this symbol AS A LIVE,
    actively-trading security?

    We're strict about "live" here. Yahoo continues to serve stale
    metadata (shortName, longName, quoteType, even the symbol itself)
    for delisted tickers long after their price feed dies. If we treat
    metadata as proof of life, a delisted symbol like CROC (the
    ProShares UltraShort Yen ETF that ProShares wound down) gets
    routed to "transient feed issue, retry" — i.e. we tell the user
    to keep hammering the same dead symbol. That's the worst kind of
    UX failure: we look broken when really their input is wrong.

    So: return True only when Yahoo serves a CURRENT price field
    (last_price / previous_close / regularMarketPrice). Pure metadata
    is treated as "not alive," which routes the failure to
    TickerNotFoundError → "we couldn't find that ticker, check the
    spelling" (and now: 'did you mean CROX?').
    """
    try:
        tk = yf.Ticker(symbol, session=_yf_session())
        # fast_info: cheap, structured, contains ONLY price-adjacent
        # fields, so any non-None value here is a strong "alive" signal.
        try:
            fi = tk.fast_info
            for k in ("last_price", "previous_close",
                      "regular_market_price", "regularMarketPrice"):
                v = fi.get(k) if isinstance(fi, dict) else getattr(fi, k, None)
                if v is not None and v == v:  # not NaN
                    return True
        except Exception:
            pass
        # Full info — heavier. We accept it as evidence ONLY if it
        # carries a non-None ``regularMarketPrice``. Don't trust
        # ``shortName`` / ``longName`` / ``quoteType`` / ``symbol`` —
        # those persist for delisted tickers indefinitely.
        try:
            info = tk.info or {}
            price = info.get("regularMarketPrice")
            if price is not None and price == price:
                return True
        except Exception:
            pass
    except Exception:
        pass
    return False


def _suggest_similar_tickers(symbol: str, max_results: int = 5) -> list:
    """Best-effort 'did you mean...' for the ticker_not_found UX.

    yfinance's Search endpoint (0.2.50+) hits Yahoo's autocomplete API
    and returns lookalike securities. We use it to power the small
    suggestion chip row that replaces the static fallback when the
    user typed a delisted or misspelled ticker (CROC → CROX, AAPLE →
    AAPL, MICROSOFT → MSFT).

    Strict filters: only EQUITY/ETF results (no FX, futures, crypto),
    skip the literal input, dedupe, cap at ``max_results``. Never
    raises — returns ``[]`` on any failure so callers can fall back
    to static suggestions cleanly.
    """
    try:
        # Search class is in yfinance 0.2.50+. Pass news_count=0 to
        # skip the news fetch we don't need.
        search = yf.Search(
            symbol, max_results=max_results, news_count=0,
            session=_yf_session(),
        )
        quotes = getattr(search, "quotes", None) or []
    except Exception:
        return []

    out: list = []
    seen: set = {symbol.upper().strip()}
    for q in quotes:
        if not isinstance(q, dict):
            continue
        sym = q.get("symbol")
        if not sym or not isinstance(sym, str):
            continue
        sym = sym.upper().strip()
        if sym in seen:
            continue
        # Skip non-equity/ETF results — futures, FX, crypto, indexes
        # rarely make for useful 'did you mean' suggestions when the
        # user typed a stock-like ticker.
        qtype = (q.get("quoteType") or "").upper()
        if qtype and qtype not in ("EQUITY", "ETF"):
            continue
        out.append(sym)
        seen.add(sym)
        if len(out) >= max_results:
            break
    return out


def fetch_stock_data(symbol: str, period: str = "7y") -> pd.DataFrame:
    """Download OHLCV history for ``symbol``.

    Yahoo's chart endpoint occasionally returns an empty dataframe for
    perfectly valid tickers — usually rate-limiting, sometimes a brief
    API hiccup. The original implementation surfaced that as
    ``"No data returned for X. Check the ticker symbol."`` which is
    misleading (the ticker is fine — Yahoo is just being flaky) and
    poisons the UX: the user looks at the form, second-guesses their
    spelling, and walks away.

    The new flow:
      1. Try ``Ticker.history`` (the original path).
      2. If it returned empty, retry once after a short backoff —
         absorbs most transient hiccups.
      3. If still empty, fall back to ``yf.download`` which uses a
         slightly different code path internally and can succeed when
         ``Ticker.history`` has hit a rate-limit.
      4. If yfinance is fully blocked (Yahoo throttling our cloud IP
         range), fall back to Stooq — a different vendor entirely.
         Returned frame carries ``df.attrs['data_source'] = 'stooq'``.
      5. If every path still failed, classify by polling info /
         fast_info — if Yahoo recognises the symbol we raise
         ``TickerDataUnavailableError`` (transient, user should retry);
         otherwise ``TickerNotFoundError`` (real spelling problem).
    """
    # ── Path 1+2: Ticker.history with one short retry ──────────────────
    sess = _yf_session()
    last_err: Optional[BaseException] = None
    for attempt in range(2):
        try:
            df = yf.Ticker(symbol, session=sess).history(
                period=period, auto_adjust=True,
            )
        except Exception as e:
            last_err = e
            df = None
        if df is not None and not df.empty:
            return _normalise_ohlcv(df)
        if attempt == 0:
            # Short backoff before retrying. Long enough that Yahoo's
            # rate-limiter has a chance to forget about us, short enough
            # that the user still sees the prediction within a few seconds.
            time.sleep(0.4)

    # ── Path 3: yf.download fallback ───────────────────────────────────
    # yf.download uses a different request path under the hood. When
    # Ticker.history is being throttled, yf.download often still works.
    try:
        df = yf.download(
            symbol,
            period=period,
            progress=False,
            auto_adjust=True,
            threads=False,
            session=sess,
        )
        if df is not None and not df.empty:
            return _normalise_ohlcv(df)
    except Exception as e:
        last_err = e

    # ── Path 4: Stooq fallback (different vendor entirely) ─────────────
    # When all of Yahoo is throttling us — even via a Chrome-impersonated
    # session — the next best move is a different data provider. Stooq
    # is free, no auth, and reliable from cloud datacenter IPs. The
    # tradeoff is unadjusted prices, which matters for tickers with
    # recent splits but is fine for most of our universe.
    try:
        df = _fetch_via_stooq(symbol, period)
        if df is not None and not df.empty:
            print(f"[data_fetcher] {symbol}: served via Stooq fallback "
                  f"(yfinance returned empty {3} times)")
            normalised = _normalise_ohlcv(df)
            normalised.attrs["data_source"] = "stooq"
            return normalised
    except Exception as e:
        last_err = e

    # ── Classification ─────────────────────────────────────────────────
    # All four fetch paths returned empty. Decide whether to tell the
    # user "your ticker is wrong" or "Yahoo is being flaky, retry."
    # Strict liveness check (live price required, not just metadata)
    # means delisted symbols correctly route to the not-found branch.
    if _ticker_exists(symbol):
        raise TickerDataUnavailableError(
            f"Yahoo Finance returned no price data for '{symbol}' right now. "
            f"This is usually a transient feed issue — try again in a moment."
        )
    # Symbol is unknown / delisted. Best-effort fuzzy suggestions
    # ('did you mean CROX?'). Empty list is fine — the UI falls back
    # to its static suggestion chips.
    suggestions = _suggest_similar_tickers(symbol)
    raise TickerNotFoundError(
        f"Yahoo Finance doesn't recognise the symbol '{symbol}'. "
        f"Double-check the spelling.",
        suggestions=suggestions,
    )


def fetch_stock_info(symbol: str) -> dict:
    try:
        info = yf.Ticker(symbol, session=_yf_session()).info
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
        tk = yf.Ticker(symbol, session=_yf_session())
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


def fetch_earnings_data(symbol: str) -> dict:
    """
    Fetch earnings calendar and historical beat/miss rates.
    Returns dict with:
      - days_to_next_earnings: days until next scheduled earnings
      - last_beat: 1 if last earnings beat, -1 if missed, 0 if unknown
      - beat_rate_ytd: % of earnings beats in YTD (0-1)
      - surprise_pct: most recent EPS surprise %
    """
    default = {
        "days_to_next_earnings": float('nan'),
        "last_beat": float('nan'),
        "beat_rate_ytd": float('nan'),
        "surprise_pct": float('nan'),
        "past_dates_surprise": [],
    }
    try:
        tk = yf.Ticker(symbol, session=_yf_session())

        # ── Days to next earnings ────────────────────────────────────────
        # Primary: tk.calendar (cleanest, but yfinance sometimes returns
        # empty). Fallback: tk.get_earnings_dates which includes future
        # scheduled dates too — filter to ones in the next 180 days.
        days_to_earnings = float('nan')
        try:
            calendar = tk.calendar
            if calendar is not None and 'Earnings Date' in calendar.index:
                earnings_date = calendar.loc['Earnings Date', 0]
                if pd.notna(earnings_date):
                    if hasattr(earnings_date, 'date'):
                        ed = earnings_date.date() if callable(getattr(earnings_date, 'date', None)) else earnings_date
                    else:
                        ed = earnings_date
                    days_to_earnings = (pd.Timestamp(ed) - pd.Timestamp(datetime.now().date())).days
        except Exception:
            pass

        # Fallback 1: look in get_earnings_dates() for the nearest future date
        if not (isinstance(days_to_earnings, (int, float)) and days_to_earnings == days_to_earnings):
            try:
                edf_all = tk.get_earnings_dates(limit=8)
                if edf_all is not None and len(edf_all) > 0:
                    now_ts = pd.Timestamp.now()
                    if edf_all.index.tz is not None:
                        now_ts = now_ts.tz_localize(edf_all.index.tz)
                    future = edf_all[edf_all.index > now_ts].sort_index()
                    if len(future) > 0:
                        next_ed = future.index[0]
                        days_to_earnings = (pd.Timestamp(next_ed).tz_localize(None)
                                            - pd.Timestamp(datetime.now().date())).days
            except Exception:
                pass

        # Fallback 2: info dict sometimes carries an earningsTimestamp
        # (epoch seconds). Used when the other two are empty.
        if not (isinstance(days_to_earnings, (int, float)) and days_to_earnings == days_to_earnings):
            try:
                info = tk.info or {}
                ts_epoch = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
                if ts_epoch:
                    next_ed = pd.Timestamp(int(ts_epoch), unit="s")
                    delta = (next_ed.normalize() - pd.Timestamp(datetime.now().date())).days
                    if 0 <= delta <= 365:
                        days_to_earnings = delta
            except Exception:
                pass

        # ── Historical earnings beat/miss from get_earnings_dates() ─────
        # In addition to scalar "latest" stats we also keep the full past
        # earnings timeline as a list of (date, surprise) tuples. This lets
        # engineer_features compute per-row features (days_since_last_earnings,
        # post-earnings drift phase) for historical training rows — without
        # this, training rows all carry the SAME scalar "days to next
        # earnings" value, which is useless for learning drift patterns.
        last_beat = float('nan')
        beat_rate = float('nan')
        surprise_pct = float('nan')
        past_dates_surprise: list = []   # [(pd.Timestamp, surprise_pct_as_decimal), ...]
        try:
            edf = tk.get_earnings_dates(limit=12)
            if edf is not None and len(edf) > 0:
                # Filter to past dates only
                now_ts = pd.Timestamp.now()
                if edf.index.tz is not None:
                    now_ts = now_ts.tz_localize(edf.index.tz)
                past = edf[edf.index <= now_ts].copy()

                if len(past) > 0 and 'Surprise(%)' in past.columns:
                    surprises = past['Surprise(%)'].dropna()
                    if len(surprises) > 0:
                        surprise_pct = float(surprises.iloc[0]) / 100.0  # normalise
                        last_beat = 1.0 if surprises.iloc[0] > 0 else (-1.0 if surprises.iloc[0] < 0 else 0.0)
                        beats = (surprises > 0).sum()
                        beat_rate = float(beats) / len(surprises)

                    # Build the past-earnings timeline for per-row features.
                    # Oldest-first so engineer_features can iterate forward.
                    timeline = past.sort_index().copy()
                    for idx, row in timeline.iterrows():
                        sp_col = row.get('Surprise(%)') if 'Surprise(%)' in timeline.columns else None
                        sp = float(sp_col) / 100.0 if sp_col is not None and pd.notna(sp_col) else 0.0
                        past_dates_surprise.append((pd.Timestamp(idx).tz_localize(None) if idx.tzinfo else pd.Timestamp(idx), sp))
        except Exception:
            pass

        return {
            "days_to_next_earnings": days_to_earnings,
            "last_beat": last_beat,
            "beat_rate_ytd": beat_rate,
            "surprise_pct": surprise_pct,
            "past_dates_surprise": past_dates_surprise,
        }
    except Exception:
        return default


def fetch_options_data(symbol: str, current_price: float) -> dict:
    """
    Fetch options chain and derive market signals:
      - iv_percentile: IV percentile (0-1)
      - put_call_ratio: puts / calls by open interest
      - call_put_vol_ratio: calls / puts by volume
      - max_pain: level where max options expire worthless
      - atm_iv: at-the-money implied volatility
      - skew: put IV / call IV (tail risk indicator)
    """
    try:
        tk = yf.Ticker(symbol, session=_yf_session())
        options_chain = tk.options

        if not options_chain:
            return {
                "iv_percentile": float('nan'),
                "put_call_ratio": float('nan'),
                "call_put_vol_ratio": float('nan'),
                "max_pain": float('nan'),
                "atm_iv": float('nan'),
                "skew": float('nan'),
            }

        # Get nearest-term expiration (usually most liquid)
        nearest_exp = options_chain[0]
        opt = tk.option_chain(nearest_exp)
        calls = opt.calls
        puts = opt.puts

        # Calculate put/call ratio by open interest
        if not (calls.empty or puts.empty):
            total_call_oi = calls['openInterest'].sum()
            total_put_oi = puts['openInterest'].sum()
            put_call_ratio = total_put_oi / (total_call_oi + 1e-9)

            # Call/put volume ratio
            total_call_vol = calls['volume'].sum()
            total_put_vol = puts['volume'].sum()
            call_put_vol_ratio = total_call_vol / (total_put_vol + 1e-9)

            # ATM implied volatility (close to current price)
            atm_calls = calls[np.abs(calls['strike'] - current_price) < current_price * 0.05]
            atm_iv = float(atm_calls['impliedVolatility'].mean()) if not atm_calls.empty else float('nan')

            # IV skew: put IV / call IV for same strike (tail risk)
            merged = calls[['strike', 'impliedVolatility']].merge(
                puts[['strike', 'impliedVolatility']], on='strike', suffixes=('_call', '_put')
            )
            if not merged.empty:
                skew = (merged['impliedVolatility_put'] / (merged['impliedVolatility_call'] + 1e-9)).mean()
            else:
                skew = float('nan')

            # Max pain (simplified: strike with highest OI)
            all_oi = pd.concat([
                calls[['strike', 'openInterest']].assign(type='call'),
                puts[['strike', 'openInterest']].assign(type='put')
            ])
            if not all_oi.empty:
                max_pain = float(all_oi.loc[all_oi['openInterest'].idxmax(), 'strike'])
            else:
                max_pain = float('nan')

        else:
            put_call_ratio = float('nan')
            call_put_vol_ratio = float('nan')
            atm_iv = float('nan')
            skew = float('nan')
            max_pain = float('nan')

        # IV percentile (simplified: use ATM IV as proxy, would need historical IV for true percentile)
        iv_percentile = float('nan')  # Requires IV history

        return {
            "iv_percentile": iv_percentile,
            "put_call_ratio": put_call_ratio,
            "call_put_vol_ratio": call_put_vol_ratio,
            "max_pain": max_pain,
            "atm_iv": atm_iv,
            "skew": skew,
        }
    except Exception:
        return {
            "iv_percentile": float('nan'),
            "put_call_ratio": float('nan'),
            "call_put_vol_ratio": float('nan'),
            "max_pain": float('nan'),
            "atm_iv": float('nan'),
            "skew": float('nan'),
        }


def _fetch_series(symbol: str, period: str = "7y") -> pd.Series:
    """Fetch a single close-price series; return empty Series on failure."""
    try:
        t  = yf.Ticker(symbol, session=_yf_session())
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
    earnings_data: Optional[dict] = None,
    options_data: Optional[dict] = None,
    sentiment_ctx: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build feature matrix. spy_close / vix_close / sector_close are optional
    enrichment series aligned to df's index.
    fundamentals: optional dict of scalar fundamental values to broadcast.
    earnings_data: optional dict with earnings metrics (days_to_earnings, etc.)
    options_data: optional dict with options-derived metrics (IV, put/call ratio, etc.)
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

    # ── Fundamental features ──────────────────────────────────────────────
    # NOTE: Fundamentals are point-in-time snapshots. Broadcasting to ALL
    # rows would leak current info into historical training data. We only
    # apply them to the last 252 rows (1 year) to prevent leakage while
    # still giving the model a window to learn from them.
    if fundamentals and isinstance(fundamentals, dict):
        cur = float(close.iloc[-1])
        _fund_mask = pd.Series(False, index=df.index)
        _fund_mask.iloc[-min(252, len(df)):] = True

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

        # Mask: zero out fundamental features for rows older than 1 year
        # to prevent point-in-time leakage into historical training data
        fund_cols = [c for c in feats.columns if c.startswith("fund_")
                     or c in ("analyst_upside", "analyst_range_pct",
                              "analyst_sentiment", "short_interest")]
        for c in fund_cols:
            feats.loc[~_fund_mask, c] = 0.0

    # ── Earnings features ────────────────────────────────────────────────────
    # Same leakage concern: only apply to recent rows
    _earn_mask = pd.Series(False, index=df.index)
    _earn_mask.iloc[-min(252, len(df)):] = True
    if earnings_data and isinstance(earnings_data, dict):
        # Days to next earnings (divide by 252 to normalize)
        dte = earnings_data.get("days_to_next_earnings", float("nan"))
        if dte and not np.isnan(dte):
            feats["days_to_earnings"] = dte
            feats["earnings_proximity"] = 1.0 / (1.0 + abs(dte) / 30.0)  # closer = higher value
        else:
            feats["days_to_earnings"] = 0.0
            feats["earnings_proximity"] = 0.5  # neutral if unknown

        # Beat/miss history
        beat_rate = earnings_data.get("beat_rate_ytd", float("nan"))
        if beat_rate and not np.isnan(beat_rate):
            feats["earnings_beat_rate"] = beat_rate
        else:
            feats["earnings_beat_rate"] = 0.5  # neutral default

        last_beat = earnings_data.get("last_beat", float("nan"))
        if last_beat and not np.isnan(last_beat):
            feats["last_earnings_beat"] = last_beat  # 1 = beat, -1 = miss, 0 = neutral
        else:
            feats["last_earnings_beat"] = 0.0

        # Surprise magnitude
        surprise = earnings_data.get("surprise_pct", float("nan"))
        if surprise and not np.isnan(surprise):
            feats["earnings_surprise_pct"] = surprise
        else:
            feats["earnings_surprise_pct"] = 0.0

        # Post-earnings drift features REMOVED 2026-04-23 after benchmark
        # regression (-1.25pp vs after-events). Hypothesis: modern markets
        # have arbitraged the classic PEAD pattern away, or 12-quarter
        # history is too sparse for XGBoost to find a stable signal without
        # overfitting. Leaving this as a comment marker so we know it was
        # tried and why it's absent. To re-test: restore the block from
        # commit history.

        # Mask scalar earnings features for older rows
        earn_cols = [c for c in feats.columns if c.startswith("earnings_")
                     or c in ("days_to_earnings", "last_earnings_beat")]
        for c in earn_cols:
            feats.loc[~_earn_mask, c] = 0.0

    # ── Options market features ───────────────────────────────────────────────
    # Also masked to prevent leakage
    if options_data and isinstance(options_data, dict):
        # Implied volatility
        iv = options_data.get("atm_iv", float("nan"))
        if iv and not np.isnan(iv):
            feats["options_atm_iv"] = iv
        else:
            feats["options_atm_iv"] = 0.0

        # Put/call ratio (elevated = bearish, depressed = bullish)
        pc_ratio = options_data.get("put_call_ratio", float("nan"))
        if pc_ratio and not np.isnan(pc_ratio):
            feats["options_put_call_ratio"] = pc_ratio
        else:
            feats["options_put_call_ratio"] = 1.0  # neutral

        # Call/put volume ratio (inverse: high vol ratio = bullish)
        cpvol_ratio = options_data.get("call_put_vol_ratio", float("nan"))
        if cpvol_ratio and not np.isnan(cpvol_ratio):
            feats["options_call_put_vol"] = cpvol_ratio
        else:
            feats["options_call_put_vol"] = 1.0  # neutral

        # Max pain (distance from current price indicates consensus target)
        cur_price = float(close.iloc[-1]) if len(close) > 0 else 1.0
        max_pain = options_data.get("max_pain", float("nan"))
        if max_pain and not np.isnan(max_pain) and cur_price > 0:
            feats["options_max_pain_diff"] = (max_pain - cur_price) / cur_price
        else:
            feats["options_max_pain_diff"] = 0.0

        # IV skew (elevated put IV relative to call = tail risk premium)
        skew = options_data.get("skew", float("nan"))
        if skew and not np.isnan(skew):
            feats["options_iv_skew"] = skew
        else:
            feats["options_iv_skew"] = 1.0  # neutral

        # Mask options features for older rows
        opt_cols = [c for c in feats.columns if c.startswith("options_")]
        _opt_mask = pd.Series(False, index=df.index)
        _opt_mask.iloc[-min(252, len(df)):] = True
        for c in opt_cols:
            neutral = 1.0 if "ratio" in c or "skew" in c else 0.0
            feats.loc[~_opt_mask, c] = neutral

    # ── Sentiment & Market Breadth features ────────────────────────────────
    if sentiment_ctx and isinstance(sentiment_ctx, dict):
        # VIX momentum (5d % change)
        vix_mom = sentiment_ctx.get("vix_5d_momentum")
        if vix_mom is not None and len(vix_mom) > 0:
            vix_align = vix_mom.reindex(df.index).ffill()
            feats["sent_vix_momentum_5d"] = vix_align
        else:
            feats["sent_vix_momentum_5d"] = 0.0

        # Sector dispersion (rotation signal)
        sec_disp = sentiment_ctx.get("sector_dispersion_21d")
        if sec_disp is not None and len(sec_disp) > 0:
            disp_align = sec_disp.reindex(df.index).ffill()
            feats["sent_sector_dispersion_21d"] = disp_align
        else:
            feats["sent_sector_dispersion_21d"] = 0.0

        # Average sector momentum
        avg_mom = sentiment_ctx.get("avg_sector_momentum")
        if avg_mom is not None and len(avg_mom) > 0:
            mom_align = avg_mom.reindex(df.index).ffill()
            feats["sent_avg_sector_momentum"] = mom_align
        else:
            feats["sent_avg_sector_momentum"] = 0.0

        # Dollar momentum (21d)
        dollar_mom = sentiment_ctx.get("dollar_momentum_21d")
        if dollar_mom is not None and len(dollar_mom) > 0:
            dollar_align = dollar_mom.reindex(df.index).ffill()
            feats["sent_dollar_momentum_21d"] = dollar_align
        else:
            feats["sent_dollar_momentum_21d"] = 0.0

        # 10Y Treasury level (normalized)
        tsy_level = sentiment_ctx.get("treasury_10y_level")
        if tsy_level is not None and len(tsy_level) > 0:
            tsy_align = tsy_level.reindex(df.index).ffill()
            feats["sent_treasury_10y_level"] = tsy_align
        else:
            feats["sent_treasury_10y_level"] = 0.0

        # 10Y Treasury momentum (21d change)
        tsy_mom = sentiment_ctx.get("treasury_10y_momentum_21d")
        if tsy_mom is not None and len(tsy_mom) > 0:
            tsy_mom_align = tsy_mom.reindex(df.index).ffill()
            feats["sent_treasury_10y_momentum_21d"] = tsy_mom_align
        else:
            feats["sent_treasury_10y_momentum_21d"] = 0.0

        # Yield curve spread
        yc_spread = sentiment_ctx.get("yield_curve_spread")
        if yc_spread is not None and len(yc_spread) > 0:
            yc_align = yc_spread.reindex(df.index).ffill()
            feats["sent_yield_curve_spread"] = yc_align
        else:
            feats["sent_yield_curve_spread"] = 0.0

        # Put/Call ratio
        pc_ratio = sentiment_ctx.get("put_call_ratio")
        if pc_ratio is not None and len(pc_ratio) > 0:
            pc_align = pc_ratio.reindex(df.index).ffill()
            feats["sent_put_call_ratio"] = pc_align
        else:
            feats["sent_put_call_ratio"] = 1.0  # neutral

        # Put/Call 20-day MA
        pc_ma20 = sentiment_ctx.get("put_call_ma20")
        if pc_ma20 is not None and len(pc_ma20) > 0:
            pc_ma_align = pc_ma20.reindex(df.index).ffill()
            feats["sent_put_call_ma20"] = pc_ma_align
        else:
            feats["sent_put_call_ma20"] = 1.0  # neutral

        # ── Commodity regime features ──────────────────────────────────
        # Oil level (z-score) — "is oil in an unusual regime right now?"
        oil_z = sentiment_ctx.get("oil_level_z")
        if oil_z is not None and len(oil_z) > 0:
            feats["sent_oil_level_z"] = oil_z.reindex(df.index).ffill()
        else:
            feats["sent_oil_level_z"] = 0.0

        # Oil momentum (21d) — captures spikes from geopolitical events
        oil_mom = sentiment_ctx.get("oil_momentum_21d")
        if oil_mom is not None and len(oil_mom) > 0:
            feats["sent_oil_momentum_21d"] = oil_mom.reindex(df.index).ffill()
        else:
            feats["sent_oil_momentum_21d"] = 0.0

        # Oil volatility (OVX normalized) — supply/demand uncertainty
        oil_vol = sentiment_ctx.get("oil_vol_normalized")
        if oil_vol is not None and len(oil_vol) > 0:
            feats["sent_oil_vol"] = oil_vol.reindex(df.index).ffill()
        else:
            feats["sent_oil_vol"] = 0.0

        # Gold momentum — flight-to-safety / inflation regime
        gold_mom = sentiment_ctx.get("gold_momentum_21d")
        if gold_mom is not None and len(gold_mom) > 0:
            feats["sent_gold_momentum_21d"] = gold_mom.reindex(df.index).ffill()
        else:
            feats["sent_gold_momentum_21d"] = 0.0

        # Copper momentum — global growth signal
        copper_mom = sentiment_ctx.get("copper_momentum_21d")
        if copper_mom is not None and len(copper_mom) > 0:
            feats["sent_copper_momentum_21d"] = copper_mom.reindex(df.index).ffill()
        else:
            feats["sent_copper_momentum_21d"] = 0.0

        # Credit stress (HYG/LQD ratio deviation from quarter avg)
        credit = sentiment_ctx.get("credit_stress")
        if credit is not None and len(credit) > 0:
            feats["sent_credit_stress"] = credit.reindex(df.index).ffill()
        else:
            feats["sent_credit_stress"] = 0.0

    else:
        # Default neutral values if no sentiment context
        feats["sent_vix_momentum_5d"] = 0.0
        feats["sent_sector_dispersion_21d"] = 0.0
        feats["sent_avg_sector_momentum"] = 0.0
        feats["sent_dollar_momentum_21d"] = 0.0
        feats["sent_treasury_10y_level"] = 0.0
        feats["sent_treasury_10y_momentum_21d"] = 0.0
        feats["sent_yield_curve_spread"] = 0.0
        feats["sent_put_call_ratio"] = 1.0
        feats["sent_put_call_ma20"] = 1.0
        # Commodity defaults (all neutral = 0)
        feats["sent_oil_level_z"] = 0.0
        feats["sent_oil_momentum_21d"] = 0.0
        feats["sent_oil_vol"] = 0.0
        feats["sent_gold_momentum_21d"] = 0.0
        feats["sent_copper_momentum_21d"] = 0.0
        feats["sent_credit_stress"] = 0.0

    # ── News Sentiment & Fear/Greed features (point-in-time, recent only) ──
    # These are "current snapshot" features — only populated for the most
    # recent rows (like fundamentals) to prevent look-ahead bias.
    _news_ctx = sentiment_ctx.get("news_sentiment") if sentiment_ctx else None
    if _news_ctx and isinstance(_news_ctx, dict):
        _news_mask = pd.Series(False, index=df.index)
        _news_mask.iloc[-min(63, len(df)):] = True  # last ~3 months

        feats["sent_news_mean"] = 0.0
        feats["sent_news_std"] = 0.0
        feats["sent_news_positive_ratio"] = 0.5
        feats["sent_news_volume"] = 0.0
        feats["sent_news_trend"] = 0.0
        feats["sent_news_volume_trend"] = 0.0
        feats["sent_fear_greed"] = 0.5
        feats["sent_fear_greed_momentum"] = 0.0

        feats.loc[_news_mask, "sent_news_mean"] = _news_ctx.get("news_sentiment_mean", 0.0)
        feats.loc[_news_mask, "sent_news_std"] = _news_ctx.get("news_sentiment_std", 0.0)
        feats.loc[_news_mask, "sent_news_positive_ratio"] = _news_ctx.get("news_positive_ratio", 0.5)
        feats.loc[_news_mask, "sent_news_volume"] = min(_news_ctx.get("news_article_count", 0) / 50.0, 2.0)
        feats.loc[_news_mask, "sent_news_trend"] = _news_ctx.get("news_sentiment_trend", 0.0)
        feats.loc[_news_mask, "sent_news_volume_trend"] = _news_ctx.get("news_volume_trend", 0.0)
        feats.loc[_news_mask, "sent_fear_greed"] = _news_ctx.get("fear_greed_score", 0.5)
        feats.loc[_news_mask, "sent_fear_greed_momentum"] = _news_ctx.get("fear_greed_momentum", 0.0)
    else:
        feats["sent_news_mean"] = 0.0
        feats["sent_news_std"] = 0.0
        feats["sent_news_positive_ratio"] = 0.5
        feats["sent_news_volume"] = 0.0
        feats["sent_news_trend"] = 0.0
        feats["sent_news_volume_trend"] = 0.0
        feats["sent_fear_greed"] = 0.5
        feats["sent_fear_greed_momentum"] = 0.0

    # ── Macro event-calendar features (FOMC / CPI / NFP) ───────────────────
    # Each row gets "normalized days until next event" (0 = today, 1 = far)
    # plus a proximity flag for "event within 5 days." The model learns how
    # short-horizon returns behave differently in event-weeks vs quiet weeks.
    try:
        from macro_calendar import (
            FOMC_DATES, CPI_DATES, NFP_DATES,
            event_distance_series, event_proximity_flag,
        )
        feats["macro_days_to_fomc"]   = event_distance_series(df.index, FOMC_DATES, 30)
        feats["macro_days_to_cpi"]    = event_distance_series(df.index, CPI_DATES, 30)
        feats["macro_days_to_nfp"]    = event_distance_series(df.index, NFP_DATES, 30)
        feats["macro_fomc_week"]      = event_proximity_flag(df.index, FOMC_DATES, 5)
        feats["macro_cpi_week"]       = event_proximity_flag(df.index, CPI_DATES, 5)
        feats["macro_nfp_week"]       = event_proximity_flag(df.index, NFP_DATES, 5)
    except Exception:
        # Fail-soft defaults if the calendar module is unavailable
        feats["macro_days_to_fomc"]   = 1.0
        feats["macro_days_to_cpi"]    = 1.0
        feats["macro_days_to_nfp"]    = 1.0
        feats["macro_fomc_week"]      = 0.0
        feats["macro_cpi_week"]       = 0.0
        feats["macro_nfp_week"]       = 0.0

    return feats




# ─── Market context fetcher ───────────────────────────────────────────────────

def fetch_sentiment_context(period: str = "7y") -> dict:
    """
    Fetch sentiment & market breadth data:
      - VIX momentum (5d, term structure proxy)
      - Put/Call ratio from ^CPC
      - Sector ETF dispersion (rotation regime)
      - Dollar index (UUP) momentum
      - 10Y Treasury yield & yield curve spread
    Returns dict with Series (may be empty or None if fetch fails).
    """
    sentiment = {}

    try:
        # VIX 5d momentum
        vix_data = _fetch_series("^VIX", period)
        if vix_data is not None and len(vix_data) > 0:
            sentiment["vix_5d_momentum"] = vix_data.pct_change(5)
        else:
            sentiment["vix_5d_momentum"] = None
    except Exception:
        sentiment["vix_5d_momentum"] = None

    # Put/Call ratio (legacy ^CPC has been delisted by Yahoo; the model was
    # already handling the absence gracefully. Skip the fetch entirely to
    # avoid noisy 404 logs on every run. Revisit if we find a replacement.)
    sentiment["put_call_ratio"] = None
    sentiment["put_call_ma20"] = None

    try:
        # Market breadth: sector dispersion
        sector_rets = {}
        for name, ticker in SECTOR_ETFS.items():
            try:
                sec_data = _fetch_series(ticker, period)
                if sec_data is not None and len(sec_data) > 0:
                    sector_rets[name] = sec_data.pct_change(21)
            except Exception:
                pass

        if len(sector_rets) > 3:
            # Compute dispersion: max - min sector return
            sector_df = pd.DataFrame(sector_rets)
            sentiment["sector_dispersion_21d"] = sector_df.max(axis=1) - sector_df.min(axis=1)
            sentiment["avg_sector_momentum"] = sector_df.mean(axis=1)
        else:
            sentiment["sector_dispersion_21d"] = None
            sentiment["avg_sector_momentum"] = None
    except Exception:
        sentiment["sector_dispersion_21d"] = None
        sentiment["avg_sector_momentum"] = None

    try:
        # Dollar index momentum (UUP)
        uup = _fetch_series("UUP", period)
        if uup is not None and len(uup) > 0:
            sentiment["dollar_momentum_21d"] = uup.pct_change(21)
        else:
            sentiment["dollar_momentum_21d"] = None
    except Exception:
        sentiment["dollar_momentum_21d"] = None

    try:
        # 10Y Treasury yield (^TNX)
        tnx = _fetch_series("^TNX", period)
        if tnx is not None and len(tnx) > 0:
            # Normalize: (current - 3%) / 3%
            sentiment["treasury_10y_level"] = (tnx - 3.0) / 3.0
            sentiment["treasury_10y_momentum_21d"] = tnx.diff(21)
        else:
            sentiment["treasury_10y_level"] = None
            sentiment["treasury_10y_momentum_21d"] = None
    except Exception:
        sentiment["treasury_10y_level"] = None
        sentiment["treasury_10y_momentum_21d"] = None

    try:
        # 10Y-2Y yield spread proxy (^TNX - ^IRX)
        tnx = _fetch_series("^TNX", period)
        irx = _fetch_series("^IRX", period)
        if tnx is not None and irx is not None and len(tnx) > 0 and len(irx) > 0:
            tnx_align = tnx.reindex(irx.index, method='ffill')
            sentiment["yield_curve_spread"] = tnx_align - irx
        else:
            sentiment["yield_curve_spread"] = None
    except Exception:
        sentiment["yield_curve_spread"] = None

    # ── Commodity regime features ──────────────────────────────────────
    # Oil (WTI continuous futures) — affects energy, transport, airlines
    try:
        oil = _fetch_series("CL=F", period)
        if oil is not None and len(oil) > 0:
            # Z-score normalized level over 200d window (regime signal)
            oil_ma = oil.rolling(200, min_periods=60).mean()
            oil_sd = oil.rolling(200, min_periods=60).std()
            sentiment["oil_level_z"] = (oil - oil_ma) / (oil_sd + 1e-9)
            # 21-day momentum (% change) — captures spikes like Iran war shocks
            sentiment["oil_momentum_21d"] = oil.pct_change(21)
        else:
            sentiment["oil_level_z"] = None
            sentiment["oil_momentum_21d"] = None
    except Exception:
        sentiment["oil_level_z"] = None
        sentiment["oil_momentum_21d"] = None

    # Oil volatility (OVX — the "VIX for oil"); rising = supply/demand
    # uncertainty, often geopolitical
    try:
        ovx = _fetch_series("^OVX", period)
        if ovx is not None and len(ovx) > 0:
            # Normalized: (level - 30) / 30. OVX ~30 is typical, >50 is stressed
            sentiment["oil_vol_normalized"] = (ovx - 30.0) / 30.0
        else:
            sentiment["oil_vol_normalized"] = None
    except Exception:
        sentiment["oil_vol_normalized"] = None

    # Gold — inflation hedge / flight-to-safety indicator
    try:
        gold = _fetch_series("GLD", period)
        if gold is not None and len(gold) > 0:
            sentiment["gold_momentum_21d"] = gold.pct_change(21)
        else:
            sentiment["gold_momentum_21d"] = None
    except Exception:
        sentiment["gold_momentum_21d"] = None

    # Copper — "Dr. Copper" — global growth / industrial demand signal
    try:
        copper = _fetch_series("CPER", period)
        if copper is not None and len(copper) > 0:
            sentiment["copper_momentum_21d"] = copper.pct_change(21)
        else:
            sentiment["copper_momentum_21d"] = None
    except Exception:
        sentiment["copper_momentum_21d"] = None

    # Credit stress — HY/IG spread proxy via HYG/LQD ratio change
    # When credit stress spikes, junk bonds drop relative to investment grade
    try:
        hyg = _fetch_series("HYG", period)
        lqd = _fetch_series("LQD", period)
        if hyg is not None and lqd is not None and len(hyg) > 0 and len(lqd) > 0:
            lqd_align = lqd.reindex(hyg.index, method='ffill')
            credit_ratio = hyg / (lqd_align + 1e-9)
            # Change vs 63-day (quarter) rolling mean — risk-on/off shift
            credit_ma = credit_ratio.rolling(63, min_periods=30).mean()
            sentiment["credit_stress"] = (credit_ratio - credit_ma) / (credit_ma + 1e-9)
        else:
            sentiment["credit_stress"] = None
    except Exception:
        sentiment["credit_stress"] = None

    return sentiment


def fetch_market_context(period: str = "7y", sector: str = None, include_sentiment: bool = True) -> dict:
    """
    Download SPY, VIX, sector data, and optionally sentiment context.
    Returns dict with 'spy', 'vix', 'sector', and optionally 'sentiment' Series.
    """
    spy = _fetch_series("SPY", period)
    vix = _fetch_series("^VIX", period)

    sector_series = None
    if sector and sector in SECTOR_ETFS:
        sector_series = _fetch_series(SECTOR_ETFS[sector], period)

    result = {"spy": spy, "vix": vix, "sector": sector_series}

    if include_sentiment:
        result["sentiment"] = fetch_sentiment_context(period)

    return result


# ─── Target Variables ─────────────────────────────────────────────────────────

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    close   = df["Close"]
    targets = pd.DataFrame(index=df.index)
    for name, days in HORIZONS.items():
        # Log-returns: ln(P_future / P_now)
        # More symmetric than raw pct returns, better for magnitude prediction.
        # Convert back to pct return at prediction time via: pct = exp(log_ret) - 1
        log_ret = np.log(close.shift(-days) / close)
        targets[f"target_{name}"] = log_ret

        # Risk-adjusted target: log-return / HISTORICAL volatility (no look-ahead)
        # Use lagged rolling vol so we only divide by what was known at prediction time
        hist_rv = close.pct_change().rolling(max(days, 21)).std().shift(1) * np.sqrt(252)
        targets[f"target_radj_{name}"] = log_ret / (hist_rv + 1e-9)

        # Direction classification target: 1 if up > threshold, 0 otherwise
        # Volatility-aware threshold: half the 75th percentile of recent absolute returns
        daily_ret = close.pct_change()
        rolling_vol_pctile = daily_ret.abs().rolling(63).quantile(0.75).shift(1)
        vol_threshold = (rolling_vol_pctile * np.sqrt(days) * 0.5).clip(lower=0.003, upper=0.05)
        raw_ret = close.shift(-days) / close - 1
        targets[f"target_dir_{name}"] = (raw_ret > vol_threshold).astype(int)

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
