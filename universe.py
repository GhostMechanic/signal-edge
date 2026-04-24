"""
universe.py
───────────────────────────────────────────────────────────────────────────────
Fetches and caches S&P 500, NASDAQ-100, and Dow Jones 30 constituent tickers.
Uses Wikipedia as the primary source (no API key required).
Cache refreshes weekly so daily scans don't re-fetch the same lists.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR  = ".predictions"
CACHE_FILE = os.path.join(CACHE_DIR, "universe_cache.json")
CACHE_TTL_DAYS = 7   # refresh weekly

# ── Hardcoded Dow Jones 30 (rarely changes, used as fallback) ─────────────────
DOW_30 = [
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
    "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT",
]

# ── Hardcoded NASDAQ-100 (stable enough; Wikipedia backup) ────────────────────
NASDAQ_100 = [
    "AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD","AMGN",
    "AMZN","ANSS","APP","ARM","ASML","AVGO","AZN","BIIB","BKNG","BKR",
    "CCEP","CDNS","CDW","CEG","CHTR","CMCSA","COST","CPRT","CRWD","CSCO",
    "CSGP","CSX","CTAS","CTSH","DDOG","DLTR","DXCM","EA","EXC","FANG",
    "FAST","FTNT","GEHC","GFS","GILD","GOOG","GOOGL","HON","IDXX","ILMN",
    "INTC","INTU","ISRG","KDP","KHC","KLAC","LRCX","LULU","MAR","MCHP",
    "MDB","MDLZ","META","MNST","MRNA","MRVL","MSFT","MU","NFLX","NVDA",
    "NXPI","ODFL","ON","ORLY","PANW","PAYX","PCAR","PDD","PEP","PYPL",
    "QCOM","REGN","ROP","ROST","SBUX","SMCI","SNPS","SPLK","TEAM","TMUS",
    "TSLA","TTD","TXN","VRSK","VRTX","WBD","WDAY","XEL","ZS","ZM",
]


def _load_cache() -> Optional[dict]:
    """Load cached universe data if still fresh."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
        if datetime.now() - cached_at > timedelta(days=CACHE_TTL_DAYS):
            return None   # expired
        return data
    except Exception:
        return None


def _save_cache(sp500: list, nasdaq: list, dow: list):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump({
            "cached_at": datetime.now().isoformat(),
            "sp500":     sp500,
            "nasdaq100": nasdaq,
            "dow30":     dow,
        }, f)


def _fetch_sp500_wiki() -> list:
    """Scrape S&P 500 tickers from Wikipedia."""
    try:
        import pandas as pd
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            header=0
        )
        df = tables[0]
        # Column is 'Symbol' or 'Ticker symbol'
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = df[col].str.replace(".", "-", regex=False).tolist()
        return [t.strip() for t in tickers if isinstance(t, str) and t.strip()]
    except Exception as e:
        logger.warning(f"Wikipedia S&P 500 fetch failed: {e}")
        return []


def _fetch_nasdaq100_wiki() -> list:
    """Scrape NASDAQ-100 tickers from Wikipedia."""
    try:
        import pandas as pd
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            header=0
        )
        for tbl in tables:
            cols = [str(c).lower() for c in tbl.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                col_idx = next(i for i, c in enumerate(cols) if "ticker" in c or "symbol" in c)
                tickers = tbl.iloc[:, col_idx].tolist()
                tickers = [str(t).strip() for t in tickers if isinstance(t, str) and len(t) < 10]
                if len(tickers) > 50:
                    return tickers
    except Exception as e:
        logger.warning(f"Wikipedia NASDAQ-100 fetch failed: {e}")
    return NASDAQ_100   # fallback


def get_sp500_tickers() -> list:
    """Return S&P 500 ticker list (cached or freshly fetched)."""
    cache = _load_cache()
    if cache and cache.get("sp500"):
        return cache["sp500"]
    tickers = _fetch_sp500_wiki()
    if not tickers:
        # Minimal fallback with well-known large caps
        tickers = _get_sp500_fallback()
    _save_cache(tickers, get_nasdaq100_tickers.__wrapped__() if hasattr(get_nasdaq100_tickers, "__wrapped__") else NASDAQ_100, DOW_30)
    return tickers


def get_nasdaq100_tickers() -> list:
    """Return NASDAQ-100 ticker list (cached or freshly fetched)."""
    cache = _load_cache()
    if cache and cache.get("nasdaq100"):
        return cache["nasdaq100"]
    return _fetch_nasdaq100_wiki()


def get_dow30_tickers() -> list:
    """Return Dow Jones 30 tickers (stable list)."""
    cache = _load_cache()
    if cache and cache.get("dow30"):
        return cache["dow30"]
    return DOW_30


def get_full_universe() -> dict:
    """
    Return combined universe with source labels.
    Fetches all three indices and caches them together.
    Returns: {"sp500": [...], "nasdaq100": [...], "dow30": [...], "all": [...unique...]}
    """
    cache = _load_cache()
    if cache:
        sp500     = cache.get("sp500", [])
        nasdaq100 = cache.get("nasdaq100", NASDAQ_100)
        dow30     = cache.get("dow30", DOW_30)
    else:
        sp500     = _fetch_sp500_wiki() or _get_sp500_fallback()
        nasdaq100 = _fetch_nasdaq100_wiki()
        dow30     = DOW_30
        _save_cache(sp500, nasdaq100, dow30)

    all_tickers = sorted(set(sp500 + nasdaq100 + dow30))
    return {
        "sp500":     sp500,
        "nasdaq100": nasdaq100,
        "dow30":     dow30,
        "all":       all_tickers,
        "counts": {
            "sp500":     len(sp500),
            "nasdaq100": len(nasdaq100),
            "dow30":     len(dow30),
            "total":     len(all_tickers),
        }
    }


def refresh_universe() -> dict:
    """Force-refresh the universe cache (ignores TTL)."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    return get_full_universe()


# ── Fallback S&P 500 list (top ~100 by market cap) ───────────────────────────
def _get_sp500_fallback() -> list:
    """Minimal S&P 500 fallback if Wikipedia is unavailable."""
    return [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","UNH","LLY",
        "JPM","XOM","V","AVGO","PG","MA","COST","HD","ABBV","MRK","CVX","ADBE",
        "CRM","ACN","PEP","NFLX","LIN","AMD","TMO","DHR","ABT","CSCO","WMT","BAC",
        "MCD","NKE","INTC","QCOM","TXN","MS","RTX","UNP","GE","HON","IBM","CAT",
        "NOW","AMAT","GS","BLK","SYK","SPGI","ADP","ISRG","BKNG","GILD","VRTX",
        "PLD","CB","MDLZ","PANW","DE","CI","MO","TJX","ZTS","REGN","BSX","LRCX",
        "SO","MMC","DUK","SHW","AON","FCX","ICE","CME","WM","CL","ITW","NSC",
        "GD","NOC","LMT","USB","PNC","TFC","EMR","ETN","APD","ECL","HCA","MCK",
        "PSA","CTAS","EW","KLAC","CDNS","SNPS","ROST","ORLY","AUTO","FAST","ODFL",
        "VRSK","CPRT","PAYX","BIIB","ILMN","AMGN","IDXX","DXCM","IQV","DHI","LEN",
    ]
