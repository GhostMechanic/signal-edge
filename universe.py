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
# Used as a fallback when _fetch_nasdaq100_wiki() fails. Keep this list
# aligned to the real NASDAQ-100; missing tickers here cause silent
# off-universe rejections on perfectly valid mega-cap names. Recent
# additions verified manually:
#   • ORCL (Oracle, NASDAQ:ORCL) — added to NASDAQ-100 in 2023.
#     Surfaced when an ORCL prediction got stamped
#     trade_pass_reason="symbol_not_in_universe" despite Oracle being
#     both an S&P 500 and NASDAQ-100 constituent.
NASDAQ_100 = [
    "AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD","AMGN",
    "AMZN","ANSS","APP","ARM","ASML","AVGO","AZN","BIIB","BKNG","BKR",
    "CCEP","CDNS","CDW","CEG","CHTR","CMCSA","COST","CPRT","CRWD","CSCO",
    "CSGP","CSX","CTAS","CTSH","DDOG","DLTR","DXCM","EA","EXC","FANG",
    "FAST","FTNT","GEHC","GFS","GILD","GOOG","GOOGL","HON","IDXX","ILMN",
    "INTC","INTU","ISRG","KDP","KHC","KLAC","LRCX","LULU","MAR","MCHP",
    "MDB","MDLZ","META","MNST","MRNA","MRVL","MSFT","MU","NFLX","NVDA",
    "NXPI","ODFL","ON","ORCL","ORLY","PANW","PAYX","PCAR","PDD","PEP",
    "PYPL","QCOM","REGN","ROP","ROST","SBUX","SMCI","SNPS","SPLK","TEAM",
    "TMUS","TSLA","TTD","TXN","VRSK","VRTX","WBD","WDAY","XEL","ZS","ZM",
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


# A real browser User-Agent. Wikipedia routinely 403s the default
# Python / pandas user agents from known datacenter IPs (Render, AWS,
# Fly, etc.) — which silently broke the public-ledger pipeline because
# the universe lookup would fall through to the curated fallback list,
# and any ticker not in that list got is_public_ledger=false even when
# it was a legitimate S&P 500 / NASDAQ-100 constituent. Sending a real
# Chrome user agent removes that gate. Kept here as a module-level
# constant so both fetches use the same string.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _fetch_wiki_html(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch a Wikipedia page with a real browser User-Agent.

    pd.read_html doesn't let us customise headers, so we fetch the HTML
    via requests and hand the text to read_html instead. Wikipedia
    serves 403 to the default python-urllib/requests UAs from
    datacenter IPs; this header makes us look like a browser and the
    fetch succeeds from Render's egress range too.
    """
    try:
        import requests
        r = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=timeout)
        if r.status_code != 200:
            logger.warning(
                "Wikipedia fetch %s returned HTTP %s — falling back to "
                "hardcoded universe lists.",
                url, r.status_code,
            )
            return None
        return r.text
    except Exception as e:
        logger.warning(f"Wikipedia fetch {url} failed: {e}")
        return None


def _fetch_sp500_wiki() -> list:
    """Scrape S&P 500 tickers from Wikipedia."""
    html = _fetch_wiki_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    if html is None:
        return []
    try:
        import pandas as pd
        tables = pd.read_html(html, header=0)
        df = tables[0]
        # Column is 'Symbol' or 'Ticker symbol'
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = df[col].str.replace(".", "-", regex=False).tolist()
        return [t.strip() for t in tickers if isinstance(t, str) and t.strip()]
    except Exception as e:
        logger.warning(f"Wikipedia S&P 500 parse failed: {e}")
        return []


def _fetch_nasdaq100_wiki() -> list:
    """Scrape NASDAQ-100 tickers from Wikipedia."""
    html = _fetch_wiki_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    if html is None:
        return NASDAQ_100   # fallback
    try:
        import pandas as pd
        tables = pd.read_html(html, header=0)
        for tbl in tables:
            cols = [str(c).lower() for c in tbl.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                col_idx = next(i for i, c in enumerate(cols) if "ticker" in c or "symbol" in c)
                tickers = tbl.iloc[:, col_idx].tolist()
                tickers = [str(t).strip() for t in tickers if isinstance(t, str) and len(t) < 10]
                if len(tickers) > 50:
                    return tickers
    except Exception as e:
        logger.warning(f"Wikipedia NASDAQ-100 parse failed: {e}")
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
    """Minimal S&P 500 fallback if Wikipedia is unavailable.

    NOTE: this is NOT the full 500-ticker list — it's a curated subset
    of the largest names plus commonly-queried mid-caps. Production
    should hit `_fetch_sp500_wiki()` first and only fall back here on
    network failure. Missing tickers here cause silent off-universe
    rejections on perfectly valid S&P 500 names. When you notice a
    legit S&P 500 ticker getting trade_pass_reason=symbol_not_in_universe
    in the logs, add it to this list with a brief comment.

    Manual additions over the curated baseline:
      • EBAY (eBay) — surfaced when an EBAY 1-year +52% prediction
        got rejected as off-universe despite EBAY being a current
        S&P 500 constituent.
    """
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
        # Manually curated additions — see comment block above.
        "EBAY","ORCL",
        # Recent S&P 500 additions (2023-2024). Without these, calls
        # on names that JUST joined the index get marked
        # is_public_ledger=false whenever the Wikipedia fetch fails
        # (which happens often on Render's datacenter IPs — Wiki
        # serves a 403 to known cloud ranges). Surfaced when DELL
        # 1-year predictions at 80% confidence kept landing private
        # despite DELL being added to the index in September 2024.
        "DELL","PLTR","GEV","KKR","CRWD","ERIE","DECK","SMCI",
        "BLDR","VLTO","INVH","TPL","TRGP","HUBB","WSM","TKO",
        "AXON","UBER","DASH","COIN","APP","TTD","ABNB","SHOP",
        # Additional surfaced misses from the data-integrity audit
        # (category F — high-confidence rows off-ledger with no dedupe
        # ancestor, i.e. excluded purely by the universe check).
        # Each one verified as a current S&P 500 constituent before
        # adding; non-S&P names like FUN, POET, MCH stayed out by design.
        #   • TTWO (Take-Two Interactive) — S&P 500 + NASDAQ-100,
        #     surfaced because the user kept testing chart fixes on it
        #     and the predictions weren't landing on the public ledger
        #     despite confidence ≥ 78%.
        "TTWO",
    ]
