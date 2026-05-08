"""Finnhub API client — news + earnings calendar + company profile.

Powers the /stock/[symbol] page's news feed, upcoming-earnings card,
and company-profile fallbacks (when yfinance is missing fields).

Free tier limits (Finnhub Basic, 2026):
  - 60 calls/min
  - All endpoints required for our pages are in the free tier
  - No commercial-use restrictions for ad-supported sites

Caching strategy: file-backed JSON cache under .finnhub_cache/ with
per-endpoint TTLs. We aggressively cache at the symbol level because
the SEO pages render up to 600 tickers and most fields don't change
intraday — news refreshes hourly, earnings calendar daily, company
profile weekly. Intent: a single nightly batch should cost ~600
total calls (one per ticker), well under the per-minute limit.

Set FINNHUB_API_KEY in env to enable. When unset, every function
returns an empty/None result and logs once — calling code can
gracefully degrade (page sections hide rather than render empty).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

API_BASE = "https://finnhub.io/api/v1"
API_KEY = os.environ.get("FINNHUB_API_KEY")

CACHE_DIR = Path(".finnhub_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Per-endpoint TTLs in seconds. Tuned for the SEO page's read pattern:
# news refreshes most often (within-day relevance), earnings + profile
# barely move week-to-week.
TTL_NEWS_SECONDS         = 60 * 60           # 1 hour
TTL_EARNINGS_SECONDS     = 60 * 60 * 12      # 12 hours
TTL_PROFILE_SECONDS      = 60 * 60 * 24 * 7  # 7 days
TTL_RECOMMENDATION_SECONDS = 60 * 60 * 24    # 1 day

# Soft per-process rate limit. Free tier is 60/min; we cap at 50/min
# and sleep briefly when we hit it. This is single-process — production
# is one Render container, so process-local is sufficient.
_RATE_BUCKET: list[float] = []
_RATE_LIMIT_PER_MINUTE = 50

# One-shot warning when API key is missing — don't spam logs once per
# call.
_WARNED_NO_KEY = False


# ─── Public API ──────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    """Mirrors a Finnhub `/company-news` item, trimmed to the fields the page renders."""
    headline: str
    source: str
    summary: str
    url: str
    image: Optional[str]
    datetime_unix: int   # seconds since epoch — what Finnhub returns
    category: Optional[str] = None
    related: Optional[str] = None


@dataclass
class EarningsItem:
    """One row from `/calendar/earnings`. Future and past, depending on date range."""
    date: str            # YYYY-MM-DD
    eps_estimate: Optional[float]
    eps_actual: Optional[float]
    revenue_estimate: Optional[float]
    revenue_actual: Optional[float]
    hour: Optional[str]  # 'bmo' (before market open) | 'amc' (after market close) | 'dmh' | None
    quarter: Optional[int]
    year: Optional[int]


@dataclass
class CompanyProfile:
    """Lightweight profile block — fills in fields yfinance.info misses or returns wrong."""
    name: Optional[str]
    ticker: str
    exchange: Optional[str]
    industry: Optional[str]
    ipo_date: Optional[str]
    logo_url: Optional[str]
    weburl: Optional[str]
    market_cap: Optional[float]
    shares_outstanding: Optional[float]
    country: Optional[str]
    currency: Optional[str]


def get_company_news(symbol: str, days_back: int = 7) -> list[NewsItem]:
    """Latest company-specific headlines for `symbol`, newest first.

    Returns up to 50 items by default. Empty list when:
      - FINNHUB_API_KEY is unset
      - Finnhub returns an error
      - Cache is fresh and empty (truly no news)
    """
    if not _ensure_key():
        return []

    today = date.today()
    earlier = today - timedelta(days=days_back)
    params = {
        "symbol": symbol.upper(),
        "from":   earlier.isoformat(),
        "to":     today.isoformat(),
    }
    cache_key = f"news_{symbol.upper()}_{days_back}d"
    raw = _fetch_with_cache(
        endpoint="/company-news",
        params=params,
        cache_key=cache_key,
        ttl_seconds=TTL_NEWS_SECONDS,
    )
    if not isinstance(raw, list):
        return []

    items: list[NewsItem] = []
    seen_urls: set[str] = set()  # Finnhub occasionally double-reports the same headline
    for r in raw:
        if not isinstance(r, dict):
            continue
        url = (r.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        items.append(NewsItem(
            headline = (r.get("headline") or "").strip(),
            source   = (r.get("source") or "").strip(),
            summary  = (r.get("summary") or "").strip(),
            url      = url,
            image    = (r.get("image") or None) or None,
            datetime_unix = int(r.get("datetime") or 0),
            category = r.get("category") or None,
            related  = r.get("related")  or None,
        ))
    items.sort(key=lambda x: x.datetime_unix, reverse=True)
    return items


def get_earnings_calendar(
    symbol: str,
    *,
    days_forward: int = 60,
    days_backward: int = 90,
) -> list[EarningsItem]:
    """Earnings rows for `symbol` from (today − days_backward) to (today + days_forward).

    Returns past quarters (with actuals) AND the next upcoming quarter
    (with estimates) so the page can render both 'next earnings' and
    'last 4 quarters' from a single call.
    """
    if not _ensure_key():
        return []

    today = date.today()
    start = today - timedelta(days=days_backward)
    end   = today + timedelta(days=days_forward)
    params = {
        "symbol": symbol.upper(),
        "from":   start.isoformat(),
        "to":     end.isoformat(),
    }
    cache_key = f"earnings_{symbol.upper()}_{days_backward}b_{days_forward}f"
    raw = _fetch_with_cache(
        endpoint="/calendar/earnings",
        params=params,
        cache_key=cache_key,
        ttl_seconds=TTL_EARNINGS_SECONDS,
    )

    # /calendar/earnings returns { "earningsCalendar": [...] }
    rows = (raw or {}).get("earningsCalendar") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        return []

    out: list[EarningsItem] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        out.append(EarningsItem(
            date             = (r.get("date") or "")[:10],
            eps_estimate     = _safe_float(r.get("epsEstimate")),
            eps_actual       = _safe_float(r.get("epsActual")),
            revenue_estimate = _safe_float(r.get("revenueEstimate")),
            revenue_actual   = _safe_float(r.get("revenueActual")),
            hour             = r.get("hour") or None,
            quarter          = _safe_int(r.get("quarter")),
            year             = _safe_int(r.get("year")),
        ))
    out.sort(key=lambda x: x.date or "", reverse=True)
    return out


def get_company_profile(symbol: str) -> Optional[CompanyProfile]:
    """Static-ish company profile — name, exchange, industry, IPO date, logo URL.

    yfinance.info covers most of this but logo_url + reliable IPO date
    only come from Finnhub. Cached aggressively (1 week) since the
    underlying data rarely moves.
    """
    if not _ensure_key():
        return None

    params = {"symbol": symbol.upper()}
    raw = _fetch_with_cache(
        endpoint="/stock/profile2",
        params=params,
        cache_key=f"profile_{symbol.upper()}",
        ttl_seconds=TTL_PROFILE_SECONDS,
    )
    if not isinstance(raw, dict) or not raw.get("ticker"):
        # Empty {} comes back when Finnhub doesn't have the ticker
        # (rare for canonical-universe names, common for OTC).
        return None

    return CompanyProfile(
        name              = raw.get("name") or None,
        ticker            = (raw.get("ticker") or symbol).upper(),
        exchange          = raw.get("exchange") or None,
        industry          = raw.get("finnhubIndustry") or None,
        ipo_date          = raw.get("ipo") or None,
        logo_url          = raw.get("logo") or None,
        weburl            = raw.get("weburl") or None,
        market_cap        = _safe_float(raw.get("marketCapitalization")),
        shares_outstanding = _safe_float(raw.get("shareOutstanding")),
        country           = raw.get("country") or None,
        currency          = raw.get("currency") or None,
    )


@dataclass
class CompanyMetrics:
    """Subset of Finnhub /stock/metric `metric` block we surface on the page.

    Used as a fallback when yfinance.info comes back empty (which is
    perpetually the case on cloud-hosted runtimes like Render — Yahoo
    rate-limits or blocks data-center IP ranges, while residential IPs
    work fine).
    """
    pe_ttm: Optional[float]
    forward_pe: Optional[float]
    peg_ratio: Optional[float]
    div_yield: Optional[float]
    beta: Optional[float]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]
    avg_volume_10d: Optional[float]
    avg_volume_3m: Optional[float]
    eps_ttm: Optional[float]
    revenue_per_share_ttm: Optional[float]
    book_value_per_share: Optional[float]
    profit_margin: Optional[float]
    return_on_equity: Optional[float]
    debt_to_equity: Optional[float]
    current_ratio: Optional[float]


def get_company_metrics(symbol: str) -> Optional[CompanyMetrics]:
    """Finnhub `/stock/metric?metric=all` — the financial-fundamentals
    block. Free-tier endpoint, cached aggressively (1 day).

    Field name notes — Finnhub's `metric` dict uses period-suffixed
    field names. We pick canonical keys and tolerate the API's
    occasional renames.
    """
    if not _ensure_key():
        return None

    raw = _fetch_with_cache(
        endpoint="/stock/metric",
        params={"symbol": symbol.upper(), "metric": "all"},
        cache_key=f"metrics_{symbol.upper()}",
        ttl_seconds=60 * 60 * 24,  # 1 day
    )
    if not isinstance(raw, dict):
        return None
    metric = raw.get("metric") or {}
    if not isinstance(metric, dict):
        return None

    def _pick(*keys: str) -> Optional[float]:
        """Try multiple Finnhub field names — `metric` schema has
        evolved and different names appear for the same concept
        across plan tiers."""
        for k in keys:
            v = metric.get(k)
            f = _safe_float(v)
            if f is not None:
                return f
        return None

    return CompanyMetrics(
        pe_ttm                = _pick("peTTM", "peExclExtraTTM", "peNormalizedAnnual"),
        forward_pe            = _pick("forwardPE", "peNormalizedAnnual"),
        peg_ratio             = _pick("pegRatio"),
        # Finnhub returns yield as a percentage already (1.45 = 1.45%).
        # Don't multiply.
        div_yield             = _pick(
            "currentDividendYieldTTM",
            "dividendYieldIndicatedAnnual",
            "dividendYield5Y",
        ),
        beta                  = _pick("beta"),
        fifty_two_week_high   = _pick("52WeekHigh"),
        fifty_two_week_low    = _pick("52WeekLow"),
        avg_volume_10d        = _pick("10DayAverageTradingVolume"),
        avg_volume_3m         = _pick("3MonthAverageTradingVolume"),
        eps_ttm               = _pick("epsTTM", "epsNormalizedAnnual"),
        revenue_per_share_ttm = _pick("revenuePerShareTTM"),
        book_value_per_share  = _pick("bookValuePerShareAnnual", "tangibleBookValuePerShareAnnual"),
        profit_margin         = _pick("netProfitMarginTTM", "netProfitMargin5Y"),
        return_on_equity      = _pick("roeTTM", "roeRfy"),
        debt_to_equity        = _pick("totalDebt/totalEquityAnnual", "longTermDebt/equityAnnual"),
        current_ratio         = _pick("currentRatioAnnual", "currentRatioQuarterly"),
    )


def get_recommendation_trends(symbol: str) -> list[dict]:
    """Analyst recommendation distribution (strongBuy / buy / hold / sell / strongSell)
    over time. One row per month, newest first. Free-tier endpoint.

    Used on the page's "Analyst consensus" card alongside the yfinance
    target-price block.
    """
    if not _ensure_key():
        return []

    raw = _fetch_with_cache(
        endpoint="/stock/recommendation",
        params={"symbol": symbol.upper()},
        cache_key=f"recs_{symbol.upper()}",
        ttl_seconds=TTL_RECOMMENDATION_SECONDS,
    )
    if not isinstance(raw, list):
        return []
    # Trim to last 12 months; sort newest first.
    trimmed = raw[:12]
    trimmed.sort(key=lambda r: r.get("period") or "", reverse=True)
    return trimmed


# ─── Internals ───────────────────────────────────────────────────────────────

def _ensure_key() -> bool:
    """Return True when FINNHUB_API_KEY is set; warn once otherwise."""
    global _WARNED_NO_KEY
    if API_KEY:
        return True
    if not _WARNED_NO_KEY:
        logger.warning(
            "FINNHUB_API_KEY is not set. Finnhub-backed page sections "
            "(news, earnings, recommendation trends, company profile) "
            "will render empty. Sign up at finnhub.io for a free key "
            "and set FINNHUB_API_KEY in env."
        )
        _WARNED_NO_KEY = True
    return False


def _fetch_with_cache(
    *,
    endpoint: str,
    params: dict,
    cache_key: str,
    ttl_seconds: int,
) -> Any:
    """File-backed cache fetcher. Returns the cached payload when fresh,
    else hits the network, persists, and returns. On network error,
    falls back to stale cache (better than empty)."""
    cache_path = CACHE_DIR / f"{cache_key}.json"

    # Fresh cache? Use it.
    if cache_path.exists():
        age_seconds = time.time() - cache_path.stat().st_mtime
        if age_seconds < ttl_seconds:
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                # Corrupt cache file — fall through to refetch.
                pass

    # Hit the network.
    try:
        _rate_limit_throttle()
        url = f"{API_BASE}{endpoint}"
        full_params = {**params, "token": API_KEY}
        resp = requests.get(url, params=full_params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        try:
            cache_path.write_text(json.dumps(payload))
        except Exception as exc:  # noqa: BLE001
            logger.warning("finnhub cache write failed for %s: %s", cache_key, exc)
        return payload
    except requests.RequestException as exc:
        logger.warning(
            "finnhub fetch %s failed: %s. Falling back to stale cache "
            "if present.", endpoint, exc,
        )
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                pass
        return None
    except Exception as exc:  # noqa: BLE001
        logger.exception("finnhub %s unexpected error: %s", endpoint, exc)
        return None


def _rate_limit_throttle() -> None:
    """Soft 50/min throttle — sleep briefly when the bucket is full.

    Tracks call timestamps in a process-local list, prunes older than
    60s, and sleeps until the oldest falls off when the bucket is full.
    Cheap and good enough for our ~600/night batch volume.
    """
    now = time.time()
    cutoff = now - 60
    while _RATE_BUCKET and _RATE_BUCKET[0] < cutoff:
        _RATE_BUCKET.pop(0)
    if len(_RATE_BUCKET) >= _RATE_LIMIT_PER_MINUTE:
        wait = _RATE_BUCKET[0] + 60 - now + 0.05
        if wait > 0:
            logger.debug("finnhub rate limit reached, sleeping %.2fs", wait)
            time.sleep(wait)
    _RATE_BUCKET.append(time.time())


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ─── CLI smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"\n=== Finnhub smoke test — {sym} ===\n")

    news = get_company_news(sym, days_back=3)
    print(f"News (last 3d): {len(news)} items")
    for n in news[:3]:
        ts = datetime.fromtimestamp(n.datetime_unix, tz=timezone.utc).isoformat()
        print(f"  [{ts}] {n.source}: {n.headline[:80]}")

    earnings = get_earnings_calendar(sym, days_forward=120, days_backward=120)
    print(f"\nEarnings (±120d): {len(earnings)} items")
    for e in earnings[:4]:
        print(f"  {e.date}  Q{e.quarter} {e.year}  est={e.eps_estimate}  act={e.eps_actual}  hr={e.hour}")

    profile = get_company_profile(sym)
    print(f"\nProfile:")
    if profile:
        print(f"  {profile.name} ({profile.ticker}) — {profile.industry} — {profile.country}")
        print(f"  IPO {profile.ipo_date}  logo {profile.logo_url}")
    else:
        print("  (none)")

    recs = get_recommendation_trends(sym)
    print(f"\nRecommendation trends: {len(recs)} months")
    for r in recs[:3]:
        print(f"  {r.get('period')}  buy={r.get('buy')}  hold={r.get('hold')}  sell={r.get('sell')}")
