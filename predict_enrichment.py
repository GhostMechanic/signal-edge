"""Post-prediction enrichment.

Augments the /api/predict response with the additional context the
receipt page needs to show "what's actually driving this call" beyond
the raw model output:

  • next_earnings   — date + EPS estimate + days_until (Finnhub)
  • news_sentiment  — keyword-scored from the headlines we already
                       fetch (no extra API call)
  • prior_calls     — last N settled predictions on this ticker +
                       hit/miss aggregates (db helper)
  • volume_context  — today's vs 30d-average daily volume (yfinance)
  • macro_events    — Fed FOMC / CPI / NFP within horizon window
                       (hardcoded 2026 schedule — small enough to ship
                       without a calendar API)
  • risk_flags      — quick-read flag list from the quote fields
                       (extreme P/E, micro-cap, high beta, etc.)

Each section is optional and degrades cleanly when its inputs are
missing — frontend renders the section empty rather than showing a
half-broken card.

Pure module — no FastAPI routes, no module-level network calls.
Callers thread the data they want enriched in and consume what comes
back. Safe to import from any context.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Public dataclasses ──────────────────────────────────────────────────────

@dataclass
class NextEarnings:
    date: str             # YYYY-MM-DD
    days_until: int
    hour: Optional[str]   # "bmo" | "amc" | None
    eps_estimate: Optional[float]


@dataclass
class RecentEarning:
    """Past earnings date inside the price_30d window — used by the
    chart to render a vertical marker so the user can see exactly
    which day the price reacted to the most recent print."""
    date: str
    eps_estimate: Optional[float]
    eps_actual: Optional[float]
    surprise_pct: Optional[float]   # (actual - est) / |est| * 100


@dataclass
class NewsSentiment:
    score: float          # -1.0 (very bearish) → +1.0 (very bullish)
    label: str            # "bullish" | "slightly bullish" | "neutral" | "slightly bearish" | "bearish"
    headline_count: int
    days: int             # window we sampled


@dataclass
class PriorCallSummary:
    last_n: int
    hit: int
    missed: int
    partial: int
    open_count: int
    avg_return_pct: Optional[float]


@dataclass
class VolumeContext:
    today_volume: Optional[int]
    avg_30d_volume: Optional[int]
    ratio: Optional[float]   # today / avg30d
    label: str               # "above avg" | "below avg" | "in line" | "unavailable"


@dataclass
class MacroEvent:
    date: str
    days_until: int
    name: str                # "Fed FOMC", "CPI", "NFP", "FOMC Minutes"
    severity: str            # "high" | "medium"


@dataclass
class RiskFlag:
    code: str                # "extreme_pe" | "micro_cap" | "high_beta" | "near_52w_high" | "near_52w_low" | "low_volume"
    label: str               # human-readable short label
    detail: str              # 1-line explanation
    tone: str                # "warn" | "info"


@dataclass
class SectorRelative:
    """How the symbol has performed against its sector ETF over the
    last 30 days. Frames the model's call against the macro / sector
    backdrop — outperforming the sector is meaningfully different
    from "the whole sector is up."""
    sector: str              # GICS-style label, e.g. "Technology"
    etf_symbol: str          # e.g. "XLK"
    days: int                # window we sampled (typically 30)
    symbol_return_pct: float       # % return over the window
    sector_return_pct: float       # % return over the window
    delta_pp: float                # symbol - sector, in pp
    label: str               # "outperforming" | "underperforming" | "in line"


@dataclass
class PredictEnrichment:
    next_earnings: Optional[NextEarnings] = None
    recent_earnings: list[RecentEarning] = field(default_factory=list)
    news_sentiment: Optional[NewsSentiment] = None
    prior_calls: Optional[PriorCallSummary] = None
    volume_context: Optional[VolumeContext] = None
    macro_events: list[MacroEvent] = field(default_factory=list)
    risk_flags: list[RiskFlag] = field(default_factory=list)
    sector_relative: Optional[SectorRelative] = None

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        if self.next_earnings:    out["next_earnings"]    = asdict(self.next_earnings)
        if self.recent_earnings:  out["recent_earnings"]  = [asdict(e) for e in self.recent_earnings]
        if self.news_sentiment:   out["news_sentiment"]   = asdict(self.news_sentiment)
        if self.prior_calls:      out["prior_calls"]      = asdict(self.prior_calls)
        if self.volume_context:   out["volume_context"]   = asdict(self.volume_context)
        if self.macro_events:     out["macro_events"]     = [asdict(e) for e in self.macro_events]
        if self.risk_flags:       out["risk_flags"]       = [asdict(f) for f in self.risk_flags]
        if self.sector_relative:  out["sector_relative"]  = asdict(self.sector_relative)
        return out


# ─── Public entrypoint ───────────────────────────────────────────────────────

def enrich_prediction(
    *,
    symbol: str,
    horizon_days_max: int,
    quote_info: Optional[dict] = None,
    price_history_30d: Optional[list[dict]] = None,
) -> PredictEnrichment:
    """Compute all enrichments for a symbol + the longest horizon under
    consideration. Resilient — every section is wrapped so a single
    fetch failure can't take the others down with it.

    Args:
        symbol:           Uppercase ticker.
        horizon_days_max: Longest horizon in days (e.g. 365 for 1y).
                           Used to scope earnings + macro events to the
                           prediction's window.
        quote_info:       Anything matching what fetch_stock_info returns —
                           market_cap, pe_ratio, beta, fifty_two_week_high/low,
                           avg_volume.
        price_history_30d: Last 30 daily {date, price[, volume]} bars
                           from the predict path. When entries include
                           a `volume` field, used for volume_context.
    """
    out = PredictEnrichment()
    sym = (symbol or "").upper().strip()
    if not sym:
        return out

    # ── Earnings (next + recent past) ───────────────────────────────────
    try:
        out.next_earnings = _next_earnings(sym, horizon_days_max)
    except Exception as exc:  # noqa: BLE001
        logger.warning("enrich %s: next_earnings failed: %s", sym, exc)
    try:
        # Past earnings inside the chart's 30-day historical window.
        # Used by the frontend to render vertical markers on the
        # historical line so the user can see the day the price
        # reacted to the print.
        out.recent_earnings = _recent_earnings(sym, days_back=45)
    except Exception as exc:  # noqa: BLE001
        logger.warning("enrich %s: recent_earnings failed: %s", sym, exc)

    # ── News sentiment ──────────────────────────────────────────────────
    try:
        out.news_sentiment = _news_sentiment(sym)
    except Exception as exc:  # noqa: BLE001
        logger.warning("enrich %s: news_sentiment failed: %s", sym, exc)

    # ── Prior calls ─────────────────────────────────────────────────────
    try:
        out.prior_calls = _prior_calls_summary(sym)
    except Exception as exc:  # noqa: BLE001
        logger.warning("enrich %s: prior_calls failed: %s", sym, exc)

    # ── Volume context ──────────────────────────────────────────────────
    try:
        out.volume_context = _volume_context(quote_info, price_history_30d)
    except Exception as exc:  # noqa: BLE001
        logger.warning("enrich %s: volume_context failed: %s", sym, exc)

    # ── Macro events ────────────────────────────────────────────────────
    try:
        out.macro_events = _macro_events_in_window(horizon_days_max)
    except Exception as exc:  # noqa: BLE001
        logger.warning("enrich %s: macro_events failed: %s", sym, exc)

    # ── Risk flags ──────────────────────────────────────────────────────
    try:
        out.risk_flags = _risk_flags(quote_info)
    except Exception as exc:  # noqa: BLE001
        logger.warning("enrich %s: risk_flags failed: %s", sym, exc)

    # ── Sector-relative context ─────────────────────────────────────────
    try:
        out.sector_relative = _sector_relative(sym, quote_info, price_history_30d)
    except Exception as exc:  # noqa: BLE001
        logger.warning("enrich %s: sector_relative failed: %s", sym, exc)

    return out


# ─── Section impls ───────────────────────────────────────────────────────────

def _next_earnings(symbol: str, horizon_days_max: int) -> Optional[NextEarnings]:
    """Pull the soonest upcoming earnings date from Finnhub. Returns
    None when Finnhub isn't configured, no upcoming earnings on the
    calendar, or every earnings date sits beyond the horizon window
    (so the catalyst is irrelevant to this prediction)."""
    try:
        from finnhub_client import get_earnings_calendar
    except Exception:
        return None

    rows = get_earnings_calendar(symbol, days_forward=max(180, horizon_days_max + 30), days_backward=0)
    if not rows:
        return None

    today = date.today()
    upcoming: list = []
    for r in rows:
        try:
            d = date.fromisoformat((r.date or "")[:10])
        except (ValueError, TypeError):
            continue
        if d < today:
            continue
        upcoming.append((d, r))
    if not upcoming:
        return None
    upcoming.sort(key=lambda t: t[0])
    soonest_date, soonest_row = upcoming[0]
    days_until = (soonest_date - today).days
    return NextEarnings(
        date=soonest_date.isoformat(),
        days_until=days_until,
        hour=soonest_row.hour,
        eps_estimate=soonest_row.eps_estimate,
    )


def _recent_earnings(symbol: str, *, days_back: int = 45) -> list[RecentEarning]:
    """Past earnings dates within the last `days_back` days. Frontend
    uses these to render vertical markers on the chart's historical
    line — lets the user spot 'oh that's the day they reported and
    the stock dropped 8%'.

    Returns at most 2 events (one quarter is normal; two would be a
    long batch). Surprise % when both estimate + actual are available.
    """
    try:
        from finnhub_client import get_earnings_calendar
    except Exception:
        return []
    rows = get_earnings_calendar(
        symbol,
        days_forward=0,
        days_backward=days_back,
    )
    if not rows:
        return []
    today = date.today()
    out: list[RecentEarning] = []
    for r in rows:
        try:
            d = date.fromisoformat((r.date or "")[:10])
        except (ValueError, TypeError):
            continue
        if d > today or (today - d).days > days_back:
            continue
        surprise = None
        if r.eps_actual is not None and r.eps_estimate is not None:
            denom = abs(r.eps_estimate) if r.eps_estimate != 0 else 0.01
            surprise = round((r.eps_actual - r.eps_estimate) / denom * 100, 2)
        out.append(RecentEarning(
            date=d.isoformat(),
            eps_estimate=r.eps_estimate,
            eps_actual=r.eps_actual,
            surprise_pct=surprise,
        ))
    out.sort(key=lambda e: e.date, reverse=True)
    return out[:2]


# Simple keyword sentiment lexicon. Positive / negative leans drawn
# from typical financial-headline patterns. Not a sentiment model —
# just a cheap directional skew indicator suitable for "is the news
# wind blowing the same direction as the model?".
_POSITIVE_WORDS = {
    "beat", "beats", "rally", "surge", "soars", "soared", "soar",
    "upgrade", "upgraded", "raised", "boost", "boosts", "boosted",
    "record", "high", "highs", "outperform", "strong", "tops", "topped",
    "wins", "win", "profit", "profits", "growth", "grew", "expand",
    "expanded", "expansion", "approves", "approved", "approval",
    "buyback", "dividend", "increase", "increased",
}
_NEGATIVE_WORDS = {
    "miss", "missed", "drop", "drops", "fall", "falls", "fell", "plunge",
    "plunges", "plunged", "downgrade", "downgraded", "lower", "lowered",
    "weak", "weaker", "warning", "lawsuit", "investigation", "fraud",
    "loss", "losses", "decline", "declined", "cut", "cuts", "slashed",
    "slumps", "tumble", "tumbles", "tumbled", "crash", "crashes",
    "delay", "delays", "delayed", "fine", "fines", "probe", "concerns",
    "headwinds", "layoffs", "bankruptcy", "default",
}


def _news_sentiment(symbol: str, *, days: int = 7) -> Optional[NewsSentiment]:
    """Cheap keyword-based sentiment over the headlines we already fetch
    from Finnhub for the page. Range: -1 (very bearish) → +1 (very
    bullish). Returns None when no headlines are available."""
    try:
        from finnhub_client import get_company_news
    except Exception:
        return None

    items = get_company_news(symbol, days_back=days)
    if not items:
        return None

    pos_hits = 0
    neg_hits = 0
    for n in items:
        headline = (n.headline or "").lower()
        for word in _POSITIVE_WORDS:
            if word in headline:
                pos_hits += 1
                break
        for word in _NEGATIVE_WORDS:
            if word in headline:
                neg_hits += 1
                break

    total_signal = pos_hits + neg_hits
    if total_signal == 0:
        score = 0.0
    else:
        score = (pos_hits - neg_hits) / max(total_signal, 1)
    score = max(-1.0, min(1.0, round(score, 3)))

    if score >= 0.4:
        label = "bullish"
    elif score >= 0.1:
        label = "slightly bullish"
    elif score <= -0.4:
        label = "bearish"
    elif score <= -0.1:
        label = "slightly bearish"
    else:
        label = "neutral"

    return NewsSentiment(
        score=score,
        label=label,
        headline_count=len(items),
        days=days,
    )


def _prior_calls_summary(symbol: str, *, last_n: int = 5) -> Optional[PriorCallSummary]:
    """Aggregate last N public-ledger predictions on this ticker. Counts
    hit/miss/partial/open + average return on settled rows."""
    try:
        from db import get_symbol_predictions
    except Exception:
        return None

    rows = get_symbol_predictions(symbol, limit=last_n, offset=0)
    if not rows:
        return None

    hit = partial = missed = open_count = 0
    returns: list[float] = []
    for r in rows:
        verdict = (r.get("verdict") or "OPEN").upper()
        if verdict == "HIT":
            hit += 1
        elif verdict == "MISSED":
            missed += 1
        elif verdict == "PARTIAL":
            partial += 1
        else:
            open_count += 1
        ar = r.get("actual_return")
        if ar is not None:
            try:
                returns.append(float(ar))
            except (TypeError, ValueError):
                pass

    avg_ret = round(sum(returns) / len(returns), 4) if returns else None

    return PriorCallSummary(
        last_n=len(rows),
        hit=hit,
        missed=missed,
        partial=partial,
        open_count=open_count,
        avg_return_pct=avg_ret,
    )


def _volume_context(
    quote_info: Optional[dict],
    price_history_30d: Optional[list[dict]],
) -> Optional[VolumeContext]:
    """Compare today's volume to the trailing 30d average. Falls back
    cleanly when either side of the ratio is missing."""
    if not quote_info and not price_history_30d:
        return None

    avg_30d = None
    if quote_info and isinstance(quote_info, dict):
        v = quote_info.get("avg_volume")
        if isinstance(v, (int, float)) and v > 0:
            avg_30d = float(v)

    today_vol: Optional[float] = None
    if price_history_30d:
        last = price_history_30d[-1] if price_history_30d else None
        if isinstance(last, dict):
            v = last.get("volume")
            if isinstance(v, (int, float)) and v > 0:
                today_vol = float(v)

    if today_vol is None and avg_30d is None:
        return None

    ratio: Optional[float] = None
    label = "unavailable"
    if today_vol is not None and avg_30d is not None and avg_30d > 0:
        ratio = round(today_vol / avg_30d, 2)
        if ratio >= 1.5:
            label = "above avg"
        elif ratio <= 0.6:
            label = "below avg"
        else:
            label = "in line"

    return VolumeContext(
        today_volume=int(today_vol) if today_vol is not None else None,
        avg_30d_volume=int(avg_30d) if avg_30d is not None else None,
        ratio=ratio,
        label=label,
    )


# Hardcoded 2026 macro schedule. Conservative coverage — Fed FOMC
# meeting weeks (high) plus CPI release dates (high) plus NFP first
# Fridays (medium). Drop-in replacement with a real calendar API
# whenever we want longer-tail coverage; this is enough for "is there
# a Fed event inside this prediction's window?" which is the
# narrative beat that matters.
_MACRO_SCHEDULE_2026 = [
    # Fed FOMC meetings (announced schedule)
    ("2026-01-28", "Fed FOMC", "high"),
    ("2026-03-18", "Fed FOMC", "high"),
    ("2026-05-06", "Fed FOMC", "high"),
    ("2026-06-17", "Fed FOMC", "high"),
    ("2026-07-29", "Fed FOMC", "high"),
    ("2026-09-16", "Fed FOMC", "high"),
    ("2026-10-28", "Fed FOMC", "high"),
    ("2026-12-09", "Fed FOMC", "high"),
    # CPI release schedule (BLS published mid-month)
    ("2026-05-13", "CPI", "high"),
    ("2026-06-11", "CPI", "high"),
    ("2026-07-15", "CPI", "high"),
    ("2026-08-12", "CPI", "high"),
    ("2026-09-10", "CPI", "high"),
    ("2026-10-15", "CPI", "high"),
    ("2026-11-12", "CPI", "high"),
    ("2026-12-10", "CPI", "high"),
    # Non-farm payrolls — first Fridays
    ("2026-05-08", "NFP", "medium"),
    ("2026-06-05", "NFP", "medium"),
    ("2026-07-03", "NFP", "medium"),
    ("2026-08-07", "NFP", "medium"),
    ("2026-09-04", "NFP", "medium"),
    ("2026-10-02", "NFP", "medium"),
    ("2026-11-06", "NFP", "medium"),
    ("2026-12-04", "NFP", "medium"),
]


def _macro_events_in_window(horizon_days_max: int) -> list[MacroEvent]:
    """Return any macro events that fall inside the prediction window.
    Skips events more than horizon_days_max in the future (irrelevant
    to the call) and events in the past."""
    today = date.today()
    cutoff = today + timedelta(days=horizon_days_max)
    out: list[MacroEvent] = []
    for d_str, name, severity in _MACRO_SCHEDULE_2026:
        try:
            d = date.fromisoformat(d_str)
        except ValueError:
            continue
        if d < today:
            continue
        if d > cutoff:
            continue
        out.append(MacroEvent(
            date=d_str,
            days_until=(d - today).days,
            name=name,
            severity=severity,
        ))
    return out


def _risk_flags(quote_info: Optional[dict]) -> list[RiskFlag]:
    """Auto-derived risk chips from quote fundamentals. Each flag fires
    on a known threshold; tone="warn" for things a user should think
    about, tone="info" for context-only flags."""
    flags: list[RiskFlag] = []
    if not quote_info or not isinstance(quote_info, dict):
        return flags

    pe = _safe_float(quote_info.get("pe_ratio"))
    fwd_pe = _safe_float(quote_info.get("forward_pe"))
    market_cap = _safe_float(quote_info.get("market_cap"))
    beta = _safe_float(quote_info.get("beta"))
    avg_volume = _safe_float(quote_info.get("avg_volume"))
    high_52w = _safe_float(quote_info.get("fifty_two_week_high") or quote_info.get("52w_high"))
    low_52w = _safe_float(quote_info.get("fifty_two_week_low") or quote_info.get("52w_low"))
    current = _safe_float(quote_info.get("current_price"))

    # Extreme P/E — classic "earnings anomaly or one-off" flag.
    if pe is not None and pe >= 200:
        flags.append(RiskFlag(
            code="extreme_pe",
            label=f"Extreme P/E ({pe:.0f}x)",
            detail=(
                f"Trailing P/E ratio of {pe:.0f}x is far above market norms — "
                "could indicate earnings collapse, a one-off charge, or unusual valuation."
            ),
            tone="warn",
        ))
    elif pe is not None and pe < 0:
        flags.append(RiskFlag(
            code="negative_pe",
            label="Negative P/E",
            detail="Trailing earnings are negative — the stock has lost money over the last 12 months.",
            tone="warn",
        ))

    # Micro/small-cap.
    if market_cap is not None:
        if market_cap < 300_000_000:
            flags.append(RiskFlag(
                code="micro_cap",
                label="Micro-cap (<$300M)",
                detail="Small market cap typically means thinner liquidity and higher volatility on news.",
                tone="warn",
            ))
        elif market_cap < 2_000_000_000:
            flags.append(RiskFlag(
                code="small_cap",
                label="Small-cap (<$2B)",
                detail="Small-cap names often see larger price moves on lower volume.",
                tone="info",
            ))

    # Beta extremes.
    if beta is not None:
        if beta >= 2.0:
            flags.append(RiskFlag(
                code="high_beta",
                label=f"High beta ({beta:.2f})",
                detail=f"Beta of {beta:.2f} means roughly {beta:.1f}x the market's swings.",
                tone="warn",
            ))
        elif beta < 0:
            flags.append(RiskFlag(
                code="negative_beta",
                label=f"Negative beta ({beta:.2f})",
                detail="Stock historically moves opposite the market — defensive characteristic.",
                tone="info",
            ))

    # 52-week extremes.
    if current is not None and high_52w is not None and low_52w is not None and high_52w > low_52w:
        position_pct = (current - low_52w) / (high_52w - low_52w) * 100
        if position_pct >= 95:
            flags.append(RiskFlag(
                code="near_52w_high",
                label="At 52W high",
                detail=f"Trading {position_pct:.0f}% of the way through the 52-week range — limited upside room.",
                tone="info",
            ))
        elif position_pct <= 5:
            flags.append(RiskFlag(
                code="near_52w_low",
                label="At 52W low",
                detail=f"Trading {position_pct:.0f}% of the way through the 52-week range — limited downside room.",
                tone="info",
            ))

    # Low-volume names.
    if avg_volume is not None and avg_volume < 200_000:
        flags.append(RiskFlag(
            code="low_volume",
            label="Low volume",
            detail=f"Avg daily volume {avg_volume:,.0f} — fills may slip, spreads may widen.",
            tone="warn",
        ))

    return flags


# GICS-style sector → SPDR sector ETF mapping. yfinance returns sector
# labels in either GICS form ("Technology", "Health Care") or the
# Yahoo-flavoured form ("Technology", "Healthcare", "Consumer
# Defensive", "Consumer Cyclical"). We normalise both into the same
# ETF lookup. Unknown / missing sectors fall through to None and the
# signal is suppressed rather than rendering "Sector relative: ?".
_SECTOR_TO_ETF = {
    # Yahoo Finance variants
    "Technology":              "XLK",
    "Healthcare":              "XLV",
    "Health Care":             "XLV",
    "Financial Services":      "XLF",
    "Financials":              "XLF",
    "Consumer Cyclical":       "XLY",
    "Consumer Discretionary":  "XLY",
    "Consumer Defensive":      "XLP",
    "Consumer Staples":        "XLP",
    "Energy":                  "XLE",
    "Utilities":               "XLU",
    "Real Estate":             "XLRE",
    "Basic Materials":         "XLB",
    "Materials":               "XLB",
    "Industrials":             "XLI",
    "Communication Services":  "XLC",
}


def _sector_relative(
    symbol: str,
    quote_info: Optional[dict],
    price_history_30d: Optional[list[dict]],
) -> Optional[SectorRelative]:
    """Compare the symbol's 30-day return against its sector ETF's 30d
    return. Captures the 'is this name moving with its sector or
    against it' read that's hard to get from individual fundamentals.
    """
    if not quote_info or not price_history_30d or len(price_history_30d) < 5:
        return None
    sector = quote_info.get("sector")
    if not sector or not isinstance(sector, str) or sector == "N/A":
        return None
    etf = _SECTOR_TO_ETF.get(sector)
    if not etf or etf.upper() == symbol.upper():
        # Don't compare a sector ETF to itself.
        return None

    # Symbol's 30-day return — first valid → last valid close.
    sym_first = None
    sym_last = None
    for p in price_history_30d:
        if isinstance(p, dict) and isinstance(p.get("price"), (int, float)) and p["price"] > 0:
            if sym_first is None:
                sym_first = float(p["price"])
            sym_last = float(p["price"])
    if sym_first is None or sym_last is None or sym_first <= 0:
        return None
    sym_ret = ((sym_last - sym_first) / sym_first) * 100

    # Fetch the ETF's same-window return. Cheap call — yfinance caches
    # under the hood and a sector ETF gets queried often enough to be
    # warm. Wrapped in try/except so a yfinance hiccup just suppresses
    # this signal without nuking the rest of the enrichment.
    try:
        from data_fetcher import fetch_stock_data
        df = fetch_stock_data(etf, period="2mo")
    except Exception as exc:
        logger.debug("sector_relative: fetch_stock_data(%s) failed: %s", etf, exc)
        return None
    if df is None or "Close" not in df.columns or len(df) < 5:
        return None
    closes = [float(c) for c in df["Close"].tolist() if c == c and c > 0]
    if len(closes) < 5:
        return None
    # Match the window length to the symbol's price_history when we can.
    days = min(len(closes), len(price_history_30d))
    etf_first = closes[-days]
    etf_last = closes[-1]
    if etf_first <= 0:
        return None
    etf_ret = ((etf_last - etf_first) / etf_first) * 100

    delta_pp = sym_ret - etf_ret
    if abs(delta_pp) < 1.0:
        label = "in line"
    elif delta_pp > 0:
        label = "outperforming"
    else:
        label = "underperforming"

    return SectorRelative(
        sector=sector,
        etf_symbol=etf,
        days=days,
        symbol_return_pct=round(sym_ret, 2),
        sector_return_pct=round(etf_ret, 2),
        delta_pp=round(delta_pp, 2),
        label=label,
    )


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
