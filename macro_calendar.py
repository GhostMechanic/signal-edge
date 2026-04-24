"""
macro_calendar.py — scheduled macro event dates + "days-to-next" helpers

Three calendars:
  • FOMC  — Fed rate-decision meeting dates (8/year, 2-day meetings;
            decision lands on Day 2 at ~2pm ET).
  • CPI   — BLS monthly Consumer Price Index release. Roughly mid-month.
  • NFP   — BLS Nonfarm Payrolls (the "jobs report"). Always the first
            Friday of the month, 8:30am ET.

Dates are hardcoded 2024–2026 (covers our walk-forward training windows
and the next ~24 months of production use). Re-check against official
sources annually — see `MACRO_CALENDAR_SOURCES` below.

Why hardcoded vs API:
  • These dates are public knowledge months in advance, don't change.
  • No API dependency = no failure mode during training or at inference.
  • NFP dates are algorithmically derivable (1st Friday) — included
    hardcoded for clarity + safety against DST edge cases.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional, Union, List

import pandas as pd


# ─── Data sources (for annual refresh) ────────────────────────────────
MACRO_CALENDAR_SOURCES = {
    "FOMC": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    "CPI":  "https://www.bls.gov/schedule/news_release/cpi.htm",
    "NFP":  "https://www.bls.gov/schedule/news_release/empsit.htm",
}


# ─── FOMC meeting decision dates ──────────────────────────────────────
# Official: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_DATES: List[date] = [
    # 2024
    date(2024, 1, 31),
    date(2024, 3, 20),
    date(2024, 5, 1),
    date(2024, 6, 12),
    date(2024, 7, 31),
    date(2024, 9, 18),
    date(2024, 11, 7),
    date(2024, 12, 18),
    # 2025
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 10),
    # 2026 (projected — will release official dates late 2025)
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
]


# ─── CPI release dates ────────────────────────────────────────────────
# Typically 2nd or 3rd Tuesday–Thursday of each month, ~8:30am ET.
# Official: https://www.bls.gov/schedule/news_release/cpi.htm
CPI_DATES: List[date] = [
    # 2024
    date(2024, 1, 11),
    date(2024, 2, 13),
    date(2024, 3, 12),
    date(2024, 4, 10),
    date(2024, 5, 15),
    date(2024, 6, 12),
    date(2024, 7, 11),
    date(2024, 8, 14),
    date(2024, 9, 11),
    date(2024, 10, 10),
    date(2024, 11, 13),
    date(2024, 12, 11),
    # 2025
    date(2025, 1, 15),
    date(2025, 2, 12),
    date(2025, 3, 12),
    date(2025, 4, 10),
    date(2025, 5, 13),
    date(2025, 6, 11),
    date(2025, 7, 15),
    date(2025, 8, 12),
    date(2025, 9, 11),
    date(2025, 10, 15),
    date(2025, 11, 13),
    date(2025, 12, 10),
    # 2026 (projected — mid-month pattern)
    date(2026, 1, 14),
    date(2026, 2, 11),
    date(2026, 3, 11),
    date(2026, 4, 15),
    date(2026, 5, 13),
    date(2026, 6, 10),
    date(2026, 7, 15),
    date(2026, 8, 12),
    date(2026, 9, 10),
    date(2026, 10, 14),
    date(2026, 11, 12),
    date(2026, 12, 10),
]


# ─── NFP release dates (first Friday of each month, 8:30am ET) ────────
# Always the first Friday; occasionally shifts if holiday. Hardcoded
# through 2026 for safety.
NFP_DATES: List[date] = [
    # 2024
    date(2024, 1, 5), date(2024, 2, 2),  date(2024, 3, 8),
    date(2024, 4, 5), date(2024, 5, 3),  date(2024, 6, 7),
    date(2024, 7, 5), date(2024, 8, 2),  date(2024, 9, 6),
    date(2024, 10, 4),date(2024, 11, 1), date(2024, 12, 6),
    # 2025
    date(2025, 1, 10), date(2025, 2, 7),  date(2025, 3, 7),
    date(2025, 4, 4),  date(2025, 5, 2),  date(2025, 6, 6),
    date(2025, 7, 3),  date(2025, 8, 1),  date(2025, 9, 5),
    date(2025, 10, 3), date(2025, 11, 7), date(2025, 12, 5),
    # 2026 (first Friday — subject to minor release-date shifts)
    date(2026, 1, 9),  date(2026, 2, 6),  date(2026, 3, 6),
    date(2026, 4, 3),  date(2026, 5, 1),  date(2026, 6, 5),
    date(2026, 7, 2),  date(2026, 8, 7),  date(2026, 9, 4),
    date(2026, 10, 2), date(2026, 11, 6), date(2026, 12, 4),
]


# ─── Core helpers ─────────────────────────────────────────────────────
def _to_date(x: Union[date, datetime, pd.Timestamp, str]) -> date:
    """Normalize various date-ish inputs to datetime.date."""
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime().date()
    if isinstance(x, str):
        return pd.Timestamp(x).to_pydatetime().date()
    raise TypeError(f"Cannot coerce {type(x)} to date")


def days_until_next(
    target_dates: List[date],
    as_of: Union[date, datetime, pd.Timestamp, str, None] = None,
    max_days: int = 365,
) -> Optional[int]:
    """
    Given a sorted list of event dates and a reference date, return the
    number of days until the NEXT event on or after the reference date.

    Returns None if:
      • No upcoming event within `max_days` (e.g. we're past the end of
        the hardcoded calendar).

    Returns 0 if the reference date IS an event date (the event is today).
    """
    as_of_d = _to_date(as_of) if as_of is not None else date.today()
    for ev in target_dates:
        delta = (ev - as_of_d).days
        if delta < 0:
            continue
        if delta > max_days:
            return None
        return delta
    return None


def days_to_next_fomc(as_of=None) -> Optional[int]:
    return days_until_next(FOMC_DATES, as_of=as_of)


def days_to_next_cpi(as_of=None) -> Optional[int]:
    return days_until_next(CPI_DATES, as_of=as_of)


def days_to_next_nfp(as_of=None) -> Optional[int]:
    return days_until_next(NFP_DATES, as_of=as_of)


def days_since_last(target_dates: List[date],
                    as_of: Union[date, datetime, pd.Timestamp, str, None] = None,
                    max_days: int = 365) -> Optional[int]:
    """Days since the most recent event on or before as_of."""
    as_of_d = _to_date(as_of) if as_of is not None else date.today()
    for ev in reversed(target_dates):
        delta = (as_of_d - ev).days
        if delta < 0:
            continue
        if delta > max_days:
            return None
        return delta
    return None


# ─── Time-series helper for training ───────────────────────────────────
def event_distance_series(
    index: pd.DatetimeIndex,
    target_dates: List[date],
    normalize_by: int = 30,
) -> pd.Series:
    """
    Build a time series (one value per date in `index`) of "normalized days
    until the next event" — i.e. days_to_next / normalize_by, clipped to
    [0, 1]. We use a normalized form so the model sees event proximity on
    a comparable scale to other [0,1]-ish features.

    A value near 0 → event is imminent. A value of 1 → event is far away
    (or no event in window).
    """
    vals = []
    for ts in index:
        d = days_until_next(target_dates, as_of=ts, max_days=normalize_by)
        if d is None:
            vals.append(1.0)
        else:
            vals.append(min(d / normalize_by, 1.0))
    return pd.Series(vals, index=index, dtype=float)


def event_proximity_flag(
    index: pd.DatetimeIndex,
    target_dates: List[date],
    window_days: int = 5,
) -> pd.Series:
    """Boolean-as-float series: 1.0 if an event is within `window_days`
    ahead, else 0.0. Useful for capturing "event-week" regime effects."""
    target_set = {d for d in target_dates}
    vals = []
    for ts in index:
        d_today = _to_date(ts)
        window = (d_today + timedelta(days=k) for k in range(0, window_days + 1))
        vals.append(1.0 if any(w in target_set for w in window) else 0.0)
    return pd.Series(vals, index=index, dtype=float)


# ─── Convenience: one-shot catalyst snapshot for UI ──────────────────
def upcoming_catalysts(
    as_of=None,
    within_days: int = 45,
    include_past_days: int = 0,
) -> list[dict]:
    """
    Return a sorted list of macro catalysts within `within_days` of as_of.
    Each entry is: {kind, date, days_out, label}.
    Used by the Analyze page's Catalysts panel.
    """
    as_of_d = _to_date(as_of) if as_of is not None else date.today()
    lower = as_of_d - timedelta(days=include_past_days)
    upper = as_of_d + timedelta(days=within_days)

    events = []
    for kind, lst, label in [
        ("FOMC", FOMC_DATES, "Fed rate decision"),
        ("CPI",  CPI_DATES,  "CPI release"),
        ("NFP",  NFP_DATES,  "Jobs report"),
    ]:
        for d in lst:
            if lower <= d <= upper:
                events.append({
                    "kind": kind,
                    "date": d.isoformat(),
                    "days_out": (d - as_of_d).days,
                    "label": label,
                })
    events.sort(key=lambda e: e["date"])
    return events
