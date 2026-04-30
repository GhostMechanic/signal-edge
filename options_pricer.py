"""
options_pricer.py
-----------------
Live mid-price valuation for an open paper-options position.

The system stores option positions as multi-leg strategies in
`paper_trades.instrument_data` with leg shape
`{action: BUY|SELL, type: CALL|PUT, strike: float}` and a single
position-level `expiry_date`. To close such a position before expiry the
user needs the current liquidation value — which means pricing every
leg at the live market and netting them by sign.

Source priority per leg (best to worst):

  mid        — (bid + ask) / 2 when both sides are present and ask >= bid
  last       — lastPrice when bid/ask is unusable but last > 0
  intrinsic  — max(0, S − K) for calls / max(0, K − S) for puts. ONLY
               used when the chain has the strike but the quote is
               all-zero (deep OTM with pennies of time value). It's
               defensible there because the option really is worth ~zero.
               NOT used when the chain itself is unreachable — see below.
  unavailable — the chain fetch failed (after the snap fallback), or the
                strike isn't on the chain. Refusing to price beats
                inventing a number.

Snap-to-chain. yfinance only returns chains for the specific expiry dates
it knows about (mostly third-Friday LEAPS plus weeklies). The asking
flow stores `expiry_date = today + N days` which often falls between two
real chain dates. If we can't get an exact match, we snap to the nearest
expiry within a tolerance (default 7 days for the pricer; larger at
trade-open). Trades opened after the snap-at-open landed always have
real chain dates so this only matters for legacy trades.

Position-level source = worst of any leg (mid > last > intrinsic >
unavailable). Any leg `unavailable` → whole position unavailable; we
never partially fill.

This module is the single source of option-pricing truth: the early-close
endpoint and the daily user EOD MTM both call into here so the two
surfaces never disagree.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# yfinance is imported lazily inside the functions that actually hit the
# network (_fetch_spot, _list_chain_expiries, etc.). Loading it at module
# import would force every consumer — including the pure-string helper
# `_format_display_strings` and unit tests — to depend on it. Kept as a
# typing reference only; do NOT call yf.* without re-importing locally.
try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


_SOURCE_RANK = {
    "mid":         3,
    "last":        2,
    "intrinsic":   1,
    "unavailable": 0,
}

# Tolerance for the pricer's snap fallback. Beyond this delta we don't
# trust that the closest chain expiry meaningfully matches the recorded
# one — the option's time value would diverge too much. The trade-open
# path snaps without tolerance because it picks the nearest at request
# time and records the real date.
PRICER_SNAP_TOLERANCE_DAYS = 7


def _worst_source(a: str, b: str) -> str:
    return a if _SOURCE_RANK.get(a, 0) <= _SOURCE_RANK.get(b, 0) else b


def _is_finite_pos(x: Any) -> bool:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f > 0


def _safe_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _fetch_spot(symbol: str) -> Optional[float]:
    """Latest close for the underlying. Used for intrinsic fallback."""
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(period="5d", auto_adjust=False)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        last = df["Close"].dropna()
        if last.empty:
            return None
        return float(last.iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("options_pricer: spot fetch failed for %s: %s", symbol, exc)
        return None


def _list_chain_expiries(symbol: str) -> Optional[List[str]]:
    """Return yfinance's available expiry dates for the symbol, or None
    if the listing itself fails. Empty list is a valid "no chains
    available" signal — caller distinguishes."""
    try:
        tk = yf.Ticker(symbol)
        opts = tk.options
        if opts is None:
            return []
        return list(opts)
    except Exception as exc:  # noqa: BLE001
        logger.warning("options_pricer: tk.options failed for %s: %s", symbol, exc)
        return None


def snap_expiry_to_chain(
    symbol: str,
    target_expiry_iso: str,
    *,
    tolerance_days: Optional[int] = None,
) -> Optional[str]:
    """
    Find the closest available chain expiry to `target_expiry_iso`.

    Returns:
      - target_expiry_iso if it's already on the chain
      - the closest available expiry within tolerance_days
      - None when no expiries are available, or when the closest is
        beyond tolerance_days

    `tolerance_days=None` means "no tolerance — always pick the
    closest." Used by the trade-open path. The pricer passes
    PRICER_SNAP_TOLERANCE_DAYS so it doesn't price legacy garbage.

    Exposed at module level (and not underscore-prefixed) so the open
    path and backfill can reuse it.
    """
    available = _list_chain_expiries(symbol)
    if not available:
        return None
    if target_expiry_iso in available:
        return target_expiry_iso

    try:
        target = datetime.strptime(target_expiry_iso, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None

    best: Optional[str] = None
    best_delta: Optional[int] = None
    for s in available:
        try:
            d = datetime.strptime(s, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        delta = abs((d - target).days)
        if best_delta is None or delta < best_delta:
            best = s
            best_delta = delta

    if best is None:
        return None
    if tolerance_days is not None and best_delta is not None and best_delta > tolerance_days:
        return None
    return best


def list_chain_strikes(
    symbol: str,
    expiry_iso: str,
    leg_type: str,
) -> List[float]:
    """
    Return the sorted list of strikes available on the live chain for
    (symbol, expiry, type). Used by snap_strike_to_chain.

    Empty list when the chain itself is unreachable or has no rows for
    that type.
    """
    try:
        tk = yf.Ticker(symbol)
        chain = tk.option_chain(expiry_iso)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "list_chain_strikes: %s @ %s failed: %s", symbol, expiry_iso, exc,
        )
        return []
    df = chain.calls if (leg_type or "").upper() == "CALL" else chain.puts
    if df is None or df.empty or "strike" not in df.columns:
        return []
    try:
        return sorted(float(s) for s in df["strike"].tolist() if s == s)  # filter NaN
    except Exception:  # noqa: BLE001
        return []


# Default strike-snap tolerance. 10% of the target strike. Catches the
# typical $5/$10 strike intervals real chains use even on high-priced
# names (AMZN at $260 has $5 strikes, so the worst-case snap distance
# is $2.50 — well under 10%). Beyond 10% we refuse rather than fill
# at a strike that's meaningfully different from what was quoted.
STRIKE_SNAP_TOLERANCE_PCT = 0.10


def snap_strike_to_chain(
    symbol: str,
    expiry_iso: str,
    leg_type: str,
    target_strike: float,
    *,
    tolerance_pct: float = STRIKE_SNAP_TOLERANCE_PCT,
    available_strikes: Optional[List[float]] = None,
) -> Optional[float]:
    """
    Find the closest strike on the live chain to `target_strike`.

    Returns:
      - target_strike if it's exactly on the chain
      - the closest available strike when within tolerance_pct
      - None when no chain available, or closest is outside tolerance

    `available_strikes` lets the caller pass a pre-fetched strike list
    to avoid re-fetching the chain when snapping multiple legs against
    the same expiry.
    """
    target = _safe_float(target_strike)
    if target is None or target <= 0:
        return None

    strikes = (
        available_strikes
        if available_strikes is not None
        else list_chain_strikes(symbol, expiry_iso, leg_type)
    )
    if not strikes:
        return None

    closest = min(strikes, key=lambda k: abs(k - target))
    delta = abs(closest - target)
    if tolerance_pct is not None and delta > target * tolerance_pct:
        return None
    return closest


def _fetch_leg_quote(
    symbol: str,
    target_expiry_iso: str,
    strike: float,
    leg_type: str,
    chain_cache: Dict[str, Any],
    snap_cache: Dict[str, Optional[str]],
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    """
    Resolve a single leg quote against the live chain.

    Returns (quote_or_None, status, chain_expiry):
      - status="ok"                 → quote_or_None has the row's bid/ask/last,
                                       any of which may be None or zero.
                                       chain_expiry is the date the chain came from.
      - status="chain_unreachable"  → quote=None, chain_expiry=None
      - status="strike_not_found"   → quote=None, chain_expiry=<reachable date>

    Snap behavior: tries the exact target first. If unreachable, snaps
    to the nearest available expiry within PRICER_SNAP_TOLERANCE_DAYS
    and tries that. If still unreachable, returns chain_unreachable.
    """
    # Resolve which expiry we'll actually price against. Cache so legs
    # sharing the same target only snap once.
    if target_expiry_iso in snap_cache:
        chain_expiry = snap_cache[target_expiry_iso]
    else:
        chain_expiry = snap_expiry_to_chain(
            symbol, target_expiry_iso,
            tolerance_days=PRICER_SNAP_TOLERANCE_DAYS,
        )
        snap_cache[target_expiry_iso] = chain_expiry

    if chain_expiry is None:
        return None, "chain_unreachable", None

    # Cache the chain object per resolved expiry — many legs share it.
    chain = chain_cache.get(chain_expiry)
    if chain is None:
        try:
            tk = yf.Ticker(symbol)
            chain = tk.option_chain(chain_expiry)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "options_pricer: chain fetch failed for %s @ %s: %s",
                symbol, chain_expiry, exc,
            )
            chain_cache[chain_expiry] = False
            return None, "chain_unreachable", chain_expiry
        chain_cache[chain_expiry] = chain
    elif chain is False:
        return None, "chain_unreachable", chain_expiry

    df = chain.calls if leg_type.upper() == "CALL" else chain.puts
    if df is None or df.empty:
        return None, "strike_not_found", chain_expiry

    try:
        match = df[(df["strike"] - float(strike)).abs() < 0.01]
    except Exception:  # noqa: BLE001
        return None, "strike_not_found", chain_expiry
    if match.empty:
        return None, "strike_not_found", chain_expiry

    row = match.iloc[0].to_dict()
    return (
        {
            "bid":  _safe_float(row.get("bid")),
            "ask":  _safe_float(row.get("ask")),
            "last": _safe_float(row.get("lastPrice")),
            # Implied vol from the chain. Used downstream to compute
            # display-only Black-Scholes greeks (delta, theta). Trade
            # math doesn't read this — see methodology § 4.5.
            "iv":   _safe_float(row.get("impliedVolatility")),
        },
        "ok",
        chain_expiry,
    )


def _price_leg(
    leg: Dict[str, Any],
    quote: Optional[Dict[str, Any]],
    quote_status: str,
    spot: Optional[float],
    *,
    days_to_expiry: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Resolve a single leg's price-per-share + source given the quote-fetch
    result. The status governs which fallbacks are allowed:

      ok                 → mid → last → intrinsic-at-spot
      strike_not_found   → unavailable (don't synthesize)
      chain_unreachable  → unavailable (don't synthesize)

    intrinsic-at-spot is only used when the chain confirmed the strike
    but the quote feed had nothing for it — that's a legitimate
    "deep OTM with pennies of time value" case. Synthesizing a price
    for unreachable chains would let the system fill at fictional
    numbers, which the integrity contract forbids (see methodology § 1).
    """
    strike = _safe_float(leg.get("strike")) or 0.0
    leg_type = (leg.get("type") or "").upper()
    action = (leg.get("action") or "").upper()

    out: Dict[str, Any] = {
        "strike":     strike,
        "type":       leg_type,
        "action":     action,
        "bid":        None,
        "ask":        None,
        "last":       None,
        "iv":         None,
        # Display-only greeks. The methodology (§ 4.5) is explicit:
        # trade math reads bid/ask/last, not these. Surfaced for
        # context in the UI's expansion panel.
        "delta":      None,
        "theta":      None,
        "used_price": None,
        "source":     "unavailable",
    }

    if quote_status != "ok" or quote is None:
        return out

    out["bid"]  = quote.get("bid")
    out["ask"]  = quote.get("ask")
    out["last"] = quote.get("last")

    # Greeks: display-only. Computed from chain-reported IV when
    # available; if we can't sanitize the IV we leave them at None
    # and the UI shows dashes rather than fabricated numbers.
    try:
        from options_greeks import bs_delta, bs_theta, bs_iv_from_chain
        clean_iv = bs_iv_from_chain(quote.get("iv"))
        out["iv"] = clean_iv
        if clean_iv is not None and spot is not None and days_to_expiry is not None:
            out["delta"] = bs_delta(spot, strike, clean_iv, days_to_expiry, leg_type)
            out["theta"] = bs_theta(spot, strike, clean_iv, days_to_expiry, leg_type)
    except Exception as exc:  # noqa: BLE001
        logger.debug("greeks compute failed for %s %s: %s", leg_type, strike, exc)
        out["iv"] = None
        out["delta"] = None
        out["theta"] = None

    bid = quote.get("bid")
    ask = quote.get("ask")
    # Prefer mid when both sides are present and the spread isn't inverted.
    if _is_finite_pos(bid) and _is_finite_pos(ask) and ask >= bid:
        out["used_price"] = round((bid + ask) / 2.0, 4)
        out["source"]     = "mid"
        return out

    last = quote.get("last")
    if _is_finite_pos(last):
        out["used_price"] = round(float(last), 4)
        out["source"]     = "last"
        return out

    # All-zero quote on a strike the chain confirmed exists. The option
    # really is worth ~zero plus pennies of time value we can't see.
    # Intrinsic at spot is the defensible value.
    if spot is not None and leg_type in ("CALL", "PUT"):
        if leg_type == "CALL":
            intrinsic = max(0.0, spot - strike)
        else:
            intrinsic = max(0.0, strike - spot)
        out["used_price"] = round(intrinsic, 4)
        out["source"]     = "intrinsic"
        return out

    return out


def value_option_position(
    trade_row: Dict[str, Any],
    *,
    spot_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Value an open option paper trade at current live prices.

    `trade_row` is a paper_trades row with kind='option'. Must carry
    `symbol`, `qty`, and `instrument_data` (with `legs`, `expiry_date`,
    `premium_per_contract`, `cash_locked`).

    `spot_override` lets the EOD pass pass an already-fetched close to
    avoid re-hitting yfinance for every option on the same symbol.

    Returns:
      {
        ok:               bool,
        source:           "mid"|"last"|"intrinsic"|"unavailable",
        liq_per_contract: float,   # current liquidation value, $/contract (signed)
        position_value:   float,   # liq_per_contract * qty
        pnl_per_contract: float,   # liq − cost, $/contract
        pnl_total:        float,
        cash_back:        float,   # cash_locked + pnl_total (what the user nets on close)
        per_leg:          [ {..., chain_expiry, status} ],
        chain_expiry:     str | None,  # actual expiry the chain came from (post-snap)
        recorded_expiry:  str | None,  # what the trade row said
        fetched_at:       ISO timestamp,
        error:            str | None,
      }

    `ok=False` means the caller MUST refuse the close (or fall back to
    cost basis for MTM) — it never silently fills at a fictional price.

    When `chain_expiry != recorded_expiry` we priced against a snapped
    chain. Surface that delta in the receipt so the user can see the
    fill came from a chain a few days off the recorded date.
    """
    symbol = (trade_row.get("symbol") or "").upper().strip()
    ins = trade_row.get("instrument_data") or {}
    legs: List[Dict[str, Any]] = list(ins.get("legs") or [])
    expiry_date = ins.get("expiry_date")
    qty = int(float(trade_row.get("qty") or 0))
    cost_per = float(ins.get("premium_per_contract") or 0)
    cash_locked = float(ins.get("cash_locked") or 0)

    fetched_at = datetime.now(timezone.utc).isoformat()
    base = {
        "ok":               False,
        "source":           "unavailable",
        "liq_per_contract": 0.0,
        "position_value":   0.0,
        "pnl_per_contract": 0.0,
        "pnl_total":        0.0,
        "cash_back":        0.0,
        "per_leg":          [],
        "chain_expiry":     None,
        "recorded_expiry":  expiry_date,
        "underlying_spot":  None,    # surfaced for the UI's distance metrics
        "fetched_at":       fetched_at,
        "error":            None,
    }

    if not symbol or not legs or not expiry_date or qty <= 0:
        base["error"] = "trade row missing symbol / legs / expiry_date / qty"
        return base

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if expiry_date <= today:
        # Expired — settlement path owns this, not the live pricer.
        base["error"] = "position is expired; use the expiry settlement path"
        return base

    spot = spot_override if spot_override is not None else _fetch_spot(symbol)
    base["underlying_spot"] = spot

    # Days-to-expiry used by the greeks helper. Computed once per call.
    try:
        from datetime import date as _date_cls
        target = datetime.strptime(expiry_date, "%Y-%m-%d").date()
        days_to_expiry = max(0, (target - datetime.now(timezone.utc).date()).days)
    except (TypeError, ValueError):
        days_to_expiry = 0

    chain_cache: Dict[str, Any] = {}
    snap_cache: Dict[str, Optional[str]] = {}
    per_leg_out: List[Dict[str, Any]] = []
    liq_price_per_contract = 0.0  # $/share — × 100 to get $/contract
    worst = "mid"
    chain_expiry_used: Optional[str] = None

    for leg in legs:
        leg_type = (leg.get("type") or "").upper()
        if leg_type not in ("CALL", "PUT"):
            continue
        quote, status, leg_chain_expiry = _fetch_leg_quote(
            symbol, expiry_date,
            _safe_float(leg.get("strike")) or 0.0,
            leg_type, chain_cache, snap_cache,
        )
        priced = _price_leg(leg, quote, status, spot, days_to_expiry=days_to_expiry)
        priced["chain_expiry"] = leg_chain_expiry
        priced["status"]       = status
        per_leg_out.append(priced)

        if leg_chain_expiry and chain_expiry_used is None:
            chain_expiry_used = leg_chain_expiry

        worst = _worst_source(worst, priced["source"])
        if priced["source"] == "unavailable" or priced["used_price"] is None:
            base["per_leg"]    = per_leg_out
            base["source"]     = "unavailable"
            base["chain_expiry"] = chain_expiry_used
            base["error"]      = _format_unavailable_reason(
                symbol, leg, expiry_date, status,
            )
            return base

        sign = +1.0 if (leg.get("action") or "").upper() == "BUY" else -1.0
        liq_price_per_contract += sign * float(priced["used_price"])

    liq_per_contract = round(liq_price_per_contract * 100.0, 2)
    position_value   = round(liq_per_contract * qty, 2)
    pnl_per_contract = round(liq_per_contract - cost_per, 2)
    pnl_total        = round(pnl_per_contract * qty, 2)
    cash_back        = round(cash_locked + pnl_total, 2)

    return {
        "ok":               True,
        "source":           worst,
        "liq_per_contract": liq_per_contract,
        "position_value":   position_value,
        "pnl_per_contract": pnl_per_contract,
        "pnl_total":        pnl_total,
        "cash_back":        cash_back,
        "per_leg":          per_leg_out,
        "chain_expiry":     chain_expiry_used,
        "recorded_expiry":  expiry_date,
        "underlying_spot":  spot,
        "days_to_expiry":   days_to_expiry,
        "fetched_at":       fetched_at,
        "error":            None,
    }


def price_legs_at_open(
    symbol: str,
    expiry_iso: str,
    legs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Price every leg of a strategy at trade-open time. Used by the open
    path to lock in real entry premiums instead of trusting the
    asking-flow's Black-Scholes estimate.

    Stricter than `value_option_position`: refuses if any leg's source
    isn't `mid` or `last`. Intrinsic-at-spot is fine for valuing an
    already-open position whose strike has zero quotes — but at OPEN
    time we'd be locking the user into a fictional cost basis. Same
    reasoning for `unavailable`. The honest move is to refuse the open
    and tell the user why.

    The caller (db.open_user_paper_option_trade) is expected to pass
    `expiry_iso` already snapped to a real chain date — no further
    snapping is needed here.

    Returns:
      {
        "ok":           bool,
        "chain_expiry": str | None,   # the chain the legs priced against
        "per_leg":      [ {strike, type, action, bid, ask, last,
                           used_price, source, status, chain_expiry} ],
        "error":        str | None,
      }
    """
    out: Dict[str, Any] = {
        "ok":           False,
        "chain_expiry": None,
        "per_leg":      [],
        "error":        None,
    }

    if not symbol or not expiry_iso or not legs:
        out["error"] = "missing symbol / expiry / legs"
        return out

    spot = _fetch_spot(symbol)
    chain_cache: Dict[str, Any] = {}
    snap_cache: Dict[str, Optional[str]] = {}
    chain_expiry_used: Optional[str] = None
    per_leg_out: List[Dict[str, Any]] = []
    # Cache strike lists per (expiry, type) so we don't re-fetch the
    # chain for each leg's snap. Most strategies share a single expiry
    # and may share types (e.g., bull call spread = two CALL legs).
    strikes_cache: Dict[Tuple[str, str], List[float]] = {}

    try:
        target = datetime.strptime(expiry_iso, "%Y-%m-%d").date()
        open_dte = max(0, (target - datetime.now(timezone.utc).date()).days)
    except (TypeError, ValueError):
        open_dte = None

    for leg in legs:
        leg_type = (leg.get("type") or "").upper()
        if leg_type not in ("CALL", "PUT"):
            out["per_leg"] = per_leg_out
            out["error"]   = f"unsupported leg type: {leg.get('type')!r}"
            return out

        # Snap the leg's strike to the nearest real strike on the chain
        # before pricing. The asking flow's `_strike_choices` produces
        # theoretical strikes (spot ± half-sigma, etc.) that often
        # don't exist on real chains — see methodology § 4.5. Snapping
        # gives the caller a fillable price; the leg's `original_strike`
        # field preserves the asked-for value for receipt drift.
        target_strike = _safe_float(leg.get("strike")) or 0.0
        cache_key = (expiry_iso, leg_type)
        if cache_key not in strikes_cache:
            strikes_cache[cache_key] = list_chain_strikes(
                symbol, expiry_iso, leg_type,
            )
        snapped_strike = snap_strike_to_chain(
            symbol, expiry_iso, leg_type, target_strike,
            available_strikes=strikes_cache[cache_key],
        )
        if snapped_strike is None:
            # Either no chain at this expiry, or no strike close enough
            # to the target. The pricer refuses; caller raises.
            out["per_leg"]      = per_leg_out
            out["chain_expiry"] = chain_expiry_used
            out["error"]        = (
                f"strike not on the live chain (within "
                f"{int(STRIKE_SNAP_TOLERANCE_PCT*100)}%): "
                f"{symbol} {leg.get('action')} {leg_type} ${target_strike:.2f} "
                f"@ {expiry_iso}"
            )
            return out

        quote, status, leg_chain_expiry = _fetch_leg_quote(
            symbol, expiry_iso, snapped_strike,
            leg_type, chain_cache, snap_cache,
        )
        # Build a leg dict with the snapped strike so _price_leg's
        # output (and any downstream consumers) see the real value.
        leg_for_pricing = {**leg, "strike": snapped_strike}
        priced = _price_leg(leg_for_pricing, quote, status, spot, days_to_expiry=open_dte)
        priced["chain_expiry"]   = leg_chain_expiry
        priced["status"]         = status
        priced["original_strike"] = (
            target_strike if abs(snapped_strike - target_strike) > 0.005 else None
        )
        per_leg_out.append(priced)

        if leg_chain_expiry and chain_expiry_used is None:
            chain_expiry_used = leg_chain_expiry

        # Open path is strict: only mid or last is fillable. Intrinsic
        # at spot for a 365d call would lock in $0 cost basis on what
        # might be a $40-premium option. Unavailable means we never had
        # a quote at all. Either way, refuse.
        if priced["source"] not in ("mid", "last"):
            out["per_leg"]      = per_leg_out
            out["chain_expiry"] = chain_expiry_used
            out["error"]        = _format_unfillable_reason(
                symbol, leg, expiry_iso, priced["source"], status,
            )
            return out

    return {
        "ok":           True,
        "chain_expiry": chain_expiry_used,
        "per_leg":      per_leg_out,
        "error":        None,
    }


def snap_strategy_to_real_chain(
    strategy: Dict[str, Any],
    symbol: str,
) -> Dict[str, Any]:
    """
    Reshape a freshly-built theoretical strategy into a chain-anchored
    one. Used by the asking flow (options_analyzer.generate_options_report)
    so that what the user sees on /prediqt is exactly what they'll fill
    at when they take the play. No drift between quote and fill.

    Pipeline:
      1. Snap expiry to nearest available chain date.
      2. For each leg, snap strike to nearest available strike on that
         chain (within STRIKE_SNAP_TOLERANCE_PCT).
      3. Fetch live mids for every leg via price_legs_at_open.
      4. Replace each leg's Black-Scholes premium with the live
         used_price; preserve original_strike + estimated_premium.
      5. Recompute strategy numerics from the live legs via
         options_analyzer.recompute_strategy_numerics.
      6. Stamp expiry_date / chain_expiry / expiry_days_actual on the
         strategy and return.

    If any step fails (no chain available, strikes outside tolerance,
    legs can't fill at mid/last), returns the original strategy with
    `numeric.tradable = False` and an `untradeable_reason` field. The
    open path's tradable check will then refuse the trade with a
    clear error rather than persist a bad strategy.
    """
    if not symbol or not isinstance(strategy, dict):
        return strategy

    legs = strategy.get("legs") or []
    expiry_days = strategy.get("expiry_days")
    strategy_type = strategy.get("strategy")
    if not legs or not expiry_days or not strategy_type:
        return strategy

    requested_expiry_dt = datetime.now(timezone.utc) + timedelta(days=int(expiry_days))
    requested_expiry = requested_expiry_dt.strftime("%Y-%m-%d")

    snapped_expiry = snap_expiry_to_chain(symbol, requested_expiry)
    if snapped_expiry is None:
        return _mark_untradeable(
            strategy, f"no chain available for {symbol} near {requested_expiry}",
        )

    # Pre-snap each leg's strike against the snapped expiry's chain.
    # We do this here (not just inside price_legs_at_open) so the
    # caller can see strike snaps happened for receipt drift even
    # before the open path runs.
    strikes_cache: Dict[Tuple[str, str], List[float]] = {}
    snapped_legs: List[Dict[str, Any]] = []
    for leg in legs:
        leg_type = (leg.get("type") or "").upper()
        if leg_type not in ("CALL", "PUT"):
            return _mark_untradeable(
                strategy, f"unsupported leg type: {leg.get('type')!r}",
            )
        target_strike = _safe_float(leg.get("strike")) or 0.0
        cache_key = (snapped_expiry, leg_type)
        if cache_key not in strikes_cache:
            strikes_cache[cache_key] = list_chain_strikes(
                symbol, snapped_expiry, leg_type,
            )
        snapped = snap_strike_to_chain(
            symbol, snapped_expiry, leg_type, target_strike,
            available_strikes=strikes_cache[cache_key],
        )
        if snapped is None:
            return _mark_untradeable(
                strategy,
                f"strike not on chain: {leg.get('action')} {leg_type} "
                f"${target_strike:.2f} @ {snapped_expiry} "
                f"(within {int(STRIKE_SNAP_TOLERANCE_PCT*100)}%)",
            )
        snapped_legs.append({
            **leg,
            "strike": snapped,
            "original_strike": (
                target_strike if abs(snapped - target_strike) > 0.005 else None
            ),
        })

    # Live-price the snapped legs.
    priced = price_legs_at_open(symbol, snapped_expiry, snapped_legs)
    if not priced.get("ok"):
        return _mark_untradeable(
            strategy, f"live pricing refused: {priced.get('error') or 'unknown'}",
        )

    # Build the live legs with chain-real strike + chain-real premium.
    live_legs: List[Dict[str, Any]] = []
    for orig, p in zip(snapped_legs, priced.get("per_leg") or []):
        live_legs.append({
            **orig,
            "estimated_premium": orig.get("premium"),
            "premium":           p.get("used_price"),
            "live_quote": {
                "bid":            p.get("bid"),
                "ask":            p.get("ask"),
                "last":           p.get("last"),
                "used_price":     p.get("used_price"),
                "source":         p.get("source"),
                "chain_expiry":   p.get("chain_expiry"),
                "snapped_strike": orig.get("strike"),
                "original_strike": orig.get("original_strike"),
            },
        })

    # Recompute the numeric block from the live values.
    try:
        from options_analyzer import recompute_strategy_numerics
        live_numeric = recompute_strategy_numerics(strategy_type, live_legs)
    except Exception as exc:  # noqa: BLE001
        return _mark_untradeable(
            strategy, f"numerics recompute failed: {exc}",
        )

    # expiry_days_actual reflects the post-snap calendar distance —
    # may differ from the requested expiry_days by a few days.
    try:
        snapped_dt = datetime.strptime(snapped_expiry, "%Y-%m-%d").date()
        days_actual = max(0, (snapped_dt - datetime.now(timezone.utc).date()).days)
    except (TypeError, ValueError):
        days_actual = int(expiry_days)

    # Refresh the legacy display-string fields from the live numeric block.
    # These strings (estimated_cost, max_profit, max_loss, breakeven, iv_used)
    # were written by options_analyzer's per-strategy constructor against the
    # ORIGINAL theoretical legs (pre-snap strikes, Black-Scholes premiums).
    # If we don't refresh them here, the `**strategy` spread below carries
    # them through unchanged — and any frontend reading the strings instead
    # of the numeric block (as prediqt-web does) shows pre-snap BS values
    # while the rest of the card shows post-snap live mids. That was the
    # MSFT 410/380 bear-put-spread bug found on the public ledger Apr 30.
    refreshed_strings = _format_display_strings(strategy_type, live_legs, live_numeric)

    return {
        **strategy,
        **refreshed_strings,                          # estimated_cost, max_profit, etc.
        "legs":                live_legs,
        "numeric":             live_numeric,
        "expiry_date":         snapped_expiry,        # real chain date
        "original_expiry_date": requested_expiry,     # what was asked
        "expiry_days_actual":  days_actual,
        "chain_expiry":        priced.get("chain_expiry") or snapped_expiry,
    }


def _format_display_strings(
    strategy_type: str,
    legs: List[Dict[str, Any]],
    numeric: Dict[str, Any],
) -> Dict[str, str]:
    """Build the legacy estimated_cost / max_profit / max_loss / breakeven
    display strings from a (live) numeric block. Mirrors the shape each
    constructor in options_analyzer writes — kept here so snap can refresh
    them without round-tripping through Black-Scholes.

    Strategy types this knows how to refresh: Long Call, Long Put,
    Bull Call Spread, Bear Put Spread, Iron Condor, Long Straddle,
    Cash-Secured Put. Unknown types get an empty dict (no override),
    so the original strings carry through unchanged.
    """
    cost       = numeric.get("cost_per_contract")
    max_profit = numeric.get("max_profit_per_contract")
    max_loss   = numeric.get("max_loss_per_contract")
    breakevens = numeric.get("breakevens") or []

    def _fmt_dollars(x: Any) -> str:
        try:
            return f"${float(x):.2f}"
        except (TypeError, ValueError):
            return "—"

    def _first_be() -> str:
        if not breakevens:
            return "—"
        return _fmt_dollars(breakevens[0])

    st = (strategy_type or "").strip()

    if st == "Long Call":
        return {
            "estimated_cost": f"{_fmt_dollars(cost)} / contract",
            "max_profit":     "Unlimited (stock price - breakeven)",
            "max_loss":       f"{_fmt_dollars(max_loss)} per contract (premium paid)",
            "breakeven":      _first_be(),
        }

    if st == "Long Put":
        return {
            "estimated_cost": f"{_fmt_dollars(cost)} / contract",
            "max_profit":     f"{_fmt_dollars(max_profit)} / contract (stock → $0)",
            "max_loss":       f"{_fmt_dollars(max_loss)} / contract (premium paid)",
            "breakeven":      _first_be(),
        }

    if st == "Bull Call Spread":
        # Find the SELL CALL strike for the "(at $X+)" tail.
        sell = next(
            (l for l in legs if (l.get("action") or "").upper() == "SELL"
             and (l.get("type") or "").upper() == "CALL"),
            None,
        )
        K_sell = float(sell.get("strike")) if sell else None
        max_profit_str = f"{_fmt_dollars(max_profit)} / contract"
        if K_sell is not None:
            max_profit_str += f" (at ${K_sell:.2f}+)"
        return {
            "estimated_cost": f"{_fmt_dollars(cost)} net debit / contract",
            "max_profit":     max_profit_str,
            "max_loss":       f"{_fmt_dollars(max_loss)} / contract (net premium paid)",
            "breakeven":      _first_be(),
        }

    if st == "Bear Put Spread":
        return {
            "estimated_cost": f"{_fmt_dollars(cost)} net debit / contract",
            "max_profit":     f"{_fmt_dollars(max_profit)} / contract",
            "max_loss":       f"{_fmt_dollars(max_loss)} / contract",
            "breakeven":      _first_be(),
        }

    if st == "Iron Condor":
        # Find the inner short strikes for the "(stock stays between)" tail.
        sell_put = next(
            (l for l in legs if (l.get("action") or "").upper() == "SELL"
             and (l.get("type") or "").upper() == "PUT"),
            None,
        )
        sell_call = next(
            (l for l in legs if (l.get("action") or "").upper() == "SELL"
             and (l.get("type") or "").upper() == "CALL"),
            None,
        )
        net_credit = numeric.get("max_profit_per_contract")  # is_credit ⇒ credit
        max_profit_str = f"{_fmt_dollars(net_credit)} / contract"
        if sell_put is not None and sell_call is not None:
            try:
                Kp = float(sell_put["strike"])
                Kc = float(sell_call["strike"])
                max_profit_str += f" (stock stays between ${Kp:g}–${Kc:g})"
            except (TypeError, ValueError, KeyError):
                pass
        # Two breakevens: be_down (down) / be_up (up)
        if len(breakevens) >= 2:
            be_str = (
                f"{_fmt_dollars(breakevens[0])} (down) / "
                f"{_fmt_dollars(breakevens[1])} (up)"
            )
        else:
            be_str = _first_be()
        return {
            "estimated_cost": f"{_fmt_dollars(net_credit)} net credit / contract",
            "max_profit":     max_profit_str,
            "max_loss":       f"{_fmt_dollars(max_loss)} / contract (stock breaks outside wings)",
            "breakeven":      be_str,
        }

    if st == "Long Straddle":
        be_str = (
            f"{_fmt_dollars(breakevens[0])} (down) / "
            f"{_fmt_dollars(breakevens[1])} (up)"
            if len(breakevens) >= 2 else _first_be()
        )
        return {
            "estimated_cost": f"{_fmt_dollars(cost)} / contract",
            "max_profit":     "Unlimited (large move in either direction)",
            "max_loss":       f"{_fmt_dollars(max_loss)} / contract (no movement at expiry)",
            "breakeven":      be_str,
        }

    if st == "Cash-Secured Put":
        # cost_per_contract on credit strategies stores -credit by convention.
        try:
            credit_dollars = -float(cost) if cost is not None else None
        except (TypeError, ValueError):
            credit_dollars = None
        # Find SELL PUT strike for the "(keep full premium if stock > $K)" tail.
        sell = next(
            (l for l in legs if (l.get("action") or "").upper() == "SELL"
             and (l.get("type") or "").upper() == "PUT"),
            None,
        )
        K = float(sell.get("strike")) if sell else None
        max_profit_str = f"{_fmt_dollars(credit_dollars)} / contract"
        if K is not None:
            max_profit_str += f" (keep full premium if stock > ${K:g})"
        return {
            "estimated_cost": f"{_fmt_dollars(credit_dollars)} net credit / contract",
            "max_profit":     max_profit_str,
            "max_loss":       f"{_fmt_dollars(max_loss)} / contract (stock → $0)",
            "breakeven":      _first_be(),
        }

    # Unknown strategy — leave the original strings alone.
    return {}


def _mark_untradeable(strategy: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Return strategy with numeric.tradable=False + a reason field."""
    numeric = dict(strategy.get("numeric") or {})
    numeric["tradable"] = False
    numeric["untradeable_reason"] = reason
    return {**strategy, "numeric": numeric}


def _format_unfillable_reason(
    symbol: str,
    leg: Dict[str, Any],
    expiry: str,
    source: str,
    status: str,
) -> str:
    """Human-readable reason a leg can't be filled at open."""
    leg_desc = (
        f"{leg.get('action')} {leg.get('type')} ${leg.get('strike')} @ {expiry}"
    )
    if source == "intrinsic":
        return (
            f"no live quote for {symbol} {leg_desc} — refusing to fill "
            "at intrinsic value (would ignore time value). Try a more "
            "liquid strike or a closer expiry."
        )
    if status == "chain_unreachable":
        return (
            f"no chain available for {symbol} near {expiry} "
            f"(within ±{PRICER_SNAP_TOLERANCE_DAYS}d)"
        )
    if status == "strike_not_found":
        return f"strike not on the live chain: {symbol} {leg_desc}"
    return f"unable to price {symbol} {leg_desc} (source={source})"


def _format_unavailable_reason(
    symbol: str,
    leg: Dict[str, Any],
    expiry: str,
    status: str,
) -> str:
    """Human-readable reason a leg couldn't be priced."""
    leg_desc = (
        f"{leg.get('action')} {leg.get('type')} ${leg.get('strike')} @ {expiry}"
    )
    if status == "chain_unreachable":
        return (
            f"no chain available for {symbol} near {expiry} "
            f"(within ±{PRICER_SNAP_TOLERANCE_DAYS}d)"
        )
    if status == "strike_not_found":
        return f"strike not on the live chain: {symbol} {leg_desc}"
    return f"unable to price {symbol} {leg_desc}"
