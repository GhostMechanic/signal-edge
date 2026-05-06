"""
api/main.py
──────────────────────────────────────────────────────────────
FastAPI wrapper over the Prediqt Python stack.

Keeps ALL existing ML code untouched — this file just exposes
Python functions (get_full_analytics, get_quick_signal,
target_hit_analyzer, etc.) as HTTP endpoints so the Next.js
frontend at prediqt-web can fetch real data instead of the
hardcoded placeholders in the React components.

Run locally:
    source .venv/bin/activate
    uvicorn api.main:app --reload --port 8000

Then open:
    http://localhost:8000/docs                        → interactive API explorer
    http://localhost:8000/api/health
    http://localhost:8000/api/analytics/summary       → three honesty reads
    http://localhost:8000/api/analytics/per-horizon   → 5 horizon cards
    http://localhost:8000/api/analytics/simulated-portfolio
    http://localhost:8000/api/predictions?filter=scored&days=30&sort=newest&limit=20
"""

import math
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth import (
    CurrentUser,
    get_current_user,
    get_optional_user,
    require_verified_email,
)


def _json_safe(obj: Any) -> Any:
    """
    Recursively coerce a payload into JSON-safe primitives. Handles three
    nasties that escape FastAPI's default serializer:
      - NaN / Infinity floats (Python json writes literal NaN, browsers reject)
      - numpy ndarrays (no native JSON encoder; convert to nested lists)
      - numpy scalars (np.float64 etc.; unwrap via .item())
    """
    # Native Python primitives — handle floats first to catch NaN/Inf.
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (str, bytes, bool, int)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in obj]

    # numpy ndarray (multi-element) → recursive list. Check tolist() before
    # item() because ndarrays have item() too but it raises on >1-element.
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            pass

    # numpy scalar → Python scalar
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass

    # Last resort: stringify so the response at least serializes. Better than
    # killing the whole request because of one weird object.
    try:
        return str(obj)
    except Exception:
        return None


# ─── App instance ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Prediqt API",
    description="HTTP wrapper over the Prediqt ML stack.",
    version="0.2.0",
)


# ─── CORS ────────────────────────────────────────────────────────────────────
# Default list covers local dev + the canonical Vercel production URL. Extra
# origins can be supplied via CORS_EXTRA_ORIGINS (comma-separated) for custom
# domains, preview deploys on other hosts, etc.
_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    # The actual deployed Vercel project name is `predqt-web` (no "i" in
    # the middle). Earlier copies of this allowlist had `prediqt-web`
    # which doesn't match anything live and silently broke production.
    # Keep both spellings in case anyone ever forks under the alternate
    # name; both are no-ops if no project resolves there.
    "https://predqt-web.vercel.app",
    "https://prediqt-web.vercel.app",
]
_EXTRA_ORIGINS = [
    o.strip()
    for o in (os.getenv("CORS_EXTRA_ORIGINS", "") or "").split(",")
    if o.strip()
]
ALLOWED_ORIGINS = _DEFAULT_ORIGINS + _EXTRA_ORIGINS
# Vercel preview deploys (predqt-web-{hash}.vercel.app) always allowed.
# Same dual-spelling note applies — match either the actual project name
# or the legacy intended one.
ALLOW_ORIGIN_REGEX = r"https://(predqt|prediqt)-web-[a-z0-9-]+\.vercel\.app"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Startup: open the SQLite prediction store ───────────────────────────────
# Initializing eagerly at boot (rather than lazily on first prediction)
# means a misconfigured DB path / missing schema fails loudly *now* instead
# of silently corrupting an actual write later. The health snapshot logged
# here is the canary the user reads if Track Record is empty after a deploy.
@app.on_event("startup")
def _init_prediction_store() -> None:
    try:
        import prediction_store
        prediction_store.init_db()
        h = prediction_store.health_check()
        print(
            f"[startup] prediction_store ready @ {h['db_path']} "
            f"({h['predictions']} predictions, {h['horizons']} horizons, "
            f"{h['scores']} scores; latest={h['latest']})"
        )
    except Exception as e:
        # We don't crash the server — the rest of the API still works for
        # fundamentals/quotes/regime — but the operator needs to see this.
        print(f"[startup] prediction_store INIT FAILED: {type(e).__name__}: {e}")


# ─── Shared cache ────────────────────────────────────────────────────────────
# Enriching analytics (target-hit daily OHLC scan) is slow on a cold run.
# Every endpoint that touches analytics shares this cache so a single page
# load of Track Record (which hits 3–4 endpoints) only pays the cost once.
_ANALYTICS_CACHE: Dict[str, Any] = {}
_ANALYTICS_TTL_SECONDS = 300  # 5 minutes

# Current market-regime classification cache. Regime swings happen on
# weeks-to-months timescales, so caching for 15 min is generous —
# avoids hammering yfinance + the regime detector on every page load.
_MARKET_REGIME_CACHE: Dict[str, Any] = {}
_MARKET_REGIME_TTL_SECONDS = 900  # 15 minutes


def _get_enriched_analytics() -> Dict[str, Any]:
    """
    Returns the full analytics dict with target-hit enrichment applied,
    cached for 5 minutes. All analytics endpoints share this one source
    of truth so they never disagree with each other.
    """
    cached_data = _ANALYTICS_CACHE.get("data")
    cached_ts = _ANALYTICS_CACHE.get("timestamp")

    if cached_data is not None and cached_ts is not None:
        age = (datetime.utcnow() - cached_ts).total_seconds()
        if age < _ANALYTICS_TTL_SECONDS:
            return cached_data

    # Cache miss — recompute. Route through `db.get_full_analytics` so
    # Supabase-mode runs get the live counters (asking_events_total,
    # scored_any, target/checkpoint/expiration rates) overlaid on top
    # of the legacy SQLite analytics. Going straight to
    # prediction_logger_v2.get_full_analytics here used to surface the
    # frozen 2026-04-25 cutover snapshot on the landing page's "Calls
    # on the record" widget.
    from db import get_full_analytics, USE_SUPABASE
    from target_hit_analyzer import (
        enrich_predictions_with_target_hit,
        compute_target_hit_aggregates,
    )

    pa = get_full_analytics()

    # The legacy target_hit_analyzer iterates over the SQLite
    # `predictions_table` (per-horizon rows from before the cutover) to
    # compute target_hit_rate / target_hit_count / target_definitive.
    # In Supabase mode those rates already arrived from the rating_*
    # columns via `_supabase_prediction_counts`, and re-running the
    # analyzer here would clobber them with stale per-horizon SQLite
    # numbers (e.g. "142 hit · 147 resolved" against 98 total calls —
    # impossible if the units agree). Skip the analyzer in Supabase
    # mode; keep it for file-mode dev.
    if not USE_SUPABASE:
        try:
            pa["predictions_table"] = enrich_predictions_with_target_hit(
                pa.get("predictions_table", [])
            )
            pa.update(compute_target_hit_aggregates(pa["predictions_table"]))
        except Exception:
            pa.setdefault("target_hit_rate", 0)
            pa.setdefault("target_hit_count", 0)
            pa.setdefault("target_definitive", 0)
            pa.setdefault("target_per_horizon", {})
    else:
        # Defensive: if the Supabase counter didn't populate these fields
        # (e.g. transient client error), keep them zeroed rather than
        # falling through to legacy SQLite numbers that don't match the
        # Supabase headline count.
        pa.setdefault("target_hit_rate", 0)
        pa.setdefault("target_hit_count", 0)
        pa.setdefault("target_definitive", 0)
        pa.setdefault("target_per_horizon", {})

    _ANALYTICS_CACHE["data"] = pa
    _ANALYTICS_CACHE["timestamp"] = datetime.utcnow()
    return pa


# ─── Health check ────────────────────────────────────────────────────────────
@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Simple check that the API is alive. Used by the frontend + Render."""
    return {
        "status": "ok",
        "service": "prediqt-api",
        "version": "0.2.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ─── Analytics: summary ──────────────────────────────────────────────────────
@app.get("/api/analytics/summary")
def analytics_summary() -> Dict[str, Any]:
    """
    Top-line scoreboard: three honesty reads + sample sizes.
    Fuels the landing-page ScoreboardTeaser component.
    """
    from prediction_logger_v2 import get_current_model_version

    pa = _get_enriched_analytics()
    model_version = get_current_model_version()

    scored_any = pa.get("scored_any", 0)
    wins = pa.get("direction_correct_any", 0)
    checkpoint_win_rate = (wins / scored_any * 100.0) if scored_any > 0 else 0

    return {
        "model_version":      model_version,
        # `total_predictions` is per-horizon (one row per asking-event ×
        # horizon). Useful for "how many *scoring evaluations* have we
        # done" — but consumers wanting "how many *asking events* / how
        # many *calls*" want `asking_events_total` instead.
        "total_predictions":  pa.get("total_predictions", 0),
        "asking_events_total": pa.get("total_analyses", 0),
        "scored_any":         scored_any,
        "scored_final":       pa.get("scored_final", 0),

        "target_hit_rate":    round(pa.get("target_hit_rate", 0), 1),
        "target_hit_count":   pa.get("target_hit_count", 0),
        "target_definitive":  pa.get("target_definitive", 0),

        "checkpoint_win_rate": round(checkpoint_win_rate, 1),
        "checkpoint_wins":     wins,
        "checkpoint_losses":   scored_any - wins,

        "expiration_win_rate": round(pa.get("live_accuracy", 0), 1),
        "expiration_correct":  pa.get("direction_correct_final", 0),
        "expiration_matured":  pa.get("scored_final", 0),
    }


# ─── New Supabase-backed endpoints (Phase 1 anchor docs) ─────────────────────
#
# These three endpoints surface the Phase 1 backend additions:
#   • /api/analytics/accuracy-bands → methodology § 6 three accuracy bands
#                                      (all/traded/passed) + conviction lift
#   • /api/portfolio/summary        → the model's actual paper portfolio:
#                                      cash, equity, return, equity curve
#   • /api/ledger                   → the public-tier ledger view (limited
#                                      fields: ticker, direction, horizon,
#                                      verdict, traded — no trade details)
#
# All three require USE_SUPABASE=true (the legacy SQLite store doesn't have
# the `traded` column or model_paper_portfolio tables). When Supabase mode
# is off, they return a 503-shaped error response so the frontend can
# render a "coming soon" state instead of crashing.


def _supabase_unavailable_response(reason: str) -> Dict[str, Any]:
    """Shape a uniform error response when Supabase mode is off but the
    endpoint requires it. Frontend can key off `available=false` to
    render a placeholder."""
    return {
        "available": False,
        "reason":    reason,
        "hint":      "Set USE_SUPABASE=true in the API service env.",
    }


@app.get("/api/analytics/accuracy-bands")
def analytics_accuracy_bands(public_only: bool = True) -> Dict[str, Any]:
    """
    The three accuracy bands from methodology § 6:
        all      — every settled prediction
        traded   — settled where the model committed paper money (conviction)
        passed   — settled where the model watched but didn't trade
    Plus the headline `conviction_lift` metric: hit_rate(traded) - hit_rate(all).

    `public_only=true` (default) restricts to public-ledger predictions —
    the same set the public Track Record sees. Set false to include
    member-only predictions (used for member-tier analytics).
    """
    try:
        from db import get_accuracy_bands
        return _json_safe({
            "available": True,
            **get_accuracy_bands(public_only=public_only),
        })
    except NotImplementedError as exc:
        return _supabase_unavailable_response(str(exc))


@app.get("/api/portfolio/summary")
def portfolio_summary() -> Dict[str, Any]:
    """
    The model's actual paper portfolio. cash + open positions (qty × entry)
    valued live; equity_curve persists daily MTM points (anchor § 7).

    This is distinct from /api/analytics/simulated-portfolio, which is the
    legacy "what if the model traded every prediction equal-weight" simulation.
    The new endpoint surfaces the *curated* portfolio shaped by the
    TRADE/PASS rule (anchor § 2.1).
    """
    try:
        from db import get_model_portfolio_summary
        return _json_safe({
            "available": True,
            **get_model_portfolio_summary(),
        })
    except NotImplementedError as exc:
        return _supabase_unavailable_response(str(exc))


# ─── User paper trading (Phase 2.0 — DEV_USER_ID single-user mode) ───────────
#
# Three endpoints back the "make your own trade, track your own performance"
# surface that flips on the Track Record page. They mirror the model's
# paper-trading discipline (anchor § 7) — $10,000 starting capital, equal-
# weight $400 positions, 25-position cap, no edits, no deletes — but the
# state lives in `paper_portfolios` + `paper_trades` (per-user) instead of
# `model_paper_portfolio` + `model_paper_trades` (singleton).
#
# Auth note: today there's only DEV_USER_ID. The db helpers route everything
# through `_current_user_id()` which falls back to the dev user when no JWT
# is present. When real auth ships in Phase 2.5, the FastAPI dep will
# extract user_id from the Supabase JWT and pass it explicitly — the helpers
# already accept an optional user_id param so the swap is one line.

# Horizon → calendar days. Same mapping used elsewhere in this module.
_HORIZON_DAYS = {
    "3 Day": 3, "1 Week": 7, "1 Month": 30,
    "1 Quarter": 90, "1 Year": 365,
}


def _resolve_close_on_or_before(symbol: str, target_iso_date: str) -> Optional[float]:
    """
    Fetch the close price for `symbol` on `target_iso_date` (YYYY-MM-DD).
    Falls back to the nearest trading day on or before the target — markets
    don't trade weekends/holidays, so a horizon that lands on Saturday
    rolls to Friday's close.

    Used for the "model mechanical exit" comparison on horizon-ended user
    trades. Returns None on data-feed failure so the caller can render a
    "comparison unavailable" footnote instead of crashing.
    """
    try:
        from data_fetcher import fetch_stock_data
        # 1y window is wide enough to comfortably contain any horizon's
        # expiry day from a recently-opened trade. Cheaper than always
        # fetching max history.
        df = fetch_stock_data(symbol, period="1y")
        if df is None or "Close" not in df.columns or len(df) == 0:
            return None
        # Index is timezone-aware datetime in yfinance; coerce to date.
        target = target_iso_date
        # Walk the index in reverse to find the latest close on/before target.
        for idx, row in df.iloc[::-1].iterrows():
            try:
                d = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            except Exception:
                continue
            if d <= target:
                price = float(row["Close"])
                if price == price and price > 0:
                    return round(price, 4)
                return None
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[paper-trade] _resolve_close_on_or_before({symbol},{target_iso_date}) failed: {exc}")
        return None


@app.get("/api/me/predictions")
def my_predictions(
    cu: CurrentUser = Depends(get_current_user),
    status: str = "open",  # "open" | "settled" | "all"
    limit: int = 50,
) -> Dict[str, Any]:
    """
    The signed-in user's own predictions, filtered by status.

    Powers the /me personal dashboard. Returns the user's predictions
    table rows (NOT the public ledger — these include the user's
    Watched calls, which the public ledger strips), enriched per-row
    with the current live price + a few derived numbers the dashboard
    cards render straight onto progress bars:

      • current_price        — fresh from yfinance (cached 15min server
                                side via /api/quote/<sym> path).
      • progress_pct          — how far the stock has moved from entry
                                toward target (positive = toward target,
                                negative = toward stop). For LONG calls
                                this is (current − entry) / (target −
                                entry); for SHORT, mirrored.
      • distance_to_target_pct — magnitude in % from current to target.
      • distance_to_stop_pct   — magnitude in % from current to stop.
      • days_until_judgment   — days to horizon_ends_at; negative when
                                past expiry (still OPEN means scoring
                                hasn't run yet).

    Service-role read so RLS doesn't block the auth'd user from seeing
    their own rows. user_id filter applied explicitly.
    """
    from db import _service_client

    valid_status = {"open", "settled", "all"}
    if status not in valid_status:
        return {"error": "invalid_status", "valid": sorted(valid_status)}

    client = _service_client()
    q = client.table("predictions").select("*").eq("user_id", cu.id)
    if status == "open":
        q = q.eq("verdict", "OPEN")
    elif status == "settled":
        q = q.in_("verdict", ["HIT", "PARTIAL", "MISSED"])

    try:
        rows = (
            q.order("created_at", desc=True)
             .limit(min(max(limit, 1), 200))
             .execute()
        ).data or []
    except Exception as exc:
        print(f"[me/predictions] query failed for {cu.id}: {exc}")
        return {"error": "query_failed", "detail": str(exc)[:200]}

    # Fetch current prices once per unique symbol — saves a round trip
    # when the user has multiple horizons of the same ticker.
    unique_symbols = list({r.get("symbol", "").upper() for r in rows if r.get("symbol")})
    price_cache: Dict[str, Optional[float]] = {}
    for sym in unique_symbols:
        try:
            price_cache[sym] = _resolve_current_price(sym)
        except Exception:
            price_cache[sym] = None

    now_utc = datetime.now(timezone.utc)
    enriched: list = []
    for r in rows:
        sym     = (r.get("symbol") or "").upper()
        direction = r.get("direction") or "Neutral"
        entry   = float(r.get("entry_price") or 0)
        target  = float(r.get("target_price") or 0) if r.get("target_price") is not None else None
        stop    = float(r.get("stop_price")   or 0) if r.get("stop_price")   is not None else None
        current = price_cache.get(sym)

        progress_pct: Optional[float] = None
        distance_to_target_pct: Optional[float] = None
        distance_to_stop_pct: Optional[float] = None
        days_until_judgment: Optional[float] = None

        if current is not None and entry > 0 and target is not None:
            denom = target - entry
            if abs(denom) > 1e-9:
                progress_raw = (current - entry) / denom
                # Mirror for SHORT — for a Bearish call, we WANT the
                # stock to fall; positive progress = current is closer
                # to target than entry (in the Bearish sense).
                if direction == "Bearish":
                    progress_raw = -progress_raw
                progress_pct = round(progress_raw * 100, 2)
            distance_to_target_pct = round(abs((target - current) / current) * 100, 2) if current > 0 else None
        if current is not None and stop is not None and current > 0:
            distance_to_stop_pct = round(abs((stop - current) / current) * 100, 2)

        ends_at_iso = r.get("horizon_ends_at")
        if ends_at_iso:
            try:
                ends_at = datetime.fromisoformat(ends_at_iso.replace("Z", "+00:00"))
                if ends_at.tzinfo is None:
                    ends_at = ends_at.replace(tzinfo=timezone.utc)
                days_until_judgment = round(
                    (ends_at - now_utc).total_seconds() / 86400, 2
                )
            except Exception:
                pass

        enriched.append({
            **r,
            "current_price":           current,
            "progress_pct":            progress_pct,
            "distance_to_target_pct":  distance_to_target_pct,
            "distance_to_stop_pct":    distance_to_stop_pct,
            "days_until_judgment":     days_until_judgment,
        })

    return {
        "user_id": cu.id,
        "status":  status,
        "count":   len(enriched),
        "rows":    enriched,
    }


@app.get("/api/paper-portfolio")
def user_paper_portfolio(
    cu: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    The user's own paper portfolio: cash, open positions, recent closes,
    and the equity curve. Same shape as /api/portfolio/summary (the model's
    book) so the frontend can re-use the same components.

    Each open trade is enriched server-side with:
      - `current_price` + `unrealised_pnl_live` — direction-aware MTM
      - `horizon_ends_at` — derived from opened_at + horizon duration
      - `horizon_ended` — bool, true when now > horizon_ends_at
      - `days_past_horizon` — when expired, how many days past
      - `mechanical_close_price` + `mechanical_pnl` — when expired, the
        price + P&L the model would have closed at on horizon-end day.
        Surfacing this lets the user see their *lift* over the model's
        mechanical exit (have they added value by holding?).

    No auto-close — the user's portfolio is the user's call. Horizon
    expiry just adds context; positions stay open until manually closed.
    Methodology § 7's auto-close rule applies to the model's book only.
    """
    # Auto-settle any expired option trades before reading the book.
    # Per Phase B design: options resolve at expiry only, no manual close.
    # Settlement is idempotent and runs in O(open_options) — cheap.
    # Pass cu.id explicitly so this works even if the ContextVar isn't
    # propagating cleanly (defensive — the auth dep is now async, so
    # the var should be set, but we don't want a fatal traceback if
    # somehow it isn't).
    try:
        from db import settle_expired_option_trades
        settled = settle_expired_option_trades(user_id=cu.id)
        if settled > 0:
            print(f"[paper-portfolio] auto-settled {settled} expired option trade(s)")
    except TypeError:
        # Older signature without user_id kwarg — fall through.
        try:
            from db import settle_expired_option_trades as _ssot
            settled = _ssot()
        except Exception as exc:  # noqa: BLE001
            print(f"[paper-portfolio] settle_expired_option_trades failed (non-fatal): {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"[paper-portfolio] settle_expired_option_trades failed (non-fatal): {exc}")

    try:
        from db import get_user_paper_portfolio
        book = get_user_paper_portfolio(user_id=cu.id)
    except NotImplementedError as exc:
        return _supabase_unavailable_response(str(exc))
    except TypeError:
        # Backward compat — if get_user_paper_portfolio doesn't take a
        # user_id kwarg yet, fall through to the ContextVar path.
        from db import get_user_paper_portfolio as _gupp
        book = _gupp()

    # Enrich each open trade with a live current price + unrealised P&L.
    # We do this in the API layer (not db.py) because db.py is the
    # Supabase boundary, not the market-data boundary.
    open_trades = book.get("open_trades") or []
    if open_trades:
        from datetime import datetime, timezone, timedelta
        now_utc = datetime.now(timezone.utc)
        # Cache lookups by symbol so a portfolio with multiple positions
        # in the same ticker only hits the data feed once.
        price_cache: Dict[str, Optional[float]] = {}
        live_open_value = 0.0
        for tr in open_trades:
            sym = (tr.get("symbol") or "").upper().strip()
            kind = (tr.get("kind") or "equity").lower()

            # Option trades: skip equity-style live MTM. Options resolve
            # at expiry only (Phase B design — no intra-trade greeks
            # modeling). Surface the underlying price so the row can show
            # "Underlying now $X" but unrealised P&L stays null until
            # expiry. Locked-cash contributes to the open_value figure.
            if kind == "option":
                if sym and sym not in price_cache:
                    price_cache[sym] = _resolve_current_price(sym)
                tr["current_price"] = price_cache.get(sym)
                tr["unrealised_pnl_live"] = None
                tr["horizon_ended"] = False  # options have own expiry semantics
                ins = tr.get("instrument_data") or {}
                live_open_value += float(ins.get("cash_locked") or 0)
                continue

            if not sym:
                tr["current_price"] = None
                tr["unrealised_pnl_live"] = None
                tr["horizon_ended"] = False
                continue
            if sym not in price_cache:
                price_cache[sym] = _resolve_current_price(sym)
            cur = price_cache[sym]
            tr["current_price"] = cur
            qty = float(tr.get("qty") or 0)
            entry = float(tr.get("entry_price") or 0)
            direction = (tr.get("direction") or "LONG").upper()

            # Live MTM
            if cur is not None and qty > 0 and entry > 0:
                if direction == "LONG":
                    pnl = (cur - entry) * qty
                else:
                    pnl = (entry - cur) * qty
                tr["unrealised_pnl_live"] = round(pnl, 2)
                live_open_value += cur * qty
            else:
                tr["unrealised_pnl_live"] = None
                live_open_value += entry * qty  # fall back to entry

            # Horizon-ended state
            horizon = tr.get("horizon")
            opened_at = tr.get("opened_at")
            horizon_days = _HORIZON_DAYS.get(horizon)
            if opened_at and horizon_days:
                try:
                    # Postgres returns ISO with offset; tolerate trailing 'Z'.
                    opened_dt = datetime.fromisoformat(
                        opened_at.replace("Z", "+00:00")
                    )
                    if opened_dt.tzinfo is None:
                        opened_dt = opened_dt.replace(tzinfo=timezone.utc)
                    ends_at = opened_dt + timedelta(days=horizon_days)
                    tr["horizon_ends_at"] = ends_at.isoformat()
                    horizon_ended = now_utc > ends_at
                    tr["horizon_ended"] = horizon_ended
                    if horizon_ended:
                        days_past = (now_utc - ends_at).total_seconds() / 86400
                        tr["days_past_horizon"] = int(days_past)
                        # Resolve the close on horizon-end day so we can
                        # show the model's mechanical exit P&L.
                        target_date = ends_at.strftime("%Y-%m-%d")
                        mech_price = _resolve_close_on_or_before(sym, target_date)
                        if mech_price is not None and qty > 0 and entry > 0:
                            if direction == "LONG":
                                mech_pnl = (mech_price - entry) * qty
                            else:
                                mech_pnl = (entry - mech_price) * qty
                            tr["mechanical_close_price"] = mech_price
                            tr["mechanical_pnl"] = round(mech_pnl, 2)
                        else:
                            tr["mechanical_close_price"] = None
                            tr["mechanical_pnl"] = None
                except Exception as exc:  # noqa: BLE001
                    print(f"[paper-trade] horizon parse failed for {tr.get('id')}: {exc}")
                    tr["horizon_ended"] = False
            else:
                tr["horizon_ended"] = False

        # Reflect the live mark in the portfolio top-line so the front
        # face shows real-time return, not just a cash + entry sum.
        cash = float(book.get("cash") or 0)
        book["open_value_live"] = round(live_open_value, 2)
        book["portfolio_value_live"] = round(cash + live_open_value, 2)
        starting = float(book.get("starting_capital") or 10000)
        book["total_return_pct_live"] = (
            round((cash + live_open_value - starting) / starting * 100, 4)
            if starting else 0.0
        )

    return _json_safe(book)


def _rr_from_logger_meta(meta: Dict[str, Any]) -> Optional[float]:
    """
    Compute reward / risk from prediction_logger_v2's out_meta dict.
    Reads entry_price / stop_price / target_price (the methodology-clamped
    values that the trade-attachment phase stamped onto the record), not
    the pre-computed `risk_reward` field. This makes the API the sole
    source of truth for R:R: even if some upstream path produces a stale
    or wrong `risk_reward` number, the API response always derives R:R
    from the same entry/stop/target the row was persisted with — which is
    what /api/ledger/{id} will return when the frontend re-reads the row.

    Returns None when any input is missing or risk is zero (Neutral
    predictions, broken plans, etc.) so the frontend can fall back
    cleanly. Rounded to 4 decimal places to match the pre-fix shape.
    """
    e = meta.get("entry_price")
    s = meta.get("stop_price")
    t = meta.get("target_price")
    if e is None or s is None or t is None:
        return None
    try:
        risk = abs(float(e) - float(s))
        reward = abs(float(t) - float(e))
    except (TypeError, ValueError):
        return None
    if risk <= 0:
        return None
    return round(reward / risk, 4)


def _resolve_current_price(symbol: str) -> Optional[float]:
    """
    Fetch the latest available close for `symbol`. Used by the paper-trade
    open path to fill at current market — never trust the client.

    Falls back to None on any data-fetcher hiccup; the caller surfaces
    that as a "couldn't pull market price" error so the user knows
    why the trade didn't open.
    """
    try:
        from data_fetcher import fetch_stock_data
        df = fetch_stock_data(symbol, period="5d")
        if df is None or "Close" not in df.columns or len(df) == 0:
            return None
        last = float(df["Close"].iloc[-1])
        if last != last or last <= 0:  # NaN or non-positive
            return None
        return round(last, 4)
    except Exception as exc:  # noqa: BLE001
        print(f"[paper-trade] _resolve_current_price({symbol}) failed: {exc}")
        return None


@app.post("/api/paper-trades")
def open_user_paper_trade_endpoint(
    body: Dict[str, Any] = Body(...),
    cu: CurrentUser = Depends(get_current_user),  # noqa: B008
) -> Dict[str, Any]:
    """
    Open a paper trade for the current user against a persisted prediction.
    Fills at current market price (server-fetched), not the prediction's
    persisted thesis entry.

    Body shape:
        {
            "prediction_id": "<uuid>",   # required — the call to back
            "qty":           <number>,   # required — share count, > 0
        }

    The fill price is resolved server-side from the same data feed the
    rest of the app uses. The model's planned entry is recorded on the
    trade row's `meta.planned_entry` for drift auditing — we want the
    user vs model comparison to be honest about execution timing.

    Validation lives in db.open_user_paper_trade(): prediction must exist,
    qty must be positive, the thesis can't have already played out (price
    crossed past target/stop), user needs cash, no double-position on a
    single symbol, and the 25-position cap holds. Any guardrail trip
    surfaces as `{ok: false, error: "..."}` with HTTP 200.
    """
    prediction_id = (body.get("prediction_id") or "").strip()
    qty = body.get("qty")

    if not prediction_id:
        return {"ok": False, "error": "prediction_id is required"}
    if qty is None:
        return {"ok": False, "error": "qty is required"}

    try:
        from db import open_user_paper_trade

        # Look up the prediction's symbol so we can pull a fresh quote
        # before the helper runs. We do this here (rather than inside
        # the helper) to keep db.py free of data-fetcher imports — db.py
        # is the Supabase boundary, not the market-data boundary.
        from db import _client, USE_SUPABASE
        fill_price: Optional[float] = None
        # Track WHY we don't have a fill price so the user gets a useful
        # error. The two failure modes — prediction not found vs market
        # data feed failed — call for different fixes (frontend/auth bug
        # vs transient retry), so we don't want to collapse them under
        # one misleading message.
        lookup_failed = False
        if USE_SUPABASE:
            try:
                pred = (
                    _client().table("predictions")
                    .select("symbol")
                    .eq("id", prediction_id)
                    .single()
                    .execute()
                ).data or {}
                sym = (pred.get("symbol") or "").upper().strip()
                if sym:
                    fill_price = _resolve_current_price(sym)
                else:
                    # .single() returned data but no symbol — rare; treat
                    # as a lookup miss so the user sees the right message.
                    lookup_failed = True
            except Exception as exc:  # noqa: BLE001
                # PGRST116 fires when .single() finds 0 or >1 rows. Both
                # are "API can't resolve this prediction" from the user's
                # perspective; surface that, not a fake market-data error.
                lookup_failed = True
                print(f"[paper-trade] symbol lookup failed for {prediction_id}: {exc}")

        if lookup_failed:
            return {
                "ok": False,
                "error": (
                    "couldn't find that prediction. it may not have been "
                    "saved, or it belongs to a different account. try "
                    "asking again, then take the play from the fresh "
                    "receipt."
                ),
            }
        if fill_price is None:
            return {
                "ok": False,
                "error": (
                    "couldn't pull current market price for this ticker. "
                    "try again in a moment — the market data feed may be "
                    "stale."
                ),
            }

        result = open_user_paper_trade(
            prediction_id=prediction_id,
            qty=float(qty),
            fill_price=fill_price,
        )
        return _json_safe({"ok": True, **result})
    except NotImplementedError as exc:
        return _supabase_unavailable_response(str(exc))
    except ValueError as exc:
        # Guardrail trip — return as a soft failure so the frontend can
        # render the message inline without bubbling it up as a network
        # error. Voice doc § 3: "no hiding" — the user sees exactly why
        # the trade didn't go through.
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — last-ditch surface to client
        return {"ok": False, "error": f"unexpected error: {exc}"}


@app.post("/api/paper-trades/{trade_id}/close")
def close_user_paper_trade_endpoint(
    trade_id: str,
    body: Dict[str, Any] = Body(...),
    cu: CurrentUser = Depends(get_current_user),  # noqa: B008
) -> Dict[str, Any]:
    """
    Manually close an open paper trade. Equity trades close at the
    client-supplied current price (preserved for backwards compat with
    the existing equity flow). Option trades close at server-fetched
    live mid-price via options_pricer — the body's current_price is
    ignored for options because the spread is what fills, not a single
    number, and the client can't be trusted to compute it.

    Dispatch is by `paper_trades.kind`:
      kind = 'equity'  → db.close_user_paper_trade(trade_id, current_price)
      kind = 'option'  → db.close_user_paper_option_trade(trade_id)

    Body shape:
        {
            "current_price": <number>   # required for equity, ignored for options
        }

    Returns (equity):
        { ok: true, trade_id, exit_price, realised_pnl, cash }
    Returns (option):
        { ok: true, trade_id, exit_price, realised_pnl, cash, cash_back,
          source, per_leg, liq_per_contract, position_value }
    """
    try:
        from db import _client, _current_user_id, USE_SUPABASE
        if not USE_SUPABASE:
            return _supabase_unavailable_response(
                "paper-trades close requires Supabase mode."
            )
        # Resolve `kind` server-side so we never trust the client to tell
        # us which path to take.
        uid = _current_user_id()
        kind_res = (
            _client().table("paper_trades")
                .select("kind")
                .eq("id", trade_id)
                .eq("user_id", uid)
                .single()
                .execute()
        )
        kind_row = kind_res.data
        if not kind_row:
            return {"ok": False, "error": f"trade not found: {trade_id}"}
        kind = (kind_row.get("kind") or "equity").lower()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"unexpected error: {exc}"}

    if kind == "option":
        try:
            from db import close_user_paper_option_trade
            result = close_user_paper_option_trade(trade_id=trade_id)
            return _json_safe({"ok": True, **result})
        except NotImplementedError as exc:
            return _supabase_unavailable_response(str(exc))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"unexpected error: {exc}"}

    # Equity (default).
    current_price = body.get("current_price")
    if current_price is None:
        return {"ok": False, "error": "current_price is required"}

    try:
        from db import close_user_paper_trade
        result = close_user_paper_trade(
            trade_id=trade_id,
            current_price=float(current_price),
        )
        return _json_safe({"ok": True, **result})
    except NotImplementedError as exc:
        return _supabase_unavailable_response(str(exc))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"unexpected error: {exc}"}


@app.post("/api/paper-trades/options")
def open_user_paper_option_trade_endpoint(
    body: Dict[str, Any] = Body(...),
    cu: CurrentUser = Depends(get_current_user),  # noqa: B008
) -> Dict[str, Any]:
    """
    Open an options paper trade against a persisted prediction's
    options_strategy. Phase B — expiry-only resolution (no manual close).

    Body shape:
        {
            "prediction_id": "<uuid>",   # required
            "qty":           <integer>,  # required — number of contracts (≥1)
        }

    The prediction must have an options_strategy persisted on its row
    (Phase B.1 migration). Strategy is taken at face value — same
    expiry_days, legs, premiums, and breakevens that were shown to the
    user on the asking flow at /prediqt.

    Cash impact:
      - Debit strategies (Long Call, Bull Call Spread, etc.): premium
        debited from cash. cash_locked = premium.
      - Credit strategies (Iron Condor, Cash-Secured Put): credit
        received, max-loss locked as margin. cash_locked = max_loss − credit.

    On expiry, /api/paper-portfolio auto-settles the trade by
    resolving the underlying close on expiry-day and computing P&L
    from the legs' intrinsic values.

    Any guardrail trip surfaces as `{ok: false, error: "..."}`.
    """
    prediction_id = (body.get("prediction_id") or "").strip()
    qty           = body.get("qty")

    if not prediction_id:
        return {"ok": False, "error": "prediction_id is required"}
    if qty is None:
        return {"ok": False, "error": "qty is required"}

    try:
        from db import open_user_paper_option_trade
        result = open_user_paper_option_trade(
            prediction_id=prediction_id,
            qty=int(qty),
        )
        return _json_safe({"ok": True, **result})
    except NotImplementedError as exc:
        return _supabase_unavailable_response(str(exc))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"unexpected error: {exc}"}


@app.get("/api/ledger/{prediction_id}")
def ledger_detail(prediction_id: str) -> Dict[str, Any]:
    """
    Full detail for a single prediction by its Supabase UUID.

    Used by the public ledger's expandable rows. Returns every field on
    the row — including paid-tier fields (entry/stop/target prices,
    confidence, predicted_return/price, regime, model_version, etc.).

    TODO (Phase 2 — auth gating): currently un-gated because there's only
    one user and no auth. When accounts ship, this endpoint should:
      - Inspect the caller's tier from the JWT
      - Strip paid-tier fields for non-paid users
      - Strip member-tier fields for anonymous visitors
    Per data-model anchor § 6 the public-tier fields are id, symbol,
    direction, horizon, verdict, traded, created_at, horizon_ends_at.
    """
    try:
        from db import get_prediction_detail
        row = get_prediction_detail(prediction_id)
        if row is None:
            return {
                "available": True,
                "found":     False,
                "message":   f"prediction {prediction_id} not found",
            }
        return _json_safe({
            "available": True,
            "found":     True,
            "row":       row,
        })
    except NotImplementedError as exc:
        return _supabase_unavailable_response(str(exc))


@app.get("/api/ledger")
def public_ledger(
    limit:        int  = 50,
    offset:       int  = 0,
    only_settled: bool = False,
) -> Dict[str, Any]:
    """
    Paginated public-tier prediction ledger. Returns only the fields a
    public visitor sees per data-model anchor § 6:
        id, symbol, direction, horizon, verdict, traded,
        created_at, horizon_ends_at

    Members and paid users get richer reads via separate endpoints (TODO).

    Query params:
        limit         (default 50)  — max rows to return
        offset        (default 0)   — pagination cursor
        only_settled  (default false) — restrict to verdict != 'OPEN'
    """
    # Defensive bounds — don't let a client request 100k rows in one shot.
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))

    try:
        from db import get_public_ledger
        rows = get_public_ledger(
            limit=limit, offset=offset, only_settled=only_settled,
        )
        return _json_safe({
            "available":    True,
            "rows":         rows,
            "limit":        limit,
            "offset":       offset,
            "only_settled": only_settled,
            "count":        len(rows),
        })
    except NotImplementedError as exc:
        return _supabase_unavailable_response(str(exc))


# ─── Analytics: per-horizon ──────────────────────────────────────────────────
HORIZON_ORDER = ["3 Day", "1 Week", "1 Month", "1 Quarter", "1 Year"]


@app.get("/api/analytics/per-horizon")
def analytics_per_horizon() -> Dict[str, Any]:
    """
    Breakdown of the three honesty reads across each prediction horizon.
    Fuels the 5 horizon cards on the Track Record page.
    """
    pa = _get_enriched_analytics()
    per_horizon = pa.get("per_horizon", {})
    target_per_horizon = pa.get("target_per_horizon", {})

    # Checkpoint win rate per horizon — walk preds_table, find the latest
    # scored interval per prediction, tally wins/losses per horizon.
    checkpoint_by_hz: Dict[str, Dict[str, int]] = {}
    for p in pa.get("predictions_table", []):
        hz = p.get("horizon", "")
        d = checkpoint_by_hz.setdefault(hz, {"scored": 0, "correct": 0})
        for interval in ["365d", "180d", "90d", "60d", "30d", "14d", "7d", "3d", "1d"]:
            s = p.get(f"score_{interval}")
            if s in ("✓", "✗"):
                d["scored"] += 1
                if s == "✓":
                    d["correct"] += 1
                break

    horizons = []
    for name in HORIZON_ORDER:
        h_data = per_horizon.get(name, {})
        t_data = target_per_horizon.get(name, {})
        c_data = checkpoint_by_hz.get(name, {"scored": 0, "correct": 0})

        total = h_data.get("total", 0)
        scored_final = h_data.get("scored", 0)
        correct_final = h_data.get("correct", 0)
        acc_final = h_data.get("accuracy", 0)

        c_scored = c_data["scored"]
        c_correct = c_data["correct"]
        c_rate = (c_correct / c_scored * 100) if c_scored > 0 else 0

        t_hit = t_data.get("hit", 0)
        t_def = t_data.get("definitive", 0)
        t_rate = t_data.get("rate", 0)

        horizons.append({
            "name":  name,
            "total": total,
            "target": {
                "rate":           round(t_rate, 1),
                "hit_count":      t_hit,
                "resolved_count": t_def,
                "has_data":       t_def > 0,
            },
            "checkpoint": {
                "rate":          round(c_rate, 1),
                "correct_count": c_correct,
                "scored_count":  c_scored,
                "has_data":      c_scored > 0,
            },
            "expiration": {
                "rate":          round(acc_final, 1),
                "correct_count": correct_final,
                "matured_count": scored_final,
                "has_data":      scored_final > 0,
            },
        })

    return {"horizons": horizons}


# ─── Analytics: per-horizon HISTORY (rolling rates over time) ───────────────
@app.get("/api/analytics/per-horizon-history")
def analytics_per_horizon_history(range: str = "30d") -> Dict[str, Any]:
    """
    Time-series version of /api/analytics/per-horizon. For each horizon ×
    metric (target / checkpoint / expiration), returns a list of points
    [{date, rate, n}] showing how the rolling rate has evolved.

    The chart uses a *rolling-window* rate (not cumulative) so the line
    reflects recent trends instead of converging asymptotically. The window
    width scales with the requested range so short ranges stay responsive
    and long ranges stay smooth.
    """
    range_window = {
        "7d":  {"days": 7,   "window": 7,   "step": 1},
        "30d": {"days": 30,  "window": 14,  "step": 1},
        "90d": {"days": 90,  "window": 30,  "step": 3},
        "all": {"days": None, "window": 60,  "step": 7},
    }
    if range not in range_window:
        return {
            "error": "invalid_range",
            "valid": list(range_window.keys()),
        }
    cfg = range_window[range]

    pa = _get_enriched_analytics()
    preds = pa.get("predictions_table", [])

    # Build event lists: for each prediction, figure out for each metric
    # (a) the date it resolved and (b) whether it won. Predictions that
    # haven't resolved a given metric are skipped for that metric.
    events_by_horizon_metric: Dict[str, Dict[str, list]] = {}

    for p in preds:
        hz = p.get("horizon", "")
        if not hz:
            continue

        # Target — definitive iff target_hit is True or (False AND expired).
        th = p.get("target_hit")
        if th is True:
            d = p.get("target_hit_date") or _resolved_date(p)
            if d:
                events_by_horizon_metric.setdefault(hz, {}).setdefault(
                    "target", []
                ).append((d, True))
        elif th is False and p.get("horizon_expired"):
            d = _resolved_date(p)
            if d:
                events_by_horizon_metric.setdefault(hz, {}).setdefault(
                    "target", []
                ).append((d, False))

        # Checkpoint — the latest scored interval. Date is approximated
        # using _resolved_date since exact checkpoint timestamps aren't
        # stamped per interval; we need an ordering, not precision.
        ck = _latest_checkpoint_score(p)
        if ck is not None:
            d = _resolved_date(p)
            if d:
                events_by_horizon_metric.setdefault(hz, {}).setdefault(
                    "checkpoint", []
                ).append((d, ck))

        # Expiration — only matured predictions (final ✓/✗).
        fr = p.get("final_result")
        if fr in ("✓", "✗"):
            d = _resolved_date(p)
            if d:
                events_by_horizon_metric.setdefault(hz, {}).setdefault(
                    "expiration", []
                ).append((d, fr == "✓"))

    # Sort each event list by date.
    for hz_dict in events_by_horizon_metric.values():
        for ev_list in hz_dict.values():
            ev_list.sort(key=lambda t: t[0])

    # Build sample dates across the requested range.
    today = datetime.utcnow().date()
    if cfg["days"] is None:
        # All-time: derive start from earliest event across all metrics.
        all_dates = []
        for hz_dict in events_by_horizon_metric.values():
            for ev_list in hz_dict.values():
                if ev_list:
                    all_dates.append(ev_list[0][0])
        if not all_dates:
            return {"range": range, "horizons": {}}
        try:
            earliest = min(
                datetime.strptime(d, "%Y-%m-%d").date() for d in all_dates
            )
        except Exception:
            earliest = today - timedelta(days=180)
        start = earliest
    else:
        start = today - timedelta(days=cfg["days"])

    sample_dates = []
    cur = start
    while cur <= today:
        sample_dates.append(cur)
        cur += timedelta(days=cfg["step"])

    # For each (horizon, metric) compute a rolling-window rate at each
    # sample date. Window: last `cfg["window"]` days inclusive.
    HORIZONS = ["3 Day", "1 Week", "1 Month", "1 Quarter", "1 Year"]
    METRICS = ["target", "checkpoint", "expiration"]
    horizons_out: Dict[str, Any] = {}

    for hz in HORIZONS:
        per_metric: Dict[str, Any] = {}
        hz_events = events_by_horizon_metric.get(hz, {})
        for m in METRICS:
            ev_list = hz_events.get(m, [])
            if not ev_list:
                per_metric[m] = []
                continue

            # Pre-parse dates once.
            parsed = []
            for d_str, won in ev_list:
                try:
                    d = datetime.strptime(d_str, "%Y-%m-%d").date()
                    parsed.append((d, won))
                except Exception:
                    continue

            series = []
            for sd in sample_dates:
                window_start = sd - timedelta(days=cfg["window"])
                wins = 0
                total = 0
                for d, won in parsed:
                    if window_start < d <= sd:
                        total += 1
                        if won:
                            wins += 1
                if total > 0:
                    series.append({
                        "date": sd.isoformat(),
                        "rate": round(wins / total * 100, 1),
                        "n":    total,
                    })
            per_metric[m] = series

        horizons_out[hz] = per_metric

    return _json_safe({
        "range":  range,
        "window_days": cfg["window"],
        "step_days":   cfg["step"],
        "horizons":    horizons_out,
    })


# ─── Analytics: momentum (streak + rolling + recent outcomes) ───────────────
def _resolved_date(pred: Dict[str, Any]) -> Optional[str]:
    """
    Best-effort 'when did this prediction resolve?' date. Prefers the
    target-hit date for target-based outcomes; otherwise derives from
    prediction date + horizon days for maturation; returns None if unresolved.
    """
    # If target was hit, the hit date is the natural resolution moment.
    if pred.get("target_hit") is True and pred.get("target_hit_date"):
        return pred.get("target_hit_date")

    # Otherwise, for matured predictions, approximate by prediction date
    # + horizon days. Good enough for bucketing.
    horizon_days = {
        "3 Day": 3, "1 Week": 7, "1 Month": 30,
        "1 Quarter": 90, "1 Year": 365,
    }.get(pred.get("horizon", ""), None)
    if not horizon_days:
        return None
    try:
        start = datetime.strptime(pred["date"], "%Y-%m-%d")
        return (start + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
    except Exception:
        return None


def _latest_checkpoint_score(pred: Dict[str, Any]) -> Optional[bool]:
    """Latest (longest interval) scored ✓/✗ from the predictions_table row."""
    for iv in ("365d", "180d", "90d", "60d", "30d", "14d", "7d", "3d", "1d"):
        s = pred.get(f"score_{iv}")
        if s in ("✓", "✗"):
            return s == "✓"
    return None


@app.get("/api/analytics/momentum")
def analytics_momentum(
    metric: str = "checkpoint",  # "checkpoint" | "expiration" | "target"
    rolling_window: int = 30,
    rolling_days: int = 90,
) -> Dict[str, Any]:
    """
    Momentum signals for the Dashboard:
      - current_streak: consecutive ✓ (or ✗) from the most recent resolutions
      - rolling_series: daily rolling win rate over the last `rolling_days` days
      - recent_results: the N most-recently-resolved predictions with outcomes
    """
    pa = _get_enriched_analytics()
    preds = pa.get("predictions_table", [])

    # Resolve each prediction under the chosen metric → list of (date, is_win)
    resolved = []
    for p in preds:
        is_hit = _is_hit(p, metric)
        if is_hit is None:
            continue
        rd = _resolved_date(p)
        if not rd:
            continue
        resolved.append({
            "prediction_id": p.get("prediction_id", ""),
            "horizon":       p.get("horizon", ""),
            "symbol":        p.get("symbol", ""),
            "date":          p.get("date", ""),      # prediction date
            "resolved_date": rd,
            "is_win":        bool(is_hit),
            "predicted_return": p.get("predicted_return", 0),
        })

    # Sort by resolved_date asc
    resolved.sort(key=lambda r: r["resolved_date"])

    # Current streak: count consecutive identical outcomes from the end.
    current_streak = 0
    streak_kind: Optional[str] = None  # "win" | "loss"
    if resolved:
        last = resolved[-1]["is_win"]
        streak_kind = "win" if last else "loss"
        for r in reversed(resolved):
            if r["is_win"] == last:
                current_streak += 1
            else:
                break

    # Rolling win rate — compute daily cumulative-window series over the
    # last `rolling_days` days using a `rolling_window` size. Returns an
    # array of {date, rate, sample} points.
    today = datetime.utcnow().date()
    series = []
    if resolved:
        earliest = datetime.strptime(
            resolved[0]["resolved_date"], "%Y-%m-%d"
        ).date()
        start_day = max(earliest, today - timedelta(days=rolling_days))

        # Pre-bucket outcomes by date for fast lookup
        outcomes_by_date: Dict[str, list] = {}
        for r in resolved:
            outcomes_by_date.setdefault(r["resolved_date"], []).append(r["is_win"])

        # Walk day-by-day
        d = start_day
        while d <= today:
            # Window = [d - rolling_window, d]
            window_start = d - timedelta(days=rolling_window)
            wins = 0
            total = 0
            cursor = window_start
            while cursor <= d:
                key = cursor.strftime("%Y-%m-%d")
                if key in outcomes_by_date:
                    for w in outcomes_by_date[key]:
                        total += 1
                        if w:
                            wins += 1
                cursor += timedelta(days=1)
            rate = (wins / total * 100) if total > 0 else None
            series.append({
                "date":   d.strftime("%Y-%m-%d"),
                "rate":   round(rate, 1) if rate is not None else None,
                "sample": total,
            })
            d += timedelta(days=1)

    # Recent results — last 10 resolved predictions
    recent = list(reversed(resolved[-10:])) if resolved else []
    recent_out = [
        {
            "prediction_id":    r["prediction_id"],
            "symbol":           r["symbol"],
            "horizon":          r["horizon"],
            "date":             r["date"],
            "resolved_date":    r["resolved_date"],
            "is_win":           r["is_win"],
            "predicted_return": r["predicted_return"],
        }
        for r in recent
    ]

    # Overall stats on the resolved pool
    total_resolved = len(resolved)
    overall_rate = (
        round(sum(1 for r in resolved if r["is_win"]) / total_resolved * 100, 1)
        if total_resolved > 0 else 0.0
    )

    return {
        "metric":          metric,
        "current_streak":  current_streak,
        "streak_kind":     streak_kind,
        "overall_rate":    overall_rate,
        "n_resolved":      total_resolved,
        "rolling_window":  rolling_window,
        "rolling_series":  series,
        "recent_results":  recent_out,
    }


# ─── Analytics: notable highlights (biggest wins & biggest misses) ──────────
@app.get("/api/analytics/highlights")
def analytics_highlights(limit: int = 3) -> Dict[str, Any]:
    """
    Ranks scored predictions by absolute price movement, splits into wins
    (direction correct) vs misses (direction wrong), and returns the top-N
    of each for a story-driven display on the Track Record page.

    Two design constraints driven by past UI noise:

    1. Restrict to SHORT horizons (3 Day, 1 Week). The "actual return"
       ranking is computed against the most-mature scored interval, which
       lands at most 7 days out for these horizons — a meaningful proxy
       for whether the call was right. For 1M/1Q/1Y predictions, the 1d
       move that gets stamped here is premature, so they're excluded.

    2. Dedupe by `prediction_id`. A single inference run emits five horizon
       rows in the log; if three of them all called DOWN and the stock
       dropped 19%, the leaderboard used to show three rows for what is
       essentially one notable call. We now keep the highest-confidence
       short-horizon row per run, and annotate it with `agreement_count`
       (how many OTHER horizons in the same run made the same directional
       call) so the user can see the model was consistent across horizons.
    """
    pa = _get_enriched_analytics()
    preds = pa.get("predictions_table", [])

    SHORT_HORIZONS = {"3 Day", "1 Week"}
    INTERVAL_KEYS = ("1d", "3d", "7d", "14d", "30d", "60d", "90d", "180d", "365d")

    # First pass: gather every horizon's predicted direction by run, so we
    # can compute the agreement annotation later. Includes ALL horizons —
    # not just the short ones — because a 3D win backed by 1M/1Q/1Y
    # agreement is more notable than a lone-horizon hit.
    runs_by_id: Dict[str, List[str]] = {}
    for p in preds:
        pid = p.get("prediction_id", "") or ""
        if not pid:
            continue
        direction = (p.get("direction", "") or "").upper()
        if direction in ("UP", "DOWN"):
            runs_by_id.setdefault(pid, []).append(direction)

    # Second pass: collect candidate rows. Only short horizons qualify;
    # within a prediction_id, keep the highest-confidence row.
    candidates_by_pid: Dict[str, Dict[str, Any]] = {}

    for p in preds:
        if p.get("horizon", "") not in SHORT_HORIZONS:
            continue

        # Pick the most-mature scored return available for this horizon.
        actual_return = None
        interval = None
        for iv in reversed(INTERVAL_KEYS):
            r = p.get(f"return_{iv}")
            s = p.get(f"score_{iv}")
            if r is not None and s in ("✓", "✗"):
                actual_return = float(r)  # already in percent
                interval = iv
                break
        if actual_return is None:
            continue

        pred_return = p.get("predicted_return", 0) or 0
        dir_sign = 1 if pred_return >= 0 else -1
        pnl_pct = actual_return * dir_sign  # correct-direction → positive
        confidence = float(p.get("confidence", 0) or 0)

        pid = p.get("prediction_id", "") or ""
        existing = candidates_by_pid.get(pid)
        # Keep the highest-confidence short horizon for this run.
        # Ties broken by larger |pnl_pct| (more dramatic call wins).
        if (
            existing is None
            or confidence > existing["_conf_for_dedupe"]
            or (
                confidence == existing["_conf_for_dedupe"]
                and abs(pnl_pct) > abs(existing["pnl_pct"])
            )
        ):
            candidates_by_pid[pid] = {
                "prediction_id":     pid,
                "symbol":            p.get("symbol", ""),
                "horizon":           p.get("horizon", ""),
                "date":              p.get("date", ""),
                "direction":         p.get("direction", ""),
                "predicted_return":  round(pred_return, 2),
                "actual_return":     round(actual_return, 2),
                "pnl_pct":           round(pnl_pct, 2),
                "interval_scored":   interval,
                "confidence":        round(confidence, 1),
                "_conf_for_dedupe":  confidence,
            }

    # Annotate each candidate with run-level agreement.
    for pid, entry in candidates_by_pid.items():
        run_dirs = runs_by_id.get(pid, [])
        rep_dir = (entry.get("direction", "") or "").upper()
        # `agreement_count` = how many OTHER horizons in the run agreed
        # on direction (the chosen horizon is excluded from its own count).
        agreement = sum(1 for d in run_dirs if d == rep_dir) - 1
        entry["agreement_count"] = max(0, agreement)
        entry["total_horizons"]  = len(run_dirs)
        entry.pop("_conf_for_dedupe", None)

    candidates = list(candidates_by_pid.values())

    wins   = [e for e in candidates if e["pnl_pct"] > 0]
    misses = [e for e in candidates if e["pnl_pct"] < 0]

    # Biggest wins: top by positive pnl
    wins.sort(key=lambda e: e["pnl_pct"], reverse=True)
    # Biggest misses: top by magnitude (pnl is negative, so sort asc)
    misses.sort(key=lambda e: e["pnl_pct"])

    return {
        "wins":         wins[:limit],
        "misses":       misses[:limit],
        "total_wins":   len(wins),
        "total_misses": len(misses),
    }


# ─── Analytics: aggregate feature importance ────────────────────────────────
@app.get("/api/analytics/feature-importance")
def analytics_feature_importance() -> Dict[str, Any]:
    """
    Aggregate how often each feature appears in a prediction's top-signal-
    drivers list. Reveals the model's personality: "Model leans on momentum
    60% of the time, never uses NFP surprise", etc.
    """
    pa = _get_enriched_analytics()
    preds = pa.get("predictions_table", [])

    counts: Dict[str, int] = {}
    total = len(preds)
    for p in preds:
        feats = p.get("top_features", []) or []
        for f in feats:
            if isinstance(f, str) and f:
                counts[f] = counts.get(f, 0) + 1

    items = [
        {
            "feature": name,
            "count":   c,
            "share":   round(c / total * 100, 1) if total > 0 else 0.0,
        }
        for name, c in counts.items()
    ]
    items.sort(key=lambda x: x["count"], reverse=True)

    return {
        "total_predictions":   total,
        "distinct_features":   len(items),
        "features":            items,
    }


# ─── Live market regime indicator ───────────────────────────────────────────
@app.get("/api/market/regime")
def market_regime() -> Dict[str, Any]:
    """
    Today's market backdrop — runs the same regime detector that tags
    individual predictions, but on fresh SPY data. Powers the "Right now"
    indicator on the Track Record page so users can see whether the
    market is currently Bull / Sideways / Bear, and contextualize the
    per-regime track record against the live state of the world.

    Cached 15 minutes since regimes evolve on weeks-to-months timescales.
    """
    cached = _MARKET_REGIME_CACHE.get("data")
    cached_ts = _MARKET_REGIME_CACHE.get("timestamp")
    if cached and cached_ts:
        age = (datetime.utcnow() - cached_ts).total_seconds()
        if age < _MARKET_REGIME_TTL_SECONDS:
            return cached

    try:
        from data_fetcher import fetch_stock_data
        from regime_detector import detect_regime
    except Exception as e:
        return {
            "error":  "import_failed",
            "label":  "Unknown",
            "detail": str(e),
        }

    try:
        spy_df = fetch_stock_data("SPY", period="2y")
    except Exception as e:
        return {
            "error":  "spy_fetch_failed",
            "label":  "Unknown",
            "detail": str(e),
        }
    if spy_df is None or len(spy_df) < 200:
        return {
            "error": "insufficient_data",
            "label": "Unknown",
            "hint":  "Need at least 200 trading days of SPY history.",
        }

    # VIX is a bonus signal — fetch attempt is allowed to fail without
    # invalidating the regime call.
    vix_close = None
    try:
        vix_df = fetch_stock_data("^VIX", period="2y")
        if vix_df is not None and "Close" in vix_df.columns:
            vix_close = vix_df["Close"]
    except Exception:
        pass

    try:
        # Pass SPY's own close as both `df` and `spy_close` so the SPY-vs-
        # 200MA signal still fires. detect_regime works fine even when
        # both are the same series — its checks are independent.
        result = detect_regime(
            spy_df, spy_close=spy_df["Close"], vix_close=vix_close,
        )
    except Exception as e:
        traceback.print_exc()
        return {
            "error":  "regime_detection_failed",
            "label":  "Unknown",
            "detail": str(e),
        }

    # Reshape `signals` (tuples) into JSON-safe dicts.
    signals_out = []
    for sig in result.get("signals", []):
        try:
            name, value, vote, label = sig
            num = float(value) if isinstance(value, (int, float)) else 0.0
            signals_out.append({
                "name":  str(name),
                "value": round(num, 4),
                "vote":  int(vote),
                "label": str(label),
            })
        except Exception:
            continue

    payload = {
        "label":      result["label"],
        "score":      result["score"],
        "score_norm": result["score_norm"],
        "signals":    signals_out,
        "underlying": "SPY",
        "as_of":      datetime.utcnow().isoformat() + "Z",
    }

    _MARKET_REGIME_CACHE["data"] = payload
    _MARKET_REGIME_CACHE["timestamp"] = datetime.utcnow()
    return _json_safe(payload)


# ─── Analytics: per-regime breakdown ────────────────────────────────────────
@app.get("/api/analytics/per-regime")
def analytics_per_regime() -> Dict[str, Any]:
    """
    Bull / Bear / Sideways breakdown using the same 3 honesty reads as the
    per-horizon endpoint. Shows which market backdrops the model handles best.
    """
    pa = _get_enriched_analytics()
    preds = pa.get("predictions_table", [])

    # by_regime tracks the metric counts; feature_counts_by_regime tracks
    # how often each feature appears in the top-3 drivers list of any
    # prediction made under each regime. Powers the "what does the model
    # lean on in Bull vs Bear" back-of-card story.
    by_regime: Dict[str, Dict[str, int]] = {}
    feature_counts_by_regime: Dict[str, Dict[str, int]] = {}

    for p in preds:
        reg = (p.get("regime", "") or "Unknown").strip() or "Unknown"
        d = by_regime.setdefault(reg, {
            "total":              0,
            "target_hit":         0,
            "target_resolved":    0,
            "checkpoint_wins":    0,
            "checkpoint_scored":  0,
            "expiration_wins":    0,
            "expiration_matured": 0,
        })
        d["total"] += 1

        # Tally feature appearances per regime.
        feats = p.get("top_features", []) or []
        if isinstance(feats, list):
            fc = feature_counts_by_regime.setdefault(reg, {})
            for f in feats:
                if isinstance(f, str) and f:
                    fc[f] = fc.get(f, 0) + 1

        # Target
        th = p.get("target_hit")
        if th is True:
            d["target_hit"] += 1
            d["target_resolved"] += 1
        elif th is False and p.get("horizon_expired"):
            d["target_resolved"] += 1

        # Checkpoint
        for iv in ("365d", "180d", "90d", "60d", "30d", "14d", "7d", "3d", "1d"):
            s = p.get(f"score_{iv}")
            if s in ("✓", "✗"):
                d["checkpoint_scored"] += 1
                if s == "✓":
                    d["checkpoint_wins"] += 1
                break

        # Expiration
        fr = p.get("final_result")
        if fr == "✓":
            d["expiration_wins"] += 1
            d["expiration_matured"] += 1
        elif fr == "✗":
            d["expiration_matured"] += 1

    # Stable order: bull → sideways → bear → unknown/other (alphabetical last).
    PREFERRED = ["Bull", "Sideways", "Bear"]
    rows = []
    for name in PREFERRED:
        if name in by_regime:
            rows.append((name, by_regime.pop(name)))
    for name in sorted(by_regime.keys()):
        rows.append((name, by_regime[name]))

    regimes_out = []
    for name, d in rows:
        # Top features for this regime — sorted by appearance count, top 6
        # so the back of the card has just enough for a real story without
        # turning into another full leaderboard.
        feat_counts = feature_counts_by_regime.get(name, {})
        # Total = sum of appearances; share is per-appearance, not per-pred,
        # since a single prediction surfaces multiple features in its top-3.
        feat_total = sum(feat_counts.values())
        top_features = []
        for fname, fcount in sorted(
            feat_counts.items(), key=lambda t: t[1], reverse=True
        )[:6]:
            top_features.append({
                "feature": fname,
                "count":   fcount,
                "share":   round(fcount / feat_total * 100, 1) if feat_total else 0.0,
            })

        regimes_out.append({
            "name":  name,
            "total": d["total"],
            "target": {
                "rate":           round(d["target_hit"] / d["target_resolved"] * 100, 1)
                                   if d["target_resolved"] else 0,
                "hit_count":      d["target_hit"],
                "resolved_count": d["target_resolved"],
                "has_data":       d["target_resolved"] > 0,
            },
            "checkpoint": {
                "rate":          round(d["checkpoint_wins"] / d["checkpoint_scored"] * 100, 1)
                                  if d["checkpoint_scored"] else 0,
                "correct_count": d["checkpoint_wins"],
                "scored_count":  d["checkpoint_scored"],
                "has_data":      d["checkpoint_scored"] > 0,
            },
            "expiration": {
                "rate":          round(d["expiration_wins"] / d["expiration_matured"] * 100, 1)
                                  if d["expiration_matured"] else 0,
                "correct_count": d["expiration_wins"],
                "matured_count": d["expiration_matured"],
                "has_data":      d["expiration_matured"] > 0,
            },
            "top_features": top_features,
        })

    return {"regimes": regimes_out}


# ─── Analytics: per-symbol breakdown ────────────────────────────────────────
@app.get("/api/analytics/per-symbol")
def analytics_per_symbol() -> Dict[str, Any]:
    """
    Ticker-level summary: total calls, avg confidence, and all three honesty
    reads (target / checkpoint / expiration) per symbol. Lets power users see
    which tickers the model calls best.
    """
    pa = _get_enriched_analytics()
    preds = pa.get("predictions_table", [])

    by_symbol: Dict[str, Dict[str, Any]] = {}
    for p in preds:
        sym = p.get("symbol", "")
        if not sym:
            continue
        d = by_symbol.setdefault(sym, {
            "total":              0,
            "conf_sum":           0.0,
            "conf_n":             0,
            "target_hit":         0,
            "target_resolved":    0,
            "checkpoint_wins":    0,
            "checkpoint_scored":  0,
            "expiration_wins":    0,
            "expiration_matured": 0,
        })
        d["total"] += 1

        conf = p.get("confidence", 0) or 0
        if conf > 0:
            d["conf_sum"] += float(conf)
            d["conf_n"] += 1

        # Target (generous)
        th = p.get("target_hit")
        if th is True:
            d["target_hit"] += 1
            d["target_resolved"] += 1
        elif th is False and p.get("horizon_expired"):
            d["target_resolved"] += 1

        # Checkpoint (latest scored interval)
        for iv in ("365d", "180d", "90d", "60d", "30d", "14d", "7d", "3d", "1d"):
            s = p.get(f"score_{iv}")
            if s in ("✓", "✗"):
                d["checkpoint_scored"] += 1
                if s == "✓":
                    d["checkpoint_wins"] += 1
                break

        # Expiration (strict)
        fr = p.get("final_result")
        if fr == "✓":
            d["expiration_wins"] += 1
            d["expiration_matured"] += 1
        elif fr == "✗":
            d["expiration_matured"] += 1

    rows = []
    for sym, d in by_symbol.items():
        rows.append({
            "symbol":         sym,
            "total":          d["total"],
            "avg_confidence": round(d["conf_sum"] / d["conf_n"], 1) if d["conf_n"] else 0,
            "target": {
                "rate":           round(d["target_hit"] / d["target_resolved"] * 100, 1)
                                   if d["target_resolved"] else 0,
                "hit_count":      d["target_hit"],
                "resolved_count": d["target_resolved"],
                "has_data":       d["target_resolved"] > 0,
            },
            "checkpoint": {
                "rate":          round(d["checkpoint_wins"] / d["checkpoint_scored"] * 100, 1)
                                  if d["checkpoint_scored"] else 0,
                "correct_count": d["checkpoint_wins"],
                "scored_count":  d["checkpoint_scored"],
                "has_data":      d["checkpoint_scored"] > 0,
            },
            "expiration": {
                "rate":          round(d["expiration_wins"] / d["expiration_matured"] * 100, 1)
                                  if d["expiration_matured"] else 0,
                "correct_count": d["expiration_wins"],
                "matured_count": d["expiration_matured"],
                "has_data":      d["expiration_matured"] > 0,
            },
        })

    # Default sort: total calls desc. Frontend can re-sort client-side.
    rows.sort(key=lambda r: r["total"], reverse=True)

    return {"symbols": rows}


# ─── Analytics: confidence calibration ──────────────────────────────────────
_CALIBRATION_BUCKETS = [
    (50, 60),
    (60, 70),
    (70, 80),
    (80, 90),
    (90, 100),
]


def _is_hit(pred: Dict[str, Any], metric: str) -> Optional[bool]:
    """
    Is this prediction a "hit" under `metric`? Returns None when the
    prediction isn't eligible for this metric (e.g. horizon not matured
    for expiration).
    """
    if metric == "target":
        th = pred.get("target_hit")
        if th is True:
            return True
        if th is False and pred.get("horizon_expired"):
            return False
        return None

    if metric == "checkpoint":
        for iv in ("365d", "180d", "90d", "60d", "30d", "14d", "7d", "3d", "1d"):
            s = pred.get(f"score_{iv}")
            if s in ("✓", "✗"):
                return s == "✓"
        return None

    # expiration
    fr = pred.get("final_result")
    if fr == "✓":
        return True
    if fr == "✗":
        return False
    return None


@app.get("/api/analytics/calibration")
def analytics_calibration(
    metric: str = "expiration",      # "target" | "checkpoint" | "expiration"
    horizon: Optional[str] = None,   # e.g. "3 Day" — None = all horizons
) -> Dict[str, Any]:
    """
    Reliability-diagram data: predicted-confidence buckets vs actual hit
    rate in each bucket. A perfectly calibrated model has actual rate ≈
    bucket midpoint (the diagonal); bars above the diagonal = underconfident,
    below = overconfident.

    Returns empty-but-shaped data if no predictions are eligible for the
    chosen metric/horizon, so the UI can render the axes without blowing up.
    """
    pa = _get_enriched_analytics()
    preds = pa.get("predictions_table", [])

    # Optional horizon narrowing
    if horizon:
        preds = [p for p in preds if p.get("horizon", "") == horizon]

    # Keep only predictions that are eligible (i.e. have a resolved outcome
    # under this metric) and within our visualized confidence range (>=50%).
    eligible = []
    for p in preds:
        hit = _is_hit(p, metric)
        if hit is None:
            continue
        conf = p.get("confidence", 0) or 0
        if conf < 50:
            continue
        eligible.append({"confidence": float(conf), "hit": bool(hit)})

    buckets_out = []
    total_hits = 0
    total_scored = 0
    weighted_conf_sum = 0.0
    for lo, hi in _CALIBRATION_BUCKETS:
        # Upper edge inclusive for the last bucket so 100% ends up somewhere.
        in_bucket = [
            p
            for p in eligible
            if (lo <= p["confidence"] < hi)
            or (hi == 100 and p["confidence"] == 100)
        ]
        n = len(in_bucket)
        hits = sum(1 for p in in_bucket if p["hit"])
        rate = (hits / n * 100) if n > 0 else 0.0
        avg_conf = (sum(p["confidence"] for p in in_bucket) / n) if n > 0 else (lo + hi) / 2

        buckets_out.append({
            "min_conf":    lo,
            "max_conf":    hi,
            "midpoint":    (lo + hi) / 2,
            "avg_conf":    round(avg_conf, 1),
            "actual_rate": round(rate, 1),
            "hits":        hits,
            "sample_size": n,
            "has_data":    n > 0,
        })
        total_hits += hits
        total_scored += n
        weighted_conf_sum += sum(p["confidence"] for p in in_bucket)

    overall_rate = (total_hits / total_scored * 100) if total_scored > 0 else 0.0
    avg_conf_weighted = (
        weighted_conf_sum / total_scored if total_scored > 0 else 0.0
    )

    # Mean absolute calibration error — weighted by sample size.
    if total_scored > 0:
        mae = sum(
            b["sample_size"] * abs(b["actual_rate"] - b["avg_conf"])
            for b in buckets_out
            if b["has_data"]
        ) / total_scored
    else:
        mae = 0.0

    # Diagnostic string for the UI — simple bucketing of MAE.
    if total_scored == 0:
        verdict = "not_enough_data"
    elif mae < 5:
        verdict = "well_calibrated"
    elif mae < 10:
        verdict = "slightly_miscalibrated"
    else:
        verdict = "miscalibrated"

    # Directional bias — avg_actual minus avg_conf. Negative ⇒ overconfident
    # (claimed higher than delivered), positive ⇒ underconfident.
    bias = overall_rate - avg_conf_weighted

    return {
        "metric":            metric,
        "buckets":           buckets_out,
        "overall_rate":      round(overall_rate, 1),
        "avg_confidence":    round(avg_conf_weighted, 1),
        "mean_abs_error":    round(mae, 2),
        "bias":              round(bias, 2),
        "verdict":           verdict,
        "n_scored":          total_scored,
    }


# ─── Analytics: simulated portfolio ──────────────────────────────────────────
_INTERVAL_DAYS = {
    "1d": 1, "3d": 3, "7d": 7, "14d": 14, "30d": 30,
    "60d": 60, "90d": 90, "180d": 180, "365d": 365,
}
_STARTING_CAPITAL = 10_000.0

# Benchmark cache — SPY daily history is hit against yfinance, which is slow
# and flaky; cache aggressively and refresh hourly.
_BENCHMARK_CACHE: Dict[str, Any] = {}
_BENCHMARK_TTL_SECONDS = 3600  # 1 hour


def _fetch_spy_daily():
    """
    Return a DataFrame of SPY's daily close prices, cached hourly.
    Returns None on any failure so callers degrade gracefully.
    """
    cached_data = _BENCHMARK_CACHE.get("spy")
    cached_ts = _BENCHMARK_CACHE.get("spy_ts")
    if cached_data is not None and cached_ts is not None:
        if (datetime.utcnow() - cached_ts).total_seconds() < _BENCHMARK_TTL_SECONDS:
            return cached_data

    try:
        from data_fetcher import fetch_stock_data
        df = fetch_stock_data("SPY", period="2y")
        _BENCHMARK_CACHE["spy"] = df
        _BENCHMARK_CACHE["spy_ts"] = datetime.utcnow()
        return df
    except Exception:
        return None


def _build_benchmark_curve(
    strategy_curve,
    starting_capital: float,
):
    """
    Build a SPY buy-and-hold equity curve sampled at the strategy's own
    curve dates. That gives us two apples-to-apples lines — same dates,
    same starting capital, same endpoint in time — so the portfolio chart
    can answer "is this better than just buying the index?" at a glance.

    Returns (curve, stats) or (None, None) if SPY data is unavailable.
    """
    if not strategy_curve:
        return None, None

    spy = _fetch_spy_daily()
    if spy is None or spy.empty:
        return None, None

    # Normalize SPY index to plain YYYY-MM-DD strings for lookup
    import pandas as pd
    spy_series = spy["Close"].copy()
    spy_series.index = pd.to_datetime(spy_series.index).strftime("%Y-%m-%d")
    # Drop duplicates in case yfinance returns any
    spy_series = spy_series[~spy_series.index.duplicated(keep="last")]

    def _nearest_price(date_str: str):
        """SPY close on `date_str`, or the most recent prior trading day."""
        if date_str in spy_series.index:
            return float(spy_series.loc[date_str])
        prior = spy_series.index[spy_series.index <= date_str]
        if len(prior) == 0:
            return None
        return float(spy_series.loc[prior[-1]])

    # Anchor SPY to the first strategy date
    first_date = strategy_curve[0]["date"]
    anchor_price = _nearest_price(first_date)
    if anchor_price is None or anchor_price == 0:
        return None, None

    curve = []
    for point in strategy_curve:
        d = point["date"]
        price = _nearest_price(d)
        if price is None:
            continue
        equity = starting_capital * (price / anchor_price)
        curve.append({"date": d, "equity": round(equity, 2)})

    if not curve:
        return None, None

    final = curve[-1]["equity"]
    total_return_pct = (final / starting_capital - 1) * 100

    # Peak drawdown along the way
    running_max = curve[0]["equity"]
    max_dd = 0.0
    for p in curve:
        running_max = max(running_max, p["equity"])
        dd = ((p["equity"] - running_max) / running_max * 100) if running_max > 0 else 0
        max_dd = min(max_dd, dd)

    stats = {
        "final_value":      round(final, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_dd, 2),
    }
    return curve, stats


def _build_curve_and_stats(trades):
    """
    Walk a list of trades (with exit_date + pnl_pct) and produce an equity
    curve + summary stats. Each trade is sized at 5% of CURRENT equity.
    """
    if not trades:
        return {
            "equity_curve":       [],
            "final_value":        _STARTING_CAPITAL,
            "total_return_pct":   0.0,
            "n_trades":           0,
            "wins":               0,
            "losses":             0,
            "max_drawdown_pct":   0.0,
            "targets_hit":        0,
        }

    equity = _STARTING_CAPITAL
    curve = [{"date": trades[0]["date"], "equity": round(equity, 2)}]
    wins = losses = targets_hit = 0
    running_max = equity
    max_dd = 0.0

    for t in trades:
        position = equity * 0.05
        pnl = position * t["pnl_pct"]
        equity += pnl
        curve.append({"date": t["date"], "equity": round(equity, 2)})
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
        running_max = max(running_max, equity)
        dd = ((equity - running_max) / running_max * 100) if running_max > 0 else 0
        max_dd = min(max_dd, dd)
        if t.get("realized") == "target":
            targets_hit += 1

    total_return = (equity / _STARTING_CAPITAL - 1) * 100
    return {
        "equity_curve":       curve,
        "final_value":        round(equity, 2),
        "total_return_pct":   round(total_return, 2),
        "n_trades":           len(trades),
        "wins":               wins,
        "losses":             losses,
        "max_drawdown_pct":   round(max_dd, 2),
        "targets_hit":        targets_hit,
    }


@app.get("/api/analytics/simulated-portfolio")
def simulated_portfolio() -> Dict[str, Any]:
    """
    Build TWO parallel equity curves from scored predictions:
      - hold:        every trade held to its scheduled checkpoint
      - take_profit: exit at target_hit_date when target was hit,
                     else fall back to hold behavior

    Used by the Track Record "If you followed every signal" chart.
    """
    pa = _get_enriched_analytics()
    scored_preds = [
        p for p in pa.get("predictions_table", [])
        if p.get("final_result") in ("✓", "✗")
    ]

    if len(scored_preds) < 3:
        return {
            "starting_capital": _STARTING_CAPITAL,
            "hold":             _build_curve_and_stats([]),
            "take_profit":      _build_curve_and_stats([]),
            "note":             "Not enough matured predictions yet — at least 3 required.",
        }

    trades_hold = []
    trades_tp = []

    for pred in scored_preds:
        actual_ret = None
        scored_interval = None
        for interval in ["365d", "180d", "90d", "60d", "30d", "14d", "7d", "3d", "1d"]:
            r = pred.get(f"return_{interval}")
            if r is not None:
                actual_ret = r / 100.0
                scored_interval = interval
                break
        if actual_ret is None:
            continue

        try:
            pred_date = datetime.strptime(pred["date"], "%Y-%m-%d")
        except Exception:
            continue
        exit_date = pred_date + timedelta(days=_INTERVAL_DAYS[scored_interval])

        pred_ret = pred.get("predicted_return", 0) or 0
        dir_sign = 1 if pred_ret >= 0 else -1
        hold_pnl_pct = actual_ret * dir_sign

        trades_hold.append({
            "date":    exit_date.strftime("%Y-%m-%d"),
            "pnl_pct": hold_pnl_pct,
        })

        # Take-profit: exit at target-hit date if target was hit, else hold
        th = pred.get("target_hit")
        th_date_str = pred.get("target_hit_date")
        if th is True and th_date_str:
            try:
                tp_exit = datetime.strptime(th_date_str, "%Y-%m-%d")
            except Exception:
                tp_exit = exit_date
            tp_pnl_pct = abs(pred_ret / 100.0)
            realized = "target"
        else:
            tp_exit = exit_date
            tp_pnl_pct = hold_pnl_pct
            realized = "expiration"

        trades_tp.append({
            "date":     tp_exit.strftime("%Y-%m-%d"),
            "pnl_pct":  tp_pnl_pct,
            "realized": realized,
        })

    trades_hold.sort(key=lambda t: t["date"])
    trades_tp.sort(key=lambda t: t["date"])

    hold = _build_curve_and_stats(trades_hold)
    take_profit = _build_curve_and_stats(trades_tp)

    # SPY benchmark sampled at the take-profit curve's dates (longer of the
    # two for most portfolios). Frontend uses the same curve for both cards,
    # since SPY is a single buy-and-hold reference.
    benchmark_curve, benchmark_stats = _build_benchmark_curve(
        take_profit["equity_curve"] or hold["equity_curve"],
        _STARTING_CAPITAL,
    )

    payload = {
        "starting_capital": _STARTING_CAPITAL,
        "hold":             hold,
        "take_profit":      take_profit,
    }
    if benchmark_curve is not None:
        payload["benchmark"] = {
            "name":         "SPY",
            "label":        "SPY buy-and-hold",
            "equity_curve": benchmark_curve,
            **(benchmark_stats or {}),
        }
    return payload


# ─── Live inference (Prediqt page) ──────────────────────────────────────────
# Per-symbol cache — fresh inference is ~0.5-1s warm, but rapid re-requests
# (the page fetching on mount + the user hitting refresh) deserve instant
# answers. TTL is short so the cached prediction never goes stale within a
# user session.
_INFERENCE_CACHE: Dict[str, Any] = {}
_INFERENCE_TTL_SECONDS = 300  # 5 minutes

# ─── Auto-retrain policy ────────────────────────────────────────────────────
# Models on disk get automatically upgraded in the background when any of
# these are stale. Users see the existing prediction now and an upgraded one
# on their next visit — no manual "retrain" button to expose to the public.
EXPECTED_TRAIN_DATA_PERIOD = "10y"   # bumped from 7y; legacy models auto-upgrade
MODEL_MAX_AGE_DAYS         = 3       # 3-day cadence catches earnings + recent
                                     # regime shifts without thrashing the
                                     # ledger with too many model versions
EXPECTED_HORIZONS_FOR_FRESH = {"3 Day", "1 Week", "1 Month", "1 Quarter", "1 Year"}


def _check_model_freshness(predictor) -> Tuple[bool, str]:
    """
    Decide whether the loaded model deserves an automatic retrain. Returns
    (needs_refresh, human_readable_reason). Reasons are stamped into the
    response so the UI can surface a subtle "upgrading" indicator.

    Order matters: schema mismatch is the most decisive signal (legacy
    models trained on a different period), so it's checked first.
    """
    saved_period = getattr(predictor, "train_data_period", None)
    if saved_period != EXPECTED_TRAIN_DATA_PERIOD:
        return True, (
            f"Trained on {saved_period or 'unknown'} of history; "
            f"current schema is {EXPECTED_TRAIN_DATA_PERIOD}"
        )

    trained_at_iso = getattr(predictor, "trained_at_iso", None)
    if trained_at_iso:
        try:
            trained_at = datetime.fromisoformat(trained_at_iso)
            age_days = (datetime.utcnow() - trained_at).total_seconds() / 86400
            if age_days > MODEL_MAX_AGE_DAYS:
                return True, f"Model is {int(age_days)} days old"
        except (ValueError, TypeError):
            pass  # Bad timestamp — don't force a retrain on parse error alone

    # Skipped horizons on a "should-be-mature" model means something
    # failed during the last training run. Try again — maybe yfinance was
    # flaky, or the older 7y window genuinely couldn't fit 1Y.
    horizons_with_models = set(getattr(predictor, "l1_members", {}).keys())
    missing = EXPECTED_HORIZONS_FOR_FRESH - horizons_with_models
    if missing:
        return True, (
            f"Missing horizon{'s' if len(missing) > 1 else ''}: "
            f"{', '.join(sorted(missing))}"
        )

    return False, ""


def _dispatch_auto_retrain(sym: str, reason: str) -> bool:
    """
    Fire-and-forget: queue a background retrain for `sym`. Idempotent —
    if a job is already running for this ticker we leave it alone.
    Returns True if a new job was launched, False if one was already
    running or dispatch failed.

    Differs from start_training() in that we deliberately bypass the
    "already on disk" guard — the whole point is to upgrade an existing
    model in place. The training worker overwrites the joblib on save,
    so the model file gets replaced atomically once training finishes.
    """
    _purge_stale_jobs()

    existing = _get_job(sym)
    if existing and existing.get("stage") in (
        _STAGE_QUEUED, _STAGE_FETCH, _STAGE_TRAIN, _STAGE_SAVE,
    ):
        # Already running — don't double-queue. The reason is informational
        # only; the existing job will produce a fresh model regardless of
        # which condition triggered it.
        return False

    try:
        t = threading.Thread(
            target=_train_symbol_worker,
            args=(sym,),
            name=f"auto-retrain-{sym}",
            daemon=True,
        )
        _set_job(
            sym,
            stage=_STAGE_QUEUED,
            started_at=datetime.utcnow(),
            error=None,
            progress=0.0,
            auto_retrain=True,
            auto_retrain_reason=reason,
        )
        t.start()
        print(f"[auto-retrain {sym}] dispatched: {reason}")
        return True
    except Exception as e:
        print(f"[auto-retrain {sym}] dispatch failed: {e}")
        return False


def _models_dir() -> str:
    """
    Resolve the directory where trained per-symbol models live. Matches the
    path used by StockPredictor._model_path() (see model.py:427).
    """
    # api/main.py lives under Stock Bot/api/; models live under Stock Bot/.models/
    api_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(api_dir, "..", ".models"))


def _list_available_symbols() -> list:
    """
    Scan the models directory for trained per-symbol joblib files. Looks for
    the `{SYMBOL}_model.joblib` naming convention used by StockPredictor.
    Returns symbol strings sorted alphabetically. Empty list on error.
    """
    try:
        models_dir = _models_dir()
        if not os.path.isdir(models_dir):
            return []
        out = []
        for name in os.listdir(models_dir):
            if name.endswith("_model.joblib"):
                # Strip the "_model.joblib" suffix (13 chars).
                out.append(name[:-13].upper())
            elif name.endswith(".pkl"):
                # Legacy format fallback
                out.append(name[:-4].upper())
        return sorted(set(out))
    except Exception:
        return []


@app.get("/api/symbols")
def list_symbols() -> Dict[str, Any]:
    """
    All symbols with a trained model on disk. Powers the /prediqt picker.
    Annotated with whether each symbol has at least one prediction in the
    log, and the most recent prediction date for it, so the picker can
    surface "active" tickers.
    """
    symbols = _list_available_symbols()
    if not symbols:
        return {"symbols": []}

    pa = _get_enriched_analytics()
    preds_by_sym: Dict[str, Any] = {}
    for p in pa.get("predictions_table", []):
        sym = (p.get("symbol", "") or "").upper()
        if not sym:
            continue
        entry = preds_by_sym.setdefault(
            sym, {"count": 0, "latest_date": ""}
        )
        entry["count"] += 1
        d = p.get("date", "") or ""
        if d > entry["latest_date"]:
            entry["latest_date"] = d

    out = []
    for sym in symbols:
        info = preds_by_sym.get(sym, {"count": 0, "latest_date": ""})
        out.append({
            "symbol":       sym,
            "has_log":      info["count"] > 0,
            "log_count":    info["count"],
            "latest_date":  info["latest_date"] or None,
        })

    # Sort: most recently-predicted first, then alphabetical for ties.
    out.sort(key=lambda s: (s["latest_date"] or "", s["symbol"]), reverse=True)
    return {"symbols": out}


@app.get("/api/predict/{symbol}")
def predict_symbol(
    symbol: str,
    force: bool = False,
    horizon: Optional[str] = None,
    cu: CurrentUser = Depends(require_verified_email),  # noqa: B008
) -> Dict[str, Any]:
    """
    Fresh model inference for a single symbol — loads the pre-trained
    predictor, fetches the latest OHLC + market context, and runs
    .predict() without logging to the official prediction log.

    Cached per-symbol for 5 minutes to absorb rapid re-requests. Pass
    `?force=true` to bypass the cache (used by the "Run again" button so
    users can see live progress + a fresh price each click).
    """
    import traceback
    # Fresh-resolve the public framework version at the top of every
    # request so log_prediction_v2 (when we get there) and the response
    # payload both stamp the current truth from model_versions.json,
    # regardless of when the predictor instance was last loaded into
    # memory. Pre-fix, a long-running uvicorn cached predictors with
    # `model_version` attribute that didn't exist (defaulted to "1.0")
    # and every prediction got that stamp even after the file moved
    # to "4.0".
    from prediction_logger_v2 import get_current_model_version as _gmv

    sym = (symbol or "").upper().strip()
    if not sym:
        return {"error": "invalid_symbol"}

    # Cache hit? Skip when caller explicitly forces a refresh.
    if not force:
        cached = _INFERENCE_CACHE.get(sym)
        if cached is not None:
            data, ts = cached
            if (datetime.utcnow() - ts).total_seconds() < _INFERENCE_TTL_SECONDS:
                return data

    # ── Anchor § 8.1 same-day dedupe — short-circuit ─────────────────────
    # Before we burn cycles loading the model + running inference, check
    # whether this user already has a prediction for (symbol, horizon)
    # logged today (UTC). If yes, return a `dedupe_hit` payload immediately
    # and let the frontend route to the existing record. This both saves
    # work and gives users a clearer mental model: "you've already asked
    # this — here it is" rather than silently producing a fresh-looking
    # response that secretly maps to the same row.
    #
    # Bypassed when ?force=true so the "Run again" affordance still re-runs
    # the model (the existing row is then updated in-place by the dedupe
    # path inside log_prediction_v2).
    if not force and horizon:
        try:
            from db import find_todays_prediction, set_request_user
            set_request_user(cu.id)
            existing_pred_id = find_todays_prediction(
                symbol=sym,
                horizon_code=horizon,
                user_id=cu.id,
            )
            if existing_pred_id:
                # Look up the existing row's created_at + a short symbol for
                # the toast message. Best-effort — if the lookup fails, we
                # still surface the dedupe hit, just without the timestamp.
                asked_at_iso: Optional[str] = None
                try:
                    from db import get_prediction_detail as _detail
                    sb_row = _detail(existing_pred_id)
                    if sb_row:
                        asked_at_iso = sb_row.get("created_at")
                except Exception:
                    asked_at_iso = None

                return {
                    "dedupe_hit":            True,
                    "existing_prediction_id": existing_pred_id,
                    "asked_at":              asked_at_iso,
                    "symbol":                sym,
                    "horizon":               horizon,
                    "reason": (
                        "Already asked today (UTC) — anchor § 8.1 same-day "
                        "dedupe. Frontend routes to the existing record."
                    ),
                }
        except Exception as exc:
            # Lookup failure is non-fatal: fall through to the normal
            # predict path. Worst case we run inference and log_prediction_v2
            # dedupes downstream, which preserves correctness.
            print(f"[predict {sym}] dedupe pre-check failed: {exc}")

    # Top-level safety net. Anything that escapes the inner blocks below
    # gets caught here, the traceback is printed to uvicorn's stdout, and
    # we return a structured JSON error so the frontend can show the real
    # cause instead of a bare 500.
    try:
        # Load predictor (pre-trained). Use StockPredictor directly since
        # model.py's load_predictor() helper is legacy — the real load path is
        # predictor.load_model() which reads {SYMBOL}_model.joblib from .models/.
        try:
            from model import StockPredictor
        except Exception as e:
            return {"error": "import_failed", "detail": str(e)}

        # Quick existence check so we fail fast + cheap when the file isn't there.
        _model_file = os.path.join(_models_dir(), f"{sym}_model.joblib")
        if not os.path.exists(_model_file):
            return {
                "error": "model_not_found",
                "symbol": sym,
                "hint": "No pre-trained model on disk for this ticker. Train it first via the batch pipeline.",
            }

        predictor = StockPredictor(sym)
        # Bypass the 24h staleness check inside load_model — we serve the
        # model regardless of age. The auto-retrain detector below decides
        # whether to dispatch a background upgrade.
        try:
            loaded = predictor.load_model(max_age_hours=24 * 365)
        except Exception as e:
            print(f"[predict {sym}] load_model crashed:")
            traceback.print_exc()
            return {
                "error": "model_load_failed",
                "symbol": sym,
                "detail": f"{type(e).__name__}: {e}",
            }
        if not loaded:
            # load_model() distinguishes WHY it failed via
            # predictor.load_failure_reason. We translate "stale weights
            # need a retrain" cases into the same `model_not_found`
            # response shape that the frontend already handles — that
            # auto-dispatches training and shows the progress UI without
            # a separate code path. The hint string is observability-only
            # (Render logs, debug tooling) so the backend stays honest
            # about what happened.
            reason = getattr(predictor, "load_failure_reason", None)
            if reason in ("training_version_mismatch", "stale", "corrupted"):
                # Kick the retrain off proactively so by the time the
                # frontend's TrainingState mounts and polls /api/train-status,
                # the worker is already running. _dispatch_auto_retrain is
                # idempotent — if start_training() fires a moment later,
                # it'll see the existing job and no-op.
                _dispatch_auto_retrain(sym, reason)
                return {
                    "error": "model_not_found",
                    "symbol": sym,
                    "hint": (
                        f"Model on disk needs to be retrained ({reason}). "
                        "Auto-training dispatched."
                    ),
                }
            # file_missing or any other unhandled reason — surface the
            # original error so the upstream existence-check or future
            # diagnostics still trip cleanly.
            return {
                "error": "model_load_failed",
                "symbol": sym,
                "hint": (
                    f"Model file exists but couldn't be hydrated "
                    f"(reason={reason or 'unknown'}). Check the meta JSON "
                    "or retrain."
                ),
            }

        # ── Auto-retrain detector ──────────────────────────────────────
        # We serve the current prediction immediately; if the model is
        # stale (legacy schema, too old, or missing horizons), we kick off
        # a background retrain. The user's NEXT visit gets the upgraded
        # model. The auto_refresh dict gets stamped into the response so
        # the UI can surface a subtle "upgrading" indicator without
        # blocking the user on a 2-4 minute training run.
        auto_refresh: Dict[str, Any] = {"pending": False, "reason": ""}
        try:
            needs_refresh, refresh_reason = _check_model_freshness(predictor)
            if needs_refresh:
                dispatched = _dispatch_auto_retrain(sym, refresh_reason)
                # Even if dispatch found an existing job, surface the
                # pending state so the UI shows the indicator until the
                # next request lands a fresh model.
                auto_refresh = {
                    "pending": True,
                    "reason": refresh_reason,
                    "dispatched": dispatched,
                }
        except Exception as e:
            # Detector / dispatcher must never break inference. Print and
            # carry on serving the current model.
            print(f"[predict {sym}] auto-retrain check failed: {e}")

        # Fetch fresh data
        try:
            from data_fetcher import (
                fetch_stock_data,
                fetch_market_context,
                TickerNotFoundError,
                TickerDataUnavailableError,
            )
        except Exception as e:
            return {"error": "import_failed", "detail": str(e)}

        try:
            df = fetch_stock_data(sym, period="2y")
        except TickerNotFoundError as e:
            # Yahoo doesn't recognise the symbol → real "check spelling"
            # case. Don't print a traceback; this isn't a backend bug.
            # Forward Yahoo Search suggestions ('did you mean CROX?')
            # so the UI can render dynamic chips instead of the same
            # static fallback list every time.
            return {
                "error": "ticker_not_found",
                "symbol": sym,
                "detail": str(e),
                "suggestions": list(getattr(e, "suggestions", []) or []),
            }
        except TickerDataUnavailableError as e:
            # Symbol is valid, Yahoo's feed is just being flaky right
            # now. Retryable on the user's side — surface that.
            return {
                "error": "data_temporarily_unavailable",
                "symbol": sym,
                "detail": str(e),
                "retryable": True,
            }
        except Exception as e:
            print(f"[predict {sym}] fetch_stock_data crashed:")
            traceback.print_exc()
            return {"error": "data_fetch_failed", "symbol": sym, "detail": str(e)}

        try:
            market_ctx = fetch_market_context()
        except Exception:
            market_ctx = None  # predict() tolerates None

        # Run inference
        try:
            predictions = predictor.predict(df, market_ctx=market_ctx)
        except Exception as e:
            print(f"[predict {sym}] predict() crashed:")
            traceback.print_exc()
            return {
                "error": "inference_failed",
                "symbol": sym,
                "detail": f"{type(e).__name__}: {e}",
            }

        # ── Contrarian haircut MUST run before log_prediction_v2 ────────
        # Methodology § 4.3 contrarian guardrail. Compute horizon-adjusted
        # divergence vs analyst consensus and apply the 0.85× confidence
        # haircut when the model is strongly contrarian + directionally
        # opposed. Critically: this has to mutate predictions[h]["confidence"]
        # *before* log_prediction_v2 runs, otherwise saved Supabase rows
        # carry the pre-haircut value while the live response carries
        # the post-haircut value — same prediction, two different numbers
        # depending on which page you look at. (This was a real bug from
        # 2026-04-29 through 2026-04-30; receipt and live pages diverged
        # silently on every contrarian call.)
        consensus_target = None
        try:
            cached = _QUOTE_CACHE.get(sym)
            if cached is not None:
                payload_q, ts_q = cached
                if (datetime.utcnow() - ts_q).total_seconds() < _QUOTE_TTL_SECONDS:
                    consensus_target = payload_q.get("analyst_target_mean")
            if consensus_target is None:
                from data_fetcher import fetch_fundamentals as _fetch_fund
                _fund = _fetch_fund(sym) or {}
                consensus_target = _fund.get("target_mean")
        except Exception as exc:  # noqa: BLE001
            print(f"[predict {sym}] consensus target fetch failed: {exc}")
            consensus_target = None

        # Snapshot current_price for the consensus loop. The downstream
        # block re-derives it from df, but the haircut needs it now.
        try:
            _current_price_snapshot = float(df["Close"].iloc[-1])
        except Exception:
            _current_price_snapshot = None

        try:
            from consensus_check import (
                evaluate as _consensus_evaluate,
                haircut_confidence as _consensus_haircut,
            )
            for h, hdata in predictions.items():
                if not isinstance(hdata, dict) or hdata.get("skipped"):
                    continue
                # Use the module-level _HORIZON_DAYS constant. The local
                # `HORIZON_DAYS = {...}` assignment further down in this
                # function shadowed the (then non-existent) global lookup
                # here as a local-not-yet-assigned read, raising
                # UnboundLocalError. Routing both reads through the
                # module constant fixes the scope hazard.
                horizon_days = _HORIZON_DAYS.get(h)
                if not horizon_days:
                    continue
                check = _consensus_evaluate(
                    consensus_target_price=consensus_target,
                    current_price=hdata.get("current_price") or _current_price_snapshot,
                    predicted_price=hdata.get("predicted_price"),
                    horizon_days=horizon_days,
                )
                hdata["consensus_check"] = {
                    "has_consensus":             check.has_consensus,
                    "horizon_days":              check.horizon_days,
                    "consensus_implied_annual":  check.consensus_implied_annual,
                    "consensus_implied_horizon": check.consensus_implied_horizon,
                    "model_return_horizon":      check.model_return_horizon,
                    "divergence_pp":             check.divergence_pp,
                    "is_directionally_opposed":  check.is_directionally_opposed,
                    "is_strong_contrarian":      check.is_strong_contrarian,
                    "apply_haircut":             check.apply_haircut,
                }
                if check.apply_haircut:
                    pre = hdata.get("confidence")
                    if pre is not None:
                        hdata["confidence_pre_haircut"] = pre
                        hdata["confidence"] = _consensus_haircut(pre, check)
        except Exception as exc:  # noqa: BLE001
            print(f"[predict {sym}] consensus check failed: {exc}")
            traceback.print_exc()

        # Log this prediction to the official log so it counts toward Track
        # Record. Per anchor § 8.1, log_prediction_v2 dedupes on
        # (user_id, symbol, user_horizon_code, today_utc) — if this user
        # already has a same-symbol-same-horizon prediction logged today,
        # the logger returns the existing prediction_id and skips the
        # write entirely (no duplicate row, no second model_paper_trade
        # attempt). The model still ran fresh inference above and the
        # user still sees the live numbers in the response; we just don't
        # double-record the call. Failures here are non-fatal: the user
        # still sees their fresh inference; we just lose the log entry.
        #
        # `persisted_pred_id` is the row id the prediction got in Supabase
        # (or the SQLite legacy store). Surfaced on the response so the
        # frontend can fetch the methodology trade plan via /api/ledger/{id}.
        # Defaults to "" if logging fails or there's nothing to log.
        persisted_pred_id: str = ""
        try:
            from prediction_logger_v2 import log_prediction_v2

            # Filter out skipped / non-dict entries — the logger expects
            # full prediction dicts with predicted_return etc.
            #
            # ALSO drop suppressed ("No Vision") horizons: they don't enter
            # the public ledger and don't count toward the published track
            # record. The model still computed them for internal use, but
            # we only stand behind calls that pass our quality gate. If
            # every horizon is suppressed, nothing gets logged for this run.
            preds_for_log = {
                h: v
                for h, v in predictions.items()
                if isinstance(v, dict)
                and not v.get("skipped")
                and not v.get("suppress")
            }
            n_suppressed = sum(
                1 for v in predictions.values()
                if isinstance(v, dict) and v.get("suppress")
            )
            if preds_for_log:
                # regime_cache is a dict ({"label": "Bull", "score_norm": ..., ...}),
                # NOT a string. The earlier `isinstance(str)` check made every
                # API-logged prediction land as "Unknown" — which is why By
                # Market Backdrop on Track Record had nothing to render.
                regime_val = getattr(predictor, "regime_cache", None)
                if isinstance(regime_val, dict):
                    regime_str = regime_val.get("label") or "Unknown"
                elif isinstance(regime_val, str) and regime_val:
                    regime_str = regime_val
                else:
                    regime_str = "Unknown"

                # Build a feature_importance dict from the predictor's
                # SHAP-ranked selected_feats so each logged horizon gets
                # populated `top_features`. Without this, every API
                # prediction lands in the log with `top_features: []`,
                # which starves the per-regime feature breakdown on
                # Track Record. We use rank-based synthetic importance
                # (1.0, 0.9, 0.8, …) since selected_feats is a list, not
                # a {feat: importance} mapping — the logger cares about
                # ordering, not absolute magnitude.
                fi_dict: Dict[str, Dict[str, float]] = {}
                try:
                    selected = getattr(predictor, "selected_feats", {}) or {}
                    for h, feats in selected.items():
                        if not feats:
                            continue
                        fi_dict[h] = {
                            f: round(max(0.1, 1.0 - (i * 0.1)), 4)
                            for i, f in enumerate(list(feats)[:10])
                        }
                except Exception:
                    fi_dict = {}

                # Compute options strategies up-front so they ride into
                # the prediction insert. Each persisted prediction row
                # carries the strategy recommended for ITS horizon, so
                # the user can later paper-trade that specific play
                # from the detail page (Phase B.1).
                options_strategies_for_log = None
                try:
                    from options_analyzer import generate_options_report
                    # Pass `symbol=sym` so each strategy gets snapped to the
                    # live chain (expiry + strikes + premiums) before
                    # persistence — methodology § 4.5. The user sees real,
                    # tradable values instead of Black-Scholes theoretical
                    # ones, and the open path opens cleanly.
                    options_strategies_for_log = generate_options_report(
                        predictions, df, symbol=sym,
                    )
                except Exception as e:
                    print(f"[predict {sym}] options strategies for persist skipped: {e}")
                    options_strategies_for_log = None

                # `_gmv()` is hoisted at the top of predict_symbol —
                # gives a fresh read of the framework version per
                # request rather than relying on the predictor's
                # load-time stamp.
                # `out_meta` is populated by log_prediction_v2 with the
                # trade-attachment outcome (traded?, trade_pass_reason,
                # entry/stop/target, derived R:R). We forward those fields
                # on the predict response so the frontend can render a
                # specific "no equity trade — risk geometry below 1:1"
                # state when the methodology suppressed the trade plan
                # for geometry reasons (PassReason.POOR_RISK_REWARD).
                _logger_meta: dict = {}
                # Pass user_id explicitly — belt-and-suspenders so the
                # write doesn't silently rely on the ContextVar making it
                # through every layer of FastAPI's dispatch. If the async
                # auth dep fixed propagation, this is redundant; if it
                # didn't, this is the only thing keeping the prediction
                # off the legacy fallback path.
                persisted_pred_id = log_prediction_v2(
                    symbol=sym,
                    predictions=preds_for_log,
                    feature_importance=fi_dict if fi_dict else None,
                    model_version=_gmv(),
                    regime=regime_str,
                    # Pass OHLC so trade attachment (anchor § 4.4) computes
                    # entry/stop/target and the model's TRADE/PASS commits
                    # before the prediction lands.
                    ohlc_df=df,
                    user_id=cu.id,
                    # User-selected horizon — gets stamped as the canonical
                    # ledger row so the public ledger reflects what the
                    # user actually asked, not a default 1-month fallback.
                    user_horizon_code=horizon,
                    # Per-horizon options strategies; the logger picks the
                    # canonical-horizon's strategy and persists it.
                    options_strategies=options_strategies_for_log,
                    out_meta=_logger_meta,
                )
                # If log_prediction_v2 returned "" the Supabase write was
                # refused (and we explicitly chose not to fall back to
                # SQLite — see prediction_logger_v2.py). Don't pretend the
                # call succeeded; surface a real error to the frontend so
                # the user sees a retry affordance instead of a phantom
                # receipt the ledger can't show.
                if not persisted_pred_id:
                    print(
                        f"[predict {sym}] log_prediction_v2 returned empty "
                        f"id — Supabase write was refused; surfacing error."
                    )
                    return {
                        "error": "log_persist_failed",
                        "symbol": sym,
                        "hint": (
                            "Prediction couldn't be saved to the public "
                            "ledger. Try again in a moment, and if it "
                            "keeps happening check Render logs for the "
                            "underlying Supabase failure."
                        ),
                    }
                # Bust the analytics cache so this fresh prediction shows up
                # in Track Record immediately, not after the 5-min TTL expires.
                _ANALYTICS_CACHE.clear()
                print(
                    f"[predict {sym}] logged {len(preds_for_log)} horizons "
                    f"({n_suppressed} suppressed, withheld from ledger) "
                    f"+ analytics cache busted (id={persisted_pred_id})"
                )
            elif n_suppressed > 0:
                print(
                    f"[predict {sym}] all {n_suppressed} horizons suppressed — "
                    f"nothing logged to public ledger"
                )
        except Exception as e_log:
            print(f"[predict {sym}] logging skipped: {e_log}")
            traceback.print_exc()

        # Fill in any horizons predict() silently dropped (insufficient
        # training data, all L1 members failed, etc.) so the frontend can
        # show a clear "Not available" state for all 5 horizons.
        EXPECTED_HORIZONS = ["3 Day", "1 Week", "1 Month", "1 Quarter", "1 Year"]
        for h in EXPECTED_HORIZONS:
            if h not in predictions:
                predictions[h] = {
                    "skipped": True,
                    "skip_reason": "Model didn't produce a prediction for this horizon (likely insufficient training samples).",
                }

        # Stamp each horizon with its expiration date (today + N days) and
        # attach the top-3 selected features so the UI can show drivers.
        # Uses the module-level _HORIZON_DAYS constant — used to redefine
        # locally here, which shadowed the same name elsewhere in this
        # function and caused UnboundLocalError in the consensus check.
        today_utc = datetime.utcnow().date()
        selected = getattr(predictor, "selected_feats", {}) or {}
        for h, hdata in predictions.items():
            if not isinstance(hdata, dict):
                continue
            if h in _HORIZON_DAYS:
                hdata["horizon_end"] = (
                    today_utc + timedelta(days=_HORIZON_DAYS[h])
                ).isoformat()
            # Top features for this horizon — sourced from selected_feats which
            # is SHAP-ranked at training time. Trimmed to 3 for compact UI.
            tf = selected.get(h)
            if tf:
                hdata["top_features"] = list(tf)[:3]

        # Current price + today's % change snapshot
        try:
            current_price = float(df["Close"].iloc[-1])
        except Exception:
            current_price = None

        # Contrarian guardrail moved to BEFORE log_prediction_v2 above —
        # see the long comment up there. consensus_check has already
        # stamped per-horizon `consensus_check` blocks AND applied any
        # confidence haircut by the time we reach this point.

        today_change_pct = None
        try:
            if len(df) >= 2:
                prev_close = float(df["Close"].iloc[-2])
                if prev_close > 0:
                    today_change_pct = (current_price - prev_close) / prev_close * 100
        except Exception:
            today_change_pct = None

        # Last 30 closes for the inline sparkline next to the big price.
        try:
            tail = df["Close"].tail(30)
            import pandas as _pd
            dates = _pd.to_datetime(tail.index).strftime("%Y-%m-%d").tolist()
            price_30d = [
                {"date": d, "price": round(float(p), 2)}
                for d, p in zip(dates, tail.tolist())
            ]
        except Exception:
            price_30d = []

        # Model age — surface if it's getting stale so the UI can prompt a retrain.
        model_age_days = None
        try:
            import json as _json
            meta_p = os.path.join(_models_dir(), f"{sym}_meta.json")
            if os.path.exists(meta_p):
                with open(meta_p) as _f:
                    _meta = _json.load(_f)
                trained_at = _meta.get("trained_at")
                if trained_at:
                    # ISO date or pandas timestamp string
                    try:
                        ta_dt = datetime.fromisoformat(trained_at.replace("Z", ""))
                    except Exception:
                        ta_dt = datetime.strptime(trained_at[:10], "%Y-%m-%d")
                    model_age_days = (datetime.utcnow() - ta_dt).days
        except Exception:
            model_age_days = None

        # Options strategy per horizon — fail soft so the trade plan still
        # ships if options chains can't be fetched (illiquid tickers).
        # Reuse the value computed for the persistence path above when
        # available (set inside the `if preds_for_log:` block); otherwise
        # compute fresh.
        options_strategies = locals().get("options_strategies_for_log")
        if not options_strategies:
            try:
                from options_analyzer import generate_options_report
                options_strategies = generate_options_report(
                    predictions, df, symbol=sym,
                )
            except Exception as e:
                print(f"[predict {sym}] options report skipped: {e}")
                options_strategies = None

        # ── Trade plan (entry/stop/target/ATR + per-horizon R:R) ───────
        # Computed inline so a single bad path in analyzer.generate_analysis
        # doesn't drop the whole panel. We try to enrich with the full analyzer
        # narrative as a bonus, but the core levels always ship.
        trade_plan = None
        try:
            import numpy as np

            # ATR (20-day)
            lookback = min(20, len(df) - 1) if len(df) > 1 else 1
            highs = df["High"].iloc[-lookback:].values
            lows = df["Low"].iloc[-lookback:].values
            prev_closes = (
                df["Close"].iloc[-lookback - 1 : -1].values
                if len(df) > lookback
                else df["Close"].iloc[-lookback:].values
            )
            tr = np.maximum(
                highs - lows,
                np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)),
            )
            atr = float(np.mean(tr)) if len(tr) > 0 else 0.02 * (current_price or 1.0)
            atr_pct = (atr / current_price * 100) if current_price else 2.0

            # Entry zone — tight band: 2 ATR below current → just above current.
            entry_low_raw = (current_price or 0) * (1 - 2 * atr_pct / 100)
            entry_high_raw = (current_price or 0) * 1.005
            stop_loss_raw = entry_low_raw - atr

            # First target — pick 1-Month prediction, fall back to whichever
            # non-skipped horizon comes first, fall back to ATR-based stretch.
            first_target_raw = None
            for h in ("1 Month", "1 Week", "3 Day", "1 Quarter", "1 Year"):
                hp = predictions.get(h, {})
                if isinstance(hp, dict) and not hp.get("skipped"):
                    pp = hp.get("predicted_price")
                    if pp is not None:
                        first_target_raw = float(pp)
                        break
            if first_target_raw is None:
                first_target_raw = (current_price or 0) * (1 + 3 * atr_pct / 100)

            # Per-horizon risk/reward
            per_horizon = {}
            risk = max(0.0001, entry_high_raw - stop_loss_raw)
            for h in EXPECTED_HORIZONS:
                hdata = predictions.get(h, {})
                if hdata.get("skipped"):
                    continue
                target_price = hdata.get("predicted_price")
                if target_price is None:
                    continue
                reward = float(target_price) - entry_low_raw
                rr = reward / risk if risk > 0 else 0
                per_horizon[h] = {
                    "target_price": round(float(target_price), 2),
                    "risk_reward":  round(float(rr), 2),
                    "risk_pct":     round(((entry_high_raw - stop_loss_raw) / entry_high_raw) * 100, 2)
                                     if entry_high_raw else None,
                }

            # Try the analyzer's narrative as a bonus — never fatal.
            rec = None
            conf = None
            narrative = None
            supports_out = None
            resistances_out = None
            try:
                from analyzer import generate_analysis
                from data_fetcher import fetch_stock_info
                stock_info = {}
                try:
                    stock_info = fetch_stock_info(sym) or {}
                except Exception:
                    stock_info = {}
                analysis = generate_analysis(df, predictions, stock_info=stock_info)
                rec = analysis.get("recommendation")
                conf = analysis.get("confidence")
                narrative = analysis.get("narrative")
                ez = analysis.get("entry_zone") or {}
                supports_out = ez.get("supports")
                resistances_out = ez.get("resistances")
            except Exception as e_an:
                print(f"[predict {sym}] analyzer narrative skipped: {e_an}")
                # Cheap fallback: classify from 1-Month return
                hp1m = predictions.get("1 Month", {}) or {}
                ret1m = hp1m.get("predicted_return", 0) or 0
                if ret1m > 0.05:        rec, conf = "Strong Buy",  "High"
                elif ret1m > 0.02:      rec, conf = "Buy",          "Medium"
                elif ret1m > -0.02:     rec, conf = "Hold",         "Low"
                elif ret1m > -0.05:     rec, conf = "Sell",         "Medium"
                else:                    rec, conf = "Strong Sell",  "High"

            trade_plan = {
                "entry_low":      round(entry_low_raw, 2),
                "entry_high":     round(entry_high_raw, 2),
                "stop_loss":      round(stop_loss_raw, 2),
                "first_target":   round(first_target_raw, 2),
                "atr":            round(atr, 2),
                "atr_pct":        round(atr_pct, 1),
                "supports":       supports_out,
                "resistances":    resistances_out,
                "recommendation": rec,
                "confidence":     conf,
                "narrative":      narrative,
                "per_horizon":    per_horizon,
            }
        except Exception as e:
            print(f"[predict {sym}] trade plan inline computation failed: {e}")
            traceback.print_exc()
            trade_plan = None

        payload = {
            "symbol":            sym,
            "current_price":     current_price,
            "today_change_pct":  today_change_pct,
            "price_30d":         price_30d,
            "predictions":       predictions,
            "trade_plan":        trade_plan,
            "options_strategies": options_strategies,
            "model_age_days":    model_age_days,
            "generated_at":      datetime.utcnow().isoformat() + "Z",
            # Same fresh-read pattern as the log_prediction_v2 call
            # above — the response payload reflects the current
            # framework version, not whatever the predictor instance
            # cached when it was first loaded.
            "model_version":     _gmv(),
            # The persisted ledger id — frontend uses this to fetch the
            # methodology trade plan (Wilder's-ATR clamped entry/stop/target)
            # via /api/ledger/{id}. Empty string when logging skipped or
            # failed (e.g. all horizons suppressed).
            "prediction_id":     persisted_pred_id,
            # When non-zero `pending`, the UI shows a subtle "upgrading
            # in background" badge so the user knows the next visit will
            # produce a fresher model.
            "model_refresh":     auto_refresh,
            # Trade-attachment outcome from log_prediction_v2's out_meta.
            # `trade_pass_reason` is the canonical PassReason enum value
            # (e.g. "poor_risk_reward") when the model passed on the
            # equity trade, None when it traded. The frontend uses this
            # to render specific suppression states ("no trade — risk
            # geometry") instead of the generic "trade plan not available"
            # fallback. `risk_reward` is the geometry's R:R for telemetry
            # / display. Both default to None when log_prediction_v2 was
            # skipped (suppressed horizons, etc.).
            "trade_pass_reason": (locals().get("_logger_meta") or {}).get("trade_pass_reason"),
            # Recompute R:R inline from the same _logger_meta entry/stop/
            # target values that prediction_logger_v2 saved. Pre-fix the
            # response read the pre-computed `risk_reward` field, which
            # on MU surfaced 0.51 even though entry $640.20 / stop
            # $588.984 / target $681.67 in Supabase produce 0.81 (the
            # 8% stop ceiling clamp). Recomputing here from the saved
            # methodology values guarantees the API number is internally
            # consistent with what got persisted, regardless of whatever
            # produced the pre-computed field.
            "trade_risk_reward": _rr_from_logger_meta(locals().get("_logger_meta") or {}),
        }
        # Strip NaN/Inf so the JSON parses cleanly in the browser.
        try:
            payload = _json_safe(payload)
        except Exception as e:
            print(f"[predict {sym}] _json_safe crashed:")
            traceback.print_exc()
            return {
                "error": "serialization_failed",
                "symbol": sym,
                "detail": f"{type(e).__name__}: {e}",
            }
        _INFERENCE_CACHE[sym] = (payload, datetime.utcnow())
        return payload

    except Exception as e:
        # Last-resort catch — anything not caught above lands here.
        print(f"[predict {sym}] uncaught exception:")
        traceback.print_exc()
        return {
            "error": "unexpected",
            "symbol": sym,
            "detail": f"{type(e).__name__}: {e}",
        }


# ─── Conviction history (per-symbol drift over time) ────────────────────────
# ─── Stock quote (fundamentals + extended price history) ────────────────────
_QUOTE_CACHE: Dict[str, Any] = {}
_QUOTE_TTL_SECONDS = 900  # 15 min — fundamentals don't move intraday


@app.get("/api/quote/{symbol}")
def stock_quote(symbol: str) -> Dict[str, Any]:
    """
    Stock-quote-style metadata for the Prediqt symbol page: name, sector,
    market cap, P/E, 52-week range, dividend yield, analyst target, plus
    a year of daily closes for the targets-on-price chart.

    Cached per-symbol for 15 min — fundamentals only update on quarterly
    earnings releases, and intraday price moves don't change the static
    fields. Cache prevents repeated yfinance calls when the user
    navigates between Prediqt tickers in quick succession.
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"error": "missing_symbol", "symbol": ""}

    cached = _QUOTE_CACHE.get(sym)
    if cached is not None:
        payload, ts = cached
        if (datetime.utcnow() - ts).total_seconds() < _QUOTE_TTL_SECONDS:
            return payload

    try:
        from data_fetcher import (
            fetch_stock_data,
            fetch_stock_info,
            fetch_fundamentals,
        )
    except Exception as e:
        return {
            "error":  "import_failed",
            "symbol": sym,
            "detail": str(e),
        }

    info = {}
    fundamentals = {}
    try:
        info = fetch_stock_info(sym) or {}
    except Exception as e:
        print(f"[quote {sym}] fetch_stock_info failed: {e}")
    try:
        fundamentals = fetch_fundamentals(sym) or {}
    except Exception as e:
        print(f"[quote {sym}] fetch_fundamentals failed: {e}")

    # Helper: yfinance returns NaN/None for missing fields. Normalize.
    def _clean(v):
        if v is None:
            return None
        try:
            f = float(v)
            if f != f:  # NaN
                return None
            return f
        except (TypeError, ValueError):
            return v

    # 1y of daily closes — gives the targets-on-price chart enough
    # context to show meaningful trend without dwarfing the targets.
    price_history = []
    try:
        df = fetch_stock_data(sym, period="1y")
        if df is not None and "Close" in df.columns and len(df) > 0:
            for idx, row in df.iterrows():
                try:
                    date_str = (
                        idx.strftime("%Y-%m-%d")
                        if hasattr(idx, "strftime")
                        else str(idx)[:10]
                    )
                    price = float(row["Close"])
                    if price == price:  # not NaN
                        price_history.append({"date": date_str, "price": round(price, 2)})
                except Exception:
                    continue
    except Exception as e:
        print(f"[quote {sym}] price history failed: {e}")

    current_price = price_history[-1]["price"] if price_history else None
    prev_price = (
        price_history[-2]["price"]
        if len(price_history) >= 2
        else None
    )
    today_change_pct = None
    if current_price is not None and prev_price:
        today_change_pct = round(
            (current_price - prev_price) / prev_price * 100, 2
        )

    description = info.get("description") or ""
    if len(description) > 280:
        description = description[:277].rstrip() + "…"

    payload = {
        "symbol":         sym,
        "name":           info.get("name") or sym,
        "sector":         info.get("sector") or None,
        "industry":       info.get("industry") or None,
        "market_cap":     _clean(info.get("market_cap")),
        "pe_ratio":       _clean(info.get("pe_ratio")),
        "forward_pe":     _clean(fundamentals.get("forward_pe")),
        "peg_ratio":      _clean(fundamentals.get("peg_ratio")),
        "div_yield":      _clean(fundamentals.get("div_yield")),
        "beta":           _clean(info.get("beta")),
        "fifty_two_week_high": _clean(info.get("52w_high")),
        "fifty_two_week_low":  _clean(info.get("52w_low")),
        "avg_volume":     _clean(info.get("avg_volume")),
        "analyst_target_mean": _clean(fundamentals.get("target_mean")),
        "analyst_target_low":  _clean(fundamentals.get("target_low")),
        "analyst_target_high": _clean(fundamentals.get("target_high")),
        "n_analysts":     _clean(fundamentals.get("n_analysts")),
        "description":    description if description and description != "N/A" else None,
        "current_price":  current_price,
        "today_change_pct": today_change_pct,
        "price_history":  price_history,
        "as_of":          datetime.utcnow().isoformat() + "Z",
    }

    payload = _json_safe(payload)
    _QUOTE_CACHE[sym] = (payload, datetime.utcnow())
    return payload


@app.get("/api/predict/{symbol}/conviction-history")
def conviction_history(symbol: str, days: int = 14) -> Dict[str, Any]:
    """
    Returns a per-horizon time series of how the model's prediction for this
    ticker has shifted over the past `days` days. Powers the "is the model's
    view steady or whipsawing?" sparkline strip on the Prediqt page.

    Each horizon block contains an ordered list of {date, predicted_return,
    confidence, direction} points — one per logged run. The frontend renders
    a tiny sparkline so the user can see at a glance whether conviction has
    been stable, drifting, or oscillating.
    """
    sym = symbol.upper().strip()
    if not sym:
        return {"error": "missing_symbol", "symbol": ""}

    days = max(1, min(int(days or 14), 90))

    try:
        from prediction_logger_v2 import _load_log
    except Exception as e:
        return {
            "symbol": sym,
            "days": days,
            "error": "logger_unavailable",
            "detail": f"{type(e).__name__}: {e}",
            "horizons": {},
        }

    cutoff = datetime.utcnow() - timedelta(days=days)
    entries = _load_log() or []

    # Filter to this symbol within the window. Logger uses local naive
    # timestamps; treat them as UTC for comparison purposes — a one-off
    # tz mismatch on the boundary is acceptable for a 14-day window.
    matched = []
    for e in entries:
        if (e.get("symbol", "") or "").upper() != sym:
            continue
        ts_raw = e.get("timestamp", "") or ""
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            # Pre-2.7 logger ISO strings sometimes have 'Z' suffix etc.
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", ""))
            except Exception:
                continue
        if ts < cutoff:
            continue
        matched.append((ts, e))

    matched.sort(key=lambda t: t[0])

    HORIZONS = ["3 Day", "1 Week", "1 Month", "1 Quarter", "1 Year"]
    horizons_out: Dict[str, Any] = {h: {"points": []} for h in HORIZONS}

    for ts, e in matched:
        date_iso = ts.strftime("%Y-%m-%d")
        for h in HORIZONS:
            hdata = (e.get("horizons") or {}).get(h)
            if not hdata:
                continue
            pred_return = hdata.get("predicted_return")
            if pred_return is None:
                continue
            conf = hdata.get("confidence", 0) or 0
            direction = "up" if (pred_return or 0) >= 0 else "down"
            horizons_out[h]["points"].append({
                "date": date_iso,
                "timestamp": ts.isoformat(),
                "predicted_return": float(pred_return),
                "confidence": float(conf),
                "direction": direction,
            })

    # Drop horizons with no data so the frontend doesn't render empty cards.
    horizons_out = {h: v for h, v in horizons_out.items() if v["points"]}

    return _json_safe({
        "symbol": sym,
        "days": days,
        "total_runs": len({e["timestamp"] for _, e in matched}),
        "horizons": horizons_out,
    })


# ─── Async training for unknown tickers ─────────────────────────────────────
# A ticker that isn't in the trained roster yet can be added on demand. The
# user clicks "Train & predict" → we queue a background thread that fetches
# 10y of OHLC, engineers features, trains the ensemble, and saves the joblib.
# The frontend polls /api/train-status/{symbol} every few seconds until
# "complete", then calls /api/predict/{symbol} as usual.
#
# Concurrency: a single global semaphore caps concurrent trainings at 1.
# Training is CPU-heavy and would trash the API's tail latency if unbounded.
_TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}
_TRAINING_JOBS_LOCK = threading.Lock()
_TRAINING_SEMAPHORE = threading.Semaphore(1)

# Stages the UI can map to progress messages.
_STAGE_FETCH      = "fetching_data"
_STAGE_TRAIN      = "training"
_STAGE_SAVE       = "saving"
_STAGE_DONE       = "complete"
_STAGE_FAILED     = "failed"
_STAGE_QUEUED     = "queued"


def _set_job(symbol: str, **updates) -> None:
    """Thread-safe mutation of a training job record."""
    with _TRAINING_JOBS_LOCK:
        rec = _TRAINING_JOBS.setdefault(symbol, {})
        rec.update(updates)


def _get_job(symbol: str) -> Optional[Dict[str, Any]]:
    with _TRAINING_JOBS_LOCK:
        rec = _TRAINING_JOBS.get(symbol)
        return dict(rec) if rec is not None else None


def _purge_stale_jobs(max_age_seconds: int = 3600) -> None:
    """Drop completed/failed records older than `max_age_seconds`."""
    cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
    with _TRAINING_JOBS_LOCK:
        stale = []
        for sym, rec in _TRAINING_JOBS.items():
            if rec.get("stage") in (_STAGE_DONE, _STAGE_FAILED):
                completed = rec.get("completed_at")
                if completed and completed < cutoff:
                    stale.append(sym)
        for sym in stale:
            _TRAINING_JOBS.pop(sym, None)


def _train_symbol_worker(symbol: str) -> None:
    """
    Runs in a background thread. Updates the _TRAINING_JOBS record at each
    stage so the frontend poll surfaces live progress. Catches every failure
    mode so the thread can never leave a job stuck.
    """
    sym = symbol.upper()
    _set_job(sym, stage=_STAGE_QUEUED, started_at=datetime.utcnow(), error=None)

    with _TRAINING_SEMAPHORE:
        # ── Fetch ─────────────────────────────────────────────────────
        _set_job(
            sym,
            stage=_STAGE_FETCH,
            progress=0.1,
            message="Fetching 10 years of price history…",
        )

        try:
            from data_fetcher import (
                fetch_stock_data,
                fetch_market_context,
                fetch_fundamentals,
                fetch_earnings_data,
                fetch_options_data,
            )
        except Exception as e:
            _set_job(
                sym,
                stage=_STAGE_FAILED,
                completed_at=datetime.utcnow(),
                error=f"import_failed: {e}",
            )
            return

        try:
            # 10y matches the Streamlit baseline that seeded the original
            # roster — keeps confidence calibration consistent across every
            # ticker the model has ever seen. Adds ~30% to fetch+train time
            # (still 2-4 min total, well within "click and wait" UX).
            # Tickers younger than 10y just return their full available
            # history; yfinance silently caps at the listing date.
            df = fetch_stock_data(sym, period="10y")
        except Exception as e:
            _set_job(
                sym,
                stage=_STAGE_FAILED,
                completed_at=datetime.utcnow(),
                error=f"Ticker not found or no price data: {e}",
            )
            return

        if df is None or len(df) < 100:
            _set_job(
                sym,
                stage=_STAGE_FAILED,
                completed_at=datetime.utcnow(),
                error="Not enough price history to train a model (need 100+ trading days).",
            )
            return

        # Best-effort aux data — all are optional for training
        try:
            market_ctx = fetch_market_context()
        except Exception:
            market_ctx = None
        try:
            fundamentals = fetch_fundamentals(sym)
        except Exception:
            fundamentals = None
        try:
            earnings_data = fetch_earnings_data(sym)
        except Exception:
            earnings_data = None
        try:
            current_price = float(df["Close"].iloc[-1])
            options_data = fetch_options_data(sym, current_price)
        except Exception:
            options_data = None

        # ── Train ─────────────────────────────────────────────────────
        _set_job(
            sym,
            stage=_STAGE_TRAIN,
            progress=0.4,
            message="Training the ensemble across all horizons…",
        )

        try:
            from model import StockPredictor
            predictor = StockPredictor(sym)
            predictor.train(
                df,
                market_ctx=market_ctx,
                fundamentals=fundamentals,
                earnings_data=earnings_data,
                options_data=options_data,
            )
        except Exception as e:
            _set_job(
                sym,
                stage=_STAGE_FAILED,
                completed_at=datetime.utcnow(),
                error=f"Training failed: {e}",
            )
            return

        # ── Save ──────────────────────────────────────────────────────
        _set_job(
            sym,
            stage=_STAGE_SAVE,
            progress=0.9,
            message="Saving model to disk…",
        )

        try:
            # Stamp the model with the period it was trained on so the
            # auto-retrain detector can spot legacy models that need an
            # upgrade (e.g., 7y → 10y bumps).
            predictor.save_model(train_data_period=EXPECTED_TRAIN_DATA_PERIOD)
        except Exception as e:
            _set_job(
                sym,
                stage=_STAGE_FAILED,
                completed_at=datetime.utcnow(),
                error=f"Saving failed: {e}",
            )
            return

        # ── Done ──────────────────────────────────────────────────────
        _set_job(
            sym,
            stage=_STAGE_DONE,
            progress=1.0,
            message="Model ready.",
            completed_at=datetime.utcnow(),
        )

        # Bust the inference cache so the next /api/predict/{sym} hit
        # uses the freshly-trained weights.
        _INFERENCE_CACHE.pop(sym, None)


@app.post("/api/train/{symbol}")
def start_training(symbol: str) -> Dict[str, Any]:
    """
    Kick off a background training job for `symbol`. Returns immediately;
    the frontend polls /api/train-status/{symbol} for progress.

    No-ops cleanly if the model already exists on disk, or if training is
    already in flight for this ticker.
    """
    sym = (symbol or "").upper().strip()
    if not sym or not sym.isalnum():
        return {"error": "invalid_symbol"}

    # Already trained?
    model_path = os.path.join(_models_dir(), f"{sym}_model.joblib")
    if os.path.exists(model_path):
        return {"symbol": sym, "stage": _STAGE_DONE, "already_trained": True}

    # Drop stale records so we don't return yesterday's "complete" to a user
    # who wants a retrain.
    _purge_stale_jobs()

    existing = _get_job(sym)
    if existing and existing.get("stage") in (_STAGE_QUEUED, _STAGE_FETCH, _STAGE_TRAIN, _STAGE_SAVE):
        return {"symbol": sym, **existing, "already_running": True}

    # Launch worker thread
    t = threading.Thread(
        target=_train_symbol_worker,
        args=(sym,),
        name=f"train-{sym}",
        daemon=True,
    )
    _set_job(sym, stage=_STAGE_QUEUED, started_at=datetime.utcnow(), error=None, progress=0.0)
    t.start()

    return {"symbol": sym, "stage": _STAGE_QUEUED, "started": True}


@app.get("/api/train-status/{symbol}")
def training_status(symbol: str) -> Dict[str, Any]:
    """
    Poll this endpoint every few seconds after starting a training job.
    Returns the live stage + progress + any error message. If the model
    already exists on disk (and we have no active record), reports
    "complete" so the frontend can skip straight to inference.
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"error": "invalid_symbol"}

    rec = _get_job(sym)
    if rec:
        # Serialize datetimes for JSON
        payload = {"symbol": sym}
        for k, v in rec.items():
            if isinstance(v, datetime):
                payload[k] = v.isoformat() + "Z"
            else:
                payload[k] = v
        return payload

    # No active record — check disk
    model_path = os.path.join(_models_dir(), f"{sym}_model.joblib")
    if os.path.exists(model_path):
        return {"symbol": sym, "stage": _STAGE_DONE, "on_disk": True}

    return {"symbol": sym, "stage": "untrained"}


# ─── Per-symbol price cache (used by prediction detail page) ────────────────
_PRICE_CACHE: Dict[str, Any] = {}
_PRICE_TTL_SECONDS = 900  # 15 minutes — symbol price history doesn't change intraday


def _fetch_price_history(symbol: str):
    """
    Fetch ~2 years of daily close prices for a symbol, cached for 15 minutes
    per ticker. Returns None on any failure so callers can degrade.
    """
    sym = symbol.upper()
    cached = _PRICE_CACHE.get(sym)
    if cached is not None:
        data, ts = cached
        if (datetime.utcnow() - ts).total_seconds() < _PRICE_TTL_SECONDS:
            return data
    try:
        from data_fetcher import fetch_stock_data
        df = fetch_stock_data(sym, period="2y")
        _PRICE_CACHE[sym] = (df, datetime.utcnow())
        return df
    except Exception:
        return None


_HORIZON_CODE_TO_NAME_MAIN = {
    "3d": "3 Day",
    "1w": "1 Week",
    "1m": "1 Month",
    "1q": "1 Quarter",
    "1y": "1 Year",
}
_HORIZON_DAYS_MAIN = {
    "3 Day": 3, "1 Week": 7, "1 Month": 30,
    "1 Quarter": 90, "1 Year": 365,
}

_CHECKPOINT_INTERVALS = ("1d", "3d", "7d", "14d", "30d", "60d", "90d", "180d", "365d")


def _compute_interval_scores(
    price_curve: list,
    entry_date_str: str,
    entry_price: float,
    direction: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Walk the price curve and fill in per-interval check scores.

    For each interval (1d, 3d, ... 365d), find the close price closest
    to (entry_date + N calendar days) — only past dates count. Returns
    the same shape the legacy SQLite path produced:
        { "1d": { "score": "✓" | "✗" | None, "return": float | None,
                  "price": float | None }, ... }

    Direction-aware: a short call where the price drops scores ✓.
    Returns null entries for intervals that haven't elapsed yet so the
    UI can render them as pending dashes.
    """
    out: Dict[str, Dict[str, Any]] = {
        iv: {"score": None, "return": None, "price": None}
        for iv in _CHECKPOINT_INTERVALS
    }

    if not price_curve or not entry_date_str or entry_price <= 0:
        return out

    try:
        entry_dt = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
    except Exception:
        return out

    today = datetime.now(timezone.utc).date()
    direction_up = (direction or "").upper() == "UP"

    # price_curve is a list of {"date": "YYYY-MM-DD", "price": float}
    # already filtered to dates >= entry_date - 7. Index by date for
    # fast lookup.
    by_date = {p["date"]: float(p["price"]) for p in price_curve}
    sorted_dates = sorted(by_date.keys())

    for iv in _CHECKPOINT_INTERVALS:
        try:
            n_days = int(iv.rstrip("d"))
        except ValueError:
            continue

        target_date = entry_dt + timedelta(days=n_days)
        if target_date > today:
            # Interval hasn't elapsed yet — leave pending.
            continue

        # Find the closest available trading-day close on or after the
        # target date. yfinance returns business days only, so a target
        # falling on a weekend lands on the next trading day.
        target_iso = target_date.isoformat()
        match_date = next((d for d in sorted_dates if d >= target_iso), None)
        if match_date is None:
            # Past the curve — interval elapsed but yfinance is missing
            # data for that day. Leave pending.
            continue

        actual_price = by_date[match_date]
        ret_pct = ((actual_price - entry_price) / entry_price) * 100.0
        # Direction-aware: for shorts, a negative return is a win.
        favourable = ret_pct >= 0 if direction_up else ret_pct <= 0
        out[iv] = {
            "score": "✓" if favourable else "✗",
            "return": round(ret_pct, 2),
            "price": round(actual_price, 2),
        }

    return out


def _synthesize_detail_from_supabase(
    sb_row: Dict[str, Any],
    row_id: str,
) -> Dict[str, Any]:
    """
    Adapt a Supabase predictions row to the PredictionDetail shape the
    /track-record/[rowId] page consumes.

    Supabase rows are per-canonical-horizon (one row = one prediction at
    one horizon). The shape needs the legacy fields the SQLite-era page
    grew up with: top_features (left empty — Supabase doesn't carry per-
    horizon SHAP features), scores_full (all pending — Supabase tracks
    the three rating_* columns instead, and those drive the page's
    other surfaces), price_history (fetched fresh from yfinance from
    the prediction date forward).

    The hero, price chart, and key stats grid all render meaningfully.
    The "checkpoint history" and "top signal drivers" sections degrade
    to empty/pending — acceptable until the per-horizon detail table
    lands in Supabase (separate migration, not blocking).
    """
    sym = (sb_row.get("symbol") or "").upper()
    horizon_code = sb_row.get("horizon") or "1m"
    horizon_name = _HORIZON_CODE_TO_NAME_MAIN.get(horizon_code, horizon_code)
    horizon_days = _HORIZON_DAYS_MAIN.get(horizon_name)

    # Supabase stores `predicted_return` already in percent (per
    # db.insert_prediction line 504: round(pred_return * 100, 4)).
    pred_return_pct = float(sb_row.get("predicted_return") or 0)
    pred_price = float(sb_row.get("predicted_price") or 0)
    entry_price = sb_row.get("entry_price")
    confidence = float(sb_row.get("confidence") or 0)
    direction_label = (sb_row.get("direction") or "").lower()
    direction = "UP" if direction_label.startswith("bull") else (
        "DOWN" if direction_label.startswith("bear") else "—"
    )

    verdict = (sb_row.get("verdict") or "OPEN").upper()
    final_result = (
        "✓" if verdict == "HIT"
        else "✗" if verdict == "MISSED"
        else "pending"
    )

    # Date strings the frontend expects in YYYY-MM-DD.
    created_iso = sb_row.get("created_at") or ""
    pred_date_str = created_iso[:10] if created_iso else ""
    horizon_end_iso = sb_row.get("horizon_ends_at") or ""
    horizon_end_str = horizon_end_iso[:10] if horizon_end_iso else None

    # Price history: fetch from yfinance starting a few days before
    # the prediction date and ending at the horizon end (or today).
    price_curve: list = []
    df = _fetch_price_history(sym)
    if df is not None and not df.empty and pred_date_str:
        try:
            import pandas as pd
            series = df["Close"].copy()
            series.index = pd.to_datetime(series.index).strftime("%Y-%m-%d")
            series = series[~series.index.duplicated(keep="last")]
            try:
                pre_ctx = (
                    datetime.strptime(pred_date_str, "%Y-%m-%d")
                    - timedelta(days=7)
                ).strftime("%Y-%m-%d")
            except Exception:
                pre_ctx = pred_date_str
            sliced = series[series.index >= pre_ctx]
            # Upper bound is whichever is later: horizon_end or today.
            # Pre-expiry calls keep their original behavior (no future
            # data exists, the slice naturally ends at the latest bar).
            # Post-expiry calls now extend through today so users see
            # how the stock actually moved after judgment passed —
            # previously the chart hard-stopped at horizon_end which
            # made stale-feeling receipts (CROX scored MISSED on May 3
            # but the chart only went to May 1, hiding 4 days of
            # actual price action).
            today_str = datetime.utcnow().strftime("%Y-%m-%d")
            upper_bound = (
                today_str
                if (not horizon_end_str or today_str > horizon_end_str)
                else horizon_end_str
            )
            sliced = sliced[sliced.index <= upper_bound]
            price_curve = [
                {"date": d, "price": round(float(v), 2)}
                for d, v in sliced.items()
            ]
        except Exception as exc:
            print(f"[synthesize_detail] price slice failed for {sym}: {exc}")
            price_curve = []

    # Per-interval checkpoint slots — computed on-the-fly from the
    # price curve we already fetched above. For each elapsed interval
    # (1d, 3d, 7d, ... 365d) we look up the close at entry_date+N and
    # decide direction-correct. Intervals that haven't elapsed stay
    # null so the UI renders them as pending dashes.
    #
    # Replaces the previous "all null" stub — Supabase rows now get the
    # same per-interval read the legacy SQLite path produced.
    full_scores = _compute_interval_scores(
        price_curve=price_curve,
        entry_date_str=pred_date_str,
        entry_price=float(entry_price or 0),
        direction=direction,
    )

    pred_id = sb_row.get("id") or row_id
    horizon_slug = horizon_name.lower().replace(" ", "-") or "unknown"
    computed_row_id = f"{pred_id}-{horizon_slug}"

    return {
        "row_id":                  computed_row_id,
        "prediction_id":           pred_id,
        "symbol":                  sym,
        "date":                    pred_date_str,
        "horizon":                 horizon_name,
        "horizon_end":             horizon_end_str,
        "predicted_return_pct":    pred_return_pct,
        "predicted_price":         pred_price,
        "current_price":           float(entry_price or 0),
        "confidence":              confidence,
        "direction":               direction,
        "model_version":           str(sb_row.get("model_version") or ""),
        "regime":                  str(sb_row.get("regime") or ""),
        "final_result":            final_result,
        "top_features":            [],

        # Target-hit fields don't exist on Supabase rows the same way;
        # the rating_target column carries the verdict instead. Leave
        # the legacy fields null/false and let the UI degrade.
        "target_hit":              None,
        "day_target_hit":          None,
        "target_hit_date":         None,
        "target_hit_price":        None,
        "peak_favorable_move_pct": None,
        "peak_fav_day":            None,
        "peak_fav_price":          None,
        "horizon_expired":         verdict != "OPEN",

        "scores_full":             full_scores,
        "price_history":           price_curve,
    }


@app.get("/api/predictions/{row_id}")
def prediction_detail(row_id: str) -> Dict[str, Any]:
    """
    Full detail for a single prediction row (keyed by row_id which is
    `{prediction_id}-{horizon-slug}`). Includes a price-history slice from
    prediction_date to the horizon end so the frontend can render a chart
    with entry, target, and current-price markers.
    """
    pa = _get_enriched_analytics()
    preds = pa.get("predictions_table", [])

    # Locate the row — row_id is composed client-side but we can match it by
    # decomposing: last segment(s) are horizon-slug, rest is prediction_id.
    target_pred = None
    for p in preds:
        _pid = p.get("prediction_id", "") or ""
        _h = (p.get("horizon", "") or "").lower().replace(" ", "-")
        if f"{_pid}-{_h}" == row_id:
            target_pred = p
            break

    if target_pred is None:
        # SQLite miss — fall back to Supabase. Post-cutover predictions
        # live in the Supabase predictions table only and never get
        # written to the legacy SQLite store, so this is the *primary*
        # path for any prediction made after 2026-04-25.
        #
        # The Ledger link sends the bare prediction UUID (no horizon
        # slug). If the path included one anyway (legacy compound
        # row_id format `{uuid}-{horizon-slug}`), strip the trailing
        # segments to recover the UUID. UUIDs are 5 hyphen-separated
        # parts; anything beyond is the horizon hint.
        parts = row_id.split("-")
        bare_uuid = "-".join(parts[:5]) if len(parts) > 5 else row_id
        try:
            from db import get_prediction_detail as _supabase_detail
            sb_row = _supabase_detail(bare_uuid)
        except Exception as exc:
            print(f"[prediction_detail] Supabase fallback failed for {bare_uuid}: {exc}")
            sb_row = None
        if sb_row is None:
            return {"error": "not_found", "row_id": row_id}
        return _synthesize_detail_from_supabase(sb_row, row_id)

    # Full enriched shape (include ALL score intervals, not just ✓/✗)
    INTERVALS = ("1d", "3d", "7d", "14d", "30d", "60d", "90d", "180d", "365d")
    full_scores: Dict[str, Dict[str, Any]] = {}
    for iv in INTERVALS:
        s = target_pred.get(f"score_{iv}")
        r = target_pred.get(f"return_{iv}")
        pr = target_pred.get(f"price_{iv}")
        full_scores[iv] = {
            "score":  s,  # "✓" | "✗" | "—" | None
            "return": r,
            "price":  pr,
        }

    # Price history slice covering prediction → horizon end
    price_curve = []
    horizon_days = {
        "3 Day": 3, "1 Week": 7, "1 Month": 30,
        "1 Quarter": 90, "1 Year": 365,
    }.get(target_pred.get("horizon", ""), None)

    pred_date_str = target_pred.get("date", "")
    horizon_end_str: Optional[str] = None
    if horizon_days and pred_date_str:
        try:
            pd_dt = datetime.strptime(pred_date_str, "%Y-%m-%d")
            horizon_end_str = (pd_dt + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
        except Exception:
            horizon_end_str = None

    df = _fetch_price_history(target_pred.get("symbol", ""))
    if df is not None and not df.empty and pred_date_str:
        import pandas as pd
        series = df["Close"].copy()
        series.index = pd.to_datetime(series.index).strftime("%Y-%m-%d")
        series = series[~series.index.duplicated(keep="last")]

        # Start 5 trading days before the prediction for context; end at
        # horizon_end or today, whichever is earlier, then +a few days buffer.
        start_mask = series.index >= pred_date_str
        # Give a small pre-context window (5 calendar days)
        try:
            pre_ctx = (datetime.strptime(pred_date_str, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
            start_mask = series.index >= pre_ctx
        except Exception:
            pass
        sliced = series[start_mask]
        if horizon_end_str:
            sliced = sliced[sliced.index <= horizon_end_str]
        price_curve = [
            {"date": d, "price": round(float(v), 2)}
            for d, v in sliced.items()
        ]

    _pred_id = target_pred.get("prediction_id", "") or ""
    _horizon = target_pred.get("horizon", "") or ""
    _horizon_slug = _horizon.lower().replace(" ", "-") or "unknown"
    _sym = target_pred.get("symbol", "") or ""
    _date = target_pred.get("date", "") or ""
    computed_row_id = (
        f"{_pred_id}-{_horizon_slug}"
        if _pred_id
        else f"{_sym}-{_date}-{_horizon_slug}"
    )

    return {
        "row_id":                  computed_row_id,
        "prediction_id":           _pred_id,
        "symbol":                  _sym,
        "date":                    _date,
        "horizon":                 _horizon,
        "horizon_end":             horizon_end_str,
        "predicted_return_pct":    target_pred.get("predicted_return", 0),
        "predicted_price":         target_pred.get("predicted_price", 0),
        "current_price":           target_pred.get("current_price", 0),
        "confidence":              target_pred.get("confidence", 0),
        "direction":               target_pred.get("direction", ""),
        "model_version":           target_pred.get("model_version", ""),
        "regime":                  target_pred.get("regime", ""),
        "final_result":            target_pred.get("final_result", "pending"),
        "top_features":            target_pred.get("top_features", []),

        "target_hit":              target_pred.get("target_hit"),
        "day_target_hit":          target_pred.get("day_target_hit"),
        "target_hit_date":         target_pred.get("target_hit_date"),
        "target_hit_price":        target_pred.get("target_hit_price"),
        "peak_favorable_move_pct": target_pred.get("peak_favorable_move_pct"),
        "peak_fav_day":            target_pred.get("peak_fav_day"),
        "peak_fav_price":          target_pred.get("peak_fav_price"),
        "horizon_expired":         target_pred.get("horizon_expired", False),

        "scores_full":             full_scores,
        "price_history":           price_curve,
    }


# ─── Predictions log (paginated, filtered, sorted) ──────────────────────────
@app.get("/api/predictions")
def predictions_log(
    filter: str = "all",            # "all" | "scored" | "pending"
    days: Optional[int] = None,     # last N days (None = no time filter)
    sort: str = "newest",           # "newest" | "oldest"
    horizon: Optional[str] = None,  # e.g. "3 Day", "1 Week" (None = any)
    metric: Optional[str] = None,   # "target" | "checkpoint" | "expiration"
    symbol: Optional[str] = None,   # e.g. "AAPL" (None = any)
    min_confidence: Optional[float] = None,  # only predictions >= this confidence
    from_date: Optional[str] = None,         # YYYY-MM-DD lower bound on prediction date
    to_date: Optional[str] = None,           # YYYY-MM-DD upper bound (inclusive)
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Paginated prediction log with status / time / sort / horizon / metric
    filters. Each row has enough context for the Track Record expandable
    rows (target hit, peak favorable move, top features, per-interval scores).

    `metric` restricts rows to those that contribute to a specific honesty
    read's denominator:
      - "target"     → rows where target is definitively resolved (hit, or
                       horizon expired without a hit)
      - "checkpoint" → rows with at least one scored interval
      - "expiration" → rows whose full horizon has matured (✓/✗ final result)
    """
    pa = _get_enriched_analytics()
    preds = list(pa.get("predictions_table", []))

    # Status filter
    if filter == "scored":
        preds = [p for p in preds if p.get("final_result") in ("✓", "✗")]
    elif filter == "pending":
        preds = [p for p in preds if p.get("final_result") not in ("✓", "✗")]

    # Time-range filter (by prediction date)
    if days is not None and days > 0:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        preds = [p for p in preds if p.get("date", "") >= cutoff]

    # Horizon filter (exact match on the horizon label)
    if horizon:
        preds = [p for p in preds if p.get("horizon", "") == horizon]

    # Symbol filter (case-insensitive exact match)
    if symbol:
        sym_upper = symbol.upper()
        preds = [p for p in preds if (p.get("symbol", "") or "").upper() == sym_upper]

    # Confidence floor
    if min_confidence is not None:
        preds = [p for p in preds if (p.get("confidence", 0) or 0) >= min_confidence]

    # Explicit date window (prediction date)
    if from_date:
        preds = [p for p in preds if p.get("date", "") >= from_date]
    if to_date:
        preds = [p for p in preds if p.get("date", "") <= to_date]

    # Metric eligibility filter
    if metric == "target":
        preds = [
            p for p in preds
            if p.get("target_hit") is True
            or (p.get("target_hit") is False and p.get("horizon_expired"))
        ]
    elif metric == "checkpoint":
        # At least one interval scored ✓ or ✗
        _INTERVALS = ("1d", "3d", "7d", "14d", "30d", "60d", "90d", "180d", "365d")
        preds = [
            p for p in preds
            if any(p.get(f"score_{iv}") in ("✓", "✗") for iv in _INTERVALS)
        ]
    elif metric == "expiration":
        preds = [p for p in preds if p.get("final_result") in ("✓", "✗")]

    # Sort
    preds.sort(key=lambda p: p.get("date", ""), reverse=(sort == "newest"))

    total_filtered = len(preds)
    page = preds[offset:offset + limit]

    # Shape each row for the frontend — only what the UI needs
    INTERVALS = ["1d", "3d", "7d", "14d", "30d", "60d", "90d", "180d", "365d"]
    output = []
    for p in page:
        scores = {}
        returns = {}
        for iv in INTERVALS:
            s = p.get(f"score_{iv}")
            r = p.get(f"return_{iv}")
            if s in ("✓", "✗"):
                scores[iv] = s
            if r is not None:
                returns[iv] = r

        # Row-level unique identifier. `prediction_id` alone is NOT unique —
        # one "run" (e.g. GOOG on 2026-04-24) emits multiple rows, one per
        # horizon (3 Day, 1 Week, 1 Month, 1 Quarter, 1 Year) that all share
        # the same prediction_id. Composing with the horizon slug makes it
        # unique per row, suitable for React keys and deep-link URLs.
        _pred_id = p.get("prediction_id", "") or ""
        _horizon = p.get("horizon", "") or ""
        _horizon_slug = _horizon.lower().replace(" ", "-") or "unknown"
        _sym = p.get("symbol", "") or ""
        _date = p.get("date", "") or ""
        row_id = (
            f"{_pred_id}-{_horizon_slug}"
            if _pred_id
            else f"{_sym}-{_date}-{_horizon_slug}"
        )

        output.append({
            "row_id":                  row_id,
            "prediction_id":           _pred_id,
            "symbol":                  _sym,
            "date":                    _date,
            "horizon":                 _horizon,
            "predicted_return_pct":    p.get("predicted_return", 0),
            "predicted_price":         p.get("predicted_price", 0),
            "current_price":           p.get("current_price", 0),
            "confidence":              p.get("confidence", 0),
            "direction":               p.get("direction", ""),
            "model_version":           p.get("model_version", ""),
            "regime":                  p.get("regime", ""),
            "final_result":            p.get("final_result", "pending"),
            "top_features":            p.get("top_features", []),

            # Target-hit fields (from target_hit_analyzer)
            "target_hit":              p.get("target_hit"),
            "day_target_hit":          p.get("day_target_hit"),
            "target_hit_date":         p.get("target_hit_date"),
            "target_hit_price":        p.get("target_hit_price"),
            "peak_favorable_move_pct": p.get("peak_favorable_move_pct"),
            "peak_fav_day":            p.get("peak_fav_day"),
            "peak_fav_price":          p.get("peak_fav_price"),
            "horizon_expired":         p.get("horizon_expired", False),

            # Per-interval scores & returns
            "scores":  scores,
            "returns": returns,
        })

    return {
        "total":       pa.get("total_predictions", 0),
        "filtered":    total_filtered,
        "offset":      offset,
        "limit":       limit,
        "predictions": output,
    }
