"""
db.py  —  Prediqt data-access layer
────────────────────────────────────────────────────────────────────────────
A thin abstraction over "where does the data live?"

  • USE_SUPABASE=false (default)  → operations go through legacy file-backed
                                    functions in watchlist.py / paper_trader.py
                                    / prediction_logger_v2.py. Nothing changes.
  • USE_SUPABASE=true             → operations talk to Supabase Postgres.

Every function takes an optional `user_id` kwarg. When in file mode, user_id
is ignored (single-user). When in Supabase mode:
  • If you're authed via Supabase Auth, user_id defaults to auth.uid().
  • If you're running locally without auth, set DEV_USER_ID in .env and all
    operations run as that user. This lets you iterate against real Supabase
    without building login yet.

Design goals:
  1. The page code (app.py, *_page.py) keeps calling the same shims it does
     today — `load_watchlist()`, `add_to_watchlist(...)`, etc. Nothing in the
     UI layer needs to know which backend it's talking to.
  2. supabase-py is a LAZY import: if you haven't installed it, file mode
     still works fine. You only pay the dependency cost when USE_SUPABASE=true.
  3. Each backend is a self-contained function — no class hierarchies, no
     adapters. One if/else per operation keeps this easy to read and easy
     to delete when we fully cut over.
"""

from __future__ import annotations

import contextvars
import os
import logging
from typing import Any, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Load .env if python-dotenv is installed (best-effort; not required in prod)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass


# ──────────────────────────────────────────────────────────────────────────
# Feature flag + config
# ──────────────────────────────────────────────────────────────────────────

def _bool_env(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


USE_SUPABASE: bool = _bool_env("USE_SUPABASE", default=False)
SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY: Optional[str] = os.getenv("SUPABASE_ANON_KEY")
# Service-role key bypasses RLS. Only use in trusted server-side contexts
# (e.g. Stripe webhook worker, prediction-scoring cron). Never ship to client.
SUPABASE_SERVICE_ROLE_KEY: Optional[str] = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Local-dev "acting as" user. Only used when USE_SUPABASE=true AND there is
# no authed session (e.g. running Streamlit locally before we wire up auth).
DEV_USER_ID: Optional[str] = os.getenv("DEV_USER_ID")


# ──────────────────────────────────────────────────────────────────────────
# Supabase client (lazy, cached)
# ──────────────────────────────────────────────────────────────────────────

def _is_phase1_dev_mode() -> bool:
    """Phase 1 dev mode: no real auth yet, we're acting as DEV_USER_ID.
    In this mode we use the service_role key (bypasses RLS) because there
    is no authed JWT to satisfy the RLS policies. Phase 2 replaces this
    with proper supabase.auth sessions and the anon key."""
    return bool(DEV_USER_ID) and bool(SUPABASE_SERVICE_ROLE_KEY)


@lru_cache(maxsize=1)
def _client():
    """Return a cached Supabase client.
    In Phase 1 dev mode (DEV_USER_ID set) this returns the service-role
    client so writes scoped to DEV_USER_ID aren't blocked by RLS.
    In Phase 2 (real auth) this returns the anon client and relies on the
    authed session's JWT to satisfy RLS."""
    if not USE_SUPABASE:
        raise RuntimeError("db._client() called while USE_SUPABASE=false")
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL is not set. See SUPABASE_SETUP.md.")
    try:
        from supabase import create_client
    except ImportError as exc:
        raise ImportError(
            "supabase-py is not installed. Run: "
            "pip install 'supabase>=2.0' python-dotenv"
        ) from exc

    if _is_phase1_dev_mode():
        # Phase 1 local dev — impersonate DEV_USER_ID via service role.
        logger.debug("db: using service-role client (Phase 1 dev mode)")
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    # Phase 2+ — anon client, RLS enforced via authed session JWT.
    if not SUPABASE_ANON_KEY:
        raise RuntimeError(
            "SUPABASE_ANON_KEY is not set and no DEV_USER_ID/SERVICE_ROLE "
            "fallback available. See SUPABASE_SETUP.md."
        )
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


@lru_cache(maxsize=1)
def _service_client():
    """Service-role client — bypasses RLS. Server-side only (e.g. Stripe
    webhook worker, prediction-scoring cron). Never expose to a browser."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing. "
            "Only set SUPABASE_SERVICE_ROLE_KEY in trusted server contexts."
        )
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# Per-request user context. The FastAPI auth dependency
# (api.auth.get_current_user) writes the verified user id here at the
# start of every authenticated request. _current_user_id() reads it
# back without needing to thread the value through every db function
# call site explicitly. ContextVars are task-scoped, so concurrent
# requests can't see each other's value.
_request_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "prediqt_request_user_id", default=None,
)


def set_request_user(user_id: Optional[str]) -> None:
    """Stash the authenticated user id for the current request. Called
    from `api.auth.get_current_user` after JWT verification."""
    _request_user_id.set(user_id)


def _current_user_id(user_id: Optional[str] = None) -> str:
    """
    Resolve the user_id for the current request.

    Order of resolution:
      1. Caller-supplied user_id wins. Used by background workers
         (scoring_worker, retrain_pending) that operate on a specific
         user without an HTTP request context.
      2. The per-request ContextVar set by `api.auth.get_current_user`
         on every authenticated FastAPI request.
      3. DEV_USER_ID fallback. Only fires when ALLOW_DEV_USER_FALLBACK=1
         is also set in env. The double-gate keeps a misconfigured
         production deploy from accidentally running every request as
         the dev user.
      4. Otherwise raise.

    Production: ALLOW_DEV_USER_FALLBACK is unset, so this is strict
    (a missing user_id outside a request is a real bug). Local dev:
    ALLOW_DEV_USER_FALLBACK=1 + DEV_USER_ID lets you keep working
    without standing up a real auth flow.
    """
    if user_id:
        return user_id

    ctx_uid = _request_user_id.get()
    if ctx_uid:
        return ctx_uid

    allow_dev = (os.getenv("ALLOW_DEV_USER_FALLBACK", "").strip().lower()
                 in ("1", "true", "yes", "on"))
    if allow_dev and DEV_USER_ID:
        return DEV_USER_ID

    raise RuntimeError(
        "No user_id available. The authenticated endpoint should pass "
        "user_id explicitly or call set_request_user() via the auth "
        "dependency. For local dev without auth: "
        "ALLOW_DEV_USER_FALLBACK=1 + DEV_USER_ID."
    )


# ──────────────────────────────────────────────────────────────────────────
# Public health check — call this once at startup to verify config.
# ──────────────────────────────────────────────────────────────────────────

def backend_info() -> dict:
    """Return a one-shot summary of which backend is active and why."""
    info = {
        "backend": "supabase" if USE_SUPABASE else "file",
        "use_supabase_flag": USE_SUPABASE,
        "supabase_url_set": bool(SUPABASE_URL),
        "supabase_anon_set": bool(SUPABASE_ANON_KEY),
        "supabase_service_role_set": bool(SUPABASE_SERVICE_ROLE_KEY),
        "dev_user_id_set": bool(DEV_USER_ID),
        "phase1_dev_mode": _is_phase1_dev_mode() if USE_SUPABASE else False,
    }
    if USE_SUPABASE:
        try:
            _client()
            info["client_ok"] = True
        except Exception as exc:
            info["client_ok"] = False
            info["client_error"] = str(exc)
    return info


# ═══════════════════════════════════════════════════════════════════════════
# WATCHLIST
# ═══════════════════════════════════════════════════════════════════════════

def load_watchlist(user_id: Optional[str] = None) -> list[str]:
    """Return a list of uppercase ticker symbols for the given user."""
    if not USE_SUPABASE:
        # Legacy path — ignore user_id; there's only one user.
        from watchlist import load_watchlist as _file_load
        return _file_load()

    uid = _current_user_id(user_id)
    try:
        res = (
            _client().table("watchlists")
            .select("symbol")
            .eq("user_id", uid)
            .order("added_at", desc=False)
            .execute()
        )
        return [row["symbol"] for row in (res.data or []) if row.get("symbol")]
    except Exception as exc:
        logger.exception("load_watchlist failed: %s", exc)
        return []


def save_watchlist(symbols: list[str], user_id: Optional[str] = None) -> list[str]:
    """
    Replace the user's watchlist with `symbols`. Returns the cleaned list.
    The file backend writes the whole file; the Supabase backend computes a
    diff and inserts/deletes only what changed, to avoid churning rows.
    """
    clean = [s.upper().strip() for s in (symbols or []) if s]
    clean = list(dict.fromkeys(clean))

    if not USE_SUPABASE:
        from watchlist import save_watchlist as _file_save
        return _file_save(clean)

    uid = _current_user_id(user_id)
    try:
        current = set(load_watchlist(user_id=uid))
        desired = set(clean)
        to_add = desired - current
        to_remove = current - desired

        if to_add:
            _client().table("watchlists").insert([
                {"user_id": uid, "symbol": s} for s in to_add
            ]).execute()

        if to_remove:
            (
                _client().table("watchlists")
                .delete()
                .eq("user_id", uid)
                .in_("symbol", list(to_remove))
                .execute()
            )
        return clean
    except Exception as exc:
        logger.exception("save_watchlist failed: %s", exc)
        return clean


def add_to_watchlist(symbol: str, user_id: Optional[str] = None) -> list[str]:
    """Add a single symbol. Idempotent — existing symbols are a no-op."""
    sym = (symbol or "").upper().strip()
    if not sym:
        return load_watchlist(user_id=user_id)

    if not USE_SUPABASE:
        from watchlist import add_to_watchlist as _file_add
        return _file_add(sym)

    uid = _current_user_id(user_id)
    try:
        _client().table("watchlists").upsert(
            {"user_id": uid, "symbol": sym},
            on_conflict="user_id,symbol",
        ).execute()
    except Exception as exc:
        logger.exception("add_to_watchlist failed: %s", exc)
    return load_watchlist(user_id=uid)


def remove_from_watchlist(symbol: str, user_id: Optional[str] = None) -> list[str]:
    """Remove a single symbol. Idempotent."""
    sym = (symbol or "").upper().strip()
    if not sym:
        return load_watchlist(user_id=user_id)

    if not USE_SUPABASE:
        from watchlist import remove_from_watchlist as _file_remove
        return _file_remove(sym)

    uid = _current_user_id(user_id)
    try:
        (
            _client().table("watchlists")
            .delete()
            .eq("user_id", uid)
            .eq("symbol", sym)
            .execute()
        )
    except Exception as exc:
        logger.exception("remove_from_watchlist failed: %s", exc)
    return load_watchlist(user_id=uid)


# ═══════════════════════════════════════════════════════════════════════════
# PAPER PORTFOLIO  (stubs — to be filled out when we migrate paper_trader.py)
# ═══════════════════════════════════════════════════════════════════════════
# For now these shim through to the file-backed versions so UI keeps working.
# Leaving the signatures here so the surface area is obvious.

def get_portfolio_stats(user_id: Optional[str] = None) -> dict:
    if not USE_SUPABASE:
        from paper_trader import get_portfolio_stats as _file_stats
        return _file_stats()
    # TODO: port paper_trader logic to query paper_portfolios + paper_trades.
    # Not implemented yet — Phase 1 Step 2.
    raise NotImplementedError(
        "Supabase paper-trading backend not yet implemented. Keep "
        "USE_SUPABASE=false until we migrate paper_trader.py."
    )


def update_portfolio(user_id: Optional[str] = None) -> dict:
    if not USE_SUPABASE:
        from paper_trader import update_portfolio as _file_update
        return _file_update()
    raise NotImplementedError("See get_portfolio_stats TODO.")


def reset_portfolio(user_id: Optional[str] = None) -> dict:
    if not USE_SUPABASE:
        from paper_trader import reset_portfolio as _file_reset
        return _file_reset()
    raise NotImplementedError("See get_portfolio_stats TODO.")


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTIONS LOG
# ═══════════════════════════════════════════════════════════════════════════

# Mapping from prediction_logger_v2's long-form horizon names to the codes the
# Supabase predictions schema uses.
_HORIZON_NAME_TO_CODE: dict[str, str] = {
    "3 Day":     "3d",
    "1 Week":    "1w",
    "1 Month":   "1m",
    "1 Quarter": "1q",
    "1 Year":    "1y",
}

_HORIZON_CODE_TO_DAYS: dict[str, int] = {
    "3d": 3, "1w": 7, "1m": 30, "1q": 90, "1y": 365,
}

# Priority for picking the canonical horizon when inserting one row per
# asking event. Mirrors the same priority used in prediction_logger_v2's
# trade-attachment helper so the trade plan's horizon and the persisted
# row's horizon agree.
_HORIZON_PRIORITY: list[str] = ["1 Month", "1 Quarter", "1 Year", "1 Week", "3 Day"]

# Confidence floor for a prediction to land on the public ledger. Matches
# the comment in 0001_init.sql.
_PUBLIC_LEDGER_CONFIDENCE_MIN: float = 55.0
_PUBLIC_LEDGER_DEDUPE_HOURS: int = 1


_HORIZON_CODE_TO_NAME: dict[str, str] = {
    code: name for name, code in _HORIZON_NAME_TO_CODE.items()
}


def _pick_canonical_horizon(record: dict) -> Optional[str]:
    """Return the long-form horizon name (e.g. '1 Month') we'll persist as
    the single Supabase predictions row for this asking event.

    Priority:
      1. If the record carries a `user_horizon_code` (3d|1w|1m|1q|1y),
         honor it — the user asked for that specific horizon, log it.
      2. Otherwise, fall back to the static priority list (1m → 1q → 1y → 1w → 3d)
         for any path that didn't pick a horizon (CLI batch scans, etc.).

    Returns None when neither path yields a usable horizon.
    """
    horizons = record.get("horizons") or {}
    user_code = record.get("user_horizon_code")
    if user_code:
        user_name = _HORIZON_CODE_TO_NAME.get(user_code)
        if user_name and user_name in horizons:
            return user_name
        # If the user picked a horizon the model didn't return (e.g. it
        # was suppressed for low signal), fall through to priority pick
        # so we still log *something* rather than dropping the call.
    for h in _HORIZON_PRIORITY:
        if h in horizons:
            return h
    return None


def _is_public_ledger(
    symbol: str, confidence: float, horizon_code: str, user_id: str
) -> bool:
    """Apply the public-ledger guardrails from 0001_init.sql:
       (a) confidence >= 55
       (b) symbol in canonical universe
       (c) no duplicate (user_id, symbol, horizon) within the last hour.
    Defaults to False on any error — conservative is_public_ledger errs
    toward NOT publishing if anything is uncertain."""
    if confidence < _PUBLIC_LEDGER_CONFIDENCE_MIN:
        return False

    # Canonical universe = S&P 500 + NASDAQ 100 + Dow 30. universe.py
    # exposes a single `get_full_universe()` helper that returns all
    # three lists; we flatten + dedupe here. Earlier code imported a
    # non-existent `canonical_universe` symbol, swallowed the
    # ImportError silently, and treated EVERY prediction as
    # off-universe — meaning is_public_ledger landed False for every
    # ticker, including S&P 500 names. That kept the public ledger
    # empty even after RLS was unblocked.
    try:
        from universe import get_full_universe
        u = get_full_universe()
        canon: set[str] = set()
        for key in ("sp500", "nasdaq100", "dow30"):
            for t in (u.get(key) or []):
                canon.add(str(t).strip().upper())
        if symbol.strip().upper() not in canon:
            return False
    except Exception as exc:
        logger.exception("canonical universe lookup failed: %s", exc)
        return False

    try:
        from datetime import datetime, timezone, timedelta
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=_PUBLIC_LEDGER_DEDUPE_HOURS)
        ).isoformat()
        # Service-role client for the same reason the insert path uses
        # it: backend trusts the auth dep, RLS is redundant for backend
        # reads scoped by explicit user_id. The cached anon `_client()`
        # has no JWT context so RLS would block this even though the
        # user owns the rows.
        res = (
            _service_client().table("predictions")
              .select("id", count="exact")
              .eq("user_id",  user_id)
              .eq("symbol",   symbol)
              .eq("horizon",  horizon_code)
              .gte("created_at", cutoff)
              .execute()
        )
        if (res.count or 0) > 0:
            return False
    except Exception as exc:
        logger.exception("public-ledger dedupe check failed: %s", exc)
        return False

    return True


def find_todays_prediction(
    symbol: str,
    horizon_code: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[str]:
    """
    Anchor § 8.1 same-day dedupe lookup.

    Return the prediction_id of an existing prediction for (user, symbol,
    horizon) on the current UTC day, or None if there isn't one. Callers
    (notably `log_prediction_v2`) use this to skip writing a duplicate
    row when a user re-asks the same symbol+horizon within a UTC day.

    Day boundary is UTC midnight (matches the predictions.created_at
    timezone). Off-market re-asks crossing UTC midnight (e.g. 11:55 PM
    ET ≈ 3:55 AM UTC next day) intentionally do not dedupe — that's a
    new asking day.

    File mode (USE_SUPABASE=false): delegates to prediction_store. The
    legacy SQLite schema has no user_id column and stores all five
    horizons under one prediction_id, so file-mode dedupe is per-symbol
    per-day (slightly broader than Supabase mode).

    Supabase mode (USE_SUPABASE=true): exact match on
    (user_id, symbol, horizon_code, current_utc_day).
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return None

    if not USE_SUPABASE:
        try:
            return prediction_store.find_todays_prediction(sym)
        except Exception as exc:
            logger.exception(
                "prediction_store.find_todays_prediction failed: %s", exc
            )
            return None

    if not horizon_code:
        # Supabase rows persist one canonical horizon per prediction_id;
        # without a horizon code we can't perform a deterministic dedupe.
        return None

    uid = _current_user_id(user_id)
    from datetime import datetime, timezone, timedelta
    now_utc = datetime.now(timezone.utc)
    day_start = (
        now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    ).isoformat()
    day_end = (
        now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        + timedelta(days=1)
    ).isoformat()

    try:
        res = (
            _client().table("predictions")
              .select("id")
              .eq("user_id", uid)
              .eq("symbol", sym)
              .eq("horizon", horizon_code)
              .gte("created_at", day_start)
              .lt("created_at", day_end)
              .order("created_at", desc=False)
              .limit(1)
              .execute()
        )
    except Exception as exc:
        logger.exception(
            "find_todays_prediction lookup failed for %s/%s: %s",
            sym, horizon_code, exc,
        )
        return None

    if res.data:
        return res.data[0].get("id")
    return None


def insert_prediction(
    record: dict,
    user_id: Optional[str] = None,
) -> str:
    """
    Persist a prediction.

    File mode (`USE_SUPABASE=false`): delegates to
    `prediction_store.insert_prediction(record)` so the legacy SQLite
    multi-row-per-asking-event shape continues to work unchanged.

    Supabase mode (`USE_SUPABASE=true`): inserts ONE row into the
    predictions table — the canonical horizon (1 Month preferred,
    falling back through 1 Quarter / 1 Year / 1 Week / 3 Day per
    `_HORIZON_PRIORITY`). The non-canonical horizons are intentionally
    not persisted to Supabase in this iteration; the per-horizon detail
    view continues to read from the SQLite store until a separate
    `prediction_horizons` table is added.

    The record dict is the same shape `prediction_logger_v2.log_prediction_v2`
    builds. Required keys: `prediction_id`, `symbol`, `horizons`,
    `model_version`, `regime`, `timestamp`. Optional keys (anchor § 4.4
    fields, populated by the trade-attachment phase): `traded`,
    `entry_price`, `stop_price`, `target_price`.

    Returns the persisted prediction's id (the UUID Supabase wrote, or
    whatever the SQLite store returns).
    """
    if not USE_SUPABASE:
        prediction_store.insert_prediction(record)
        return record.get("prediction_id", "")

    uid = _current_user_id(user_id)

    # 1. Pick the canonical horizon. Bail if the record has nothing usable.
    canonical = _pick_canonical_horizon(record)
    if canonical is None:
        raise ValueError(
            f"insert_prediction: record has no recognized horizon "
            f"(keys={list((record.get('horizons') or {}).keys())})"
        )

    h = record["horizons"][canonical]
    horizon_code = _HORIZON_NAME_TO_CODE[canonical]

    # 2. Direction from sign of predicted return — with the methodology
    #    § 4.1 deadband. Tiny-magnitude calls (|return| ≤ 0.5%) are
    #    classified Neutral so the public ledger never carries a "Bullish"
    #    label sitting on a near-zero point estimate.
    pred_return = float(h.get("predicted_return", 0) or 0)
    from trade_decision import direction_from_return  # local import to avoid cycles at module load
    direction = direction_from_return(pred_return)

    confidence = float(h.get("confidence", 0) or 0)

    # 3. Public ledger guardrails. Computed Python-side per 0001 comment;
    #    Postgres can't reach the canonical universe set without round-tripping.
    is_public = _is_public_ledger(
        symbol=record["symbol"],
        confidence=confidence,
        horizon_code=horizon_code,
        user_id=uid,
    )

    # 4. Horizon endpoints. starts_at = the timestamp the model produced
    #    the prediction; ends_at = starts_at + the horizon's interval.
    from datetime import datetime, timezone, timedelta
    timestamp_iso = record.get("timestamp")
    if timestamp_iso:
        try:
            starts_at = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
            if starts_at.tzinfo is None:
                starts_at = starts_at.replace(tzinfo=timezone.utc)
        except Exception:
            starts_at = datetime.now(timezone.utc)
    else:
        starts_at = datetime.now(timezone.utc)
    ends_at = starts_at + timedelta(days=_HORIZON_CODE_TO_DAYS[horizon_code])

    # 5a. Pull the canonical-horizon's options strategy out of the record
    #     so it lands on this row. options_strategies is keyed by the
    #     full horizon name ("1 Month", "1 Quarter"...) — same key as
    #     `canonical` from _pick_canonical_horizon. Skip the placeholder
    #     error shape generate_options_report writes when a per-horizon
    #     build fails (e.g. illiquid options chain).
    options_strategy_for_row = None
    options_strategies = record.get("options_strategies") or {}
    if isinstance(options_strategies, dict):
        candidate = options_strategies.get(canonical)
        if isinstance(candidate, dict) and not candidate.get("error"):
            options_strategy_for_row = candidate

    # 5. Build the row. Drop keys with None so Postgres defaults apply
    #    (created_at, verdict='OPEN', rating_*='pending', etc.).
    row = {
        "id":                record["prediction_id"],
        "user_id":           uid,
        "symbol":            record["symbol"],
        "horizon":           horizon_code,
        "predicted_return":  round(pred_return * 100, 4),  # decimal → percent
        "predicted_price":   round(float(h.get("predicted_price", 0) or 0), 4),
        "confidence":        round(confidence, 2),
        "direction":         direction,
        "model_version":     record.get("model_version") or "1.0",
        "regime":            record.get("regime") or "Unknown",
        "is_public_ledger":  is_public,
        "horizon_starts_at": starts_at.isoformat(),
        "horizon_ends_at":   ends_at.isoformat(),
        "traded":            bool(record.get("traded", False)),
        "entry_price":       record.get("entry_price"),
        "stop_price":        record.get("stop_price"),
        "target_price":      record.get("target_price"),
        "options_strategy":  options_strategy_for_row,
    }
    row = {k: v for k, v in row.items() if v is not None}

    # Backend write: use the service-role client to bypass RLS. The
    # cached anon client (`_client()`) has no per-request JWT attached,
    # so RLS rejects every insert with 42501 even when row.user_id is
    # correct. Authorization is enforced by the API layer (auth dep
    # verifies the JWT, predict path passes cu.id explicitly into the
    # row.user_id we set above) — RLS is then redundant for backend
    # writes. Reads from authenticated frontend continue to use _client
    # so RLS still enforces per-user isolation on the public surface.
    res = _service_client().table("predictions").insert(row).execute()

    # supabase-py returns a list; pull the assigned id back. Fall through
    # to the request-side id on the off-chance the response is empty.
    if res.data:
        return res.data[0].get("id") or record["prediction_id"]
    return record["prediction_id"]


def get_accuracy_bands(public_only: bool = True) -> dict:
    """
    Return the three accuracy bands from methodology § 6:
        • all      — every settled prediction
        • traded   — settled predictions where traded=true (conviction calls)
        • passed   — settled predictions where traded=false (watched)

    For each band: counts of HIT / PARTIAL / MISSED, plus rates.

    Output shape (always, even when bands are empty):
        {
            "all":      {"total": N, "hit": h, "partial": p, "missed": m,
                         "hit_rate": float, "partial_rate": float, "miss_rate": float},
            "traded":   {...same...},
            "passed":   {...same...},
            "conviction_lift": float,   # hit_rate(traded) - hit_rate(all)
        }

    File mode raises — the legacy SQLite store doesn't have the `traded`
    column, so the bands are uncomputable. Any caller that needs the new
    bands should run with USE_SUPABASE=true.
    """
    if not USE_SUPABASE:
        raise NotImplementedError(
            "get_accuracy_bands requires Supabase mode (the SQLite legacy "
            "store has no `traded` column). Set USE_SUPABASE=true."
        )

    c = _client()
    q = c.table("predictions").select("verdict, traded").neq("verdict", "OPEN")
    if public_only:
        q = q.eq("is_public_ledger", True)
    rows = (q.execute()).data or []

    def _band(filtered: list[dict]) -> dict:
        total   = len(filtered)
        hit     = sum(1 for r in filtered if r["verdict"] == "HIT")
        partial = sum(1 for r in filtered if r["verdict"] == "PARTIAL")
        missed  = sum(1 for r in filtered if r["verdict"] == "MISSED")
        return {
            "total":         total,
            "hit":           hit,
            "partial":       partial,
            "missed":        missed,
            "hit_rate":      round(hit     / total, 4) if total else 0.0,
            "partial_rate":  round(partial / total, 4) if total else 0.0,
            "miss_rate":     round(missed  / total, 4) if total else 0.0,
        }

    band_all    = _band(rows)
    band_traded = _band([r for r in rows if r.get("traded") is True])
    band_passed = _band([r for r in rows if r.get("traded") is False])

    # Conviction lift — the headline meta-judgment metric from § 6.1.
    conviction_lift = round(band_traded["hit_rate"] - band_all["hit_rate"], 4)

    return {
        "all":             band_all,
        "traded":          band_traded,
        "passed":          band_passed,
        "conviction_lift": conviction_lift,
    }


def get_model_portfolio_summary() -> dict:
    """
    Return the model paper portfolio's headline numbers and the equity curve
    for the Track Record portfolio surface. Public — no auth required.

    Shape:
        {
            "cash":             float,
            "starting_capital": float,
            "open_positions":   int,
            "open_value":       float,    # Σ(qty × entry_price), best-effort
            "portfolio_value":  float,    # cash + open_value
            "total_return_pct": float,    # (portfolio_value - start) / start * 100
            "equity_curve":     [{date, equity}, ...],
            "max_open_positions": int,
        }

    File mode raises — there's no model paper portfolio in the legacy code.
    """
    if not USE_SUPABASE:
        raise NotImplementedError(
            "get_model_portfolio_summary requires Supabase mode (the model "
            "paper portfolio lives in Postgres tables created by 0002)."
        )

    c = _client()
    p_res = (
        c.table("model_paper_portfolio")
         .select("cash, starting_capital, equity_curve, "
                 "max_open_positions, created_at")
         .eq("id", 1)
         .single()
         .execute()
    )
    p = p_res.data or {}
    if not p:
        return {
            "cash": 0, "starting_capital": 0, "open_positions": 0,
            "open_value": 0, "portfolio_value": 0, "total_return_pct": 0.0,
            "equity_curve": [], "max_open_positions": 25,
            "started_at": None, "benchmark_sp500_return_pct": None,
        }

    # Open trades — pull the full row shape for each one so the Track
    # Record "TheBook" panel can render the position list. We keep the
    # query targeted to fields the public-facing surface needs (no
    # exit_price or realised_pnl — those are the closed-trade summary
    # surface). Quote-time current_price is not stored on the row; the
    # frontend can decorate with a live yfinance pull if desired (today
    # we surface the trade plan as it was committed, which is what the
    # ledger row stamps and what we score against).
    t_res = (
        c.table("model_paper_trades")
         .select(
             "id, prediction_id, symbol, direction, entry_price, "
             "qty, notional, target_price, stop_price, opened_at, status"
         )
         .eq("status", "open")
         .order("opened_at", desc=True)
         .execute()
    )
    open_rows = t_res.data or []
    open_value = sum(float(r["qty"]) * float(r["entry_price"]) for r in open_rows)

    cash = float(p["cash"])
    start = float(p["starting_capital"])
    portfolio_value = round(cash + open_value, 2)

    # Started date — when the paper portfolio went live. Surfaces on
    # the landing page as "Tracking since YYYY-MM-DD" so visitors see
    # the model is honest about its age. Coerce to YYYY-MM-DD only —
    # ISO timestamp is too noisy for display.
    started_iso = p.get("created_at")
    started_short: Optional[str] = None
    if isinstance(started_iso, str) and len(started_iso) >= 10:
        started_short = started_iso[:10]

    # S&P 500 benchmark over the same window — methodology § 7
    # ("alongside the portfolio number, two reference numbers are
    # shown: S&P 500 return over same window…"). Best-effort: when
    # the yfinance call fails or there isn't enough history, return
    # None and the UI gracefully shows a dash.
    benchmark_pct: Optional[float] = None
    if started_short:
        try:
            benchmark_pct = _spy_return_since(started_short)
        except Exception as exc:  # noqa: BLE001
            logger.warning("S&P benchmark fetch failed: %s", exc)

    return {
        "cash":               cash,
        "starting_capital":   start,
        "open_positions":     len(open_rows),
        "open_value":         round(open_value, 2),
        "portfolio_value":    portfolio_value,
        "total_return_pct":   round((portfolio_value - start) / start * 100, 4) if start else 0.0,
        "equity_curve":       p.get("equity_curve") or [],
        "max_open_positions": int(p.get("max_open_positions") or 25),
        # New: full open-position rows so the Track Record back face can
        # render the list. Existing consumers that don't read `positions`
        # are unaffected — additive change to the response shape.
        "positions":          open_rows,
        "started_at":         started_short,
        "benchmark_sp500_return_pct": benchmark_pct,
    }


def _spy_return_since(start_date_iso: str) -> Optional[float]:
    """
    Total return on SPY (proxy for S&P 500) from `start_date_iso` to
    today's last close, expressed as a percentage. Used by
    get_model_portfolio_summary to surface a benchmark next to the
    model's portfolio number.

    Returns None when yfinance has no usable data for the window.
    Pure-best-effort — the model's portfolio surface stays useful
    even if the benchmark call fails.
    """
    try:
        import yfinance as yf
    except Exception:  # noqa: BLE001
        return None

    try:
        # Fetch slightly more history than needed; we'll trim by date.
        df = yf.Ticker("SPY").history(period="2y", auto_adjust=True)
    except Exception:  # noqa: BLE001
        return None

    if df is None or df.empty or "Close" not in df.columns:
        return None

    # Find the close on or after start_date.
    start_close: Optional[float] = None
    for idx, row in df.iterrows():
        try:
            d = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
        except Exception:  # noqa: BLE001
            continue
        if d >= start_date_iso:
            v = float(row["Close"])
            if v == v and v > 0:
                start_close = v
                break

    if start_close is None or start_close <= 0:
        return None

    last_close = float(df["Close"].dropna().iloc[-1])
    if not (last_close == last_close) or last_close <= 0:
        return None

    return round((last_close - start_close) / start_close * 100, 4)


def ensure_user_paper_portfolio(user_id: Optional[str] = None) -> str:
    """
    Ensure the user's paper_portfolios row exists. The 0001 migration has
    a profile-creation trigger that auto-inserts an empty portfolio at
    signup, but in dev mode (DEV_USER_ID) we may have a profile that
    pre-dates the trigger. This is a defensive check — cheap, idempotent,
    safe to call every time before reading or writing.

    Returns the resolved user_id.
    """
    if not USE_SUPABASE:
        raise NotImplementedError(
            "ensure_user_paper_portfolio requires Supabase mode."
        )
    uid = _current_user_id(user_id)
    c = _client()
    res = c.table("paper_portfolios").select("user_id").eq("user_id", uid).execute()
    if not (res.data or []):
        try:
            c.table("paper_portfolios").insert({"user_id": uid}).execute()
        except Exception as exc:
            logger.exception(
                "ensure_user_paper_portfolio: insert failed for %s: %s",
                uid, exc,
            )
            raise
    return uid


def get_user_paper_portfolio(user_id: Optional[str] = None) -> dict:
    """
    Read the user's paper portfolio summary + open trades + recent
    closed trades. Mirrors the shape of `get_model_portfolio_summary()`
    so the frontend can re-use the same components against either
    endpoint.

    Returns:
        {
            "available":         True,
            "cash":              float,
            "starting_capital":  float,
            "open_positions":    int,
            "open_value":        float,    # Σ(qty × entry_price)
            "portfolio_value":   float,    # cash + open_value (MTM proxy)
            "total_return_pct":  float,
            "equity_curve":      [{date, equity}, ...],
            "open_trades":       [paper_trades rows...],
            "recent_closed":     [paper_trades rows, last 25...],
        }
    """
    if not USE_SUPABASE:
        raise NotImplementedError(
            "get_user_paper_portfolio requires Supabase mode."
        )

    uid = ensure_user_paper_portfolio(user_id)
    c = _client()

    p_res = (
        c.table("paper_portfolios")
         .select("cash, starting_capital, equity_curve")
         .eq("user_id", uid)
         .single()
         .execute()
    )
    p = p_res.data or {}

    # Defensive SELECT — `kind` and `instrument_data` only exist after
    # migration 0007 lands. If they don't, fall back to the equity-only
    # column set so the portfolio still loads (option features just
    # won't work until the migration is applied).
    OPEN_FIELDS_FULL = (
        "id, symbol, direction, entry_price, qty, opened_at, "
        "stop_loss, take_profit, confidence, horizon, status, "
        "unrealised_pnl, meta, kind, instrument_data"
    )
    OPEN_FIELDS_LEGACY = (
        "id, symbol, direction, entry_price, qty, opened_at, "
        "stop_loss, take_profit, confidence, horizon, status, "
        "unrealised_pnl, meta"
    )
    CLOSED_FIELDS_FULL = (
        "id, symbol, direction, entry_price, exit_price, qty, "
        "opened_at, closed_at, stop_loss, take_profit, status, "
        "realised_pnl, horizon, kind, instrument_data"
    )
    CLOSED_FIELDS_LEGACY = (
        "id, symbol, direction, entry_price, exit_price, qty, "
        "opened_at, closed_at, stop_loss, take_profit, status, "
        "realised_pnl, horizon"
    )

    try:
        open_res = (
            c.table("paper_trades")
             .select(OPEN_FIELDS_FULL)
             .eq("user_id", uid)
             .eq("status", "open")
             .order("opened_at", desc=True)
             .execute()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "paper_trades open SELECT failed with full schema (%s); "
            "retrying without kind/instrument_data — apply migration "
            "0007 to enable option trades.", exc,
        )
        open_res = (
            c.table("paper_trades")
             .select(OPEN_FIELDS_LEGACY)
             .eq("user_id", uid)
             .eq("status", "open")
             .order("opened_at", desc=True)
             .execute()
        )
    open_trades = open_res.data or []

    # Attach a live valuation to each open option trade so the frontend
    # can render "Close at $X.XX" with the current spread visible. Best-
    # effort: a failed pricer call lands as live={ok: False, error: ...}
    # and the row stays open with the close button disabled. Equity rows
    # don't get a live block — server-fetch on equity close lives at the
    # close endpoint, not the portfolio read.
    if any((tr.get("kind") or "").lower() == "option" for tr in open_trades):
        try:
            from options_pricer import value_option_position
        except Exception as exc:  # noqa: BLE001
            logger.warning("options_pricer import failed: %s", exc)
            value_option_position = None  # type: ignore[assignment]
        if value_option_position is not None:
            for tr in open_trades:
                if (tr.get("kind") or "").lower() != "option":
                    continue
                try:
                    tr["live"] = value_option_position(tr)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "value_option_position failed for trade %s: %s",
                        tr.get("id"), exc,
                    )
                    tr["live"] = {
                        "ok": False, "source": "unavailable",
                        "error": f"valuation failed: {exc}",
                    }

    try:
        closed_res = (
            c.table("paper_trades")
             .select(CLOSED_FIELDS_FULL)
             .eq("user_id", uid)
             .neq("status", "open")
             .order("closed_at", desc=True)
             .limit(25)
             .execute()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "paper_trades closed SELECT failed with full schema (%s); "
            "retrying without kind/instrument_data.", exc,
        )
        closed_res = (
            c.table("paper_trades")
             .select(CLOSED_FIELDS_LEGACY)
             .eq("user_id", uid)
             .neq("status", "open")
             .order("closed_at", desc=True)
             .limit(25)
             .execute()
        )
    closed_trades = closed_res.data or []

    cash = float(p.get("cash") or 0)
    starting = float(p.get("starting_capital") or 10000)
    open_value = sum(
        float(r.get("qty") or 0) * float(r.get("entry_price") or 0)
        for r in open_trades
    )
    portfolio_value = round(cash + open_value, 2)
    total_return_pct = (
        round((portfolio_value - starting) / starting * 100, 4)
        if starting else 0.0
    )

    return {
        "available":          True,
        "cash":               cash,
        "starting_capital":   starting,
        "open_positions":     len(open_trades),
        "open_value":         round(open_value, 2),
        "portfolio_value":    portfolio_value,
        "total_return_pct":   total_return_pct,
        "equity_curve":       p.get("equity_curve") or [],
        "open_trades":        open_trades,
        "recent_closed":      closed_trades,
        "max_open_positions": MAX_USER_OPEN_POSITIONS,
    }


# ─── Open paper trade ────────────────────────────────────────────────────────

# 25-position cap mirrors the model's portfolio (anchor § 7). Users
# follow the same discipline — keeps user/model comparisons honest and
# protects against a forever-accumulating open book.
MAX_USER_OPEN_POSITIONS = 25


def open_user_paper_trade(
    prediction_id: str,
    qty: float,
    user_id: Optional[str] = None,
    fill_price: Optional[float] = None,
) -> dict:
    """
    Open a paper trade for the user, derived from a persisted prediction
    row. Returns the inserted trade, the new cash balance, and the actual
    fill price used.

    The trade fills at `fill_price` — the API layer should always supply
    the current market price (server-fetched, never trust the client).
    The prediction's persisted `entry_price` is treated as the model's
    *thesis* entry only; we record it as `meta.planned_entry` so drift
    between thesis and execution is auditable. When `fill_price` is not
    supplied (e.g., from older callers), we fall back to the persisted
    entry — that path is for symmetry with the model's own paper book,
    where fill = thesis by definition.

    Validates (in order):
      1. prediction exists and carries the symbol + thesis levels
      2. qty > 0
      3. fill price is positive
      4. thesis hasn't already played out: for LONG, fill must sit below
         the target and above the stop; for SHORT, the inverse. If the
         market has crossed the target or stop already, the call's R:R
         is moot — refuse with a clear "thesis already played out"
         message instead of silently logging a bad trade.
      5. user has enough cash for qty × fill_price
      6. user doesn't already have an open position in this symbol
      7. user is under the 25-position cap

    Raises ValueError on any guardrail trip with a human-readable
    message; the API endpoint surfaces it as `{ok: false, error: ...}`.
    """
    if not USE_SUPABASE:
        raise NotImplementedError(
            "open_user_paper_trade requires Supabase mode."
        )

    uid = ensure_user_paper_portfolio(user_id)
    c = _client()

    # 1. Fetch the prediction's persisted trade plan.
    pred_res = (
        c.table("predictions")
         .select(
             "id, symbol, direction, horizon, confidence, "
             "entry_price, stop_price, target_price, predicted_return"
         )
         .eq("id", prediction_id)
         .single()
         .execute()
    )
    pred = pred_res.data
    if not pred:
        raise ValueError(f"prediction not found: {prediction_id}")

    symbol = (pred.get("symbol") or "").upper().strip()
    if not symbol:
        raise ValueError("prediction has no symbol")

    planned_entry_raw = pred.get("entry_price")
    planned_entry = (
        float(planned_entry_raw)
        if planned_entry_raw is not None and float(planned_entry_raw) > 0
        else None
    )
    target_price = pred.get("target_price")
    stop_price = pred.get("stop_price")
    confidence = pred.get("confidence")
    horizon = pred.get("horizon")

    # Direction: anchor stamps Bullish/Bearish/Neutral on predictions.
    # Map to LONG/SHORT for the trade row. Neutral defaults to LONG —
    # there's no neutral position to take; the user wouldn't pick this.
    dir_label = (pred.get("direction") or "").lower()
    if dir_label.startswith("bear"):
        direction = "SHORT"
    else:
        direction = "LONG"

    # 2. qty
    qty_f = float(qty or 0)
    if qty_f <= 0:
        raise ValueError("qty must be greater than 0")

    # 3. resolve fill price. Prefer caller-supplied current market price;
    # fall back to the persisted thesis entry if the API hasn't fetched
    # one (legacy / model-trade symmetry path).
    if fill_price is not None and float(fill_price) > 0:
        actual_fill = float(fill_price)
    elif planned_entry is not None:
        actual_fill = planned_entry
    else:
        raise ValueError(
            "no fill price available — the prediction has no persisted "
            "entry and no current market price was supplied."
        )

    # 4. thesis-played-out guardrail. If price has already moved through
    # the target or stop, opening this trade today doesn't reflect the
    # methodology the user is trying to track. The trade plan was for
    # *this* setup, not yesterday's setup at today's price.
    target_f = float(target_price) if target_price is not None else None
    stop_f = float(stop_price) if stop_price is not None else None
    if direction == "LONG":
        if target_f is not None and actual_fill >= target_f:
            raise ValueError(
                f"thesis already played out: {symbol} is at "
                f"${actual_fill:.2f}, at or above the model's "
                f"${target_f:.2f} target. The call's window has closed."
            )
        if stop_f is not None and actual_fill <= stop_f:
            raise ValueError(
                f"thesis invalidated: {symbol} is at ${actual_fill:.2f}, "
                f"at or below the model's ${stop_f:.2f} stop. Opening "
                f"this trade today would already be in the loss zone."
            )
    else:  # SHORT
        if target_f is not None and actual_fill <= target_f:
            raise ValueError(
                f"thesis already played out: {symbol} is at "
                f"${actual_fill:.2f}, at or below the model's "
                f"${target_f:.2f} target. The call's window has closed."
            )
        if stop_f is not None and actual_fill >= stop_f:
            raise ValueError(
                f"thesis invalidated: {symbol} is at ${actual_fill:.2f}, "
                f"at or above the model's ${stop_f:.2f} stop. Opening "
                f"this trade today would already be in the loss zone."
            )

    notional = round(qty_f * actual_fill, 2)

    # 5. cash check
    p_res = (
        c.table("paper_portfolios")
         .select("cash")
         .eq("user_id", uid)
         .single()
         .execute()
    )
    cash = float((p_res.data or {}).get("cash") or 0)
    if cash + 0.005 < notional:  # tiny epsilon for rounding
        raise ValueError(
            f"insufficient cash: ${cash:.2f} available, ${notional:.2f} needed"
        )

    # 6. no double-position on this symbol
    dupe_res = (
        c.table("paper_trades")
         .select("id", count="exact")
         .eq("user_id", uid)
         .eq("symbol", symbol)
         .eq("status", "open")
         .execute()
    )
    if (dupe_res.count or 0) > 0:
        raise ValueError(f"you already have an open position in {symbol}")

    # 7. position cap
    open_count_res = (
        c.table("paper_trades")
         .select("id", count="exact")
         .eq("user_id", uid)
         .eq("status", "open")
         .execute()
    )
    open_count = open_count_res.count or 0
    if open_count >= MAX_USER_OPEN_POSITIONS:
        raise ValueError(
            f"position cap reached: {open_count} of "
            f"{MAX_USER_OPEN_POSITIONS} slots open"
        )

    # Insert the trade. RLS is bypassed by the service-role client in
    # dev mode — when real auth ships, the policies in 0001_init.sql
    # will gate this on auth.uid() = user_id.
    trade_row: Dict[str, Any] = {
        "user_id":    uid,
        "symbol":     symbol,
        "direction":  direction,
        "entry_price": round(actual_fill, 4),
        "qty":        round(qty_f, 4),
        "stop_loss":  round(stop_f, 4) if stop_f is not None else None,
        "take_profit": round(target_f, 4) if target_f is not None else None,
        "confidence": round(float(confidence), 2) if confidence is not None else None,
        "horizon":    horizon,
        "meta":       {
            "prediction_id": prediction_id,
            "planned_entry": (
                round(planned_entry, 4) if planned_entry is not None else None
            ),
            "fill_drift_pct": (
                round((actual_fill - planned_entry) / planned_entry * 100, 2)
                if planned_entry is not None and planned_entry > 0
                else None
            ),
        },
    }
    trade_row = {k: v for k, v in trade_row.items() if v is not None}

    insert_res = c.table("paper_trades").insert(trade_row).execute()
    new_trade = (insert_res.data or [{}])[0]

    # Decrement cash. We do this AFTER the insert so a failed insert
    # doesn't burn the user's cash; if the cash update fails post-insert
    # we leave the trade in place and surface the error — the cash
    # state is recoverable on the next portfolio read.
    new_cash = round(cash - notional, 2)
    c.table("paper_portfolios").update(
        {"cash": new_cash}
    ).eq("user_id", uid).execute()

    return {
        "trade_id":     new_trade.get("id"),
        "trade":        new_trade,
        "cash":         new_cash,
        "notional":     notional,
        "fill_price":   round(actual_fill, 4),
        "planned_entry": (
            round(planned_entry, 4) if planned_entry is not None else None
        ),
    }


def close_user_paper_trade(
    trade_id: str,
    current_price: float,
    user_id: Optional[str] = None,
) -> dict:
    """
    Manually close an open paper trade at the given price. Restores the
    notional to cash and adds the realised P&L. Returns the updated
    trade snapshot + new cash.

    For LONG: pnl = (exit - entry) × qty
    For SHORT: pnl = (entry - exit) × qty

    Validates that the trade exists, is owned by the user, and is open.
    """
    if not USE_SUPABASE:
        raise NotImplementedError(
            "close_user_paper_trade requires Supabase mode."
        )

    uid = _current_user_id(user_id)
    c = _client()

    t_res = (
        c.table("paper_trades")
         .select(
             "id, user_id, symbol, direction, entry_price, qty, status"
         )
         .eq("id", trade_id)
         .single()
         .execute()
    )
    trade = t_res.data
    if not trade:
        raise ValueError(f"trade not found: {trade_id}")
    if trade.get("user_id") != uid:
        raise ValueError("trade does not belong to this user")
    if trade.get("status") != "open":
        raise ValueError(
            f"trade is not open (status={trade.get('status')})"
        )

    qty = float(trade.get("qty") or 0)
    entry = float(trade.get("entry_price") or 0)
    direction = trade.get("direction") or "LONG"
    exit_price = float(current_price or 0)
    if exit_price <= 0:
        raise ValueError("current_price must be positive")

    if direction == "LONG":
        pnl = (exit_price - entry) * qty
    else:
        pnl = (entry - exit_price) * qty
    pnl = round(pnl, 2)

    # Update trade row.
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    c.table("paper_trades").update({
        "status":       "closed",
        "exit_price":   round(exit_price, 4),
        "closed_at":    now_iso,
        "realised_pnl": pnl,
    }).eq("id", trade_id).eq("user_id", uid).execute()

    # Restore cash + add realised P&L.
    p_res = (
        c.table("paper_portfolios")
         .select("cash")
         .eq("user_id", uid)
         .single()
         .execute()
    )
    cash = float((p_res.data or {}).get("cash") or 0)
    notional = round(qty * entry, 2)
    new_cash = round(cash + notional + pnl, 2)
    c.table("paper_portfolios").update(
        {"cash": new_cash}
    ).eq("user_id", uid).execute()

    return {
        "trade_id":    trade_id,
        "exit_price":  round(exit_price, 4),
        "realised_pnl": pnl,
        "cash":        new_cash,
    }


# ─── Option paper trades (Phase B) ───────────────────────────────────────────


def _option_intrinsic_per_contract(legs: list, underlying_price: float) -> float:
    """
    Sum of leg intrinsic values per contract at expiry, signed for long
    (+) vs short (−) positions. Multiplied by 100 (the equity option
    multiplier) gives the dollar payoff per contract.

    BUY  CALL @ K → +max(0, S - K)
    SELL CALL @ K → −max(0, S - K)
    BUY  PUT  @ K → +max(0, K - S)
    SELL PUT  @ K → −max(0, K - S)
    """
    total = 0.0
    S = float(underlying_price)
    for leg in legs or []:
        try:
            action = (leg.get("action") or "").upper()
            kind   = (leg.get("type")   or "").upper()
            K      = float(leg.get("strike") or 0)
            sign   = +1 if action == "BUY" else -1
            if kind == "CALL":
                intrinsic = max(0.0, S - K)
            elif kind == "PUT":
                intrinsic = max(0.0, K - S)
            else:
                continue
            total += sign * intrinsic
        except Exception:
            continue
    return total


def open_user_paper_option_trade(
    prediction_id: str,
    qty: int,
    user_id: Optional[str] = None,
) -> dict:
    """
    Open an options paper trade against a persisted prediction's
    options_strategy. Locks cash equal to the strategy's max-loss-per-
    contract × qty (per Phase B design — cash_at_risk model).

    qty is the number of contracts. Each contract represents 100 shares
    of the underlying per CBOE spec — the strategy's premium and P&L
    math are per-contract dollar amounts.

    Validates (in order):
      1. prediction exists with a tradable options_strategy
      2. qty > 0
      3. user has enough cash for cash_at_risk
      4. user doesn't already have an open option in the same symbol+strategy
      5. user is under the 25-position cap

    Returns the inserted trade + new cash + receipt fields. On any
    guardrail trip, raises ValueError with a human-readable message.
    """
    if not USE_SUPABASE:
        raise NotImplementedError(
            "open_user_paper_option_trade requires Supabase mode."
        )

    uid = ensure_user_paper_portfolio(user_id)
    c = _client()

    # 1. Prediction + strategy lookup
    pred_res = (
        c.table("predictions")
         .select("id, symbol, horizon, options_strategy")
         .eq("id", prediction_id)
         .single()
         .execute()
    )
    pred = pred_res.data
    if not pred:
        raise ValueError(f"prediction not found: {prediction_id}")
    symbol = (pred.get("symbol") or "").upper().strip()
    if not symbol:
        raise ValueError("prediction has no symbol")

    strategy = pred.get("options_strategy")
    if not strategy or not isinstance(strategy, dict):
        raise ValueError(
            "this prediction has no persisted options strategy — only "
            "predictions made after the Phase B migration carry one. "
            "Re-ask on /prediqt to get a fresh strategy."
        )
    numeric = strategy.get("numeric") or {}
    if not numeric.get("tradable"):
        raise ValueError(
            f"the {strategy.get('strategy', 'options')} strategy isn't "
            "available for paper trading yet."
        )

    # Asking-flow estimates — Black-Scholes prices the model produced at
    # request time. We persist them as planned_* on the trade row for
    # audit and drift-reporting, but the LIVE values below win for cash
    # math, persisted entry, and everything user-facing. See
    # methodology § 4.5 (open path) — same posture as equity.
    planned_cost_per_contract     = float(numeric.get("cost_per_contract") or 0)
    planned_max_loss_per_contract = float(numeric.get("max_loss_per_contract") or 0)
    planned_is_credit             = bool(numeric.get("is_credit"))
    planned_breakevens            = list(numeric.get("breakevens") or [])
    planned_max_profit_per_contract = numeric.get("max_profit_per_contract")
    if planned_max_loss_per_contract <= 0:
        raise ValueError(
            "strategy has no defined max-loss — paper trading requires "
            "a finite worst case."
        )

    # 2. qty (contracts)
    try:
        qty_int = int(qty)
    except Exception:
        raise ValueError("qty must be a positive integer (number of contracts)")
    if qty_int <= 0:
        raise ValueError("qty must be greater than 0")

    strategy_type = strategy.get("strategy") or "Unknown"

    # 3. Snap expiry to a real chain date (no tolerance — closest wins).
    # See methodology § 4.5 (snap-to-chain).
    from datetime import datetime, timezone, timedelta
    expiry_days = int(strategy.get("expiry_days") or 30)
    now_utc = datetime.now(timezone.utc)
    requested_expiry_dt = now_utc + timedelta(days=expiry_days)
    requested_expiry_date = requested_expiry_dt.strftime("%Y-%m-%d")

    try:
        from options_pricer import snap_expiry_to_chain
        snapped = snap_expiry_to_chain(symbol, requested_expiry_date)
    except Exception as exc:  # noqa: BLE001
        logger.warning("open option trade: snap call failed: %s", exc)
        snapped = None

    if snapped is None:
        raise ValueError(
            f"no live options chain available for {symbol} — can't open "
            "an options paper trade without a real expiry. Try a more "
            "liquid name or come back when the chain feed is reachable."
        )
    expiry_date = snapped
    try:
        snapped_dt = datetime.strptime(expiry_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        expiry_days_actual = max(0, (snapped_dt.date() - now_utc.date()).days)
    except (TypeError, ValueError):
        expiry_days_actual = expiry_days

    # 4. Live-price every leg against the snapped chain. Refuses if any
    # leg can't fill at mid or last — we never lock in entry premiums
    # the system can't honestly stand behind.
    try:
        from options_pricer import price_legs_at_open
        priced = price_legs_at_open(
            symbol, expiry_date, strategy.get("legs") or [],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("open option trade: live leg pricing crashed")
        raise ValueError(f"couldn't price live legs: {exc}") from exc

    if not priced.get("ok"):
        raise ValueError(
            f"can't open at live mid: {priced.get('error') or 'unknown error'}"
        )

    # Build the live-legs list. Each leg keeps its action/type/strike,
    # but `premium` becomes the live used_price (in per-share dollars,
    # matching options_analyzer's leg shape) and we attach the per-leg
    # bid/ask receipt for the trade row.
    planned_legs = list(strategy.get("legs") or [])
    priced_legs = priced.get("per_leg") or []
    if len(priced_legs) != len(planned_legs):
        raise ValueError(
            "live leg count mismatch — pricer returned "
            f"{len(priced_legs)} legs, strategy has {len(planned_legs)}"
        )

    live_legs: list = []
    for orig, p in zip(planned_legs, priced_legs):
        # Strike snap: priced["strike"] is the chain-real strike the
        # leg was actually filled against; original_strike captures
        # what the asking flow asked for, when they differ. Same audit
        # pattern as expiry snap and premium drift.
        snapped_strike = p.get("strike")
        original_strike = p.get("original_strike")
        live_legs.append({
            **orig,
            # If a strike snap happened, the persisted strike is the
            # snapped one; the asked-for strike survives as
            # original_strike. recompute_strategy_numerics reads the
            # leg's `strike` and `premium` and produces correct
            # breakevens / max-loss against the snapped values.
            "strike": snapped_strike if snapped_strike is not None else orig.get("strike"),
            "original_strike": original_strike,  # only set when snap fired; null otherwise
            # Estimated premium from the asking flow, kept for audit.
            "estimated_premium": orig.get("premium"),
            # Live premium replaces the asking-flow estimate.
            "premium": p.get("used_price"),
            "live_quote": {
                "bid":         p.get("bid"),
                "ask":         p.get("ask"),
                "last":        p.get("last"),
                "used_price":  p.get("used_price"),
                "source":      p.get("source"),
                "chain_expiry": p.get("chain_expiry"),
                "snapped_strike": snapped_strike,
                "original_strike": original_strike,
            },
        })

    # 5. Recompute strategy numerics from live legs.
    try:
        from options_analyzer import recompute_strategy_numerics
        live_numeric = recompute_strategy_numerics(strategy_type, live_legs)
    except Exception as exc:  # noqa: BLE001
        logger.exception("open option trade: live numerics recompute failed")
        raise ValueError(f"couldn't reshape strategy from live legs: {exc}") from exc

    cost_per_contract     = float(live_numeric.get("cost_per_contract") or 0)
    max_loss_per_contract = float(live_numeric.get("max_loss_per_contract") or 0)
    is_credit             = bool(live_numeric.get("is_credit"))
    if max_loss_per_contract <= 0:
        raise ValueError(
            "recomputed numerics report no defined max-loss — refusing "
            "to open at an undefined risk."
        )

    # cash_at_risk under the LIVE numbers.
    if is_credit:
        cash_locked_per_contract = round(max_loss_per_contract - abs(cost_per_contract), 2)
    else:
        cash_locked_per_contract = round(cost_per_contract, 2)
    cash_locked = round(cash_locked_per_contract * qty_int, 2)
    if cash_locked < 0:
        raise ValueError("computed negative cash-at-risk — strategy data is malformed")

    # 6. cash check
    p_res = (
        c.table("paper_portfolios")
         .select("cash")
         .eq("user_id", uid)
         .single()
         .execute()
    )
    cash = float((p_res.data or {}).get("cash") or 0)
    if cash + 0.005 < cash_locked:
        raise ValueError(
            f"insufficient cash: ${cash:.2f} available, "
            f"${cash_locked:.2f} needed to open this {strategy_type} "
            f"at live prices"
        )

    # 7. dupe check — same symbol + same strategy already open?
    existing_res = (
        c.table("paper_trades")
         .select("id, instrument_data")
         .eq("user_id", uid)
         .eq("symbol", symbol)
         .eq("status", "open")
         .eq("kind", "option")
         .execute()
    )
    for row in (existing_res.data or []):
        ins = row.get("instrument_data") or {}
        if ins.get("strategy_type") == strategy_type:
            raise ValueError(
                f"you already have an open {strategy_type} on {symbol}"
            )

    # 8. position cap (combined equity + option count)
    open_count_res = (
        c.table("paper_trades")
         .select("id", count="exact")
         .eq("user_id", uid)
         .eq("status", "open")
         .execute()
    )
    open_count = open_count_res.count or 0
    if open_count >= MAX_USER_OPEN_POSITIONS:
        raise ValueError(
            f"position cap reached: {open_count} of "
            f"{MAX_USER_OPEN_POSITIONS} slots open"
        )

    # Cost drift vs the asking-flow estimate. Receipt surfaces this so
    # the user sees the gap between what they were quoted and what they
    # filled at — same pattern as equity drift.
    cost_drift_pct: Optional[float]
    if planned_cost_per_contract not in (0, 0.0):
        cost_drift_pct = round(
            (cost_per_contract - planned_cost_per_contract)
            / abs(planned_cost_per_contract) * 100.0,
            2,
        )
    else:
        cost_drift_pct = None

    # Build instrument_data (everything the settlement path needs)
    instrument_data: Dict[str, Any] = {
        "strategy_type":            strategy_type,
        "direction":                strategy.get("direction"),
        "complexity":               strategy.get("complexity"),
        "expiry_days":              expiry_days,            # user's requested horizon
        "expiry_days_actual":       expiry_days_actual,     # after snap
        "expiry_date":              expiry_date,            # real chain date (post-snap)
        "original_expiry_date":     requested_expiry_date,  # pre-snap, for audit
        "chain_expiry":             priced.get("chain_expiry"),  # what legs priced against
        "legs":                     live_legs,
        "premium_per_contract":     round(cost_per_contract, 2),
        "max_profit_per_contract":  live_numeric.get("max_profit_per_contract"),
        "max_loss_per_contract":    round(max_loss_per_contract, 2),
        "breakevens":               live_numeric.get("breakevens") or [],
        "is_credit":                is_credit,
        "cash_locked_per_contract": cash_locked_per_contract,
        "cash_locked":              cash_locked,
        "rationale":                strategy.get("rationale"),
        "risk_note":                strategy.get("risk_note"),
        # Asking-flow shadow — what the user was quoted before we hit
        # the live chain. Audit only; live values above are authoritative.
        "planned": {
            "cost_per_contract":       round(planned_cost_per_contract, 2),
            "max_loss_per_contract":   round(planned_max_loss_per_contract, 2),
            "max_profit_per_contract": planned_max_profit_per_contract,
            "breakevens":              planned_breakevens,
            "is_credit":               planned_is_credit,
        },
        "cost_drift_pct":           cost_drift_pct,
    }

    # Direction LONG/SHORT for top-level field — options spreads aren't
    # cleanly long/short, but we need a value. Pick LONG for debit
    # strategies (we own them), SHORT for credit (we sold them).
    top_direction = "SHORT" if is_credit else "LONG"

    trade_row: Dict[str, Any] = {
        "user_id":      uid,
        "symbol":       symbol,
        "kind":         "option",
        "direction":    top_direction,
        # entry_price is per-contract premium in dollars (positive for
        # debit, negative for credit). Reused so the column has meaning
        # without adding option-specific columns.
        "entry_price":  round(cost_per_contract, 4),
        "qty":          qty_int,
        "horizon":      strategy.get("horizon"),
        "instrument_data": instrument_data,
        "meta": {
            "prediction_id": prediction_id,
        },
    }
    trade_row = {k: v for k, v in trade_row.items() if v is not None}

    insert_res = c.table("paper_trades").insert(trade_row).execute()
    new_trade = (insert_res.data or [{}])[0]

    # Lock cash for the worst-case loss. For debit strategies this is
    # the premium paid; for credit strategies it's the margin gap.
    new_cash = round(cash - cash_locked, 2)
    c.table("paper_portfolios").update(
        {"cash": new_cash}
    ).eq("user_id", uid).execute()

    return {
        "trade_id":           new_trade.get("id"),
        "trade":              new_trade,
        "cash":               new_cash,
        "cash_locked":        cash_locked,
        "premium_per_contract":       round(cost_per_contract, 2),
        "planned_premium_per_contract": round(planned_cost_per_contract, 2),
        "cost_drift_pct":     cost_drift_pct,
        "is_credit":          is_credit,
        "expiry_date":        expiry_date,
        "expiry_days_actual": expiry_days_actual,
        "expiry_days_requested": expiry_days,
        "chain_expiry":       priced.get("chain_expiry"),
        "strategy_type":      strategy_type,
    }


def settle_expired_option_trades(user_id: Optional[str] = None) -> int:
    """
    Walk all open option trades for the user, settle any whose expiry
    date has passed. Settlement: resolve underlying close on the
    expiry date, compute realised P&L from the legs' intrinsic values
    at that price, mark the trade closed, refund locked cash + apply
    P&L to the user's cash balance.

    Returns the count of settled trades.

    Called automatically from /api/paper-portfolio so the user sees
    the resolved state without having to do anything special.
    """
    if not USE_SUPABASE:
        return 0

    uid = _current_user_id(user_id)
    c = _client()

    # Pull all open option trades for this user. If the kind column
    # doesn't exist yet (migration 0007 not applied), no-op — there
    # can't be any options trades to settle anyway.
    try:
        res = (
            c.table("paper_trades")
             .select("id, symbol, qty, instrument_data, entry_price, kind, status")
             .eq("user_id", uid)
             .eq("status", "open")
             .eq("kind", "option")
             .execute()
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "settle_expired_option_trades skipped — paper_trades.kind "
            "column missing (apply migration 0007). Error: %s", exc,
        )
        return 0
    rows = res.data or []
    if not rows:
        return 0

    from datetime import datetime, timezone
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Late import — keep db.py free of yfinance at module-import time.
    try:
        from data_fetcher import fetch_stock_data
    except Exception as exc:  # noqa: BLE001
        logger.warning("settle_expired_option_trades: data_fetcher import failed: %s", exc)
        return 0

    settled = 0
    for tr in rows:
        ins = tr.get("instrument_data") or {}
        expiry_date = ins.get("expiry_date")
        if not expiry_date or expiry_date >= today_utc:
            continue  # not yet expired

        # Resolve underlying close on or before expiry day.
        sym = (tr.get("symbol") or "").upper().strip()
        try:
            df = fetch_stock_data(sym, period="1y")
            settle_price = None
            if df is not None and "Close" in df.columns:
                # Walk back from latest to find a close ≤ expiry_date.
                for idx, prow in df.iloc[::-1].iterrows():
                    try:
                        d = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                    except Exception:
                        continue
                    if d <= expiry_date:
                        v = float(prow["Close"])
                        if v == v and v > 0:
                            settle_price = round(v, 4)
                        break
        except Exception as exc:  # noqa: BLE001
            logger.warning("settle %s: resolve close failed: %s", tr.get("id"), exc)
            settle_price = None

        if settle_price is None:
            # Couldn't get a settle price; leave the trade open and try
            # again next portfolio read.
            continue

        # Compute realised P&L per contract.
        legs = ins.get("legs") or []
        intrinsic_per = _option_intrinsic_per_contract(legs, settle_price)
        cost_per      = float(ins.get("premium_per_contract") or 0)
        # Unified formula: pnl_per_contract = intrinsic*100 − cost
        pnl_per       = round(intrinsic_per * 100 - cost_per, 2)
        qty           = int(float(tr.get("qty") or 0))
        total_pnl     = round(pnl_per * qty, 2)
        cash_locked   = float(ins.get("cash_locked") or 0)

        # Update the trade row.
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            c.table("paper_trades").update({
                "status":       "closed",
                "exit_price":   settle_price,  # underlying close at expiry
                "closed_at":    now_iso,
                "realised_pnl": total_pnl,
            }).eq("id", tr.get("id")).eq("user_id", uid).execute()

            # Refund locked cash + apply realised P&L.
            p_res = (
                c.table("paper_portfolios")
                 .select("cash")
                 .eq("user_id", uid)
                 .single()
                 .execute()
            )
            cur_cash = float((p_res.data or {}).get("cash") or 0)
            new_cash = round(cur_cash + cash_locked + total_pnl, 2)
            c.table("paper_portfolios").update(
                {"cash": new_cash}
            ).eq("user_id", uid).execute()
            settled += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("settle %s: write failed: %s", tr.get("id"), exc)
            continue

    return settled


def backfill_option_expiry_dates(*, dry_run: bool = False) -> dict:
    """
    One-shot fix for legacy option trades whose `expiry_date` was
    computed as `today + N days` (calendar arithmetic) and so doesn't
    exist on real options chains. Walks every open option paper trade,
    snaps the recorded date to the nearest real chain expiry via
    options_pricer.snap_expiry_to_chain, and persists the update.

    Preserves the original guess as `instrument_data.original_expiry_date`
    so the audit trail stays intact. Idempotent — running twice is a
    no-op once everything's snapped.

    Returns a summary dict for logs:
        {
            "checked":    N,
            "updated":    int,
            "unchanged":  int,    # already on a real chain (or already snapped)
            "unsnappable": int,   # no chain available for symbol
            "errors":     [{"trade_id": ..., "error": ...}, ...],
        }

    `dry_run=True` runs the snap and reports counts without writing.
    """
    if not USE_SUPABASE:
        raise NotImplementedError(
            "backfill_option_expiry_dates requires Supabase mode."
        )

    c = _client()
    summary: dict = {
        "checked":     0,
        "updated":     0,
        "unchanged":   0,
        "unsnappable": 0,
        "errors":      [],
    }

    try:
        from options_pricer import snap_expiry_to_chain
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"options_pricer import failed: {exc}") from exc

    rows = (
        c.table("paper_trades")
         .select("id, symbol, instrument_data, user_id")
         .eq("status", "open")
         .eq("kind", "option")
         .execute()
    ).data or []

    summary["checked"] = len(rows)

    for tr in rows:
        tid = tr.get("id")
        try:
            symbol = (tr.get("symbol") or "").upper().strip()
            ins = tr.get("instrument_data") or {}
            current = ins.get("expiry_date")
            if not symbol or not current:
                summary["errors"].append({
                    "trade_id": tid, "error": "missing symbol or expiry_date",
                })
                continue

            snapped = snap_expiry_to_chain(symbol, current)
            if snapped is None:
                summary["unsnappable"] += 1
                continue

            if snapped == current:
                summary["unchanged"] += 1
                continue

            new_ins = {**ins, "expiry_date": snapped}
            # Only stamp the original on the first snap — don't clobber
            # an audit trail that's already there.
            if not new_ins.get("original_expiry_date"):
                new_ins["original_expiry_date"] = current

            if dry_run:
                summary["updated"] += 1
                continue

            c.table("paper_trades").update({
                "instrument_data": new_ins,
            }).eq("id", tid).execute()
            summary["updated"] += 1

        except Exception as exc:  # noqa: BLE001
            logger.exception("backfill_option_expiry_dates: trade %s failed", tid)
            summary["errors"].append({
                "trade_id": tid, "error": str(exc)[:200],
            })

    return summary


def close_user_paper_option_trade(
    trade_id: str,
    user_id: Optional[str] = None,
) -> dict:
    """
    Close an open user option trade at the live mid-price (or last /
    intrinsic fallback) determined by `options_pricer.value_option_position`.

    The methodology's no-early-close rule binds `model_paper_trades` only
    — see anchors/02-methodology.md § 4.2. User paper trades are the
    user's book, so this path is allowed and is the option-side
    counterpart to `close_user_paper_trade` for equity.

    The fill price is server-fetched at the moment of close — the
    endpoint never trusts a price the client supplies. The close receipt
    captures bid/ask/source per leg in `instrument_data.close_quote` so
    the user can see exactly what spread they crossed.

    Refuses cleanly (raises ValueError) when the pricer can't get a
    quote — silent fallbacks to fictional prices would let the system
    settle at numbers we can't defend.
    """
    if not USE_SUPABASE:
        raise NotImplementedError(
            "close_user_paper_option_trade requires Supabase mode."
        )

    uid = _current_user_id(user_id)
    c = _client()

    t_res = (
        c.table("paper_trades")
         .select(
             "id, user_id, symbol, qty, kind, status, entry_price, "
             "instrument_data"
         )
         .eq("id", trade_id)
         .single()
         .execute()
    )
    trade = t_res.data
    if not trade:
        raise ValueError(f"trade not found: {trade_id}")
    if trade.get("user_id") != uid:
        raise ValueError("trade does not belong to this user")
    if trade.get("status") != "open":
        raise ValueError(
            f"trade is not open (status={trade.get('status')})"
        )
    if trade.get("kind") != "option":
        raise ValueError(
            "this trade is not an option — use the equity close path"
        )

    from options_pricer import value_option_position
    valuation = value_option_position(trade)
    if not valuation.get("ok"):
        # The pricer couldn't get a usable quote — the user gets a clear
        # refusal rather than a fictional fill. They can retry; if the
        # chain comes back, so does the close path.
        raise ValueError(
            f"can't price this position right now — {valuation.get('error') or 'unavailable'}"
        )

    pnl_total   = float(valuation["pnl_total"])
    cash_back   = float(valuation["cash_back"])
    liq_per     = float(valuation["liq_per_contract"])
    cash_locked = float((trade.get("instrument_data") or {}).get("cash_locked") or 0)

    # Build the close receipt: enough detail to audit the fill later.
    close_quote = {
        "method":            "early_mid",
        "source":            valuation["source"],
        "liq_per_contract":  liq_per,
        "position_value":    valuation["position_value"],
        "pnl_per_contract":  valuation["pnl_per_contract"],
        "pnl_total":         pnl_total,
        "cash_back":         cash_back,
        "per_leg":           valuation["per_leg"],
        "fetched_at":        valuation["fetched_at"],
    }

    # Merge close_quote into existing instrument_data (don't clobber the
    # rest of the option metadata — strategy_type, legs, etc.).
    new_instrument_data = {
        **(trade.get("instrument_data") or {}),
        "close_quote": close_quote,
    }

    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()

    c.table("paper_trades").update({
        "status":          "closed",
        "exit_price":      round(liq_per, 4),  # dollars/contract — symmetric with entry_price
        "closed_at":       now_iso,
        "realised_pnl":    round(pnl_total, 2),
        "instrument_data": new_instrument_data,
    }).eq("id", trade_id).eq("user_id", uid).execute()

    # Refund locked cash + apply realised P&L, mirroring the settle path.
    p_res = (
        c.table("paper_portfolios")
         .select("cash")
         .eq("user_id", uid)
         .single()
         .execute()
    )
    cur_cash = float((p_res.data or {}).get("cash") or 0)
    new_cash = round(cur_cash + cash_locked + pnl_total, 2)
    c.table("paper_portfolios").update(
        {"cash": new_cash}
    ).eq("user_id", uid).execute()

    return {
        "trade_id":         trade_id,
        "exit_price":       round(liq_per, 4),
        "realised_pnl":     round(pnl_total, 2),
        "cash":             new_cash,
        "cash_back":        round(cash_back, 2),
        "source":           valuation["source"],
        "per_leg":          valuation["per_leg"],
        "liq_per_contract": liq_per,
        "position_value":   valuation["position_value"],
    }


def get_prediction_detail(prediction_id: str) -> Optional[dict]:
    """
    Full row for a single prediction, keyed by its Supabase UUID.

    Returns *all* fields — including paid-tier ones (entry_price, stop_price,
    target_price, predicted_return, predicted_price, confidence,
    model_version, regime, etc.).

    TODO (Phase 2 — auth gating): when accounts ship, this function should
    accept the requesting user's tier and strip paid-tier fields when the
    requester isn't a paid member. Today we have one user and no auth, so
    everything is returned and the consumer (frontend) shows it.

    Returns None when the row doesn't exist — caller can treat as 404.
    """
    if not USE_SUPABASE:
        raise NotImplementedError(
            "get_prediction_detail requires Supabase mode."
        )

    res = (
        _client().table("predictions")
            .select("*")
            .eq("id", prediction_id)
            .single()
            .execute()
    )
    return res.data or None


def get_public_ledger(
    limit: int = 50,
    offset: int = 0,
    only_settled: bool = False,
) -> list[dict]:
    """
    Public-ledger rows — the public-tier view per data-model anchor § 6.

    Returns a list of dicts with ONLY the public fields:
        id, symbol, direction, horizon, verdict, traded,
        created_at, horizon_ends_at

    No confidence, no entry/stop/target, no model_version or regime.
    Members and paid users get richer reads via separate functions
    (still TODO).

    Filters:
        only_settled=True restricts to verdict != 'OPEN'.
    """
    if not USE_SUPABASE:
        raise NotImplementedError(
            "get_public_ledger requires Supabase mode."
        )

    q = (
        _client().table("predictions")
            .select("id, symbol, direction, horizon, verdict, traded, "
                    "created_at, horizon_ends_at")
            .eq("is_public_ledger", True)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
    )
    if only_settled:
        q = q.neq("verdict", "OPEN")

    return (q.execute()).data or []


def _supabase_prediction_counts(public_only: bool = False) -> dict:
    """
    Live counters pulled directly from the Supabase predictions table.
    These overlay the stale SQLite-era counts in get_full_analytics().

    Why this exists: the legacy SQLite store stopped being written to at
    the 2026-04-25 cutover, so any analytics field that flows through
    prediction_logger_v2.get_full_analytics() (which reads SQLite) is
    frozen at that snapshot. The landing-page "Calls on the record"
    widget showed stale 56 / 263 / 25 even when Supabase had grown well
    past those counts. Routing these specific counters through Supabase
    fixes the user-visible numbers without requiring the full per-horizon
    detail migration to ship first.

    Each Supabase predictions row is one canonical horizon per asking
    event (see `insert_prediction` — only `_pick_canonical_horizon`'s
    pick is persisted), so `asking_events_total == total_predictions`
    in Supabase mode. That's a deliberate semantic shift from SQLite
    where total_predictions was per-horizon (5× asking_events on
    average).

    Returns an empty dict when not in Supabase mode or on any error so
    callers can fall through to the legacy values without special-casing.
    """
    if not USE_SUPABASE:
        return {}
    try:
        client = _client()

        def _count(filter_fn=None) -> int:
            q = client.table("predictions").select("id", count="exact")
            if public_only:
                q = q.eq("is_public_ledger", True)
            if filter_fn is not None:
                q = filter_fn(q)
            try:
                res = q.execute()
                return int(res.count or 0)
            except Exception as exc:
                logger.exception("supabase count query failed: %s", exc)
                return 0

        total = _count()

        # ── Three honesty reads, all from Supabase rating_* columns ─────────
        # Methodology § 4.3 / § 5: each prediction carries three independent
        # ratings (rating_target / rating_checkpoint / rating_expiration),
        # each in {'pending', 'hit', 'miss'}. The Scoreboard's three cards
        # map 1:1 to these. Computing rates straight from the rating_*
        # columns means every card uses the same units (per-asking-event,
        # post-cutover Supabase) — no SQLite-era per-horizon mix-in.
        #
        # Until predictions mature past their windows the rating fields
        # stay 'pending', so all three rates honestly read "—" / "0 of 0"
        # in the UI. That's the expected option-1 empty state for a fresh
        # ledger; rates fill in as windows close.

        # Generous — Target Hit Rate (anchor § 4.3 rating_target).
        target_hit = _count(lambda q: q.eq("rating_target", "hit"))
        target_miss = _count(lambda q: q.eq("rating_target", "miss"))
        target_definitive = target_hit + target_miss
        target_hit_rate = (target_hit / target_definitive * 100) if target_definitive > 0 else 0.0

        # Medium — Checkpoint Win Rate (rating_checkpoint).
        checkpoint_hit = _count(lambda q: q.eq("rating_checkpoint", "hit"))
        checkpoint_miss = _count(lambda q: q.eq("rating_checkpoint", "miss"))
        scored_any = checkpoint_hit + checkpoint_miss
        quick_accuracy = (checkpoint_hit / scored_any * 100) if scored_any > 0 else 0.0

        # Strict — Expiration Win Rate (rating_expiration).
        expiration_hit = _count(lambda q: q.eq("rating_expiration", "hit"))
        expiration_miss = _count(lambda q: q.eq("rating_expiration", "miss"))
        scored_final = expiration_hit + expiration_miss
        live_accuracy = (expiration_hit / scored_final * 100) if scored_final > 0 else 0.0

        return {
            # Each Supabase row is one asking event. Surface the same
            # number under both legacy keys so consumers reading either
            # one (asking_events_total / total_predictions) see truth.
            "total_analyses":          total,
            "total_predictions":       total,

            # Target (Generous). target_hit_rate is read directly by
            # api/main.py:228 — overlaying it here means we don't need to
            # also overlay it after the target_hit_analyzer runs.
            "target_hit_rate":         round(target_hit_rate, 1),
            "target_hit_count":        target_hit,
            "target_definitive":       target_definitive,

            # Checkpoint (Medium). api/main.py derives checkpoint_win_rate
            # from `direction_correct_any / scored_any` — wiring those
            # fields to checkpoint_hit / scored_any gives the right rate.
            "scored_any":              scored_any,
            "scored_quick":            scored_any,
            "direction_correct_any":   checkpoint_hit,
            "direction_correct_quick": checkpoint_hit,
            "quick_accuracy":          round(quick_accuracy, 1),

            # Expiration (Strict). expiration_win_rate is read from
            # live_accuracy at api/main.py:236.
            "scored_final":            scored_final,
            "direction_correct_final": expiration_hit,
            "live_accuracy":           round(live_accuracy, 1),
        }
    except Exception as exc:
        logger.exception("_supabase_prediction_counts failed: %s", exc)
        return {}


def get_full_analytics(public_only: bool = False) -> dict:
    """
    Returns the same shape as prediction_logger_v2.get_full_analytics.
    public_only=True restricts to predictions with is_public_ledger=true
    (this powers the public Track Record view).

    File mode delegates to the legacy SQLite-backed analytics.

    Supabase mode returns a HYBRID shape during the cutover:
      • The new Track Record bits (`accuracy_bands`, `portfolio`, `ledger`)
        come from the Supabase tables — accurate, current, includes traded.
      • Top-line counts (`total_analyses`, `total_predictions`, `scored_any`,
        `scored_final`, `direction_correct_*`, `live_accuracy`) come from
        Supabase via `_supabase_prediction_counts` and overlay the stale
        legacy SQLite values.
      • The remaining legacy bits (`per_horizon`, `per_symbol`,
        `confidence_calibration`, `regime_accuracy`, `accuracy_over_time`,
        `predictions_table`) currently fall back to the SQLite analytics
        so existing per-horizon UI keeps rendering. These will move to
        Supabase once the per-horizon detail table lands (separate
        migration, not blocking Phase 1 backend completion).
    """
    if not USE_SUPABASE:
        from prediction_logger_v2 import get_full_analytics as _file_analytics
        return _file_analytics()

    # Hybrid: new shape from Supabase, legacy shape from SQLite for
    # things not yet migrated. The legacy fallback is best-effort —
    # if the SQLite store is empty, we return empty placeholders.
    try:
        from prediction_logger_v2 import get_full_analytics as _file_analytics
        legacy = _file_analytics()
    except Exception as exc:
        logger.exception("legacy analytics fallback failed: %s", exc)
        legacy = {}

    bands     = get_accuracy_bands(public_only=public_only)
    portfolio = get_model_portfolio_summary()
    ledger    = get_public_ledger(limit=100, offset=0)
    counts    = _supabase_prediction_counts(public_only=public_only)

    # Overlay the new bits on the legacy shape. Supabase wins for any keys
    # we explicitly compute here — including the top-line counts that
    # otherwise read as the frozen 2026-04-25 SQLite snapshot.
    out = dict(legacy)
    out.update(counts)
    out["accuracy_bands"]   = bands
    out["portfolio"]        = portfolio
    out["ledger"]           = ledger
    out["analytics_source"] = "hybrid_supabase_sqlite"
    return out


# ═══════════════════════════════════════════════════════════════════════════
# USAGE  (Phase 1 scaffolding for rate limits — not yet enforced in UI)
# ═══════════════════════════════════════════════════════════════════════════

def log_usage(kind: str, meta: Optional[dict] = None, user_id: Optional[str] = None) -> None:
    """Insert a usage event. No-op in file mode."""
    if not USE_SUPABASE:
        return
    uid = _current_user_id(user_id)
    try:
        _client().table("usage_events").insert({
            "user_id": uid,
            "kind": kind,
            "meta": meta or {},
        }).execute()
    except Exception as exc:
        logger.exception("log_usage failed: %s", exc)


def count_usage_this_month(kind: str, user_id: Optional[str] = None) -> int:
    """Return # of events of `kind` by this user in the current calendar month."""
    if not USE_SUPABASE:
        return 0  # Rate limits not enforced in file mode.
    uid = _current_user_id(user_id)
    try:
        from datetime import datetime, timezone
        month_start = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        res = (
            _client().table("usage_events")
            .select("id", count="exact")
            .eq("user_id", uid)
            .eq("kind", kind)
            .gte("created_at", month_start.isoformat())
            .execute()
        )
        return res.count or 0
    except Exception as exc:
        logger.exception("count_usage_this_month failed: %s", exc)
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# SUBSCRIPTION  (read-only from user side; writes come from Stripe webhook)
# ═══════════════════════════════════════════════════════════════════════════

def get_subscription(user_id: Optional[str] = None) -> dict:
    """Return the user's current subscription row, or a 'free' default."""
    default = {"plan": "free", "status": "active"}
    if not USE_SUPABASE:
        return default
    uid = _current_user_id(user_id)
    try:
        res = (
            _client().table("subscriptions")
            .select("*")
            .eq("user_id", uid)
            .single()
            .execute()
        )
        return res.data or default
    except Exception:
        return default
