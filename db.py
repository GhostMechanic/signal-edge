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


def _current_user_id(user_id: Optional[str] = None) -> str:
    """
    Resolve the user_id for the current request.
      1. Caller-supplied user_id wins.
      2. Otherwise fall back to DEV_USER_ID env var (for local dev before auth).
      3. TODO (Phase 2): read from Supabase Auth session via
         st.session_state or a helper that wraps supabase.auth.get_user().
    """
    if user_id:
        return user_id
    if DEV_USER_ID:
        return DEV_USER_ID
    raise RuntimeError(
        "No user_id available. Either pass user_id explicitly, set DEV_USER_ID "
        "in .env for local dev, or wire up Supabase Auth (Phase 2)."
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
# PREDICTIONS LOG  (stubs — migrate prediction_logger_v2 next)
# ═══════════════════════════════════════════════════════════════════════════

def get_full_analytics(public_only: bool = False) -> dict:
    """
    Returns the same shape as prediction_logger_v2.get_full_analytics.
    public_only=True restricts to predictions with is_public_ledger=true
    (this powers the public Track Record view).
    """
    if not USE_SUPABASE:
        from prediction_logger_v2 import get_full_analytics as _file_analytics
        return _file_analytics()
    raise NotImplementedError(
        "Supabase analytics backend not yet implemented. Keep "
        "USE_SUPABASE=false until we migrate prediction_logger_v2."
    )


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
