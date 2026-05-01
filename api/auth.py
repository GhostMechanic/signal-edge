"""
api/auth.py
-----------
FastAPI auth integration for Supabase Auth-issued JWTs.

The frontend signs the user in via @supabase/ssr (browser + server
cookies), then forwards the access token to this backend on every
authenticated request as `Authorization: Bearer <access_token>`. We
verify the signature, extract the user's id (the `sub` claim), and
return a CurrentUser the endpoint can use.

Two signing algorithms in play:

  ES256 — modern Supabase projects (post-2024) sign tokens with an
          ECC P-256 key. The public key lives at
          ${SUPABASE_URL}/auth/v1/.well-known/jwks.json. We verify
          using PyJWKClient which fetches and caches the JWKS, then
          uses the `kid` in the JWT header to pick the right key.

  HS256 — legacy projects (or tokens issued before a project's recent
          ES256 rotation, still valid until they expire). Verified
          with a shared secret in SUPABASE_JWT_SECRET.

We pick the algorithm by reading the unverified header's `alg` claim
first, then verify with the matching path. Tokens whose `alg` is
neither are refused.

Dev fallback. When `ALLOW_DEV_USER_FALLBACK=1` AND `DEV_USER_ID` is
set, requests without an Authorization header resolve to that dev
user. Production deploys must not set ALLOW_DEV_USER_FALLBACK.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import jwt  # PyJWT 2.4+ (PyJWKClient lives here)
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Load .env before any helper reads env vars. This file may import
# before db.py (where load_dotenv lives historically), so we call it
# here too — load_dotenv is idempotent, calling twice does nothing.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

logger = logging.getLogger(__name__)


# IMPORTANT: never cache these at module-load time. Helpers below read
# os.getenv() each call so a .env edit + uvicorn --reload picks up
# correctly without a fresh process. Both vars also intentionally
# allowed to be None at import — the helpers raise informative errors
# on first use, not on import.
SUPABASE_JWT_AUDIENCE = "authenticated"


def _allow_dev_fallback() -> bool:
    val = os.getenv("ALLOW_DEV_USER_FALLBACK", "").strip().lower()
    if val not in ("1", "true", "yes", "on"):
        return False
    return bool(os.getenv("DEV_USER_ID"))


@dataclass
class CurrentUser:
    """The minimum we need to know about the requester to do work on
    their behalf. Populated either from a verified JWT or from the
    dev fallback."""
    id: str
    email: Optional[str] = None
    email_verified: bool = False
    via_dev_fallback: bool = False


def _email_verified_from_payload(payload: dict) -> bool:
    """Read email-verified status from a Supabase JWT.

    Supabase populates this at `user_metadata.email_verified` (the
    flag in raw_user_meta_data on the auth.users row). Older or
    custom setups may put it at the top level. We check both, then
    fall back to True when neither is set — by the time a valid
    signed JWT reaches us, the user has already passed Supabase's
    sign-in gate, which enforces email confirmation when the
    "Confirm Email" setting is on. Defaulting to True here means an
    unfilled metadata field doesn't accidentally lock everyone out.

    Set to False explicitly only if user_metadata.email_verified is
    present and false.
    """
    user_meta = payload.get("user_metadata") or {}
    if "email_verified" in user_meta:
        return bool(user_meta["email_verified"])
    if "email_verified" in payload:
        return bool(payload["email_verified"])
    return True


_bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _jwks_client_for(jwks_uri: str) -> "jwt.PyJWKClient":
    """One PyJWKClient per JWKS URI; PyJWKClient itself caches keys
    internally (default 5 min TTL) and refetches when a `kid` isn't
    found, so we get key-rotation handling for free."""
    return jwt.PyJWKClient(jwks_uri)


def _jwks_client() -> "jwt.PyJWKClient":
    """Resolve the JWKS client for the current SUPABASE_URL. Reads
    env at call time so a fresh uvicorn pick up of .env works without
    code changes."""
    supabase_url = os.getenv("SUPABASE_URL")
    if not supabase_url:
        raise RuntimeError(
            "SUPABASE_URL not set — cannot fetch JWKS to verify ES256 tokens."
        )
    return _jwks_client_for(
        f"{supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
    )


def _verify_supabase_jwt(token: str) -> dict:
    """Validate signature + claims against whichever algorithm the
    token's header declares. Raises HTTPException(401) on any
    failure."""
    # Read the unverified header so we know which path to take. This
    # is safe because we don't trust the body until we've verified
    # below — `alg` selection just routes us to the right key source.
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        logger.info("jwt header parse failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="malformed auth token",
        )
    alg = (header.get("alg") or "").upper()

    try:
        if alg == "ES256":
            signing_key = _jwks_client().get_signing_key_from_jwt(token).key
            payload = jwt.decode(
                token,
                signing_key,
                algorithms=["ES256"],
                audience=SUPABASE_JWT_AUDIENCE,
            )
        elif alg == "HS256":
            secret = os.getenv("SUPABASE_JWT_SECRET")
            if not secret:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        "received an HS256 token but SUPABASE_JWT_SECRET "
                        "is not configured. Either set it in env (legacy "
                        "projects only) or your Supabase project should "
                        "be using ES256 — check Project Settings → API → "
                        "JWT Keys."
                    ),
                )
            payload = jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                audience=SUPABASE_JWT_AUDIENCE,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"unsupported jwt algorithm: {alg!r}",
            )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="auth token expired",
        )
    except jwt.InvalidAudienceError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="auth token audience mismatch",
        )
    except jwt.PyJWTError as exc:
        logger.info("jwt validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid auth token",
        )
    return payload


def get_current_user(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> CurrentUser:
    """
    Required-auth dependency. Use on every endpoint that mutates
    user state (predict, paper-trade open/close, etc.).

    Order of resolution:
      1. Authorization: Bearer <jwt> → verify, return CurrentUser.
      2. ALLOW_DEV_USER_FALLBACK + DEV_USER_ID → return that user.
      3. Otherwise → 401.
    """
    from db import set_request_user

    if creds and creds.credentials:
        payload = _verify_supabase_jwt(creds.credentials)
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="auth token missing user id",
            )
        # email_verified flag lives at user_metadata.email_verified in
        # Supabase JWTs (see _email_verified_from_payload). The
        # require_verified_email dependency reads cu.email_verified to
        # gate the predict path; everything else is fine with either.
        cu = CurrentUser(
            id=str(sub),
            email=payload.get("email"),
            email_verified=_email_verified_from_payload(payload),
        )
        request.state.current_user = cu
        set_request_user(cu.id)
        return cu

    if _allow_dev_fallback():
        dev_id = os.getenv("DEV_USER_ID") or ""
        cu = CurrentUser(
            id=dev_id, email=None, email_verified=True, via_dev_fallback=True,
        )
        request.state.current_user = cu
        set_request_user(cu.id)
        return cu

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="authentication required",
    )


def get_optional_user(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[CurrentUser]:
    """
    Optional-auth dependency. Use on read endpoints that branch
    response shape by tier (anonymous → public, member → +member fields,
    paid → +paid fields per data-model § 6).

    Returns None when there's no valid session — endpoint then renders
    the public-tier view.
    """
    from db import set_request_user

    if creds and creds.credentials:
        try:
            payload = _verify_supabase_jwt(creds.credentials)
        except HTTPException:
            # Bad token on a public endpoint — treat as anonymous
            # rather than 401. The point of the endpoint is to be
            # readable without auth.
            return None
        sub = payload.get("sub")
        if not sub:
            return None
        cu = CurrentUser(
            id=str(sub),
            email=payload.get("email"),
            email_verified=_email_verified_from_payload(payload),
        )
        request.state.current_user = cu
        set_request_user(cu.id)
        return cu

    if _allow_dev_fallback():
        dev_id = os.getenv("DEV_USER_ID") or ""
        cu = CurrentUser(
            id=dev_id, email=None, email_verified=True, via_dev_fallback=True,
        )
        request.state.current_user = cu
        set_request_user(cu.id)
        return cu

    return None


def require_verified_email(cu: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """
    Stricter wrapper for endpoints that produce public-ledger artifacts
    (predict). The public ledger is forever; we don't accept entries
    from accounts that haven't verified they own their email.

    Dev fallback bypasses this — DEV_USER_ID is treated as verified by
    convention so local dev still works.
    """
    if cu.via_dev_fallback:
        return cu
    if not cu.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "email-verification required. check your inbox for the "
                "confirmation link before making predictions."
            ),
        )
    return cu
