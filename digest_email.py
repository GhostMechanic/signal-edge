"""
digest_email.py
───────────────
Daily user digest — runs after the EOD scoring pass.

For each authenticated user with at least one event in the last
trading day, builds a small HTML email summarizing:

  • Calls that just settled (HIT / PARTIAL / MISSED with verdict
    counts and per-call symbol + horizon).
  • Their open commitments closing soonest (next 7 days).
  • Their personal portfolio delta if they paper-trade
    (cash + open MTM vs starting capital).

Sends via Resend (https://resend.com — clean API, $0 free tier covers
~3k emails/month). Same script could swap to Postmark / SendGrid /
SES with a 4-line API call change.

Methodology-honest framing: nothing predictive in the email. Just a
digest of events that already happened. Re-engagement at zero
methodology cost.

Deployment:
  1. Sign up at resend.com, verify a sending domain (or use
     onboarding@resend.dev for testing — has rate limits).
  2. Set GitHub repo secrets:
       • RESEND_API_KEY      — re_*** from Resend dashboard
       • DIGEST_FROM_ADDRESS — e.g. "Prediqt <digest@yourdomain.com>"
  3. Run via the existing score.yml workflow:
       python -m digest_email
     OR add a new GitHub Actions cron job (recommended:
     22:00 UTC weekdays so the EOD scoring at 20:15 has finished
     and prices are settled).

No-op safe: if RESEND_API_KEY isn't set, prints what it WOULD send
and exits 0. Lets the script live in the repo and on Render without
breaking anything when no email key is configured.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ── Config (env-driven) ──────────────────────────────────────────────────────
RESEND_API_KEY        = os.getenv("RESEND_API_KEY", "")
DIGEST_FROM_ADDRESS   = os.getenv(
    "DIGEST_FROM_ADDRESS",
    "Prediqt <onboarding@resend.dev>",
)
DIGEST_REPLY_TO       = os.getenv("DIGEST_REPLY_TO", "")
RESEND_API_URL        = "https://api.resend.com/emails"

# Look-back window — events between (now - DIGEST_WINDOW_HOURS) and now
# are considered "today's" events for digest purposes. 26h covers the
# Mon-after-weekend gap when Friday closes were 72h ago.
DIGEST_WINDOW_HOURS   = int(os.getenv("DIGEST_WINDOW_HOURS", "26"))


@dataclass
class UserDigestData:
    user_id:        str
    email:          str
    settled_today:  List[Dict[str, Any]]    # predictions that scored in window
    open_soon:      List[Dict[str, Any]]    # open predictions expiring in 7d
    portfolio:      Optional[Dict[str, Any]]  # paper portfolio summary, if any


# ── Data layer ──────────────────────────────────────────────────────────────

def collect_user_data(user_id: str, email: str, now_utc: datetime) -> Optional[UserDigestData]:
    """Pull this user's settled-today, open-soon, and portfolio from
    Supabase. Returns None when there's nothing to report (so the
    digest skips them entirely instead of sending a no-news email)."""
    from db import _service_client

    client = _service_client()
    cutoff = now_utc - timedelta(hours=DIGEST_WINDOW_HOURS)
    seven_days_out = now_utc + timedelta(days=7)

    # Settled today — predictions whose scored_at landed in window.
    settled = (
        client.table("predictions")
              .select("id, symbol, horizon, direction, verdict, "
                      "actual_return, predicted_return, scored_at, traded")
              .eq("user_id", user_id)
              .in_("verdict", ["HIT", "PARTIAL", "MISSED"])
              .gte("scored_at", cutoff.isoformat())
              .order("scored_at", desc=True)
              .execute()
    ).data or []

    # Open + judgment soon — predictions still OPEN whose
    # horizon_ends_at is within the next 7 days. Limit to 5 — the
    # digest is a digest, not a wall.
    open_soon = (
        client.table("predictions")
              .select("id, symbol, horizon, direction, confidence, "
                      "entry_price, target_price, horizon_ends_at")
              .eq("user_id", user_id)
              .eq("verdict", "OPEN")
              .lte("horizon_ends_at", seven_days_out.isoformat())
              .order("horizon_ends_at", desc=False)
              .limit(5)
              .execute()
    ).data or []

    # User's paper portfolio — optional. None of these endpoints fail
    # the digest if absent (user might just not paper-trade).
    portfolio: Optional[Dict[str, Any]] = None
    try:
        pp = (
            client.table("paper_portfolios")
                  .select("cash, starting_capital, equity_curve")
                  .eq("user_id", user_id)
                  .single()
                  .execute()
        ).data
        if pp:
            portfolio = {
                "cash":             float(pp.get("cash") or 0),
                "starting_capital": float(pp.get("starting_capital") or 10000),
                "equity_curve":     pp.get("equity_curve") or [],
            }
    except Exception:
        portfolio = None

    if not settled and not open_soon:
        return None

    return UserDigestData(
        user_id        = user_id,
        email          = email,
        settled_today  = settled,
        open_soon      = open_soon,
        portfolio      = portfolio,
    )


def all_users_with_email() -> List[Dict[str, str]]:
    """Pull (user_id, email) for every signed-up user. Reads the
    `profiles` table (created by Supabase auth's handle_new_user
    trigger). Falls back to an empty list when profiles isn't
    accessible — the digest then no-ops cleanly instead of crashing
    the workflow."""
    from db import _service_client

    client = _service_client()
    try:
        rows = (
            client.table("profiles")
                  .select("id, email")
                  .execute()
        ).data or []
    except Exception as exc:
        logger.warning("profiles lookup failed: %s", exc)
        return []

    out: List[Dict[str, str]] = []
    for r in rows:
        uid   = r.get("id")
        email = (r.get("email") or "").strip().lower()
        if uid and email and "@" in email:
            out.append({"id": uid, "email": email})
    return out


# ── HTML rendering ──────────────────────────────────────────────────────────

def render_email_html(d: UserDigestData) -> str:
    """Build the email body. Brand-cyan / surface-1 colors lifted from
    the web app's design tokens so the digest reads as the same
    product surface."""
    settled_count = len(d.settled_today)
    hits     = sum(1 for r in d.settled_today if r.get("verdict") == "HIT")
    partials = sum(1 for r in d.settled_today if r.get("verdict") == "PARTIAL")
    misses   = sum(1 for r in d.settled_today if r.get("verdict") == "MISSED")

    settled_rows_html = "".join(
        _render_settled_row(r) for r in d.settled_today
    )
    open_rows_html = "".join(
        _render_open_row(r) for r in d.open_soon
    )

    # Portfolio line — only when paper portfolio exists AND has at
    # least one equity curve point (so we have something honest to
    # report).
    portfolio_html = ""
    if d.portfolio and d.portfolio.get("equity_curve"):
        ec = d.portfolio["equity_curve"]
        latest = ec[-1] if ec else None
        if latest and isinstance(latest, dict) and "equity" in latest:
            equity = float(latest["equity"])
            start = d.portfolio["starting_capital"]
            ret_pct = ((equity - start) / start * 100) if start > 0 else 0
            tone = "#06d6a0" if ret_pct >= 0 else "#e04a4a"
            sign = "+" if ret_pct >= 0 else ""
            portfolio_html = f"""
              <p style="margin:24px 0 0;color:#8898aa;font-size:13px;line-height:1.6">
                <span style="color:#f0f4fa;font-weight:700">
                  Your portfolio: ${equity:,.2f}
                </span>
                <span style="color:{tone};font-weight:700;margin-left:6px">
                  {sign}{ret_pct:.2f}%
                </span>
                <span style="color:#4a5a6e;margin-left:6px">
                  from ${start:,.0f} start
                </span>
              </p>
            """

    settled_summary = (
        f"<strong style='color:#f0f4fa'>{settled_count} call{'' if settled_count == 1 else 's'}</strong> "
        f"settled — "
        f"<span style='color:#06d6a0'>{hits} HIT</span>, "
        f"<span style='color:#f59e0b'>{partials} PARTIAL</span>, "
        f"<span style='color:#e04a4a'>{misses} MISSED</span>"
    ) if settled_count > 0 else (
        "<span style='color:#8898aa'>Nothing settled today.</span>"
    )

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prediqt — your daily digest</title>
</head>
<body style="margin:0;padding:0;background:#030508;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
  <div style="max-width:560px;margin:0 auto;padding:32px 24px;color:#f0f4fa">
    <!-- Header -->
    <p style="margin:0;color:#06d6a0;font-size:11px;font-weight:800;letter-spacing:0.18em;text-transform:uppercase">
      Prediqt · daily digest
    </p>
    <h1 style="margin:8px 0 4px;font-size:24px;font-weight:900;line-height:1.2;color:#f0f4fa">
      Today's record
    </h1>
    <p style="margin:0;color:#8898aa;font-size:13px">
      {datetime.now(timezone.utc).strftime("%A, %B %-d")}
    </p>

    <!-- Settled summary -->
    <p style="margin:24px 0 12px;color:#f0f4fa;font-size:14px;line-height:1.6">
      {settled_summary}
    </p>

    {("<table cellpadding='0' cellspacing='0' style='width:100%;border-collapse:collapse;margin-top:8px;border-radius:8px;overflow:hidden;background:#0a0e14'>" + settled_rows_html + "</table>") if settled_rows_html else ""}

    {portfolio_html}

    <!-- Open soon -->
    {("<h2 style='margin:32px 0 12px;font-size:16px;font-weight:800;color:#f0f4fa'>Closing soon</h2><p style='margin:0 0 12px;color:#8898aa;font-size:12px'>Open calls expiring in the next 7 days. Watch your inbox.</p><table cellpadding='0' cellspacing='0' style='width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;background:#0a0e14'>" + open_rows_html + "</table>") if open_rows_html else ""}

    <!-- Footer -->
    <p style="margin:32px 0 0;padding-top:24px;border-top:1px solid #14202e;color:#4a5a6e;font-size:11px;line-height:1.6">
      You're receiving this because you've made predictions on Prediqt.
      <br>
      <a href="https://prediqt-web.vercel.app/me" style="color:#06d6a0;text-decoration:none">View on the web</a>
      &nbsp;·&nbsp;
      <a href="https://prediqt-web.vercel.app" style="color:#06d6a0;text-decoration:none">prediqt</a>
    </p>
  </div>
</body>
</html>"""


def _render_settled_row(r: Dict[str, Any]) -> str:
    sym       = r.get("symbol") or "—"
    horizon   = r.get("horizon") or "—"
    verdict   = r.get("verdict") or "—"
    direction = r.get("direction") or "Neutral"
    actual    = r.get("actual_return")
    actual_str = f"{actual:+.2f}%" if actual is not None else "—"
    tone = (
        "#06d6a0" if verdict == "HIT"
        else "#f59e0b" if verdict == "PARTIAL"
        else "#e04a4a"
    )
    arrow = "↑" if direction == "Bullish" else "↓" if direction == "Bearish" else "→"
    return f"""
      <tr style="border-bottom:1px solid #14202e">
        <td style="padding:10px 14px;color:#f0f4fa;font-weight:800;font-size:13px;font-variant-numeric:tabular-nums">
          {sym} <span style="color:#8898aa;font-weight:600;margin-left:4px">{arrow} {horizon}</span>
        </td>
        <td style="padding:10px 14px;text-align:right;font-size:11px;font-weight:800;letter-spacing:0.1em;color:{tone};text-transform:uppercase">
          {verdict}
        </td>
        <td style="padding:10px 14px;text-align:right;color:#8898aa;font-size:12px;font-variant-numeric:tabular-nums">
          {actual_str}
        </td>
      </tr>
    """


def _render_open_row(r: Dict[str, Any]) -> str:
    sym       = r.get("symbol") or "—"
    horizon   = r.get("horizon") or "—"
    direction = r.get("direction") or "Neutral"
    conf      = r.get("confidence") or 0
    ends_at   = (r.get("horizon_ends_at") or "")[:10]
    arrow = "↑" if direction == "Bullish" else "↓" if direction == "Bearish" else "→"
    return f"""
      <tr style="border-bottom:1px solid #14202e">
        <td style="padding:10px 14px;color:#f0f4fa;font-weight:800;font-size:13px;font-variant-numeric:tabular-nums">
          {sym} <span style="color:#8898aa;font-weight:600;margin-left:4px">{arrow} {horizon}</span>
        </td>
        <td style="padding:10px 14px;text-align:right;color:#8898aa;font-size:12px">
          ends {ends_at}
        </td>
        <td style="padding:10px 14px;text-align:right;color:#06d6a0;font-size:12px;font-weight:700;font-variant-numeric:tabular-nums">
          {conf:.0f}%
        </td>
      </tr>
    """


# ── Send ────────────────────────────────────────────────────────────────────

def send_email(to_addr: str, subject: str, html: str) -> bool:
    """Send via Resend. Returns True on success.

    No-op safe: when RESEND_API_KEY is empty, prints the would-send
    payload and returns True. Lets us test in dry-run without a key
    AND lets the cron run without breaking when no key is configured.
    """
    if not RESEND_API_KEY:
        print(f"[digest dry-run] would email {to_addr}: {subject!r}")
        return True
    payload: Dict[str, Any] = {
        "from":    DIGEST_FROM_ADDRESS,
        "to":      [to_addr],
        "subject": subject,
        "html":    html,
    }
    if DIGEST_REPLY_TO:
        payload["reply_to"] = DIGEST_REPLY_TO
    try:
        res = requests.post(
            RESEND_API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type":  "application/json",
            },
            timeout=10,
        )
        if res.status_code >= 400:
            logger.warning("Resend %d: %s", res.status_code, res.text[:300])
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        logger.exception("send_email exception: %s", exc)
        return False


# ── Orchestrator ────────────────────────────────────────────────────────────

def run_digest() -> Dict[str, Any]:
    """Walk every signed-up user; for each, collect data + send if
    there's something to report. Returns a per-run summary."""
    now_utc = datetime.now(timezone.utc)
    users   = all_users_with_email()
    summary = {
        "scanned":   len(users),
        "skipped":   0,
        "sent":      0,
        "failed":    0,
        "dry_run":   not RESEND_API_KEY,
    }
    for u in users:
        try:
            data = collect_user_data(u["id"], u["email"], now_utc)
            if data is None:
                summary["skipped"] += 1
                continue
            html    = render_email_html(data)
            subject = _build_subject(data)
            ok      = send_email(u["email"], subject, html)
            if ok:
                summary["sent"] += 1
            else:
                summary["failed"] += 1
        except Exception as exc:
            logger.exception("digest user %s failed: %s", u["id"], exc)
            summary["failed"] += 1
    return summary


def _build_subject(d: UserDigestData) -> str:
    n = len(d.settled_today)
    if n == 0:
        return "Prediqt — your open calls update"
    hits = sum(1 for r in d.settled_today if r.get("verdict") == "HIT")
    if n == 1:
        return "Prediqt — 1 call settled"
    return f"Prediqt — {n} calls settled ({hits} HIT)"


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cli() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    result = run_digest()
    print(f"digest: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
