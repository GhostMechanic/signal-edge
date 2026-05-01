"""
model_portfolio.py
------------------
Repository interface for the model's paper portfolio. Defines the *shape*
of the read/write operations the prediction-insertion path needs, plus
two implementations:

  • InMemoryPortfolioRepo  — single-process, non-durable. For tests.
  • SupabaseModelPortfolioRepo — talks to the Supabase tables created by
                                  migrations 0002 + 0003. Production path.

Use `get_portfolio_repo()` to pick the right one based on env config (same
pattern as db.py — USE_SUPABASE=true selects the Supabase backend).

Read paths:
  • get_state() — cash, open positions count, open symbols set
                  (everything the TRADE rule needs)

Write paths:
  • open_trade(req) — atomic "decrement cash + insert model_paper_trades
                      row" via the open_model_trade RPC from 0003.

We deliberately do NOT expose `update_cash()` directly — every mutation
goes through a typed operation so the audit trail is complete by
construction.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AbstractSet, Optional, Protocol
import uuid

logger = logging.getLogger(__name__)


# ─── Value types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PortfolioState:
    """Snapshot of the model's portfolio at the moment a TRADE decision is
    being made. Everything the § 2.1 rule needs in one shape."""
    cash:                float
    open_positions_count: int
    open_symbols:        AbstractSet[str]
    starting_capital:    float
    max_open_positions:  int


@dataclass(frozen=True)
class OpenTradeRequest:
    """All the fields needed to write a `model_paper_trades` row. Built by
    the orchestrator from the prediction + trade plan. Frozen because once
    submitted it should not mutate."""
    prediction_id: str
    symbol:        str
    direction:     str             # 'LONG' | 'SHORT'
    entry_price:   float
    target_price:  float
    stop_price:    float
    qty:           float
    notional:      float


@dataclass(frozen=True)
class OpenTradeResult:
    """What the repository returns after a successful open. The trade_id
    is what gets stored (or referenced) on the prediction row, and the
    new_cash is what the dashboard's intra-day live number will reflect."""
    trade_id:  str
    opened_at: datetime
    new_cash:  float


# ─── Repository protocol ──────────────────────────────────────────────────────

class ModelPortfolioRepo(Protocol):
    """The minimum surface the orchestrator needs. The Supabase
    implementation will satisfy this with a service-role client; the
    in-memory implementation below satisfies it for tests."""

    def get_state(self) -> PortfolioState: ...

    def open_trade(self, req: OpenTradeRequest) -> OpenTradeResult:
        """Atomic operation:
          1. Verify cash >= req.notional and no open trade exists for symbol.
          2. Insert a row into model_paper_trades (status='open').
          3. Decrement model_paper_portfolio.cash by req.notional.
        Returns OpenTradeResult on success.

        Implementations MUST raise on any rule violation rather than
        silently no-op'ing — the orchestrator already pre-checks via the
        decision module, but the repo is the last line of defense against
        a race condition between two simultaneous predictions for the
        same symbol."""


# ─── In-memory implementation (testing only) ──────────────────────────────────

@dataclass
class InMemoryPortfolioRepo:
    """Single-process portfolio used by self-tests and any local-dev path
    where Supabase isn't running. Not threadsafe; not durable; loses state
    on process exit. Do not import for production."""

    cash:               float = 10_000.00
    starting_capital:   float = 10_000.00
    max_open_positions: int   = 25
    _trades: dict[str, OpenTradeRequest] = field(default_factory=dict)

    def get_state(self) -> PortfolioState:
        return PortfolioState(
            cash                 = self.cash,
            open_positions_count = len(self._trades),
            open_symbols         = frozenset(t.symbol.upper() for t in self._trades.values()),
            starting_capital     = self.starting_capital,
            max_open_positions   = self.max_open_positions,
        )

    def open_trade(self, req: OpenTradeRequest) -> OpenTradeResult:
        sym = req.symbol.strip().upper()
        if any(t.symbol.strip().upper() == sym for t in self._trades.values()):
            raise ValueError(f"refusing to open: {sym} already has an open trade")
        if self.cash < req.notional:
            raise ValueError(
                f"refusing to open: cash {self.cash:.2f} < notional {req.notional:.2f}"
            )
        if len(self._trades) >= self.max_open_positions:
            raise ValueError(
                f"refusing to open: book full ({len(self._trades)}/{self.max_open_positions})"
            )

        trade_id = str(uuid.uuid4())
        self._trades[trade_id] = req
        self.cash = round(self.cash - req.notional, 2)
        return OpenTradeResult(
            trade_id  = trade_id,
            opened_at = datetime.now(timezone.utc),
            new_cash  = self.cash,
        )


# ─── Supabase implementation (production) ─────────────────────────────────────

class SupabaseModelPortfolioRepo:
    """Production repo backed by the Supabase tables from migrations
    0002 + 0003. Uses the service-role client (bypasses RLS) — the model's
    portfolio is a global system asset, not user-scoped data.

    Reads come from two simple SELECTs against `model_paper_portfolio`
    (singleton id=1) and `model_paper_trades`. Writes go through the
    `open_model_trade` RPC defined in 0003 so the cash-decrement and
    trade-insert happen atomically (SELECT FOR UPDATE serializes
    concurrent opens; partial unique index backstops the symbol guard).
    """

    def __init__(self, client=None):
        # Lazy-resolve the client so importing this module doesn't require
        # supabase-py to be installed when running tests with the in-mem repo.
        self._client = client

    def _service_client(self):
        if self._client is not None:
            return self._client
        # Defer to db.py's _service_client so we follow the project pattern
        # (env-driven URL + service-role key, lru_cached).
        from db import _service_client as _project_service_client
        self._client = _project_service_client()
        return self._client

    def get_state(self) -> PortfolioState:
        c = self._service_client()

        portfolio_res = (
            c.table("model_paper_portfolio")
             .select("cash, starting_capital, max_open_positions")
             .eq("id", 1)
             .single()
             .execute()
        )
        if not portfolio_res.data:
            raise RuntimeError(
                "model_paper_portfolio singleton row missing — apply "
                "migration 0002 before using this repo."
            )

        trades_res = (
            c.table("model_paper_trades")
             .select("symbol")
             .eq("status", "open")
             .execute()
        )
        rows = trades_res.data or []

        return PortfolioState(
            cash                 = float(portfolio_res.data["cash"]),
            open_positions_count = len(rows),
            open_symbols         = frozenset(
                r["symbol"].strip().upper() for r in rows if r.get("symbol")
            ),
            starting_capital     = float(portfolio_res.data["starting_capital"]),
            max_open_positions   = int(portfolio_res.data["max_open_positions"]),
        )

    def open_trade(self, req: OpenTradeRequest) -> OpenTradeResult:
        c = self._service_client()

        # Single RPC = single transaction. Errors raised inside the function
        # come back as exceptions on the Python side with the descriptive
        # messages from 0003 ('insufficient_cash', 'book_full',
        # 'symbol_already_open', etc.) so the prediction logger can surface
        # which guardrail tripped.
        try:
            res = c.rpc(
                "open_model_trade",
                {
                    "p_prediction_id": req.prediction_id,
                    "p_symbol":        req.symbol.strip().upper(),
                    "p_direction":     req.direction,
                    "p_entry_price":   float(req.entry_price),
                    "p_target_price":  float(req.target_price),
                    "p_stop_price":    float(req.stop_price),
                    "p_qty":           float(req.qty),
                    "p_notional":      float(req.notional),
                },
            ).execute()
        except Exception as exc:
            # supabase-py wraps Postgres errors; surface them with the trade
            # context so logs are actionable.
            raise RuntimeError(
                f"open_model_trade RPC failed for {req.symbol} "
                f"(prediction_id={req.prediction_id}): {exc}"
            ) from exc

        # The RPC returns a one-row table — supabase-py exposes it as a list.
        rows = res.data or []
        if not rows:
            raise RuntimeError(
                f"open_model_trade returned no rows for {req.symbol}"
            )
        row = rows[0]
        return OpenTradeResult(
            trade_id  = row["trade_id"],
            opened_at = datetime.fromisoformat(
                str(row["opened_at"]).replace("Z", "+00:00")
            ),
            new_cash  = float(row["new_cash"]),
        )


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_portfolio_repo() -> ModelPortfolioRepo:
    """Pick the right repo based on env. Mirrors the db.py USE_SUPABASE
    pattern. Falls back to in-memory if Supabase is not configured.

    Returning the in-memory repo when Supabase is off is a deliberate dev
    choice: the prediction logger keeps working in legacy mode without
    crashing, but the cash math evaporates between processes. Anything
    that actually depends on the model's portfolio being durable must run
    with USE_SUPABASE=true.
    """
    use_sb = os.getenv("USE_SUPABASE", "").strip().lower() in ("1", "true", "yes", "on")
    if not use_sb:
        logger.warning(
            "model_portfolio: USE_SUPABASE not set; returning in-memory repo "
            "(cash math will not persist across processes)."
        )
        return InMemoryPortfolioRepo()
    return SupabaseModelPortfolioRepo()


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    repo = InMemoryPortfolioRepo()

    # Initial state.
    s = repo.get_state()
    assert s.cash == 10_000.00
    assert s.open_positions_count == 0
    assert s.open_symbols == frozenset()
    print(f"Initial state: cash={s.cash}, open={s.open_positions_count}")

    # Open a trade.
    req = OpenTradeRequest(
        prediction_id="pred-1", symbol="NVDA", direction="LONG",
        entry_price=800.0, target_price=880.0, stop_price=776.0,
        qty=0.5, notional=400.0,
    )
    res = repo.open_trade(req)
    s = repo.get_state()
    assert s.cash == 9_600.00
    assert s.open_positions_count == 1
    assert "NVDA" in s.open_symbols
    print(f"After 1 open: cash={s.cash}, open_symbols={set(s.open_symbols)}")

    # Refuse to double-open same symbol.
    try:
        repo.open_trade(req._replace(prediction_id="pred-2") if False else
                        OpenTradeRequest(
                            prediction_id="pred-2", symbol="NVDA", direction="LONG",
                            entry_price=800, target_price=880, stop_price=776,
                            qty=0.5, notional=400,
                        ))
    except ValueError as e:
        print(f"Double-open guard: {e}")
    else:
        raise AssertionError("expected double-open to fail")

    # Refuse if cash is insufficient.
    repo.cash = 100.0
    try:
        repo.open_trade(OpenTradeRequest(
            prediction_id="pred-3", symbol="MSFT", direction="LONG",
            entry_price=400, target_price=440, stop_price=388,
            qty=1.0, notional=400,
        ))
    except ValueError as e:
        print(f"Insufficient-cash guard: {e}")
    else:
        raise AssertionError("expected insufficient-cash open to fail")

    print("model_portfolio.py self-test OK")
