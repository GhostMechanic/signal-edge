"""
paper_trader.py  –  User Paper-Trading Portfolio
────────────────────────────────────────────────────
A personal paper-trading sandbox where users pick signals
from the watchlist and track their own performance.

The user chooses which signals to act on, selects position
size and horizon, and can close trades early.  This is the
user's playground — fully resettable.

The *system's* track record (every signal, never resettable)
lives on the Track Record page, separate from this.
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

PORTFOLIO_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".predictions", "paper_portfolio.json",
)
STARTING_CAPITAL = 10_000.0
MAX_POSITION_PCT = 0.20     # 20% max single position
MAX_TOTAL_EXPOSURE = 1.00   # 100% total exposure (no leverage)

# Preset position sizes the user can pick from
POSITION_SIZES = {
    "Small ($250)": 250,
    "Medium ($500)": 500,
    "Large ($1,000)": 1000,
    "XL ($2,000)": 2000,
}

HORIZON_OPTIONS = {
    "3 Day": 3,
    "1 Week": 5,
    "1 Month": 21,
    "1 Quarter": 63,
}


def _load_portfolio() -> dict:
    """Load the portfolio from disk."""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return _new_portfolio()


def _new_portfolio() -> dict:
    return {
        "starting_capital": STARTING_CAPITAL,
        "cash": STARTING_CAPITAL,
        "positions": [],        # open positions
        "closed": [],           # closed trades
        "equity_curve": [],     # daily snapshots: {date, equity}
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
    }


def _save_portfolio(pf: dict):
    os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
    pf["last_updated"] = datetime.now().isoformat()
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(pf, f, indent=2, default=str)


def reset_portfolio() -> dict:
    """Reset the portfolio to starting state."""
    pf = _new_portfolio()
    _save_portfolio(pf)
    return pf


def user_open_trade(
    symbol: str,
    direction: str,           # "LONG" or "SHORT"
    entry_price: float,
    position_value: float,    # dollar amount chosen by user
    confidence: float,
    horizon: str,
    horizon_days: int,
) -> tuple:
    """
    Open a paper trade chosen by the user.
    Returns (success: bool, message: str).
    """
    pf = _load_portfolio()

    if position_value < 10:
        return False, "Position too small (min $10)"

    if position_value > pf["cash"]:
        return False, f"Not enough cash (${pf['cash']:,.2f} available)"

    if entry_price <= 0:
        return False, "Invalid entry price"

    shares = position_value / entry_price

    position = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S") + f"_{symbol}_{horizon}",
        "symbol": symbol,
        "direction": direction,
        "entry_price": round(entry_price, 4),
        "shares": round(shares, 6),
        "position_value": round(position_value, 2),
        "confidence": round(confidence, 1),
        "horizon": horizon,
        "horizon_days": horizon_days,
        "entry_date": datetime.now().strftime("%Y-%m-%d"),
        "target_close_date": (
            datetime.now() + timedelta(days=horizon_days)
        ).strftime("%Y-%m-%d"),
    }

    pf["cash"] -= position_value
    pf["positions"].append(position)
    _save_portfolio(pf)
    return True, f"Opened {direction} {symbol} — ${position_value:,.0f} ({horizon})"


def user_close_trade(position_id: str) -> tuple:
    """
    Close a specific position early by its id.
    Returns (success: bool, message: str).
    """
    pf = _load_portfolio()

    target = None
    for i, pos in enumerate(pf["positions"]):
        if pos.get("id") == position_id:
            target = pf["positions"].pop(i)
            break

    if target is None:
        return False, "Position not found"

    # Fetch current price
    sym = target["symbol"]
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="5d")
        cur_price = float(hist["Close"].iloc[-1]) if len(hist) > 0 else target["entry_price"]
    except Exception:
        cur_price = target["entry_price"]

    # Calculate P&L
    if target["direction"] == "LONG":
        pnl = (cur_price - target["entry_price"]) * target["shares"]
    else:
        pnl = (target["entry_price"] - cur_price) * target["shares"]

    closed_trade = {
        **target,
        "exit_price": round(cur_price, 4),
        "exit_date": datetime.now().strftime("%Y-%m-%d"),
        "realised_pnl": round(pnl, 2),
        "realised_pct": round(pnl / target["position_value"] * 100, 2),
        "correct_direction": pnl > 0,
        "closed_early": True,
    }

    pf["closed"].append(closed_trade)
    pf["cash"] += target["position_value"] + pnl
    _save_portfolio(pf)
    return True, f"Closed {sym} — P&L ${pnl:+,.2f} ({closed_trade['realised_pct']:+.1f}%)"


def update_portfolio() -> dict:
    """
    Update all open positions with current prices.
    Close positions that have reached their horizon.
    Snapshot the equity curve.
    """
    pf = _load_portfolio()

    if not pf["positions"]:
        # Still snapshot equity even with no positions
        _snapshot_equity(pf)
        _save_portfolio(pf)
        return pf

    # Fetch current prices for all open symbols
    symbols = list(set(p["symbol"] for p in pf["positions"]))
    current_prices = {}
    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period="5d")
            if len(hist) > 0:
                current_prices[sym] = float(hist["Close"].iloc[-1])
        except Exception:
            pass

    today = datetime.now().strftime("%Y-%m-%d")
    still_open = []
    for pos in pf["positions"]:
        # Backfill id for legacy positions
        if "id" not in pos:
            pos["id"] = pos.get("entry_date", today) + f"_{pos['symbol']}_{pos.get('horizon', 'unknown')}"
        sym = pos["symbol"]
        if sym not in current_prices:
            still_open.append(pos)
            continue

        cur_price = current_prices[sym]
        pos["current_price"] = cur_price

        # Calculate unrealised P&L
        if pos["direction"] == "LONG":
            pnl = (cur_price - pos["entry_price"]) * pos["shares"]
        else:
            pnl = (pos["entry_price"] - cur_price) * pos["shares"]
        pos["unrealised_pnl"] = round(pnl, 2)
        pos["unrealised_pct"] = round(pnl / pos["position_value"] * 100, 2)

        # Check if we should close this position
        if today >= pos.get("target_close_date", "9999-12-31"):
            # Close the position
            closed_trade = {
                **pos,
                "exit_price": cur_price,
                "exit_date": today,
                "realised_pnl": round(pnl, 2),
                "realised_pct": round(pnl / pos["position_value"] * 100, 2),
                "correct_direction": (
                    (pnl > 0 and pos["direction"] == "LONG") or
                    (pnl > 0 and pos["direction"] == "SHORT")
                ),
            }
            pf["closed"].append(closed_trade)
            pf["cash"] += pos["position_value"] + pnl  # return capital + P&L
        else:
            still_open.append(pos)

    pf["positions"] = still_open
    _snapshot_equity(pf)
    _save_portfolio(pf)
    return pf


def _total_equity(pf: dict) -> float:
    """Total portfolio value = cash + open positions at cost."""
    return pf["cash"] + sum(p["position_value"] for p in pf["positions"])


def _total_equity_mark(pf: dict) -> float:
    """Total portfolio value marked to market."""
    equity = pf["cash"]
    for pos in pf["positions"]:
        unrealised = pos.get("unrealised_pnl", 0)
        equity += pos["position_value"] + unrealised
    return equity


def _snapshot_equity(pf: dict):
    """Record today's equity for the equity curve."""
    today = datetime.now().strftime("%Y-%m-%d")
    equity = _total_equity_mark(pf)
    # Don't duplicate same-day entries
    if pf["equity_curve"] and pf["equity_curve"][-1]["date"] == today:
        pf["equity_curve"][-1]["equity"] = round(equity, 2)
    else:
        pf["equity_curve"].append({"date": today, "equity": round(equity, 2)})


def get_portfolio_stats() -> dict:
    """
    Compute portfolio analytics.
    Returns dict with all key metrics.
    """
    pf = _load_portfolio()
    pf = update_portfolio()  # refresh first

    equity = _total_equity_mark(pf)
    total_return_pct = (equity / pf["starting_capital"] - 1) * 100

    # Closed trade stats
    closed = pf.get("closed", [])
    n_trades = len(closed)
    wins = [t for t in closed if t.get("realised_pnl", 0) > 0]
    losses = [t for t in closed if t.get("realised_pnl", 0) <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

    avg_win = np.mean([t["realised_pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t["realised_pnl"]) for t in losses]) if losses else 0
    profit_factor = (sum(t["realised_pnl"] for t in wins) /
                     max(1, sum(abs(t["realised_pnl"]) for t in losses))) if losses else float('inf')

    total_realised = sum(t.get("realised_pnl", 0) for t in closed)
    total_unrealised = sum(p.get("unrealised_pnl", 0) for p in pf["positions"])

    # Equity curve metrics
    curve = pf.get("equity_curve", [])
    max_drawdown = 0
    sharpe = None   # None means "not enough data yet" — UI renders "—"
    if len(curve) >= 2:
        equities = [c["equity"] for c in curve]
        # Max drawdown
        peak = equities[0]
        for e in equities:
            peak = max(peak, e)
            dd = (peak - e) / peak * 100
            max_drawdown = max(max_drawdown, dd)
        # Daily returns for Sharpe — requires enough data AND non-trivial
        # variance. Without the minimum-std guard, `std + 1e-9` in the
        # denominator produces astronomical values (we were seeing -3.8e7).
        daily_rets = [equities[i] / equities[i-1] - 1
                      for i in range(1, len(equities))]
        if len(daily_rets) >= 5:
            mean_ret = float(np.mean(daily_rets))
            std_ret = float(np.std(daily_rets))
            # Require meaningful variance (≥ 0.01% daily stdev) before
            # reporting Sharpe — otherwise the ratio is meaningless.
            if std_ret > 1e-4:
                sharpe = (mean_ret / std_ret) * np.sqrt(252)
                # Clamp to a sane display range — real-world Sharpes are
                # rarely outside [-5, 5]; anything bigger is almost
                # certainly numerical noise.
                sharpe = float(max(-5.0, min(5.0, sharpe)))

    # Direction accuracy (did the predicted direction match?)
    direction_correct = sum(1 for t in closed if t.get("correct_direction", False))
    direction_acc = direction_correct / n_trades * 100 if n_trades > 0 else 0

    return {
        "equity": round(equity, 2),
        "starting_capital": pf["starting_capital"],
        "total_return_pct": round(total_return_pct, 2),
        "total_realised_pnl": round(total_realised, 2),
        "total_unrealised_pnl": round(total_unrealised, 2),
        "cash": round(pf["cash"], 2),
        "n_open_positions": len(pf["positions"]),
        "n_closed_trades": n_trades,
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        # profit_factor is None when there are no losing trades yet (the
        # UI renders "—" so users don't see a meaningless ∞ or 999.99).
        "profit_factor": (round(profit_factor, 2)
                          if losses and profit_factor != float('inf') else None),
        "max_drawdown_pct": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 2) if sharpe is not None else None,
        "direction_accuracy": round(direction_acc, 1),
        "open_positions": pf["positions"],
        "closed_trades": closed[-20:],  # last 20
        "equity_curve": curve,
    }


def get_available_cash() -> float:
    """Return current cash balance."""
    pf = _load_portfolio()
    return round(pf["cash"], 2)
