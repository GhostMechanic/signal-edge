"""
portfolio_page.py — User Paper-Trading Desk
Your personal sandbox: pick signals, size positions, track your P&L.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from paper_trader import (
    get_portfolio_stats, reset_portfolio, update_portfolio,
    user_open_trade, user_close_trade, get_available_cash,
    POSITION_SIZES, HORIZON_OPTIONS,
)

# ── Design Tokens (mirror app.py's brand palette — keep in sync) ─────────────
BG_PRIMARY    = "#030508"
BG_SURFACE    = "#0a0e14"
BG_CARD       = "#0f1319"
BG_ELEVATED   = "#141a22"
BORDER        = "#1a2233"
BORDER_ACCENT = "#1e3050"
TEXT_PRIMARY   = "#f0f4fa"
TEXT_SECONDARY = "#8898aa"
TEXT_MUTED     = "#4a5a6e"
GREEN          = "#10b981"
GREEN_DIM      = "#064e3b"
RED            = "#ef4444"
RED_DIM        = "#450a0a"
AMBER          = "#f59e0b"
AMBER_DIM      = "#451a03"
BLUE           = "#2080e5"
CYAN           = "#06d6a0"
BRAND_GRAD     = "linear-gradient(135deg, #2080e5 0%, #06d6a0 100%)"
# Chart palette — brand-first, convention-safe
CHART_UP        = CYAN
CHART_UP_FILL   = "rgba(6,214,160,0.22)"
CHART_DOWN      = "#e04a4a"
CHART_DOWN_FILL = "rgba(224,74,74,0.22)"
CHART_NEUTRAL   = BLUE
CHART_GRID      = "rgba(255,255,255,0.04)"

STARTING_CAPITAL = 10_000.0


def render_portfolio_page():
    st.markdown(
        "<div class='page-header'>"
        "<div class='ph-eyebrow'>Paper Trading</div>"
        "<h1>Your <span class='grad'>portfolio</span>.</h1>"
        "<p class='ph-sub'>Pick signals, size your bets, and see exactly "
        "how you'd perform — without risking real money.</p>"
        "</div>", unsafe_allow_html=True)

    try:
        stats = get_portfolio_stats()
    except Exception as e:
        st.error(f"Portfolio error: {e}")
        return

    # ── Hero value card ──────────────────────────────────────────────────
    equity = stats["equity"]
    total_ret = stats["total_return_pct"]
    ret_color = GREEN if total_ret >= 0 else RED
    ret_arrow = "▲" if total_ret >= 0 else "▼"

    st.markdown(
        f"<div style='background:linear-gradient(145deg, {BG_CARD}, {BG_SURFACE});border:1px solid {BORDER};"
        f"border-radius:14px;padding:24px 28px;margin-bottom:20px;position:relative;overflow:hidden'>"
        f"<div style='position:absolute;top:0;left:0;right:0;height:3px;background:{BRAND_GRAD}'></div>"
        f"<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px'>"
        f"<div>"
        f"<div style='color:{TEXT_MUTED};font-size:0.6rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:1.5px;margin-bottom:6px'>Portfolio Value</div>"
        f"<div style='color:{TEXT_PRIMARY};font-size:2.4rem;font-weight:900;letter-spacing:-1px'>${equity:,.2f}</div>"
        f"<div style='color:{ret_color};font-size:0.85rem;font-weight:700;margin-top:4px'>"
        f"{ret_arrow} {total_ret:+.2f}% all-time</div>"
        f"</div>"
        f"<div style='display:flex;gap:28px'>"
        + "".join(
            f"<div style='text-align:center'>"
            f"<div style='color:{TEXT_MUTED};font-size:0.55rem;font-weight:600;text-transform:uppercase;letter-spacing:1px'>{lbl}</div>"
            f"<div style='color:{clr};font-size:1.15rem;font-weight:800;margin-top:4px'>{val}</div>"
            f"</div>"
            for lbl, val, clr in [
                ("Cash", f"${stats['cash']:,.0f}", GREEN if stats['cash'] > 0 else RED),
                ("Open", str(stats["n_open_positions"]), TEXT_PRIMARY),
                ("Closed", str(stats["n_closed_trades"]), TEXT_PRIMARY),
                ("Win Rate",
                 f"{stats['win_rate']:.0f}%" if stats["n_closed_trades"] > 0 else "—",
                 GREEN if stats["win_rate"] >= 50 else (RED if stats["n_closed_trades"] > 0 else TEXT_MUTED)),
            ]
        )
        + f"</div></div></div>", unsafe_allow_html=True)

    # ── Stats row ────────────────────────────────────────────────────────
    # Render Sharpe only when we have enough data. When paper-trading just
    # started (< ~5 daily snapshots) or when all equity points are identical,
    # Sharpe is meaningless and the backend returns None → show "—".
    _sharpe_raw = stats.get('sharpe_ratio')
    _sharpe_str = f"{_sharpe_raw:+.2f}" if _sharpe_raw is not None else "—"
    _sharpe_sub = ("not enough data yet" if _sharpe_raw is None
                   else "annualized, daily returns")

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, sub in [
        (m1, "Realised P&L",   f"${stats['total_realised_pnl']:+,.2f}", "closed trades"),
        (m2, "Unrealised P&L", f"${stats['total_unrealised_pnl']:+,.2f}", "open positions"),
        (m3, "Max Drawdown",   f"{stats['max_drawdown_pct']:.1f}%",       "worst peak-to-trough"),
        (m4, "Sharpe Ratio",   _sharpe_str,                                _sharpe_sub),
    ]:
        # Color by sign only when the value has a real numeric sign.
        if "+" in val:
            _v_color = CHART_UP
        elif "-" in val and val != "—":
            _v_color = CHART_DOWN
        else:
            _v_color = TEXT_PRIMARY
        col.markdown(
            f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
            f"border:1px solid {BORDER};border-radius:12px;padding:16px;text-align:center;"
            f"position:relative;overflow:hidden'>"
            # Brand hairline
            f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
            f"background:{BRAND_GRAD};opacity:0.55'></div>"
            f"<div style='color:{CYAN};font-size:0.56rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.14em'>{label}</div>"
            f"<div style='font-family:\"Inter\",sans-serif;color:{_v_color};"
            f"font-size:1.2rem;font-weight:900;letter-spacing:-0.02em;"
            f"font-variant-numeric:tabular-nums;margin-top:8px'>{val}</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.6rem;margin-top:4px'>{sub}</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='height:20px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # BUYING POWER BAR
    # ══════════════════════════════════════════════════════════════════════
    cash = stats["cash"]
    invested = equity - cash
    cash_pct = (cash / equity * 100) if equity > 0 else 0
    _bp_color = GREEN if cash_pct > 30 else (AMBER if cash_pct > 10 else RED)
    _bp_warn = ""
    if cash_pct <= 10 and equity > 0:
        _bp_warn = (
            f"<div style='color:{RED};font-size:0.68rem;font-weight:600;margin-top:6px'>"
            f"Low buying power — close positions or wait for trades to expire to free up cash.</div>"
        )

    st.markdown(
        f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;"
        f"padding:14px 18px;margin-bottom:20px'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>"
        f"<div style='color:{TEXT_MUTED};font-size:0.55rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:1px'>Buying Power</div>"
        f"<div style='display:flex;gap:16px;align-items:center'>"
        f"<span style='color:{TEXT_MUTED};font-size:0.65rem'>Invested: "
        f"<span style='color:{TEXT_SECONDARY};font-weight:700'>${invested:,.2f}</span></span>"
        f"<span style='color:{TEXT_MUTED};font-size:0.65rem'>Available: "
        f"<span style='color:{_bp_color};font-weight:700'>${cash:,.2f}</span></span>"
        f"</div></div>"
        f"<div style='width:100%;height:6px;background:{BG_SURFACE};border-radius:3px;overflow:hidden'>"
        f"<div style='height:100%;width:{cash_pct:.1f}%;background:{_bp_color};border-radius:3px;"
        f"transition:width 0.3s'></div></div>"
        f"<div style='display:flex;justify-content:space-between;margin-top:4px'>"
        f"<span style='color:{TEXT_MUTED};font-size:0.55rem'>{cash_pct:.0f}% available</span>"
        f"<span style='color:{TEXT_MUTED};font-size:0.55rem'>of ${equity:,.2f} total</span>"
        f"</div>"
        f"{_bp_warn}"
        f"</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # NEW TRADE FORM
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        f"<div style='color:{TEXT_MUTED};font-size:0.6rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:2px;margin-bottom:4px'>New Trade</div>",
        unsafe_allow_html=True)

    # How it works explanation
    st.markdown(
        f"<div style='background:{BG_CARD};border:1px solid {BORDER_ACCENT};border-radius:8px;"
        f"padding:12px 16px;margin-bottom:14px'>"
        f"<span style='color:{CYAN};font-weight:700;font-size:0.72rem'>How it works:</span>"
        f"<span style='color:{TEXT_SECONDARY};font-size:0.72rem;line-height:1.6'>"
        f" Pick a stock, choose your direction (long = betting it goes up, short = betting it goes down), "
        f"set your position size, and select a horizon. "
        f"The <b style='color:{TEXT_PRIMARY}'>horizon</b> is how long you're holding the trade — "
        f"when the horizon expires, the position automatically closes at the current market price and "
        f"locks in your P&L. You can also close any trade early if you want to take profits or cut losses.</span>"
        f"</div>", unsafe_allow_html=True)

    # Check if watchlist pre-filled a trade
    prefill_sym = st.session_state.get("paper_trade_prefill", {}).get("symbol", "")
    prefill_dir = st.session_state.get("paper_trade_prefill", {}).get("direction", "LONG")
    prefill_conf = st.session_state.get("paper_trade_prefill", {}).get("confidence", 0)

    with st.container():
        tc1, tc2, tc3, tc4 = st.columns([2, 1.2, 1.5, 1.5])
        with tc1:
            trade_sym = st.text_input(
                "Ticker", value=prefill_sym,
                placeholder="AAPL",
                key="trade_ticker_input",
            ).upper().strip()
        with tc2:
            dir_options = ["LONG", "SHORT"]
            dir_idx = 0 if prefill_dir == "LONG" else 1
            trade_dir = st.selectbox("Direction", dir_options, index=dir_idx, key="trade_dir_sel")
        with tc3:
            size_label = st.selectbox("Position Size", list(POSITION_SIZES.keys()), index=1, key="trade_size_sel")
            trade_value = POSITION_SIZES[size_label]
        with tc4:
            horizon_label = st.selectbox("Horizon", list(HORIZON_OPTIONS.keys()), index=1, key="trade_horizon_sel")
            trade_days = HORIZON_OPTIONS[horizon_label]

        # Confidence display if pre-filled
        _conf_note = ""
        if prefill_conf > 0:
            _conf_note = (
                f"<span style='color:{CYAN};font-size:0.7rem;margin-left:8px'>"
                f"Signal confidence: {prefill_conf:.0f}%</span>"
            )

        # ── Available cash + warnings ───────────────────────────────────
        _size_warn = ""
        if trade_value > cash:
            _size_warn = (
                f"<span style='color:{RED};font-size:0.68rem;font-weight:600;margin-left:8px'>"
                f"Exceeds available cash</span>"
            )
        st.markdown(
            f"<div style='color:{TEXT_MUTED};font-size:0.72rem;margin-bottom:8px'>"
            f"Available: <span style='color:{_bp_color};font-weight:700'>${cash:,.2f}</span>"
            f"{_conf_note}{_size_warn}</div>",
            unsafe_allow_html=True)

        # ── Preview / Confirm flow ───────────────────────────────────────
        from datetime import datetime as _dt, timedelta as _td

        _preview = st.session_state.get("_trade_preview", None)
        _preview_stale = False
        if _preview:
            # Check if form inputs changed since preview was generated
            _preview_stale = (
                _preview.get("symbol") != trade_sym
                or _preview.get("direction") != trade_dir
                or _preview.get("position_value") != trade_value
                or _preview.get("horizon") != horizon_label
            )
            if _preview_stale:
                st.session_state.pop("_trade_preview", None)
                _preview = None

        if not _preview:
            # Step 1: Preview button
            if st.button("Preview Trade", type="primary", use_container_width=False, key="preview_trade_btn"):
                if not trade_sym:
                    st.warning("Enter a ticker symbol")
                else:
                    try:
                        tk = yf.Ticker(trade_sym)
                        hist = tk.history(period="5d")
                        if len(hist) == 0:
                            st.error(f"Could not fetch price for {trade_sym}")
                        else:
                            cur_price = float(hist["Close"].iloc[-1])
                            _shares = trade_value / cur_price
                            _close_date = (_dt.now() + _td(days=trade_days)).strftime("%Y-%m-%d")
                            st.session_state["_trade_preview"] = {
                                "symbol": trade_sym,
                                "direction": trade_dir,
                                "entry_price": cur_price,
                                "shares": _shares,
                                "position_value": trade_value,
                                "confidence": prefill_conf,
                                "horizon": horizon_label,
                                "horizon_days": trade_days,
                                "close_date": _close_date,
                            }
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error fetching price: {e}")
        else:
            # Step 2: Show preview card + Confirm / Cancel
            _p = _preview
            _dir_clr = GREEN if _p["direction"] == "LONG" else RED
            _dir_label = "Long (betting up)" if _p["direction"] == "LONG" else "Short (betting down)"
            _conf_str = f" · Signal conf: {_p['confidence']:.0f}%" if _p["confidence"] > 0 else ""

            st.markdown(
                f"<div style='background:linear-gradient(145deg,{BG_ELEVATED},{BG_CARD});border:1px solid {CYAN}40;"
                f"border-radius:10px;padding:18px 20px;margin-bottom:12px'>"
                f"<div style='color:{CYAN};font-size:0.6rem;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:1.5px;margin-bottom:10px'>Trade Preview</div>"
                f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:14px'>"
                f"<span style='color:{TEXT_PRIMARY};font-size:1.4rem;font-weight:900'>{_p['symbol']}</span>"
                f"<span style='color:{_dir_clr};font-size:0.7rem;font-weight:700;background:{BG_SURFACE};"
                f"padding:3px 10px;border-radius:4px;border:1px solid {_dir_clr}30'>{_dir_label}</span>"
                f"</div>"
                f"<div style='display:flex;gap:24px;flex-wrap:wrap'>"
                + "".join(
                    f"<div>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:600;text-transform:uppercase;"
                    f"letter-spacing:0.8px;margin-bottom:3px'>{lbl}</div>"
                    f"<div style='color:{clr};font-size:0.95rem;font-weight:700'>{val}</div>"
                    f"</div>"
                    for lbl, val, clr in [
                        ("Entry Price", f"${_p['entry_price']:,.2f}", TEXT_PRIMARY),
                        ("Position Size", f"${_p['position_value']:,.0f}", TEXT_PRIMARY),
                        ("Shares", f"{_p['shares']:,.4f}", TEXT_SECONDARY),
                        ("Horizon", _p["horizon"], TEXT_SECONDARY),
                        ("Auto-Closes", _p["close_date"], AMBER),
                        ("Remaining Cash", f"${cash - _p['position_value']:,.2f}",
                         GREEN if cash - _p['position_value'] > 0 else RED),
                    ]
                )
                + f"</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.65rem;margin-top:10px;font-style:italic'>"
                f"Position will auto-close on {_p['close_date']} at market price{_conf_str}</div>"
                f"</div>", unsafe_allow_html=True)

            _confirm_c1, _confirm_c2, _confirm_c3 = st.columns([1, 1, 4])
            with _confirm_c1:
                if st.button("Confirm Trade", type="primary", use_container_width=True, key="confirm_trade_btn"):
                    ok, msg = user_open_trade(
                        symbol=_p["symbol"],
                        direction=_p["direction"],
                        entry_price=_p["entry_price"],
                        position_value=_p["position_value"],
                        confidence=_p["confidence"],
                        horizon=_p["horizon"],
                        horizon_days=_p["horizon_days"],
                    )
                    st.session_state.pop("_trade_preview", None)
                    st.session_state.pop("paper_trade_prefill", None)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            with _confirm_c2:
                if st.button("Cancel", use_container_width=True, key="cancel_trade_btn"):
                    st.session_state.pop("_trade_preview", None)
                    st.rerun()

    st.markdown(f"<div style='height:20px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # EQUITY CURVE
    # ══════════════════════════════════════════════════════════════════════
    curve = stats.get("equity_curve", [])

    # Branded section header (always shown)
    st.markdown(
        f"<div class='sec-head' style='display:flex;align-items:center;gap:10px;margin-bottom:14px'>"
        f"<div class='sec-bar' style='width:3px;height:14px;background:{BRAND_GRAD};border-radius:2px'></div>"
        f"<span style='color:{TEXT_PRIMARY};font-size:0.7rem;font-weight:800;"
        f"text-transform:uppercase;letter-spacing:0.14em'>Equity Curve</span>"
        f"<span style='color:{TEXT_MUTED};font-size:0.62rem;font-weight:500;margin-left:6px'>"
        f"· portfolio value over time</span>"
        f"</div>",
        unsafe_allow_html=True)

    if len(curve) >= 2:
        curve_df = pd.DataFrame(curve)
        curve_df["date"] = pd.to_datetime(curve_df["date"])

        _eq_positive = curve_df["equity"].iloc[-1] >= STARTING_CAPITAL
        _eq_color = CHART_UP if _eq_positive else CHART_DOWN
        _eq_fill = "rgba(6,214,160,0.10)" if _eq_positive else "rgba(224,74,74,0.10)"

        _y_min = min(curve_df["equity"].min(), STARTING_CAPITAL)
        _y_max = max(curve_df["equity"].max(), STARTING_CAPITAL)
        _y_pad = (_y_max - _y_min) * 0.08 if _y_max > _y_min else 100
        _y_range = [_y_min - _y_pad, _y_max + _y_pad]

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=curve_df["date"], y=curve_df["equity"],
            mode="lines",
            line=dict(color=_eq_color, width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor=_eq_fill,
            name="Portfolio Equity",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Equity: $%{y:,.2f}<extra></extra>",
        ))
        fig_eq.add_hline(y=STARTING_CAPITAL, line_dash="dot",
                         line_color="rgba(255,255,255,0.2)",
                         annotation_text="$10K start",
                         annotation_font_color=TEXT_MUTED,
                         annotation_font_size=10)
        fig_eq.update_layout(
            height=280, template="plotly_dark",
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=11),
            xaxis=dict(gridcolor=CHART_GRID, showgrid=False),
            yaxis=dict(
                gridcolor=CHART_GRID,
                title=None, range=_y_range,
                tickprefix="$", tickformat=",",
            ),
            showlegend=False,
            hoverlabel=dict(bgcolor=BG_ELEVATED, font_color=TEXT_PRIMARY, bordercolor=BORDER_ACCENT, font_family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_eq, use_container_width=True, config={"displayModeBar": False})
    else:
        # Empty state — branded card with explanation
        _n_points = len(curve)
        _sub = ("Your equity curve shows how your portfolio value changes day-by-day. "
                "It needs at least 2 daily snapshots to render — check back tomorrow or "
                "after closing a few trades.")
        if _n_points == 1:
            _sub = ("You have 1 daily snapshot so far. The curve will populate as more days "
                    "of trading data accumulate.")
        st.markdown(
            f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
            f"border:1px solid {BORDER};border-radius:14px;padding:32px 24px;text-align:center;"
            f"position:relative;overflow:hidden;margin-bottom:16px'>"
            f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
            f"background:{BRAND_GRAD};opacity:0.55'></div>"
            f"<div style='font-size:1.4rem;margin-bottom:8px;opacity:0.55'>📈</div>"
            f"<div style='color:{TEXT_PRIMARY};font-size:0.9rem;font-weight:700;margin-bottom:6px'>"
            f"Equity curve building…</div>"
            f"<div style='color:{TEXT_SECONDARY};font-size:0.75rem;max-width:460px;margin:0 auto;line-height:1.55'>"
            f"{_sub}</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='height:16px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # OPEN POSITIONS — with close buttons
    # ══════════════════════════════════════════════════════════════════════
    open_pos = stats.get("open_positions", [])

    # ── Expiring trades alert ────────────────────────────────────────────
    if open_pos:
        from datetime import datetime as _dt_alert, timedelta as _td_alert
        _today_alert = _dt_alert.now().strftime("%Y-%m-%d")
        _week_alert = (_dt_alert.now() + _td_alert(days=7)).strftime("%Y-%m-%d")
        _exp_today = [p for p in open_pos if p.get("target_close_date", "") <= _today_alert]
        _exp_week = [p for p in open_pos
                     if _today_alert < p.get("target_close_date", "") <= _week_alert]
        if _exp_today or _exp_week:
            _parts = []
            if _exp_today:
                _parts.append(
                    f"<span style='color:{RED};font-weight:700'>{len(_exp_today)} closing today</span>"
                    f" ({', '.join(p['symbol'] for p in _exp_today[:4])})"
                )
            if _exp_week:
                _parts.append(
                    f"<span style='color:{AMBER};font-weight:700'>{len(_exp_week)} this week</span>"
                    f" ({', '.join(p['symbol'] for p in _exp_week[:4])})"
                )
            st.markdown(
                f"<div style='background:linear-gradient(145deg,{BG_ELEVATED},{BG_CARD});"
                f"border:1px solid {AMBER}30;border-radius:10px;padding:14px 18px;margin-bottom:16px;"
                f"display:flex;align-items:center;gap:12px'>"
                f"<div style='font-size:1.1rem'>⏰</div>"
                f"<div>"
                f"<div style='color:{TEXT_PRIMARY};font-size:0.78rem;font-weight:700;margin-bottom:2px'>Expiring Trades</div>"
                f"<div style='color:{TEXT_SECONDARY};font-size:0.72rem'>{' · '.join(_parts)}</div>"
                f"</div></div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='sec-head' style='display:flex;align-items:center;gap:10px;margin-bottom:14px'>"
        f"<div class='sec-bar' style='width:3px;height:14px;background:{BRAND_GRAD};border-radius:2px'></div>"
        f"<span style='color:{TEXT_PRIMARY};font-size:0.7rem;font-weight:800;"
        f"text-transform:uppercase;letter-spacing:0.14em'>Open Positions</span>"
        f"<span style='color:{TEXT_MUTED};font-size:0.62rem;font-weight:500;margin-left:6px'>· {len(open_pos)} active</span>"
        f"</div>",
        unsafe_allow_html=True)

    if open_pos:
        # Table header — use st.columns to match row widths exactly
        _hdr = (f"color:{TEXT_MUTED};font-size:0.52rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.08em")
        _hdr_cols = st.columns([11, 1.3])
        with _hdr_cols[0]:
            st.markdown(
                f"<div style='display:flex;align-items:center;padding:4px 16px;gap:4px'>"
                f"<div style='flex:1;{_hdr}'>Stock</div>"
                f"<div style='flex:0.7;{_hdr};text-align:center'>Side</div>"
                f"<div style='flex:1;{_hdr};text-align:right'>Entry</div>"
                f"<div style='flex:1;{_hdr};text-align:right'>Value</div>"
                f"<div style='flex:0.8;{_hdr};text-align:right'>P&L</div>"
                f"<div style='flex:0.8;{_hdr};text-align:right'>P&L %</div>"
                f"<div style='flex:1;{_hdr};text-align:center'>Horizon</div>"
                f"<div style='flex:1;{_hdr};text-align:center'>Expires</div>"
                f"</div>", unsafe_allow_html=True)
        with _hdr_cols[1]:
            st.markdown(
                f"<div style='{_hdr};text-align:center;padding-top:4px'>Action</div>",
                unsafe_allow_html=True)

        from datetime import datetime as _dt

        for pos in open_pos:
            pnl = pos.get("unrealised_pnl", 0)
            pnl_pct = pos.get("unrealised_pct", 0)
            pnl_clr = GREEN if pnl >= 0 else RED
            dir_clr = GREEN if pos["direction"] == "LONG" else RED
            pos_id = pos.get("id", f"{pos.get('entry_date', '')}_{pos['symbol']}_{pos.get('horizon', '')}")

            # Calculate days remaining
            _close_date = pos.get("target_close_date", "")
            _days_left = "—"
            _days_clr = TEXT_MUTED
            if _close_date:
                try:
                    _delta = (_dt.strptime(_close_date, "%Y-%m-%d") - _dt.now()).days
                    if _delta <= 0:
                        _days_left = "Today"
                        _days_clr = AMBER
                    elif _delta == 1:
                        _days_left = "1 day"
                        _days_clr = AMBER
                    else:
                        _days_left = f"{_delta}d"
                        _days_clr = TEXT_SECONDARY if _delta > 3 else AMBER
                except Exception:
                    pass

            # Inline row: data columns on left, Close button on right
            _row_cols = st.columns([11, 1.3])
            with _row_cols[0]:
                st.markdown(
                    f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;"
                    f"padding:10px 16px;margin-bottom:4px;display:flex;align-items:center;gap:4px;"
                    f"border-left:3px solid {dir_clr}'>"
                    f"<div style='flex:1;color:{TEXT_PRIMARY};font-weight:800;font-size:0.85rem;"
                    f"font-family:\"Inter\",sans-serif'>{pos['symbol']}</div>"
                    f"<div style='flex:0.7;text-align:center'>"
                    f"<span style='color:{dir_clr};font-size:0.62rem;font-weight:800;background:{BG_SURFACE};"
                    f"padding:3px 8px;border-radius:100px;letter-spacing:0.06em'>{pos['direction']}</span></div>"
                    f"<div style='flex:1;text-align:right;color:{TEXT_SECONDARY};font-size:0.8rem;"
                    f"font-variant-numeric:tabular-nums'>${pos['entry_price']:,.2f}</div>"
                    f"<div style='flex:1;text-align:right;color:{TEXT_PRIMARY};font-size:0.8rem;font-weight:700;"
                    f"font-variant-numeric:tabular-nums'>${pos['position_value']:,.2f}</div>"
                    f"<div style='flex:0.8;text-align:right;color:{pnl_clr};font-size:0.8rem;font-weight:800;"
                    f"font-variant-numeric:tabular-nums;white-space:nowrap'>${pnl:+,.2f}</div>"
                    f"<div style='flex:0.8;text-align:right;color:{pnl_clr};font-size:0.8rem;font-weight:800;"
                    f"font-variant-numeric:tabular-nums;white-space:nowrap'>{pnl_pct:+.1f}%</div>"
                    f"<div style='flex:1;text-align:center;color:{TEXT_SECONDARY};font-size:0.75rem'>{pos.get('horizon', '—')}</div>"
                    f"<div style='flex:1;text-align:center;color:{_days_clr};font-size:0.72rem;font-weight:700'>{_days_left}</div>"
                    f"</div>", unsafe_allow_html=True)
            with _row_cols[1]:
                if st.button("Close", key=f"close_{pos_id}", use_container_width=True):
                    ok, msg = user_close_trade(pos_id)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
    else:
        # Branded empty state with gradient hairline
        st.markdown(
            f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
            f"border:1px solid {BORDER};border-radius:14px;padding:36px 24px;text-align:center;"
            f"position:relative;overflow:hidden'>"
            f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
            f"background:{BRAND_GRAD};opacity:0.55'></div>"
            f"<div style='font-size:1.4rem;margin-bottom:8px;opacity:0.55'>📊</div>"
            f"<div style='color:{TEXT_PRIMARY};font-size:0.92rem;font-weight:700;margin-bottom:4px'>"
            f"No open positions yet</div>"
            f"<div style='color:{TEXT_SECONDARY};font-size:0.75rem;max-width:380px;margin:0 auto;line-height:1.5'>"
            f"Place a trade with the form above, or hit "
            f"<span style='color:{CYAN};font-weight:700'>Paper Trade</span> on any signal "
            f"in the Watchlist to open a position.</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='height:20px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # CLOSED TRADES
    # ══════════════════════════════════════════════════════════════════════
    closed = stats.get("closed_trades", [])
    if closed:
        st.markdown(
            f"<div class='sec-head' style='display:flex;align-items:center;gap:10px;margin-bottom:14px'>"
            f"<div class='sec-bar' style='width:3px;height:14px;background:{BRAND_GRAD};border-radius:2px'></div>"
            f"<span style='color:{TEXT_PRIMARY};font-size:0.7rem;font-weight:800;"
            f"text-transform:uppercase;letter-spacing:0.14em'>Closed Trades</span>"
            f"<span style='color:{TEXT_MUTED};font-size:0.62rem;font-weight:500;margin-left:6px'>"
            f"· last {min(20, stats['n_closed_trades'])} of {stats['n_closed_trades']}</span>"
            f"</div>",
            unsafe_allow_html=True)

        # Table header
        _hdr_closed = (f"color:{TEXT_MUTED};font-size:0.52rem;font-weight:700;"
                       f"text-transform:uppercase;letter-spacing:0.08em")
        st.markdown(
            f"<div style='display:flex;align-items:center;padding:4px 16px;gap:4px'>"
            f"<div style='flex:1;{_hdr_closed}'>Stock</div>"
            f"<div style='flex:0.7;{_hdr_closed};text-align:center'>Side</div>"
            f"<div style='flex:1;{_hdr_closed};text-align:right'>Entry</div>"
            f"<div style='flex:1;{_hdr_closed};text-align:right'>Exit</div>"
            f"<div style='flex:0.8;{_hdr_closed};text-align:right'>P&L</div>"
            f"<div style='flex:0.8;{_hdr_closed};text-align:right'>Return</div>"
            f"<div style='flex:0.6;{_hdr_closed};text-align:center'>Result</div>"
            f"<div style='flex:1;{_hdr_closed};text-align:center'>Closed</div>"
            f"</div>", unsafe_allow_html=True)

        for trade in reversed(closed[-20:]):
            _pnl = trade.get("realised_pnl", 0)
            _pnl_pct = trade.get("realised_pct", 0)
            _pnl_clr = GREEN if _pnl >= 0 else RED
            _dir_clr = GREEN if trade["direction"] == "LONG" else RED
            _correct = trade.get("correct_direction")
            _result = "✓" if _correct else "✗"
            _result_clr = GREEN if _correct else RED

            st.markdown(
                f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;"
                f"padding:10px 16px;margin-bottom:4px;display:flex;align-items:center;gap:4px;"
                f"border-left:3px solid {_result_clr}'>"
                f"<div style='flex:1;color:{TEXT_PRIMARY};font-weight:800;font-size:0.82rem;"
                f"font-family:\"Inter\",sans-serif'>{trade['symbol']}</div>"
                f"<div style='flex:0.7;text-align:center'>"
                f"<span style='color:{_dir_clr};font-size:0.6rem;font-weight:800;background:{BG_SURFACE};"
                f"padding:3px 8px;border-radius:100px;letter-spacing:0.06em'>{trade['direction']}</span></div>"
                f"<div style='flex:1;text-align:right;color:{TEXT_SECONDARY};font-size:0.78rem;"
                f"font-variant-numeric:tabular-nums'>${trade['entry_price']:,.2f}</div>"
                f"<div style='flex:1;text-align:right;color:{TEXT_SECONDARY};font-size:0.78rem;"
                f"font-variant-numeric:tabular-nums'>${trade.get('exit_price', 0):,.2f}</div>"
                f"<div style='flex:0.8;text-align:right;color:{_pnl_clr};font-size:0.78rem;font-weight:800;"
                f"font-variant-numeric:tabular-nums;white-space:nowrap'>${_pnl:+,.2f}</div>"
                f"<div style='flex:0.8;text-align:right;color:{_pnl_clr};font-size:0.78rem;font-weight:800;"
                f"font-variant-numeric:tabular-nums;white-space:nowrap'>{_pnl_pct:+.1f}%</div>"
                f"<div style='flex:0.6;text-align:center;font-size:1rem;color:{_result_clr};font-weight:800'>{_result}</div>"
                f"<div style='flex:1;text-align:center;color:{TEXT_MUTED};font-size:0.72rem'>{trade.get('exit_date', '—')}</div>"
                f"</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # TRADE ANALYTICS (if closed trades exist)
    # ══════════════════════════════════════════════════════════════════════
    if stats["n_closed_trades"] > 0:
        st.markdown(f"<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='sec-head' style='display:flex;align-items:center;gap:10px;margin-bottom:14px'>"
            f"<div class='sec-bar' style='width:3px;height:14px;background:{BRAND_GRAD};border-radius:2px'></div>"
            f"<span style='color:{TEXT_PRIMARY};font-size:0.7rem;font-weight:800;"
            f"text-transform:uppercase;letter-spacing:0.14em'>Trade Analytics</span>"
            f"<span style='color:{TEXT_MUTED};font-size:0.62rem;font-weight:500;margin-left:6px'>"
            f"· based on {stats['n_closed_trades']} closed trade{'s' if stats['n_closed_trades'] != 1 else ''}</span>"
            f"</div>",
            unsafe_allow_html=True)

        # ── Format values with empty-state handling ─────────────────────
        _avg_win_raw = stats.get('avg_win', 0) or 0
        _avg_loss_raw = stats.get('avg_loss', 0) or 0
        _pf_raw = stats.get('profit_factor')

        # Avg Win — green when positive, "—" when no wins
        if _avg_win_raw > 0:
            _avg_win_str = f"${_avg_win_raw:,.2f}"
            _avg_win_color = CHART_UP
            _avg_win_sub = "per winning trade"
        else:
            _avg_win_str = "—"
            _avg_win_color = TEXT_MUTED
            _avg_win_sub = "no wins yet"

        # Avg Loss — red when there are losses, "—" when none
        if _avg_loss_raw < 0:
            _avg_loss_str = f"${_avg_loss_raw:,.2f}"
            _avg_loss_color = CHART_DOWN
            _avg_loss_sub = "per losing trade"
        else:
            _avg_loss_str = "—"
            _avg_loss_color = TEXT_MUTED
            _avg_loss_sub = "no losses yet"

        # Profit Factor — None when no losses, big number otherwise
        if _pf_raw is None:
            _pf_str = "—"
            _pf_color = TEXT_MUTED
            _pf_sub = "needs at least 1 loss"
        elif _pf_raw >= 100:
            _pf_str = "∞"
            _pf_color = CHART_UP
            _pf_sub = "no losses — all profit"
        else:
            _pf_str = f"{_pf_raw:.2f}"
            if _pf_raw >= 2.0:
                _pf_color = CHART_UP
                _pf_sub = "strong — wins dwarf losses"
            elif _pf_raw >= 1.0:
                _pf_color = CHART_UP
                _pf_sub = "profitable — wins > losses"
            else:
                _pf_color = CHART_DOWN
                _pf_sub = "unprofitable — losses > wins"

        a1, a2, a3 = st.columns(3)
        for col, label, val, v_color, sub in [
            (a1, "Avg Win",       _avg_win_str,  _avg_win_color,  _avg_win_sub),
            (a2, "Avg Loss",      _avg_loss_str, _avg_loss_color, _avg_loss_sub),
            (a3, "Profit Factor", _pf_str,       _pf_color,       _pf_sub),
        ]:
            col.markdown(
                f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
                f"border:1px solid {BORDER};border-radius:12px;padding:16px;text-align:center;"
                f"position:relative;overflow:hidden'>"
                # Brand gradient hairline
                f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
                f"background:{BRAND_GRAD};opacity:0.55'></div>"
                f"<div style='color:{CYAN};font-size:0.56rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.14em'>{label}</div>"
                f"<div style='font-family:\"Inter\",sans-serif;color:{v_color};"
                f"font-size:1.2rem;font-weight:900;letter-spacing:-0.02em;"
                f"font-variant-numeric:tabular-nums;margin-top:8px'>{val}</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.6rem;margin-top:4px'>{sub}</div>"
                f"</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # RESET — at the bottom, subtle
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(f"<div style='height:30px'></div>", unsafe_allow_html=True)
    with st.expander("Portfolio Settings"):
        st.markdown(
            f"<div style='color:{TEXT_MUTED};font-size:0.75rem;line-height:1.6;margin-bottom:12px'>"
            f"This is your personal paper-trading sandbox. Resetting clears all positions and history, "
            f"returning the portfolio to $10,000. The system's Track Record is separate and unaffected.</div>",
            unsafe_allow_html=True)

        _set1, _set2, _set3 = st.columns([1, 1, 2])
        with _set1:
            if st.button("Reset Portfolio", key="reset_paper", type="secondary"):
                reset_portfolio()
                st.success("Portfolio reset to $10,000")
                st.rerun()
        with _set2:
            # Export portfolio as CSV
            _export_rows = []
            for p in stats.get("open_positions", []):
                _export_rows.append({
                    "Status": "Open", "Symbol": p["symbol"], "Direction": p["direction"],
                    "Entry": p["entry_price"], "Value": p["position_value"],
                    "P&L": p.get("unrealised_pnl", 0), "Horizon": p.get("horizon", ""),
                    "Opened": p.get("entry_date", ""), "Closes": p.get("target_close_date", ""),
                })
            for t in stats.get("closed_trades", []):
                _export_rows.append({
                    "Status": "Closed", "Symbol": t["symbol"], "Direction": t["direction"],
                    "Entry": t["entry_price"], "Exit": t.get("exit_price", ""),
                    "P&L": t.get("realised_pnl", 0), "Return%": t.get("realised_pct", 0),
                    "Horizon": t.get("horizon", ""),
                    "Opened": t.get("entry_date", ""), "Closed": t.get("exit_date", ""),
                })
            if _export_rows:
                _csv_data = pd.DataFrame(_export_rows).to_csv(index=False)
                st.download_button(
                    label="Export CSV",
                    data=_csv_data,
                    file_name="prediqt_portfolio.csv",
                    mime="text/csv",
                )
