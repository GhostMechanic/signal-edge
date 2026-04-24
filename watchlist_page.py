"""
watchlist_page.py — Watchlist multi-stock dashboard page.
Full watchlist management: add/remove stocks, load indices, batch scan,
and live signal overview cards.
"""

import streamlit as st
import numpy as np
# Watchlist persistence routes through db.py (feature-flagged: file-backed
# by default, Supabase when USE_SUPABASE=true). get_quick_signal is a pure
# compute function — stays in watchlist.py.
from db import add_to_watchlist, remove_from_watchlist, save_watchlist
from watchlist import get_quick_signal
from universe import get_dow30_tickers, get_nasdaq100_tickers, get_sp500_tickers

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
PURPLE         = "#1e90ff"
BRAND_GRAD     = "linear-gradient(135deg, #2080e5 0%, #06d6a0 100%)"


def render_watchlist_page():
    try:
        # Compact button styling for watchlist rows
        st.markdown(
            "<style>"
            "div[data-testid='stHorizontalBlock'] button[kind='secondary'] {"
            "  padding: 2px 8px !important; font-size: 0.62rem !important;"
            "  min-height: 0 !important; height: 26px !important;"
            "  border-radius: 4px !important;"
            "}"
            "</style>",
            unsafe_allow_html=True)

        # Header
        st.markdown(
            "<div class='page-header'>"
            "<div class='ph-eyebrow'>Portfolio Watch</div>"
            "<h1>Your <span class='grad'>watchlist</span>.</h1>"
            "<p class='ph-sub'>Track tickers, load an index, and batch-scan "
            "your whole universe for the next setup.</p>"
            "</div>", unsafe_allow_html=True)

        # ── Management toolbar ───────────────────────────────────────────────
        st.markdown(
            f"<div style='color:{TEXT_MUTED};font-size:0.55rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:2px;margin-bottom:10px'>Manage Watchlist</div>",
            unsafe_allow_html=True)

        # Add stocks row
        _add_col1, _add_col2 = st.columns([4, 1])
        with _add_col1:
            new_sym = st.text_input(
                "Add ticker", value="", placeholder="AAPL, MSFT, GOOG…",
                label_visibility="collapsed", key="wl_page_add_input"
            ).upper().strip()
        with _add_col2:
            if st.button("Add", use_container_width=True, key="wl_page_add_btn"):
                if new_sym:
                    # Support comma-separated tickers
                    for _s in [s.strip() for s in new_sym.split(",") if s.strip()]:
                        if _s not in st.session_state.watchlist:
                            st.session_state.watchlist = add_to_watchlist(_s) or st.session_state.watchlist
                    st.rerun()

        # Load index row
        _idx_c1, _idx_c2, _idx_c3, _idx_c4 = st.columns(4)
        with _idx_c1:
            if st.button("DOW 30", use_container_width=True, key="wl_load_dow",
                         help="Add all Dow Jones stocks"):
                st.session_state.watchlist = save_watchlist(
                    sorted(set((st.session_state.watchlist or []) + get_dow30_tickers()))
                ) or st.session_state.watchlist
                st.rerun()
        with _idx_c2:
            if st.button("NASDAQ 100", use_container_width=True, key="wl_load_ndx",
                         help="Add NASDAQ-100 stocks"):
                st.session_state.watchlist = save_watchlist(
                    sorted(set((st.session_state.watchlist or []) + get_nasdaq100_tickers()))
                ) or st.session_state.watchlist
                st.rerun()
        with _idx_c3:
            if st.button("S&P 500", use_container_width=True, key="wl_load_sp500",
                         help="Add all S&P 500 stocks"):
                with st.spinner("Fetching S&P 500…"):
                    sp_tickers = get_sp500_tickers()
                st.session_state.watchlist = save_watchlist(
                    sorted(set((st.session_state.watchlist or []) + sp_tickers))
                ) or st.session_state.watchlist
                st.rerun()
        with _idx_c4:
            if st.session_state.watchlist:
                if st.button("Clear All", use_container_width=True, key="wl_clear_all",
                             help="Remove all stocks from watchlist"):
                    st.session_state.watchlist = save_watchlist([]) or []
                    st.rerun()

        # Scan buttons
        if st.session_state.watchlist:
            _scan_c1, _scan_c2, _scan_c3 = st.columns([2, 2, 1])
            with _scan_c1:
                if st.button("⚡ Deep Scan", use_container_width=True, key="wl_batch_scan_btn",
                             type="primary",
                             help="Full model training + prediction logging (slower, builds track record)"):
                    st.session_state.batch_scan_requested = True
                    st.rerun()
            with _scan_c2:
                if st.button("🔄 Refresh Signals", use_container_width=True, key="wl_refresh_signals",
                             help="Refresh quick technical signals (fast)"):
                    # Clear the signal cache to force re-fetch
                    get_quick_signal.clear()
                    st.rerun()

        st.markdown(f"<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Stock list ───────────────────────────────────────────────────────
        if not st.session_state.watchlist:
            st.markdown(
                f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});border:1px solid {BORDER};"
                f"border-radius:14px;padding:50px 30px;text-align:center'>"
                f"<div style='font-size:2rem;margin-bottom:12px'>⬡</div>"
                f"<div style='color:{TEXT_PRIMARY};font-size:1.1rem;font-weight:700;margin-bottom:8px'>"
                f"No stocks tracked yet</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.8rem;max-width:360px;margin:0 auto;line-height:1.6'>"
                f"Add individual tickers above, or load an entire index to get started.</div>"
                f"</div>", unsafe_allow_html=True)
        else:

            # Fetch and cache signals
            signals = {}
            _wl = st.session_state.watchlist
            _progress = st.progress(0, text=f"Loading signals for {len(_wl)} stocks…")
            for _i, sym in enumerate(_wl):
                _progress.progress((_i + 1) / len(_wl), text=f"Loading {sym}… ({_i+1}/{len(_wl)})")
                try:
                    sig = get_quick_signal(sym)
                    signals[sym] = sig
                except Exception as e:
                    signals[sym] = {
                        "symbol": sym,
                        "error": str(e),
                        "signal": "HOLD",
                        "confidence": 0,
                        "price": None,
                        "change_1d": None,
                        "rsi": None,
                        "ma50_dist": None,
                        "vol_ratio": None,
                    }
            _progress.empty()

            # Summary strip
            bullish_count = sum(1 for s in signals.values() if s.get("signal") == "BUY")
            bearish_count = sum(1 for s in signals.values() if s.get("signal") == "SELL")
            neutral_count = len(signals) - bullish_count - bearish_count
            avg_conf = int(np.mean([s.get("confidence", 0) for s in signals.values()])) if signals else 0

            _pill = (
                "display:inline-flex;align-items:center;gap:5px;"
                f"padding:6px 14px;border-radius:8px;background:{BG_CARD};"
                f"border:1px solid {BORDER}"
            )
            st.markdown(
                f"<div style='display:flex;gap:8px;flex-wrap:wrap;justify-content:center'>"
                f"<div style='{_pill}'>"
                f"<span style='color:{TEXT_MUTED};font-size:0.6rem;font-weight:600'>Bullish</span>"
                f"<span style='color:{GREEN};font-size:0.95rem;font-weight:800'>{bullish_count}</span>"
                f"</div>"
                f"<div style='{_pill}'>"
                f"<span style='color:{TEXT_MUTED};font-size:0.6rem;font-weight:600'>Bearish</span>"
                f"<span style='color:{RED};font-size:0.95rem;font-weight:800'>{bearish_count}</span>"
                f"</div>"
                f"<div style='{_pill}'>"
                f"<span style='color:{TEXT_MUTED};font-size:0.6rem;font-weight:600'>Neutral</span>"
                f"<span style='color:{AMBER};font-size:0.95rem;font-weight:800'>{neutral_count}</span>"
                f"</div>"
                f"<div style='{_pill}'>"
                f"<span style='color:{TEXT_MUTED};font-size:0.6rem;font-weight:600'>Avg Conf</span>"
                f"<span style='color:{BLUE};font-size:0.95rem;font-weight:800'>{avg_conf}%</span>"
                f"</div>"
                f"</div>", unsafe_allow_html=True)

            st.markdown(f"<div style='height:12px'></div>", unsafe_allow_html=True)

            # ── Filter + Sort controls (single row) ─────────────────────
            _ctrl_c1, _ctrl_c2, _ctrl_c3 = st.columns([3, 2, 1])
            with _ctrl_c1:
                _filter = st.radio(
                    "Filter", ["All", "Buy", "Sell", "Hold"],
                    horizontal=True, label_visibility="collapsed", key="wl_filter",
                )
            with _ctrl_c2:
                _sort_by = st.selectbox(
                    "Sort by", ["Signal Strength", "Today", "1 Month", "RSI", "Name"],
                    label_visibility="collapsed", key="wl_sort",
                )

            if _filter == "Buy":
                _filtered = {k: v for k, v in signals.items() if v.get("signal") == "BUY"}
            elif _filter == "Sell":
                _filtered = {k: v for k, v in signals.items() if v.get("signal") == "SELL"}
            elif _filter == "Hold":
                _filtered = {k: v for k, v in signals.items() if v.get("signal") == "HOLD"}
            else:
                _filtered = signals

            # Sort the filtered signals
            def _sort_key(item):
                _sym, _sd = item
                if _sort_by == "Signal Strength":
                    return -(_sd.get("confidence", 0))
                elif _sort_by == "Today":
                    return -(_sd.get("change_1d", 0) or 0)
                elif _sort_by == "1 Month":
                    return -(_sd.get("change_21d", 0) or 0)
                elif _sort_by == "RSI":
                    return _sd.get("rsi", 50) or 50
                else:
                    return _sym

            _sorted = sorted(_filtered.items(), key=_sort_key)

            # ── Table header ─────────────────────────────────────────────
            _hdr_style = (
                f"color:{TEXT_MUTED};font-size:0.55rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.8px"
            )
            st.markdown(
                f"<div style='display:flex;align-items:center;padding:8px 16px 6px;gap:4px'>"
                f"<div style='flex:1.2;{_hdr_style}'>Stock</div>"
                f"<div style='flex:1;{_hdr_style};text-align:right'>Price</div>"
                f"<div style='flex:0.8;{_hdr_style};text-align:right'>Today</div>"
                f"<div style='flex:0.8;{_hdr_style};text-align:right'>1 Month</div>"
                f"<div style='flex:0.6;{_hdr_style};text-align:center'>RSI</div>"
                f"<div style='flex:0.7;{_hdr_style};text-align:center'>MA50</div>"
                f"<div style='flex:1.2;{_hdr_style};text-align:center'>Signal</div>"
                f"<div style='flex:0.4;{_hdr_style};text-align:right'>"
                f"{len(_filtered)}/{len(signals)}</div>"
                f"</div>",
                unsafe_allow_html=True)

            # ── Dense table rows ─────────────────────────────────────────
            for sym, sig_data in _sorted:
                try:
                    if sig_data.get("error"):
                        st.markdown(
                            f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:6px;"
                            f"padding:8px 16px;margin-bottom:2px;display:flex;align-items:center'>"
                            f"<div style='flex:1.2;color:{TEXT_PRIMARY};font-weight:700;font-size:0.85rem'>{sym}</div>"
                            f"<div style='flex:4;color:{TEXT_MUTED};font-size:0.72rem;font-style:italic'>Data unavailable</div>"
                            f"</div>", unsafe_allow_html=True)
                        continue

                    price = sig_data.get("price", 0) or 0
                    change_1d = sig_data.get("change_1d", 0) or 0
                    change_21d = sig_data.get("change_21d", 0) or 0
                    signal = sig_data.get("signal", "HOLD")
                    confidence = sig_data.get("confidence", 0)
                    rsi = sig_data.get("rsi")
                    ma50_dist = sig_data.get("ma50_dist")

                    sig_color = GREEN if signal == "BUY" else (RED if signal == "SELL" else AMBER)
                    sig_bg = GREEN_DIM if signal == "BUY" else (RED_DIM if signal == "SELL" else AMBER_DIM)
                    chg_1d_color = GREEN if change_1d >= 0 else RED
                    chg_21d_color = GREEN if change_21d >= 0 else RED
                    rsi_color = RED if rsi and rsi > 70 else (GREEN if rsi and rsi < 30 else TEXT_SECONDARY)
                    ma50_color = GREEN if ma50_dist and ma50_dist > 0 else RED

                    # RSI and MA50 display strings
                    rsi_str = f"{rsi:.0f}" if rsi else "—"
                    ma50_str = f"{ma50_dist:+.1f}%" if ma50_dist is not None else "—"

                    # Confidence bar width
                    conf_w = max(8, confidence)

                    st.markdown(
                        f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:6px;"
                        f"padding:10px 16px;margin-bottom:2px;display:flex;align-items:center;gap:4px;"
                        f"border-left:3px solid {sig_color}'>"
                        # Ticker
                        f"<div style='flex:1.2;display:flex;align-items:center;gap:6px'>"
                        f"<span style='color:{TEXT_PRIMARY};font-weight:800;font-size:0.88rem'>{sym}</span>"
                        f"</div>"
                        # Price
                        f"<div style='flex:1;text-align:right'>"
                        f"<span style='color:{TEXT_PRIMARY};font-weight:600;font-size:0.82rem'>${price:,.2f}</span>"
                        f"</div>"
                        # Today change
                        f"<div style='flex:0.8;text-align:right'>"
                        f"<span style='color:{chg_1d_color};font-weight:700;font-size:0.78rem'>{change_1d:+.2f}%</span>"
                        f"</div>"
                        # 1 Month change
                        f"<div style='flex:0.8;text-align:right'>"
                        f"<span style='color:{chg_21d_color};font-weight:700;font-size:0.78rem'>{change_21d:+.2f}%</span>"
                        f"</div>"
                        # RSI
                        f"<div style='flex:0.6;text-align:center'>"
                        f"<span style='color:{rsi_color};font-weight:700;font-size:0.78rem'>{rsi_str}</span>"
                        f"</div>"
                        # MA50
                        f"<div style='flex:0.7;text-align:center'>"
                        f"<span style='color:{ma50_color};font-weight:600;font-size:0.75rem'>{ma50_str}</span>"
                        f"</div>"
                        # Signal + confidence inline
                        f"<div style='flex:1.2;display:flex;align-items:center;gap:6px;justify-content:center'>"
                        f"<span style='color:{sig_color};background:{sig_bg};padding:2px 8px;"
                        f"border-radius:4px;font-size:0.58rem;font-weight:700;letter-spacing:0.5px;"
                        f"border:1px solid {sig_color}30'>{signal}</span>"
                        f"<div style='display:flex;align-items:center;gap:4px;min-width:56px'>"
                        f"<div style='width:32px;height:3px;background:{BG_SURFACE};border-radius:2px;overflow:hidden'>"
                        f"<div style='height:100%;width:{conf_w}%;background:{sig_color};border-radius:2px'></div>"
                        f"</div>"
                        f"<span style='color:{sig_color};font-size:0.62rem;font-weight:700'>{confidence}%</span>"
                        f"</div>"
                        f"</div>"
                        # Action spacer
                        f"<div style='flex:0.4'></div>"
                        f"</div>",
                        unsafe_allow_html=True)

                    # Slim inline buttons — single row, tiny
                    _bc1, _bc2, _bc3, _bc4, _bc5 = st.columns([1, 1, 1, 1, 6])
                    with _bc1:
                        if st.button("👁 View", key=f"view_{sym}",
                                     use_container_width=True):
                            st.session_state.detail_symbol = sym
                            st.session_state.detail_return_page = "Watchlist"
                            st.session_state.current_page = "Stock Detail"
                            st.rerun()
                    with _bc2:
                        if st.button("▶ Analyze", key=f"analyze_{sym}",
                                     use_container_width=True):
                            st.session_state.selected_ticker = sym
                            st.session_state.current_page = "Analyze"
                            st.session_state.auto_analyze = True
                            st.rerun()
                    with _bc3:
                        _pt_dir = "LONG" if signal == "BUY" else ("SHORT" if signal == "SELL" else "LONG")
                        if st.button("📊 Trade", key=f"paper_trade_{sym}",
                                     use_container_width=True):
                            st.session_state.paper_trade_prefill = {
                                "symbol": sym,
                                "direction": _pt_dir,
                                "confidence": confidence,
                            }
                            st.session_state.current_page = "Portfolio"
                            st.rerun()
                    with _bc4:
                        if st.button("✕ Remove", key=f"wl_remove_{sym}",
                                     use_container_width=True):
                            st.session_state.watchlist = remove_from_watchlist(sym)
                            st.rerun()

                except Exception:
                    st.markdown(
                        f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:6px;"
                        f"padding:8px 16px;margin-bottom:2px'>"
                        f"<span style='color:{TEXT_PRIMARY};font-weight:700'>{sym}</span>"
                        f" <span style='color:{RED};font-size:0.72rem'>Display error</span>"
                        f"</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Watchlist error: {str(e)}")
