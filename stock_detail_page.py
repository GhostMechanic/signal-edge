"""
stock_detail_page.py
--------------------
Lightweight Stock Detail page for Prediqt.
Displays comprehensive info about a single stock using existing data only
(no model retraining). Designed for speed.
"""

import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

from prediction_logger_v2 import _load_log
from watchlist import get_quick_signal

# ── Design tokens (mirror app.py's brand palette — keep in sync) ─────────────
BG_PRIMARY     = "#030508"
BG_SURFACE     = "#0a0e14"
BG_CARD        = "#0f1319"
BG_ELEVATED    = "#141a22"
BORDER         = "#1a2233"
TEXT_PRIMARY    = "#f0f4fa"
TEXT_SECONDARY  = "#8898aa"
TEXT_MUTED      = "#4a5a6e"
GREEN           = "#10b981"
RED             = "#ef4444"
AMBER           = "#f59e0b"
BLUE            = "#2080e5"
CYAN            = "#06d6a0"
BRAND_GRAD      = "linear-gradient(135deg, #2080e5 0%, #06d6a0 100%)"
# Chart palette — brand-first, convention-safe
CHART_UP        = CYAN
CHART_UP_FILL   = "rgba(6,214,160,0.22)"
CHART_DOWN      = "#e04a4a"
CHART_DOWN_FILL = "rgba(224,74,74,0.22)"
CHART_NEUTRAL   = BLUE
CHART_GRID      = "rgba(255,255,255,0.04)"

# ── Plotly chart template ────────────────────────────────────────────────────
_CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=11),
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(
        gridcolor=CHART_GRID, showline=False, zeroline=False,
        tickfont=dict(color=TEXT_MUTED, size=10),
    ),
    yaxis=dict(
        gridcolor=CHART_GRID, showline=False, zeroline=False,
        tickfont=dict(color=TEXT_MUTED, size=10), side="right",
    ),
)


def _card(inner_html: str, extra_style: str = "") -> str:
    """Wrap content in a styled card div."""
    return (
        f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
        f"border:1px solid {BORDER};border-radius:14px;padding:22px 24px;"
        f"margin-bottom:16px;{extra_style}'>{inner_html}</div>"
    )


def _section_title(label: str) -> str:
    return (
        f"<div style='color:{TEXT_SECONDARY};font-size:0.68rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px'>"
        f"{label}</div>"
    )


def _fmt_large_number(n) -> str:
    """Format large numbers with B/M/K suffix."""
    if n is None:
        return "N/A"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "N/A"
    if n >= 1_000_000_000_000:
        return f"${n / 1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"${n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"${n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"${n / 1_000:.1f}K"
    return f"${n:,.0f}"


def _fmt_volume(n) -> str:
    if n is None:
        return "N/A"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "N/A"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return f"{n:,.0f}"


# ═════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ═════════════════════════════════════════════════════════════════════════════

def render_stock_detail(symbol: str):
    """
    Render the full Stock Detail page for *symbol*.

    Fast path only -- pulls from existing prediction logs and yfinance;
    never retrains the model.
    """
    symbol = symbol.upper().strip()
    if not symbol:
        st.warning("No symbol provided.")
        return

    # ── Fetch quick signal (price, change, technicals) ───────────────────────
    sig = get_quick_signal(symbol)
    price = sig.get("price")
    change_1d = sig.get("change_1d")

    if price is None:
        st.markdown(
            _card(
                f"<div style='text-align:center;padding:40px 0'>"
                f"<div style='font-size:1.5rem;margin-bottom:10px;color:{TEXT_MUTED}'>--</div>"
                f"<div style='color:{TEXT_PRIMARY};font-size:1.1rem;font-weight:700'>"
                f"Unable to load data for {symbol}</div>"
                f"<div style='color:{TEXT_SECONDARY};font-size:0.8rem;margin-top:6px'>"
                f"Check the symbol and try again.</div></div>"
            ),
            unsafe_allow_html=True,
        )
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 1. HERO HEADER
    # ─────────────────────────────────────────────────────────────────────────
    change_color = GREEN if (change_1d or 0) >= 0 else RED
    change_sign = "+" if (change_1d or 0) >= 0 else ""
    change_str = f"{change_sign}{change_1d:.2f}%" if change_1d is not None else "N/A"

    # Build a tiny 5-day sparkline description from quick signal
    change_5d = sig.get("change_5d")
    sparkline_text = ""
    if change_5d is not None:
        spark_dir = "up" if change_5d >= 0 else "down"
        spark_color = GREEN if change_5d >= 0 else RED
        sparkline_text = (
            f"<span style='color:{spark_color};font-size:0.72rem;font-weight:600;"
            f"margin-left:10px'>5d: {'+' if change_5d >= 0 else ''}{change_5d:.2f}%"
            f" {'&#9650;' if change_5d >= 0 else '&#9660;'}</span>"
        )

    signal_badge = ""
    sig_label = sig.get("signal", "HOLD")
    sig_conf = sig.get("confidence", 0)
    sig_colors = {"BUY": GREEN, "SELL": RED, "HOLD": AMBER}
    sig_color = sig_colors.get(sig_label, AMBER)
    signal_badge = (
        f"<span style='display:inline-block;padding:3px 12px;border-radius:6px;"
        f"background:{sig_color}18;border:1px solid {sig_color}44;color:{sig_color};"
        f"font-size:0.7rem;font-weight:700;letter-spacing:0.04em;margin-left:14px'>"
        f"{sig_label} {sig_conf:.0f}%</span>"
    )

    st.markdown(
        f"<div style='text-align:center;padding:10px 0 24px;position:relative'>"
        f"<div style='position:absolute;top:-20px;left:50%;transform:translateX(-50%);width:600px;height:300px;"
        f"background:radial-gradient(circle,rgba(6,214,160,0.04) 0%,rgba(37,99,235,0.02) 40%,transparent 70%);"
        f"pointer-events:none'></div>"
        f"<div style='color:{TEXT_MUTED};font-size:0.7rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.1em;margin-bottom:4px'>Stock Detail</div>"
        f"<h1 style='color:{TEXT_PRIMARY};font-size:2.2rem;font-weight:900;margin:0 0 2px;"
        f"letter-spacing:-0.03em'>{symbol}</h1>"
        f"<div style='display:flex;align-items:baseline;justify-content:center;gap:6px;flex-wrap:wrap'>"
        f"<span style='color:{TEXT_PRIMARY};font-size:1.4rem;font-weight:700'>${price:,.2f}</span>"
        f"<span style='color:{change_color};font-size:0.95rem;font-weight:700'>{change_str}</span>"
        f"{sparkline_text}{signal_badge}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. ACTIVE PREDICTIONS
    # ─────────────────────────────────────────────────────────────────────────
    all_entries = _load_log()
    sym_entries = [e for e in all_entries if e.get("symbol", "").upper() == symbol]

    pending_rows = []
    scored_results = []

    for entry in sym_entries:
        ts = entry.get("timestamp", "")[:10]
        for horizon, hdata in entry.get("horizons", {}).items():
            pred_return = hdata.get("predicted_return", 0)
            confidence = hdata.get("confidence", 0)
            direction = hdata.get("direction", "up")

            if hdata.get("final_scored"):
                scored_results.append({
                    "date": ts,
                    "horizon": horizon,
                    "predicted_return": pred_return,
                    "confidence": confidence,
                    "direction": direction,
                    "direction_correct": hdata.get("final_correct", False),
                })
            else:
                # Also count partially scored as pending if not final
                pending_rows.append({
                    "date": ts,
                    "horizon": horizon,
                    "predicted_return": pred_return,
                    "confidence": confidence,
                    "direction": direction,
                    "partial_scores": len(hdata.get("scores", {})),
                })

    # Also gather any-scored results (not just final)
    any_scored = []
    for entry in sym_entries:
        ts = entry.get("timestamp", "")[:10]
        for horizon, hdata in entry.get("horizons", {}).items():
            scores = hdata.get("scores", {})
            if scores:
                latest_key = max(scores.keys(), key=lambda k: int(k.replace("d", "")))
                any_scored.append({
                    "date": ts,
                    "horizon": horizon,
                    "direction_correct": scores[latest_key].get("direction_correct", False),
                })

    st.markdown(
        _card(
            _section_title("Active Predictions") + _render_pending_table(pending_rows)
        ),
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. PER-STOCK ACCURACY
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        _card(_section_title(f"Prediqt Accuracy on {symbol}") + _render_accuracy(symbol, scored_results, any_scored)),
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. KEY STATS  &  5. PRICE CHART  (side by side on wide screens)
    # ─────────────────────────────────────────────────────────────────────────
    col_stats, col_chart = st.columns([1, 2])

    with col_stats:
        _render_key_stats(symbol)

    with col_chart:
        _render_price_chart(symbol)


# ═════════════════════════════════════════════════════════════════════════════
# SUB-RENDERERS
# ═════════════════════════════════════════════════════════════════════════════

def _render_pending_table(pending: list) -> str:
    """Render pending (unscored) predictions as an HTML table."""
    if not pending:
        return (
            f"<div style='color:{TEXT_MUTED};font-size:0.82rem;padding:18px 0;"
            f"text-align:center'>No active predictions for this symbol.</div>"
        )

    # Sort newest first
    pending.sort(key=lambda r: r["date"], reverse=True)

    header = (
        f"<tr style='border-bottom:1px solid {BORDER}'>"
        f"<th style='color:{TEXT_MUTED};font-size:0.65rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.06em;padding:6px 10px;text-align:left'>Date</th>"
        f"<th style='color:{TEXT_MUTED};font-size:0.65rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.06em;padding:6px 10px;text-align:left'>Horizon</th>"
        f"<th style='color:{TEXT_MUTED};font-size:0.65rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.06em;padding:6px 10px;text-align:right'>Predicted</th>"
        f"<th style='color:{TEXT_MUTED};font-size:0.65rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.06em;padding:6px 10px;text-align:right'>Confidence</th>"
        f"<th style='color:{TEXT_MUTED};font-size:0.65rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.06em;padding:6px 10px;text-align:center'>Status</th>"
        f"</tr>"
    )

    rows = ""
    for r in pending[:15]:  # cap at 15 rows
        pret = r["predicted_return"] * 100
        ret_color = GREEN if pret >= 0 else RED
        ret_sign = "+" if pret >= 0 else ""
        dir_arrow = "&#9650;" if r["direction"] == "up" else "&#9660;"
        dir_color = GREEN if r["direction"] == "up" else RED

        partial = r.get("partial_scores", 0)
        status = f"<span style='color:{AMBER};font-size:0.72rem'>pending</span>"
        if partial > 0:
            status = (
                f"<span style='color:{BLUE};font-size:0.72rem'>"
                f"{partial} scored</span>"
            )

        rows += (
            f"<tr style='border-bottom:1px solid {BORDER}22'>"
            f"<td style='color:{TEXT_SECONDARY};font-size:0.78rem;padding:8px 10px'>{r['date']}</td>"
            f"<td style='color:{TEXT_PRIMARY};font-size:0.78rem;padding:8px 10px;font-weight:600'>{r['horizon']}</td>"
            f"<td style='color:{ret_color};font-size:0.78rem;padding:8px 10px;text-align:right;font-weight:600'>"
            f"<span style='color:{dir_color};margin-right:4px'>{dir_arrow}</span>"
            f"{ret_sign}{pret:.2f}%</td>"
            f"<td style='color:{TEXT_SECONDARY};font-size:0.78rem;padding:8px 10px;text-align:right'>"
            f"{r['confidence']:.0f}%</td>"
            f"<td style='padding:8px 10px;text-align:center'>{status}</td>"
            f"</tr>"
        )

    return (
        f"<table style='width:100%;border-collapse:collapse'>"
        f"{header}{rows}</table>"
    )


def _render_accuracy(symbol: str, scored_final: list, any_scored: list) -> str:
    """Render per-stock accuracy summary."""
    # Prefer final-scored; fall back to any-scored
    use_scored = scored_final if scored_final else any_scored
    n = len(use_scored)

    if n == 0:
        return (
            f"<div style='color:{TEXT_MUTED};font-size:0.82rem;padding:18px 0;"
            f"text-align:center'>No scored predictions yet for {symbol}. "
            f"Predictions are scored automatically as time passes.</div>"
        )

    correct = sum(1 for r in use_scored if r.get("direction_correct"))
    wrong = n - correct
    accuracy = (correct / n * 100) if n > 0 else 0
    acc_color = GREEN if accuracy >= 55 else AMBER if accuracy >= 45 else RED

    label = "final-scored" if scored_final else "scored"

    # W/L bar
    w_pct = (correct / n * 100) if n > 0 else 0
    l_pct = 100 - w_pct

    return (
        f"<div style='text-align:center;padding:10px 0 6px'>"
        f"<div style='color:{acc_color};font-size:1.6rem;font-weight:900;letter-spacing:-0.02em'>"
        f"{accuracy:.1f}%</div>"
        f"<div style='color:{TEXT_SECONDARY};font-size:0.78rem;margin-top:2px'>"
        f"Prediqt accuracy on <span style='color:{TEXT_PRIMARY};font-weight:700'>{symbol}</span>: "
        f"{accuracy:.1f}% across {n} {label} predictions</div>"
        f"</div>"

        # W/L breakdown
        f"<div style='display:flex;align-items:center;gap:14px;justify-content:center;"
        f"margin:14px 0 6px'>"
        f"<div style='display:flex;align-items:center;gap:5px'>"
        f"<div style='width:8px;height:8px;border-radius:50%;background:{GREEN}'></div>"
        f"<span style='color:{GREEN};font-size:0.76rem;font-weight:600'>W {correct}</span></div>"
        f"<div style='display:flex;align-items:center;gap:5px'>"
        f"<div style='width:8px;height:8px;border-radius:50%;background:{RED}'></div>"
        f"<span style='color:{RED};font-size:0.76rem;font-weight:600'>L {wrong}</span></div>"
        f"</div>"

        # Visual bar
        f"<div style='width:100%;height:6px;border-radius:3px;background:{BG_ELEVATED};"
        f"overflow:hidden;margin-top:6px'>"
        f"<div style='height:100%;width:{w_pct:.1f}%;background:{GREEN};border-radius:3px'></div>"
        f"</div>"
    )


def _render_key_stats(symbol: str):
    """Render fundamental stats card using yfinance info."""
    st.markdown(
        _section_title("Key Stats"),
        unsafe_allow_html=True,
    )

    try:
        tk = yf.Ticker(symbol)
        info = tk.info or {}
    except Exception:
        info = {}

    market_cap = info.get("marketCap")
    pe_ratio = info.get("trailingPE")
    week52_low = info.get("fiftyTwoWeekLow")
    week52_high = info.get("fiftyTwoWeekHigh")
    avg_volume = info.get("averageDailyVolume10Day") or info.get("averageVolume")

    def _stat_row(label: str, value: str) -> str:
        return (
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:10px 0;border-bottom:1px solid {BORDER}22'>"
            f"<span style='color:{TEXT_MUTED};font-size:0.78rem'>{label}</span>"
            f"<span style='color:{TEXT_PRIMARY};font-size:0.82rem;font-weight:600'>{value}</span>"
            f"</div>"
        )

    pe_str = f"{pe_ratio:.1f}" if pe_ratio is not None else "N/A"
    range_str = (
        f"${week52_low:,.2f} - ${week52_high:,.2f}"
        if week52_low is not None and week52_high is not None
        else "N/A"
    )

    stats_html = (
        _stat_row("Market Cap", _fmt_large_number(market_cap))
        + _stat_row("P/E Ratio", pe_str)
        + _stat_row("52-Week Range", range_str)
        + _stat_row("Avg Volume", _fmt_volume(avg_volume))
    )

    st.markdown(
        _card(stats_html, "padding:14px 20px"),
        unsafe_allow_html=True,
    )


def _render_price_chart(symbol: str):
    """Render a 6-month price chart using plotly."""
    st.markdown(
        _section_title("6-Month Price"),
        unsafe_allow_html=True,
    )

    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="6mo")
    except Exception:
        hist = None

    if hist is None or hist.empty:
        st.markdown(
            _card(
                f"<div style='color:{TEXT_MUTED};font-size:0.82rem;text-align:center;"
                f"padding:40px 0'>Price data unavailable.</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    hist.index = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index

    # Determine color based on period performance (brand-first, convention-safe)
    first_close = float(hist["Close"].iloc[0])
    last_close = float(hist["Close"].iloc[-1])
    _up = last_close >= first_close
    line_color = CHART_UP if _up else CHART_DOWN
    fill_color = "rgba(6,214,160,0.08)" if _up else "rgba(224,74,74,0.08)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist["Close"],
            mode="lines",
            line=dict(color=line_color, width=2),
            fill="tozeroy",
            fillcolor=fill_color,
            hovertemplate="$%{y:,.2f}<extra>%{x|%b %d}</extra>",
        )
    )

    fig.update_layout(
        **_CHART_LAYOUT,
        height=300,
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=BG_ELEVATED,
            bordercolor=BORDER,
            font=dict(color=TEXT_PRIMARY, size=12),
        ),
    )

    # Tighten y-axis around actual data
    y_min = float(hist["Close"].min()) * 0.97
    y_max = float(hist["Close"].max()) * 1.03
    fig.update_yaxes(range=[y_min, y_max])

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )
