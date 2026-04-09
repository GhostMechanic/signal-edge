"""
app.py  –  Stock Signal Engine v3.0
─────────────────────────────────────
Premium decision-engine interface.
Architecture: all analysis results stored in st.session_state so that
widget interactions (chart period, tab switches, etc.) never trigger a
re-analysis.  Analysis only runs when the user clicks Analyze.

Design philosophy:
  - Decide in 5 seconds (top hero signal card)
  - Validate in 30 seconds (evidence strip + risk metrics)
  - Deep dive in 2 minutes (tabs for chart, options, backtest, model)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_fetcher import (
    fetch_stock_data, fetch_stock_info, fetch_market_context,
    fetch_fundamentals, fetch_earnings_data, fetch_options_data,
    engineer_features, HORIZONS,
)
from model import StockPredictor, HAS_LGB, HAS_SHAP, HAS_OPTUNA
from backtester import run_backtest, backtest_summary_df
from analyzer import generate_analysis, RECOMMENDATION_COLORS
from options_analyzer import generate_options_report
from regime_detector import detect_regime
from prediction_logger import log_prediction, get_track_record


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SignalEdge — ML Stock Signals",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design tokens ────────────────────────────────────────────────────────────
BG_PRIMARY   = "#06080d"
BG_SURFACE   = "#0c1017"
BG_CARD      = "#111620"
BG_ELEVATED  = "#161d2b"
BORDER       = "#1a2235"
BORDER_ACCENT= "#1e3050"
TEXT_PRIMARY  = "#e8ecf5"
TEXT_SECONDARY= "#8494a7"
TEXT_MUTED    = "#4a5568"
GREEN        = "#10b981"
GREEN_DIM    = "#064e3b"
RED          = "#ef4444"
RED_DIM      = "#450a0a"
AMBER        = "#f59e0b"
AMBER_DIM    = "#451a03"
BLUE         = "#3b82f6"
PURPLE       = "#8b5cf6"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Reset ── */
.stApp {{ background-color: {BG_PRIMARY}; color: {TEXT_PRIMARY}; font-family: 'Inter', -apple-system, sans-serif; }}
.block-container {{ padding-top: 1rem; max-width: 1400px; }}

/* ── Surface & Cards ── */
.surface {{ background: {BG_SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; }}
.card {{ background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 20px; }}
.card-sm {{ background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 14px 16px; }}

/* ── Hero Signal Card ── */
.hero-signal {{
    border-radius: 16px; padding: 32px 28px; text-align: center;
    position: relative; overflow: hidden;
}}
.hero-signal::before {{
    content: ''; position: absolute; inset: 0; border-radius: 16px;
    background: radial-gradient(ellipse at 50% 0%, var(--glow) 0%, transparent 70%);
    opacity: 0.12;
}}

/* ── Metric Row ── */
.mrow {{ display: flex; justify-content: space-between; align-items: center;
         padding: 10px 0; border-bottom: 1px solid {BORDER}; }}
.mrow:last-child {{ border-bottom: none; }}
.mrow-label {{ color: {TEXT_SECONDARY}; font-size: 0.78rem; font-weight: 500; }}
.mrow-value {{ color: {TEXT_PRIMARY}; font-size: 0.85rem; font-weight: 700; }}

/* ── Stat Tile ── */
.stat {{
    background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 16px 14px; text-align: center;
}}
.stat-label {{ color: {TEXT_MUTED}; font-size: 0.65rem; text-transform: uppercase;
               letter-spacing: 1.2px; margin-bottom: 6px; font-weight: 600; }}
.stat-value {{ color: {TEXT_PRIMARY}; font-size: 1.15rem; font-weight: 800; line-height: 1; }}
.stat-sub   {{ color: {TEXT_SECONDARY}; font-size: 0.7rem; margin-top: 5px; }}

/* ── Horizon Strip ── */
.hstrip {{
    background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 16px; display: flex; flex-direction: column; gap: 6px;
}}
.hstrip-head {{ display: flex; justify-content: space-between; align-items: baseline; }}
.hstrip-label {{ color: {TEXT_MUTED}; font-size: 0.68rem; font-weight: 600;
                 text-transform: uppercase; letter-spacing: 1px; }}
.hstrip-price {{ color: {TEXT_PRIMARY}; font-size: 1.2rem; font-weight: 800; }}
.hstrip-ret   {{ font-size: 0.85rem; font-weight: 700; }}
.hstrip-range {{ color: {TEXT_MUTED}; font-size: 0.68rem; }}
.hstrip-conf  {{ height: 4px; border-radius: 4px; background: {BORDER}; margin-top: 4px; overflow: hidden; }}
.hstrip-conf-fill {{ height: 100%; border-radius: 4px; }}

/* ── Evidence Pill ── */
.evpill {{
    display: inline-flex; align-items: center; gap: 5px;
    border-radius: 6px; padding: 5px 10px; font-size: 0.72rem; font-weight: 600;
}}
.evpill-bull {{ background: {GREEN_DIM}; color: {GREEN}; border: 1px solid #065f4640; }}
.evpill-bear {{ background: {RED_DIM}; color: {RED}; border: 1px solid #7f1d1d40; }}
.evpill-neut {{ background: {AMBER_DIM}; color: {AMBER}; border: 1px solid #78350f40; }}

/* ── Regime Chip ── */
.regime-chip {{
    display: inline-flex; align-items: center; gap: 6px;
    border-radius: 20px; padding: 4px 14px; font-size: 0.72rem; font-weight: 700;
}}

/* ── Tooltip ── */
.tip {{
    position: relative; display: inline-block;
    cursor: help; border-bottom: 1px dotted {TEXT_MUTED};
}}
.tip .tiptext {{
    visibility: hidden; opacity: 0;
    background: {BG_ELEVATED}; color: {TEXT_SECONDARY};
    font-size: 0.73rem; line-height: 1.5;
    border: 1px solid {BORDER_ACCENT};
    border-radius: 8px; padding: 10px 14px;
    position: absolute; z-index: 100;
    bottom: 130%; left: 50%; transform: translateX(-50%);
    width: 230px; box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    transition: opacity 0.15s;
    white-space: normal; text-align: left;
}}
.tip:hover .tiptext {{ visibility: visible; opacity: 1; }}

/* ── Inline info icon tooltip (for hero card metrics) ── */
.info-icon {{
    position: relative; display: inline-block;
    cursor: help; margin-left: 4px; vertical-align: middle;
}}
.info-icon .info-tip {{
    visibility: hidden; opacity: 0;
    background: {BG_ELEVATED}; color: {TEXT_SECONDARY};
    font-size: 0.73rem; line-height: 1.5; font-weight: 400;
    border: 1px solid {BORDER_ACCENT};
    border-radius: 8px; padding: 10px 14px;
    position: absolute; z-index: 100;
    bottom: 130%; left: 50%; transform: translateX(-50%);
    width: 250px; box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    transition: opacity 0.15s;
    white-space: normal; text-align: left;
    letter-spacing: 0; text-transform: none;
}}
.info-icon:hover .info-tip {{ visibility: visible; opacity: 1; }}

/* ── Context block ── */
.ctx {{
    background: {BG_SURFACE}; border: 1px solid {BORDER_ACCENT};
    border-left: 3px solid {BLUE};
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.8rem; color: {TEXT_SECONDARY}; line-height: 1.7;
}}
.ctx b {{ color: {TEXT_PRIMARY}; }}

/* ── Trade Rec in Horizon Card ── */
.trade-rec {{
    margin-top: 8px; padding-top: 8px; border-top: 1px solid {BORDER};
    font-size: 0.7rem; line-height: 1.5;
}}
.trade-rec-label {{
    color: {TEXT_MUTED}; font-size: 0.58rem; text-transform: uppercase;
    letter-spacing: 1px; font-weight: 600; margin-bottom: 3px;
}}
.trade-rec-line {{
    font-weight: 700; font-size: 0.72rem; letter-spacing: 0.2px;
}}
.trade-rec-sub {{
    color: {TEXT_MUTED}; font-size: 0.62rem; margin-top: 2px;
}}

/* ── Options leg cards ── */
.leg-buy  {{ border-left: 3px solid {GREEN}; background: {GREEN_DIM}; }}
.leg-sell {{ border-left: 3px solid {RED}; background: {RED_DIM}; }}
.leg-card {{ border-radius: 8px; padding: 12px 14px; margin: 4px 0; }}

/* ── Tooltips ── */
.tooltip-icon {{
    display: inline-block; width: 16px; height: 16px; border-radius: 50%;
    background: {BORDER_ACCENT}; color: {BLUE}; font-size: 0.7rem; font-weight: bold;
    line-height: 16px; text-align: center; cursor: help; margin-left: 4px;
    position: relative;
}}
.tooltip-icon:hover::after {{
    content: attr(data-tip); position: absolute; bottom: 125%; left: 50%; transform: translateX(-50%);
    background: {BG_ELEVATED}; border: 1px solid {BORDER_ACCENT}; color: {TEXT_PRIMARY};
    padding: 8px 12px; border-radius: 6px; font-size: 0.75rem; font-weight: 400; white-space: nowrap;
    z-index: 1000; box-shadow: 0 4px 12px rgba(0,0,0,0.3); pointer-events: none;
}}
.tooltip-icon:hover::before {{
    content: ''; position: absolute; bottom: 115%; left: 50%; transform: translateX(-50%);
    border: 5px solid transparent; border-top: 5px solid {BG_ELEVATED}; z-index: 1000;
}}

/* ── Streamlit overrides ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {BG_SURFACE}; border-radius: 10px; padding: 3px; gap: 2px;
    border: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{ border-radius: 8px; color: {TEXT_MUTED}; padding: 8px 20px; font-weight: 600; font-size: 0.82rem; }}
.stTabs [aria-selected="true"] {{ background: {BORDER_ACCENT} !important; color: {TEXT_PRIMARY} !important; }}
button[kind="primary"] {{ background: linear-gradient(135deg,{BLUE},{PURPLE}) !important; border: none !important; font-weight: 700 !important; }}
hr {{ border-color: {BORDER} !important; margin: 0.8rem 0; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _rc(r):   return GREEN if r >= 0 else RED
def _arr(v):  return "▲" if v >= 0 else "▼"
def _cc(c):
    if c >= 70: return GREEN
    if c >= 55: return AMBER
    return "#f97316"
def _rsi_c(r):
    if r > 70: return RED
    if r < 30: return GREEN
    return AMBER

def tip(label, text):
    return f"<span class='tip'>{label}<span class='tiptext'>{text}</span></span>"


def info_icon(tooltip_text: str) -> str:
    """Creates an info icon with hover tooltip."""
    return f"<span class='tooltip-icon' data-tip='{tooltip_text}'>ℹ</span>"


def metric_with_tooltip(label: str, value: str, tooltip: str, color: str = TEXT_PRIMARY) -> str:
    """Renders a metric label with value and info icon tooltip."""
    return (
        f"<div style='display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid {BORDER}'>"
        f"<div style='display:flex;align-items:center;gap:6px'>"
        f"<span style='color:{TEXT_SECONDARY};font-size:0.75rem;font-weight:500'>{label}</span>"
        f"{info_icon(tooltip)}"
        f"</div>"
        f"<span style='color:{color};font-size:0.85rem;font-weight:700'>{value}</span>"
        f"</div>"
    )


def _trade_oneliner(strat: dict) -> tuple:
    """
    Build a compact one-liner from an options strategy dict.
    Returns (line_text, direction_color, risk_reward_text).
    """
    if not strat or strat.get("strategy") == "N/A":
        return ("No trade rec", TEXT_MUTED, "")

    name = strat["strategy"]
    legs = strat.get("legs", [])
    direction = strat.get("direction", "Neutral")

    # Pick color based on direction
    if "Bull" in direction or "bull" in direction.lower():
        dc = GREEN
    elif "Bear" in direction or "bear" in direction.lower():
        dc = RED
    else:
        dc = AMBER

    # Build the strike description
    if name == "Long Call":
        K = legs[0]["strike"]
        exp = strat["expiry_days"]
        line = f"Buy ${K:.0f} Call · {exp}d"
    elif name == "Long Put":
        K = legs[0]["strike"]
        exp = strat["expiry_days"]
        line = f"Buy ${K:.0f} Put · {exp}d"
    elif name == "Bull Call Spread":
        K1 = legs[0]["strike"]
        K2 = legs[1]["strike"]
        exp = strat["expiry_days"]
        line = f"${K1:.0f}/${K2:.0f} Call Spread · {exp}d"
    elif name == "Bear Put Spread":
        K1 = legs[0]["strike"]
        K2 = legs[1]["strike"]
        exp = strat["expiry_days"]
        line = f"${K1:.0f}/${K2:.0f} Put Spread · {exp}d"
    elif name == "Iron Condor":
        Kps = legs[1]["strike"]
        Kcs = legs[2]["strike"]
        exp = strat["expiry_days"]
        line = f"IC ${Kps:.0f}/${Kcs:.0f} · {exp}d"
    elif name == "Cash-Secured Put":
        K = legs[0]["strike"]
        exp = strat["expiry_days"]
        line = f"Sell ${K:.0f} Put · {exp}d"
    elif name == "Long Straddle":
        K = legs[0]["strike"]
        exp = strat["expiry_days"]
        line = f"${K:.0f} Straddle · {exp}d"
    else:
        line = name

    # Cost / credit
    cost_str = strat.get("estimated_cost", "")
    # Simplify to just the dollar amount
    if "$" in cost_str:
        cost_short = cost_str.split("/")[0].strip()
    else:
        cost_short = cost_str

    return (line, dc, cost_short)

TIPS = {
    "Confidence":  "Model confidence in prediction (30-95%). Calibrated against ensemble accuracy. Regime-adjusted: lower in bear markets.",
    "Dir Acc":     "Directional accuracy: % correct up/down predictions on test data the model never saw. 55%+ is meaningful.",
    "Sharpe":      "Risk-adjusted return. >1.0 = good, >2.0 = excellent. Accounts for drawdowns.",
    "Win Rate":    "Of buy signals, what % actually went up. 55%+ indicates edge.",
    "PF":          "Profit factor = gross profit / gross loss. >1.5 = meaningful, >2.0 = strong edge.",
    "Max DD":      "Largest peak-to-trough loss. Your worst case if you followed every signal.",
    "RSI":         "Momentum 0-100. >70 overbought (reversal risk). <30 oversold (bounce potential).",
    "MACD":        "Trend momentum. Bullish = strengthening, Bearish = weakening. Divergence shows weakness.",
    "Beta":        "Volatility vs S&P 500. 1.5 = 50% more volatile. Higher beta = larger swings.",
    "IV":          "Implied volatility: 21d historical vol × 1.2. Higher = options more expensive.",
    "Agreement":   "Model consensus: % of base models agreeing on direction. 100% = unanimous.",
    "WF":          "Walk-forward backtest: trained on 2yr windows, tested on next 3mo, repeated. Tests all market regimes.",
    "Regime":      "Current market state (Bull/Bear/Sideways). Affects confidence levels.",
    "RR":          "Risk/Reward = (target - entry) / (entry - stop). 3:1 or higher is ideal.",
}

def fmt_mc(mc):
    if not mc: return "N/A"
    if mc > 1e12: return f"${mc/1e12:.2f}T"
    if mc > 1e9:  return f"${mc/1e9:.1f}B"
    return f"${mc/1e6:.0f}M"


# ── Cached data fetchers (avoid re-downloading on every rerun) ───────────────

@st.cache_data(ttl=900, show_spinner=False)   # 15-min cache
def _cached_fetch_data(symbol: str, period: str):
    return fetch_stock_data(symbol, period=period)

@st.cache_data(ttl=900, show_spinner=False)
def _cached_fetch_info(symbol: str):
    return fetch_stock_info(symbol)

@st.cache_data(ttl=900, show_spinner=False)
def _cached_fetch_market(period: str, sector: str):
    return fetch_market_context(period=period, sector=sector)

@st.cache_data(ttl=900, show_spinner=False)
def _cached_fetch_fundamentals(symbol: str):
    return fetch_fundamentals(symbol)

@st.cache_data(ttl=900, show_spinner=False)
def _cached_fetch_earnings(symbol: str):
    return fetch_earnings_data(symbol)

@st.cache_data(ttl=900, show_spinner=False)
def _cached_fetch_options(symbol: str, current_price: float):
    return fetch_options_data(symbol, current_price)

def rsi_series(close):
    d = close.diff()
    g = d.clip(lower=0).ewm(com=13, min_periods=14).mean()
    l = (-d.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:6px 0 18px'>
        <div style='font-size:0.9rem;letter-spacing:3px;color:{TEXT_MUTED};font-weight:600'>◆</div>
        <div style='font-size:1.15rem;font-weight:900;color:{TEXT_PRIMARY};letter-spacing:1px;margin-top:4px'>SignalEdge</div>
        <div style='font-size:0.68rem;color:{TEXT_MUTED};margin-top:2px;letter-spacing:0.5px'>ML-Powered Stock Signals</div>
    </div>
    """, unsafe_allow_html=True)

    symbol = st.text_input(
        "Ticker", value="AAPL", placeholder="AAPL, NVDA, TSLA…",
        help="Any US stock ticker"
    ).upper().strip()

    data_period = st.selectbox("Training History", ["3y","5y","7y","10y"], index=3,
                               help="More history = more diverse market regimes. 10y recommended for best accuracy.")

    c1, c2 = st.columns(2)
    with c1: run_bt  = st.checkbox("Backtest", value=False, help="Walk-forward backtest (+1-2 min)")
    with c2: use_ctx = st.checkbox("SPY + VIX", value=True, help="Market context features")

    c3, c4 = st.columns(2)
    with c3: use_shap   = st.checkbox("SHAP", value=False, disabled=not HAS_SHAP,
                                       help="SHAP feature selection (+30s)" + ("" if HAS_SHAP else " — install shap"))
    with c4: use_optuna = st.checkbox("Optuna", value=False, disabled=not HAS_OPTUNA,
                                       help="Hyperparameter tuning (+2-3 min)" + ("" if HAS_OPTUNA else " — install optuna"))

    analyze_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    st.divider()

    # Engine info
    parts = ["XGB×5"]
    if HAS_LGB: parts.append("LGB×3")
    parts += ["RF×2", "CLS×4", "RAdj×2", "Ridge L2"]
    st.markdown(
        f"<div style='background:{BG_SURFACE};border:1px solid {BORDER};border-radius:8px;"
        f"padding:10px 12px;font-size:0.7rem'>"
        f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px;font-weight:600'>Engine</div>"
        f"<div style='color:{BLUE};font-weight:700'>{' · '.join(parts)}</div>"
        f"</div>", unsafe_allow_html=True)

    st.caption("For educational purposes only. Not financial advice.")

    # ── Prediction Track Record ─────────────────────────────────────────────
    try:
        track = get_track_record()
        if track["scored_predictions"] > 0:
            st.markdown(
                f"<div style='background:{BG_SURFACE};border:1px solid {BORDER};border-radius:8px;"
                f"padding:10px 12px;font-size:0.7rem;margin-top:8px'>"
                f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px;font-weight:600'>Track Record</div>"
                f"<div style='color:{GREEN if track['direction_accuracy'] >= 55 else AMBER};font-weight:700;font-size:0.85rem'>"
                f"{track['direction_accuracy']}% Directional Accuracy</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.6rem'>"
                f"{track['scored_predictions']} scored / {track['total_predictions']} total predictions</div>"
                f"</div>", unsafe_allow_html=True)
        elif track["total_predictions"] > 0:
            st.markdown(
                f"<div style='background:{BG_SURFACE};border:1px solid {BORDER};border-radius:8px;"
                f"padding:10px 12px;font-size:0.7rem;margin-top:8px'>"
                f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px;font-weight:600'>Track Record</div>"
                f"<div style='color:{TEXT_SECONDARY};font-size:0.7rem'>"
                f"{track['total_predictions']} predictions logged · scoring in 7–30 days</div>"
                f"</div>", unsafe_allow_html=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results = None


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS — runs on button click only
# ══════════════════════════════════════════════════════════════════════════════

if analyze_btn and symbol:
    st.session_state.results = None

    status = st.status(f"Analyzing {symbol}…", expanded=True)
    with status:
        st.write("Fetching price data…")
        try:
            df   = _cached_fetch_data(symbol, period=data_period)
            info = _cached_fetch_info(symbol)
        except Exception as e:
            st.error(f"Could not load {symbol}: {e}")
            st.stop()

        market_ctx = None
        if use_ctx:
            st.write("Fetching SPY + VIX…")
            try:
                market_ctx = _cached_fetch_market(period=data_period, sector=info.get("sector"))
            except Exception:
                pass

        st.write("Fetching fundamentals, earnings & options data…")
        try:
            fundamentals = _cached_fetch_fundamentals(symbol)
        except Exception:
            fundamentals = {}

        try:
            earnings = _cached_fetch_earnings(symbol)
        except Exception:
            earnings = {}

        try:
            current_price = float(df["Close"].iloc[-1])
            options_mkt = _cached_fetch_options(symbol, current_price)
        except Exception:
            options_mkt = {}

        n_mdls = 16 if HAS_LGB else 13
        st.write(f"Training {n_mdls}-model stacked ensemble (v5)…")
        prog = st.progress(0)
        def upd(f, msg): prog.progress(min(float(f), 1.0))

        try:
            predictor   = StockPredictor(symbol)
            predictor.train(df, market_ctx=market_ctx, fundamentals=fundamentals,
                            earnings_data=earnings, options_data=options_mkt,
                            use_shap=use_shap, use_optuna=use_optuna,
                            progress_callback=upd)
            predictions = predictor.predict(df, market_ctx=market_ctx)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

        st.write("Generating signals…")
        analysis = generate_analysis(df, predictions, info)
        options  = generate_options_report(predictions, df, focus_horizon="1 Month")

        spy_s = market_ctx.get("spy") if market_ctx else None
        vix_s = market_ctx.get("vix") if market_ctx else None
        regime_info = detect_regime(df, spy_s, vix_s)

        bt_results = None
        if run_bt:
            st.write("Running walk-forward backtest…")
            bt_prog = st.progress(0)
            def bt_cb(f, msg): bt_prog.progress(min(float(f), 1.0))
            try:
                bt_results = run_backtest(df, market_ctx=market_ctx, fundamentals=fundamentals,
                                         earnings_data=earnings, options_data=options_mkt,
                                         progress_callback=bt_cb)
            except Exception as e:
                st.warning(f"Backtest error: {e}")

    status.update(label=f"Analysis complete — {symbol}", state="complete", expanded=False)

    st.session_state.results = dict(
        symbol=symbol, df=df, info=info,
        predictions=predictions, analysis=analysis,
        options=options, predictor=predictor,
        bt_results=bt_results, regime_info=regime_info,
        market_ctx=market_ctx,
    )

    # Log prediction for future accuracy tracking
    try:
        log_prediction(symbol, predictions)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.results is None:
    st.markdown(f"""
    <div style='text-align:center;padding:80px 20px 40px'>
        <div style='font-size:0.75rem;letter-spacing:4px;color:{TEXT_MUTED};font-weight:600;text-transform:uppercase;margin-bottom:16px'>◆ SignalEdge</div>
        <h1 style='color:{TEXT_PRIMARY};font-size:2.4rem;font-weight:900;margin:0 0 8px;line-height:1.2'>
            ML-Powered Stock Signals</h1>
        <p style='color:{TEXT_SECONDARY};font-size:1rem;max-width:500px;margin:0 auto;line-height:1.6'>
            Stacked ensemble predictions across 4 time horizons.<br>
            Walk-forward backtested. Regime-aware. Options-ready.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    items = [
        ("⬆", "Price Targets", "1W / 1M / 1Q / 1Y with confidence intervals"),
        ("◎", "Confidence", "12-model ensemble agreement + calibrated accuracy"),
        ("⟲", "Backtested", "Walk-forward across bull, bear & sideways markets"),
        ("⚡", "Options", "Strategy recommendations with estimated premiums"),
    ]
    for col, (ic, ttl, dsc) in zip([c1,c2,c3,c4], items):
        with col:
            st.markdown(
                f"<div class='card' style='text-align:center;min-height:120px'>"
                f"<div style='font-size:1.4rem;color:{BLUE};margin-bottom:8px'>{ic}</div>"
                f"<div style='color:{TEXT_PRIMARY};font-weight:700;font-size:0.85rem;margin-bottom:4px'>{ttl}</div>"
                f"<div style='color:{TEXT_SECONDARY};font-size:0.75rem;line-height:1.5'>{dsc}</div>"
                f"</div>", unsafe_allow_html=True)

    # ── Dashboard: Track Record & Recent Predictions ──────────────────────────
    st.markdown(f"<div style='margin-top:60px;padding-top:40px;border-top:1px solid {BORDER}'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='text-align:center;margin-bottom:32px'>
        <div style='color:{TEXT_SECONDARY};font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px'>Track Record</div>
        <h2 style='color:{TEXT_PRIMARY};font-size:1.8rem;font-weight:700;margin:0'>Prediction Performance</h2>
    </div>
    """, unsafe_allow_html=True)

    try:
        track = get_track_record()

        # Summary cards
        scol1, scol2, scol3, scol4 = st.columns(4)

        with scol1:
            st.markdown(
                f"<div class='card-sm' style='text-align:center'>"
                f"<div style='color:{TEXT_MUTED};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Total Predictions</div>"
                f"<div style='color:{TEXT_PRIMARY};font-size:1.6rem;font-weight:800'>{track['total_predictions']}</div>"
                f"</div>", unsafe_allow_html=True)

        with scol2:
            dir_acc = track['direction_accuracy']
            acc_color = GREEN if dir_acc >= 55 else AMBER if dir_acc >= 50 else RED
            st.markdown(
                f"<div class='card-sm' style='text-align:center'>"
                f"<div style='color:{TEXT_MUTED};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Direction Accuracy</div>"
                f"<div style='color:{acc_color};font-size:1.6rem;font-weight:800'>{dir_acc:.1f}%</div>"
                f"</div>", unsafe_allow_html=True)

        with scol3:
            scored = track['scored_predictions']
            st.markdown(
                f"<div class='card-sm' style='text-align:center'>"
                f"<div style='color:{TEXT_MUTED};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Scored</div>"
                f"<div style='color:{TEXT_SECONDARY};font-size:1.6rem;font-weight:800'>{scored}</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.65rem;margin-top:4px'>of {track['total_predictions']}</div>"
                f"</div>", unsafe_allow_html=True)

        with scol4:
            pending = track['total_predictions'] - scored
            st.markdown(
                f"<div class='card-sm' style='text-align:center'>"
                f"<div style='color:{TEXT_MUTED};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Pending</div>"
                f"<div style='color:{TEXT_SECONDARY};font-size:1.6rem;font-weight:800'>{pending}</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.65rem;margin-top:4px'>being scored</div>"
                f"</div>", unsafe_allow_html=True)

        # Recent predictions
        if track.get('recent'):
            st.markdown(f"<div style='margin-top:32px'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='color:{TEXT_SECONDARY};font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px'>Recent Predictions</div>",
                unsafe_allow_html=True)

            recent_df = pd.DataFrame(track['recent'][:5])
            if not recent_df.empty:
                recent_df = recent_df[['symbol', 'date', 'horizon', 'predicted_return', 'actual_return', 'direction_correct']]
                recent_df.columns = ['Symbol', 'Date', 'Horizon', 'Predicted %', 'Actual %', 'Correct']
                recent_df['Predicted %'] = recent_df['Predicted %'].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")
                recent_df['Actual %'] = recent_df['Actual %'].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "pending")
                recent_df['Correct'] = recent_df['Correct'].apply(lambda x: "✓" if x is True else "✗" if x is False else "—")

                st.dataframe(recent_df, use_container_width=True, hide_index=True)

    except Exception:
        pass

    st.markdown(
        f"<p style='text-align:center;color:{TEXT_MUTED};margin-top:48px;font-size:0.82rem'>"
        f"Enter a ticker and click <b style='color:{TEXT_SECONDARY}'>Run Analysis</b></p>",
        unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS — from session state
# ══════════════════════════════════════════════════════════════════════════════

R           = st.session_state.results
df          = R["df"]
info        = R["info"]
predictions = R["predictions"]
analysis    = R["analysis"]
options     = R["options"]
predictor   = R["predictor"]
bt_results  = R["bt_results"]
symbol      = R["symbol"]
regime_info = R.get("regime_info", {})
market_ctx  = R.get("market_ctx")

cur   = float(df["Close"].iloc[-1])
prev  = float(df["Close"].iloc[-2])
d_chg = (cur - prev) / prev * 100


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — DECISION (5-second scan)
# ══════════════════════════════════════════════════════════════════════════════

rec    = analysis["recommendation"]
color  = analysis["color"]
conf1m = predictions["1 Month"]["confidence"]

# ── Stock identity bar ────────────────────────────────────────────────────────
id_col, price_col, stat1, stat2, stat3, stat4 = st.columns([2.5, 1.5, 1, 1, 1, 1])

with id_col:
    rlabel = regime_info.get("label", "Sideways") if regime_info else "N/A"
    rcolor = {"Bull":GREEN,"Bear":RED,"Sideways":AMBER}.get(rlabel,TEXT_MUTED)
    rbg    = {"Bull":GREEN_DIM,"Bear":RED_DIM,"Sideways":AMBER_DIM}.get(rlabel,BG_CARD)
    ricon  = {"Bull":"▲","Bear":"▼","Sideways":"→"}.get(rlabel,"")
    st.markdown(
        f"<div style='padding:4px 0'>"
        f"<span style='color:{TEXT_SECONDARY};font-size:0.75rem'>{info.get('sector','')} · {info.get('industry','')}</span><br>"
        f"<span style='color:{TEXT_PRIMARY};font-size:1.4rem;font-weight:900'>{info.get('name', symbol)}</span>"
        f"  <code style='font-size:0.78rem;background:{BG_ELEVATED};padding:2px 8px;border-radius:4px;color:{TEXT_SECONDARY}'>{symbol}</code>"
        f"  <span class='regime-chip' style='position:relative;background:{rbg};color:{rcolor};border:1px solid {rcolor}30;padding:4px 8px;border-radius:4px;display:inline-block;cursor:help'>{ricon} {rlabel}{info_icon(TIPS['Regime'])}</span>"
        f"</div>", unsafe_allow_html=True)

with price_col:
    dc = _rc(d_chg)
    st.markdown(
        f"<div style='text-align:right;padding:4px 0'>"
        f"<div style='font-size:1.8rem;font-weight:900;color:{TEXT_PRIMARY}'>${cur:,.2f}</div>"
        f"<div style='color:{dc};font-size:0.85rem;font-weight:700'>{_arr(d_chg)} {abs(d_chg):.2f}% today</div>"
        f"</div>", unsafe_allow_html=True)

for col_s, lbl, val in [
    (stat1, "Mkt Cap",   fmt_mc(info.get("market_cap"))),
    (stat2, "P/E",       f"{info.get('pe_ratio'):.1f}" if info.get('pe_ratio') else "—"),
    (stat3, "Beta",      f"{info.get('beta'):.2f}" if info.get('beta') else "—"),
    (stat4, "52W Range", f"${info.get('52w_low',0):.0f}–${info.get('52w_high',0):.0f}" if info.get('52w_low') else "—"),
]:
    with col_s:
        st.markdown(
            f"<div class='stat'>"
            f"<div class='stat-label'>{lbl}</div>"
            f"<div class='stat-value' style='font-size:0.95rem'>{val}</div>"
            f"</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Hero signal + horizon strip ──────────────────────────────────────────────
hero_col, h1, h2, h3, h4 = st.columns([1.6, 1, 1, 1, 1])

with hero_col:
    glow = GREEN if "Buy" in rec else RED if "Sell" in rec else AMBER
    st.markdown(
        f"<div class='hero-signal' style='border:1.5px solid {color}30;--glow:{glow}'>"
        f"<div style='position:relative;z-index:1'>"
        f"<div style='color:{color};font-size:2.4rem;font-weight:900;letter-spacing:2px'>{rec.upper()}</div>"
        f"<div style='color:{TEXT_MUTED};font-size:0.7rem;margin:6px 0 16px;text-transform:uppercase;letter-spacing:2px;font-weight:600'>1-Month ML Signal</div>"
        f"<div style='display:flex;justify-content:center;gap:28px'>"
        f"<div>"
        f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1px;font-weight:600'>Confidence {info_icon(TIPS['Confidence'])}</div>"
        f"<div style='color:{_cc(conf1m)};font-size:2.2rem;font-weight:900;line-height:1.2'>{conf1m:.0f}<span style='font-size:0.9rem'>%</span></div>"
        f"</div>"
        f"<div style='width:1px;background:{BORDER}'></div>"
        f"<div>"
        f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1px;font-weight:600'>Agreement {info_icon(TIPS['Agreement'])}</div>"
        f"<div style='color:{TEXT_PRIMARY};font-size:2.2rem;font-weight:900;line-height:1.2'>{predictions['1 Month']['ensemble_agreement']:.0f}<span style='font-size:0.9rem'>%</span></div>"
        f"</div>"
        f"</div>"
        f"<div style='color:{TEXT_MUTED};font-size:0.68rem;margin-top:12px'>Validated accuracy: {predictions['1 Month']['val_dir_accuracy']:.1f}% on held-out data</div>"
        f"</div></div>",
        unsafe_allow_html=True)

for col_h, h in zip([h1,h2,h3,h4], ["1 Week","1 Month","1 Quarter","1 Year"]):
    if h not in predictions:
        with col_h:
            st.markdown(f"<div class='hstrip'><div style='color:{TEXT_MUTED};font-size:0.75rem;padding:20px 0;text-align:center'>Not enough data<br>for {h}</div></div>", unsafe_allow_html=True)
        continue
    p   = predictions[h]
    ret = p["predicted_return"]
    rc_ = _rc(ret)
    cc_ = _cc(p["confidence"])
    # Get trade rec for this horizon
    strat_h = options.get(h, {})
    tr_line, tr_color, tr_cost = _trade_oneliner(strat_h)
    with col_h:
        st.markdown(
            f"<div class='hstrip'>"
            f"<div class='hstrip-head'>"
            f"<span class='hstrip-label'>{h}</span>"
            f"<span class='hstrip-ret' style='color:{rc_}'>{_arr(ret)} {abs(ret*100):.1f}%</span>"
            f"</div>"
            f"<div class='hstrip-price'>${p['predicted_price']:,.2f}</div>"
            f"<div class='hstrip-range'>${p['interval_low']:,.2f} — ${p['interval_high']:,.2f}</div>"
            f"<div class='hstrip-conf'><div class='hstrip-conf-fill' style='width:{p['confidence']:.0f}%;background:{cc_}'></div></div>"
            f"<div style='color:{cc_};font-size:0.68rem;font-weight:600;margin-top:2px'>{p['confidence']:.0f}% conf · {p['val_dir_accuracy']:.1f}% acc</div>"
            f"<div class='trade-rec'>"
            f"<div class='trade-rec-label'>Top Trade</div>"
            f"<div class='trade-rec-line' style='color:{tr_color}'>{tr_line}</div>"
            f"<div class='trade-rec-sub'>{tr_cost}</div>"
            f"</div>"
            f"</div>", unsafe_allow_html=True)
        # Expandable trade details
        with st.expander(f"Trade details", expanded=False):
            if strat_h and strat_h.get("strategy") != "N/A" and "error" not in strat_h:
                st.markdown(f"<div style='color:{tr_color};font-weight:800;font-size:0.95rem;margin-bottom:8px'>"
                            f"{strat_h.get('emoji','')} {strat_h['strategy']}</div>", unsafe_allow_html=True)
                # Legs
                for leg in strat_h.get("legs", []):
                    leg_cls = "leg-buy" if leg["action"] == "BUY" else "leg-sell"
                    st.markdown(
                        f"<div class='leg-card {leg_cls}'>"
                        f"<span style='font-weight:700;font-size:0.8rem'>{leg['action']}</span> "
                        f"<span style='color:{TEXT_PRIMARY};font-weight:600'>{leg['type']} ${leg['strike']:.0f}</span> "
                        f"<span style='color:{TEXT_MUTED};font-size:0.75rem'>exp {leg['expiry']}</span>"
                        f"<span style='float:right;color:{TEXT_SECONDARY};font-weight:600'>${leg['premium']:.2f}</span>"
                        f"</div>", unsafe_allow_html=True)
                # Key metrics
                st.markdown(
                    f"<div style='margin-top:10px;font-size:0.78rem;line-height:1.8'>"
                    f"<div class='mrow'><span class='mrow-label'>Est. Cost</span><span class='mrow-value'>{strat_h['estimated_cost']}</span></div>"
                    f"<div class='mrow'><span class='mrow-label'>Max Profit</span><span class='mrow-value' style='color:{GREEN}'>{strat_h['max_profit']}</span></div>"
                    f"<div class='mrow'><span class='mrow-label'>Max Loss</span><span class='mrow-value' style='color:{RED}'>{strat_h['max_loss']}</span></div>"
                    f"<div class='mrow'><span class='mrow-label'>Breakeven</span><span class='mrow-value'>{strat_h['breakeven']}</span></div>"
                    f"<div class='mrow'><span class='mrow-label'>IV Used</span><span class='mrow-value'>{strat_h['iv_used']}</span></div>"
                    f"</div>", unsafe_allow_html=True)
                # Rationale
                st.markdown(f"<div style='margin-top:8px;padding:8px 10px;background:{BG_SURFACE};"
                            f"border-radius:6px;font-size:0.72rem;color:{TEXT_SECONDARY};line-height:1.6'>"
                            f"{strat_h['rationale']}</div>", unsafe_allow_html=True)
                # Risk note
                st.markdown(f"<div style='margin-top:4px;font-size:0.65rem;color:{AMBER}'>"
                            f"⚠ {strat_h.get('risk_note','')}</div>", unsafe_allow_html=True)
                # Estimated prices disclaimer
                st.markdown(f"<div style='margin-top:6px;font-size:0.6rem;color:{TEXT_MUTED};font-style:italic'>"
                            f"Prices are Black-Scholes estimates from historical vol. Verify with your broker before trading.</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.78rem'>No strategy available for this horizon.</div>",
                            unsafe_allow_html=True)


# ── Why This Recommendation? ──────────────────────────────────────────────────
with st.expander("📊 Why This Recommendation?", expanded=False):
    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown(f"<div style='color:{TEXT_SECONDARY};font-size:0.75rem;font-weight:600;text-transform:uppercase;margin-bottom:12px'>Model Confidence Breakdown</div>", unsafe_allow_html=True)

        # Show regime impact
        if regime_info:
            rlbl = regime_info.get("label", "Sideways")
            rsc = regime_info.get("score_norm", 0.5)
            st.markdown(
                f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;padding:12px'>"
                f"<div style='color:{TEXT_SECONDARY};font-size:0.7rem;margin-bottom:6px'>🔄 Current Regime</div>"
                f"<div style='color:{TEXT_PRIMARY};font-weight:700;margin-bottom:4px'>{rlbl} Market</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.7rem'>Confidence adjusted {'down' if rlbl!='Bull' else 'normally'} for {rlbl} regime.</div>"
                f"</div>", unsafe_allow_html=True)

        # Prediction confidence
        p1m = predictions.get("1 Month", {})
        ensemble_agree = p1m.get('ensemble_agreement', 0)
        hist_acc = p1m.get('val_dir_accuracy', 0)
        ensemble_agree_color = _cc(ensemble_agree) if ensemble_agree else TEXT_SECONDARY
        st.markdown(
            f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;padding:12px;margin-top:8px'>"
            f"<div style='color:{TEXT_SECONDARY};font-size:0.7rem;margin-bottom:6px'>🎯 Prediction Factors</div>"
            f"<div style='display:flex;justify-content:space-between;margin:4px 0;font-size:0.75rem'>"
            f"<span>Ensemble Agreement:</span><span style='color:{ensemble_agree_color};font-weight:700'>{ensemble_agree:.0f}%</span>"
            f"</div>"
            f"<div style='display:flex;justify-content:space-between;margin:4px 0;font-size:0.75rem'>"
            f"<span>Historical Accuracy:</span><span style='color:{GREEN};font-weight:700'>{hist_acc:.1f}%</span>"
            f"</div>"
            f"<div style='display:flex;justify-content:space-between;margin:4px 0;font-size:0.75rem'>"
            f"<span>Final Confidence:</span><span style='color:{_cc(conf1m)};font-weight:700'>{conf1m:.0f}%</span>"
            f"</div>"
            f"</div>", unsafe_allow_html=True)

    with rec_col2:
        st.markdown(f"<div style='color:{TEXT_SECONDARY};font-size:0.75rem;font-weight:600;text-transform:uppercase;margin-bottom:12px'>Signal Strength</div>", unsafe_allow_html=True)

        # Technical signals agreement
        bull_n = sum(1 for s in analysis["signals"] if s[1]=="Bullish")
        bear_n = sum(1 for s in analysis["signals"] if s[1]=="Bearish")
        total  = len(analysis["signals"])

        st.markdown(
            f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;padding:12px'>"
            f"<div style='color:{TEXT_SECONDARY};font-size:0.7rem;margin-bottom:6px'>📈 Technical Consensus</div>"
            f"<div style='display:flex;gap:8px;margin-bottom:8px'>"
            f"<div style='flex:1;text-align:center'>"
            f"<div style='color:{GREEN};font-weight:700;font-size:0.95rem'>{bull_n}</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.65rem'>Bullish</div>"
            f"</div>"
            f"<div style='flex:1;text-align:center'>"
            f"<div style='color:{AMBER};font-weight:700;font-size:0.95rem'>{total-bull_n-bear_n}</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.65rem'>Neutral</div>"
            f"</div>"
            f"<div style='flex:1;text-align:center'>"
            f"<div style='color:{RED};font-weight:700;font-size:0.95rem'>{bear_n}</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.65rem'>Bearish</div>"
            f"</div>"
            f"</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.7rem;line-height:1.5'>ML model weights these signals differently than equal voting.</div>"
            f"</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — EVIDENCE STRIP (30-second validation)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<hr>", unsafe_allow_html=True)

ev1, ev2, ev3 = st.columns([1.2, 1.6, 1.2])

# ── Signal gauge ─────────────────────────────────────────────────────────────
with ev1:
    bull_n = sum(1 for s in analysis["signals"] if s[1]=="Bullish")
    bear_n = sum(1 for s in analysis["signals"] if s[1]=="Bearish")
    total  = len(analysis["signals"])
    neut_n = total - bull_n - bear_n
    bull_pct = int(bull_n / total * 100) if total > 0 else 0
    bear_pct = int(bear_n / total * 100) if total > 0 else 0
    neut_pct = 100 - bull_pct - bear_pct

    st.markdown(
        f"<div class='card'>"
        f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:10px'>Signal Consensus</div>"
        f"<div style='display:flex;gap:12px;margin-bottom:12px'>"
        f"<div style='flex:1;text-align:center'><div style='color:{GREEN};font-size:1.5rem;font-weight:900'>{bull_n}</div><div style='color:{TEXT_MUTED};font-size:0.65rem'>BULLISH</div></div>"
        f"<div style='flex:1;text-align:center'><div style='color:{AMBER};font-size:1.5rem;font-weight:900'>{neut_n}</div><div style='color:{TEXT_MUTED};font-size:0.65rem'>NEUTRAL</div></div>"
        f"<div style='flex:1;text-align:center'><div style='color:{RED};font-size:1.5rem;font-weight:900'>{bear_n}</div><div style='color:{TEXT_MUTED};font-size:0.65rem'>BEARISH</div></div>"
        f"</div>"
        f"<div style='background:{BORDER};border-radius:20px;height:8px;overflow:hidden;display:flex'>"
        f"<div style='width:{bull_pct}%;background:{GREEN}'></div>"
        f"<div style='width:{neut_pct}%;background:{AMBER}'></div>"
        f"<div style='width:{bear_pct}%;background:{RED}'></div>"
        f"</div></div>",
        unsafe_allow_html=True)

# ── Key signals ──────────────────────────────────────────────────────────────
with ev2:
    pills_html = ""
    for name, direction, desc in analysis["signals"]:
        cls = {"Bullish":"evpill-bull","Bearish":"evpill-bear"}.get(direction,"evpill-neut")
        icon = {"Bullish":"▲","Bearish":"▼"}.get(direction,"→")
        pills_html += f"<span class='evpill {cls}'>{icon} {name}</span>  "

    st.markdown(
        f"<div class='card'>"
        f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:10px'>Key Drivers</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:6px'>{pills_html}</div>"
        f"</div>", unsafe_allow_html=True)

# ── Trade setup ──────────────────────────────────────────────────────────────
with ev3:
    ez = analysis["entry_zone"]
    rr = ez['risk_reward']
    rr_color = GREEN if rr and rr >= 2 else AMBER if rr and rr >= 1 else RED if rr else TEXT_MUTED
    st.markdown(
        f"<div class='card'>"
        f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:10px'>Trade Setup</div>"
        f"<div class='mrow'><span class='mrow-label'>Entry</span><span class='mrow-value'>${ez['entry_low']:,.2f} – ${ez['entry_high']:,.2f}</span></div>"
        f"<div class='mrow'><span class='mrow-label'>Stop Loss</span><span class='mrow-value' style='color:{RED}'>${ez['stop_loss']:,.2f}</span></div>"
        f"<div class='mrow'><span class='mrow-label'>Target</span><span class='mrow-value' style='color:{GREEN}'>${ez['first_target']:,.2f}</span></div>"
        f"<div class='mrow'><span class='mrow-label'>Risk/Reward</span><span class='mrow-value' style='color:{rr_color}'>{rr:.1f}x</span></div>"
        f"</div>" if rr else
        f"<div class='card'>"
        f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:10px'>Trade Setup</div>"
        f"<div class='mrow'><span class='mrow-label'>Entry</span><span class='mrow-value'>${ez['entry_low']:,.2f} – ${ez['entry_high']:,.2f}</span></div>"
        f"<div class='mrow'><span class='mrow-label'>Stop Loss</span><span class='mrow-value' style='color:{RED}'>${ez['stop_loss']:,.2f}</span></div>"
        f"<div class='mrow'><span class='mrow-label'>Target</span><span class='mrow-value' style='color:{GREEN}'>${ez['first_target']:,.2f}</span></div>"
        f"</div>",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — DEEP DIVE TABS (2-minute analysis)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<hr>", unsafe_allow_html=True)

tab_chart, tab_analysis, tab_opts, tab_bt, tab_model = st.tabs([
    "📊  Chart",
    "🔍  Analysis",
    "⚡  Options",
    "⟲  Backtest",
    "◎  Model",
])


# ── TAB: CHART ───────────────────────────────────────────────────────────────

with tab_chart:
    period_map = {"3M":63,"6M":126,"1Y":252,"2Y":504,"All":len(df)}
    cp       = st.radio("Period", list(period_map.keys()), index=2, horizontal=True)
    chart_df = df.tail(period_map[cp]).copy()

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.58, 0.21, 0.21],
        vertical_spacing=0.025,
        subplot_titles=("", "", ""),
    )

    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"],  close=chart_df["Close"],
        name="Price",
        increasing=dict(line=dict(color=GREEN), fillcolor="rgba(16,185,129,0.2)"),
        decreasing=dict(line=dict(color=RED), fillcolor="rgba(239,68,68,0.2)"),
    ), row=1, col=1)

    for ma, mc, mn in [(20,AMBER,"MA20"),(50,BLUE,"MA50"),(200,PURPLE,"MA200")]:
        if len(chart_df) >= ma:
            fig.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df["Close"].rolling(ma).mean(),
                name=mn, line=dict(color=mc, width=1.2),
            ), row=1, col=1)

    target_styles = {"1 Week":("#67e8f9","dot"),"1 Month":("#fde68a","dash"),"1 Quarter":("#fca5a5","dashdot"),"1 Year":("#c4b5fd","longdash")}
    for h, (hc, hd) in target_styles.items():
        if h not in predictions:
            continue
        tp = predictions[h]["predicted_price"]
        if abs(tp - cur) / cur < 0.5:
            fig.add_hline(y=tp, line=dict(color=hc, width=1, dash=hd),
                          annotation_text=f" {h} ${tp:,.0f}",
                          annotation_font=dict(color=hc, size=9), row=1, col=1)

    vc = [GREEN if c >= o else RED for c,o in zip(chart_df["Close"], chart_df["Open"])]
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"],
                         marker_color=vc, name="Vol", showlegend=False, marker_opacity=0.6), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Volume"].rolling(20).mean(),
                             name="Vol MA20", line=dict(color=AMBER,width=1), showlegend=False), row=2, col=1)

    rsi_s = rsi_series(chart_df["Close"])
    fig.add_trace(go.Scatter(x=chart_df.index, y=rsi_s, name="RSI",
                             line=dict(color=PURPLE,width=1.6)), row=3, col=1)
    for lvl, ann, lc in [(70,"Overbought",RED),(50,"","#334155"),(30,"Oversold",GREEN)]:
        fig.add_hline(y=lvl, line=dict(color=lc,width=1,dash="dash"),
                      annotation_text=f" {ann}" if ann else "",
                      annotation_font=dict(color=lc,size=9), row=3, col=1)

    fig.update_layout(
        height=620, template="plotly_dark",
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_SURFACE,
        font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=10,r=10,t=30,b=10),
        xaxis_rangeslider_visible=False, hovermode="x unified",
    )
    fig.update_yaxes(gridcolor="#0c1525", zerolinecolor="#0c1525")
    fig.update_xaxes(gridcolor="#0c1525", showspikes=True, spikecolor="#1e3050", spikethickness=1)
    st.plotly_chart(fig, use_container_width=True)

    # Tech indicators row
    tech = analysis["technicals"]
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    for col_m, label, val, color, tkey in [
        (m1,"RSI",     str(tech["rsi"]),       _rsi_c(tech["rsi"]),    "RSI"),
        (m2,"Trend",   tech["trend"],          GREEN if "Up" in tech["trend"] else RED if "Down" in tech["trend"] else AMBER, None),
        (m3,"MACD",    tech["macd_cross"],     GREEN if tech["macd_cross"]=="Bullish" else RED, "MACD"),
        (m4,"BB Pos.", tech["bb_pos"],         TEXT_PRIMARY, None),
        (m5,"Volume",  f"{tech['vol_ratio']}x", GREEN if tech["vol_ratio"]>1.2 else TEXT_MUTED, None),
        (m6,"Beta",    f"{info.get('beta'):.2f}" if info.get('beta') else "—", TEXT_SECONDARY, "Beta"),
    ]:
        with col_m:
            lab = tip(label, TIPS[tkey]) if tkey else label
            st.markdown(
                f"<div class='stat'>"
                f"<div class='stat-label'>{lab}</div>"
                f"<div class='stat-value' style='color:{color};font-size:0.95rem'>{val}</div>"
                f"</div>", unsafe_allow_html=True)


# ── TAB: ANALYSIS ────────────────────────────────────────────────────────────

with tab_analysis:
    an1, an2 = st.columns([1, 1])

    with an1:
        st.markdown(
            f"<div class='card' style='color:{TEXT_SECONDARY};font-size:0.83rem;line-height:1.85'>"
            f"{analysis['narrative']}</div>", unsafe_allow_html=True)

    with an2:
        n_l1 = sum(1 for m in predictor.l1_members.get("1 Month", []))
        st.markdown(
            f"<div class='card'>"
            f"<div style='color:{BLUE};font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px'>Methodology</div>"
            f"<div style='color:{TEXT_SECONDARY};font-size:0.8rem;line-height:1.7'>"
            f"<b style='color:{TEXT_PRIMARY}'>Level 1:</b> {n_l1} diverse models (XGBoost, LightGBM, RandomForest, XGBClassifier) "
            f"trained with feature bagging + exponential sample weighting. "
            f"<b style='color:{TEXT_PRIMARY}'>Level 2:</b> Ridge meta-learner on out-of-fold predictions with isotonic calibration."
            f"<br><br><b style='color:{TEXT_PRIMARY}'>Features:</b> 75+ engineered signals — momentum (7 timeframes), "
            f"MA crossovers, RSI, MACD, Bollinger, Stochastic, Ichimoku, OBV, volume-price trend, "
            f"variance ratio, autocorrelation, SPY alpha, VIX regime, regime detection, candle patterns.</div>"
            f"</div>", unsafe_allow_html=True)

    # Signal detail table
    st.markdown(f"<div style='margin-top:16px;color:{TEXT_MUTED};font-size:0.65rem;text-transform:uppercase;"
                f"letter-spacing:1.5px;font-weight:600;margin-bottom:8px'>Signal Details</div>", unsafe_allow_html=True)
    sig_rows = ""
    for name, direction, desc in analysis["signals"]:
        cls = {"Bullish":"evpill-bull","Bearish":"evpill-bear"}.get(direction,"evpill-neut")
        icon = {"Bullish":"▲","Bearish":"▼"}.get(direction,"→")
        sig_rows += (
            f"<tr style='border-bottom:1px solid {BORDER}'>"
            f"<td style='padding:10px 14px;color:{TEXT_SECONDARY};font-size:0.8rem;font-weight:600;white-space:nowrap'>{name}</td>"
            f"<td style='padding:10px 14px'><span class='evpill {cls}'>{icon} {direction}</span></td>"
            f"<td style='padding:10px 14px;color:{TEXT_MUTED};font-size:0.78rem'>{desc}</td>"
            f"</tr>"
        )
    st.markdown(
        f"<table style='width:100%;border-collapse:collapse;background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;overflow:hidden'>"
        f"<thead><tr style='border-bottom:1px solid {BORDER_ACCENT}'>"
        f"<th style='padding:10px 14px;color:{TEXT_MUTED};font-size:0.65rem;text-align:left;font-weight:600;text-transform:uppercase;letter-spacing:1px'>Indicator</th>"
        f"<th style='padding:10px 14px;color:{TEXT_MUTED};font-size:0.65rem;text-align:left;font-weight:600;text-transform:uppercase;letter-spacing:1px'>Signal</th>"
        f"<th style='padding:10px 14px;color:{TEXT_MUTED};font-size:0.65rem;text-align:left;font-weight:600;text-transform:uppercase;letter-spacing:1px'>Reading</th>"
        f"</tr></thead><tbody>{sig_rows}</tbody></table>", unsafe_allow_html=True)


# ── TAB: OPTIONS ─────────────────────────────────────────────────────────────

with tab_opts:
    st.markdown(
        f"<div class='ctx' style='margin-bottom:14px'>"
        f"Strategies selected from ML <b>direction</b>, <b>magnitude</b>, and <b>confidence</b> "
        f"per horizon, combined with estimated IV. Premiums are <b>approximations</b>. "
        f"Verify with your broker.</div>", unsafe_allow_html=True)

    opt_tabs = st.tabs(list(HORIZONS.keys()))
    for opt_tab, h in zip(opt_tabs, HORIZONS.keys()):
        with opt_tab:
            if h not in predictions:
                st.info(f"Not enough training data for {h} predictions with the selected history window. Try 5y or 10y.")
                continue
            opt    = options.get(h, {})
            pred_h = predictions[h]

            if "error" in opt or opt.get("strategy","") == "N/A":
                st.warning("Could not generate strategy for this horizon.")
                continue

            sh1, sh2, sh3 = st.columns([1.2, 1.5, 1.5])

            with sh1:
                dc_map = {"Bullish":GREEN,"Bearish":RED,"Mildly Bullish":"#6ee7b7","Neutral":AMBER,"Neutral (Volatility Play)":PURPLE}
                dc = dc_map.get(opt["direction"],TEXT_SECONDARY)
                st.markdown(
                    f"<div class='card' style='text-align:center;padding:24px 16px'>"
                    f"<div style='font-size:2.2rem'>{opt['emoji']}</div>"
                    f"<div style='color:{TEXT_PRIMARY};font-size:1.15rem;font-weight:800;margin:6px 0 4px'>{opt['strategy']}</div>"
                    f"<div style='color:{dc};font-size:0.82rem;font-weight:700'>{opt['direction']}</div>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.7rem;margin-top:6px'>Complexity: {opt['complexity']}</div>"
                    f"<hr style='margin:10px 0'>"
                    f"<div style='color:{_rc(pred_h['predicted_return'])};font-size:1rem;font-weight:700'>"
                    f"{_arr(pred_h['predicted_return'])} {abs(pred_h['predicted_return']*100):.1f}% predicted</div>"
                    f"<div style='color:{_cc(pred_h['confidence'])};font-size:0.78rem'>{pred_h['confidence']:.0f}% confidence</div>"
                    f"</div>", unsafe_allow_html=True)

            with sh2:
                st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:8px'>Economics</div>", unsafe_allow_html=True)
                for lbl, val, tkey in [
                    ("Cost / Credit", opt.get("estimated_cost","N/A"), None),
                    ("Max Profit",    opt.get("max_profit","N/A"),     None),
                    ("Max Loss",      opt.get("max_loss","N/A"),       None),
                    ("Breakeven",     opt.get("breakeven","N/A"),      None),
                    ("IV Estimate",   opt.get("iv_used","N/A"),        "IV"),
                    ("Expiry",        f"~{opt.get('expiry_days',45)}d", None),
                ]:
                    lab = tip(lbl, TIPS[tkey]) if tkey else lbl
                    st.markdown(
                        f"<div class='mrow'><span class='mrow-label'>{lab}</span>"
                        f"<span class='mrow-value'>{val}</span></div>", unsafe_allow_html=True)

            with sh3:
                st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:8px'>Rationale</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='card' style='color:{TEXT_SECONDARY};font-size:0.8rem;line-height:1.7;margin-bottom:8px'>"
                    f"{opt['rationale']}</div>"
                    f"<div style='background:{AMBER_DIM};border:1px solid #78350f40;border-left:3px solid {AMBER};"
                    f"border-radius:8px;padding:10px 14px;color:{AMBER};font-size:0.75rem'>"
                    f"⚠ {opt.get('risk_note','')}</div>", unsafe_allow_html=True)

            st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin:14px 0 6px'>Trade Legs</div>", unsafe_allow_html=True)
            leg_cols = st.columns(max(len(opt.get("legs",[])), 1))
            for i, leg in enumerate(opt.get("legs",[])):
                is_buy  = leg["action"]=="BUY"
                act_col = GREEN if is_buy else RED
                typ_col = BLUE if leg["type"]=="CALL" else "#fb923c"
                leg_cls = "leg-buy" if is_buy else "leg-sell"
                with leg_cols[i]:
                    st.markdown(
                        f"<div class='leg-card {leg_cls}'>"
                        f"<div style='color:{act_col};font-size:0.75rem;font-weight:800;letter-spacing:1px'>{leg['action']}</div>"
                        f"<div style='color:{typ_col};font-size:1.15rem;font-weight:900;margin:3px 0'>{leg['type']}</div>"
                        f"<div style='color:{TEXT_PRIMARY};font-size:0.85rem'>Strike <b>${leg['strike']:,.0f}</b></div>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.73rem;margin-top:4px'>Exp: {leg['expiry']} · Prem: <b style='color:{TEXT_SECONDARY}'>${leg['premium']:.2f}</b></div>"
                        f"</div>", unsafe_allow_html=True)


# ── TAB: BACKTEST ────────────────────────────────────────────────────────────

with tab_bt:
    if bt_results is None:
        st.info("Enable **Backtest** in the sidebar and click Run Analysis.")
    else:
        st.markdown(
            f"<div class='ctx' style='margin-bottom:14px'>"
            f"<b>{tip('Walk-forward', TIPS['WF'])}</b>: fresh model trained on each 2-year window, "
            f"tested on the next 3 months, stepping 1 month. Tests across <b>bull, bear, sideways</b> markets.</div>",
            unsafe_allow_html=True)

        st.dataframe(backtest_summary_df(bt_results), hide_index=True, use_container_width=True)

        sel_bt = st.radio("Horizon", list(bt_results.keys()), horizontal=True, key="bt_h")
        r      = bt_results[sel_bt]

        if "error" in r:
            st.warning(r["error"])
        else:
            col_a, col_b = st.columns([1.6, 1])

            with col_a:
                acc_vals  = [w["dir_acc"]*100 for w in r["window_details"]]
                acc_dates = [w["start"]       for w in r["window_details"]]
                regimes   = [w["regime"]      for w in r["window_details"]]
                reg_clr   = {"Bull":GREEN,"Bear":RED,"Sideways":AMBER}

                fig_acc = go.Figure()
                fig_acc.add_hline(y=50, line=dict(color="#1e3050",dash="dash"),
                                  annotation_text=" 50% random",
                                  annotation_font=dict(color=TEXT_MUTED,size=9))
                fig_acc.add_trace(go.Bar(
                    x=acc_dates, y=acc_vals,
                    marker_color=[reg_clr.get(reg,BLUE) for reg in regimes],
                    text=[f"{v:.0f}%" for v in acc_vals],
                    textposition="outside", textfont=dict(size=9),
                ))
                fig_acc.update_layout(
                    height=280, template="plotly_dark",
                    paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_SURFACE,
                    yaxis=dict(range=[0,110], gridcolor="#0c1525"),
                    xaxis=dict(gridcolor="#0c1525"),
                    margin=dict(l=10,r=10,t=10,b=10), showlegend=False,
                )
                st.plotly_chart(fig_acc, use_container_width=True)
                st.caption("Green = Bull · Red = Bear · Amber = Sideways")

            with col_b:
                # Regime accuracy bars
                reg_acc = r.get("regime_accuracy", {})
                for regime, acc in reg_acc.items():
                    rc = reg_clr.get(regime,BLUE)
                    st.markdown(
                        f"<div style='margin:8px 0'>"
                        f"<div style='display:flex;justify-content:space-between;margin-bottom:3px'>"
                        f"<span style='color:{rc};font-weight:700;font-size:0.8rem'>{regime}</span>"
                        f"<span style='color:{rc};font-weight:700;font-size:0.8rem'>{acc}%</span></div>"
                        f"<div style='background:{BORDER};border-radius:20px;height:6px'>"
                        f"<div style='width:{int(acc)}%;background:{rc};height:6px;border-radius:20px'></div>"
                        f"</div></div>", unsafe_allow_html=True)

                st.markdown(f"<div style='margin-top:12px'>", unsafe_allow_html=True)
                for lbl, val in [
                    (tip("Dir Acc",   TIPS["Dir Acc"]),  f"{r['dir_accuracy']}%"),
                    (tip("Win Rate",  TIPS["Win Rate"]), f"{r['win_rate']}%"),
                    (tip("Sharpe",    TIPS["Sharpe"]),   str(r["strat_sharpe"])),
                    ("B&H Sharpe",                        str(r["bh_sharpe"])),
                    (tip("Max DD",    TIPS["Max DD"]),   f"{r['max_drawdown']}%"),
                    (tip("PF",        TIPS["PF"]),       str(r["profit_factor"])),
                    ("Windows",                           str(r["n_windows"])),
                ]:
                    st.markdown(
                        f"<div class='mrow'><span class='mrow-label'>{lbl}</span>"
                        f"<span class='mrow-value'>{val}</span></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Equity curve
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=r["strategy_cum"].index, y=r["strategy_cum"].values,
                name="ML Strategy", line=dict(color=GREEN,width=2.2),
                fill="tozeroy", fillcolor="rgba(16,185,129,0.04)",
            ))
            fig_eq.add_trace(go.Scatter(
                x=r["bh_cum"].index, y=r["bh_cum"].values,
                name="Buy & Hold", line=dict(color=BLUE,width=1.8,dash="dash"),
            ))
            fig_eq.add_hline(y=1.0, line=dict(color="#1e3050",width=1))
            fig_eq.update_layout(
                height=260, template="plotly_dark",
                paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_SURFACE,
                yaxis=dict(title="Portfolio Value ($1 start)", gridcolor="#0c1525"),
                xaxis=dict(gridcolor="#0c1525"),
                legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                margin=dict(l=10,r=10,t=20,b=10),
            )
            st.plotly_chart(fig_eq, use_container_width=True)


# ── TAB: MODEL ───────────────────────────────────────────────────────────────

with tab_model:

    mc1, mc2 = st.columns([1, 1.4])

    with mc1:
        st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:8px'>Validation Calibration</div>", unsafe_allow_html=True)
        st.dataframe(predictor.calibration_report(), hide_index=True, use_container_width=True)
        st.markdown(
            f"<div class='ctx' style='margin-top:8px'>"
            f"<b>Dir Acc</b> = accuracy on held-out 20% data. 50% = random, 55%+ = edge, 60%+ = strong.<br>"
            f"<b>Mean Abs Error</b> = average prediction error magnitude. Lower = more precise.</div>",
            unsafe_allow_html=True)

    with mc2:
        st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:8px'>Prediction Distribution</div>", unsafe_allow_html=True)
        h_sel  = st.selectbox("Horizon", list(HORIZONS.keys()), index=1, key="ens_h")
        p_data = predictions[h_sel]
        ens_p  = p_data.get("all_preds", np.array([p_data["predicted_return"]]))

        fig_ens = go.Figure()
        fig_ens.add_trace(go.Histogram(
            x=ens_p * 100, nbinsx=8,
            marker_color=PURPLE, marker_line=dict(color="#7c3aed",width=1), opacity=0.85,
        ))
        fig_ens.add_vline(x=p_data["predicted_return"]*100,
                          line=dict(color=GREEN,width=2.5,dash="dash"),
                          annotation_text=f" Meta: {p_data['predicted_return']*100:.2f}%",
                          annotation_font=dict(color=GREEN,size=10))
        fig_ens.add_vline(x=0, line=dict(color=TEXT_MUTED,width=1))
        fig_ens.update_layout(
            height=220, template="plotly_dark",
            paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_SURFACE,
            xaxis_title="Predicted Return (%)", yaxis=dict(title="Count",gridcolor="#0c1525"),
            xaxis=dict(gridcolor="#0c1525"),
            margin=dict(l=10,r=10,t=15,b=10), showlegend=False,
        )
        st.plotly_chart(fig_ens, use_container_width=True)
        n_mdls = p_data.get("n_models", len(ens_p))
        agree  = p_data["ensemble_agreement"]
        st.markdown(
            f"<div class='ctx'>"
            f"<b>{n_mdls} L1 models</b> → Ridge meta-learner. "
            f"Agreement: <b style='color:{_cc(agree)}'>{agree:.0f}%</b>. "
            f"Tight cluster = consensus = higher confidence.</div>", unsafe_allow_html=True)

    st.divider()

    # Feature importance
    st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:6px'>Feature Importance</div>", unsafe_allow_html=True)
    fi_h   = st.selectbox("Horizon", list(HORIZONS.keys()), index=1, key="fi_h2")
    imp_df = predictor.feature_importance(fi_h)
    if not imp_df.empty:
        fig_fi = go.Figure(go.Bar(
            x=imp_df["importance"], y=imp_df["feature"], orientation="h",
            marker=dict(color=imp_df["importance"], colorscale="Blues", showscale=False),
            text=[f"{v:.4f}" for v in imp_df["importance"]],
            textposition="outside", textfont=dict(size=9, color=TEXT_MUTED),
        ))
        fig_fi.update_layout(
            height=420, template="plotly_dark",
            paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_SURFACE,
            yaxis=dict(autorange="reversed", gridcolor="#0c1525"),
            xaxis=dict(gridcolor="#0c1525"),
            margin=dict(l=10,r=70,t=10,b=10),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # SHAP
    if HAS_SHAP:
        st.divider()
        st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:6px'>SHAP Importance</div>", unsafe_allow_html=True)
        shap_h  = st.selectbox("Horizon", list(HORIZONS.keys()), index=1, key="shap_h")
        shap_df = predictor.shap_importance(df, market_ctx, horizon=shap_h)
        if not shap_df.empty:
            fig_shap = go.Figure(go.Bar(
                x=shap_df["shap_importance"], y=shap_df["feature"], orientation="h",
                marker=dict(color=shap_df["shap_importance"], colorscale="Teal", showscale=False),
                text=[f"{v:.4f}" for v in shap_df["shap_importance"]],
                textposition="outside", textfont=dict(size=9, color=TEXT_MUTED),
            ))
            fig_shap.update_layout(
                height=420, template="plotly_dark",
                paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_SURFACE,
                yaxis=dict(autorange="reversed", gridcolor="#0c1525"),
                xaxis=dict(gridcolor="#0c1525"),
                margin=dict(l=10,r=70,t=10,b=10),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("Enable SHAP Selection in sidebar for detailed SHAP values.")

    # Regime signals
    st.divider()
    st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:8px'>Regime Signals</div>", unsafe_allow_html=True)
    if regime_info and regime_info.get("signals"):
        rsigs = regime_info["signals"]
        sig_cols = st.columns(min(len(rsigs), 4))
        for i, (sname, sval, svote, slbl) in enumerate(rsigs):
            vc = GREEN if svote > 0 else (RED if svote < 0 else AMBER)
            icon = "▲" if svote > 0 else ("▼" if svote < 0 else "→")
            val_str = f"{sval:.3f}" if isinstance(sval, float) else str(sval)
            with sig_cols[i % len(sig_cols)]:
                st.markdown(
                    f"<div class='card-sm' style='margin-bottom:6px'>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;font-weight:600'>{sname}</div>"
                    f"<div style='color:{TEXT_PRIMARY};font-size:0.88rem;font-weight:700;margin:3px 0'>{val_str}</div>"
                    f"<div style='color:{vc};font-size:0.72rem;font-weight:700'>{icon} {slbl}</div>"
                    f"</div>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='text-align:center;color:{BORDER};font-size:0.68rem;padding:20px 0 6px'>"
    f"SignalEdge is for informational and educational purposes only. "
    f"Not financial advice. Past performance ≠ future results.</div>",
    unsafe_allow_html=True)
