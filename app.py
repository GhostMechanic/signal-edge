"""
app.py  –  Prediqt v4.0
─────────────────────────────────────
Premium ML stock prediction interface.
Powered by Prediqt IQ intelligence layer.
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
import streamlit.components.v1 as _st_components
import pandas as pd
import numpy as np
from datetime import datetime
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
from prediction_logger_v2 import (
    log_prediction_v2, quick_score_predictions, score_all_intervals,
    backtest_accuracy, get_full_analytics, export_predictions_csv,
    export_predictions_dataframe, get_current_model_version,
    should_retrain, migrate_from_v1, get_feature_importance_ranking,
    deduplicate_log,
)
from model_improvement import ModelImprover
from auto_retrain import (
    should_retrain_now, get_improvement_metrics, execute_retrain,
    check_and_log_retrain_status, apply_confidence_adjustments,
)
# Watchlist persistence routes through db.py (feature-flagged: file-backed
# by default, Supabase when USE_SUPABASE=true). `get_quick_signal` is a pure
# compute — stays in watchlist.py.
from db import (
    load_watchlist, save_watchlist, add_to_watchlist, remove_from_watchlist,
)
from watchlist import get_quick_signal
from universe import get_full_universe, get_dow30_tickers, get_nasdaq100_tickers, get_sp500_tickers
from batch_scanner import run_daily_pipeline, get_last_scan_summary, get_last_screen_results
from paper_trader import (
    get_portfolio_stats, reset_portfolio, update_portfolio,
)
from learning_engine import (
    analyze_scored_predictions, get_learning_summary,
)
from sentiment_analyzer import get_sentiment_features, fetch_news_sentiment, fetch_fear_greed


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Prediqt — Smarter Stock Predictions",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design tokens (Prediqt brand palette) ───────────────────────────────────
BG_PRIMARY   = "#030508"
BG_SURFACE   = "#0a0e14"
BG_CARD      = "#0f1319"
BG_ELEVATED  = "#141a22"
BORDER       = "#1a2233"
BORDER_ACCENT= "#1e3050"
TEXT_PRIMARY  = "#f0f4fa"
TEXT_SECONDARY= "#8898aa"
TEXT_MUTED    = "#4a5a6e"
GREEN        = "#10b981"
GREEN_DIM    = "#064e3b"
RED          = "#ef4444"
RED_DIM      = "#450a0a"
AMBER        = "#f59e0b"
AMBER_DIM    = "#451a03"
BLUE         = "#2080e5"          # Prediqt deep blue  (matches landing page)
CYAN         = "#06d6a0"          # Prediqt teal/cyan accent
PURPLE       = "#1e90ff"          # lighter blue for gradient end
BRAND_GRAD   = "linear-gradient(135deg, #2080e5 0%, #06d6a0 100%)"  # core brand gradient (matches landing)
BRAND_GLOW   = "0 8px 32px rgba(32,128,229,0.35)"                   # primary-button glow
BRAND_GLOW_SM= "0 4px 14px rgba(32,128,229,0.28)"

# ── Chart palette (brand-first, convention-safe) ──────────────────────────────
# Bullish uses brand cyan (close enough to green to read as "up"), bearish stays
# a muted red, and prediction/neutral series use brand blue.
CHART_UP        = CYAN          # "#06d6a0"
CHART_UP_FILL   = "rgba(6,214,160,0.22)"
CHART_DOWN      = "#e04a4a"     # slightly muted red (easier on dark bg than pure #ef4444)
CHART_DOWN_FILL = "rgba(224,74,74,0.22)"
CHART_NEUTRAL   = BLUE          # "#2080e5"
CHART_GRID      = "rgba(255,255,255,0.04)"
# Brand colorscale for heatmaps / sequential bar fills — dark-blue → cyan
CHART_BRAND_SCALE = [
    [0.0, "#0a2e55"],   # deep navy (low)
    [0.5, "#2080e5"],   # brand blue (mid)
    [1.0, "#06d6a0"],   # brand cyan (high)
]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Reset ── */
.stApp {{ background-color: {BG_PRIMARY}; color: {TEXT_PRIMARY}; font-family: 'Inter', -apple-system, sans-serif; }}
.block-container {{ padding-top: 0rem; max-width: 1400px; }}
/* Header: keep transparent but visible so the sidebar collapse hamburger
   remains clickable. display:none on the header orphans the sidebar. */
header[data-testid="stHeader"] {{
    background: transparent !important;
    backdrop-filter: none !important;
    z-index: 100 !important;
}}
/* Hide deploy/menu buttons but preserve the sidebar collapse control */
header[data-testid="stHeader"] [data-testid="stToolbar"] {{
    background: transparent !important;
}}
header[data-testid="stHeader"] [data-testid="stToolbar"] button:not([data-testid="stSidebarCollapseButton"]):not([data-testid="collapsedControl"]):not([data-testid="stExpandSidebarButton"]) {{
    display: none !important;
}}
/* ────────────────────────────────────────────────────────────────────────
   SIDEBAR TOGGLE — Streamlit 1.50 renders TWO distinct buttons:
     1) stExpandSidebarButton  — floating; appears when sidebar is CLOSED
        (lives outside the sidebar, at top-left of the app)
     2) stSidebarCollapseButton — inline; lives INSIDE the sidebar header
        (clicked to close the sidebar)
   Style each one differently to match its role.
   ──────────────────────────────────────────────────────────────────────── */

/* ---------- The OPEN / EXPAND button — floating branded card ---------- */
[data-testid="stExpandSidebarButton"] {{
    position: relative !important;
    z-index: 99999 !important;
}}
[data-testid="stExpandSidebarButton"] button {{
    background: linear-gradient(145deg, {BG_CARD}, {BG_SURFACE}) !important;
    border: 1px solid {BORDER_ACCENT} !important;
    border-radius: 11px !important;
    width: 44px !important;
    height: 44px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: {CYAN} !important;
    box-shadow: 0 4px 18px rgba(32,128,229,0.25),
                0 0 0 1px rgba(6,214,160,0.08) !important;
    transition: transform 0.18s ease, box-shadow 0.22s ease, border-color 0.2s ease !important;
    cursor: pointer !important;
}}
[data-testid="stExpandSidebarButton"] button:hover {{
    transform: translateY(-1px) !important;
    border-color: {CYAN} !important;
    box-shadow: 0 6px 24px rgba(6,214,160,0.28),
                0 0 0 2px rgba(6,214,160,0.15) !important;
    background: {BG_ELEVATED} !important;
}}
/* Replace the Material-icon text (which can leak as "keyboard_double_arrow_right"
   when the icon font hasn't loaded) with our own branded SVG via background. */
[data-testid="stExpandSidebarButton"] button [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] button span[class*="material"] {{
    font-size: 0 !important;
    color: transparent !important;
    width: 22px !important;
    height: 22px !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2306d6a0' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'><polyline points='7 6 13 12 7 18'/><polyline points='13 6 19 12 13 18'/></svg>") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 22px 22px !important;
    display: inline-block !important;
    vertical-align: middle !important;
}}

/* ---------- The CLOSE button inside the sidebar — subtle chevron ---------- */
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: {TEXT_MUTED} !important;
    padding: 6px !important;
    border-radius: 8px !important;
    transition: all 0.18s ease !important;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover {{
    background: rgba(255,255,255,0.05) !important;
    color: {CYAN} !important;
    transform: none !important;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button [data-testid="stIconMaterial"],
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button span[class*="material"] {{
    font-size: 0 !important;
    color: transparent !important;
    width: 20px !important;
    height: 20px !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238898aa' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'><polyline points='17 6 11 12 17 18'/><polyline points='11 6 5 12 11 18'/></svg>") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 20px 20px !important;
    display: inline-block !important;
    vertical-align: middle !important;
}}
/* Hover turns the in-sidebar chevron cyan */
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover [data-testid="stIconMaterial"],
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover span[class*="material"] {{
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2306d6a0' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'><polyline points='17 6 11 12 17 18'/><polyline points='11 6 5 12 11 18'/></svg>") !important;
}}
/* Guarantee a sensible minimum width on the sidebar — don't rely on
   aria-expanded which Streamlit doesn't always set. These rules apply to any
   rendered sidebar so collapsing by the hamburger still works (it uses
   transform/width transitions that Streamlit handles). */
section[data-testid="stSidebar"]:not([aria-expanded="false"]) {{
    min-width: 264px !important;
    width: 264px !important;
    flex-shrink: 0 !important;
}}
section[data-testid="stSidebar"]:not([aria-expanded="false"]) > div:first-child {{
    min-width: 264px !important;
    width: 264px !important;
}}
/* Inner content wrapper must not collapse either */
section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
    min-width: 224px !important;
}}

/* ── Surface & Cards ── */
.surface {{ background: {BG_SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; }}
.card {{
    background: linear-gradient(145deg, {BG_CARD} 0%, {BG_SURFACE} 100%);
    border: 1px solid {BORDER}; border-radius: 14px; padding: 22px;
    backdrop-filter: blur(8px);
    transition: transform 0.22s cubic-bezier(0.4,0,0.2,1),
                border-color 0.22s, box-shadow 0.22s;
}}
.card:hover {{
    transform: translateY(-2px);
    border-color: {BORDER_ACCENT};
    box-shadow: 0 10px 30px rgba(0,0,0,0.35), 0 0 0 1px rgba(32,128,229,0.08);
}}
.card-sm {{
    background: linear-gradient(145deg, {BG_CARD} 0%, {BG_SURFACE} 100%);
    border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px;
    overflow: visible; backdrop-filter: blur(8px);
}}

/* ── Hero Signal Card ── */
.hero-signal {{
    border-radius: 18px; padding: 36px 32px; text-align: center;
    position: relative; overflow: visible;
    box-shadow: 0 8px 40px rgba(0,0,0,0.3);
}}
.hero-signal::before {{
    content: ''; position: absolute; inset: 0; border-radius: 18px;
    background: radial-gradient(ellipse at 50% 0%, var(--glow) 0%, transparent 70%);
    opacity: 0.15; pointer-events: none; z-index: 0;
}}
.hero-signal::after {{
    content: ''; position: absolute; inset: -1px; border-radius: 18px;
    background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, transparent 40%);
    pointer-events: none; z-index: 0;
}}

/* ── Metric Row ── */
.mrow {{ display: flex; justify-content: space-between; align-items: center;
         padding: 11px 0; border-bottom: 1px solid {BORDER}; }}
.mrow:last-child {{ border-bottom: none; }}
.mrow-label {{ color: {TEXT_SECONDARY}; font-size: 0.78rem; font-weight: 500; }}
.mrow-value {{ color: {TEXT_PRIMARY}; font-size: 0.85rem; font-weight: 700; letter-spacing: -0.01em; }}

/* ── Stat Tile ── */
.stat {{
    background: linear-gradient(145deg, {BG_CARD} 0%, {BG_SURFACE} 100%);
    border: 1px solid {BORDER}; border-radius: 12px;
    padding: 18px 14px; text-align: center;
    position: relative;
}}
.stat::before {{
    content: ''; position: absolute; inset: -1px; border-radius: 12px;
    background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, transparent 50%);
    pointer-events: none;
}}
.stat-label {{ color: {TEXT_MUTED}; font-size: 0.62rem; text-transform: uppercase;
               letter-spacing: 1.5px; margin-bottom: 8px; font-weight: 600; }}
.stat-value {{ color: {TEXT_PRIMARY}; font-size: 1.2rem; font-weight: 800; line-height: 1; letter-spacing: -0.02em; }}
.stat-sub   {{ color: {TEXT_SECONDARY}; font-size: 0.68rem; margin-top: 6px; }}

/* ── Horizon Strip ── */
.hstrip {{
    background: linear-gradient(145deg, {BG_CARD} 0%, {BG_SURFACE} 100%);
    border: 1px solid {BORDER}; border-radius: 12px;
    padding: 18px; display: flex; flex-direction: column; gap: 6px;
    position: relative;
    overflow: hidden;
}}
/* Brand gradient hairline at the top — matches metric cards, portfolio
   cards, and Catalysts event cards for visual consistency across the app. */
.hstrip::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1.5px;
    background: {BRAND_GRAD};
    opacity: 0.55;
    pointer-events: none;
    z-index: 1;
}}
/* Subtle top-inner highlight for extra depth */
.hstrip::after {{
    content: '';
    position: absolute;
    inset: -1px;
    border-radius: 12px;
    background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, transparent 40%);
    pointer-events: none;
}}
.hstrip-head {{ display: flex; justify-content: space-between; align-items: baseline; }}
.hstrip-label {{ color: {TEXT_MUTED}; font-size: 0.65rem; font-weight: 700;
                 text-transform: uppercase; letter-spacing: 1.2px; }}
.hstrip-price {{ color: {TEXT_PRIMARY}; font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em; }}
.hstrip-ret   {{ font-size: 0.85rem; font-weight: 700; }}
.hstrip-range {{ color: {TEXT_MUTED}; font-size: 0.68rem; }}
.hstrip-conf  {{ height: 3px; border-radius: 3px; background: {BORDER}; margin-top: 4px; overflow: hidden; }}
.hstrip-conf-fill {{ height: 100%; border-radius: 3px; }}

/* ── Evidence Pill ── */
.evpill {{
    display: inline-flex; align-items: center; gap: 5px;
    border-radius: 20px; padding: 5px 12px; font-size: 0.7rem; font-weight: 600;
}}
.evpill-bull {{ background: {GREEN_DIM}; color: {GREEN}; border: 1px solid #065f4640; }}
.evpill-bear {{ background: {RED_DIM}; color: {RED}; border: 1px solid #7f1d1d40; }}
.evpill-neut {{ background: {AMBER_DIM}; color: {AMBER}; border: 1px solid #78350f40; }}

/* ── Confidence Badges ── */
.conf-high {{ background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.4); color: {GREEN}; }}
.conf-medium {{ background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.4); color: {AMBER}; }}
.conf-low {{ background: rgba(249,115,22,0.12); border: 1px solid rgba(249,115,22,0.4); color: #f97316; }}
.conf-badge {{ border-radius: 8px; padding: 8px 12px; font-weight: 700; font-size: 0.75rem; text-align: center; }}

/* ── Confidence Arc Gauge ── */
.conf-gauge {{ position: relative; width: 120px; height: 66px; margin: 0 auto; overflow: hidden; }}
.conf-gauge svg {{ width: 120px; height: 66px; }}
.conf-gauge-text {{ position: absolute; bottom: 2px; left: 0; right: 0; text-align: center; font-weight: 800; font-size: 1.6rem; }}

/* ── Regime Chip ── */
.regime-chip {{
    display: inline-flex; align-items: center; gap: 6px;
    border-radius: 20px; padding: 5px 16px; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.3px;
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
    border-radius: 10px; padding: 12px 16px;
    position: absolute; z-index: 100;
    bottom: 130%; left: 50%; transform: translateX(-50%);
    width: 240px; box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    transition: opacity 0.2s ease;
    white-space: normal; text-align: left;
}}
.tip:hover .tiptext {{ visibility: visible; opacity: 1; }}

/* ── Inline info icon tooltip (for hero card metrics) ── */
.info-icon {{
    position: relative; display: inline-block;
    cursor: help; margin-left: 4px; vertical-align: middle;
}}
/* Tooltip is rendered in a fixed overlay div via JS — not as a child */
#se-tooltip {{
    display: none;
    position: fixed; z-index: 999999;
    background: {BG_ELEVATED}; color: {TEXT_SECONDARY};
    font-size: 0.73rem; line-height: 1.5; font-weight: 400;
    border: 1px solid {BORDER_ACCENT};
    border-radius: 8px; padding: 10px 14px;
    width: 240px; max-width: 80vw;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    white-space: normal; text-align: left;
    letter-spacing: 0; text-transform: none;
    pointer-events: none;
}}
/* Keep hidden .info-tip as data source only */
.info-icon .info-tip {{
    display: none;
}}

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
    color: {TEXT_SECONDARY}; font-size: 0.65rem; margin-top: 3px; font-style: italic;
}}

/* ── Options leg cards ── */
/* Options trade legs — brand-aligned tints instead of saturated greens/reds */
.leg-buy {{
    border-left: 3px solid {CHART_UP};
    background: linear-gradient(90deg, rgba(6,214,160,0.12), rgba(6,214,160,0.04));
}}
.leg-sell {{
    border-left: 3px solid {CHART_DOWN};
    background: linear-gradient(90deg, rgba(224,74,74,0.12), rgba(224,74,74,0.04));
}}
.leg-card {{
    border-radius: 8px;
    padding: 10px 14px;
    margin: 5px 0;
    border: 1px solid {BORDER};
    border-left-width: 3px;
    font-size: 0.8rem;
    letter-spacing: 0.005em;
}}

/* ── Tooltips ── */
.tooltip-icon {{
    display: inline-block; width: 16px; height: 16px; border-radius: 50%;
    background: {BORDER_ACCENT}; color: {BLUE}; font-size: 0.7rem; font-weight: bold;
    line-height: 16px; text-align: center; cursor: help; margin-left: 4px;
    position: relative; z-index: 2;
}}
.tooltip-icon:hover::after {{
    content: attr(data-tip);
    position: absolute; bottom: calc(100% + 10px); left: 50%; transform: translateX(-50%);
    background: {BG_ELEVATED}; border: 1px solid {BORDER_ACCENT}; color: {TEXT_PRIMARY};
    padding: 10px 14px; border-radius: 8px;
    font-size: 0.75rem; font-weight: 400; line-height: 1.5;
    letter-spacing: 0; text-transform: none; text-align: left;
    white-space: normal; width: 260px; max-width: 80vw;
    z-index: 10000; box-shadow: 0 8px 32px rgba(0,0,0,0.5); pointer-events: none;
}}
.tooltip-icon:hover::before {{
    content: ''; position: absolute; bottom: calc(100% + 4px); left: 50%; transform: translateX(-50%);
    border: 6px solid transparent; border-top-color: {BORDER_ACCENT}; z-index: 10000;
}}

/* ── Animations & Transitions ── */
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to {{ opacity: 1; }}
}}
@keyframes slideUp {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes shimmer {{
    0% {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.7; }}
}}
@keyframes glowPulse {{
    0%, 100% {{ box-shadow: 0 0 0 rgba(6,214,160,0); }}
    50% {{ box-shadow: 0 0 20px rgba(6,214,160,0.15); }}
}}

.card {{ transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); }}
.card:hover {{ border-color: {BORDER_ACCENT}; box-shadow: 0 8px 32px rgba(0,0,0,0.25); transform: translateY(-1px); }}
.stat {{ transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); }}
.stat:hover {{ transform: translateY(-2px); border-color: {BORDER_ACCENT}; box-shadow: 0 6px 24px rgba(0,0,0,0.2); }}
.hstrip {{ transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); }}
.hstrip:hover {{ border-color: {BORDER_ACCENT}; box-shadow: 0 6px 24px rgba(0,0,0,0.2); transform: translateY(-1px); }}

/* ── Expander styles ── */
.stExpander {{ transition: all 0.3s ease; }}
.stExpander:hover {{ border-color: {BORDER_ACCENT} !important; }}

/* ── Button styles ── */
button {{ transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important; }}
button:hover {{ transform: translateY(-1px); box-shadow: 0 8px 24px rgba(6,214,160,0.25) !important; }}

/* ── Streamlit overrides ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {BG_SURFACE}; border-radius: 12px; padding: 4px; gap: 2px;
    border: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px; color: {TEXT_MUTED}; padding: 9px 22px;
    font-weight: 600; font-size: 0.8rem; letter-spacing: 0.2px;
    transition: all 0.2s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {TEXT_SECONDARY}; }}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {BORDER_ACCENT} 0%, rgba(6,214,160,0.1) 100%) !important;
    color: {TEXT_PRIMARY} !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    border-bottom: 2px solid {CYAN} !important;
}}
button[kind="primary"] {{
    background: {BRAND_GRAD} !important;
    border: none !important; font-weight: 700 !important;
    box-shadow: {BRAND_GLOW_SM} !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
}}
button[kind="primary"]:hover {{
    box-shadow: {BRAND_GLOW} !important;
    filter: brightness(1.08) !important;
    transform: translateY(-1px) !important;
}}
button[kind="primary"]:active {{
    transform: translateY(0) !important;
    filter: brightness(1) !important;
}}
hr {{ border-color: {BORDER} !important; margin: 0.8rem 0; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG_PRIMARY}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER_ACCENT}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {TEXT_MUTED}; }}

/* ── Selection ── */
::selection {{ background: rgba(6,214,160,0.3); color: {TEXT_PRIMARY}; }}

/* ══════════════════════════════════════════════════════════════════════════
   SIDEBAR — brand surface, Inter typography, unified text hierarchy
   ══════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {BG_SURFACE} 0%, {BG_PRIMARY} 100%) !important;
    border-right: 1px solid {BORDER} !important;
}}
section[data-testid="stSidebar"] * {{
    font-family: 'Inter', -apple-system, sans-serif !important;
}}
/* Body copy inside sidebar */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {{
    letter-spacing: 0.005em;
}}
/* Caption treatment (st.caption) */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    color: {TEXT_MUTED} !important;
    font-size: 0.68rem !important;
    font-style: italic;
    letter-spacing: 0.02em !important;
    line-height: 1.5 !important;
    padding: 6px 2px 0;
    opacity: 0.85;
}}
/* Labels for inputs inside the sidebar — slightly tighter */
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stMultiSelect label {{
    color: {TEXT_MUTED} !important;
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    margin-bottom: 5px !important;
}}
/* Headings inside sidebar */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {{
    color: {TEXT_PRIMARY} !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.01em !important;
    font-weight: 800 !important;
}}
/* Button (non-primary) in sidebar: full-width with brand hover */
section[data-testid="stSidebar"] .stButton > button {{
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.02em !important;
}}
section[data-testid="stSidebar"] .stRadio > label {{
    font-size: 0.82rem !important;
}}

/* ── Page content fade-in ── */
.main .block-container {{
    animation: fadeIn 0.3s ease-out;
}}

/* ── Dataframe dark theme ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden;
}}
[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {{
    border-radius: 10px !important;
}}

/* ── Streamlit metric cards dark polish ── */
[data-testid="stMetric"] {{
    background: linear-gradient(145deg, {BG_CARD} 0%, {BG_SURFACE} 100%);
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-size: 0.6rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}}

/* Sidebar nav radio is styled separately in an inline <style> block near
   the nav — see `_nav_icon_css` generation. Keeping only the divider polish
   here to avoid selector-specificity conflicts. */

/* ── Sidebar dividers ── */
section[data-testid="stSidebar"] hr {{
    border-color: {BORDER} !important;
    margin: 12px 0 !important;
    opacity: 0.5;
}}

/* ── Quick action buttons ── */
.stButton > button {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_PRIMARY} !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    padding: 10px 18px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}
.stButton > button:hover {{
    border-color: {CYAN}40 !important;
    box-shadow: 0 4px 16px rgba(6,214,160,0.15) !important;
    background: {BG_ELEVATED} !important;
}}

/* ── Mobile Responsive ── */
@media (max-width: 768px) {{
    .block-container {{ padding: 0.5rem 0.5rem !important; max-width: 100% !important; }}
    .hero-signal {{ padding: 20px 14px; border-radius: 12px; }}
    .hero-signal .stat-value, .hstrip-price {{ font-size: 1rem; }}
    .card {{ padding: 14px; border-radius: 10px; }}
    .card-sm {{ padding: 10px 12px; border-radius: 8px; }}
    .stat {{ padding: 10px 8px; }}
    .stat-label {{ font-size: 0.58rem; letter-spacing: 0.8px; }}
    .stat-value {{ font-size: 0.95rem; }}
    .hstrip {{ padding: 12px; }}
    .hstrip-label {{ font-size: 0.6rem; }}
    .evpill {{ font-size: 0.65rem; padding: 4px 8px; }}
    .tooltip-icon:hover::after {{ width: 200px; max-width: 70vw; font-size: 0.68rem; }}
    .conf-gauge {{ width: 90px; height: 50px; }}
    .conf-gauge svg {{ width: 90px; height: 50px; }}
    .conf-gauge-text {{ font-size: 1.2rem; }}
    .trade-rec {{ font-size: 0.63rem; }}
    .ctx {{ font-size: 0.72rem; padding: 10px 12px; }}

    /* Stack Streamlit columns vertically */
    [data-testid="column"] {{ width: 100% !important; flex: 100% !important; min-width: 100% !important; }}

    /* Tab bar: scroll horizontally */
    .stTabs [data-baseweb="tab-list"] {{ overflow-x: auto; flex-wrap: nowrap; -webkit-overflow-scrolling: touch; }}
    .stTabs [data-baseweb="tab"] {{ font-size: 0.72rem; padding: 6px 12px; white-space: nowrap; }}
}}

@media (max-width: 480px) {{
    .block-container {{ padding: 0.3rem !important; }}
    .hero-signal {{ padding: 16px 10px; }}
    .stat-value {{ font-size: 0.85rem; }}
    .hstrip-price {{ font-size: 1rem; }}
    [data-testid="stMetricValue"] {{ font-size: 0.9rem !important; }}
    [data-testid="stMetricLabel"] {{ font-size: 0.65rem !important; }}
}}

/* ── Landing-page signature: decorative radial orbs behind the app shell ── */
.stApp::before, .stApp::after {{
    content: '';
    position: fixed;
    pointer-events: none;
    z-index: 0;
    border-radius: 50%;
    filter: blur(90px);
}}
.stApp::before {{
    /* Blue orb, upper-left */
    top: -220px; left: -180px;
    width: 640px; height: 640px;
    background: radial-gradient(circle, #2080e5 0%, transparent 68%);
    opacity: 0.42;
}}
.stApp::after {{
    /* Cyan orb, mid-right */
    top: 28%; right: -260px;
    width: 560px; height: 560px;
    background: radial-gradient(circle, #06d6a0 0%, transparent 68%);
    opacity: 0.28;
}}
/* Make sure content sits above the orbs */
.main, .block-container, section[data-testid="stSidebar"] {{
    position: relative;
    z-index: 1;
}}

/* ── Primary-button icon refinement — tighter, rounded, Inter-weight ── */
button[kind="primary"] p {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}}

/* ══════════════════════════════════════════════════════════════════════════
   LANDING-STYLE PAGE HEADER BLOCKS
   Use with: st.markdown(page_header('Eyebrow', 'Big headline <span class="grad">accent</span>', 'Optional sub'), unsafe_allow_html=True)
   ══════════════════════════════════════════════════════════════════════════ */
.page-header {{
    text-align: center;
    padding: 28px 0 32px;
    margin-bottom: 8px;
}}
.page-header .ph-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    border-radius: 100px;
    border: 1px solid rgba(6,214,160,0.35);
    background: rgba(6,214,160,0.08);
    color: {CYAN};
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 20px;
}}
.page-header .ph-eyebrow::before {{
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {CYAN};
    box-shadow: 0 0 10px {CYAN};
}}
.page-header h1 {{
    font-family: 'Inter', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.035em !important;
    color: {TEXT_PRIMARY} !important;
    line-height: 1.05 !important;
    margin: 0 0 12px !important;
}}
.page-header .grad {{
    background: linear-gradient(135deg, #2080e5 0%, #06d6a0 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}}
.page-header .ph-sub {{
    color: {TEXT_SECONDARY};
    font-size: 0.95rem;
    font-weight: 500;
    max-width: 620px;
    margin: 0 auto;
    line-height: 1.55;
}}

@media (max-width: 768px) {{
    .page-header h1 {{ font-size: 1.9rem !important; }}
    .page-header .ph-sub {{ font-size: 0.85rem; }}
}}

/* ══════════════════════════════════════════════════════════════════════════
   FORM CONTROLS — text, number, select, slider, date, multi-select, textarea
   ══════════════════════════════════════════════════════════════════════════ */

/* Baseline label treatment across all widgets */
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stMultiSelect label, .stDateInput label, .stTextArea label,
.stSlider label, .stRadio label, .stCheckbox label, .stToggle label,
.stFileUploader label, .stTimeInput label {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}}

/* Text / Number / Date / TextArea — base input chrome */
.stTextInput input, .stNumberInput input, .stDateInput input,
.stTextArea textarea, .stTimeInput input {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {TEXT_PRIMARY} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    box-shadow: none !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease !important;
}}
.stTextInput input:hover, .stNumberInput input:hover, .stDateInput input:hover,
.stTextArea textarea:hover, .stTimeInput input:hover {{
    border-color: {BORDER_ACCENT} !important;
    background: {BG_ELEVATED} !important;
}}
.stTextInput input:focus, .stNumberInput input:focus, .stDateInput input:focus,
.stTextArea textarea:focus, .stTimeInput input:focus {{
    border-color: {CYAN} !important;
    box-shadow: 0 0 0 3px rgba(6,214,160,0.15) !important;
    outline: none !important;
}}
/* Number input +/- steppers */
.stNumberInput button {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_SECONDARY} !important;
    transition: all 0.2s ease !important;
}}
.stNumberInput button:hover {{
    background: {BG_ELEVATED} !important;
    border-color: {CYAN}66 !important;
    color: {CYAN} !important;
}}

/* Selectbox / Multiselect — BaseWeb select */
div[data-baseweb="select"] > div {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}}
div[data-baseweb="select"] > div:hover {{
    border-color: {BORDER_ACCENT} !important;
    background: {BG_ELEVATED} !important;
}}
div[data-baseweb="select"]:focus-within > div {{
    border-color: {CYAN} !important;
    box-shadow: 0 0 0 3px rgba(6,214,160,0.15) !important;
}}
div[data-baseweb="select"] input,
div[data-baseweb="select"] span {{
    color: {TEXT_PRIMARY} !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}}
/* Dropdown menu popover */
div[data-baseweb="popover"] ul {{
    background: {BG_ELEVATED} !important;
    border: 1px solid {BORDER_ACCENT} !important;
    border-radius: 10px !important;
    box-shadow: 0 16px 40px rgba(0,0,0,0.45) !important;
}}
div[data-baseweb="popover"] li {{
    color: {TEXT_SECONDARY} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
}}
div[data-baseweb="popover"] li:hover {{
    background: rgba(32,128,229,0.10) !important;
    color: {TEXT_PRIMARY} !important;
}}
div[data-baseweb="popover"] li[aria-selected="true"] {{
    background: linear-gradient(90deg, rgba(32,128,229,0.20), rgba(6,214,160,0.10)) !important;
    color: {TEXT_PRIMARY} !important;
    font-weight: 600 !important;
}}
/* Multiselect selected-item tags */
div[data-baseweb="tag"] {{
    background: rgba(6,214,160,0.14) !important;
    border: 1px solid rgba(6,214,160,0.35) !important;
    border-radius: 6px !important;
    color: {CYAN} !important;
    font-weight: 600 !important;
}}

/* Slider — brand gradient track, cyan thumb */
.stSlider [data-baseweb="slider"] > div > div > div {{
    background: {BRAND_GRAD} !important;
}}
.stSlider [data-baseweb="slider"] [role="slider"] {{
    background: {CYAN} !important;
    border: 3px solid #fff !important;
    box-shadow: 0 0 0 4px rgba(6,214,160,0.25), 0 4px 12px rgba(0,0,0,0.4) !important;
    height: 18px !important;
    width: 18px !important;
}}
.stSlider [data-testid="stTickBar"] {{
    color: {TEXT_MUTED} !important;
}}

/* Checkbox — cyan check */
.stCheckbox [data-baseweb="checkbox"] div:first-child {{
    border-radius: 5px !important;
    border-color: {BORDER_ACCENT} !important;
    background: {BG_CARD} !important;
    transition: all 0.2s ease !important;
}}
.stCheckbox [data-baseweb="checkbox"][aria-checked="true"] div:first-child,
.stCheckbox input:checked + div {{
    background: {BRAND_GRAD} !important;
    border-color: transparent !important;
    box-shadow: 0 0 0 2px rgba(6,214,160,0.20) !important;
}}

/* Toggle switch — brand gradient when on */
.stToggle [role="switch"] {{
    background: {BORDER} !important;
    transition: background 0.2s ease !important;
}}
.stToggle [role="switch"][aria-checked="true"] {{
    background: {BRAND_GRAD} !important;
    box-shadow: 0 0 14px rgba(6,214,160,0.35) !important;
}}

/* Radio (non-sidebar, i.e. main content radios) */
.stRadio [role="radiogroup"] label {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.82rem !important;
}}
.stRadio [role="radio"] {{
    border-color: {BORDER_ACCENT} !important;
}}
.stRadio [role="radio"][aria-checked="true"] {{
    background: {CYAN} !important;
    border-color: {CYAN} !important;
    box-shadow: 0 0 0 3px rgba(6,214,160,0.20) !important;
}}

/* ══════════════════════════════════════════════════════════════════════════
   ALERT / MESSAGE BOXES — info / warning / success / error
   ══════════════════════════════════════════════════════════════════════════ */
div[data-testid="stAlert"] {{
    border-radius: 12px !important;
    border: 1px solid {BORDER} !important;
    backdrop-filter: blur(8px);
    padding: 14px 16px !important;
}}
/* info — brand blue */
div[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
div[data-testid="stNotificationContentInfo"] {{
    background: linear-gradient(145deg, rgba(32,128,229,0.10), rgba(6,214,160,0.05)) !important;
    border-color: rgba(32,128,229,0.40) !important;
    border-left: 3px solid {BLUE} !important;
    color: {TEXT_PRIMARY} !important;
}}
/* warning — amber */
div[data-testid="stNotificationContentWarning"] {{
    background: linear-gradient(145deg, rgba(245,158,11,0.10), rgba(245,158,11,0.04)) !important;
    border-color: rgba(245,158,11,0.40) !important;
    border-left: 3px solid {AMBER} !important;
}}
/* success — cyan */
div[data-testid="stNotificationContentSuccess"] {{
    background: linear-gradient(145deg, rgba(6,214,160,0.12), rgba(6,214,160,0.04)) !important;
    border-color: rgba(6,214,160,0.40) !important;
    border-left: 3px solid {CYAN} !important;
}}
/* error — red */
div[data-testid="stNotificationContentError"] {{
    background: linear-gradient(145deg, rgba(239,68,68,0.10), rgba(239,68,68,0.04)) !important;
    border-color: rgba(239,68,68,0.40) !important;
    border-left: 3px solid {RED} !important;
}}

/* ══════════════════════════════════════════════════════════════════════════
   EXPANDERS — sleek pill-button feel when collapsed, card when open
   ══════════════════════════════════════════════════════════════════════════ */
details[data-testid="stExpander"],
[data-testid="stExpander"] {{
    background: linear-gradient(145deg, {BG_CARD} 0%, {BG_SURFACE} 100%) !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease,
                transform 0.2s ease !important;
}}
details[data-testid="stExpander"]:hover,
[data-testid="stExpander"]:hover {{
    border-color: rgba(6,214,160,0.32) !important;
    box-shadow: 0 0 0 1px rgba(6,214,160,0.12),
                0 6px 20px rgba(6,214,160,0.08) !important;
}}
details[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary {{
    font-family: 'Inter', sans-serif !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    color: {TEXT_SECONDARY} !important;
    letter-spacing: 0.02em !important;
    padding: 9px 14px !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    transition: color 0.18s ease, background 0.18s ease !important;
}}
details[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:hover {{
    background: rgba(6,214,160,0.06) !important;
    color: {CYAN} !important;
}}
details[data-testid="stExpander"][open] summary,
[data-testid="stExpander"][open] summary {{
    border-bottom: 1px solid {BORDER} !important;
    background: linear-gradient(90deg, rgba(6,214,160,0.08), transparent 70%) !important;
    color: {CYAN} !important;
}}
/* Brand gradient hairline at the top of every expander — matches the
   rest of the card language (metric cards, portfolio cards, etc.) */
details[data-testid="stExpander"]::before,
[data-testid="stExpander"]::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1.5px;
    background: {BRAND_GRAD};
    opacity: 0.55;
    pointer-events: none;
}}
details[data-testid="stExpander"],
[data-testid="stExpander"] {{
    position: relative !important;
}}
/* Expander chevron — cyan, rotates smoothly on open */
[data-testid="stExpander"] svg {{
    color: {CYAN} !important;
    transition: transform 0.22s cubic-bezier(0.4,0,0.2,1) !important;
}}

/* ══════════════════════════════════════════════════════════════════════════
   DOWNLOAD / FORM SUBMIT BUTTONS — match quick-action style with brand hover
   ══════════════════════════════════════════════════════════════════════════ */
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {{
    background: linear-gradient(145deg, {BG_CARD}, {BG_SURFACE}) !important;
    border: 1px solid {BORDER_ACCENT} !important;
    color: {TEXT_PRIMARY} !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    padding: 10px 18px !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
}}
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {{
    border-color: {CYAN} !important;
    color: {CYAN} !important;
    background: {BG_ELEVATED} !important;
    box-shadow: 0 4px 16px rgba(6,214,160,0.18) !important;
    transform: translateY(-1px) !important;
}}

/* ══════════════════════════════════════════════════════════════════════════
   PROGRESS BARS
   ══════════════════════════════════════════════════════════════════════════ */
.stProgress > div > div {{
    background: {BORDER} !important;
    border-radius: 100px !important;
}}
.stProgress > div > div > div > div {{
    background: {BRAND_GRAD} !important;
    box-shadow: 0 0 12px rgba(6,214,160,0.35) !important;
    border-radius: 100px !important;
}}

/* ══════════════════════════════════════════════════════════════════════════
   FILE UPLOADER
   ══════════════════════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] section {{
    background: linear-gradient(145deg, {BG_CARD}, {BG_SURFACE}) !important;
    border: 1px dashed {BORDER_ACCENT} !important;
    border-radius: 12px !important;
    transition: all 0.25s ease !important;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color: {CYAN} !important;
    background: {BG_ELEVATED} !important;
    box-shadow: 0 0 0 3px rgba(6,214,160,0.10) !important;
}}

/* ══════════════════════════════════════════════════════════════════════════
   METRIC (st.metric) — brand deltas + polished value type
   ══════════════════════════════════════════════════════════════════════════ */
[data-testid="stMetricValue"] {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    color: {TEXT_PRIMARY} !important;
}}
[data-testid="stMetricDelta"] {{
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.02em !important;
}}
[data-testid="stMetricDelta"] svg {{
    height: 14px !important; width: 14px !important;
}}

/* ══════════════════════════════════════════════════════════════════════════
   TYPOGRAPHY — main content headings
   ══════════════════════════════════════════════════════════════════════════ */
.main h1, .main h2, .main h3, .main h4 {{
    font-family: 'Inter', sans-serif !important;
    color: {TEXT_PRIMARY} !important;
    letter-spacing: -0.02em !important;
}}
.main h2 {{ font-weight: 800 !important; }}
.main h3 {{ font-weight: 700 !important; }}
.main h4 {{ font-weight: 700 !important; font-size: 0.95rem !important; color: {TEXT_SECONDARY} !important; }}

/* ══════════════════════════════════════════════════════════════════════════
   SECTION HEAD — reusable eyebrow label with brand accent bar
   Use: <div class='sec-head'><span class='sec-bar'></span>Label</div>
   ══════════════════════════════════════════════════════════════════════════ */
.sec-head {{
    display: flex; align-items: center; gap: 10px;
    color: {CYAN};
    font-size: 0.65rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin: 4px 0 12px;
}}
.sec-head .sec-bar {{
    display: inline-block;
    width: 18px; height: 2px; border-radius: 2px;
    background: {BRAND_GRAD};
    box-shadow: 0 0 8px rgba(6,214,160,0.5);
}}

/* ══════════════════════════════════════════════════════════════════════════
   TABS — elevate active state with gradient underline
   ══════════════════════════════════════════════════════════════════════════ */
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(32,128,229,0.18) 0%, rgba(6,214,160,0.12) 100%) !important;
    border-bottom: 2px solid transparent !important;
    position: relative !important;
}}
.stTabs [aria-selected="true"]::after {{
    content: ''; position: absolute; left: 10%; right: 10%; bottom: -1px;
    height: 2px; border-radius: 2px;
    background: {BRAND_GRAD};
    box-shadow: 0 0 10px rgba(6,214,160,0.5);
}}

/* ══════════════════════════════════════════════════════════════════════════
   DATAFRAME — brand hover + subtle row separators
   ══════════════════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {{
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}}

/* ══════════════════════════════════════════════════════════════════════════
   LINKS in markdown/body — brand cyan
   ══════════════════════════════════════════════════════════════════════════ */
.main a, .stMarkdown a {{
    color: {CYAN} !important;
    text-decoration: none !important;
    border-bottom: 1px dotted rgba(6,214,160,0.4);
    transition: all 0.2s ease !important;
}}
.main a:hover, .stMarkdown a:hover {{
    color: {BLUE} !important;
    border-bottom-color: {BLUE};
}}

/* ══════════════════════════════════════════════════════════════════════════
   SPINNER — brand color
   ══════════════════════════════════════════════════════════════════════════ */
.stSpinner > div > div {{
    border-top-color: {CYAN} !important;
    border-right-color: {BLUE} !important;
}}

/* ══════════════════════════════════════════════════════════════════════════
   CODE blocks
   ══════════════════════════════════════════════════════════════════════════ */
code, pre {{
    font-family: 'SF Mono', 'Monaco', 'Menlo', monospace !important;
}}
.stMarkdown code {{
    background: rgba(6,214,160,0.08) !important;
    color: {CYAN} !important;
    padding: 2px 6px !important;
    border-radius: 5px !important;
    font-size: 0.85em !important;
    border: 1px solid rgba(6,214,160,0.2) !important;
}}

/* ══════════════════════════════════════════════════════════════════════════
   CAPTIONS + small text
   ══════════════════════════════════════════════════════════════════════════ */
[data-testid="stCaptionContainer"] {{
    color: {TEXT_MUTED} !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.02em !important;
}}

/* ══════════════════════════════════════════════════════════════════════════
   SEGMENTED CONTROL — st.segmented_control, brand-styled
   ══════════════════════════════════════════════════════════════════════════ */
div[data-testid="stSegmentedControl"] {{
    display: inline-flex !important;
    justify-content: flex-end !important;
    width: 100% !important;
}}
div[data-testid="stSegmentedControl"] > div[role="group"] {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    padding: 3px !important;
    gap: 2px !important;
    display: inline-flex !important;
    flex-wrap: nowrap !important;
}}
div[data-testid="stSegmentedControl"] button {{
    background: transparent !important;
    border: 1px solid transparent !important;
    color: {TEXT_MUTED} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    padding: 5px 16px !important;
    min-height: 0 !important;
    height: auto !important;
    border-radius: 7px !important;
    box-shadow: none !important;
    transition: all 0.18s ease !important;
}}
div[data-testid="stSegmentedControl"] button:hover {{
    background: rgba(255,255,255,0.04) !important;
    color: {TEXT_PRIMARY} !important;
    transform: none !important;
    border-color: transparent !important;
}}
div[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"],
div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
div[data-testid="stSegmentedControl"] button[data-selected="true"] {{
    background: {BRAND_GRAD} !important;
    color: {TEXT_PRIMARY} !important;
    box-shadow: 0 2px 10px rgba(32,128,229,0.30) !important;
    border-color: transparent !important;
}}

/* ══════════════════════════════════════════════════════════════════════════
   BORDERED CONTAINER — st.container(border=True) rendered as a premium card
   ══════════════════════════════════════════════════════════════════════════ */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: linear-gradient(145deg, {BG_CARD}, {BG_SURFACE}) !important;
    border: 1px solid {BORDER} !important;
    border-radius: 14px !important;
    position: relative !important;
    overflow: hidden !important;
    padding: 18px 18px 12px !important;
    transition: border-color 0.22s ease, box-shadow 0.22s ease, transform 0.22s ease !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"]::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1.5px;
    background: {BRAND_GRAD}; opacity: 0.65; z-index: 2;
    pointer-events: none;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    border-color: {BORDER_ACCENT} !important;
    box-shadow: 0 10px 32px rgba(0,0,0,0.28),
                0 0 0 1px rgba(32,128,229,0.12) !important;
}}
/* Trim default plotly margins when chart is nested in a bordered card */
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stPlotlyChart"] {{
    margin: 6px -6px -4px !important;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _rc(r):   return CHART_UP if r >= 0 else CHART_DOWN
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


# ── Feature name humanizer ───────────────────────────────────────────────────
# Maps ML feature names (raw column strings) to plain-English labels for
# display in the "Why this recommendation" section. Category icons help
# group them at a glance.
#
# Each entry: raw_name → (display_label, icon_emoji, one_line_tooltip)
FEATURE_LABELS = {
    # Trend / momentum
    "ret_63d":         ("3-month momentum",       "📈", "How the stock has moved over 63 trading days"),
    "ret_21d":         ("1-month momentum",       "📈", "Return over the last 21 trading days"),
    "ret_5d":          ("1-week momentum",        "📈", "Return over the last 5 trading days"),
    "mom_accel_21":    ("Momentum acceleration",  "📈", "Is 1-month momentum speeding up or slowing down?"),
    "mom_accel_5":     ("Short-term acceleration","📈", "Is 1-week momentum speeding up?"),
    "autocorr_1d":     ("Day-over-day trend",     "📈", "Does yesterday's direction tend to continue today?"),
    "autocorr_5d":     ("5-day trend consistency","📈", "Does recent direction persist day-to-day?"),
    "autocorr_10d":    ("Trend consistency",      "📈", "Does the price trend repeat itself over 10 days?"),
    "autocorr_20d":    ("20-day trend consistency","📈","Does the price trend repeat itself over 20 days?"),
    "ret_1d":          ("Yesterday's return",     "📈", "How the stock moved on the previous day"),
    "ret_3d":          ("3-day momentum",         "📈", "Return over the last 3 trading days"),
    "ret_10d":         ("2-week momentum",        "📈", "Return over the last 10 trading days"),
    "roc_30d":         ("Rate of change (30d)",   "📈", "Price change vs 30 trading days ago"),
    "roc_60d":         ("Rate of change (60d)",   "📈", "Price change vs 60 trading days ago"),
    # Moving averages
    "ma50_200_xover":  ("50d vs 200d MA crossover","📊", "Golden cross / death cross signal"),
    "ma20_50_xover":   ("20d vs 50d MA crossover","📊", "Short-term trend crossover"),
    "ma5_20_xover":    ("5d vs 20d MA crossover","📊", "Very short-term trend crossover"),
    "ma50_ratio":      ("Price vs 50d average",   "📊", "How far price is from its 50-day average"),
    "ma100_ratio":     ("Price vs 100d average",  "📊", "How far price is from its 100-day average"),
    "ma200_ratio":     ("Price vs 200d average",  "📊", "How far price is from its 200-day average"),
    "spy_ma50_ratio":  ("S&P 500 trend (50d)",   "📊", "Where the S&P 500 sits vs its 50-day average"),
    "spy_ma200_ratio": ("S&P 500 trend (200d)",  "📊", "Where the S&P 500 sits vs its 200-day average"),
    # Technicals
    "rsi_14":          ("RSI (14d)",              "🎚", "Overbought/oversold momentum (0-100)"),
    "rsi_28":          ("RSI (28d)",              "🎚", "Longer-period overbought/oversold"),
    "macd_norm":       ("MACD signal",            "〰",  "Momentum cross vs trend"),
    "macd_sig":        ("MACD signal line",       "〰",  "MACD trend indicator"),
    "macd_hist":       ("MACD histogram",         "〰",  "MACD momentum strength"),
    "stoch_k":         ("Stochastic %K",          "〰",  "Where price sits in its recent range"),
    "stoch_d":         ("Stochastic %D",          "〰",  "Smoothed stochastic oscillator"),
    "williams_r":      ("Williams %R",            "〰",  "Overbought/oversold swing indicator"),
    "cci":             ("Commodity Channel Index", "〰", "Momentum deviation from mean"),
    "bb_pct_b":        ("Bollinger %B",           "🎚", "Where price sits in its volatility band"),
    "bb_bw":           ("Bollinger bandwidth",    "🎚", "How wide the volatility band is"),
    "ichi_price_vs_cloud":("Ichimoku cloud pos.","☁",  "Price position vs the Ichimoku cloud"),
    "ichi_cloud_width":("Ichimoku cloud width",   "☁",  "Width of the Ichimoku cloud = trend strength"),
    "ichi_tk_ratio":   ("Ichimoku lines",         "☁",  "Tenkan/Kijun crossover signal"),
    # Volatility
    "vol_5d":          ("1-week volatility",      "⚡", "How much the price has wobbled over 5 days"),
    "vol_21d":         ("1-month volatility",     "⚡", "How much the price has wobbled over 21 days"),
    "vol_63d":         ("3-month volatility",     "⚡", "How much the price has wobbled over 63 days"),
    "vol_ratio_20":    ("Vol vs 20-day avg",      "⚡", "Is volatility elevated or calm right now?"),
    "vol_ratio_5":     ("Short-term vol change",  "⚡", "5-day volatility vs its typical level"),
    "vol_regime":      ("Volatility regime",      "⚡", "Calm / moderate / stressed classification"),
    "vol_trend":       ("Volatility trend",       "⚡", "Is volatility rising or falling?"),
    "atr_ratio":       ("ATR (avg daily range)",  "⚡", "Typical daily move in $"),
    "var_ratio_10d":   ("10d variance ratio",     "⚡", "Trend vs mean-reversion signal"),
    "var_ratio_21d":   ("21d variance ratio",     "⚡", "Longer-period trend-vs-noise signal"),
    # Volume / flow
    "rvol_10":         ("Relative volume (10d)",  "🔊", "Today's volume vs typical"),
    "rvol_50":         ("Relative volume (50d)",  "🔊", "Today's volume vs 50-day average"),
    "obv_trend":       ("On-balance volume",      "🔊", "Is volume flowing into or out of the stock?"),
    "vpt_trend":       ("Volume-price trend",     "🔊", "Volume-weighted price direction"),
    "candle_body_ratio":("Candle body size",      "🕯", "How decisive recent price bars have been"),
    "upper_wick_ratio":("Upper wick size",        "🕯", "Seller pressure intraday"),
    "lower_wick_ratio":("Lower wick size",        "🕯", "Buyer pressure intraday"),
    "gap_up":          ("Gap-up events",          "⤴", "Frequency of overnight upward gaps"),
    "gap_down":        ("Gap-down events",        "⤵", "Frequency of overnight downward gaps"),
    # Relative strength
    "rs_spy_20":       ("Strength vs S&P (20d)",  "⚖", "Out/underperforming the market"),
    "sector_rs_21d":   ("Strength vs sector (21d)","⚖", "Out/underperforming its sector"),
    "sector_rs_63d":   ("Strength vs sector (63d)","⚖", "3-month sector-relative strength"),
    "alpha_63d":       ("Alpha vs market",        "⚖", "Excess return vs market over 63 days"),
    "beta_63d":        ("Beta vs market",         "⚖", "How much the stock moves with the market"),
    "pos_52w_high":    ("Distance from 52w high", "📍", "How close to the year's high"),
    "pos_52w_low":     ("Distance from 52w low",  "📍", "How close to the year's low"),
    "dist_low_20d":    ("Distance from 20d low",  "📍", "Room between price and recent low"),
    "dist_high_50d":   ("Distance from 50d high", "📍", "Room between price and recent high"),
    "hl_range_norm":   ("Recent H/L range",       "📍", "How wide the recent high-low range is"),
    "hl_range_5d_avg": ("5d H/L range",           "📍", "Average 5-day trading range"),
    "pivot_dist":      ("Distance to pivot",      "📍", "How far from the classic pivot price"),
    # Regime
    "regime_vix_norm":    ("VIX regime",          "🌡", "Overall market fear level"),
    "regime_vol_ratio":   ("Volatility regime",   "🌡", "Stock vol vs its own typical"),
    "regime_spy_vs_200ma":("S&P 500 regime",     "🌡", "Broad market trend state"),
    "regime_vs_200ma":    ("Long-term trend state","🌡", "Above or below the 200-day moving average"),
    "regime_vs_50ma":     ("Medium-term trend",  "🌡", "Above or below the 50-day MA"),
    "regime_mom63":       ("3-month market mom.", "🌡", "Market-wide momentum signal"),
    # Macro / calendar
    "macro_days_to_fomc": ("Days to next FOMC",  "🏛", "Time until the next Fed rate decision"),
    "macro_days_to_cpi":  ("Days to next CPI",   "📈", "Time until the next inflation report"),
    "macro_days_to_nfp":  ("Days to next jobs rpt","👥", "Time until the next jobs report"),
    "macro_fomc_week":    ("FOMC week",          "🏛", "Within 5 days of a Fed decision"),
    "macro_cpi_week":     ("CPI week",           "📈", "Within 5 days of an inflation print"),
    "macro_nfp_week":     ("Jobs report week",   "👥", "Within 5 days of a jobs report"),
    # Sentiment / market breadth
    "sent_vix_momentum_5d":    ("VIX 5d momentum",       "😨", "Is fear rising or falling?"),
    "sent_sector_dispersion_21d":("Sector rotation",     "🔄", "How much sectors are diverging"),
    "sent_avg_sector_momentum":("Avg sector momentum",   "🔄", "How all sectors are moving on average"),
    "sent_dollar_momentum_21d":("Dollar index momentum", "💵", "Is the dollar strengthening?"),
    "sent_treasury_10y_level":("10Y Treasury yield",     "🏛", "Current 10-year yield level"),
    "sent_treasury_10y_momentum_21d":("10Y yield trend", "🏛", "Is the 10-year yield rising?"),
    "sent_yield_curve_spread":("Yield curve slope",      "🏛", "10y–3m slope — recession indicator"),
    "sent_put_call_ratio":    ("Put/call ratio",         "😨", "Bearish vs bullish options positioning"),
    "sent_put_call_ma20":     ("Put/call 20d avg",       "😨", "Smoothed options sentiment"),
    "sent_oil_level_z":       ("Oil regime (z-score)",   "🛢", "Is oil unusually high or low?"),
    "sent_oil_momentum_21d":  ("Oil momentum (21d)",     "🛢", "Recent oil price direction"),
    "sent_oil_vol":           ("Oil volatility (OVX)",   "🛢", "Uncertainty in oil prices"),
    "sent_gold_momentum_21d": ("Gold momentum (21d)",    "🪙", "Flight-to-safety signal"),
    "sent_copper_momentum_21d":("Copper momentum (21d)", "🪙", "Global growth signal"),
    "sent_credit_stress":     ("Credit market stress",   "📉", "HY vs IG bond spread proxy"),
    "sent_fear_greed":        ("Fear & greed index",     "😨", "CNN fear/greed sentiment index"),
    "sent_fear_greed_momentum":("Fear/greed momentum",   "😨", "Is fear rising or easing?"),
    "sent_news_mean":         ("News sentiment",         "📰", "Tone of recent news"),
    "sent_news_positive_ratio":("Positive news ratio",   "📰", "Share of positive-tone articles"),
    "sent_news_volume":       ("News volume",            "📰", "How much the stock is being covered"),
    "sent_news_trend":        ("News sentiment trend",   "📰", "Is coverage getting more positive?"),
    "sent_news_volume_trend": ("News volume trend",      "📰", "Is coverage heating up?"),
    # Earnings
    "days_to_earnings":       ("Days to earnings",       "📊", "Time until next earnings report"),
    "earnings_proximity":     ("Earnings proximity",     "📊", "How close we are to next earnings"),
    "earnings_beat_rate":     ("YTD beat rate",          "📊", "How often they've beaten estimates"),
    "last_earnings_beat":     ("Last earnings result",   "📊", "Whether they beat or missed last time"),
    "earnings_surprise_pct":  ("Last earnings surprise", "📊", "Size of the most recent EPS surprise"),
    # Options
    "options_atm_iv":         ("Options implied vol",    "🎲", "Market's priced-in volatility"),
    "options_put_call_ratio": ("Stock put/call ratio",   "🎲", "Options positioning for this name"),
    "options_call_put_vol":   ("Call/put volume ratio",  "🎲", "Recent options flow"),
    "options_iv_skew":        ("IV skew",                "🎲", "Put vs call implied vol — tail risk"),
    "options_max_pain_diff":  ("Distance to max pain",   "🎲", "Price vs max-pain level"),
    # Calendar / seasonal
    "month":                  ("Calendar month",         "📅", "Seasonal effect"),
    "month_sin":              ("Month cycle (sin)",      "📅", "Seasonal cycle, sine component"),
    "month_cos":              ("Month cycle (cos)",      "📅", "Seasonal cycle, cosine component"),
    "quarter":                ("Calendar quarter",       "📅", "Quarterly seasonal effect"),
    "day_of_week":            ("Day of week",            "📅", "Day-of-week seasonal effect"),
    "dow_sin":                ("Weekday cycle (sin)",    "📅", "Weekday cycle, sine component"),
    "dow_cos":                ("Weekday cycle (cos)",    "📅", "Weekday cycle, cosine component"),
    # Analyst / fundamentals
    "analyst_sentiment":      ("Analyst sentiment",      "👥", "Wall Street's avg recommendation"),
    "analyst_upside":         ("Analyst upside target",  "👥", "Avg analyst price target vs current"),
    "analyst_range_pct":      ("Analyst spread",         "👥", "How much analysts disagree"),
    "short_interest":         ("Short interest",         "📉", "Share of float sold short"),
}

def humanize_feature(raw_name: str) -> tuple[str, str, str]:
    """Return (display_label, icon, tooltip) for a raw ML feature name.
    Falls back to a title-cased version if we haven't explicitly mapped it."""
    if raw_name in FEATURE_LABELS:
        return FEATURE_LABELS[raw_name]
    # Graceful fallback: take the snake_case, title-case it
    pretty = raw_name.replace("_", " ").title()
    return (pretty, "•", "")


def info_icon(tooltip_text: str) -> str:
    """Creates an info icon with hover tooltip."""
    safe = (tooltip_text
            .replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))
    return f"<span class='tooltip-icon' data-tip=\"{safe}\">ℹ</span>"


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

    # Cost / credit — label it clearly so it's not a mystery number
    cost_str = strat.get("estimated_cost", "")
    if "$" in cost_str:
        cost_dollar = cost_str.split("/")[0].strip()   # e.g. "$1,982"
    else:
        cost_dollar = cost_str

    # Prepend a plain-English label based on whether it's a debit or credit trade
    credit_strategies = {"Bull Call Spread", "Bear Put Spread", "Long Call", "Long Put",
                         "Long Straddle", "Long Strangle"}
    if name in credit_strategies:
        cost_short = f"Est. cost to open · {cost_dollar}" if cost_dollar else ""
    else:
        cost_short = f"Est. premium received · {cost_dollar}" if cost_dollar else ""

    return (line, dc, cost_short)

TIPS = {
    "Confidence":  "How sure the model is about this call, on a 0–100% scale. It blends three things: how often the model has been right on similar past setups, how tightly the 16 base models cluster around the same answer, and how calm or choppy the market is right now. 70%+ = strong conviction, 55–70% = decent edge, under 55% = coin-flip territory.",
    "Agreement":   "How many of the 16 base models vote the same direction (up or down). 100% means every model agrees — strongest consensus. Around 50% means the committee is split and the signal is weak. High agreement + high confidence is the best setup to act on.",
    "ValAcc":      "The model was trained on older data and then tested on recent data it had never seen. This number is the % of times it correctly called the direction (up or down) on that unseen data. 50% = random guessing. 55%+ is a real edge. 60%+ is strong. It's the honest report card, not a cherry-picked backtest.",
    "Dir Acc":     "Directional accuracy: % correct up/down predictions on test data the model never saw. 55%+ is meaningful.",
    "Sharpe":      "Risk-adjusted return. >1.0 = good, >2.0 = excellent. Accounts for drawdowns.",
    "Win Rate":    "Of buy signals, what % actually went up. 55%+ indicates edge.",
    "PF":          "Profit factor = gross profit / gross loss. >1.5 = meaningful, >2.0 = strong edge.",
    "Max DD":      "Largest peak-to-trough loss. Your worst case if you followed every signal.",
    "RSI":         "Momentum 0-100. >70 overbought (reversal risk). <30 oversold (bounce potential).",
    "MACD":        "Trend momentum. Bullish = strengthening, Bearish = weakening. Divergence shows weakness.",
    "Beta":        "Volatility vs S&P 500. 1.5 = 50% more volatile. Higher beta = larger swings.",
    "IV":          "Implied volatility: 21d historical vol × 1.2. Higher = options more expensive.",
    "WF":          "Walk-forward backtest: trained on 2yr windows, tested on next 3mo, repeated. Tests all market regimes.",
    "Regime":      "Current market state (Bull/Bear/Sideways). Affects confidence levels.",
    "RR":          "Risk/Reward = (target - entry) / (entry - stop). 3:1 or higher is ideal.",
    "ML":          "ML = Machine Learning. A computer program that studies thousands of past stock charts, finds patterns humans can't easily see, and uses those patterns to predict future price moves. Here we use 16 different ML models (XGBoost, LightGBM, RandomForest, and others) and combine their votes — like a committee of analysts instead of a single opinion.",
    "ATR":         "ATR = Average True Range. A simple volatility measure that tells you how much a stock typically moves in a day. We look at the last 14 days and average the daily ranges (high minus low, accounting for overnight gaps). Example: ATR = $3 means this stock usually swings about $3 per day. We use 1.5× ATR to set a stop loss far enough away that normal daily noise won't hit it.",
    "RRSection":   "For each horizon, we compare how much you could make vs. how much you'd lose if the trade goes wrong. Reward = ML (Machine Learning) target price minus the best entry price. Risk = worst entry price minus the ATR-based stop loss. ATR (Average True Range) is a volatility measure — roughly how much the stock moves in a typical day. Ratio = reward ÷ risk. 3:1 means you make $3 for every $1 risked.",
    "RRRatio":     "Reward-to-Risk ratio. How many dollars of potential gain per dollar of potential loss. Under 1:1 = you risk more than you can win (POOR). 1–2:1 = MODERATE. 2–3:1 = FAVORABLE. 3–5:1 = VERY GOOD. 5:1+ = EXCELLENT. Long-horizon trades tend to have higher ratios because the target is further away.",
    "RRQuality":   "Plain-English grade of the R:R ratio. POOR (<1:1), MODERATE (1–2:1), FAVORABLE (2–3:1), VERY GOOD (3–5:1), EXCELLENT (5:1+). Flagged ⚠ Extreme when the predicted move is unusually large (>50%) — treat the ratio with extra skepticism.",
    "TGT":         "TGT = Target price. This is where the ML (Machine Learning) model predicts the stock will be at the end of this horizon. The +% next to it is how much you'd gain if you bought at the best entry price and sold at the target. This is the 'reward' side of the trade.",
    "STP":         "STP = Stop-loss price. The price where you'd exit to cut your losses if the trade goes against you. We calculate it as entry zone minus 1.5× ATR (ATR = Average True Range, a measure of how much the stock normally moves per day). Using ATR means the stop is far enough away that ordinary daily noise won't trigger it. The same stop is used across all horizons because it's based on today's volatility, not on the horizon length. This is the 'risk' side of the trade.",
    "RewardPct":   "Upside % if the ML (Machine Learning) target is hit, measured from the best entry price. Formula: (target − entry_low) ÷ entry_low × 100.",
    "RiskPct":     "Downside % if the stop is hit, measured from the worst entry price. Formula: (entry_high − stop) ÷ entry_high × 100. Stays constant across horizons because the stop is ATR-based (ATR = Average True Range, a daily volatility measure).",
    "Horizon":     "How far out the prediction looks. 3 Day = next 3 trading days. 1 Week = 5 trading days. 1 Month = ~21 trading days. 1 Quarter = ~63 trading days. 1 Year = ~252 trading days. Longer horizons typically show larger predicted moves and better R:R, but with more uncertainty.",
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


# ── Freemium wall ────────────────────────────────────────────────────────────
# For now, subscription state is stored in session_state.
# Later this connects to Stripe/auth backend.
if "is_subscribed" not in st.session_state:
    st.session_state.is_subscribed = True  # default True during development

FREE_PAGES = {"Dashboard", "Track Record"}  # always accessible
PRO_PAGES  = {"Analyze", "Watchlist", "Portfolio"}  # require subscription


def _render_paywall(feature_name: str = "this feature"):
    """Show a blurred upgrade CTA for non-subscribers."""
    st.markdown(f"""
    <div style='position:relative;padding:80px 20px;text-align:center'>
        <div style='position:absolute;inset:0;background:linear-gradient(180deg,transparent 0%,{BG_PRIMARY} 100%);
                    pointer-events:none;z-index:1'></div>
        <div style='position:relative;z-index:2;max-width:440px;margin:0 auto'>
            <div style='width:64px;height:64px;margin:0 auto 20px;background:{BRAND_GRAD};
                        border-radius:16px;display:flex;align-items:center;justify-content:center;
                        font-size:1.6rem;box-shadow:0 8px 32px rgba(6,214,160,0.2)'>🔒</div>
            <div style='color:{TEXT_PRIMARY};font-size:1.4rem;font-weight:800;margin-bottom:10px'>
                Unlock {feature_name}</div>
            <div style='color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.7;margin-bottom:24px'>
                Get full access to Prediqt's ML predictions, real-time analysis,
                portfolio tracking, and the complete intelligence suite.</div>
            <div style='display:inline-block;background:{BRAND_GRAD};color:white;padding:12px 32px;
                        border-radius:10px;font-weight:700;font-size:0.95rem;cursor:pointer;
                        box-shadow:0 4px 20px rgba(6,214,160,0.25)'>
                Upgrade to Prediqt Pro</div>
            <div style='color:{TEXT_MUTED};font-size:0.7rem;margin-top:12px'>
                Starting at $29/month · Cancel anytime</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── Brand logo ──
    import base64, os as _os, pathlib as _pathlib

    def _load_logo(name: str, fallback_names: list = None) -> str:
        """Load a logo from assets/ — tries png/jpg/webp/svg in order."""
        _assets = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "assets")
        names = [name] + (fallback_names or [])
        for n in names:
            for ext in [".png", ".jpg", ".jpeg", ".webp", ".svg"]:
                p = _os.path.join(_assets, n + ext)
                if _os.path.exists(p):
                    mime = {"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg",
                            "webp":"image/webp","svg":"image/svg+xml"}[ext.lstrip(".")]
                    mode = "r" if ext == ".svg" else "rb"
                    with open(p, mode) as f:
                        data = f.read()
                    raw = data.encode() if isinstance(data, str) else data
                    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"
        return ""

    # Prefer the tight-cropped transparent PNG (matches landing page);
    # fall back to older logo files if the transparent one isn't present.
    _full_sidebar_src = _load_logo(
        "logo_full_transparent",
        ["logo_full", "logo", "prediqt_logo"],
    )
    _icon_src = _load_logo(
        "logo_icon_cropped",
        ["logo_icon", "icon", "logo-icon"],
    )

    if _full_sidebar_src:
        _sidebar_logo_html = (
            f"<img src='{_full_sidebar_src}' "
            f"style='width:100%;max-width:220px;height:auto;display:block;margin:0 auto;"
            f"filter:drop-shadow(0 4px 18px rgba(32,128,229,0.25))'>"
        )
    elif _icon_src:
        _sidebar_logo_html = (
            f"<img src='{_icon_src}' "
            f"style='width:120px;height:auto;display:block;margin:0 auto;"
            f"filter:drop-shadow(0 4px 14px rgba(32,128,229,0.25))'>"
        )
    else:
        _sidebar_logo_html = ""

    st.markdown(f"""
    <div style='text-align:center;padding:12px 0 18px'>
        {_sidebar_logo_html}
    </div>
    """, unsafe_allow_html=True)

    # ── Page Navigation ──────────────────────────────────────────────────────
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"

    _nav_pages = ["Dashboard", "Analyze", "Track Record", "Watchlist", "Portfolio"]

    # Branded line-icon SVGs — rendered via CSS mask-image so the icon color
    # automatically follows the label's text color (and switches to the brand
    # gradient on active). Keep the SVGs compact — they're embedded as data URIs.
    _NAV_SVG = {
        "Dashboard": (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
            "stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
            "<rect x='3' y='3' width='7' height='7' rx='1.5'/>"
            "<rect x='14' y='3' width='7' height='7' rx='1.5'/>"
            "<rect x='3' y='14' width='7' height='7' rx='1.5'/>"
            "<rect x='14' y='14' width='7' height='7' rx='1.5'/></svg>"
        ),
        "Analyze": (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
            "stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
            "<polyline points='3 17 9 11 13 15 21 7'/>"
            "<polyline points='15 7 21 7 21 13'/></svg>"
        ),
        "Track Record": (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
            "stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
            "<path d='M6 3h12v5a6 6 0 0 1-12 0V3z'/>"
            "<path d='M6 5H3a3 3 0 0 0 3 3'/><path d='M18 5h3a3 3 0 0 1-3 3'/>"
            "<path d='M10 21h4'/><path d='M12 14v7'/></svg>"
        ),
        "Watchlist": (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
            "stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
            "<polygon points='12 2 15 9 22 9.5 17 14.5 18.5 21.5 12 18 5.5 21.5 7 14.5 2 9.5 9 9'/></svg>"
        ),
        "Portfolio": (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
            "stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
            "<path d='M21 12a9 9 0 1 1-9-9'/>"
            "<path d='M12 3a9 9 0 0 1 9 9h-9V3z'/></svg>"
        ),
    }
    _NAV_DATA_URIS = {
        k: "data:image/svg+xml;base64," + base64.b64encode(v.encode()).decode()
        for k, v in _NAV_SVG.items()
    }

    # Clamp current_page to valid nav pages (Stock Detail etc. are handled separately)
    if st.session_state.current_page not in _nav_pages:
        _stashed_page = st.session_state.current_page
    else:
        _stashed_page = None

    # Build nth-child selectors for per-option SVG masks — scoped to sidebar only
    _nav_icon_css_blocks = []
    _NAV_SEL = "section[data-testid='stSidebar'] div[data-testid='stRadio'] > div > label"
    for _i, _page in enumerate(_nav_pages, start=1):
        _uri = _NAV_DATA_URIS[_page]
        _nav_icon_css_blocks.append(
            f"{_NAV_SEL}:nth-child({_i})::after {{"
            f"  -webkit-mask-image: url('{_uri}');"
            f"  mask-image: url('{_uri}');"
            f"}}"
        )
    _nav_icon_css = "\n".join(_nav_icon_css_blocks)

    st.markdown(f"""
    <style>
    /* ── Sidebar nav: scoped to sidebar so other radios in the app aren't affected ─ */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{
        gap: 3px !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div > label {{
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        padding: 10px 14px 10px 42px !important;   /* room for left SVG icon */
        margin: 1px 0 !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        color: {TEXT_SECONDARY} !important;
        position: relative !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div > label:hover {{
        background: rgba(32,128,229,0.06) !important;
        color: {TEXT_PRIMARY} !important;
        border-color: {BORDER} !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div > label[data-checked="true"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div > label:has(input:checked) {{
        background: linear-gradient(135deg, rgba(32,128,229,0.20) 0%, rgba(6,214,160,0.12) 100%) !important;
        color: {TEXT_PRIMARY} !important;
        border-color: rgba(32,128,229,0.40) !important;
        box-shadow: 0 4px 18px rgba(32,128,229,0.18) !important;
    }}
    /* Gradient accent bar on left of active item */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div > label[data-checked="true"]::before,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div > label:has(input:checked)::before {{
        content: '';
        position: absolute;
        left: 0; top: 8px; bottom: 8px;
        width: 3px;
        border-radius: 0 3px 3px 0;
        background: {BRAND_GRAD};
        box-shadow: 0 0 10px rgba(6,214,160,0.5);
    }}
    /* Hide Streamlit's default radio dot */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div > label > div:first-child {{
        display: none !important;
    }}

    /* ── SVG icon via mask-image: color follows currentColor ──────────────── */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div > label::after {{
        content: '';
        position: absolute;
        left: 14px; top: 50%;
        width: 18px; height: 18px;
        transform: translateY(-50%);
        background-color: currentColor;
        -webkit-mask-size: contain;
        mask-size: contain;
        -webkit-mask-repeat: no-repeat;
        mask-repeat: no-repeat;
        -webkit-mask-position: center;
        mask-position: center;
        opacity: 0.75;
        transition: opacity 0.2s ease, background 0.2s ease, filter 0.2s ease;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div > label:hover::after {{
        opacity: 1;
    }}
    /* Active state: brand gradient filled icon with a soft glow */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div > label[data-checked="true"]::after,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div > label:has(input:checked)::after {{
        background-color: transparent;
        background-image: {BRAND_GRAD};
        opacity: 1;
        filter: drop-shadow(0 0 6px rgba(6,214,160,0.45));
    }}

    {_nav_icon_css}
    </style>
    """, unsafe_allow_html=True)

    def _nav_label(x):
        """Plain page name — icon is rendered via CSS ::after mask."""
        return x

    _safe_index = _nav_pages.index(st.session_state.current_page) if st.session_state.current_page in _nav_pages else 0
    _selected_page = st.radio(
        "Navigation",
        _nav_pages,
        index=_safe_index,
        format_func=_nav_label,
        label_visibility="collapsed",
        key="nav_radio",
    )
    if _stashed_page is None and _selected_page != st.session_state.current_page:
        st.session_state.current_page = _selected_page
        st.rerun()

    st.markdown(f"<div style='height:8px'></div>", unsafe_allow_html=True)
    st.divider()

    # Initialize session state for watchlist selections
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = None
    if "auto_analyze" not in st.session_state:
        st.session_state.auto_analyze = False

    # ── Analyze page controls ────────────────────────────────────────────────
    # Hardcoded for maximum accuracy — always use full history + all models
    data_period = "10y"
    run_bt = True
    use_ctx = True
    use_shap = HAS_SHAP
    use_optuna = HAS_OPTUNA
    analyze_btn = False

    # Default symbol
    symbol = "AAPL"

    if st.session_state.current_page == "Analyze":
        # Use selected ticker from watchlist if set, otherwise default to AAPL
        ticker_value = st.session_state.selected_ticker if st.session_state.selected_ticker else "AAPL"

        symbol = st.text_input(
            "Ticker", value=ticker_value, placeholder="AAPL, NVDA, TSLA…",
            help="Any US stock ticker"
        ).upper().strip()

        analyze_btn = st.button("Run Analysis", type="primary", use_container_width=True)

        # Auto-trigger analysis if coming from watchlist
        if st.session_state.auto_analyze and st.session_state.selected_ticker:
            analyze_btn = True
            st.session_state.auto_analyze = False

        st.divider()

        # Engine info
        parts = ["XGB×5"]
        if HAS_LGB: parts.append("LGB×3")
        parts += ["RF×2", "CLS×4", "RAdj×2", "Ridge L2"]
        st.markdown(
            f"<div style='background:linear-gradient(145deg,{BG_SURFACE},{BG_CARD});"
            f"border:1px solid {BORDER};border-radius:12px;padding:13px 15px;"
            f"position:relative;overflow:hidden'>"
            # top gradient hairline
            f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
            f"background:{BRAND_GRAD};opacity:0.7'></div>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'>"
            f"<div style='width:6px;height:6px;border-radius:50%;background:{CYAN};"
            f"box-shadow:0 0 10px {CYAN}'></div>"
            f"<span style='color:{CYAN};font-size:0.58rem;text-transform:uppercase;"
            f"letter-spacing:0.14em;font-weight:700'>Prediqt IQ Engine</span>"
            f"</div>"
            f"<div style='color:{TEXT_PRIMARY};font-weight:700;font-size:0.72rem;"
            f"letter-spacing:0.01em;font-family:\"Inter\",sans-serif'>"
            f"{' · '.join(parts)}</div>"
            f"</div>", unsafe_allow_html=True)

        st.caption("For educational purposes only. Not financial advice.")

    # ── Initialize watchlist (needed globally) ──────────────────────────────
    if "watchlist" not in st.session_state or st.session_state.watchlist is None:
        st.session_state.watchlist = load_watchlist() or []
    if not isinstance(st.session_state.watchlist, list):
        st.session_state.watchlist = list(st.session_state.watchlist or [])

    # ── Sidebar footer ───────────────────────────────────────────────────────
    st.markdown(f"<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='border-top:1px solid {BORDER};padding-top:14px;margin-top:auto'>
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'>
            <div style='position:relative;width:7px;height:7px'>
                <div style='position:absolute;inset:0;border-radius:50%;background:{CYAN};
                            box-shadow:0 0 10px {CYAN};animation:pulse 2.4s ease-in-out infinite'></div>
            </div>
            <span style='color:{CYAN};font-size:0.58rem;font-weight:700;text-transform:uppercase;
                         letter-spacing:0.14em'>Systems Online</span>
        </div>
        <div style='color:{TEXT_MUTED};font-size:0.6rem;line-height:1.6;letter-spacing:0.01em'>
            <span style='color:{TEXT_SECONDARY};font-weight:700'>Prediqt v4.0</span>
            &nbsp;·&nbsp;ML Intelligence Engine<br>
            <span style='font-style:italic;opacity:0.75'>For educational purposes only. Not financial advice.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results = None

# ── Background scoring on every page load ─────────────────────────────────────
# Silently scores any pending predictions without needing the user to run analysis.
# Uses a session flag so it only runs once per browser session, not on every rerun.
if "bg_scored" not in st.session_state:
    st.session_state.bg_scored = False

if not st.session_state.bg_scored:
    try:
        score_all_intervals()
        analyze_scored_predictions()  # update learning state
    except Exception:
        pass
    st.session_state.bg_scored = True


# ══════════════════════════════════════════════════════════════════════════════
# BATCH SCAN — train + log predictions for every watchlist symbol
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.get("batch_scan_requested") and st.session_state.get("watchlist"):
    st.session_state.batch_scan_requested = False
    _wl = st.session_state.watchlist[:]

    # ── Skip stocks already scanned today ────────────────────────────────
    from prediction_logger_v2 import _load_log as _batch_load_log
    import time as _time
    _today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        _existing = _batch_load_log()
        _scanned_today = set()
        for _e in _existing:
            _ts = _e.get("timestamp", "")
            if _ts.startswith(_today_str):
                _scanned_today.add(_e.get("symbol", ""))
        _to_scan = [s for s in _wl if s not in _scanned_today]
        _skipped_count = len(_wl) - len(_to_scan)
    except Exception:
        _to_scan = _wl
        _skipped_count = 0

    _batch_status = st.status(
        f"Batch scanning {len(_to_scan)} stocks"
        + (f" ({_skipped_count} already scanned today)" if _skipped_count else "")
        + "…",
        expanded=True,
    )
    _batch_logged = 0
    _batch_errors = []

    with _batch_status:
        if _skipped_count:
            st.write(f"⏭️ Skipping {_skipped_count} stocks already scanned today")

        # ── Prefetch data in parallel (I/O bound) ───────────────────────
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _prefetch(sym):
            """Fetch all data for a symbol (runs in thread)."""
            _df   = _cached_fetch_data(sym, period=data_period)
            _info = _cached_fetch_info(sym)
            _mkt  = _cached_fetch_market(period=data_period, sector=_info.get("sector"))
            _fund = fetch_fundamentals(sym)
            _earn = fetch_earnings_data(sym)
            _opts = fetch_options_data(sym, float(_df["Close"].iloc[-1])) if _df is not None and len(_df) > 0 else {}
            # Sentiment (best effort)
            _sent = None
            try:
                _sent = get_sentiment_features(sym)
            except Exception:
                pass
            return sym, _df, _info, _mkt, _fund, _earn, _opts, _sent

        # Phase 1: parallel data fetch (4 threads — enough for API calls)
        _prefetched = {}
        _fetch_start = _time.time()
        if _to_scan:
            st.write(f"📡 Fetching data for {len(_to_scan)} stocks in parallel…")
            _fetch_bar = st.progress(0, text="Fetching…")
            _done_count = 0
            with ThreadPoolExecutor(max_workers=4) as _pool:
                _futures = {_pool.submit(_prefetch, s): s for s in _to_scan}
                for _fut in as_completed(_futures):
                    _s = _futures[_fut]
                    _done_count += 1
                    _fetch_bar.progress(
                        _done_count / len(_to_scan),
                        text=f"Fetched {_done_count}/{len(_to_scan)} — {_s}",
                    )
                    try:
                        _result = _fut.result()
                        _prefetched[_result[0]] = _result[1:]
                    except Exception as _fe:
                        _batch_errors.append(f"{_s}: fetch failed — {str(_fe)[:50]}")
            _fetch_bar.empty()
            _fetch_elapsed = _time.time() - _fetch_start
            st.write(f"📡 Data fetched in {_fetch_elapsed:.0f}s")

        # Phase 2: parallel model training ─────────────────────────────
        # sklearn / LightGBM release the GIL during C-level work, so
        # threading gives real speedup for CPU-bound training.
        import threading

        def _train_one(sym, data_tuple):
            """Train model + return result dict (runs in thread)."""
            _bdf, _binfo, _bmkt, _bfund, _bearn, _bopts, _bsent = data_tuple
            if _bdf is None or (hasattr(_bdf, 'empty') and _bdf.empty):
                return sym, None, "No price data"

            # Attach sentiment
            if _bsent and _bmkt and isinstance(_bmkt, dict):
                _s = _bmkt.get("sentiment")
                if isinstance(_s, dict):
                    _s["news_sentiment"] = _bsent

            try:
                pred_obj = StockPredictor(sym)
                pred_obj.train(_bdf, market_ctx=_bmkt, fundamentals=_bfund,
                               earnings_data=_bearn, options_data=_bopts)
                preds = pred_obj.predict(_bdf, market_ctx=_bmkt)
                fi = pred_obj.get_feature_importance() if hasattr(pred_obj, 'get_feature_importance') else None
                mv = get_current_model_version()
                spy = _bmkt.get("spy") if _bmkt else None
                vix = _bmkt.get("vix") if _bmkt else None
                regime = detect_regime(_bdf, spy, vix).get("label", "Unknown") if _bdf is not None else "Unknown"

                # Confidence adjustments
                try:
                    for h in preds:
                        c = preds[h].get("confidence", 50)
                        preds[h]["confidence"] = apply_confidence_adjustments(h, c, regime)
                except Exception:
                    pass

                return sym, {"preds": preds, "fi": fi, "mv": mv, "regime": regime}, None
            except Exception as e:
                return sym, None, str(e)[:60]

        _train_start = _time.time()
        _train_total = len(_prefetched)
        _train_workers = min(3, _train_total)  # 3 parallel trainers
        _train_bar = st.progress(0, text="Training models…")
        _completed = [0]  # mutable for thread-safe counter
        _lock = threading.Lock()
        _results = {}

        st.write(f"🧠 Training {_train_total} models ({_train_workers} in parallel)…")

        with ThreadPoolExecutor(max_workers=_train_workers) as _pool:
            _futures = {
                _pool.submit(_train_one, s, d): s
                for s, d in _prefetched.items()
            }
            for _fut in as_completed(_futures):
                _s = _futures[_fut]
                try:
                    _sym_r, _res, _err = _fut.result()
                    _results[_sym_r] = (_res, _err)
                except Exception as _te:
                    _results[_s] = (None, str(_te)[:60])
                with _lock:
                    _completed[0] += 1
                    _c = _completed[0]
                _elapsed = _time.time() - _train_start
                _per = _elapsed / _c if _c else 0
                _remaining = _per * (_train_total - _c)
                _train_bar.progress(
                    _c / _train_total,
                    text=f"Trained {_c}/{_train_total} · {_s} · ~{_remaining:.0f}s left",
                )
        _train_bar.empty()

        # Log results (must be sequential — file I/O)
        for _sym, (_res, _err) in _results.items():
            if _res:
                log_prediction_v2(_sym, _res["preds"], _res["fi"], _res["mv"], _res["regime"])
                _batch_logged += 1
            elif _err:
                _batch_errors.append(f"{_sym}: {_err}")

        # Score any predictions that are now ready
        try:
            score_all_intervals()
            analyze_scored_predictions()
        except Exception:
            pass

        _total_elapsed = _time.time() - _fetch_start
        _mins = int(_total_elapsed // 60)
        _secs = int(_total_elapsed % 60)
        _batch_status.update(
            label=f"✅ Batch complete — {_batch_logged} scanned, "
                  f"{_skipped_count} skipped, "
                  f"{len(_batch_errors)} errors · {_mins}m {_secs}s",
            state="complete", expanded=False)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS — runs on button click only
# ══════════════════════════════════════════════════════════════════════════════

if analyze_btn and symbol:
    st.session_state.results = None

    # ── Compact progress bar ─────────────────────────────────────────────────
    _analysis_ph = st.empty()
    _n_steps = 6 + (1 if run_bt else 0)
    _step = [0]  # mutable container for closure access

    def _update_progress(label):
        _step[0] += 1
        _analysis_ph.progress(_step[0] / _n_steps, text=f"Analyzing {symbol} — {label}")

    def _train_progress(frac, msg):
        """Progress callback for model training (maps to step 4 range)."""
        _frac = (_step[0] + frac) / _n_steps
        _analysis_ph.progress(min(_frac, 0.99), text=f"Analyzing {symbol} — Training models")

    _update_progress("Loading data")
    try:
        df   = _cached_fetch_data(symbol, period=data_period)
        info = _cached_fetch_info(symbol)
    except Exception as e:
        _analysis_ph.empty()
        st.error(f"Could not load {symbol}: {e}")
        st.stop()

    market_ctx = None
    if use_ctx:
        _update_progress("Market context")
        try:
            market_ctx = _cached_fetch_market(period=data_period, sector=info.get("sector"))
        except Exception:
            pass

    _update_progress("Fundamentals & sentiment")
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

    # Fetch news sentiment + Fear & Greed and inject into market context
    try:
        _news_sentiment = get_sentiment_features(symbol)
        if market_ctx is None:
            market_ctx = {}
        if "sentiment" not in market_ctx or market_ctx["sentiment"] is None:
            market_ctx["sentiment"] = {}
        market_ctx["sentiment"]["news_sentiment"] = _news_sentiment
    except Exception:
        pass

    _update_progress("Training models")
    try:
        predictor   = StockPredictor(symbol)
        predictor.train(df, market_ctx=market_ctx, fundamentals=fundamentals,
                        earnings_data=earnings, options_data=options_mkt,
                        use_shap=use_shap, use_optuna=use_optuna,
                        progress_callback=_train_progress)
        predictions = predictor.predict(df, market_ctx=market_ctx)
    except Exception as e:
        _analysis_ph.empty()
        st.error(f"Training failed: {e}")
        st.stop()

    _update_progress("Generating signals")
    analysis = generate_analysis(df, predictions, info)
    options  = generate_options_report(predictions, df, focus_horizon="1 Month")

    spy_s = market_ctx.get("spy") if market_ctx else None
    vix_s = market_ctx.get("vix") if market_ctx else None
    regime_info = detect_regime(df, spy_s, vix_s)

    bt_results = None
    if run_bt:
        _update_progress("Backtesting")
        def _bt_progress(frac, msg):
            _frac = (_step[0] + frac) / _n_steps
            _analysis_ph.progress(min(_frac, 0.99), text=f"Analyzing {symbol} — Backtesting")
        try:
            bt_results = run_backtest(df, market_ctx=market_ctx, fundamentals=fundamentals,
                                     earnings_data=earnings, options_data=options_mkt,
                                     progress_callback=_bt_progress)
        except Exception as e:
            st.warning(f"Backtest error: {e}")

    _update_progress("Done")
    _analysis_ph.empty()

    st.session_state.results = dict(
        symbol=symbol, df=df, info=info,
        predictions=predictions, analysis=analysis,
        options=options, predictor=predictor,
        bt_results=bt_results, regime_info=regime_info,
        market_ctx=market_ctx,
        earnings=earnings,
        fundamentals=fundamentals,
    )

    # Log prediction for future accuracy tracking (v1 + v2)
    try:
        log_prediction(symbol, predictions)
    except Exception:
        pass

    # V2: Enhanced logging with feature tracking and backtest scoring
    try:
        rlabel = regime_info.get("label", "Unknown") if regime_info else "Unknown"
        fi = predictor.get_feature_importance() if hasattr(predictor, 'get_feature_importance') else None
        mv = get_current_model_version()

        # Apply learned confidence adjustments from past predictions
        try:
            for horizon in predictions:
                base_conf = predictions[horizon].get("confidence", 50)
                adjusted_conf = apply_confidence_adjustments(horizon, base_conf, rlabel)
                predictions[horizon]["confidence"] = adjusted_conf
        except Exception:
            pass

        log_prediction_v2(symbol, predictions, fi, mv, rlabel)

        # Instant backtest scoring for immediate accuracy feedback
        bt_acc = backtest_accuracy(symbol, df, predictions, n_samples=25)
        st.session_state.results["backtest_accuracy"] = bt_acc
    except Exception:
        pass

    # Score any pending predictions from earlier sessions
    try:
        score_all_intervals()
        analyze_scored_predictions()  # update learning state
    except Exception:
        pass

    # Clear auto-analyze flag and selected ticker after analysis completes
    st.session_state.selected_ticker = None
    st.session_state.auto_analyze = False


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTING
# ══════════════════════════════════════════════════════════════════════════════

_page = st.session_state.current_page

# ── Paywall gate for Pro pages ───────────────────────────────────────────────
if _page in PRO_PAGES and not st.session_state.is_subscribed:
    _render_paywall(_page)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD — Live overview with instant value
# ──────────────────────────────────────────────────────────────────────────────
if _page == "Dashboard":
    # ═════════════════════════════════════════════════════════════════════════
    # DATA SCOPE NOTES (Prediqt multi-tenancy model)
    # ─────────────────────────────────────────────────────────────────────────
    # UNIVERSAL (shared across all users, public-ledger / trust metrics):
    #   • get_full_analytics()          → model's prediction log, everyone's
    #                                     predictions contribute here
    #   • Market Pulse (SPY, VIX, etc.) → market data, user-agnostic
    #   • Model Accuracy, Streak, total Predictions count
    #   • Daily Briefing "top calls"    → model's highest-confidence recent calls
    #   • Recent Results list           → compact mirror of Track Record
    #
    # PER-USER (scoped to the account):
    #   • get_portfolio_stats()         → this user's paper-trading book
    #   • st.session_state.watchlist    → this user's watchlist
    #   • Today's Best Signals          → signals against THIS user's watchlist
    #   • Expiring Trades               → THIS user's positions
    #   • Portfolio Value chart         → THIS user's equity curve
    #
    # When multi-tenancy lands, thread user_id through get_portfolio_stats()
    # and load_watchlist() — get_full_analytics() stays universal.
    # ═════════════════════════════════════════════════════════════════════════

    # ── Data loading ─────────────────────────────────────────────────────────
    _now = datetime.now()
    _hour = _now.hour
    _greeting = "Good morning" if _hour < 12 else "Good afternoon" if _hour < 17 else "Good evening"

    # UNIVERSAL — model-wide analytics
    # We use Expiration Win Rate (live_accuracy) and Checkpoint Win Rate
    # (direction_correct_any / scored_any) — matching Track Record exactly.
    # No more max(live, quick) cherry-picking for the headline number.
    try:
        _pa = get_full_analytics()
        _mv = get_current_model_version()
        _total_p = _pa.get("total_predictions", 0)
        _scored = _pa.get("scored_any", 0)
        _live_acc = _pa.get("live_accuracy", 0)      # Expiration Win Rate (strict)
        _quick_acc = _pa.get("quick_accuracy", 0)    # day-1 direction (kept for briefing)
        _wins = _pa.get("direction_correct_any", 0)  # feeds Checkpoint Win Rate
    except Exception:
        _pa = {}
        _total_p = _scored = _wins = 0
        _live_acc = _quick_acc = 0
        _mv = "1.0"

    # PER-USER — this user's paper-trading portfolio
    try:
        _pstats = get_portfolio_stats()
        _port_val = _pstats.get("equity", 10000)
        _port_pnl_pct = _pstats.get("total_return_pct", 0)
        _port_pnl_dollar = _port_val - 10000
        _open_pos = _pstats.get("n_open_positions", 0)
        _port_cash = _pstats.get("cash", 10000)
        _port_win_rate = _pstats.get("win_rate", 0)
        _port_closed = _pstats.get("n_closed_trades", 0)
        _port_open_list = _pstats.get("open_positions", [])
    except Exception:
        _pstats = {}
        _port_val = 10000
        _port_pnl_pct = _port_pnl_dollar = 0
        _open_pos = 0
        _port_cash = 10000
        _port_win_rate = 0
        _port_closed = 0
        _port_open_list = []

    _wl_count = len(st.session_state.get("watchlist", []))
    _pnl_color = GREEN if _port_pnl_pct >= 0 else RED
    _losses = _scored - _wins
    _pending = _total_p - _scored

    # ── Hero row — landing-style eyebrow + gradient greeting ─────────────────
    _ph_first_name = "Marc"  # placeholder — hook into auth when available
    st.markdown(
        f"<div style='margin-bottom:28px;padding-top:14px'>"
        # Eyebrow chip
        f"<div style='display:inline-flex;align-items:center;gap:8px;padding:5px 14px;"
        f"border-radius:100px;border:1px solid rgba(6,214,160,0.35);"
        f"background:rgba(6,214,160,0.08);color:{CYAN};font-size:0.62rem;"
        f"font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:14px'>"
        f"<span style='width:5px;height:5px;border-radius:50%;background:{CYAN};"
        f"box-shadow:0 0 10px {CYAN};animation:pulse 2.4s ease-in-out infinite'></span>"
        f"Prediqt IQ · Live"
        f"</div>"
        # Gradient headline
        f"<h1 style='font-family:\"Inter\",sans-serif;font-size:2.2rem;font-weight:900;"
        f"letter-spacing:-0.035em;margin:0 0 6px;line-height:1.05;color:{TEXT_PRIMARY}'>"
        f"{_greeting}, <span style='background:{BRAND_GRAD};-webkit-background-clip:text;"
        f"background-clip:text;-webkit-text-fill-color:transparent;color:transparent'>"
        f"{_ph_first_name}.</span></h1>"
        f"<p style='color:{TEXT_SECONDARY};font-size:0.82rem;margin:0;font-weight:500'>"
        f"{_now.strftime('%A, %B %d, %Y')} &nbsp;·&nbsp; here's what the model sees today.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Daily Briefing ─────────────────────────────────────────────────────
    try:
        from daily_briefing import get_briefing
        _briefing = get_briefing()
        _brief_text = _briefing.get("summary_text", "")
        if _brief_text:
            _brief_w = _briefing.get("recent_wins", 0)
            _brief_l = _briefing.get("recent_losses", 0)
            _brief_acc = _briefing.get("accuracy", 0)
            _brief_acc_clr = GREEN if _brief_acc >= 55 else AMBER if _brief_acc >= 50 else TEXT_SECONDARY

            # ── Model state one-liner ────────────────────────────────────
            if _streak_type == "W" and _streak >= 3:
                _model_word, _model_clr = "running hot", CHART_UP
            elif _streak_type == "W" and _streak >= 2:
                _model_word, _model_clr = "on a roll", CHART_UP
            elif _streak_type == "L" and _streak >= 3:
                _model_word, _model_clr = "cooling off", CHART_DOWN
            elif _streak_type == "L" and _streak >= 2:
                _model_word, _model_clr = "taking a hit", CHART_DOWN
            elif _brief_acc >= 60:
                _model_word, _model_clr = "performing well", CHART_UP
            elif _brief_acc < 50 and _brief_acc > 0:
                _model_word, _model_clr = "below par", CHART_DOWN
            else:
                _model_word, _model_clr = "staying steady", TEXT_SECONDARY

            # ── Top picks — highest-confidence pending predictions ──────
            _top_picks = []
            try:
                _pt_brief = _pa.get("predictions_table", []) or []
                # Recent unscored high-confidence calls
                _pend = [p for p in _pt_brief
                         if p.get("final_result") == "pending"
                         and p.get("confidence", 0) >= 55]
                _pend.sort(
                    key=lambda p: (p.get("date", ""), p.get("confidence", 0)),
                    reverse=True,
                )
                _seen_syms = set()
                for _p in _pend:
                    _s = _p.get("symbol", "")
                    if _s and _s not in _seen_syms:
                        _seen_syms.add(_s)
                        _top_picks.append(_p)
                    if len(_top_picks) >= 3:
                        break
            except Exception:
                pass

            # ── Build top-picks pill row ─────────────────────────────────
            _picks_html = ""
            if _top_picks:
                for _pk in _top_picks:
                    _pk_sym = _pk.get("symbol", "")
                    _pk_dir = (_pk.get("direction") or "").lower()
                    _pk_conf = float(_pk.get("confidence", 0))
                    _pk_ret = float(_pk.get("predicted_return", 0))
                    if "bull" in _pk_dir or _pk_ret >= 2:
                        _pc, _pb, _pbd, _pi, _pl = (
                            CHART_UP, "rgba(6,214,160,0.10)",
                            "rgba(6,214,160,0.32)", "▲", "BUY")
                    elif "bear" in _pk_dir or _pk_ret <= -2:
                        _pc, _pb, _pbd, _pi, _pl = (
                            CHART_DOWN, "rgba(224,74,74,0.10)",
                            "rgba(224,74,74,0.32)", "▼", "SELL")
                    else:
                        _pc, _pb, _pbd, _pi, _pl = (
                            AMBER, "rgba(245,158,11,0.10)",
                            "rgba(245,158,11,0.32)", "→", "HOLD")
                    _picks_html += (
                        f"<span style='display:inline-flex;align-items:center;gap:7px;"
                        f"padding:5px 11px;border-radius:8px;background:{_pb};"
                        f"border:1px solid {_pbd};font-size:0.7rem;font-weight:700;"
                        f"font-variant-numeric:tabular-nums;white-space:nowrap'>"
                        f"<span style='color:{TEXT_PRIMARY};letter-spacing:0.02em'>{_pk_sym}</span>"
                        f"<span style='color:{_pc}'>{_pi} {_pl}</span>"
                        f"<span style='color:{_pc};opacity:0.75;font-weight:600'>{_pk_conf:.0f}%</span>"
                        f"</span>"
                    )

            # Compose the briefing headline
            _regime_phrase_br = f"{_regime_label.lower() if '_regime_label' in dir() else 'neutral'}"
            _headline_html = (
                f"Model is <b style='color:{_model_clr}'>{_model_word}</b> "
                f"<span style='color:{TEXT_MUTED}'>·</span> "
                f"<b style='color:{CHART_UP}'>{_brief_w}W</b>"
                f"<span style='color:{TEXT_MUTED}'>/</span>"
                f"<b style='color:{CHART_DOWN}'>{_brief_l}L</b> last 10 "
                f"<span style='color:{TEXT_MUTED}'>·</span> "
                f"{_brief_acc:.0f}% overall"
            )

            # Render the briefing card — "Model's top calls" framing so users
            # understand these are the model's highest-confidence recent calls
            # across ALL user activity, not just their own.
            _picks_row = (
                f"<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center;"
                f"margin-top:10px'>"
                f"<span style='color:{TEXT_MUTED};font-size:0.6rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.14em;margin-right:4px;"
                f"display:inline-flex;align-items:center;gap:6px'>"
                f"Model's top calls"
                f"<span style='padding:1px 6px;border-radius:100px;"
                f"background:rgba(6,214,160,0.10);border:1px solid rgba(6,214,160,0.28);"
                f"color:{CYAN};font-size:0.46rem;letter-spacing:0.12em'>PUBLIC</span>"
                f"</span>"
                f"{_picks_html}"
                f"</div>"
            ) if _picks_html else ""

            st.markdown(
                f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
                f"border:1px solid {BORDER};border-radius:14px;padding:18px 22px;"
                f"margin-bottom:22px;position:relative;overflow:hidden;"
                f"backdrop-filter:blur(8px)'>"
                # Gradient hairline
                f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
                f"background:{BRAND_GRAD};opacity:0.75'></div>"
                # Top row: icon + title + accuracy + W/L
                f"<div style='display:flex;align-items:center;gap:16px'>"
                # Icon badge
                f"<div style='width:44px;height:44px;border-radius:12px;flex-shrink:0;"
                f"background:linear-gradient(145deg,rgba(32,128,229,0.18),rgba(6,214,160,0.12));"
                f"border:1px solid rgba(6,214,160,0.35);display:flex;align-items:center;"
                f"justify-content:center'>"
                f"<svg xmlns='http://www.w3.org/2000/svg' width='22' height='22' viewBox='0 0 24 24' "
                f"fill='none' stroke='{CYAN}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
                f"<rect x='8' y='3' width='8' height='4' rx='1'/>"
                f"<path d='M8 5H5a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-3'/>"
                f"<path d='M9 13h6M9 17h4'/></svg>"
                f"</div>"
                # Title column
                f"<div style='flex:1;min-width:0'>"
                f"<div style='color:{CYAN};font-size:0.58rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.14em;margin-bottom:4px'>"
                f"Daily Briefing</div>"
                f"<div style='color:{TEXT_PRIMARY};font-size:0.82rem;line-height:1.55;"
                f"font-weight:500'>{_headline_html}</div>"
                f"</div>"
                # Right-hand accuracy + W/L
                f"<div style='display:flex;gap:20px;flex-shrink:0;align-items:center'>"
                f"<div style='text-align:center'>"
                f"<div style='font-family:\"Inter\",sans-serif;color:{_brief_acc_clr};"
                f"font-size:1.2rem;font-weight:900;letter-spacing:-0.02em;"
                f"font-variant-numeric:tabular-nums;line-height:1'>{_brief_acc:.0f}%</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-top:3px'>Accuracy</div>"
                f"</div>"
                f"<div style='width:1px;height:32px;background:{BORDER}'></div>"
                f"<div style='text-align:center'>"
                f"<div style='font-family:\"Inter\",sans-serif;font-size:1.2rem;font-weight:900;"
                f"letter-spacing:-0.02em;font-variant-numeric:tabular-nums;line-height:1'>"
                f"<span style='color:{CHART_UP}'>{_brief_w}</span>"
                f"<span style='color:{TEXT_MUTED};font-size:0.85rem;font-weight:600'>&nbsp;/&nbsp;</span>"
                f"<span style='color:{CHART_DOWN}'>{_brief_l}</span></div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-top:3px'>Last 10</div>"
                f"</div>"
                f"</div>"
                f"</div>"
                # Bottom row: top picks pills
                f"{_picks_row}"
                f"</div>",
                unsafe_allow_html=True)
    except Exception:
        pass

    # ── 4 metric cards ───────────────────────────────────────────────────────
    # Status-dashboard layout:
    #   [PERSONAL]   Your Portfolio + open positions
    #   [PERSONAL]   Today's Signals (actionable BUY/SELL count from watchlist)
    #   [PUBLIC]     Expiration Win Rate — the strict accuracy read
    #                (matches Track Record, no cherry-picking)
    #   [PUBLIC]     Recent Form — current streak + last 10 matured record
    _pnl_arrow = "▲" if _port_pnl_pct >= 0 else "▼"

    # ── Current streak (from matured predictions only, newest first) ──
    _streak = 0
    _streak_type = ""
    _last10 = []
    try:
        _preds_tbl = _pa.get("predictions_table", [])
        _scored_list = [p for p in _preds_tbl if p.get("final_result") in ("✓", "✗")]
        # Sort by date newest first so streak reflects the actual sequence
        _scored_list = sorted(_scored_list,
                              key=lambda p: p.get("date", ""), reverse=True)
        if _scored_list:
            _last_result = _scored_list[0].get("final_result", "")
            _streak_type = "W" if _last_result == "✓" else "L"
            for _sp in _scored_list:
                if _sp.get("final_result") == _last_result:
                    _streak += 1
                else:
                    break
        _last10 = [p.get("final_result") for p in _scored_list[:10]]
    except Exception:
        pass

    _l10_w = sum(1 for r in _last10 if r == "✓")
    _l10_l = sum(1 for r in _last10 if r == "✗")
    _l10_n = len(_last10)
    # Color the card based on net record: green if winning majority,
    # red if losing majority, muted if no matured predictions yet
    if _l10_n == 0:
        _l10_color = TEXT_MUTED
    elif _l10_w > _l10_l:
        _l10_color = GREEN
    elif _l10_l > _l10_w:
        _l10_color = RED
    else:
        _l10_color = AMBER
    _streak_color = GREEN if _streak_type == "W" else RED if _streak_type == "L" else TEXT_MUTED

    # ── Today's actionable watchlist signals (single scan, cap at 12) ──
    _wl_list = st.session_state.get("watchlist", [])
    _sig_buy = _sig_sell = _sig_hold = 0
    _sig_scanned = 0
    for _ws in _wl_list[:12]:
        try:
            _sig = get_quick_signal(_ws)
            if not _sig.get("error"):
                _sig_scanned += 1
                _s = _sig.get("signal", "HOLD")
                if _s == "BUY":
                    _sig_buy += 1
                elif _s == "SELL":
                    _sig_sell += 1
                else:
                    _sig_hold += 1
        except Exception:
            pass
    _sig_actionable = _sig_buy + _sig_sell

    # Big-number display: always show the number. The "All Clear" framing
    # moves to the sub-text so the visual rhythm across the 4 cards stays
    # consistent (same font size for all big values).
    if _sig_actionable >= 3:
        _sig_color = CHART_UP
    elif _sig_actionable >= 1:
        _sig_color = BLUE
    else:
        # Zero — but it's a "quiet day," not an error. Cyan to celebrate.
        _sig_color = CYAN if _wl_list else TEXT_MUTED

    _sig_big = f"{_sig_actionable}"
    _sig_big_size = "2rem"

    if not _wl_list:
        _sig_sub = "add stocks to your watchlist"
    elif _sig_actionable == 0:
        _sig_sub = (
            f"<b style='color:{CYAN}'>All Clear</b> · "
            f"all {_sig_hold} watchlist stock{'s' if _sig_hold != 1 else ''} on hold"
        )
    else:
        _sig_sub = f"{_sig_buy} BUY · {_sig_sell} SELL · {_sig_hold} hold"

    # ── Expiration Win Rate (the public headline) ──
    _exp_win = _pa.get("live_accuracy", 0) or 0
    _exp_matured = _pa.get("scored_final", 0) or 0
    _checkpoint_win = _best_acc = ((_wins / _scored) * 100) if _scored > 0 else 0
    if _exp_matured == 0:
        _exp_big = "—"
        _exp_color = TEXT_MUTED
        _exp_sub = f"no predictions matured yet · v{_mv}"
    else:
        _exp_big = f"{_exp_win:.0f}%"
        _exp_color = GREEN if _exp_win >= 55 else AMBER if _exp_win >= 50 else RED
        _exp_sub = (f"{_exp_matured} matured · "
                    f"checkpoint read {_checkpoint_win:.0f}% · v{_mv}")

    # ── Portfolio sub-text ──
    _port_sub = (
        f"{_pnl_arrow}{abs(_port_pnl_pct):.1f}% · "
        f"{_open_pos} open position{'s' if _open_pos != 1 else ''}"
    )

    # ── LAST 10 card — big number is the W/L record, sub-text is the streak ──
    if _l10_n == 0:
        _l10_big = "—"
        _l10_sub = "no matured predictions yet"
    else:
        _l10_big_html = (
            f"<span style='color:{CHART_UP}'>{_l10_w}W</span>"
            f"<span style='color:{TEXT_MUTED};font-size:0.7em;font-weight:600;"
            f"margin:0 6px'>/</span>"
            f"<span style='color:{CHART_DOWN}'>{_l10_l}L</span>"
        )
        _l10_big = _l10_big_html  # rendered as HTML inside the card
        if _streak > 0:
            _l10_sub = (
                f"current <b style='color:{_streak_color}'>{_streak}{_streak_type} streak</b> "
                f"· last {_l10_n} matured"
            )
        else:
            _l10_sub = f"last {_l10_n} matured predictions"

    # ── Expandable-card state ─────────────────────────────────────────────
    # Session-state set tracks which cards are currently expanded. Multiple
    # may be open at once (independent state per user requirements).
    if "dash_expanded" not in st.session_state:
        st.session_state.dash_expanded = set()

    # Inject CSS once — cards become clickable via an invisible overlay
    # button, so no visible "View details" chrome is needed.
    st.markdown(
        f"""<style>
        /* Hover / pressed states on the card wrapper */
        .dash-metric-card {{
            cursor: pointer;
            transition: border-color 0.22s ease, box-shadow 0.22s ease,
                        transform 0.22s ease;
        }}
        .dash-metric-card:hover {{
            border-color: {BORDER_ACCENT} !important;
            box-shadow: 0 8px 32px rgba(32,128,229,0.10);
        }}
        .dash-metric-card.is-expanded {{
            border-color: {BORDER_ACCENT} !important;
            box-shadow: 0 0 0 1px {BORDER_ACCENT};
        }}
        /* Chevron affordance in the bottom-right of the card */
        .dash-card-chevron {{
            position: absolute;
            bottom: 16px;
            right: 16px;
            color: {TEXT_MUTED};
            font-size: 0.95rem;
            line-height: 1;
            transition: transform 0.22s ease, color 0.22s ease;
            user-select: none;
            pointer-events: none;
        }}
        .dash-metric-card.is-expanded .dash-card-chevron {{
            color: {CYAN};
            transform: rotate(180deg);
        }}
        .dash-metric-card:hover .dash-card-chevron {{
            color: {CYAN};
        }}

        /* The actual click wiring is done via JS (see _st_components.html
         * below). It hides each card's hidden Streamlit button and attaches
         * a click handler to the .dash-metric-card itself. No CSS :has()
         * tricks needed. */

        /* Full-width expansion panel below the card row */
        .dash-expand-panel {{
            background: linear-gradient(180deg, {BG_ELEVATED}, {BG_SURFACE});
            border: 1px solid {BORDER_ACCENT};
            border-radius: 14px;
            padding: 22px 26px;
            margin-top: 12px;
            position: relative;
            overflow: hidden;
        }}
        .dash-expand-panel::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; height: 1.5px;
            background: {BRAND_GRAD};
            opacity: 0.65;
        }}
        .dash-expand-panel-title {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .dash-expand-panel-eyebrow {{
            color: {CYAN};
            font-size: 0.58rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.14em;
        }}

        /* ── Premium pill action buttons (signal row actions) ────────── */
        .dash-signal-actions {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 4px;
            flex-wrap: wrap;
        }}
        .dash-sig-action {{
            font-family: "Inter", sans-serif;
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            padding: 7px 16px;
            border-radius: 100px;
            cursor: pointer;
            transition: all 0.18s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            line-height: 1;
            white-space: nowrap;
            user-select: none;
            outline: none;
        }}
        .dash-sig-action-icon {{
            font-size: 0.85rem;
            font-weight: 400;
            line-height: 0.8;
            margin-right: 1px;
            opacity: 0.85;
        }}
        .dash-sig-action-arrow {{
            font-size: 0.78rem;
            line-height: 1;
            margin-left: 1px;
            opacity: 0.7;
            transition: transform 0.18s ease, opacity 0.18s ease;
        }}
        .dash-sig-action:hover .dash-sig-action-arrow {{
            transform: translateX(2px);
            opacity: 1;
        }}

        /* Trade — brand-filled primary */
        .dash-sig-action-trade {{
            background: {BRAND_GRAD};
            border: 1px solid transparent;
            color: #051018;
            box-shadow: 0 2px 14px rgba(6,214,160,0.18);
        }}
        .dash-sig-action-trade:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 20px rgba(6,214,160,0.35);
        }}

        /* View / Analyze — ghost pills */
        .dash-sig-action-view,
        .dash-sig-action-analyze {{
            background: transparent;
            border: 1px solid {BORDER_ACCENT};
            color: {TEXT_SECONDARY};
        }}
        .dash-sig-action-view:hover,
        .dash-sig-action-analyze:hover {{
            border-color: {CYAN};
            color: {TEXT_PRIMARY};
            background: rgba(6,214,160,0.05);
        }}
        </style>""",
        unsafe_allow_html=True)

    _d1, _d2, _d3, _d4 = st.columns(4)
    # Metric cards — each tagged with scope so the user can tell at a glance
    # whether they're looking at a universal (model-wide) or personal number.
    # Tuple: (col, card_id, label, value, sub, color, scope, tooltip, big_size)
    _metric_data = [
        (_d1, "portfolio", "Your Portfolio", f"${_port_val:,.0f}", _port_sub,
         _pnl_color, "user",
         "Your paper-trading portfolio value. Everyone starts at $10,000 — "
         "this is what you'd have now if you'd followed the trades you opened.",
         "2rem"),
        (_d2, "actionable", "Actionable", _sig_big, _sig_sub,
         _sig_color, "user",
         "How many BUY or SELL signals exist in your watchlist today. "
         "HOLD signals aren't counted — they mean no strong directional bias. "
         "'All Clear' means nothing in your watchlist needs attention right now.",
         _sig_big_size),
        (_d3, "expiration", "Expiration Win Rate", _exp_big, _exp_sub,
         _exp_color, "universal",
         "Percentage of predictions that were directionally correct at their "
         "scheduled full-horizon date. This is the strictest read — only counts "
         "predictions that have actually matured. Matches the Track Record page.",
         "2rem"),
        (_d4, "last10", "Last 10", _l10_big, _l10_sub,
         _l10_color, "universal",
         "The model's wins and losses across its last 10 matured predictions "
         "(ones that reached their full expiration date). 'Current streak' shows "
         "how many in a row the model has gotten right or wrong most recently.",
         "2rem"),
    ]
    for col, card_id, label, value, sub, color, scope, tooltip, big_size in _metric_data:
        _is_expanded = card_id in st.session_state.dash_expanded
        _expanded_cls = " is-expanded" if _is_expanded else ""
        _chevron_glyph = "▾"  # CSS flips it on is-expanded

        with col:
            # Tiny scope badge in the top-right corner of the card
            if scope == "universal":
                _scope_text, _scope_clr, _scope_bg, _scope_bd = (
                    "PUBLIC", CYAN, "rgba(6,214,160,0.10)", "rgba(6,214,160,0.28)"
                )
            else:
                _scope_text, _scope_clr, _scope_bg, _scope_bd = (
                    "PERSONAL", BLUE, "rgba(32,128,229,0.10)", "rgba(32,128,229,0.28)"
                )
            _scope_pill = (
                f"<span style='position:absolute;top:14px;right:16px;"
                f"padding:2px 7px;border-radius:100px;"
                f"background:{_scope_bg};border:1px solid {_scope_bd};"
                f"color:{_scope_clr};font-size:0.48rem;font-weight:700;"
                f"letter-spacing:0.12em;text-transform:uppercase'>{_scope_text}</span>"
            )
            # Info icon with native browser tooltip on hover
            _info_icon = (
                f"<span title=\"{tooltip}\" "
                f"style='display:inline-flex;align-items:center;justify-content:center;"
                f"width:13px;height:13px;border-radius:50%;border:1px solid {TEXT_MUTED};"
                f"color:{TEXT_MUTED};font-size:0.52rem;font-weight:700;cursor:help;"
                f"margin-left:6px;vertical-align:1px;font-family:Georgia,serif;"
                f"font-style:italic'>i</span>"
            )
            # Chevron affordance — rotates on expand, highlights on hover
            _chevron = (
                f"<span class='dash-card-chevron'>{_chevron_glyph}</span>"
            )
            st.markdown(
                f"<div class='dash-metric-card{_expanded_cls}' "
                f"style='background:linear-gradient(145deg, {BG_CARD}, {BG_SURFACE});"
                f"border:1px solid {BORDER};border-radius:14px;padding:22px 20px 36px;"
                f"position:relative;overflow:hidden'>"
                # Gradient hairline (kept as inline since CSS ::before conflicts
                # with the .dash-metric-card class we added)
                f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
                f"background:{BRAND_GRAD};opacity:0.75'></div>"
                f"{_scope_pill}"
                f"<div style='color:{CYAN};font-size:0.58rem;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:0.14em;margin-bottom:12px;max-width:70%;display:flex;"
                f"align-items:center'>{label}{_info_icon}</div>"
                f"<div style='font-family:\"Inter\",sans-serif;color:{color};font-size:{big_size};"
                f"font-weight:900;letter-spacing:-0.035em;line-height:1.05;"
                f"font-variant-numeric:tabular-nums'>{value}</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.65rem;margin-top:8px;"
                f"letter-spacing:0.01em'>{sub}</div>"
                f"{_chevron}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Hidden Streamlit button — the actual state toggle.
            # Rendered invisibly via JS below (which also wires the card
            # itself as the click target).
            if st.button(
                "toggle",
                key=f"dash_toggle_{card_id}",
                use_container_width=True,
                help=("Click to collapse details" if _is_expanded
                      else "Click to see more details"),
            ):
                if _is_expanded:
                    st.session_state.dash_expanded.discard(card_id)
                else:
                    st.session_state.dash_expanded.add(card_id)
                st.rerun()

    # ── Wire cards + signal rows to their hidden "toggle" buttons via JS ──
    # Strategy: find ALL Streamlit buttons whose label is exactly "toggle"
    # and hide them. This doesn't depend on fragile sibling-walking through
    # Streamlit's wrapper hierarchy. Then event-delegate clicks from the
    # card/row back to the correct hidden button via DOM proximity.
    _st_components.html(
        """
        <script>
        (function () {
          const parentDoc = window.parent.document;

          // Find ALL buttons across the page whose label is "toggle" or
          // "sig-action" — these are our hidden proxy buttons. Hide them.
          function hideToggleButtons() {
            const wrappers = parentDoc.querySelectorAll('[data-testid="stButton"]');
            wrappers.forEach((w) => {
              const btn = w.querySelector('button');
              if (!btn) return;
              const label = (btn.textContent || '').trim();

              if (label === 'toggle') {
                // For dashboard cards, the column must be the positioning
                // context AND the button must still be clickable (invisible
                // overlay). For signal rows, we can just display:none.
                const insideCard = btn.closest('[data-testid="stColumn"]') &&
                                   btn.closest('[data-testid="stColumn"]')
                                      .querySelector('.dash-metric-card');
                if (insideCard) {
                  if (w.dataset.dashHidden !== '1') {
                    w.dataset.dashHidden = '1';
                    w.style.position = 'absolute';
                    w.style.width = '0';
                    w.style.height = '0';
                    w.style.overflow = 'hidden';
                    w.style.opacity = '0';
                    w.style.pointerEvents = 'none';
                    w.style.margin = '0';
                  }
                } else {
                  if (w.dataset.dashHidden !== '1') {
                    w.dataset.dashHidden = '1';
                    w.style.display = 'none';
                  }
                }
              } else if (label === 'sig-action') {
                // Pill-button proxy — fully hidden from layout
                if (w.dataset.dashHidden !== '1') {
                  w.dataset.dashHidden = '1';
                  w.style.display = 'none';
                }
              }
            });

            // Make cards + signal rows visually interactive
            parentDoc.querySelectorAll(
              '.dash-metric-card, .dash-signal-row'
            ).forEach((el) => { el.style.cursor = 'pointer'; });
          }

          // Click a hidden "sig-action" button for the given symbol + action.
          // We match by DOM order: the 3 "sig-action" buttons in a row are
          // trade, view, analyze (in that render order). Find them as a
          // contiguous run that follows the clicked pill's signal row.
          function triggerSigAction(pill) {
            const sym = pill.getAttribute('data-sig-sym');
            const action = pill.getAttribute('data-sig-action');
            if (!sym || !action) return false;

            // Find the signal row this pill belongs to (it's inside the
            // expanded panel that follows the row in DOM order).
            const actionsContainer = pill.closest('.dash-signal-actions');
            if (!actionsContainer) return false;

            // Look forward in document order for the 3 consecutive
            // "sig-action" hidden buttons belonging to this signal.
            const allSigActionButtons = Array.from(
              parentDoc.querySelectorAll('[data-testid="stButton"] button')
            ).filter((b) => (b.textContent || '').trim() === 'sig-action');

            // The hidden buttons are rendered immediately AFTER the pill
            // HTML in DOM order. Find the first 3 "sig-action" buttons
            // that come AFTER actionsContainer.
            const following = allSigActionButtons.filter((b) => {
              const rel = actionsContainer.compareDocumentPosition(b);
              return rel & Node.DOCUMENT_POSITION_FOLLOWING;
            });
            // Sort by document order: first 3 are trade, view, analyze
            following.sort((a, b) => {
              const rel = a.compareDocumentPosition(b);
              if (rel & Node.DOCUMENT_POSITION_FOLLOWING) return -1;
              return 1;
            });
            const slot = action === 'trade' ? 0
                       : action === 'view' ? 1
                       : action === 'analyze' ? 2 : -1;
            const target = following[slot];
            if (target) {
              target.click();
              return true;
            }
            return false;
          }

          // Find the hidden toggle button paired with a clicked element.
          // Uses DOM proximity: the button nearest in document order AFTER
          // the clicked element with label "toggle".
          function findToggleAfter(el) {
            // Collect all toggle buttons on the page
            const toggles = Array.from(
              parentDoc.querySelectorAll('[data-testid="stButton"] button')
            ).filter((b) => (b.textContent || '').trim() === 'toggle');
            if (!toggles.length) return null;

            // Find the toggle whose DOM position is the earliest AFTER el.
            // Use compareDocumentPosition: returns DOCUMENT_POSITION_FOLLOWING (4)
            // when the target follows the reference.
            let best = null;
            for (const t of toggles) {
              const rel = el.compareDocumentPosition(t);
              // 4 = follows; 20 (4|16) = follows + contained by
              if (rel & Node.DOCUMENT_POSITION_FOLLOWING) {
                if (!best) { best = t; continue; }
                // Pick the earliest (one that precedes current best)
                if (best.compareDocumentPosition(t) &
                    Node.DOCUMENT_POSITION_PRECEDING) {
                  best = t;
                }
              }
            }
            return best;
          }

          // For DASHBOARD CARDS specifically, the toggle is inside the
          // same column (rendered BEFORE the expansion panel, still
          // following the card in DOM). Column-scoped lookup is simpler.
          function findCardToggle(card) {
            const col = card.closest('[data-testid="stColumn"]');
            if (!col) return null;
            const btn = col.querySelector('[data-testid="stButton"] button');
            return btn;
          }

          // ── EVENT DELEGATION on document ────────────────────────────
          if (!parentDoc._dashDelegated) {
            parentDoc._dashDelegated = true;
            parentDoc.addEventListener('click', function (ev) {
              // Pill action button (Trade / View / Analyze) — takes priority
              // so a click on a pill doesn't also fire the row-expand.
              const pill = ev.target.closest('.dash-sig-action');
              if (pill) {
                ev.preventDefault();
                ev.stopPropagation();
                triggerSigAction(pill);
                return;
              }

              // Card click
              const card = ev.target.closest('.dash-metric-card');
              if (card) {
                const btn = findCardToggle(card);
                if (btn) {
                  ev.preventDefault();
                  btn.click();
                }
                return;
              }

              // Signal row click — ignore clicks on action pills/buttons
              const sigRow = ev.target.closest('.dash-signal-row');
              if (sigRow) {
                if (ev.target.closest('[data-testid="stButton"]') ||
                    ev.target.closest('.dash-sig-action') ||
                    ev.target.closest('.dash-signal-actions')) return;
                const btn = findToggleAfter(sigRow);
                if (btn) {
                  ev.preventDefault();
                  btn.click();
                }
                return;
              }
            }, false);
          }

          hideToggleButtons();

          // Re-hide on every Streamlit rerun
          const observer = new MutationObserver(() => {
            hideToggleButtons();
          });
          observer.observe(parentDoc.body, {
            childList: true,
            subtree: true,
          });
        })();
        </script>
        """,
        height=0,
    )

    # ── Full-width expansion panels — one per expanded card ────────────
    # Ordered to match card order so panels stack predictably.
    _panel_order = ["portfolio", "actionable", "expiration", "last10"]
    _panel_meta = {
        "portfolio":  ("Your Portfolio", "PERSONAL", BLUE),
        "actionable": ("Actionable", "PERSONAL", BLUE),
        "expiration": ("Expiration Win Rate", "PUBLIC", CYAN),
        "last10":     ("Last 10", "PUBLIC", CYAN),
    }
    for _pid in _panel_order:
        if _pid not in st.session_state.dash_expanded:
            continue
        _title, _ptag, _pclr = _panel_meta[_pid]

        _pill_bg = ("rgba(32,128,229,0.10)" if _ptag == "PERSONAL"
                    else "rgba(6,214,160,0.10)")
        _pill_bd = ("rgba(32,128,229,0.28)" if _ptag == "PERSONAL"
                    else "rgba(6,214,160,0.28)")

        st.markdown(
            f"<div class='dash-expand-panel'>"
            f"<div class='dash-expand-panel-title'>"
            f"<div style='display:flex;align-items:center;gap:10px'>"
            f"<span class='dash-expand-panel-eyebrow'>{_title}</span>"
            f"<span style='padding:2px 8px;border-radius:100px;"
            f"background:{_pill_bg};border:1px solid {_pill_bd};"
            f"color:{_pclr};font-size:0.48rem;font-weight:700;letter-spacing:0.14em;"
            f"text-transform:uppercase'>{_ptag}</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Panel-specific content ────────────────────────────────────
        if _pid == "portfolio":
            # ── 1) Simple equity curve mini-chart ─────────────────────
            _p_curve = _pstats.get("equity_curve", []) if isinstance(_pstats, dict) else []
            if len(_p_curve) >= 2:
                try:
                    import plotly.graph_objects as _go_p
                    _curve_df = pd.DataFrame(_p_curve)
                    _curve_df["date"] = pd.to_datetime(_curve_df["date"])
                    _curve_df = _curve_df.sort_values("date")
                    _eq_last = float(_curve_df["equity"].iloc[-1])
                    _eq_positive = _eq_last >= 10000
                    _eq_color = CHART_UP if _eq_positive else CHART_DOWN
                    _eq_fill = ("rgba(6,214,160,0.14)" if _eq_positive
                                else "rgba(224,74,74,0.14)")

                    _y_min = min(float(_curve_df["equity"].min()), 10000.0)
                    _y_max = max(float(_curve_df["equity"].max()), 10000.0)
                    _y_pad = (_y_max - _y_min) * 0.15 if _y_max > _y_min else 100

                    _fig_p = _go_p.Figure()
                    _fig_p.add_trace(_go_p.Scatter(
                        x=_curve_df["date"], y=_curve_df["equity"],
                        mode="lines",
                        line=dict(color=_eq_color, width=2.2, shape="spline",
                                  smoothing=0.8),
                        fill="tozeroy", fillcolor=_eq_fill,
                        hovertemplate=("<b>%{x|%b %d}</b><br>"
                                       "Portfolio: <b>$%{y:,.0f}</b>"
                                       "<extra></extra>"),
                        showlegend=False,
                    ))
                    _fig_p.add_hline(
                        y=10000, line_dash="dot",
                        line_color="rgba(255,255,255,0.18)", line_width=1,
                    )
                    _fig_p.update_layout(
                        height=160, margin=dict(l=0, r=0, t=4, b=20),
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif",
                                  color=TEXT_SECONDARY, size=9),
                        xaxis=dict(showgrid=False, zeroline=False,
                                   tickfont=dict(color=TEXT_MUTED, size=9),
                                   nticks=4),
                        yaxis=dict(showgrid=False, zeroline=False,
                                   tickprefix="$", tickformat=",",
                                   range=[_y_min - _y_pad, _y_max + _y_pad],
                                   tickfont=dict(color=TEXT_MUTED, size=9),
                                   nticks=3),
                        hoverlabel=dict(bgcolor=BG_ELEVATED,
                                        font_color=TEXT_PRIMARY,
                                        bordercolor=BORDER_ACCENT,
                                        font_family="Inter, sans-serif",
                                        font_size=11),
                    )
                    st.plotly_chart(_fig_p, use_container_width=True,
                                    config={"displayModeBar": False})
                except Exception:
                    pass
            else:
                st.markdown(
                    f"<div style='background:{BG_CARD};border:1px dashed {BORDER};"
                    f"border-radius:8px;padding:18px;text-align:center;"
                    f"color:{TEXT_MUTED};font-size:0.7rem;line-height:1.5;"
                    f"margin-bottom:10px'>"
                    f"Equity curve fills in after your first daily snapshot."
                    f"</div>", unsafe_allow_html=True)

            # ── 2) Top outstanding positions + Recent closes (side by side) ──
            _oc_l, _oc_r = st.columns(2)

            # Top open (sorted by absolute P&L size so biggest movers surface)
            with _oc_l:
                st.markdown(
                    f"<div style='color:{CYAN};font-size:0.52rem;font-weight:800;"
                    f"text-transform:uppercase;letter-spacing:0.14em;margin-bottom:8px'>"
                    f"Top Outstanding &nbsp;"
                    f"<span style='color:{TEXT_MUTED};font-weight:500;"
                    f"letter-spacing:0.02em;text-transform:none'>"
                    f"· {_open_pos} open</span></div>",
                    unsafe_allow_html=True)
                if _port_open_list:
                    _top_open = sorted(
                        _port_open_list,
                        key=lambda p: abs(p.get("unrealised_pnl", 0)),
                        reverse=True,
                    )[:4]
                    for _p in _top_open:
                        _ppnl = _p.get("unrealised_pnl", 0)
                        _pppct = _p.get("unrealised_pct", 0)
                        _pc = CHART_UP if _ppnl >= 0 else CHART_DOWN
                        _pdir = _p.get("direction", "LONG")
                        _pdc = CHART_UP if _pdir == "LONG" else CHART_DOWN
                        st.markdown(
                            f"<div style='background:{BG_CARD};border:1px solid {BORDER};"
                            f"border-radius:8px;padding:8px 12px;margin-bottom:3px;"
                            f"display:flex;align-items:center;gap:8px;"
                            f"border-left:3px solid {_pdc}'>"
                            f"<span style='flex:1;color:{TEXT_PRIMARY};font-weight:800;"
                            f"font-size:0.78rem'>{_p.get('symbol', '')}</span>"
                            f"<span style='color:{_pdc};font-size:0.52rem;font-weight:800;"
                            f"background:{BG_SURFACE};padding:2px 7px;border-radius:100px;"
                            f"letter-spacing:0.06em'>{_pdir}</span>"
                            f"<span style='color:{_pc};font-size:0.74rem;font-weight:800;"
                            f"font-variant-numeric:tabular-nums;white-space:nowrap;"
                            f"min-width:72px;text-align:right'>"
                            f"${_ppnl:+,.2f}</span>"
                            f"<span style='color:{_pc};font-size:0.64rem;font-weight:700;"
                            f"font-variant-numeric:tabular-nums;white-space:nowrap;"
                            f"min-width:46px;text-align:right'>"
                            f"{_pppct:+.1f}%</span>"
                            f"</div>", unsafe_allow_html=True)
                    if len(_port_open_list) > 4:
                        st.markdown(
                            f"<div style='color:{TEXT_MUTED};font-size:0.6rem;"
                            f"margin-top:3px;text-align:right'>"
                            f"+ {len(_port_open_list) - 4} more</div>",
                            unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div style='color:{TEXT_MUTED};font-size:0.7rem;"
                        f"padding:16px;text-align:center;background:{BG_CARD};"
                        f"border:1px dashed {BORDER};border-radius:8px'>"
                        f"No open positions.</div>", unsafe_allow_html=True)

            # Recent closes
            with _oc_r:
                _closed = _pstats.get("closed_trades", []) if isinstance(_pstats, dict) else []
                st.markdown(
                    f"<div style='color:{CYAN};font-size:0.52rem;font-weight:800;"
                    f"text-transform:uppercase;letter-spacing:0.14em;margin-bottom:8px'>"
                    f"Recent Closes &nbsp;"
                    f"<span style='color:{TEXT_MUTED};font-weight:500;"
                    f"letter-spacing:0.02em;text-transform:none'>"
                    f"· {_port_closed} total</span></div>",
                    unsafe_allow_html=True)
                if _closed:
                    _recent_closed = list(reversed(_closed))[:4]
                    for _tc in _recent_closed:
                        _tpnl = _tc.get("realised_pnl", 0)
                        _tpct = _tc.get("realised_pct", 0)
                        _tc_clr = CHART_UP if _tpnl >= 0 else CHART_DOWN
                        _tresult_ok = _tc.get("correct_direction", False)
                        _tmark = "✓" if _tresult_ok else "✗"
                        _tmark_clr = CHART_UP if _tresult_ok else CHART_DOWN
                        _texit = _tc.get("exit_date", "—")
                        try:
                            _texit_short = datetime.strptime(
                                _texit, "%Y-%m-%d"
                            ).strftime("%b %d")
                        except Exception:
                            _texit_short = _texit
                        st.markdown(
                            f"<div style='background:{BG_CARD};border:1px solid {BORDER};"
                            f"border-radius:8px;padding:8px 12px;margin-bottom:3px;"
                            f"display:flex;align-items:center;gap:8px;"
                            f"border-left:3px solid {_tmark_clr}'>"
                            f"<span style='flex:1;color:{TEXT_PRIMARY};font-weight:800;"
                            f"font-size:0.78rem'>{_tc.get('symbol', '')}</span>"
                            f"<span style='color:{TEXT_MUTED};font-size:0.6rem;"
                            f"font-variant-numeric:tabular-nums;min-width:42px'>"
                            f"{_texit_short}</span>"
                            f"<span style='color:{_tc_clr};font-size:0.74rem;font-weight:800;"
                            f"font-variant-numeric:tabular-nums;white-space:nowrap;"
                            f"min-width:72px;text-align:right'>"
                            f"${_tpnl:+,.2f}</span>"
                            f"<span style='color:{_tc_clr};font-size:0.64rem;font-weight:700;"
                            f"font-variant-numeric:tabular-nums;white-space:nowrap;"
                            f"min-width:46px;text-align:right'>"
                            f"{_tpct:+.1f}%</span>"
                            f"<span style='color:{_tmark_clr};font-size:0.85rem;"
                            f"font-weight:900;width:14px;text-align:center'>{_tmark}</span>"
                            f"</div>", unsafe_allow_html=True)
                    if len(_closed) > 4:
                        st.markdown(
                            f"<div style='color:{TEXT_MUTED};font-size:0.6rem;"
                            f"margin-top:3px;text-align:right'>"
                            f"+ {len(_closed) - 4} more</div>",
                            unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div style='color:{TEXT_MUTED};font-size:0.7rem;"
                        f"padding:16px;text-align:center;background:{BG_CARD};"
                        f"border:1px dashed {BORDER};border-radius:8px'>"
                        f"No closed trades yet.</div>", unsafe_allow_html=True)

        elif _pid == "actionable":
            if not _wl_list:
                st.markdown(
                    f"<div style='color:{TEXT_MUTED};font-size:0.75rem;"
                    f"padding:16px;text-align:center;background:{BG_CARD};"
                    f"border:1px dashed {BORDER};border-radius:8px'>"
                    f"Your watchlist is empty. Add stocks on the "
                    f"<b style='color:{CYAN}'>Watchlist</b> page to see signals here."
                    f"</div>", unsafe_allow_html=True)
            elif _sig_actionable == 0:
                st.markdown(
                    f"<div style='color:{TEXT_MUTED};font-size:0.75rem;"
                    f"padding:16px;text-align:center;background:{BG_CARD};"
                    f"border:1px dashed {BORDER};border-radius:8px;line-height:1.55'>"
                    f"<b style='color:{CYAN};font-size:0.85rem'>All Clear</b><br>"
                    f"All {_sig_hold} stocks in your watchlist are on "
                    f"<b style='color:{TEXT_SECONDARY}'>HOLD</b> today — no BUY or SELL "
                    f"signals with high enough confidence.<br>"
                    f"<span style='font-size:0.68rem'>Check back tomorrow — signals refresh daily.</span>"
                    f"</div>", unsafe_allow_html=True)
            else:
                # List the actionable signals we already scanned above
                st.markdown(
                    f"<div style='color:{TEXT_MUTED};font-size:0.52rem;font-weight:700;"
                    f"text-transform:uppercase;letter-spacing:0.14em;margin-bottom:8px'>"
                    f"Ranked by confidence</div>",
                    unsafe_allow_html=True)
                # Re-scan to get full signal objects (was just counts earlier)
                _act_items = []
                for _ws in _wl_list[:12]:
                    try:
                        _sg = get_quick_signal(_ws)
                        if not _sg.get("error") and _sg.get("signal") in ("BUY", "SELL"):
                            _act_items.append(_sg)
                    except Exception:
                        pass
                _act_items.sort(key=lambda s: -s.get("confidence", 0))
                for _si in _act_items[:8]:
                    _sg_dir = _si.get("signal", "HOLD")
                    _sg_dir_clr = CHART_UP if _sg_dir == "BUY" else CHART_DOWN
                    _sg_conf = _si.get("confidence", 0)
                    _sg_prc = _si.get("price", 0)
                    _sg_chg = _si.get("change_1d", 0)
                    _sg_chg_clr = CHART_UP if _sg_chg >= 0 else CHART_DOWN
                    st.markdown(
                        f"<div style='background:{BG_CARD};border:1px solid {BORDER};"
                        f"border-radius:8px;padding:8px 14px;margin-bottom:3px;"
                        f"display:flex;align-items:center;gap:10px;"
                        f"border-left:3px solid {_sg_dir_clr}'>"
                        f"<span style='flex:1;color:{TEXT_PRIMARY};font-weight:800;"
                        f"font-size:0.82rem'>{_si.get('symbol', '')}</span>"
                        f"<span style='color:{_sg_dir_clr};font-size:0.58rem;font-weight:800;"
                        f"background:{BG_SURFACE};padding:2px 8px;border-radius:100px;"
                        f"letter-spacing:0.06em;border:1px solid {_sg_dir_clr}40'>{_sg_dir}</span>"
                        f"<span style='color:{TEXT_SECONDARY};font-size:0.74rem;"
                        f"font-variant-numeric:tabular-nums'>${_sg_prc:,.2f}</span>"
                        f"<span style='color:{_sg_chg_clr};font-size:0.72rem;font-weight:800;"
                        f"font-variant-numeric:tabular-nums;white-space:nowrap;min-width:52px;text-align:right'>"
                        f"{_sg_chg:+.2f}%</span>"
                        f"<div style='display:flex;align-items:center;gap:6px;min-width:70px'>"
                        f"<div style='flex:1;height:4px;background:{BG_SURFACE};border-radius:3px;"
                        f"overflow:hidden;min-width:40px'>"
                        f"<div style='height:100%;width:{max(8, _sg_conf)}%;background:{_sg_dir_clr};"
                        f"border-radius:3px'></div>"
                        f"</div>"
                        f"<span style='color:{_sg_dir_clr};font-size:0.64rem;font-weight:800;"
                        f"font-variant-numeric:tabular-nums'>{_sg_conf:.0f}%</span>"
                        f"</div>"
                        f"</div>", unsafe_allow_html=True)

        elif _pid == "expiration":
            # Per-horizon expiration accuracy from prediction_logger's per_horizon
            _per_h = _pa.get("per_horizon", {})
            HORDER = ["3 Day", "1 Week", "1 Month", "1 Quarter", "1 Year"]
            _h_cols = st.columns(len(HORDER))
            for _hi, _hn in enumerate(HORDER):
                _hd = _per_h.get(_hn, {})
                _hf = _hd.get("scored", 0)
                _ha = _hd.get("accuracy", 0)
                with _h_cols[_hi]:
                    if _hf > 0:
                        _clr = GREEN if _ha >= 55 else AMBER if _ha >= 50 else RED
                        _vtxt = f"{_ha:.0f}%"
                    else:
                        _clr = TEXT_MUTED
                        _vtxt = "—"
                    st.markdown(
                        f"<div style='background:{BG_CARD};border:1px solid {BORDER};"
                        f"border-radius:10px;padding:10px 8px;text-align:center'>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.48rem;font-weight:700;"
                        f"text-transform:uppercase;letter-spacing:0.12em;"
                        f"margin-bottom:4px'>{_hn}</div>"
                        f"<div style='font-family:\"Inter\",sans-serif;color:{_clr};"
                        f"font-size:1.1rem;font-weight:900;line-height:1;"
                        f"font-variant-numeric:tabular-nums'>{_vtxt}</div>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.5rem;margin-top:3px;"
                        f"font-variant-numeric:tabular-nums'>{_hf} matured</div>"
                        f"</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='color:{TEXT_MUTED};font-size:0.7rem;line-height:1.55;"
                f"padding:8px 12px;background:{BG_CARD};border-radius:8px'>"
                f"<b style='color:{TEXT_SECONDARY}'>{_exp_matured}</b> predictions "
                f"have matured across all horizons so far. The headline "
                f"<b style='color:{_exp_color}'>{_exp_win:.0f}%</b> rate is "
                f"their combined direction accuracy at expiration.<br>"
                f"For the full honesty breakdown + Target Hit / Checkpoint reads, "
                f"open the <b style='color:{CYAN}'>Track Record</b> page."
                f"</div>",
                unsafe_allow_html=True)

        elif _pid == "last10":
            # Show the actual last 10 matured predictions
            _scored_full = [
                p for p in _pa.get("predictions_table", [])
                if p.get("final_result") in ("✓", "✗")
            ]
            _scored_full.sort(key=lambda p: p.get("date", ""), reverse=True)
            _last10_items = _scored_full[:10]

            if not _last10_items:
                st.markdown(
                    f"<div style='color:{TEXT_MUTED};font-size:0.72rem;"
                    f"padding:14px;text-align:center;background:{BG_CARD};"
                    f"border:1px dashed {BORDER};border-radius:8px'>"
                    f"No matured predictions yet — the first ones will appear here "
                    f"as their scheduled horizons arrive."
                    f"</div>", unsafe_allow_html=True)
            else:
                # Column header
                _hdr_s = (f"color:{TEXT_MUTED};font-size:0.48rem;font-weight:700;"
                          f"text-transform:uppercase;letter-spacing:0.12em")
                st.markdown(
                    f"<div style='display:flex;align-items:center;padding:4px 14px;"
                    f"gap:4px;margin-bottom:4px'>"
                    f"<div style='flex:0.9;{_hdr_s}'>Stock · Date</div>"
                    f"<div style='flex:0.7;{_hdr_s}'>Horizon</div>"
                    f"<div style='flex:0.7;{_hdr_s};text-align:right'>Predicted</div>"
                    f"<div style='flex:0.7;{_hdr_s};text-align:right'>Actual</div>"
                    f"<div style='flex:0.3;{_hdr_s};text-align:center'>Result</div>"
                    f"</div>", unsafe_allow_html=True)
                for _lp in _last10_items:
                    _lsym = _lp.get("symbol", "?")
                    _ldate = _lp.get("date", "")
                    try:
                        _ldate_short = datetime.strptime(_ldate, "%Y-%m-%d").strftime("%b %d")
                    except Exception:
                        _ldate_short = _ldate
                    _lhz = _lp.get("horizon", "")
                    _lpred = _lp.get("predicted_return", 0)
                    _lresult = _lp.get("final_result", "—")
                    _lclr = GREEN if _lresult == "✓" else RED
                    _lpred_clr = GREEN if _lpred >= 0 else RED
                    _lact = None
                    for _iv in ["365d", "180d", "90d", "60d", "30d", "14d", "7d", "3d", "1d"]:
                        _rv = _lp.get(f"return_{_iv}")
                        if _rv is not None:
                            _lact = _rv
                            break
                    _lact_str = f"{_lact:+.1f}%" if _lact is not None else "—"
                    _lact_clr = GREEN if _lact and _lact > 0 else RED if _lact and _lact < 0 else TEXT_MUTED
                    st.markdown(
                        f"<div style='background:{BG_CARD};border:1px solid {BORDER};"
                        f"border-radius:8px;padding:8px 14px;margin-bottom:3px;"
                        f"display:flex;align-items:center;gap:4px;"
                        f"border-left:3px solid {_lclr}'>"
                        f"<div style='flex:0.9'>"
                        f"<div style='color:{TEXT_PRIMARY};font-weight:800;font-size:0.78rem'>{_lsym}</div>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.58rem;"
                        f"margin-top:1px'>{_ldate_short}</div>"
                        f"</div>"
                        f"<div style='flex:0.7;color:{TEXT_SECONDARY};font-size:0.66rem;"
                        f"font-weight:600'>{_lhz}</div>"
                        f"<div style='flex:0.7;text-align:right;color:{_lpred_clr};"
                        f"font-weight:800;font-size:0.72rem;font-variant-numeric:tabular-nums'>"
                        f"{_lpred:+.1f}%</div>"
                        f"<div style='flex:0.7;text-align:right;color:{_lact_clr};"
                        f"font-weight:800;font-size:0.72rem;font-variant-numeric:tabular-nums'>"
                        f"{_lact_str}</div>"
                        f"<div style='flex:0.3;text-align:center;font-size:0.9rem;"
                        f"color:{_lclr};font-weight:900'>{_lresult}</div>"
                        f"</div>", unsafe_allow_html=True)

        # Close the expand-panel div
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='height:20px'></div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # MARKET PULSE — rich tiles with sparklines + contextual deltas
    # ═════════════════════════════════════════════════════════════════════════
    try:
        import yfinance as _yf_dash

        # SPY — 30 days for trend context, but use last 5 for sparkline
        _spy_tk = _yf_dash.Ticker("SPY")
        _spy_hist = _spy_tk.history(period="1mo")
        _spy_price = float(_spy_hist["Close"].iloc[-1]) if len(_spy_hist) > 0 else 0
        _spy_chg = float(_spy_hist["Close"].pct_change().iloc[-1] * 100) if len(_spy_hist) > 1 else 0
        _spy_5d = _spy_hist["Close"].tail(5).tolist() if len(_spy_hist) >= 5 else []
        _spy_30d_high = float(_spy_hist["Close"].max()) if len(_spy_hist) > 0 else 0
        _spy_30d_low = float(_spy_hist["Close"].min()) if len(_spy_hist) > 0 else 0
        _spy_pos_pct = ((_spy_price - _spy_30d_low)
                        / max(_spy_30d_high - _spy_30d_low, 0.01)) * 100

        # VIX — 30 days for context
        _vix_tk = _yf_dash.Ticker("^VIX")
        _vix_hist = _vix_tk.history(period="1mo")
        _vix_val = float(_vix_hist["Close"].iloc[-1]) if len(_vix_hist) > 0 else 0
        _vix_5d = _vix_hist["Close"].tail(5).tolist() if len(_vix_hist) >= 5 else []
        _vix_20d_avg = (float(_vix_hist["Close"].tail(20).mean())
                        if len(_vix_hist) >= 20 else _vix_val)
        _vix_chg_vs_avg = (((_vix_val - _vix_20d_avg) / _vix_20d_avg) * 100
                           if _vix_20d_avg > 0 else 0)

        # 10Y Treasury yield — bond/rate context
        _tnx_val = None
        _tnx_chg = 0
        _tnx_5d = []
        try:
            _tnx_tk = _yf_dash.Ticker("^TNX")
            _tnx_hist = _tnx_tk.history(period="1mo")
            if len(_tnx_hist) > 0:
                # ^TNX is quoted directly as yield percent (e.g. 4.32 for 4.32%).
                # Sanity-check: if value looks like "yield × 10" (> 15 and < 100),
                # scale it back — handles older yfinance quirks gracefully.
                _raw_tnx = float(_tnx_hist["Close"].iloc[-1])
                _tnx_val = _raw_tnx / 10 if 15 < _raw_tnx < 100 else _raw_tnx
                _raw_5d = _tnx_hist["Close"].tail(5).tolist()
                _tnx_5d = [v / 10 if 15 < v < 100 else v for v in _raw_5d]
                if len(_tnx_hist) > 1:
                    _tnx_chg = float(
                        (_tnx_hist["Close"].iloc[-1] - _tnx_hist["Close"].iloc[-2])
                        / _tnx_hist["Close"].iloc[-2] * 100
                    )
        except Exception:
            pass

        # Fear & Greed
        _fg_score = 50
        _fg_label = "Neutral"
        try:
            _fg = fetch_fear_greed()
            _fg_score = _fg.get("score", 50)
            _fg_label = _fg.get("rating", "Neutral")
        except Exception:
            pass

        # Regime detection on SPY — returns label + score_norm (0..1 strength)
        _regime_label = "Unknown"
        _regime_strength = 0.5   # 0..1 normalized strength of the regime call
        try:
            _spy_df_long = _yf_dash.Ticker("SPY").history(period="1y")
            if len(_spy_df_long) > 200:
                _regime_info = detect_regime(_spy_df_long)
                _regime_label = _regime_info.get("label", "Unknown")
                _regime_strength = float(_regime_info.get("score_norm", 0.5) or 0.5)
        except Exception:
            pass
        # Convert strength to a readable label — "Strong / Moderate / Weak"
        # based on how far from the neutral 0.5 midpoint.
        _strength_dist = abs(_regime_strength - 0.5) * 2  # 0..1
        if _strength_dist >= 0.6:
            _regime_strength_label = "Strong"
        elif _strength_dist >= 0.3:
            _regime_strength_label = "Moderate"
        else:
            _regime_strength_label = "Weak"

        # Color logic
        _spy_chg_clr = CHART_UP if _spy_chg >= 0 else CHART_DOWN
        _vix_clr = CHART_UP if _vix_val < 18 else AMBER if _vix_val < 25 else CHART_DOWN
        _fg_clr = CHART_UP if _fg_score > 60 else CHART_DOWN if _fg_score < 40 else AMBER
        _regime_clr = CHART_UP if _regime_label == "Bull" else CHART_DOWN if _regime_label == "Bear" else AMBER
        _tnx_chg_clr = CHART_UP if _tnx_chg >= 0 else CHART_DOWN

        # ── Inline SVG sparkline helper ─────────────────────────────────
        def _mini_sparkline(vals, color, w=120, h=28, with_dot=True):
            if not vals or len(vals) < 2:
                return ""
            vmin, vmax = min(vals), max(vals)
            rng = (vmax - vmin) if vmax > vmin else 1
            pad = 1.5
            def _map(i, v):
                x = pad + (i / (len(vals) - 1)) * (w - 2 * pad)
                y = (h - pad) - ((v - vmin) / rng) * (h - 2 * pad)
                return x, y
            pts = [_map(i, v) for i, v in enumerate(vals)]
            pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            # Area fill = line + close path along bottom
            area_pts = pts_str + f" {pts[-1][0]:.1f},{h} {pts[0][0]:.1f},{h}"
            # Color without the "#" for inline fill-opacity
            dot_html = ""
            if with_dot:
                lx, ly = pts[-1]
                dot_html = (
                    f"<circle cx='{lx:.1f}' cy='{ly:.1f}' r='2.5' "
                    f"fill='{color}' stroke='rgba(255,255,255,0.9)' stroke-width='1'/>"
                )
            return (
                f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' "
                f"style='display:block;margin:8px auto 6px' preserveAspectRatio='none'>"
                f"<polygon points='{area_pts}' fill='{color}' fill-opacity='0.12' "
                f"stroke='none'/>"
                f"<polyline points='{pts_str}' fill='none' stroke='{color}' "
                f"stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'/>"
                f"{dot_html}"
                f"</svg>"
            )

        # ── Contextual one-liners ───────────────────────────────────────
        # SPY context
        if _spy_pos_pct >= 90:
            _spy_ctx = "At 30d high"
        elif _spy_pos_pct >= 70:
            _spy_ctx = "Near 30d high"
        elif _spy_pos_pct <= 10:
            _spy_ctx = "At 30d low"
        elif _spy_pos_pct <= 30:
            _spy_ctx = "Near 30d low"
        else:
            _spy_ctx = f"{_spy_pos_pct:.0f}% of 30d range"

        # VIX context
        if _vix_val < 15:
            _vix_band = "Complacent"
        elif _vix_val < 20:
            _vix_band = "Calm"
        elif _vix_val < 25:
            _vix_band = "Moderate"
        elif _vix_val < 30:
            _vix_band = "Elevated"
        else:
            _vix_band = "Stressed"
        _vix_ctx = (f"{_vix_band} · vs 20d avg {_vix_20d_avg:.1f}"
                    if _vix_20d_avg else _vix_band)

        # Fear & Greed context
        _fg_ctx = _fg_label.title()

        # Regime context — show trend-strength label instead of a bogus
        # "days in phase" number (which wasn't being populated reliably).
        _regime_ctx = f"{_regime_strength_label} · {int(_strength_dist * 100)}% conviction"

        # 10Y yield context
        if _tnx_val is None:
            _tnx_ctx = "rates"
        elif _tnx_chg >= 0.5:
            _tnx_ctx = "rising sharply"
        elif _tnx_chg >= 0.1:
            _tnx_ctx = "drifting higher"
        elif _tnx_chg <= -0.5:
            _tnx_ctx = "falling sharply"
        elif _tnx_chg <= -0.1:
            _tnx_ctx = "drifting lower"
        else:
            _tnx_ctx = "stable"

        st.markdown(
            "<div class='sec-head' style='margin-bottom:14px'><span class='sec-bar'></span>"
            "Market Pulse</div>",
            unsafe_allow_html=True)

        # 5-column grid (added 10Y Yield for rate context)
        _mp_cols = st.columns(5, gap="small")
        # Flex column with fixed min-height so all 5 tiles align perfectly
        _mp_card = (
            f"background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
            f"border:1px solid {BORDER};border-radius:12px;padding:14px 16px;"
            f"text-align:center;position:relative;overflow:hidden;"
            f"display:flex;flex-direction:column;min-height:168px;"
            f"box-sizing:border-box"
        )
        _mp_hairline = (
            f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
            f"background:{BRAND_GRAD};opacity:0.6'></div>"
        )
        _mp_label = (
            f"color:{CYAN};font-size:0.54rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.14em;margin-bottom:6px"
        )
        _mp_value = (
            "font-family:'Inter',sans-serif;font-size:1.15rem;font-weight:900;"
            "letter-spacing:-0.03em;font-variant-numeric:tabular-nums;line-height:1"
        )
        _mp_delta = (
            "font-size:0.68rem;font-weight:700;margin-top:4px;"
            "font-variant-numeric:tabular-nums"
        )
        # margin-top:auto pushes the context line to the card bottom so all
        # 5 tiles have the same footer position regardless of middle content.
        _mp_ctx = (
            f"color:{TEXT_MUTED};font-size:0.6rem;margin-top:auto;padding-top:6px;"
            f"font-weight:500;letter-spacing:0.01em;line-height:1.3"
        )

        # ── SPY tile ────────────────────────────────────────────────────
        with _mp_cols[0]:
            _spy_spark = _mini_sparkline(_spy_5d, _spy_chg_clr)
            st.markdown(
                f"<div style='{_mp_card}'>{_mp_hairline}"
                f"<div style='{_mp_label}'>S&amp;P 500 · SPY</div>"
                f"<div style='{_mp_value};color:{TEXT_PRIMARY}'>${_spy_price:,.2f}</div>"
                f"<div style='{_mp_delta};color:{_spy_chg_clr}'>"
                f"{'▲' if _spy_chg >= 0 else '▼'} {_spy_chg:+.2f}%</div>"
                f"{_spy_spark}"
                f"<div style='{_mp_ctx}'>{_spy_ctx}</div>"
                f"</div>", unsafe_allow_html=True)

        # ── VIX tile ────────────────────────────────────────────────────
        with _mp_cols[1]:
            _vix_spark = _mini_sparkline(_vix_5d, _vix_clr)
            _vix_delta_str = (f"{'▲' if _vix_chg_vs_avg >= 0 else '▼'} "
                              f"{_vix_chg_vs_avg:+.1f}% vs avg")
            st.markdown(
                f"<div style='{_mp_card}'>{_mp_hairline}"
                f"<div style='{_mp_label}'>VIX · Volatility</div>"
                f"<div style='{_mp_value};color:{_vix_clr}'>{_vix_val:.1f}</div>"
                f"<div style='{_mp_delta};color:{TEXT_MUTED}'>{_vix_delta_str}</div>"
                f"{_vix_spark}"
                f"<div style='{_mp_ctx}'>{_vix_ctx}</div>"
                f"</div>", unsafe_allow_html=True)

        # ── Fear & Greed tile (no sparkline; scalar gauge instead) ──────
        with _mp_cols[2]:
            _fg_fill = min(100, max(0, _fg_score))
            _fg_gauge = (
                f"<div style='width:100%;height:5px;border-radius:3px;"
                f"background:{BORDER};margin:14px 0 0;overflow:hidden'>"
                f"<div style='height:100%;width:{_fg_fill:.0f}%;"
                f"background:linear-gradient(90deg,{CHART_DOWN},{AMBER},{CHART_UP});"
                f"border-radius:3px'></div>"
                f"</div>"
            )
            st.markdown(
                f"<div style='{_mp_card}'>{_mp_hairline}"
                f"<div style='{_mp_label}'>Fear &amp; Greed</div>"
                f"<div style='{_mp_value};color:{_fg_clr}'>{_fg_score:.0f}</div>"
                f"<div style='{_mp_delta};color:{_fg_clr}'>{_fg_label}</div>"
                f"{_fg_gauge}"
                f"<div style='{_mp_ctx}'>0 = extreme fear, 100 = extreme greed</div>"
                f"</div>", unsafe_allow_html=True)

        # ── 10Y Yield tile ──────────────────────────────────────────────
        with _mp_cols[3]:
            _tnx_display = f"{_tnx_val:.2f}%" if _tnx_val is not None else "—"
            _tnx_spark = (_mini_sparkline(_tnx_5d, _tnx_chg_clr)
                          if _tnx_val is not None else "")
            _tnx_delta_str = (f"{'▲' if _tnx_chg >= 0 else '▼'} {_tnx_chg:+.2f}%"
                              if _tnx_val is not None else "—")
            st.markdown(
                f"<div style='{_mp_card}'>{_mp_hairline}"
                f"<div style='{_mp_label}'>10Y Yield</div>"
                f"<div style='{_mp_value};color:{TEXT_PRIMARY}'>{_tnx_display}</div>"
                f"<div style='{_mp_delta};color:{_tnx_chg_clr}'>{_tnx_delta_str}</div>"
                f"{_tnx_spark}"
                f"<div style='{_mp_ctx}'>Rates {_tnx_ctx}</div>"
                f"</div>", unsafe_allow_html=True)

        # ── Regime tile ─────────────────────────────────────────────────
        with _mp_cols[4]:
            # Simple regime icon (up-arrow for bull, down for bear, dash for side)
            if _regime_label == "Bull":
                _regime_icon = "▲"
            elif _regime_label == "Bear":
                _regime_icon = "▼"
            else:
                _regime_icon = "→"
            st.markdown(
                f"<div style='{_mp_card}'>{_mp_hairline}"
                f"<div style='{_mp_label}'>Market Regime</div>"
                f"<div style='{_mp_value};color:{_regime_clr};display:flex;"
                f"align-items:baseline;gap:6px;justify-content:center'>"
                f"<span style='font-size:0.9rem'>{_regime_icon}</span>{_regime_label}</div>"
                f"<div style='{_mp_delta};color:{TEXT_MUTED}'>{_regime_ctx}</div>"
                # Simple strength-bar as bottom visual instead of sparkline
                f"<div style='width:100%;height:5px;border-radius:3px;background:{BORDER};"
                f"margin:14px 0 0;overflow:hidden'>"
                f"<div style='height:100%;width:{min(100, max(8, int(_strength_dist * 100)))}%;"
                f"background:{_regime_clr};border-radius:3px'></div>"
                f"</div>"
                f"<div style='{_mp_ctx}'>Trend strength</div>"
                f"</div>", unsafe_allow_html=True)

    except Exception:
        pass  # Graceful fail — market pulse is nice-to-have

    st.markdown(f"<div style='height:28px'></div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # PERFORMANCE TRACKERS — accuracy / predictions / portfolio value over time
    # ═════════════════════════════════════════════════════════════════════════
    # Header row: sec-head label + right-aligned segmented range control
    _hd_l, _hd_r = st.columns([3, 2])
    with _hd_l:
        st.markdown(
            "<div class='sec-head' style='margin:8px 0 0'><span class='sec-bar'></span>"
            "Performance Trackers</div>",
            unsafe_allow_html=True)
    with _hd_r:
        _dash_range = st.segmented_control(
            "Time range",
            options=["7D", "1M", "3M", "All"],
            default="All",
            selection_mode="single",
            label_visibility="collapsed",
            key="dash_tracker_range",
        )
        if _dash_range is None:
            _dash_range = "All"

    st.markdown(f"<div style='height:14px'></div>", unsafe_allow_html=True)

    _tr_col1, _tr_col2, _tr_col3 = st.columns(3, gap="medium")

    # ── Shared chart constants ───────────────────────────────────────────────
    _DASH_CHART_HEIGHT = 230
    _DASH_CHART_MARGIN = dict(l=8, r=12, t=18, b=28)
    _DASH_HOVER = dict(
        bgcolor=BG_ELEVATED, font_color=TEXT_PRIMARY,
        bordercolor=BORDER_ACCENT, font_family="Inter, sans-serif",
        font_size=11,
    )

    # ── Range helpers ────────────────────────────────────────────────────────
    _RANGE_DAYS = {"7D": 7, "1M": 31, "3M": 93, "All": None}
    _range_days = _RANGE_DAYS.get(_dash_range)
    _range_cutoff_str = None
    if _range_days is not None:
        from datetime import timedelta as _td_range
        _range_cutoff_str = (_now - _td_range(days=_range_days)).strftime("%Y-%m-%d")

    def _dash_header_html(label, stat_value, stat_color, delta_html, scope="universal"):
        """Render a chart-card header with a scope tag.
        scope: "universal" (cyan PUBLIC pill) | "user" (blue PERSONAL pill)
        """
        if scope == "universal":
            _sc_text, _sc_clr, _sc_bg, _sc_bd = (
                "PUBLIC", CYAN, "rgba(6,214,160,0.10)", "rgba(6,214,160,0.28)"
            )
        else:
            _sc_text, _sc_clr, _sc_bg, _sc_bd = (
                "PERSONAL", BLUE, "rgba(32,128,229,0.10)", "rgba(32,128,229,0.28)"
            )
        _scope_pill = (
            f"<span style='padding:2px 7px;border-radius:100px;"
            f"background:{_sc_bg};border:1px solid {_sc_bd};"
            f"color:{_sc_clr};font-size:0.46rem;font-weight:700;"
            f"letter-spacing:0.12em;text-transform:uppercase'>{_sc_text}</span>"
        )
        return (
            f"<div style='display:flex;align-items:center;justify-content:space-between;"
            f"gap:8px;margin-bottom:4px'>"
            f"<div style='color:{CYAN};font-size:0.58rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.14em'>{label}</div>"
            f"{_scope_pill}"
            f"</div>"
            f"<div style='display:flex;align-items:baseline;gap:10px;margin-bottom:2px'>"
            f"<div style='font-family:\"Inter\",sans-serif;color:{stat_color};"
            f"font-size:1.55rem;font-weight:900;letter-spacing:-0.035em;"
            f"font-variant-numeric:tabular-nums;line-height:1'>{stat_value}</div>"
            f"{delta_html}"
            f"</div>"
        )

    def _dash_delta(value_str, positive):
        _c = CHART_UP if positive else CHART_DOWN
        _arrow = "▲" if positive else "▼"
        _bg = "rgba(6,214,160,0.10)" if positive else "rgba(224,74,74,0.10)"
        _bd = "rgba(6,214,160,0.30)" if positive else "rgba(224,74,74,0.30)"
        return (
            f"<span style='display:inline-flex;align-items:center;gap:4px;"
            f"padding:2px 8px;border-radius:100px;background:{_bg};"
            f"border:1px solid {_bd};color:{_c};font-size:0.62rem;font-weight:700;"
            f"font-variant-numeric:tabular-nums;letter-spacing:0.02em'>"
            f"{_arrow} {value_str}</span>"
        )

    def _dash_empty_html(label, msg):
        return (
            f"<div style='color:{CYAN};font-size:0.58rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.14em;margin-bottom:4px'>"
            f"{label}</div>"
            f"<div style='font-family:\"Inter\",sans-serif;color:{TEXT_MUTED};"
            f"font-size:1.55rem;font-weight:900;letter-spacing:-0.035em;"
            f"line-height:1;margin-bottom:40px'>—</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.72rem;text-align:center;"
            f"line-height:1.55;padding:30px 10px'>{msg}</div>"
        )

    # ── Pull scored predictions (date-ordered ASC for time-series math) ─────
    try:
        _preds_table = _pa.get("predictions_table", [])
        _preds_sorted = sorted(
            [p for p in _preds_table if p.get("date")],
            key=lambda p: p["date"]
        )
    except Exception:
        _preds_sorted = []

    # ──────────────────────────────────────────────────────────────────────
    # 1) MODEL ACCURACY OVER TIME — rolling accuracy with smooth gradient fill
    # ──────────────────────────────────────────────────────────────────────
    with _tr_col1:
        _scored_pts = [p for p in _preds_sorted if p.get("final_result") in ("✓", "✗")]
        if len(_scored_pts) >= 5:
            _dates_acc = pd.to_datetime([p["date"] for p in _scored_pts])
            _correct = [1 if p["final_result"] == "✓" else 0 for p in _scored_pts]
            _acc_series = pd.Series(_correct, index=_dates_acc).sort_index()
            # Window: roughly a third of the sample, clamped to 5–25
            _window = max(5, min(25, len(_scored_pts) // 3))
            _rolling = (_acc_series.rolling(window=_window, min_periods=max(3, _window // 2))
                                   .mean() * 100).dropna()
            # Apply time-range filter
            if _range_cutoff_str is not None:
                _rolling = _rolling[_rolling.index >= pd.to_datetime(_range_cutoff_str)]

            if len(_rolling) >= 2:
                _latest_acc = float(_rolling.iloc[-1])
                _prev_acc = float(_rolling.iloc[0])
                _delta_pp = _latest_acc - _prev_acc
                _acc_stat_clr = (CHART_UP if _latest_acc >= 55
                                 else AMBER if _latest_acc >= 50 else CHART_DOWN)
                _line_clr = CHART_UP if _latest_acc >= 50 else CHART_DOWN

                _delta_html = _dash_delta(f"{abs(_delta_pp):.1f}pp", _delta_pp >= 0)

                # Y-range: hug the data with a 6pp buffer, clamp to 0-100
                _y_lo_acc = max(0.0, float(_rolling.min()) - 6)
                _y_hi_acc = min(100.0, float(_rolling.max()) + 6)

                _fig_acc = go.Figure()
                # Invisible baseline at y_lo — fill above it bounded to visible range
                _fig_acc.add_trace(go.Scatter(
                    x=_rolling.index, y=[_y_lo_acc] * len(_rolling),
                    mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
                    showlegend=False, hoverinfo="skip",
                ))
                _fig_acc.add_trace(go.Scatter(
                    x=_rolling.index, y=_rolling.values,
                    mode="lines",
                    line=dict(color=_line_clr, width=2.4, shape="linear"),
                    fill="tonexty",
                    fillcolor=("rgba(6,214,160,0.18)" if _latest_acc >= 50
                               else "rgba(224,74,74,0.18)"),
                    hovertemplate="<b>%{x|%b %d}</b><br>Accuracy: %{y:.1f}%<extra></extra>",
                    showlegend=False,
                ))
                # Glowing end-point marker
                _fig_acc.add_trace(go.Scatter(
                    x=[_rolling.index[-1]], y=[_latest_acc],
                    mode="markers",
                    marker=dict(
                        size=10, color=_line_clr,
                        line=dict(color="rgba(255,255,255,0.9)", width=2),
                        opacity=1,
                    ),
                    showlegend=False, hoverinfo="skip",
                ))
                # 50% reference
                if _y_lo_acc < 50 < _y_hi_acc:
                    _fig_acc.add_hline(
                        y=50, line=dict(color=BORDER_ACCENT, width=1, dash="dot"))
                _fig_acc.update_layout(
                    height=_DASH_CHART_HEIGHT, margin=_DASH_CHART_MARGIN,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=10),
                    xaxis=dict(
                        showgrid=False, zeroline=False, showline=False,
                        tickfont=dict(color=TEXT_MUTED, size=9), nticks=4,
                    ),
                    yaxis=dict(
                        gridcolor=CHART_GRID, zeroline=False, showline=False,
                        ticksuffix="%", range=[_y_lo_acc, _y_hi_acc],
                        tickfont=dict(color=TEXT_MUTED, size=9), nticks=4,
                    ),
                    hoverlabel=_DASH_HOVER,
                )

                with st.container(border=True):
                    st.markdown(
                        _dash_header_html("Accuracy Trend",
                                          f"{_latest_acc:.1f}%",
                                          _acc_stat_clr,
                                          _delta_html,
                                          scope="universal"),
                        unsafe_allow_html=True)
                    st.plotly_chart(_fig_acc, use_container_width=True,
                                    config={"displayModeBar": False})
            else:
                with st.container(border=True):
                    st.markdown(_dash_empty_html(
                        "Accuracy Trend",
                        "Not enough scored predictions yet.<br>"
                        "Chart fills in as the model runs."), unsafe_allow_html=True)
        else:
            with st.container(border=True):
                st.markdown(_dash_empty_html(
                    "Accuracy Trend",
                    "Not enough scored predictions yet.<br>"
                    "Chart fills in as the model runs."), unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────
    # 2) PREDICTIONS OVER TIME — bars bucketed by range, latest in cyan
    # ──────────────────────────────────────────────────────────────────────
    with _tr_col2:
        _preds_in_range = _preds_sorted
        if _range_cutoff_str is not None:
            _preds_in_range = [p for p in _preds_sorted if p["date"] >= _range_cutoff_str]

        if len(_preds_in_range) >= 2:
            _dates_p = pd.to_datetime([p["date"] for p in _preds_in_range])
            _series = pd.Series(1, index=_dates_p).sort_index()
            # Bucket freq + cap depends on range
            if _dash_range == "7D":
                _bucketed = _series.resample("D").sum().fillna(0).astype(int)
                _bucketed = _bucketed.tail(7)
                _period_name = "Day"
            elif _dash_range == "1M":
                _bucketed = _series.resample("D").sum().fillna(0).astype(int)
                _bucketed = _bucketed.tail(30)
                _period_name = "Day"
            elif _dash_range == "3M":
                _bucketed = _series.resample("W-MON").sum().fillna(0).astype(int)
                _bucketed = _bucketed.tail(13)
                _period_name = "Week"
            else:  # All
                _bucketed = _series.resample("W-MON").sum().fillna(0).astype(int)
                _bucketed = _bucketed.tail(12)
                _period_name = "Week"

            # Bar width in ms — matches the bucket size
            _bw_ms = (86_400_000 * 0.72 if _period_name == "Day"
                      else 86_400_000 * 4.2)

            _weekly = _bucketed  # alias for downstream code
            _latest_week_n = int(_weekly.iloc[-1]) if len(_weekly) > 0 else 0
            _prev_week_n = int(_weekly.iloc[-2]) if len(_weekly) > 1 else 0
            _delta_n = _latest_week_n - _prev_week_n
            _delta_html_p = (_dash_delta(f"{abs(_delta_n)}", _delta_n >= 0)
                             if _prev_week_n > 0 else "")

            # Colors: uniform brand cyan for all bars except the latest which
            # stays full-opacity; older bars slightly faded for subtle depth.
            _n_bars = len(_weekly)
            _bar_colors = [CYAN] * _n_bars
            _bar_opacity = [0.85] * _n_bars
            if _n_bars > 0:
                _bar_opacity[-1] = 1.0  # latest bar = crisp

            _fig_p = go.Figure()
            _bar_hover = (
                "<b>%{x|%b %d}</b><br>%{y} predictions<extra></extra>"
                if _period_name == "Day"
                else "<b>Week of %{x|%b %d}</b><br>%{y} predictions<extra></extra>"
            )
            _fig_p.add_trace(go.Bar(
                x=_weekly.index, y=_weekly.values,
                marker=dict(
                    color=_bar_colors,
                    opacity=_bar_opacity,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                hovertemplate=_bar_hover,
                showlegend=False,
                width=_bw_ms,
            ))
            _fig_p.update_layout(
                height=_DASH_CHART_HEIGHT, margin=_DASH_CHART_MARGIN,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=10),
                bargap=0.35,
                xaxis=dict(
                    showgrid=False, zeroline=False, showline=False,
                    tickfont=dict(color=TEXT_MUTED, size=9), nticks=4,
                ),
                yaxis=dict(
                    gridcolor=CHART_GRID, zeroline=False, showline=False,
                    tickfont=dict(color=TEXT_MUTED, size=9), nticks=4,
                ),
                hoverlabel=_DASH_HOVER,
            )

            _pred_count = len(_preds_in_range)
            with st.container(border=True):
                st.markdown(
                    _dash_header_html("Predictions",
                                      f"{_pred_count:,}",
                                      CYAN,
                                      _delta_html_p,
                                      scope="universal"),
                    unsafe_allow_html=True)
                st.plotly_chart(_fig_p, use_container_width=True,
                                config={"displayModeBar": False})
        else:
            with st.container(border=True):
                st.markdown(_dash_empty_html(
                    "Predictions",
                    "Prediction volume appears<br>after a few runs."),
                    unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────
    # 3) PORTFOLIO VALUE OVER TIME — equity curve, baseline+tonexty fill
    # ──────────────────────────────────────────────────────────────────────
    with _tr_col3:
        _curve = []
        try:
            _curve = _pstats.get("equity_curve", []) or []
        except Exception:
            _curve = []

        if len(_curve) >= 2:
            _curve_df = pd.DataFrame(_curve).copy()
            _curve_df["date"] = pd.to_datetime(_curve_df["date"])
            _curve_df = _curve_df.sort_values("date").reset_index(drop=True)

            # Apply range filter
            if _range_cutoff_str is not None:
                _curve_df_view = _curve_df[
                    _curve_df["date"] >= pd.to_datetime(_range_cutoff_str)
                ].reset_index(drop=True)
                if len(_curve_df_view) < 2:
                    _curve_df_view = _curve_df.tail(2).reset_index(drop=True)
                _curve_df = _curve_df_view

            _eq_start = 10000.0
            _eq_latest = float(_curve_df["equity"].iloc[-1])
            _eq_first = float(_curve_df["equity"].iloc[0])
            _eq_pos = _eq_latest >= _eq_start
            _eq_color = CHART_UP if _eq_pos else CHART_DOWN
            _eq_fill_rgba = ("rgba(6,214,160,0.22)" if _eq_pos
                             else "rgba(224,74,74,0.22)")

            # Delta vs start of visible window (not vs $10k) so the delta
            # reflects what the user is currently looking at.
            _delta_dollar = _eq_latest - _eq_first
            _delta_html_eq = _dash_delta(
                f"${abs(_delta_dollar):,.0f}", _delta_dollar >= 0)

            # Y-range: hug data with a generous padding, keep $10k line visible
            _y_lo_raw = min(_curve_df["equity"].min(), _eq_start)
            _y_hi_raw = max(_curve_df["equity"].max(), _eq_start)
            _y_span = max(_y_hi_raw - _y_lo_raw, _eq_start * 0.02)
            _y_lo_eq = _y_lo_raw - _y_span * 0.15
            _y_hi_eq = _y_hi_raw + _y_span * 0.15

            _fig_eq = go.Figure()
            # Invisible baseline at y_lo for bounded fill
            _fig_eq.add_trace(go.Scatter(
                x=_curve_df["date"],
                y=[_y_lo_eq] * len(_curve_df),
                mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
                showlegend=False, hoverinfo="skip",
            ))
            _fig_eq.add_trace(go.Scatter(
                x=_curve_df["date"], y=_curve_df["equity"],
                mode="lines",
                line=dict(color=_eq_color, width=2.4, shape="linear"),
                fill="tonexty",
                fillcolor=("rgba(6,214,160,0.18)" if _eq_pos
                           else "rgba(224,74,74,0.18)"),
                hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.0f}<extra></extra>",
                showlegend=False,
            ))
            # $10k starting reference — subtle dotted horizontal line
            if _y_lo_eq < _eq_start < _y_hi_eq:
                _fig_eq.add_hline(
                    y=_eq_start,
                    line=dict(color=BORDER_ACCENT, width=1, dash="dot"))
            # Glowing end-point marker
            _fig_eq.add_trace(go.Scatter(
                x=[_curve_df["date"].iloc[-1]], y=[_eq_latest],
                mode="markers",
                marker=dict(
                    size=10, color=_eq_color,
                    line=dict(color="rgba(255,255,255,0.9)", width=2),
                    opacity=1,
                ),
                showlegend=False, hoverinfo="skip",
            ))
            _fig_eq.update_layout(
                height=_DASH_CHART_HEIGHT, margin=_DASH_CHART_MARGIN,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=10),
                xaxis=dict(
                    showgrid=False, zeroline=False, showline=False,
                    tickfont=dict(color=TEXT_MUTED, size=9), nticks=4,
                ),
                yaxis=dict(
                    gridcolor=CHART_GRID, zeroline=False, showline=False,
                    tickprefix="$", tickformat=",",
                    range=[_y_lo_eq, _y_hi_eq],
                    tickfont=dict(color=TEXT_MUTED, size=9), nticks=4,
                ),
                hoverlabel=_DASH_HOVER,
            )

            with st.container(border=True):
                st.markdown(
                    _dash_header_html("Your Portfolio Value",
                                      f"${_eq_latest:,.0f}",
                                      _eq_color,
                                      _delta_html_eq,
                                      scope="user"),
                    unsafe_allow_html=True)
                st.plotly_chart(_fig_eq, use_container_width=True,
                                config={"displayModeBar": False})
        else:
            with st.container(border=True):
                st.markdown(_dash_empty_html(
                    "Portfolio Value",
                    "Your equity curve appears here<br>"
                    "once a few paper trades have closed."),
                    unsafe_allow_html=True)

    st.markdown(f"<div style='height:24px'></div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # EXPIRING TRADES ALERT
    # ═════════════════════════════════════════════════════════════════════════
    if _port_open_list:
        from datetime import timedelta as _td_dash
        _today_str = _now.strftime("%Y-%m-%d")
        _week_str = (_now + _td_dash(days=7)).strftime("%Y-%m-%d")
        _expiring_today = [p for p in _port_open_list if p.get("target_close_date", "") <= _today_str]
        _expiring_week = [p for p in _port_open_list
                          if _today_str < p.get("target_close_date", "") <= _week_str]

        if _expiring_today or _expiring_week:
            _all_expiring = _expiring_today + _expiring_week
            # Deduplicate by symbol
            _seen_exp = set()
            _unique_exp = []
            for _ep in _all_expiring:
                if _ep["symbol"] not in _seen_exp:
                    _seen_exp.add(_ep["symbol"])
                    _unique_exp.append(_ep)

            _summary_parts = []
            if _expiring_today:
                _summary_parts.append(
                    f"<span style='color:{RED};font-weight:700'>{len(_expiring_today)} closing today</span>")
            if _expiring_week:
                _summary_parts.append(
                    f"<span style='color:{AMBER};font-weight:700'>{len(_expiring_week)} this week</span>")

            # Ticker list string
            _ticker_str = ", ".join(ep["symbol"] for ep in _unique_exp[:8])

            _exp_left, _exp_right = st.columns([5, 1])
            with _exp_left:
                st.markdown(
                    f"<div style='background:linear-gradient(145deg,{BG_ELEVATED},{BG_CARD});"
                    f"border:1px solid {AMBER}30;border-radius:10px;padding:14px 18px;"
                    f"display:flex;align-items:center;gap:12px'>"
                    f"<div style='font-size:1.1rem'>⏰</div>"
                    f"<div>"
                    f"<div style='color:{TEXT_PRIMARY};font-size:0.78rem;font-weight:700;margin-bottom:3px'>Expiring Trades</div>"
                    f"<div style='font-size:0.72rem;line-height:1.6'>{' · '.join(_summary_parts)}"
                    f" ({_ticker_str})</div>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True)
            with _exp_right:
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                if st.button("View in Portfolio →", key="exp_goto_portfolio", use_container_width=True):
                    st.session_state.current_page = "Portfolio"
                    st.rerun()

            st.markdown(f"<div style='height:12px'></div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TODAY'S BEST SIGNALS — the hero section
    # ═════════════════════════════════════════════════════════════════════════

    st.markdown(
        f"<div class='sec-head' style='display:flex;align-items:center;gap:10px;margin-bottom:14px'>"
        f"<div class='sec-bar' style='width:3px;height:14px;background:{BRAND_GRAD};border-radius:2px'></div>"
        f"<span style='color:{TEXT_PRIMARY};font-size:0.7rem;font-weight:800;"
        f"text-transform:uppercase;letter-spacing:0.14em'>Today's Best Signals</span>"
        f"<span style='padding:3px 10px;border-radius:100px;"
        f"background:rgba(32,128,229,0.10);border:1px solid rgba(32,128,229,0.28);"
        f"color:{BLUE};font-size:0.5rem;font-weight:800;letter-spacing:0.12em;"
        f"text-transform:uppercase;margin-left:4px'>From your watchlist</span>"
        f"<span style='color:{TEXT_MUTED};font-size:0.62rem;font-weight:500;margin-left:4px'>"
        f"· ranked by confidence</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    _wl = st.session_state.get("watchlist", [])
    if _wl:
        _top_signals = []
        # Fetch up to 20 signals to get a good ranked set
        for _ws in _wl[:20]:
            try:
                _sig = get_quick_signal(_ws)
                if not _sig.get("error"):
                    _top_signals.append(_sig)
            except Exception:
                pass

        # Sort: actionable signals first (BUY/SELL), then by confidence
        _top_signals.sort(
            key=lambda s: (0 if s["signal"] in ("BUY", "SELL") else 1, -s.get("confidence", 0))
        )

        if _top_signals:
            # ── Magnifying glass SVG icon (blue→cyan gradient) ──
            _analyze_svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none">'
                '<defs><linearGradient id="ag" x1="0%" y1="0%" x2="100%" y2="100%">'
                '<stop offset="0%" stop-color="#2080e5"/><stop offset="100%" stop-color="#06d6a0"/>'
                '</linearGradient></defs>'
                '<circle cx="10.5" cy="10.5" r="6.5" stroke="url(#ag)" stroke-width="2.5" fill="none"/>'
                '<line x1="15.5" y1="15.5" x2="21" y2="21" stroke="url(#ag)" stroke-width="2.5" stroke-linecap="round"/>'
                '</svg>'
            )

            # ── Premium compact button CSS ──
            st.markdown(f"""<style>
            [data-testid="stHorizontalBlock"] button[kind="secondary"] {{
                font-size: 0.64rem !important;
                letter-spacing: 0.3px !important;
            }}
            .premium-sig-btn {{
                display: inline-flex; align-items: center; justify-content: center; gap: 5px;
                padding: 5px 12px; border-radius: 6px; font-size: 0.62rem; font-weight: 600;
                cursor: pointer; transition: all 0.2s ease; border: none;
                backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
                letter-spacing: 0.3px; white-space: nowrap; text-decoration: none;
            }}
            .btn-trade {{
                background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
                border: 1px solid rgba(16,185,129,0.3) !important; color: {GREEN};
            }}
            .btn-trade:hover {{ background: linear-gradient(135deg, rgba(16,185,129,0.25), rgba(16,185,129,0.1)); }}
            .btn-view {{
                background: linear-gradient(135deg, rgba(138,152,170,0.12), rgba(138,152,170,0.04));
                border: 1px solid rgba(138,152,170,0.25) !important; color: {TEXT_SECONDARY};
            }}
            .btn-view:hover {{ background: linear-gradient(135deg, rgba(138,152,170,0.2), rgba(138,152,170,0.08)); }}
            .btn-analyze {{
                background: linear-gradient(135deg, rgba(32,128,229,0.15), rgba(6,214,160,0.08));
                border: 1px solid rgba(32,128,229,0.3) !important; color: #4db8ff;
            }}
            .btn-analyze:hover {{ background: linear-gradient(135deg, rgba(32,128,229,0.25), rgba(6,214,160,0.15)); }}
            </style>""", unsafe_allow_html=True)

            # ── Expandable-row state ─────────────────────────────────
            if "dash_signals_expanded" not in st.session_state:
                st.session_state.dash_signals_expanded = set()

            # Table header (no separate Actions column — actions move into
            # the expanded row)
            _hdr_s = (f"color:{TEXT_MUTED};font-size:0.52rem;font-weight:700;"
                      f"text-transform:uppercase;letter-spacing:0.08em")
            st.markdown(
                f"<div style='display:flex;align-items:center;padding:4px 16px;"
                f"gap:4px;margin-bottom:2px'>"
                f"<div style='flex:1.5;{_hdr_s}'>Stock</div>"
                f"<div style='flex:1;{_hdr_s};text-align:right'>Price</div>"
                f"<div style='flex:0.8;{_hdr_s};text-align:right'>Today</div>"
                f"<div style='flex:1.1;{_hdr_s};text-align:center'>Signal · Confidence</div>"
                f"<div style='width:18px;{_hdr_s}'></div>"
                f"</div>", unsafe_allow_html=True)

            for _ts in _top_signals[:8]:
                _s_sym = _ts["symbol"]
                _s_price = _ts.get("price", 0) or 0
                _s_chg = _ts.get("change_1d", 0) or 0
                _s_chg5 = _ts.get("change_5d", 0) or 0
                _s_chg21 = _ts.get("change_21d", 0) or 0
                _s_signal = _ts.get("signal", "HOLD")
                _s_conf = _ts.get("confidence", 0)
                _s_rsi = _ts.get("rsi", 0) or 0
                _s_ma50 = _ts.get("ma50_dist", 0) or 0
                _s_ma200 = _ts.get("ma200_dist", 0) or 0
                _s_vol = _ts.get("vol_ratio", 1) or 1
                # Brand palette (cyan/red, not generic green/red)
                if _s_signal == "BUY":
                    _s_color = CHART_UP
                    _s_bg = "rgba(6,214,160,0.12)"
                elif _s_signal == "SELL":
                    _s_color = CHART_DOWN
                    _s_bg = "rgba(224,74,74,0.12)"
                else:
                    _s_color = AMBER
                    _s_bg = "rgba(245,158,11,0.12)"
                _chg_c = CHART_UP if _s_chg >= 0 else CHART_DOWN
                _pt_dir = "LONG" if _s_signal == "BUY" else ("SHORT" if _s_signal == "SELL" else "LONG")

                # Confidence label — interpret the number for the user
                if _s_conf >= 70:
                    _conf_label = "High"
                elif _s_conf >= 55:
                    _conf_label = "Medium"
                else:
                    _conf_label = "Low"

                _sig_id = f"sig_{_s_sym}"
                _is_sig_expanded = _sig_id in st.session_state.dash_signals_expanded
                _sig_expanded_cls = " is-expanded" if _is_sig_expanded else ""

                # Full-width clickable row — no more split columns for actions
                st.markdown(
                    f"<div class='dash-signal-row{_sig_expanded_cls}' "
                    f"style='background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;"
                    f"padding:10px 14px;display:flex;align-items:center;gap:4px;"
                    f"border-left:3px solid {_s_color};"
                    f"transition:border-color 0.18s ease,background 0.18s ease'>"
                    f"<div style='flex:1.5;display:flex;align-items:center;gap:8px'>"
                    f"<span style='color:{TEXT_PRIMARY};font-weight:800;font-size:0.88rem;"
                    f"font-family:\"Inter\",sans-serif;letter-spacing:-0.01em'>{_s_sym}</span>"
                    f"</div>"
                    f"<div style='flex:1;text-align:right;color:{TEXT_SECONDARY};font-size:0.76rem;"
                    f"font-variant-numeric:tabular-nums'>${_s_price:,.2f}</div>"
                    f"<div style='flex:0.8;text-align:right;color:{_chg_c};font-weight:800;font-size:0.74rem;"
                    f"font-variant-numeric:tabular-nums;white-space:nowrap'>{_s_chg:+.2f}%</div>"
                    f"<div style='flex:1.1;display:flex;align-items:center;gap:8px;justify-content:center'>"
                    f"<span style='color:{_s_color};background:{_s_bg};padding:3px 10px;"
                    f"border-radius:100px;font-size:0.56rem;font-weight:800;letter-spacing:0.08em;"
                    f"border:1px solid {_s_color}40'>{_s_signal}</span>"
                    f"<div style='display:flex;align-items:center;gap:6px;min-width:74px'>"
                    f"<div style='flex:1;height:5px;background:{BG_SURFACE};border-radius:3px;overflow:hidden;"
                    f"min-width:40px'>"
                    f"<div style='height:100%;width:{max(8, _s_conf)}%;background:{_s_color};border-radius:3px'></div>"
                    f"</div>"
                    f"<span style='color:{_s_color};font-size:0.6rem;font-weight:800;"
                    f"font-variant-numeric:tabular-nums;white-space:nowrap'>{_s_conf}%</span>"
                    f"</div>"
                    f"</div>"
                    # Chevron
                    f"<span class='dash-signal-chevron' style='width:18px;color:{TEXT_MUTED};"
                    f"font-size:0.85rem;text-align:center;"
                    f"transform:{'rotate(180deg)' if _is_sig_expanded else 'none'};"
                    f"transition:transform 0.22s ease,color 0.22s ease;"
                    f"{('color:' + CYAN + ';') if _is_sig_expanded else ''}'>▾</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Hidden toggle button — JS below wires the row to click it
                if st.button(
                    "toggle",
                    key=f"sig_toggle_{_s_sym}",
                    use_container_width=True,
                    help=("Click to collapse" if _is_sig_expanded
                          else "Click to see signal details"),
                ):
                    if _is_sig_expanded:
                        st.session_state.dash_signals_expanded.discard(_sig_id)
                    else:
                        st.session_state.dash_signals_expanded.add(_sig_id)
                    st.rerun()

                # ── Expanded detail panel ───────────────────────────────
                if _is_sig_expanded:
                    # RSI interpretation
                    if _s_rsi >= 70:
                        _rsi_tag, _rsi_clr = "overbought", CHART_DOWN
                    elif _s_rsi <= 30:
                        _rsi_tag, _rsi_clr = "oversold", CHART_UP
                    else:
                        _rsi_tag, _rsi_clr = "neutral", TEXT_SECONDARY

                    _ma50_clr = CHART_UP if _s_ma50 >= 0 else CHART_DOWN
                    _ma200_clr = CHART_UP if _s_ma200 >= 0 else CHART_DOWN
                    _chg5_clr = CHART_UP if _s_chg5 >= 0 else CHART_DOWN
                    _chg21_clr = CHART_UP if _s_chg21 >= 0 else CHART_DOWN
                    _vol_tag = ("elevated" if _s_vol >= 1.5
                                else "quiet" if _s_vol <= 0.7 else "normal")
                    _vol_clr = (AMBER if _s_vol >= 1.5
                                else TEXT_MUTED if _s_vol <= 0.7 else TEXT_SECONDARY)

                    st.markdown(
                        f"<div style='background:linear-gradient(180deg,{BG_ELEVATED},{BG_SURFACE});"
                        f"border:1px solid {BORDER_ACCENT};border-top:none;"
                        f"border-radius:0 0 10px 10px;padding:16px 18px 14px;"
                        f"margin-top:-6px;margin-bottom:4px;"
                        f"border-left:3px solid {_s_color}'>"
                        # Top: technical breakdown
                        f"<div style='color:{CYAN};font-size:0.52rem;font-weight:800;"
                        f"text-transform:uppercase;letter-spacing:0.14em;margin-bottom:10px'>"
                        f"Technical Snapshot</div>"
                        f"<div style='display:grid;"
                        f"grid-template-columns:repeat(auto-fit,minmax(90px,1fr));"
                        f"gap:10px;margin-bottom:14px'>"
                        # 5D change
                        f"<div>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                        f"text-transform:uppercase;letter-spacing:0.12em'>5 Day</div>"
                        f"<div style='color:{_chg5_clr};font-family:\"Inter\",sans-serif;"
                        f"font-size:0.95rem;font-weight:900;font-variant-numeric:tabular-nums;"
                        f"margin-top:3px'>{_s_chg5:+.2f}%</div>"
                        f"</div>"
                        # 21D change
                        f"<div>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                        f"text-transform:uppercase;letter-spacing:0.12em'>21 Day</div>"
                        f"<div style='color:{_chg21_clr};font-family:\"Inter\",sans-serif;"
                        f"font-size:0.95rem;font-weight:900;font-variant-numeric:tabular-nums;"
                        f"margin-top:3px'>{_s_chg21:+.2f}%</div>"
                        f"</div>"
                        # RSI
                        f"<div>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                        f"text-transform:uppercase;letter-spacing:0.12em'>RSI</div>"
                        f"<div style='font-family:\"Inter\",sans-serif;font-size:0.95rem;"
                        f"font-weight:900;font-variant-numeric:tabular-nums;margin-top:3px;"
                        f"color:{TEXT_PRIMARY}'>{_s_rsi:.0f}"
                        f"<span style='color:{_rsi_clr};font-size:0.56rem;font-weight:700;"
                        f"margin-left:5px;font-variant-numeric:normal'>{_rsi_tag}</span></div>"
                        f"</div>"
                        # MA50 distance
                        f"<div>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                        f"text-transform:uppercase;letter-spacing:0.12em'>vs MA50</div>"
                        f"<div style='color:{_ma50_clr};font-family:\"Inter\",sans-serif;"
                        f"font-size:0.95rem;font-weight:900;font-variant-numeric:tabular-nums;"
                        f"margin-top:3px'>{_s_ma50:+.2f}%</div>"
                        f"</div>"
                        # MA200 distance
                        f"<div>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                        f"text-transform:uppercase;letter-spacing:0.12em'>vs MA200</div>"
                        f"<div style='color:{_ma200_clr};font-family:\"Inter\",sans-serif;"
                        f"font-size:0.95rem;font-weight:900;font-variant-numeric:tabular-nums;"
                        f"margin-top:3px'>{_s_ma200:+.2f}%</div>"
                        f"</div>"
                        # Volume ratio
                        f"<div>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                        f"text-transform:uppercase;letter-spacing:0.12em'>Volume</div>"
                        f"<div style='font-family:\"Inter\",sans-serif;font-size:0.95rem;"
                        f"font-weight:900;font-variant-numeric:tabular-nums;margin-top:3px;"
                        f"color:{TEXT_PRIMARY}'>{_s_vol:.1f}×"
                        f"<span style='color:{_vol_clr};font-size:0.56rem;font-weight:700;"
                        f"margin-left:5px;font-variant-numeric:normal'>{_vol_tag}</span></div>"
                        f"</div>"
                        f"</div>"
                        # Signal summary line
                        f"<div style='color:{TEXT_SECONDARY};font-size:0.72rem;line-height:1.55;"
                        f"padding:8px 12px;background:{BG_CARD};border-radius:6px;"
                        f"border-left:2px solid {_s_color}'>"
                        f"<b style='color:{TEXT_PRIMARY}'>{_s_sym}</b> signal is "
                        f"<b style='color:{_s_color}'>{_s_signal}</b> "
                        f"with <b style='color:{_s_color}'>{_conf_label.lower()} confidence</b> "
                        f"({_s_conf}%). Direction bias for the simulated trade: "
                        f"<b style='color:{CHART_UP if _pt_dir == 'LONG' else CHART_DOWN}'>{_pt_dir}</b>."
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True)

                    # Premium pill-style action buttons — HTML with hidden
                    # Streamlit button overlays (same pattern as the card
                    # click-expand). JS wires `.dash-sig-action` clicks to
                    # the matching hidden button.
                    st.markdown(
                        f"<div class='dash-signal-actions'>"
                        f"<button class='dash-sig-action dash-sig-action-trade' "
                        f"data-sig-sym='{_s_sym}' data-sig-action='trade'>"
                        f"<span class='dash-sig-action-icon'>+</span>"
                        f"Paper Trade</button>"
                        f"<button class='dash-sig-action dash-sig-action-view' "
                        f"data-sig-sym='{_s_sym}' data-sig-action='view'>"
                        f"View</button>"
                        f"<button class='dash-sig-action dash-sig-action-analyze' "
                        f"data-sig-sym='{_s_sym}' data-sig-action='analyze'>"
                        f"Full Analysis"
                        f"<span class='dash-sig-action-arrow'>→</span>"
                        f"</button>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Hidden Streamlit buttons that actually trigger the
                    # state changes — marked with label "sig-action" so our
                    # global JS hider catches them too.
                    if st.button("sig-action", key=f"dash_trade_{_s_sym}",
                                 use_container_width=True):
                        st.session_state.paper_trade_prefill = {
                            "symbol": _s_sym,
                            "direction": _pt_dir,
                            "confidence": _s_conf,
                        }
                        st.session_state.current_page = "Portfolio"
                        st.rerun()
                    if st.button("sig-action", key=f"dash_view_{_s_sym}",
                                 use_container_width=True):
                        st.session_state.detail_symbol = _s_sym
                        st.session_state.detail_return_page = "Dashboard"
                        st.session_state.current_page = "Stock Detail"
                        st.rerun()
                    if st.button("sig-action", key=f"dash_analyze_{_s_sym}",
                                 use_container_width=True):
                        st.session_state.selected_ticker = _s_sym
                        st.session_state.current_page = "Analyze"
                        st.session_state.auto_analyze = True
                        st.rerun()
                    st.markdown(f"<div style='height:6px'></div>",
                                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='color:{TEXT_MUTED};font-size:0.8rem;padding:20px'>Loading signals…</div>",
                unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
            f"border:1px solid {BORDER};border-radius:14px;padding:36px 24px;text-align:center;"
            f"position:relative;overflow:hidden'>"
            f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
            f"background:{BRAND_GRAD};opacity:0.55'></div>"
            f"<div style='font-size:1.4rem;margin-bottom:8px;opacity:0.55'>👀</div>"
            f"<div style='color:{TEXT_PRIMARY};font-size:0.92rem;font-weight:700;margin-bottom:4px'>"
            f"No watchlist yet</div>"
            f"<div style='color:{TEXT_SECONDARY};font-size:0.75rem;max-width:400px;margin:0 auto;line-height:1.5'>"
            f"Add stocks on the <span style='color:{CYAN};font-weight:700'>Watchlist</span> page "
            f"to see daily BUY/SELL signals ranked by confidence — right here on your Dashboard.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"<div style='height:24px'></div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # YOUR PORTFOLIO (compact glance) + RECENT RESULTS — side by side
    # ═════════════════════════════════════════════════════════════════════════

    _dash_left, _dash_right = st.columns([1, 1])

    # ── Portfolio Glance ─────────────────────────────────────────────────────
    with _dash_left:
        st.markdown(
            f"<div class='sec-head' style='display:flex;align-items:center;gap:10px;margin-bottom:14px'>"
            f"<div class='sec-bar' style='width:3px;height:14px;background:{BRAND_GRAD};border-radius:2px'></div>"
            f"<span style='color:{TEXT_PRIMARY};font-size:0.7rem;font-weight:800;"
            f"text-transform:uppercase;letter-spacing:0.14em'>Your Portfolio</span>"
            f"<span style='padding:3px 10px;border-radius:100px;"
            f"background:rgba(32,128,229,0.10);border:1px solid rgba(32,128,229,0.28);"
            f"color:{BLUE};font-size:0.5rem;font-weight:800;letter-spacing:0.12em;"
            f"text-transform:uppercase;margin-left:4px'>Personal</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Compact stats
        _wr_color = GREEN if _port_win_rate >= 50 else RED if _port_closed > 0 else TEXT_MUTED
        _wr_str = f"{_port_win_rate:.0f}%" if _port_closed > 0 else "—"
        st.markdown(
            f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});border:1px solid {BORDER};"
            f"border-radius:12px;padding:18px 20px;margin-bottom:10px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px'>"
            f"<div>"
            f"<div style='color:{TEXT_PRIMARY};font-size:1.5rem;font-weight:900'>${_port_val:,.2f}</div>"
            f"<div style='color:{_pnl_color};font-size:0.75rem;font-weight:700;margin-top:2px'>"
            f"{_pnl_arrow} ${abs(_port_pnl_dollar):,.2f} ({_port_pnl_pct:+.1f}%)</div>"
            f"</div>"
            f"<div style='display:flex;gap:20px'>"
            f"<div style='text-align:center'>"
            f"<div style='color:{TEXT_MUTED};font-size:0.48rem;font-weight:600;text-transform:uppercase;letter-spacing:0.8px'>Open</div>"
            f"<div style='color:{TEXT_PRIMARY};font-size:1rem;font-weight:800;margin-top:2px'>{_open_pos}</div>"
            f"</div>"
            f"<div style='text-align:center'>"
            f"<div style='color:{TEXT_MUTED};font-size:0.48rem;font-weight:600;text-transform:uppercase;letter-spacing:0.8px'>Win Rate</div>"
            f"<div style='color:{_wr_color};font-size:1rem;font-weight:800;margin-top:2px'>{_wr_str}</div>"
            f"</div>"
            f"<div style='text-align:center'>"
            f"<div style='color:{TEXT_MUTED};font-size:0.48rem;font-weight:600;text-transform:uppercase;letter-spacing:0.8px'>Cash</div>"
            f"<div style='color:{GREEN if _port_cash > 1000 else AMBER if _port_cash > 0 else RED};"
            f"font-size:1rem;font-weight:800;margin-top:2px'>${_port_cash:,.0f}</div>"
            f"</div>"
            f"</div></div>"
            # Open positions preview (up to 3)
            + (
                f"<div style='border-top:1px solid {BORDER};padding-top:10px'>"
                + "".join(
                    f"<div style='display:flex;justify-content:space-between;align-items:center;"
                    f"padding:4px 0'>"
                    f"<div style='display:flex;align-items:center;gap:8px'>"
                    f"<span style='color:{TEXT_PRIMARY};font-weight:700;font-size:0.78rem'>{p['symbol']}</span>"
                    f"<span style='color:{GREEN if p['direction'] == 'LONG' else RED};"
                    f"font-size:0.55rem;font-weight:700'>{p['direction']}</span>"
                    f"</div>"
                    f"<span style='color:{GREEN if p.get('unrealised_pnl', 0) >= 0 else RED};"
                    f"font-size:0.78rem;font-weight:700'>${p.get('unrealised_pnl', 0):+,.2f}</span>"
                    f"</div>"
                    for p in _port_open_list[:3]
                )
                + (f"<div style='color:{TEXT_MUTED};font-size:0.62rem;margin-top:4px'>+ {_open_pos - 3} more</div>"
                   if _open_pos > 3 else "")
                + f"</div>"
                if _port_open_list else
                f"<div style='border-top:1px solid {BORDER};padding-top:10px;"
                f"color:{TEXT_MUTED};font-size:0.72rem;text-align:center'>No open positions</div>"
            )
            + f"</div>",
            unsafe_allow_html=True,
        )

        if st.button("Open Portfolio", use_container_width=True, key="dash_portfolio_btn"):
            st.session_state.current_page = "Portfolio"
            st.rerun()

    # ── Recent Results (UNIVERSAL — mirrors the public Track Record) ────────
    with _dash_right:
        st.markdown(
            f"<div class='sec-head' style='display:flex;align-items:center;gap:10px;margin-bottom:14px'>"
            f"<div class='sec-bar' style='width:3px;height:14px;background:{BRAND_GRAD};border-radius:2px'></div>"
            f"<span style='color:{TEXT_PRIMARY};font-size:0.7rem;font-weight:800;"
            f"text-transform:uppercase;letter-spacing:0.14em'>Recent Results</span>"
            f"<span style='padding:3px 10px;border-radius:100px;"
            f"background:rgba(6,214,160,0.10);border:1px solid rgba(6,214,160,0.28);"
            f"color:{CYAN};font-size:0.5rem;font-weight:800;letter-spacing:0.12em;"
            f"text-transform:uppercase;margin-left:4px'>Public ledger</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        try:
            _preds = _pa.get("predictions_table", [])
            _recent_scored = [p for p in _preds if p.get("final_result") in ("✓", "✗")][:10]
            if _recent_scored:
                # Win/Loss summary
                _rw = sum(1 for p in _recent_scored if p.get("final_result") == "✓")
                _rl = len(_recent_scored) - _rw
                st.markdown(
                    f"<div style='display:flex;gap:8px;margin-bottom:10px'>"
                    f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:6px;"
                    f"padding:6px 12px;display:flex;align-items:center;gap:6px'>"
                    f"<span style='color:{TEXT_MUTED};font-size:0.55rem;font-weight:600'>Last {len(_recent_scored)}</span>"
                    f"<span style='color:{GREEN};font-size:0.82rem;font-weight:800'>{_rw}W</span>"
                    f"<span style='color:{TEXT_MUTED};font-size:0.6rem'>·</span>"
                    f"<span style='color:{RED};font-size:0.82rem;font-weight:800'>{_rl}L</span>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                for _rp in _recent_scored:
                    _r_sym = _rp.get("symbol", "?")
                    _r_horizon = _rp.get("horizon", "")
                    _r_ret = _rp.get("predicted_return", 0)
                    _r_actual = _rp.get("actual_return", 0)
                    _r_result = _rp.get("final_result", "—")
                    _r_color = GREEN if _r_result == "✓" else RED
                    _r_icon = "✓" if _r_result == "✓" else "✗"

                    st.markdown(
                        f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;"
                        f"padding:10px 14px;margin-bottom:4px;border-left:3px solid {_r_color};"
                        f"display:flex;align-items:center;justify-content:space-between'>"
                        f"<div style='display:flex;align-items:center;gap:10px'>"
                        f"<span style='color:{TEXT_PRIMARY};font-weight:700;font-size:0.82rem;min-width:45px'>{_r_sym}</span>"
                        f"<span style='color:{TEXT_MUTED};font-size:0.58rem'>{_r_horizon}</span>"
                        f"<span style='color:{GREEN if _r_ret > 0 else RED};font-size:0.72rem;"
                        f"font-weight:600'>{_r_ret:+.1f}%</span>"
                        f"</div>"
                        f"<div style='color:{_r_color};font-size:1.1rem;font-weight:900'>{_r_icon}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:12px;"
                    f"padding:30px;text-align:center'>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.8rem'>Results appear as predictions are scored.</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

        if st.button("View Full Track Record", use_container_width=True, key="dash_track_btn"):
            st.session_state.current_page = "Track Record"
            st.rerun()

    # ── Quick Actions ────────────────────────────────────────────────────────
    st.markdown(f"<div style='height:24px'></div>", unsafe_allow_html=True)

    _qa1, _qa2, _qa3 = st.columns(3)
    with _qa1:
        if st.button("Analyze a Stock", use_container_width=True, key="dash_analyze", type="primary"):
            st.session_state.current_page = "Analyze"
            st.rerun()
    with _qa2:
        if st.button("Deep Scan Watchlist", use_container_width=True, key="dash_batch"):
            st.session_state.batch_scan_requested = True
            st.session_state.current_page = "Watchlist"
            st.rerun()
    with _qa3:
        if st.button("Open Watchlist", use_container_width=True, key="dash_watchlist"):
            st.session_state.current_page = "Watchlist"
            st.rerun()

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='text-align:center;margin-top:28px;padding-top:16px;border-top:1px solid {BORDER}'>"
        f"<div style='color:{TEXT_MUTED};font-size:0.55rem;line-height:1.6;max-width:550px;margin:0 auto'>"
        f"Prediqt is an educational research tool. Predictions are model-generated and not financial advice. "
        f"Past performance does not guarantee future results.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: TRACK RECORD
# ──────────────────────────────────────────────────────────────────────────────
if _page == "Track Record":
    from track_record_page import render_track_record
    render_track_record()
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: STOCK DETAIL — lightweight stock view (no retraining)
# ──────────────────────────────────────────────────────────────────────────────
if _page == "Stock Detail":
    from stock_detail_page import render_stock_detail
    _detail_sym = st.session_state.get("detail_symbol", "AAPL")
    # Back button
    if st.button("← Back", key="detail_back"):
        st.session_state.current_page = st.session_state.get("detail_return_page", "Dashboard")
        st.rerun()
    render_stock_detail(_detail_sym)
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: WATCHLIST  (pulls existing watchlist tab content into its own page)
# ──────────────────────────────────────────────────────────────────────────────
if _page == "Watchlist":
    from watchlist_page import render_watchlist_page
    render_watchlist_page()
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: PORTFOLIO  (pulls existing paper trading tab into its own page)
# ──────────────────────────────────────────────────────────────────────────────
if _page == "Portfolio":
    from portfolio_page import render_portfolio_page
    render_portfolio_page()
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: ANALYZE — main analysis flow
# ──────────────────────────────────────────────────────────────────────────────
assert _page == "Analyze", f"Unknown page: {_page}"

if st.session_state.results is None:
    st.markdown(f"""
    <div class='page-header'>
        <div class='ph-eyebrow'>Powered by Prediqt IQ</div>
        <h1>Know the move<br><span class='grad'>before it happens.</span></h1>
        <p class='ph-sub'>Enter a ticker in the sidebar and hit <b style='color:{CYAN}'>Run Analysis</b> for
        multi-horizon predictions, signal reasoning, and trade setups — all from one engine.</p>
    </div>
    """, unsafe_allow_html=True)
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
earnings    = R.get("earnings", {}) or {}
fundamentals= R.get("fundamentals", {}) or {}

cur   = float(df["Close"].iloc[-1])
prev  = float(df["Close"].iloc[-2])
d_chg = (cur - prev) / prev * 100


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — DECISION (5-second scan)
# ══════════════════════════════════════════════════════════════════════════════

rec    = analysis["recommendation"]
color  = analysis["color"]
conf1m = predictions.get("1 Month", predictions.get("3 Day", {})).get("confidence", 50)

# ── Stock identity bar ────────────────────────────────────────────────────────
id_col, price_col, stat1, stat2, stat3, stat4 = st.columns([2, 1.5, 1.3, 1.3, 1.3, 1.3])

with id_col:
    rlabel = regime_info.get("label", "Sideways") if regime_info else "N/A"
    # Regime chip colors — brand palette; dim tints derived with rgba math
    rcolor = {
        "Bull":     CHART_UP,
        "Bear":     CHART_DOWN,
        "Sideways": AMBER,
    }.get(rlabel, TEXT_MUTED)
    rbg = {
        "Bull":     "rgba(6,214,160,0.10)",
        "Bear":     "rgba(224,74,74,0.10)",
        "Sideways": "rgba(245,158,11,0.10)",
    }.get(rlabel, BG_CARD)
    rbd = {
        "Bull":     "rgba(6,214,160,0.32)",
        "Bear":     "rgba(224,74,74,0.32)",
        "Sideways": "rgba(245,158,11,0.32)",
    }.get(rlabel, BORDER)
    ricon = {"Bull": "▲", "Bear": "▼", "Sideways": "→"}.get(rlabel, "◆")
    st.markdown(
        f"<div style='padding:6px 0;line-height:1.35'>"
        # Sector · Industry — cyan eyebrow style
        f"<div style='color:{CYAN};font-size:0.56rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.14em;margin-bottom:6px'>"
        f"{info.get('sector','—')} &nbsp;·&nbsp; {info.get('industry','—')}</div>"
        # Company name + ticker pill + regime pill, all aligned
        f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap'>"
        f"<span style='font-family:\"Inter\",sans-serif;color:{TEXT_PRIMARY};"
        f"font-size:1.45rem;font-weight:900;letter-spacing:-0.02em'>"
        f"{info.get('name', symbol)}</span>"
        f"<span style='display:inline-flex;align-items:center;padding:3px 10px;"
        f"border-radius:100px;background:rgba(32,128,229,0.10);"
        f"border:1px solid rgba(32,128,229,0.30);color:{BLUE};"
        f"font-family:\"Inter\",sans-serif;font-size:0.64rem;font-weight:800;"
        f"letter-spacing:0.08em'>{symbol}</span>"
        f"<span style='display:inline-flex;align-items:center;gap:5px;"
        f"padding:3px 10px;border-radius:100px;background:{rbg};"
        f"border:1px solid {rbd};color:{rcolor};font-size:0.62rem;"
        f"font-weight:700;letter-spacing:0.08em;text-transform:uppercase'>"
        f"<span style='font-size:0.7rem'>{ricon}</span>{rlabel}</span>"
        f"</div></div>", unsafe_allow_html=True)

with price_col:
    dc = _rc(d_chg)
    _chg_bg = "rgba(6,214,160,0.10)" if d_chg >= 0 else "rgba(224,74,74,0.10)"
    _chg_bd = "rgba(6,214,160,0.32)" if d_chg >= 0 else "rgba(224,74,74,0.32)"
    st.markdown(
        f"<div style='text-align:right;padding:4px 0'>"
        f"<div style='font-family:\"Inter\",sans-serif;font-size:2rem;font-weight:900;"
        f"color:{TEXT_PRIMARY};letter-spacing:-0.03em;"
        f"font-variant-numeric:tabular-nums;line-height:1'>${cur:,.2f}</div>"
        f"<div style='display:inline-flex;align-items:center;gap:5px;"
        f"padding:3px 10px;border-radius:100px;"
        f"background:{_chg_bg};border:1px solid {_chg_bd};margin-top:6px'>"
        f"<span style='color:{dc};font-size:0.74rem;font-weight:700;"
        f"font-variant-numeric:tabular-nums;letter-spacing:0.01em'>"
        f"{_arr(d_chg)} {abs(d_chg):.2f}% today</span>"
        f"</div>"
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
            f"<div class='stat-value' style='font-size:0.95rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{val}</div>"
            f"</div>", unsafe_allow_html=True)

st.markdown(f"<div style='height:1px;background:linear-gradient(90deg,transparent,{BORDER_ACCENT},transparent);margin:12px 0 16px'></div>", unsafe_allow_html=True)

# ── Hero signal + horizon strip ──────────────────────────────────────────────
hero_col, h1, h2, h3, h4, h5 = st.columns([1.9, 1, 1, 1, 1, 1])

with hero_col:
    # Glow uses brand cyan/muted-red/amber — keeps the signal visually
    # "hot" without raw greens/reds.
    glow = CHART_UP if "Buy" in rec else CHART_DOWN if "Sell" in rec else AMBER
    conf_class = "conf-high" if conf1m >= 70 else "conf-medium" if conf1m >= 55 else "conf-low"
    conf_label = "HIGH" if conf1m >= 70 else "MEDIUM" if conf1m >= 55 else "LOW"
    st.markdown(
        f"<div class='hero-signal' style='border:1.5px solid {color}30;--glow:{glow}'>"
        f"<div style='position:relative;z-index:1'>"
        # Prediqt IQ badge moved ABOVE the signal — reads as a provenance stamp
        f"<div style='display:inline-flex;align-items:center;gap:6px;"
        f"padding:4px 11px;border-radius:100px;"
        f"background:rgba(6,214,160,0.10);border:1px solid rgba(6,214,160,0.32);"
        f"margin-bottom:12px'>"
        f"<span style='width:5px;height:5px;border-radius:50%;background:{CYAN};"
        f"box-shadow:0 0 8px {CYAN}'></span>"
        f"<span style='color:{CYAN};font-size:0.54rem;font-weight:700;"
        f"letter-spacing:0.14em;text-transform:uppercase'>Prediqt IQ</span></div>"
        # Headline recommendation
        f"<div style='font-family:\"Inter\",sans-serif;color:{color};"
        f"font-size:2rem;font-weight:900;letter-spacing:-0.02em;"
        f"white-space:nowrap;line-height:1'>{rec.upper()}</div>"
        # Sublabel
        f"<div style='color:{TEXT_MUTED};font-size:0.58rem;margin:8px 0 20px;"
        f"text-transform:uppercase;letter-spacing:0.14em;font-weight:700;"
        f"overflow:visible'>1-Month ML Signal {info_icon(TIPS['ML'])}</div>"
        f"<div style='display:flex;justify-content:center;gap:20px;margin-bottom:16px'>"
        f"<div style='flex:1'>"
        f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:6px'>Confidence {info_icon(TIPS['Confidence'])}</div>"
        f"<div class='conf-gauge'>"
        f"<svg viewBox='0 0 120 66'>"
        f"<path d='M10 60 A50 50 0 0 1 110 60' fill='none' stroke='{BORDER}' stroke-width='10' stroke-linecap='round'/>"
        f"<path d='M10 60 A50 50 0 0 1 110 60' fill='none' stroke='url(#cg)' stroke-width='10' stroke-linecap='round' stroke-dasharray='{(conf1m / 100) * 157:.1f} 157'/>"
        f"<defs><linearGradient id='cg'>"
        f"<stop offset='0%' stop-color='{CHART_DOWN}'/>"
        f"<stop offset='50%' stop-color='{AMBER}'/>"
        f"<stop offset='100%' stop-color='{CHART_UP}'/>"
        f"</linearGradient></defs>"
        f"</svg>"
        f"<div class='conf-gauge-text' style='color:{_cc(conf1m)}'>{conf1m:.0f}%</div>"
        f"</div>"
        f"<div style='color:{TEXT_MUTED};font-size:0.58rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.12em;margin-top:6px'>"
        f"{conf_label}</div>"
        f"</div>"
        f"<div style='width:1px;background:{BORDER}'></div>"
        f"<div style='flex:1'>"
        f"<div style='color:{CYAN};font-size:0.54rem;text-transform:uppercase;"
        f"letter-spacing:0.14em;font-weight:700;margin-bottom:6px'>"
        f"Agreement {info_icon(TIPS['Agreement'])}</div>"
        f"<div style='font-family:\"Inter\",sans-serif;color:{TEXT_PRIMARY};"
        f"font-size:1.9rem;font-weight:900;letter-spacing:-0.03em;line-height:1;"
        f"font-variant-numeric:tabular-nums'>"
        f"{predictions.get('1 Month', predictions.get('3 Day', {})).get('ensemble_agreement', 0):.0f}"
        f"<span style='font-size:0.85rem;color:{TEXT_MUTED};font-weight:700'>%</span></div>"
        f"<div style='color:{TEXT_MUTED};font-size:0.58rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.12em;margin-top:6px'>"
        f"Models Align</div>"
        f"</div>"
        f"</div>"
        # Validated-accuracy footer — brand cyan for the accuracy %
        f"<div style='color:{TEXT_MUTED};font-size:0.68rem;margin-top:14px;"
        f"padding-top:12px;border-top:1px solid {BORDER};overflow:visible;"
        f"line-height:1.5'>"
        f"Validated accuracy: "
        f"<b style='color:{CHART_UP};font-variant-numeric:tabular-nums'>"
        f"{predictions.get('1 Month', predictions.get('3 Day', {})).get('val_dir_accuracy', 0):.1f}%</b> "
        f"on held-out data {info_icon(TIPS['ValAcc'])}</div>"
        f"</div></div>",
        unsafe_allow_html=True)

# ── Horizon target dates (trading days forward from last bar) ────────────────
# Horizons are in TRADING days (3, 5, 21, 63, 252). Anchor from the last bar in
# the data (not today) so weekends/holidays/stale data don't shift the target.
# Uses US federal calendar as a close proxy for NYSE holidays — good enough
# for display; adds ≤1-2 days drift at the 1-year horizon vs. the real NYSE cal.
try:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    _hz_bday = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
except Exception:
    _hz_bday = pd.offsets.BDay()
_hz_anchor = pd.Timestamp(df.index[-1]).normalize()
_hz_anchor_str = _hz_anchor.strftime("%b %d, %Y")
horizon_dates = {
    _h: (_hz_anchor + _hz_bday * _d).strftime("%b %d, %Y")
    for _h, _d in HORIZONS.items()
}

for col_h, h in zip([h1,h2,h3,h4,h5], ["3 Day","1 Week","1 Month","1 Quarter","1 Year"]):
    if h not in predictions:
        with col_h:
            st.markdown(f"<div class='hstrip'><div style='color:{TEXT_MUTED};font-size:0.75rem;padding:20px 0;text-align:center'>Not enough data<br>for {h}</div></div>", unsafe_allow_html=True)
        continue
    p   = predictions[h]

    # ── Quality gate: suppress low-quality predictions ──────────────
    if p.get("suppress"):
        with col_h:
            _suppress_reason = p.get("suppress_reason", "Low quality signal")
            st.markdown(
                f"<div class='hstrip' style='min-height:320px;display:flex;flex-direction:column;justify-content:center;align-items:center'>"
                f"<div style='color:{AMBER};font-size:1.3rem;font-weight:800;margin-bottom:8px'>◇ NO SIGNAL</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.72rem;text-align:center;max-width:140px;line-height:1.4'>{_suppress_reason}</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.6rem;margin-top:12px'>Quality gate active</div>"
                f"</div>", unsafe_allow_html=True)
        continue

    ret = p["predicted_return"]
    rc_ = _rc(ret)
    cc_ = _cc(p["confidence"])
    # Get trade rec for this horizon
    strat_h = options.get(h, {})
    tr_line, tr_color, tr_cost = _trade_oneliner(strat_h)

    # Flag predictions that look model-overfit. Trust the model's own
    # is_extreme signal, which means the prediction is outside the training
    # distribution (true extrapolation). Large-but-in-distribution predictions
    # on volatile stocks are not flagged.
    is_extreme_h = p.get("is_extreme", False)
    raw_ret = p.get("predicted_return_raw", ret)
    was_shrunk = abs(raw_ret - ret) > 0.01
    if is_extreme_h:
        shrink_note = f" — pulled back from raw {raw_ret*100:+.0f}%" if was_shrunk else ""
        extreme_badge = f"<span style='display:inline-block;padding:2px 6px;border-radius:4px;background:#f59e0b22;color:{AMBER};font-size:0.58rem;font-weight:700;margin-left:4px' title='Outside training distribution — confidence reduced{shrink_note}'>⚠ EXTREME</span>"
    else:
        extreme_badge = ""

    with col_h:
        h_conf_class = "conf-high" if p['confidence'] >= 70 else "conf-medium" if p['confidence'] >= 55 else "conf-low"
        # Confidence pill colored by tier so the eye can scan a row of 5
        # cards and immediately see which horizons the model feels strong on.
        _conf_pill_bg = (
            "rgba(6,214,160,0.12)" if p['confidence'] >= 70 else
            "rgba(245,158,11,0.12)" if p['confidence'] >= 55 else
            "rgba(224,74,74,0.12)"
        )
        _conf_pill_bd = (
            "rgba(6,214,160,0.32)" if p['confidence'] >= 70 else
            "rgba(245,158,11,0.32)" if p['confidence'] >= 55 else
            "rgba(224,74,74,0.32)"
        )
        st.markdown(
            f"<div class='hstrip' style='display:flex;flex-direction:column;min-height:320px'>"
            # Header row — horizon + target date on the left, confidence pill on the right
            f"<div class='hstrip-head' style='align-items:flex-start'>"
            f"<div style='display:flex;flex-direction:column;gap:3px;min-width:0'>"
            f"<span style='color:{CYAN};font-size:0.58rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.14em' "
            f"title='{HORIZONS[h]} trading days forward from last close ({_hz_anchor_str})'>"
            f"{h}{extreme_badge}</span>"
            f"<span style='color:{TEXT_MUTED};font-size:0.58rem;letter-spacing:0.02em;"
            f"font-weight:500;white-space:nowrap'>by {horizon_dates[h]}</span>"
            f"</div>"
            f"<span style='display:inline-flex;align-items:center;padding:2px 8px;"
            f"border-radius:100px;background:{_conf_pill_bg};border:1px solid {_conf_pill_bd};"
            f"color:{cc_};font-size:0.62rem;font-weight:700;flex-shrink:0;"
            f"font-variant-numeric:tabular-nums;letter-spacing:0.02em'>"
            f"{p['confidence']:.0f}%</span>"
            f"</div>"
            # Target + Return row
            f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
            f"margin-top:12px'>"
            f"<div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.54rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:4px'>Target</div>"
            f"<div style='font-family:\"Inter\",sans-serif;color:{TEXT_PRIMARY};"
            f"font-size:1.05rem;font-weight:900;letter-spacing:-0.02em;"
            f"font-variant-numeric:tabular-nums;line-height:1;white-space:nowrap'>"
            f"${p['predicted_price']:,.2f}</div>"
            f"</div>"
            f"<div style='text-align:right'>"
            f"<div style='color:{TEXT_MUTED};font-size:0.54rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:4px'>Return</div>"
            f"<div style='font-family:\"Inter\",sans-serif;color:{rc_};"
            f"font-size:1.05rem;font-weight:900;letter-spacing:-0.02em;"
            f"font-variant-numeric:tabular-nums;line-height:1;white-space:nowrap'>"
            f"{_arr(ret)}&nbsp;{abs(ret*100):.1f}%</div>"
            f"</div>"
            f"</div>"
            # Range + accuracy micro-line
            f"<div style='margin-top:12px;padding-top:10px;border-top:1px solid {BORDER};"
            f"font-size:0.66rem;color:{TEXT_MUTED};letter-spacing:0.01em;"
            f"font-variant-numeric:tabular-nums'>"
            f"${p['interval_low']:,.2f} – ${p['interval_high']:,.2f}"
            f"<span style='opacity:0.6'>&nbsp;·&nbsp;</span>"
            f"acc {p['val_dir_accuracy']:.1f}%"
            f"</div>"
            # Suggested trade
            f"<div style='margin-top:10px;padding-top:10px;border-top:1px solid {BORDER};"
            f"flex-grow:1;display:flex;flex-direction:column'>"
            f"<div style='color:{CYAN};font-size:0.52rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.14em;margin-bottom:5px'>"
            f"Suggested Trade</div>"
            f"<div style='color:{tr_color};font-size:0.78rem;font-weight:800;"
            f"letter-spacing:0.01em;min-height:20px;line-height:1.3'>{tr_line}</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.62rem;margin-top:auto;"
            f"padding-top:6px;font-style:italic'>{tr_cost}</div>"
            f"</div>"
            f"</div>", unsafe_allow_html=True)
        # Expandable trade details — designed for narrow (~180px) columns.
        # Every element is single-line where possible, with tight typography.
        with st.expander("Trade details", expanded=False):
            if strat_h and strat_h.get("strategy") != "N/A" and "error" not in strat_h:

                _strat_dir = strat_h.get("direction", "Neutral")
                _strat_dir_color = (
                    CHART_UP if "Bull" in _strat_dir or "bull" in _strat_dir.lower()
                    else CHART_DOWN if "Bear" in _strat_dir or "bear" in _strat_dir.lower()
                    else AMBER
                )
                _iv_used_str = str(strat_h.get("iv_used", "")).strip()

                # ── Strategy header — one compact line ────────────────────
                # Name + direction chip stacked if narrow, side-by-side if
                # there's room. flex-wrap handles the narrow-column case.
                st.markdown(
                    f"<div style='margin-bottom:10px'>"
                    f"<div style='color:{TEXT_PRIMARY};font-family:\"Inter\",sans-serif;"
                    f"font-size:0.88rem;font-weight:800;letter-spacing:-0.01em;"
                    f"line-height:1.15;margin-bottom:6px'>"
                    f"{strat_h['strategy']}</div>"
                    f"<div style='display:flex;gap:5px;flex-wrap:wrap;align-items:center'>"
                    f"<span style='display:inline-flex;align-items:center;"
                    f"padding:1px 7px;border-radius:100px;"
                    f"background:{_strat_dir_color}1a;"
                    f"border:1px solid {_strat_dir_color}44;"
                    f"color:{_strat_dir_color};font-size:0.52rem;font-weight:700;"
                    f"letter-spacing:0.12em;text-transform:uppercase'>"
                    f"{_strat_dir}</span>"
                    + (f"<span style='display:inline-flex;align-items:center;"
                       f"padding:1px 7px;border-radius:100px;"
                       f"background:{BG_ELEVATED};border:1px solid {BORDER};"
                       f"color:{TEXT_SECONDARY};font-size:0.52rem;font-weight:700;"
                       f"letter-spacing:0.12em;text-transform:uppercase'>"
                       f"IV {_iv_used_str}</span>" if _iv_used_str else "")
                    + f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # ── Legs — true single-row compact table ─────────────────
                _legs = strat_h.get("legs", [])
                if _legs:
                    _leg_rows = []
                    for leg in _legs:
                        _buy = leg["action"] == "BUY"
                        _leg_color = CHART_UP if _buy else CHART_DOWN
                        _arrow = "▲" if _buy else "▼"
                        # Single-line: arrow + action + type $strike + premium
                        _leg_rows.append(
                            f"<div style='display:flex;align-items:center;gap:6px;"
                            f"padding:6px 8px;border-radius:6px;"
                            f"background:{('rgba(6,214,160,0.07)' if _buy else 'rgba(224,74,74,0.07)')};"
                            f"border-left:2px solid {_leg_color};"
                            f"font-size:0.7rem;font-variant-numeric:tabular-nums;"
                            f"line-height:1.15'>"
                            f"<span style='color:{_leg_color};font-weight:800;"
                            f"font-size:0.7rem'>{_arrow}</span>"
                            f"<span style='color:{_leg_color};font-weight:800;"
                            f"letter-spacing:0.04em'>{leg['action']}</span>"
                            f"<span style='color:{TEXT_PRIMARY};font-weight:700'>"
                            f"{leg['type']}&nbsp;${leg['strike']:.0f}</span>"
                            f"<span style='flex:1;text-align:right;color:{TEXT_SECONDARY};"
                            f"font-weight:700'>${leg['premium']:.2f}</span>"
                            f"</div>"
                        )
                    st.markdown(
                        "<div style='display:flex;flex-direction:column;gap:4px;"
                        "margin-bottom:12px'>"
                        + "".join(_leg_rows)
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                # ── P/L + breakeven — compact stacked layout ─────────────
                # Raw API values like "$662.00 (stock stays between $259 and $288)"
                # overflow narrow columns. Split the $ amount from the
                # parenthetical note, stack them on separate lines.
                import re as _re_ts
                def _split_value_note(s: str) -> tuple[str, str]:
                    """Return (amount, note) from strings like 'X (note)'.
                    Strips '/ contract' noise and handles missing parenthetical."""
                    if not s:
                        return ("—", "")
                    s = (s.replace(" / contract", "")
                          .replace("/ contract", "")
                          .strip())
                    # Match first "(...)" — some strings (breakeven) have two
                    # parentheticals like "$A (down) / $B (up)"; those should
                    # NOT be split (they're already short enough).
                    if s.count("(") > 1:
                        return (s, "")
                    m = _re_ts.match(r'^(.+?)\s*\((.+)\)\s*$', s)
                    if m:
                        return (m.group(1).strip(), m.group(2).strip())
                    return (s, "")

                _label_style = (
                    f"color:{TEXT_MUTED};font-size:0.54rem;font-weight:700;"
                    f"text-transform:uppercase;letter-spacing:0.12em"
                )
                _note_style = (
                    f"color:{TEXT_MUTED};font-size:0.62rem;line-height:1.4;"
                    f"margin-top:2px;letter-spacing:0.005em"
                )

                def _stat_row(label: str, raw: str, value_color: str,
                              value_weight: str = "800",
                              show_divider: bool = False) -> str:
                    if not raw:
                        return ""
                    amount, note = _split_value_note(raw)
                    _div = (f"border-top:1px solid {BORDER};margin-top:8px;padding-top:8px"
                            if show_divider else "margin-top:8px")
                    _note_html = (f"<div style='{_note_style}'>{note}</div>" if note else "")
                    return (
                        f"<div style='{_div}'>"
                        f"<div style='{_label_style};margin-bottom:2px'>{label}</div>"
                        f"<div style='color:{value_color};font-weight:{value_weight};"
                        f"font-size:0.85rem;font-variant-numeric:tabular-nums;"
                        f"letter-spacing:-0.005em;line-height:1.15'>"
                        f"{amount}</div>"
                        f"{_note_html}"
                        f"</div>"
                    )

                _stats_html = (
                    f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
                    f"border:1px solid {BORDER};border-radius:8px;padding:12px 14px'>"
                )
                # First row (no divider/margin above)
                _first = True
                for _lbl, _key, _clr in [
                    ("Max profit",  "max_profit",     CHART_UP),
                    ("Max loss",    "max_loss",       CHART_DOWN),
                    ("Cost",        "estimated_cost", TEXT_PRIMARY),
                ]:
                    _raw = strat_h.get(_key, "")
                    if not _raw:
                        continue
                    _row = _stat_row(_lbl, _raw, _clr, show_divider=not _first)
                    _stats_html += _row
                    _first = False

                # Breakeven gets its own section with divider
                _be = strat_h.get('breakeven', '')
                if _be:
                    _be_clean = (_be.replace(" / contract", "")
                                    .replace("/ contract", "").strip())
                    _stats_html += (
                        f"<div style='border-top:1px solid {BORDER};"
                        f"margin-top:10px;padding-top:8px'>"
                        f"<div style='{_label_style};margin-bottom:3px'>Breakeven</div>"
                        f"<div style='color:{TEXT_PRIMARY};font-weight:700;"
                        f"font-size:0.75rem;font-variant-numeric:tabular-nums;"
                        f"line-height:1.3'>{_be_clean}</div>"
                        f"</div>"
                    )
                _stats_html += "</div>"
                st.markdown(_stats_html, unsafe_allow_html=True)

                # ── Rationale — compact cyan-accent block ────────────────
                _rationale = strat_h.get("rationale", "") or ""
                if _rationale:
                    st.markdown(
                        f"<div style='margin-top:10px;padding:8px 10px;"
                        f"background:rgba(6,214,160,0.04);"
                        f"border-left:2px solid rgba(6,214,160,0.35);"
                        f"border-radius:4px;font-size:0.66rem;"
                        f"color:{TEXT_SECONDARY};line-height:1.5'>"
                        f"<span style='color:{CYAN};font-size:0.48rem;font-weight:700;"
                        f"text-transform:uppercase;letter-spacing:0.14em;"
                        f"display:block;margin-bottom:3px'>Why this strategy</span>"
                        f"{_rationale}</div>",
                        unsafe_allow_html=True,
                    )

                # ── Risk note — only if present, compact amber block ────
                _risk_txt = strat_h.get("risk_note", "") or ""
                if _risk_txt:
                    st.markdown(
                        f"<div style='margin-top:6px;padding:8px 10px;"
                        f"background:rgba(245,158,11,0.05);"
                        f"border-left:2px solid rgba(245,158,11,0.35);"
                        f"border-radius:4px;font-size:0.64rem;"
                        f"color:{AMBER};line-height:1.5'>"
                        f"<span style='font-size:0.48rem;font-weight:700;"
                        f"text-transform:uppercase;letter-spacing:0.14em;"
                        f"display:block;margin-bottom:3px'>Risk</span>"
                        f"{_risk_txt}</div>",
                        unsafe_allow_html=True,
                    )

                # ── Small disclaimer ─────────────────────────────────────
                st.markdown(
                    f"<div style='margin-top:8px;font-size:0.54rem;"
                    f"color:{TEXT_MUTED};font-style:italic;line-height:1.4'>"
                    f"Black-Scholes estimates. Verify with your broker.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='color:{TEXT_MUTED};font-size:0.72rem;"
                    f"padding:8px 4px'>No strategy for this horizon.</div>",
                    unsafe_allow_html=True,
                )


# ── Risk/Reward Ratio Cards ────────────────────────────────────────────────────
# Uses the SAME dollar-level math as the Trade Setup card:
#   reward  = predicted_price − entry_low      (best case entry to ML target)
#   risk    = entry_high     − stop_loss       (worst case entry to ATR stop)
#   R/R     = reward / risk
# This keeps all four horizon cards consistent with the Trade Setup number.
st.markdown(f"<div style='margin:28px 0 14px;'></div>", unsafe_allow_html=True)
st.markdown(
    f"<div style='display:flex;flex-wrap:wrap;align-items:baseline;gap:10px;margin-bottom:12px'>"
    f"<div class='sec-head' style='margin:0'><span class='sec-bar'></span>"
    f"Risk / Reward by Horizon {info_icon(TIPS['RRSection'])}</div>"
    f"<span style='color:{TEXT_MUTED};font-size:0.64rem;font-weight:500'>"
    f"· reward = ML{info_icon(TIPS['ML'])} target − best entry · "
    f"risk = worst entry − ATR{info_icon(TIPS['ATR'])} stop</span>"
    f"</div>",
    unsafe_allow_html=True)

rr1, rr2, rr3, rr4, rr5 = st.columns(5)

# Pull the shared trade-setup levels once so all four cards use the same stop/entry
_ez        = analysis["entry_zone"]
_entry_low  = _ez["entry_low"]
_entry_high = _ez["entry_high"]
_stop_loss  = _ez["stop_loss"]
_risk_pts   = max(_entry_high - _stop_loss, 0.01)   # worst-case entry to stop (dollars at risk)

_first_rr_card = True
for col, horizon in zip([rr1, rr2, rr3, rr4, rr5], ["3 Day", "1 Week", "1 Month", "1 Quarter", "1 Year"]):
    with col:
        if horizon not in predictions:
            st.markdown(f"<div class='card-sm'><div style='color:{TEXT_MUTED};font-size:0.75rem;text-align:center;padding:12px 0'>N/A</div></div>", unsafe_allow_html=True)
            continue

        p = predictions[horizon]
        pred_price = p.get("predicted_price", 0)
        pred_ret   = p.get("predicted_return", 0)
        conf       = p.get("confidence", 50)

        # Reward = distance from best entry to ML predicted price for this horizon
        reward_pts = max(pred_price - _entry_low, 0)

        # R/R = reward / risk  (same formula as Trade Setup card)
        rr_ratio = round(reward_pts / _risk_pts, 1) if _risk_pts > 0 else 0

        # % gain from best entry → target; % loss from worst entry → stop
        reward_pct = (reward_pts / _entry_low * 100) if _entry_low > 0 else 0
        risk_pct   = (_risk_pts  / _entry_high * 100) if _entry_high > 0 else 0

        # Flag extreme predictions (model likely overfit on this horizon)
        is_extreme = abs(reward_pct) > 50 or rr_ratio > 8

        if is_extreme:
            rr_label = "⚠ Extreme"
            if rr_ratio >= 3:
                rr_color = GREEN
            elif rr_ratio >= 2:
                rr_color = GREEN
            elif rr_ratio >= 1:
                rr_color = AMBER
            else:
                rr_color = RED
        elif rr_ratio >= 3:
            rr_color = GREEN
            rr_label = "Excellent" if rr_ratio >= 5 else "Very Good"
        elif rr_ratio >= 2:
            rr_color = GREEN
            rr_label = "Favorable"
        elif rr_ratio >= 1:
            rr_color = AMBER
            rr_label = "Moderate"
        else:
            rr_color = RED
            rr_label = "Poor"

        # Cap bar fill visually at 6:1 so extreme ratios don't distort
        bar_fill = min((rr_ratio / 6) * 100, 100)

        _reward_tip = TIPS["RewardPct"].replace("'", "&#39;").replace('"', "&quot;")
        _risk_tip   = TIPS["RiskPct"].replace("'", "&#39;").replace('"', "&quot;")

        # Only show info icons on the first card to avoid clutter
        _hi = info_icon(TIPS['Horizon']) if _first_rr_card else ""
        _ri = info_icon(TIPS['RRRatio']) if _first_rr_card else ""
        _qi = info_icon(TIPS['RRQuality']) if _first_rr_card else ""
        _ti = info_icon(TIPS['TGT']) if _first_rr_card else ""
        _si = info_icon(TIPS['STP']) if _first_rr_card else ""
        _gcol = "48px" if _first_rr_card else "40px"

        st.markdown(
            f"<div class='card-sm' style='padding:14px;min-height:160px;display:flex;flex-direction:column;overflow:visible'>"
            # Header row: horizon + R:R ratio
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;overflow:visible'>"
            f"<span style='color:{TEXT_MUTED};font-size:0.66rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px'>{horizon} {_hi}</span>"
            f"<span style='color:{rr_color};font-size:0.85rem;font-weight:800;font-variant-numeric:tabular-nums'>{rr_ratio:.1f}:1 {_ri}</span>"
            f"</div>"
            # Progress bar
            f"<div style='background:{BORDER};border-radius:3px;height:4px;overflow:hidden;margin-bottom:8px'>"
            f"<div style='width:{bar_fill:.1f}%;background:{rr_color};height:100%;border-radius:3px;transition:width 0.3s'></div>"
            f"</div>"
            # Quality label
            f"<div style='color:{rr_color};font-size:0.7rem;font-weight:700;margin-bottom:12px;text-transform:uppercase;letter-spacing:0.5px;overflow:visible'>{rr_label} {_qi}</div>"
            # Target row
            f"<div style='display:grid;grid-template-columns:{_gcol} 1fr auto;gap:8px;align-items:center;font-size:0.625rem;margin-bottom:6px;overflow:visible'>"
            f"<span style='color:{TEXT_MUTED};font-weight:600'>TGT {_ti}</span>"
            f"<span style='color:{TEXT_PRIMARY};font-weight:700;font-variant-numeric:tabular-nums'>${pred_price:,.2f}</span>"
            f"<span style='background:{GREEN}22;color:{GREEN};border-radius:3px;padding:2px 7px;font-weight:800;font-size:0.6rem;font-variant-numeric:tabular-nums;white-space:nowrap' title='{_reward_tip}'>+{reward_pct:.1f}%</span>"
            f"</div>"
            # Stop row
            f"<div style='display:grid;grid-template-columns:{_gcol} 1fr auto;gap:8px;align-items:center;font-size:0.625rem;flex-grow:1;overflow:visible'>"
            f"<span style='color:{TEXT_MUTED};font-weight:600'>STP {_si}</span>"
            f"<span style='color:{TEXT_PRIMARY};font-weight:700;font-variant-numeric:tabular-nums'>${_stop_loss:,.2f}</span>"
            f"<span style='background:{RED}22;color:{RED};border-radius:3px;padding:2px 7px;font-weight:800;font-size:0.6rem;font-variant-numeric:tabular-nums;white-space:nowrap' title='{_risk_tip}'>−{risk_pct:.1f}%</span>"
            f"</div>"
            f"</div>", unsafe_allow_html=True)
        _first_rr_card = False


# ── Why This Recommendation? ──────────────────────────────────────────────────
with st.expander("Why this recommendation?", expanded=False):

    # ── Plain-English summary narrative at the top ───────────────────────────
    p1m = predictions.get("1 Month", predictions.get("3 Day", {}))
    ensemble_agree = p1m.get('ensemble_agreement', 0)
    hist_acc = p1m.get('val_dir_accuracy', 0)
    ensemble_agree_color = _cc(ensemble_agree) if ensemble_agree else TEXT_SECONDARY
    _regime_label = (regime_info.get("label", "Sideways")
                     if regime_info else "Sideways")

    # Build a narrative sentence that explains the recommendation
    _conf_word = "strong" if conf1m >= 70 else ("moderate" if conf1m >= 55 else "weak")
    if "Buy" in rec:
        _rec_dir = "bullish"
    elif "Sell" in rec:
        _rec_dir = "bearish"
    else:
        _rec_dir = "neutral"
    _agree_phrase = (
        f"its 16 sub-models are in strong agreement ({ensemble_agree:.0f}%)"
        if ensemble_agree >= 65 else
        f"its 16 sub-models are split ({ensemble_agree:.0f}% agreement)"
        if ensemble_agree >= 45 else
        f"its 16 sub-models disagree meaningfully ({ensemble_agree:.0f}% agreement)"
    )
    _regime_phrase = {
        "Bull":     "the broader market is in a bull regime",
        "Bear":     "the broader market is in a bear regime",
        "Sideways": "the broader market is chopping sideways",
    }.get(_regime_label, "the market regime is unclear")

    _narrative = (
        f"Prediqt's 1-month call is <b style='color:{color}'>{rec.upper()}</b> "
        f"with <b style='color:{_cc(conf1m)}'>{_conf_word} confidence</b> ({conf1m:.0f}%). "
        f"Under the hood, {_agree_phrase}, and {_regime_phrase}. "
        f"On similar setups, the model has been right "
        f"<b style='color:{CHART_UP}'>{hist_acc:.1f}%</b> of the time on held-out data."
    )

    st.markdown(
        f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
        f"border:1px solid {BORDER};border-radius:12px;padding:16px 20px;"
        f"margin-bottom:18px;position:relative;overflow:hidden'>"
        # Hairline
        f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
        f"background:{BRAND_GRAD};opacity:0.65'></div>"
        f"<div style='color:{CYAN};font-size:0.58rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.14em;margin-bottom:8px'>"
        f"Summary</div>"
        f"<div style='color:{TEXT_PRIMARY};font-size:0.86rem;line-height:1.65;"
        f"font-weight:400;letter-spacing:0.005em'>{_narrative}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── 2-col: Confidence factors + Signal mix (balanced heights) ───────────
    rec_col1, rec_col2 = st.columns(2, gap="medium")

    with rec_col1:
        st.markdown(
            "<div class='sec-head' style='margin-bottom:12px'>"
            "<span class='sec-bar'></span>Confidence Factors</div>",
            unsafe_allow_html=True)

        # Three bars showing the three inputs that build confidence
        _factor_rows = []
        # 1. Model agreement
        _factor_rows.append((
            "Model agreement",
            f"{ensemble_agree:.0f}%",
            ensemble_agree / 100.0,
            ensemble_agree_color,
            "How aligned the 16 sub-models are. High = unanimous, low = split.",
        ))
        # 2. Historical accuracy
        _factor_rows.append((
            "Historical accuracy",
            f"{hist_acc:.1f}%",
            hist_acc / 100.0,
            _cc(hist_acc),
            "How often the model has been right on similar setups.",
        ))
        # 3. Regime fit
        _regime_score = 1.0 if _regime_label == "Bull" else (0.5 if _regime_label == "Sideways" else 0.3)
        _regime_color = CHART_UP if _regime_label == "Bull" else (AMBER if _regime_label == "Sideways" else CHART_DOWN)
        _factor_rows.append((
            "Market regime",
            _regime_label,
            _regime_score,
            _regime_color,
            "Models perform best in trending markets, worst in chop.",
        ))
        # Final composite
        _factor_rows.append((
            "Final confidence",
            f"{conf1m:.0f}%",
            conf1m / 100.0,
            _cc(conf1m),
            "The combined signal after weighing all three inputs above.",
        ))

        _bars_html = (
            f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
            f"border:1px solid {BORDER};border-radius:10px;padding:16px 18px'>"
        )
        for i, (label, val_str, pct, clr, desc) in enumerate(_factor_rows):
            _is_last = (i == len(_factor_rows) - 1)
            _border_top = (f"border-top:1px solid {BORDER};padding-top:12px;margin-top:12px"
                           if _is_last else "")
            _weight = "800" if _is_last else "600"
            _bars_html += (
                f"<div style='{_border_top};margin-bottom:{'0' if _is_last else '12px'}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
                f"margin-bottom:5px'>"
                f"<span style='color:{TEXT_SECONDARY};font-size:0.76rem;font-weight:{_weight}'>"
                f"{label}</span>"
                f"<span style='color:{clr};font-size:0.78rem;font-weight:800;"
                f"font-variant-numeric:tabular-nums'>{val_str}</span>"
                f"</div>"
                f"<div style='background:{BORDER};border-radius:100px;"
                f"height:5px;overflow:hidden;margin-bottom:6px'>"
                f"<div style='width:{min(100, max(0, pct*100)):.1f}%;"
                f"background:{clr};height:100%;border-radius:100px'></div></div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.64rem;line-height:1.4'>"
                f"{desc}</div>"
                f"</div>"
            )
        _bars_html += "</div>"
        st.markdown(_bars_html, unsafe_allow_html=True)

    with rec_col2:
        st.markdown(
            "<div class='sec-head' style='margin-bottom:12px'>"
            "<span class='sec-bar'></span>Technical Signal Mix</div>",
            unsafe_allow_html=True)

        # Technical signals grouped by side — list them individually, not just counts.
        # That way user sees "Oh, Price > 200MA is bullish, RSI is bullish, etc."
        bull_signals = [s for s in analysis["signals"] if s[1] == "Bullish"]
        bear_signals = [s for s in analysis["signals"] if s[1] == "Bearish"]
        neut_signals = [s for s in analysis["signals"] if s[1] not in ("Bullish", "Bearish")]
        total = len(analysis["signals"])

        def _signal_row(signal_list, side_label, side_color, side_bg):
            if not signal_list:
                return ""
            chips = "".join(
                f"<span style='display:inline-flex;align-items:center;padding:3px 9px;"
                f"border-radius:100px;background:{side_bg};border:1px solid {side_color}44;"
                f"color:{side_color};font-size:0.66rem;font-weight:700;margin:0 4px 4px 0;"
                f"letter-spacing:0.02em'>{s[0]}</span>"
                for s in signal_list
            )
            return (
                f"<div style='margin-bottom:10px'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
                f"margin-bottom:6px'>"
                f"<span style='color:{side_color};font-size:0.56rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.14em'>{side_label}</span>"
                f"<span style='color:{side_color};font-size:0.72rem;font-weight:800;"
                f"font-variant-numeric:tabular-nums'>{len(signal_list)}/{total}</span>"
                f"</div>"
                f"<div style='display:flex;flex-wrap:wrap'>{chips}</div>"
                f"</div>"
            )

        _mix_html = (
            f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
            f"border:1px solid {BORDER};border-radius:10px;padding:16px 18px'>"
            + _signal_row(bull_signals, "Bullish signals", CHART_UP, "rgba(6,214,160,0.08)")
            + _signal_row(neut_signals, "Neutral signals", AMBER,    "rgba(245,158,11,0.08)")
            + _signal_row(bear_signals, "Bearish signals", CHART_DOWN,"rgba(224,74,74,0.08)")
            + f"<div style='border-top:1px solid {BORDER};padding-top:10px;margin-top:4px;"
              f"color:{TEXT_MUTED};font-size:0.66rem;line-height:1.55'>"
              f"The ML model weights these signals differently than equal voting — "
              f"some carry more predictive power than others.</div>"
            + f"</div>"
        )
        st.markdown(_mix_html, unsafe_allow_html=True)

    # ── Top Signal Drivers — humanized names + inline descriptions ──────────
    st.markdown(
        "<div class='sec-head' style='margin:24px 0 6px'>"
        "<span class='sec-bar'></span>Top Signal Drivers</div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{TEXT_MUTED};font-size:0.7rem;margin-bottom:14px;"
        f"line-height:1.6;max-width:640px'>"
        f"These are the <b style='color:{TEXT_SECONDARY}'>5 features the model "
        f"paid the most attention to</b> when making this call — "
        f"picked from 160+ signals it tracks. A longer bar means the model "
        f"weighted that feature more heavily. Hover over any row for a deeper "
        f"explanation.</div>",
        unsafe_allow_html=True)

    try:
        imp_df = predictor.feature_importance("1 Month") if "1 Month" in predictions else predictor.feature_importance("3 Day")
        if not imp_df.empty:
            top_feats = imp_df.nlargest(5, "importance")
            max_imp = top_feats["importance"].max()

            _drivers_html = (
                f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
                f"border:1px solid {BORDER};border-radius:10px;padding:16px 18px;"
                f"position:relative;overflow:hidden'>"
                # Brand gradient hairline
                f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
                f"background:{BRAND_GRAD};opacity:0.55'></div>"
            )
            for i, (_, row) in enumerate(top_feats.iterrows()):
                fname = row["feature"]
                imp = row["importance"]
                display_label, icon, description = humanize_feature(fname)
                bw = (imp / max_imp) * 100
                _is_last = i == len(top_feats) - 1
                _divider = (f"border-bottom:1px solid {BORDER};padding-bottom:12px;margin-bottom:12px"
                            if not _is_last else "")
                # Escape tooltip for safe HTML attribute
                _tt = (description.replace("&", "&amp;").replace('"', "&quot;")
                              .replace("'", "&#39;").replace("<", "&lt;")
                              .replace(">", "&gt;"))
                _drivers_html += (
                    f"<div style='{_divider}' title='{_tt}'>"
                    # Top row: icon + label on left, importance value on right
                    f"<div style='display:flex;align-items:center;gap:10px;"
                    f"margin-bottom:5px'>"
                    f"<span style='font-size:0.95rem;min-width:22px;"
                    f"text-align:center'>{icon}</span>"
                    f"<div style='flex:1;color:{TEXT_PRIMARY};"
                    f"font-size:0.8rem;font-weight:700;letter-spacing:0.005em'>"
                    f"{display_label}</div>"
                    f"<span style='color:{TEXT_SECONDARY};font-size:0.68rem;"
                    f"font-weight:600;font-variant-numeric:tabular-nums;"
                    f"min-width:48px;text-align:right'>{imp:.3f}</span>"
                    f"</div>"
                    # Inline description — always visible, not hover-only
                    f"<div style='color:{TEXT_MUTED};font-size:0.68rem;"
                    f"line-height:1.5;padding-left:32px;margin-bottom:6px'>"
                    f"{description or '—'}</div>"
                    # Importance bar
                    f"<div style='background:{BORDER};border-radius:100px;"
                    f"height:5px;overflow:hidden;margin-left:32px'>"
                    f"<div style='width:{bw:.1f}%;background:{BRAND_GRAD};"
                    f"height:100%;border-radius:100px'></div></div>"
                    f"</div>"
                )
            _drivers_html += "</div>"
            st.markdown(_drivers_html, unsafe_allow_html=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — EVIDENCE STRIP (30-second validation)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"<div style='margin:36px 0 16px;'></div>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center;margin-bottom:22px'>"
    f"<div style='display:inline-flex;align-items:center;gap:8px;padding:4px 14px;"
    f"border-radius:100px;border:1px solid rgba(6,214,160,0.35);"
    f"background:rgba(6,214,160,0.08);color:{CYAN};font-size:0.6rem;font-weight:700;"
    f"letter-spacing:0.14em;text-transform:uppercase;margin-bottom:12px'>"
    f"Deeper Analysis</div>"
    f"<h2 style='color:{TEXT_PRIMARY};font-family:\"Inter\",sans-serif;"
    f"font-size:1.5rem;font-weight:800;letter-spacing:-0.02em;margin:0'>"
    f"Signal Evidence &amp; Backtesting</h2>"
    f"</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CATALYSTS — moved to the top of Deeper Analysis. Shows upcoming events
# the model is aware of + active macro regime factors. Full width row.
# ══════════════════════════════════════════════════════════════════════════════
try:
    from macro_calendar import upcoming_catalysts as _upcoming_macro
    from datetime import date as _date

    # ── Gather catalysts (this-ticker earnings + universal macro) ──────────
    _catalysts = []

    # Earnings (ticker-specific)
    _earn_days = earnings.get("days_to_next_earnings") if earnings else None
    if _earn_days is not None and _earn_days == _earn_days:  # NaN-safe
        _earn_days = int(_earn_days)
        if 0 <= _earn_days <= 90:
            _beat_rate = earnings.get("beat_rate_ytd")
            _last_beat = earnings.get("last_beat")
            _beat_str = ""
            if _beat_rate is not None and _beat_rate == _beat_rate:
                _beat_pct = int(round(_beat_rate * 100))
                _beat_str = f"{_beat_pct}% beat rate YTD"
            elif _last_beat is not None and _last_beat == _last_beat:
                _beat_str = "beat last qtr" if _last_beat > 0 else (
                    "missed last qtr" if _last_beat < 0 else "in-line last qtr")
            _catalysts.append({
                "kind": "EARNINGS",
                "days_out": _earn_days,
                "label": f"{symbol} earnings",
                "sub": _beat_str or "",
            })

    # Macro events (universal) — dedupe to next occurrence of each kind so
    # we don't double-count (e.g. two jobs reports within 45 days).
    try:
        _macro_events = _upcoming_macro(as_of=_date.today(), within_days=45)
    except Exception:
        _macro_events = []
    _seen_kinds = set()
    for _ev in sorted(_macro_events, key=lambda e: e["days_out"]):
        if _ev["kind"] in _seen_kinds:
            continue
        _seen_kinds.add(_ev["kind"])
        _catalysts.append({
            "kind": _ev["kind"],
            "days_out": _ev["days_out"],
            "label": _ev["label"],
            "sub": "",
        })

    # Sort by proximity
    _catalysts.sort(key=lambda c: c["days_out"])

    # Urgency color by days_out
    def _cat_color(days):
        if days <= 3:   return CHART_DOWN,    "rgba(224,74,74,0.10)",  "rgba(224,74,74,0.32)",  "HIGH"
        if days <= 14:  return AMBER,         "rgba(245,158,11,0.10)", "rgba(245,158,11,0.32)", "MED"
        return          (CHART_UP,            "rgba(6,214,160,0.10)",  "rgba(6,214,160,0.32)",  "LOW")

    # ── Gather regime factors from market_ctx.sentiment ────────────────────
    _sentiment = (market_ctx or {}).get("sentiment", {}) or {}

    def _last_val(s):
        """Safely pick the last non-NaN value from a pandas Series or return None."""
        try:
            if s is None or len(s) == 0:
                return None
            s = s.dropna()
            return float(s.iloc[-1]) if len(s) else None
        except Exception:
            return None

    _oil_mom  = _last_val(_sentiment.get("oil_momentum_21d"))
    _gold_mom = _last_val(_sentiment.get("gold_momentum_21d"))
    _tsy_lvl  = _last_val(_sentiment.get("treasury_10y_level"))
    _tsy_mom  = _last_val(_sentiment.get("treasury_10y_momentum_21d"))
    _credit   = _last_val(_sentiment.get("credit_stress"))

    # VIX from market_ctx
    _vix_val_latest = None
    try:
        _vix_series = (market_ctx or {}).get("vix")
        _vix_val_latest = _last_val(_vix_series) if _vix_series is not None else None
    except Exception:
        pass

    def _pill(label, value, interp, clr, arrow=""):
        return {"label": label, "value": value, "interp": interp, "clr": clr, "arrow": arrow}

    _regime_pills = []
    if _oil_mom is not None:
        _pct = _oil_mom * 100
        _arrow = "▲" if _pct >= 0 else "▼"
        if abs(_pct) >= 5:
            _interp = "energy tailwind" if _pct >= 5 else "energy headwind"
            _clr = CHART_UP if _pct >= 0 else CHART_DOWN
        elif abs(_pct) >= 2:
            _interp = "modestly " + ("up" if _pct >= 0 else "down")
            _clr = TEXT_SECONDARY
        else:
            _interp = "quiet"
            _clr = TEXT_MUTED
        _regime_pills.append(_pill("Oil 21d", f"{_arrow} {abs(_pct):.1f}%", _interp, _clr, _arrow))

    if _tsy_lvl is not None:
        _abs_rate = _tsy_lvl * 3.0 + 3.0
        _rising = _tsy_mom is not None and _tsy_mom > 0.1
        _falling = _tsy_mom is not None and _tsy_mom < -0.1
        if _rising:
            _interp, _clr, _arrow = "rising — headwind for multiples", CHART_DOWN, "▲"
        elif _falling:
            _interp, _clr, _arrow = "falling — tailwind for multiples", CHART_UP, "▼"
        else:
            _interp, _clr, _arrow = "stable", TEXT_SECONDARY, "→"
        _regime_pills.append(_pill("10Y yield", f"{_abs_rate:.2f}%", _interp, _clr, _arrow))

    if _vix_val_latest is not None:
        if _vix_val_latest < 15:
            _interp, _clr = "complacent", AMBER
        elif _vix_val_latest < 20:
            _interp, _clr = "calm", CHART_UP
        elif _vix_val_latest < 25:
            _interp, _clr = "moderate", AMBER
        elif _vix_val_latest < 30:
            _interp, _clr = "elevated", CHART_DOWN
        else:
            _interp, _clr = "stressed", CHART_DOWN
        _regime_pills.append(_pill("VIX", f"{_vix_val_latest:.1f}", _interp, _clr))

    if _gold_mom is not None:
        _pct = _gold_mom * 100
        _arrow = "▲" if _pct >= 0 else "▼"
        if abs(_pct) >= 3:
            _interp = ("defensive bid" if _pct >= 3 else "risk-on rotation")
            _clr = AMBER if _pct >= 3 else CHART_UP
        else:
            _interp = "neutral"
            _clr = TEXT_MUTED
        _regime_pills.append(_pill("Gold 21d", f"{_arrow} {abs(_pct):.1f}%", _interp, _clr, _arrow))

    if _credit is not None:
        if _credit > 0.005:
            _interp, _clr, _arrow = "easing", CHART_UP, "▲"
        elif _credit < -0.005:
            _interp, _clr, _arrow = "widening", CHART_DOWN, "▼"
        else:
            _interp, _clr, _arrow = "stable", TEXT_SECONDARY, "→"
        _regime_pills.append(_pill("Credit", f"{_credit*100:+.2f}%", _interp, _clr, _arrow))

    # ── Render Catalysts section ───────────────────────────────────────────
    _total_events = len(_catalysts)
    _n_label = f"{_total_events} on deck" if _total_events else "none scheduled"

    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:10px;margin-bottom:12px'>"
        f"<div class='sec-head' style='margin:0'><span class='sec-bar'></span>Catalysts</div>"
        f"<span style='color:{TEXT_MUTED};font-size:0.68rem;font-weight:600'>"
        f"· {_n_label} · these are the events the model is aware of</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if _catalysts:
        _event_cards_html = "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-bottom:14px'>"
        for _c in _catalysts[:6]:
            _clr, _bg, _bd, _tag = _cat_color(_c["days_out"])
            _days_label = "today" if _c["days_out"] == 0 else (
                "tomorrow" if _c["days_out"] == 1 else f"{_c['days_out']}d out")
            _kind_icon = {
                "EARNINGS": "📊",
                "FOMC":     "🏛",
                "CPI":      "📈",
                "NFP":      "👥",
            }.get(_c["kind"], "◆")
            _event_cards_html += (
                f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
                f"border:1px solid {BORDER};border-left:3px solid {_clr};border-radius:10px;"
                f"padding:12px 14px;position:relative'>"
                f"<div style='display:flex;align-items:center;justify-content:space-between;"
                f"margin-bottom:6px'>"
                f"<span style='font-size:0.9rem'>{_kind_icon}</span>"
                f"<span style='color:{_clr};font-size:0.48rem;font-weight:700;"
                f"letter-spacing:0.14em;background:{_bg};border:1px solid {_bd};"
                f"padding:2px 7px;border-radius:100px'>{_tag}</span>"
                f"</div>"
                f"<div style='color:{TEXT_PRIMARY};font-size:0.78rem;font-weight:700;"
                f"letter-spacing:-0.005em;line-height:1.2;margin-bottom:4px'>{_c['label']}</div>"
                f"<div style='color:{_clr};font-size:0.7rem;font-weight:700;"
                f"font-variant-numeric:tabular-nums'>{_days_label}</div>"
                + (f"<div style='color:{TEXT_MUTED};font-size:0.62rem;margin-top:4px'>"
                   f"{_c['sub']}</div>" if _c["sub"] else "")
                + f"</div>"
            )
        _event_cards_html += "</div>"
        st.markdown(_event_cards_html, unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;"
            f"padding:14px 16px;color:{TEXT_MUTED};font-size:0.76rem;margin-bottom:14px'>"
            f"No scheduled catalysts in the next 45 days.</div>",
            unsafe_allow_html=True,
        )

    if _regime_pills:
        _pill_html_parts = [
            f"<div style='color:{TEXT_MUTED};font-size:0.56rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.14em;margin-right:6px;"
            f"display:inline-flex;align-items:center'>Regime factors</div>"
        ]
        for _p in _regime_pills:
            _pill_html_parts.append(
                f"<span style='display:inline-flex;align-items:center;gap:6px;"
                f"padding:5px 12px;border-radius:100px;"
                f"background:{BG_CARD};border:1px solid {BORDER};font-size:0.68rem;"
                f"font-weight:600;margin:0 6px 6px 0'>"
                f"<span style='color:{TEXT_MUTED};font-size:0.58rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.1em'>{_p['label']}</span>"
                f"<span style='color:{_p['clr']};font-weight:800;"
                f"font-variant-numeric:tabular-nums'>{_p['value']}</span>"
                f"<span style='color:{TEXT_SECONDARY};font-size:0.64rem;font-weight:500'>"
                f"· {_p['interp']}</span>"
                f"</span>"
            )
        st.markdown(
            "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:2px;"
            "margin-bottom:22px'>" + "".join(_pill_html_parts) + "</div>",
            unsafe_allow_html=True,
        )
except Exception as _catalysts_exc:
    st.caption(f"(Catalysts panel unavailable: {type(_catalysts_exc).__name__})")


# ══════════════════════════════════════════════════════════════════════════════
# Evidence cards — 2-col layout: Consensus + Drivers stacked on left, Trade
# Setup on the right. Keeps heights balanced and eliminates the dead space
# the old 3-col layout had under the short cards.
# ══════════════════════════════════════════════════════════════════════════════
ev_left, ev_right = st.columns([1, 1.2], gap="medium")

# ── Signal gauge ─────────────────────────────────────────────────────────────
with ev_left:
    bull_n = sum(1 for s in analysis["signals"] if s[1]=="Bullish")
    bear_n = sum(1 for s in analysis["signals"] if s[1]=="Bearish")
    total  = len(analysis["signals"])
    neut_n = total - bull_n - bear_n
    bull_pct = int(bull_n / total * 100) if total > 0 else 0
    bear_pct = int(bear_n / total * 100) if total > 0 else 0
    neut_pct = 100 - bull_pct - bear_pct

    st.markdown(
        f"<div class='card'>"
        f"<div class='sec-head'><span class='sec-bar'></span>Signal Consensus</div>"
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

# ── Key signals (stacked below Signal Consensus in the same left column) ────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    pills_html = ""
    for name, direction, desc in analysis["signals"]:
        cls = {"Bullish":"evpill-bull","Bearish":"evpill-bear"}.get(direction,"evpill-neut")
        icon = {"Bullish":"▲","Bearish":"▼"}.get(direction,"→")
        pills_html += f"<span class='evpill {cls}'>{icon} {name}</span>  "

    st.markdown(
        f"<div class='card'>"
        f"<div class='sec-head'><span class='sec-bar'></span>Key Drivers</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:6px'>{pills_html}</div>"
        f"</div>", unsafe_allow_html=True)

# ── Trade setup (right column) ──────────────────────────────────────────────
with ev_right:
    ez = analysis["entry_zone"]
    rr = ez['risk_reward']
    rr_color = GREEN if rr and rr >= 2 else AMBER if rr and rr >= 1 else RED if rr else TEXT_MUTED

    # Pre-compute the dollar components so we can show the equation
    _ts_reward = round(ez['first_target'] - ez['entry_low'], 2)
    _ts_risk   = round(ez['entry_high']   - ez['stop_loss'],  2)
    _atr_val   = ez.get('atr', 0)
    _atr_pct   = ez.get('atr_pct', 0)

    st.markdown(
        f"<div class='card'>"
        # Header
        f"<div class='sec-head'><span class='sec-bar'></span>Trade Setup</div>"

        # Entry row
        f"<div class='mrow'>"
        f"<span class='mrow-label'>Entry Zone</span>"
        f"<span class='mrow-value'>${ez['entry_low']:,.2f} – ${ez['entry_high']:,.2f}</span>"
        f"</div>"
        f"<div style='font-size:0.62rem;color:{TEXT_MUTED};margin:-4px 0 8px 0;padding-left:2px'>"
        f"Support level → current price</div>"

        # Stop loss row
        f"<div class='mrow'>"
        f"<span class='mrow-label'>Stop Loss</span>"
        f"<span class='mrow-value' style='color:{RED}'>${ez['stop_loss']:,.2f}</span>"
        f"</div>"
        f"<div style='font-size:0.62rem;color:{TEXT_MUTED};margin:-4px 0 8px 0;padding-left:2px;overflow:visible'>"
        f"2nd support − ATR{info_icon(TIPS['ATR'])} (${_atr_val:,.2f} / {_atr_pct:.1f}%)</div>"

        # Target row
        f"<div class='mrow'>"
        f"<span class='mrow-label'>Target</span>"
        f"<span class='mrow-value' style='color:{GREEN}'>${ez['first_target']:,.2f}</span>"
        f"</div>"
        f"<div style='font-size:0.62rem;color:{TEXT_MUTED};margin:-4px 0 12px 0;padding-left:2px;overflow:visible'>"
        f"ML{info_icon(TIPS['ML'])} 1-month prediction or first resistance</div>"

        # ── Compact R/R summary — reward / risk / ratio in a single block ──
        f"<div style='border-top:1px solid {BORDER};margin:10px 0 12px'></div>"

        # Reward + Risk on two compact lines
        f"<div style='display:grid;grid-template-columns:auto 1fr auto;"
        f"gap:6px 14px;align-items:center;font-size:0.74rem;line-height:1.4;"
        f"font-variant-numeric:tabular-nums'>"
        # Reward row
        f"<span style='color:{TEXT_MUTED};font-size:0.58rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.12em'>Reward</span>"
        f"<span style='color:{TEXT_SECONDARY};font-size:0.66rem'>"
        f"Target − Best entry</span>"
        f"<span style='color:{CHART_UP};font-weight:800'>+${_ts_reward:,.2f}</span>"
        # Risk row
        f"<span style='color:{TEXT_MUTED};font-size:0.58rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.12em'>Risk</span>"
        f"<span style='color:{TEXT_SECONDARY};font-size:0.66rem'>"
        f"Worst entry − Stop</span>"
        f"<span style='color:{CHART_DOWN};font-weight:800'>−${_ts_risk:,.2f}</span>"
        f"</div>"

        # Headline R/R row — big number + quality badge together
        f"<div style='display:flex;align-items:center;justify-content:space-between;"
        f"gap:10px;padding:12px 14px;margin-top:12px;border-radius:8px;"
        f"background:linear-gradient(145deg,{BG_ELEVATED},{BG_CARD});"
        f"border:1px solid {BORDER}'>"
        f"<div>"
        f"<div style='color:{TEXT_MUTED};font-size:0.54rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.14em;margin-bottom:2px'>"
        f"Risk / reward</div>"
        f"<div style='font-family:\"Inter\",sans-serif;color:{rr_color};"
        f"font-size:1.4rem;font-weight:900;letter-spacing:-0.02em;line-height:1;"
        f"font-variant-numeric:tabular-nums'>"
        f"{rr:.1f}×</div>"
        f"</div>"
        f"<span style='display:inline-flex;align-items:center;padding:4px 12px;"
        f"border-radius:100px;background:{rr_color}1a;border:1px solid {rr_color}44;"
        f"color:{rr_color};font-size:0.62rem;font-weight:800;letter-spacing:0.12em;"
        f"text-transform:uppercase'>"
        f"{'Excellent' if rr and rr >= 3 else 'Favorable' if rr and rr >= 2 else 'Moderate' if rr and rr >= 1 else 'Poor'}"
        f"</span>"
        f"</div>"

        f"</div>",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — DEEP DIVE TABS (2-minute analysis)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"<div style='margin:36px 0 16px;'></div>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center;margin-bottom:20px'>"
    f"<div style='display:inline-flex;align-items:center;gap:8px;padding:4px 14px;"
    f"border-radius:100px;border:1px solid rgba(6,214,160,0.35);"
    f"background:rgba(6,214,160,0.08);color:{CYAN};font-size:0.6rem;font-weight:700;"
    f"letter-spacing:0.14em;text-transform:uppercase;margin-bottom:12px'>"
    f"Deep Dive</div>"
    f"<h2 style='color:{TEXT_PRIMARY};font-family:\"Inter\",sans-serif;"
    f"font-size:1.5rem;font-weight:800;letter-spacing:-0.02em;margin:0'>"
    f"Charts · Analysis · Strategy</h2>"
    f"</div>",
    unsafe_allow_html=True)

st.markdown("<hr style='margin:16px 0'>", unsafe_allow_html=True)

tab_chart, tab_analysis, tab_opts, tab_bt, tab_model = st.tabs([
    "Chart",
    "Analysis",
    "Options",
    "Backtest",
    "Model",
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
        increasing=dict(line=dict(color=CHART_UP),   fillcolor=CHART_UP_FILL),
        decreasing=dict(line=dict(color=CHART_DOWN), fillcolor=CHART_DOWN_FILL),
    ), row=1, col=1)

    # MA palette — each line a clearly distinct, readable color. Avoids the
    # old overlap where MA50 used brand-blue (same as RSI AND the 1Q target).
    _ma_palette = [
        (20,  "#f59e0b", "MA 20"),    # amber — fast MA
        (50,  "#a78bfa", "MA 50"),    # soft lavender
        (200, "#60a5fa", "MA 200"),   # sky blue — slowest, coolest
    ]
    for ma, mc, mn in _ma_palette:
        if len(chart_df) >= ma:
            fig.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df["Close"].rolling(ma).mean(),
                name=mn, line=dict(color=mc, width=1.4),
            ), row=1, col=1)

    # Horizon prediction lines — all cyan (the brand's "forecast" color)
    # differentiated by line pattern and annotation. This makes them visually
    # cohesive ("these are all model forecasts") while still distinct from MAs.
    target_styles = {
        "1 Week":    ("dot",      0.55),
        "1 Month":   ("dash",     0.75),
        "1 Quarter": ("dashdot",  0.90),
        "1 Year":    ("longdash", 1.00),
    }
    for h, (hd, alpha) in target_styles.items():
        if h not in predictions:
            continue
        tp = predictions[h]["predicted_price"]
        if abs(tp - cur) / cur < 0.5:
            # Blend cyan with white based on horizon length — further out = faded
            _hc = f"rgba(6,214,160,{alpha:.2f})"
            fig.add_hline(y=tp, line=dict(color=_hc, width=1.2, dash=hd),
                          annotation_text=f"  {h} ${tp:,.0f}",
                          annotation_position="right",
                          annotation_font=dict(color=_hc, size=9, family="Inter"),
                          row=1, col=1)

    vc = [CHART_UP if c >= o else CHART_DOWN for c, o in zip(chart_df["Close"], chart_df["Open"])]
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"],
                         marker_color=vc, name="Vol", showlegend=False, marker_opacity=0.55), row=2, col=1)
    # Volume MA — muted so it doesn't compete with price MAs visually
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Volume"].rolling(20).mean(),
                             name="Vol MA 20",
                             line=dict(color="#6b7c90", width=1.2, dash="dot"),
                             showlegend=False), row=2, col=1)

    # RSI — distinct violet so it never collides with any price-pane line.
    rsi_s = rsi_series(chart_df["Close"])
    fig.add_trace(go.Scatter(x=chart_df.index, y=rsi_s, name="RSI",
                             line=dict(color="#8b5cf6", width=1.6)), row=3, col=1)
    for lvl, ann, lc in [(70, "Overbought", CHART_DOWN), (50, "", "#334155"), (30, "Oversold", CHART_UP)]:
        fig.add_hline(y=lvl, line=dict(color=lc,width=1,dash="dash"),
                      annotation_text=f" {ann}" if ann else "",
                      annotation_font=dict(color=lc,size=9), row=3, col=1)

    fig.update_layout(
        height=620, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=10,r=10,t=30,b=10),
        xaxis_rangeslider_visible=False, hovermode="x unified",
    )
    fig.update_yaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID)
    fig.update_xaxes(gridcolor=CHART_GRID, showspikes=True, spikecolor=BORDER_ACCENT, spikethickness=1)
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
    an1, an2 = st.columns([1.25, 1])

    # ── Left: structured analysis card ───────────────────────────────────────
    with an1:
        tech      = analysis["technicals"]
        ez        = analysis["entry_zone"]
        sigs      = analysis["signals"]
        rec       = analysis["recommendation"]
        rec_color = analysis["color"]
        conf_str  = analysis["confidence"]

        bull_n    = sum(1 for s in sigs if s[1] == "Bullish")
        bear_n    = sum(1 for s in sigs if s[1] == "Bearish")
        total_n   = max(len(sigs), 1)
        neut_n    = total_n - bull_n - bear_n

        # ── Helpers ──
        # NOTE: using &#36; instead of '$' so Streamlit's MathJax doesn't
        # interpret paired dollar signs as LaTeX math mode.
        def _chip(label, value_html, sub_html, accent):
            return (
                f"<div style='flex:1;min-width:140px;padding:11px 13px;background:{BG_ELEVATED};"
                f"border:1px solid {BORDER};border-radius:8px;border-left:3px solid {accent}'>"
                f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:0.6px;font-weight:600;margin-bottom:5px'>{label}</div>"
                f"<div style='color:{TEXT_PRIMARY};font-size:0.95rem;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.25'>{value_html}</div>"
                f"<div style='color:{accent};font-size:0.66rem;margin-top:3px;font-weight:600'>{sub_html}</div>"
                f"</div>"
            )

        def _ma_chip(label, val, is_above):
            c = GREEN if is_above else RED
            arrow = "▲" if is_above else "▼"
            word = "above" if is_above else "below"
            return _chip(label, f"&#36;{val:,.2f}", f"{arrow} {word}", c)

        # RSI classification
        rsi_val = tech["rsi"]
        if rsi_val >= 70:
            rsi_c, rsi_lbl = RED, "Overbought"
        elif rsi_val <= 30:
            rsi_c, rsi_lbl = GREEN, "Oversold"
        elif rsi_val >= 50:
            rsi_c, rsi_lbl = GREEN, "Bullish zone"
        else:
            rsi_c, rsi_lbl = AMBER, "Neutral"

        macd_c     = GREEN if tech["macd_cross"] == "Bullish" else RED
        macd_arrow = "▲" if tech["macd_cross"] == "Bullish" else "▼"

        # Trend color
        tr_lbl = tech["trend"]
        tr_c   = GREEN if "Up" in tr_lbl else RED if "Down" in tr_lbl else AMBER

        # Bollinger
        bb_c = (GREEN if "Lower" in tech["bb_pos"]
                else RED if "Upper" in tech["bb_pos"]
                else AMBER)
        bb_arrow = ("▼" if "Upper" in tech["bb_pos"]
                    else "▲" if "Lower" in tech["bb_pos"]
                    else "→")

        # Risk/Reward
        rr      = ez.get("risk_reward", 0) or 0
        rr_c    = GREEN if rr >= 2 else AMBER if rr >= 1 else RED
        rr_lbl  = "Favorable" if rr >= 2 else "Marginal" if rr >= 1 else "Poor"

        # Signal bar widths
        bull_w = (bull_n / total_n) * 100
        neut_w = (neut_n / total_n) * 100
        bear_w = (bear_n / total_n) * 100

        # ── Build the full analysis card ──
        html = []

        # Overview line
        html.append(
            f"<div class='card' style='padding:22px 22px 18px'>"
            f"<div style='color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.6;margin-bottom:18px'>"
            f"<span style='color:{TEXT_PRIMARY};font-weight:600'>{info.get('name', symbol)}</span> "
            f"is trading at <b style='color:{TEXT_PRIMARY};font-variant-numeric:tabular-nums'>&#36;{analysis['current_price']:,.2f}</b>. "
            f"The model issues a <b style='color:{rec_color}'>{rec}</b> call with "
            f"<b style='color:{TEXT_PRIMARY}'>{conf_str.lower()} confidence</b>."
            f"</div>"
        )

        # Technical Trend
        html.append(
            f"<div style='margin-bottom:18px'>"
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>"
            f"<span style='color:{TEXT_MUTED};font-size:0.62rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600'>Technical Trend</span>"
            f"<span style='display:inline-block;padding:3px 10px;border-radius:4px;background:{tr_c}1f;color:{tr_c};font-size:0.7rem;font-weight:700;border:1px solid {tr_c}33'>{tr_lbl}</span>"
            f"</div>"
            f"<div style='display:flex;gap:8px;flex-wrap:wrap'>"
            f"{_ma_chip('20-day MA', tech['ma20'], tech['above_ma20'])}"
            f"{_ma_chip('50-day MA', tech['ma50'], tech['above_ma50'])}"
            f"{_ma_chip('200-day MA', tech['ma200'], tech['above_ma200'])}"
            f"</div>"
            f"</div>"
        )

        # Momentum — pre-compute display strings to keep f-strings clean
        macd_val_str  = f"{tech['macd']:+.3f}"
        macd_sub_str  = f"{macd_arrow} {tech['macd_cross']}"
        bb_sub_str    = f"{bb_arrow} {tech['bb_pct']:.0f}% of band"
        html.append(
            f"<div style='margin-bottom:18px'>"
            f"<div style='color:{TEXT_MUTED};font-size:0.62rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:10px'>Momentum</div>"
            f"<div style='display:flex;gap:8px;flex-wrap:wrap'>"
            f"{_chip('RSI (14)', f'{rsi_val:.1f}', rsi_lbl, rsi_c)}"
            f"{_chip('MACD', macd_val_str, macd_sub_str, macd_c)}"
            f"{_chip('Bollinger', tech['bb_pos'], bb_sub_str, bb_c)}"
            f"</div>"
            f"</div>"
        )

        # Signal Summary with visual bar
        html.append(
            f"<div style='margin-bottom:18px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>"
            f"<span style='color:{TEXT_MUTED};font-size:0.62rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600'>Signal Summary</span>"
            f"<span style='font-size:0.75rem;font-weight:600'>"
            f"<span style='color:{GREEN}'>▲ {bull_n} bullish</span>"
            f"<span style='color:{TEXT_MUTED}'> · </span>"
            f"<span style='color:{RED}'>▼ {bear_n} bearish</span>"
            + (f"<span style='color:{TEXT_MUTED}'> · {neut_n} neutral</span>" if neut_n > 0 else "")
            + f"</span></div>"
            f"<div style='display:flex;height:6px;border-radius:3px;overflow:hidden;background:{BG_ELEVATED}'>"
            + (f"<div style='width:{bull_w:.1f}%;background:{GREEN}'></div>" if bull_n else "")
            + (f"<div style='width:{neut_w:.1f}%;background:{TEXT_MUTED}'></div>" if neut_n else "")
            + (f"<div style='width:{bear_w:.1f}%;background:{RED}'></div>" if bear_n else "")
            + f"</div></div>"
        )

        # Trade Setup grid (4 cells)
        html.append(
            f"<div style='margin-bottom:4px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px'>"
            f"<span style='color:{TEXT_MUTED};font-size:0.62rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600'>Trade Setup</span>"
            f"<span style='display:inline-block;padding:3px 10px;border-radius:4px;background:{rr_c}1f;color:{rr_c};font-size:0.7rem;font-weight:700;border:1px solid {rr_c}33'>R/R {rr:.1f}× · {rr_lbl}</span>"
            f"</div>"
            f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px'>"
            # Entry Zone
            f"<div style='padding:10px 12px;background:{BG_ELEVATED};border:1px solid {BORDER};border-radius:8px'>"
            f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:0.6px;font-weight:600;margin-bottom:5px'>Entry Zone</div>"
            f"<div style='color:{TEXT_PRIMARY};font-size:0.9rem;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.3'>&#36;{ez['entry_low']:,.2f}</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.68rem;margin-top:2px'>to &#36;{ez['entry_high']:,.2f}</div>"
            f"</div>"
            # Stop Loss
            f"<div style='padding:10px 12px;background:{BG_ELEVATED};border:1px solid {BORDER};border-radius:8px;border-left:3px solid {RED}'>"
            f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:0.6px;font-weight:600;margin-bottom:5px'>Stop Loss</div>"
            f"<div style='color:{RED};font-size:0.95rem;font-weight:700;font-variant-numeric:tabular-nums'>&#36;{ez['stop_loss']:,.2f}</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.66rem;margin-top:3px'>{((ez['stop_loss']/analysis['current_price'])-1)*100:+.1f}%</div>"
            f"</div>"
            # Target
            f"<div style='padding:10px 12px;background:{BG_ELEVATED};border:1px solid {BORDER};border-radius:8px;border-left:3px solid {GREEN}'>"
            f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:0.6px;font-weight:600;margin-bottom:5px'>Target</div>"
            f"<div style='color:{GREEN};font-size:0.95rem;font-weight:700;font-variant-numeric:tabular-nums'>&#36;{ez['first_target']:,.2f}</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.66rem;margin-top:3px'>{((ez['first_target']/analysis['current_price'])-1)*100:+.1f}%</div>"
            f"</div>"
            # ATR
            f"<div style='padding:10px 12px;background:{BG_ELEVATED};border:1px solid {BORDER};border-radius:8px'>"
            f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:0.6px;font-weight:600;margin-bottom:5px'>ATR</div>"
            f"<div style='color:{TEXT_PRIMARY};font-size:0.95rem;font-weight:700;font-variant-numeric:tabular-nums'>&#36;{ez['atr']:.2f}</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.66rem;margin-top:3px'>{ez['atr_pct']:.1f}% daily</div>"
            f"</div>"
            f"</div>"
            f"</div>"
        )

        # Risk note (beta)
        beta = info.get("beta")
        if beta:
            if abs(beta - 1) < 0.15:
                vol_word, vol_c = "roughly in line with", TEXT_SECONDARY
            elif beta > 1:
                vol_word, vol_c = "more volatile than", AMBER
            else:
                vol_word, vol_c = "less volatile than", GREEN
            html.append(
                f"<div style='display:flex;align-items:center;gap:12px;padding:10px 14px;background:{BG_ELEVATED};border:1px solid {BORDER};border-radius:8px;margin-top:14px'>"
                f"<span style='color:{TEXT_MUTED};font-size:0.62rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600'>Risk</span>"
                f"<span style='color:{TEXT_SECONDARY};font-size:0.78rem'>Beta <b style='color:{TEXT_PRIMARY};font-variant-numeric:tabular-nums'>{beta:.2f}</b> — <span style='color:{vol_c};font-weight:600'>{vol_word} the market</span></span>"
                f"</div>"
            )

        # Disclaimer
        html.append(
            f"<div style='color:{TEXT_MUTED};font-size:0.68rem;line-height:1.5;margin-top:14px;padding-top:12px;border-top:1px solid {BORDER};font-style:italic'>"
            f"This analysis is generated by a machine-learning model for informational purposes only. Past performance does not guarantee future results. Always do your own research before making investment decisions."
            f"</div>"
        )

        # Close card
        html.append("</div>")

        st.markdown("".join(html), unsafe_allow_html=True)

    # ── Right: methodology card (refreshed) ──────────────────────────────────
    with an2:
        n_l1 = sum(1 for m in predictor.l1_members.get("1 Month", []))
        st.markdown(
            f"<div class='card' style='padding:22px'>"
            f"<div style='color:{BLUE};font-size:0.62rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:14px'>Methodology</div>"

            # Level 1
            f"<div style='padding:12px 14px;background:{BG_ELEVATED};border:1px solid {BORDER};border-radius:8px;border-left:3px solid {BLUE};margin-bottom:10px'>"
            f"<div style='color:{TEXT_PRIMARY};font-size:0.78rem;font-weight:700;margin-bottom:4px'>Level 1 · Base Models</div>"
            f"<div style='color:{TEXT_SECONDARY};font-size:0.76rem;line-height:1.55'>{n_l1} diverse learners (XGBoost, LightGBM, Random Forest, XGBClassifier) trained with feature bagging and exponential sample weighting.</div>"
            f"</div>"

            # Level 2
            f"<div style='padding:12px 14px;background:{BG_ELEVATED};border:1px solid {BORDER};border-radius:8px;border-left:3px solid {CYAN};margin-bottom:14px'>"
            f"<div style='color:{TEXT_PRIMARY};font-size:0.78rem;font-weight:700;margin-bottom:4px'>Level 2 · Meta-Learner</div>"
            f"<div style='color:{TEXT_SECONDARY};font-size:0.76rem;line-height:1.55'>Ridge regression over out-of-fold L1 predictions with isotonic confidence calibration.</div>"
            f"</div>"

            # Features block
            f"<div style='color:{TEXT_MUTED};font-size:0.62rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:8px'>Features (75+)</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:5px'>"
            + "".join(
                f"<span style='padding:3px 9px;background:{BG_ELEVATED};border:1px solid {BORDER};border-radius:12px;color:{TEXT_SECONDARY};font-size:0.68rem;font-weight:500'>{tag}</span>"
                for tag in [
                    "Momentum × 7", "MA crossovers", "RSI", "MACD", "Bollinger",
                    "Stochastic", "Ichimoku", "OBV", "VP trend",
                    "Variance ratio", "Autocorrelation", "SPY alpha",
                    "VIX regime", "Regime detect", "Candle patterns",
                ]
            )
            + f"</div>"
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

    # ── Stock Comparison View ─────────────────────────────────────────────
    with st.expander("Compare with another stock"):
        compare_sym = st.text_input("Enter a symbol to compare", key="compare_sym",
                                    placeholder="e.g., MSFT, AAPL, GOOGL").strip().upper()
        if compare_sym and compare_sym != symbol:
            with st.spinner(f"Analyzing {compare_sym}…"):
                try:
                    cmp_df = fetch_stock_data(compare_sym, period="2y")
                    cmp_info = fetch_stock_info(compare_sym)
                    cmp_mkt = fetch_market_context()
                    cmp_fund = fetch_fundamentals(compare_sym)
                    cmp_earn = fetch_earnings_data(compare_sym)
                    cmp_price = float(cmp_df["Close"].iloc[-1])
                    cmp_opts = fetch_options_data(compare_sym, cmp_price)

                    cmp_pred = StockPredictor(compare_sym)
                    cmp_pred.train(cmp_df, cmp_mkt, fundamentals=cmp_fund,
                                  earnings_data=cmp_earn, options_data=cmp_opts)
                    cmp_preds = cmp_pred.predict(cmp_df, cmp_mkt)

                    # Side-by-side comparison table
                    st.markdown(f"#### {symbol} vs {compare_sym}")
                    def _dir_label(ret):
                        if ret > 0.001: return "▲ Bullish"
                        if ret < -0.001: return "▼ Bearish"
                        return "→ Neutral"

                    comp_rows = []
                    for h_name in HORIZONS:
                        h1 = predictions.get(h_name, {})
                        h2 = cmp_preds.get(h_name, {})
                        h1_ret = h1.get('predicted_return', 0)
                        h2_ret = h2.get('predicted_return', 0)
                        comp_rows.append({
                            "Horizon": h_name,
                            f"{symbol} Return": f"{h1_ret*100:+.1f}%",
                            f"{symbol} Conf": f"{h1.get('confidence', 0):.0f}%",
                            f"{symbol} Dir": _dir_label(h1_ret) if h1 else "—",
                            f"{compare_sym} Return": f"{h2_ret*100:+.1f}%",
                            f"{compare_sym} Conf": f"{h2.get('confidence', 0):.0f}%",
                            f"{compare_sym} Dir": _dir_label(h2_ret) if h2 else "—",
                        })
                    comp_df = pd.DataFrame(comp_rows)
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

                    # Price chart overlay (normalised)
                    st.markdown("**Normalised Price Performance (6M)**")
                    fig_cmp = go.Figure()
                    n_bars = min(126, len(df), len(cmp_df))
                    s1_close = df["Close"].iloc[-n_bars:]
                    s2_close = cmp_df["Close"].iloc[-n_bars:]
                    s1_norm = s1_close / s1_close.iloc[0] * 100
                    s2_norm = s2_close / s2_close.iloc[0] * 100
                    fig_cmp.add_trace(go.Scatter(x=s1_norm.index, y=s1_norm.values,
                                                 mode="lines", name=symbol,
                                                 line=dict(color=CYAN, width=2.2)))
                    fig_cmp.add_trace(go.Scatter(x=s2_norm.index, y=s2_norm.values,
                                                 mode="lines", name=compare_sym,
                                                 line=dict(color=BLUE, width=2.2, dash="dash")))
                    fig_cmp.update_layout(
                        height=300, template="plotly_dark",
                        margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=11),
                        xaxis=dict(gridcolor=CHART_GRID),
                        yaxis=dict(gridcolor=CHART_GRID, title="Normalised (base=100)"),
                        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)
                except Exception as cmp_err:
                    st.error(f"Could not analyze {compare_sym}: {cmp_err}")
        elif compare_sym and compare_sym == symbol:
            st.info("Enter a different symbol to compare against.")


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
                st.markdown("<div class='sec-head'><span class='sec-bar'></span>Economics</div>", unsafe_allow_html=True)
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
                st.markdown("<div class='sec-head'><span class='sec-bar'></span>Rationale</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='card' style='color:{TEXT_SECONDARY};font-size:0.8rem;line-height:1.7;margin-bottom:8px'>"
                    f"{opt['rationale']}</div>"
                    f"<div style='background:{AMBER_DIM};border:1px solid #78350f40;border-left:3px solid {AMBER};"
                    f"border-radius:8px;padding:10px 14px;color:{AMBER};font-size:0.75rem'>"
                    f"⚠ {opt.get('risk_note','')}</div>", unsafe_allow_html=True)

            st.markdown("<div class='sec-head' style='margin:14px 0 6px'><span class='sec-bar'></span>Trade Legs</div>", unsafe_allow_html=True)
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
                reg_clr   = {"Bull": CHART_UP, "Bear": CHART_DOWN, "Sideways": AMBER}

                fig_acc = go.Figure()
                fig_acc.add_hline(y=50, line=dict(color=BORDER_ACCENT, dash="dash"),
                                  annotation_text=" 50% random",
                                  annotation_font=dict(color=TEXT_MUTED, size=9))
                fig_acc.add_trace(go.Bar(
                    x=acc_dates, y=acc_vals,
                    marker_color=[reg_clr.get(reg, CHART_NEUTRAL) for reg in regimes],
                    text=[f"{v:.0f}%" for v in acc_vals],
                    textposition="outside", textfont=dict(size=9, color=TEXT_SECONDARY),
                ))
                fig_acc.update_layout(
                    height=280, template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=11),
                    yaxis=dict(range=[0, 110], gridcolor=CHART_GRID),
                    xaxis=dict(gridcolor=CHART_GRID),
                    margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                )
                st.plotly_chart(fig_acc, use_container_width=True)
                st.caption("Cyan = Bull · Red = Bear · Amber = Sideways")

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
                name="ML Strategy", line=dict(color=CHART_UP, width=2.4),
                fill="tozeroy", fillcolor=CHART_UP_FILL,
            ))
            fig_eq.add_trace(go.Scatter(
                x=r["bh_cum"].index, y=r["bh_cum"].values,
                name="Buy & Hold", line=dict(color=CHART_NEUTRAL, width=1.8, dash="dash"),
            ))
            fig_eq.add_hline(y=1.0, line=dict(color=BORDER_ACCENT, width=1))
            fig_eq.update_layout(
                height=260, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=11),
                yaxis=dict(title="Portfolio Value ($1 start)", gridcolor=CHART_GRID),
                xaxis=dict(gridcolor=CHART_GRID),
                legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_eq, use_container_width=True)


# ── TAB: MODEL ───────────────────────────────────────────────────────────────

with tab_model:

    mc1, mc2 = st.columns([1, 1.4])

    with mc1:
        st.markdown("<div class='sec-head'><span class='sec-bar'></span>Validation Calibration</div>", unsafe_allow_html=True)
        st.dataframe(predictor.calibration_report(), hide_index=True, use_container_width=True)
        st.markdown(
            f"<div class='ctx' style='margin-top:8px'>"
            f"<b>Dir Acc</b> = accuracy on held-out 20% data. 50% = random, 55%+ = edge, 60%+ = strong.<br>"
            f"<b>Mean Abs Error</b> = average prediction error magnitude. Lower = more precise.</div>",
            unsafe_allow_html=True)

    with mc2:
        st.markdown("<div class='sec-head'><span class='sec-bar'></span>Prediction Distribution</div>", unsafe_allow_html=True)
        h_sel  = st.selectbox("Horizon", list(HORIZONS.keys()), index=1, key="ens_h")
        p_data = predictions[h_sel]
        ens_p  = p_data.get("all_preds", np.array([p_data["predicted_return"]]))

        fig_ens = go.Figure()
        fig_ens.add_trace(go.Histogram(
            x=ens_p * 100, nbinsx=8,
            marker_color=CHART_NEUTRAL, marker_line=dict(color=BLUE, width=1), opacity=0.85,
        ))
        fig_ens.add_vline(x=p_data["predicted_return"]*100,
                          line=dict(color=CHART_UP, width=2.5, dash="dash"),
                          annotation_text=f" Meta: {p_data['predicted_return']*100:.2f}%",
                          annotation_font=dict(color=CHART_UP, size=10))
        fig_ens.add_vline(x=0, line=dict(color=TEXT_MUTED, width=1))
        fig_ens.update_layout(
            height=220, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=11),
            xaxis_title="Predicted Return (%)", yaxis=dict(title="Count", gridcolor=CHART_GRID),
            xaxis=dict(gridcolor=CHART_GRID),
            margin=dict(l=10, r=10, t=15, b=10), showlegend=False,
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
    st.markdown("<div class='sec-head'><span class='sec-bar'></span>Feature Importance</div>", unsafe_allow_html=True)
    fi_h   = st.selectbox("Horizon", list(HORIZONS.keys()), index=1, key="fi_h2")
    imp_df = predictor.feature_importance(fi_h)
    if not imp_df.empty:
        fig_fi = go.Figure(go.Bar(
            x=imp_df["importance"], y=imp_df["feature"], orientation="h",
            marker=dict(color=imp_df["importance"], colorscale=CHART_BRAND_SCALE, showscale=False),
            text=[f"{v:.4f}" for v in imp_df["importance"]],
            textposition="outside", textfont=dict(size=9, color=TEXT_MUTED),
        ))
        fig_fi.update_layout(
            height=420, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed", gridcolor=CHART_GRID),
            xaxis=dict(gridcolor=CHART_GRID),
            margin=dict(l=10,r=70,t=10,b=10),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # SHAP
    if HAS_SHAP:
        st.divider()
        st.markdown("<div class='sec-head'><span class='sec-bar'></span>SHAP Importance</div>", unsafe_allow_html=True)
        shap_h  = st.selectbox("Horizon", list(HORIZONS.keys()), index=1, key="shap_h")
        shap_df = predictor.shap_importance(df, market_ctx, horizon=shap_h)
        if not shap_df.empty:
            fig_shap = go.Figure(go.Bar(
                x=shap_df["shap_importance"], y=shap_df["feature"], orientation="h",
                marker=dict(color=shap_df["shap_importance"], colorscale=CHART_BRAND_SCALE, showscale=False),
                text=[f"{v:.4f}" for v in shap_df["shap_importance"]],
                textposition="outside", textfont=dict(size=9, color=TEXT_MUTED),
            ))
            fig_shap.update_layout(
                height=420, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed", gridcolor=CHART_GRID),
                xaxis=dict(gridcolor=CHART_GRID),
                margin=dict(l=10,r=70,t=10,b=10),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("Enable SHAP Selection in sidebar for detailed SHAP values.")

    # Regime signals
    st.divider()
    st.markdown("<div class='sec-head'><span class='sec-bar'></span>Regime Signals</div>", unsafe_allow_html=True)
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
    f"<div style='text-align:center;padding:40px 0 16px;border-top:1px solid {BORDER};margin-top:40px'>"
    f"<div style='display:inline-flex;align-items:center;gap:6px;margin-bottom:8px'>"
    f"<div style='width:20px;height:20px;background:{BRAND_GRAD};border-radius:5px;"
    f"display:flex;align-items:center;justify-content:center;font-size:0.6rem;color:white;font-weight:900'>⬡</div>"
    f"<span style='color:{TEXT_MUTED};font-size:0.72rem;font-weight:700;letter-spacing:0.3px'>Prediqt</span>"
    f"</div>"
    f"<div style='color:{TEXT_MUTED};font-size:0.62rem;line-height:1.6;max-width:400px;margin:0 auto'>"
    f"Prediqt is for informational and educational purposes only. Not financial advice. Past performance does not guarantee future results."
    f"</div>"
    f"</div>",
    unsafe_allow_html=True)
