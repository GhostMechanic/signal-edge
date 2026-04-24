"""
predictions_page.py — Predictions full analytics dashboard page.
Extracted from app.py tab_predictions block.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

from prediction_logger_v2 import (
    migrate_from_v1, deduplicate_log, score_all_intervals,
    get_full_analytics, get_current_model_version,
    get_feature_importance_ranking,
)
from learning_engine import analyze_scored_predictions, get_learning_summary
from auto_retrain import should_retrain_now, execute_retrain, get_improvement_metrics
from model_improvement import ModelImprover
from data_fetcher import (
    fetch_market_context, fetch_fundamentals, fetch_earnings_data,
    fetch_options_data,
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
PURPLE         = "#1e90ff"
BRAND_GRAD     = "linear-gradient(135deg, #2080e5 0%, #06d6a0 100%)"
# Chart palette — brand-first, convention-safe
CHART_UP        = CYAN
CHART_UP_FILL   = "rgba(6,214,160,0.22)"
CHART_DOWN      = "#e04a4a"
CHART_DOWN_FILL = "rgba(224,74,74,0.22)"
CHART_NEUTRAL   = BLUE
CHART_GRID      = "rgba(255,255,255,0.04)"


def info_icon(tooltip_text: str) -> str:
    """Creates an info icon with hover tooltip."""
    safe = (tooltip_text
            .replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))
    return f"<span class='tooltip-icon' data-tip=\"{safe}\">ℹ</span>"


def render_predictions_page(symbol: str, df):
    """Render the predictions / track record page.

    Parameters
    ----------
    symbol : str
        The currently selected stock ticker (needed for retrain context).
    df : pd.DataFrame
        The price DataFrame for *symbol* (needed for retrain context).
    """
    # Migrate v1 predictions on first load
    try:
        migrate_from_v1()
    except Exception:
        pass

    # One-time deduplication (runs once per session)
    if "dedup_done" not in st.session_state:
        try:
            dedup_result = deduplicate_log()
            if dedup_result["removed"] > 0:
                st.toast(f"Cleaned {dedup_result['removed']} duplicate predictions ({dedup_result['original']} → {dedup_result['cleaned']})")
            st.session_state.dedup_done = True
        except Exception:
            st.session_state.dedup_done = True

    # Score any pending predictions
    try:
        score_result = score_all_intervals()
        if score_result.get("error_details"):
            st.warning(f"Scoring issues ({score_result['errors']} errors): {'; '.join(score_result['error_details'][:5])}")
    except Exception as _score_exc:
        score_result = {"scored": 0}
        st.error(f"Scoring failed: {_score_exc}")

    # Update learning engine with latest scored data
    try:
        _learn_result = analyze_scored_predictions()
        _learn_summary = get_learning_summary()
    except Exception:
        _learn_result = {}
        _learn_summary = {}

    # ── Auto-retrain: run automatically when conditions are met ──────────
    if "auto_retrain_done" not in st.session_state:
        try:
            _should_rt, _rt_reason = should_retrain_now()
            if _should_rt:
                _rt_mkt = fetch_market_context()
                _rt_fund = fetch_fundamentals(symbol)
                _rt_earn = fetch_earnings_data(symbol)
                _rt_price = float(df["Close"].iloc[-1])
                _rt_opts = fetch_options_data(symbol, _rt_price)
                _rt_result = execute_retrain(df, _rt_mkt, _rt_fund, _rt_earn, _rt_opts)
                st.session_state.auto_retrain_done = True
                st.session_state.last_retrain_result = _rt_result
                st.toast(f"Model auto-retrained to v{_rt_result.get('new_model_version', '?')} — {_rt_reason}")
            else:
                st.session_state.auto_retrain_done = False
        except Exception:
            st.session_state.auto_retrain_done = False

    # Get full analytics
    try:
        pred_analytics = get_full_analytics()
    except Exception:
        pred_analytics = {"total_predictions": 0, "scored_any": 0, "predictions_table": []}

    # ── Section Header ──
    st.markdown(
        f"<div style='text-align:center;margin-bottom:32px'>"
        f"<div style='color:{TEXT_SECONDARY};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px'>Prediqt IQ</div>"
        f"<h2 style='color:{TEXT_PRIMARY};font-size:1.6rem;font-weight:700;margin:0'>Track Record & Model Learning</h2>"
        f"<div style='color:{TEXT_MUTED};font-size:0.8rem;margin-top:8px'>Every prediction is logged, scored, and used to improve the model.</div>"
        f"</div>", unsafe_allow_html=True)

    # ── Summary Cards ──
    pc1, pc2, pc3, pc4, pc5 = st.columns(5)

    total_preds = pred_analytics.get("total_predictions", 0)
    scored_any = pred_analytics.get("scored_any", 0)
    scored_quick = pred_analytics.get("scored_quick", 0)
    dir_correct_any = pred_analytics.get("direction_correct_any", 0)
    quick_acc = pred_analytics.get("quick_accuracy", 0)
    live_acc = pred_analytics.get("live_accuracy", 0)

    best_acc = max(live_acc, quick_acc) if scored_any > 0 else 0
    acc_color = GREEN if best_acc >= 55 else AMBER if best_acc >= 50 else RED if best_acc > 0 else TEXT_MUTED

    with pc1:
        st.markdown(
            f"<div class='stat'>"
            f"<div class='stat-label'>Total Predictions</div>"
            f"<div class='stat-value'>{total_preds}</div>"
            f"</div>", unsafe_allow_html=True)
    with pc2:
        st.markdown(
            f"<div class='stat'>"
            f"<div class='stat-label'>Scored</div>"
            f"<div class='stat-value'>{scored_any}</div>"
            f"<div class='stat-sub'>of {total_preds}</div>"
            f"</div>", unsafe_allow_html=True)
    with pc3:
        st.markdown(
            f"<div class='stat'>"
            f"<div class='stat-label'>Quick Check (1d)</div>"
            f"<div class='stat-value' style='color:{acc_color}'>{quick_acc:.1f}%</div>"
            f"<div class='stat-sub'>{scored_quick} scored</div>"
            f"</div>", unsafe_allow_html=True)
    with pc4:
        st.markdown(
            f"<div class='stat'>"
            f"<div class='stat-label'>Live Accuracy</div>"
            f"<div class='stat-value' style='color:{acc_color}'>{live_acc:.1f}%</div>"
            f"<div class='stat-sub'>final scores</div>"
            f"</div>", unsafe_allow_html=True)
    with pc5:
        mv = get_current_model_version()
        st.markdown(
            f"<div class='stat'>"
            f"<div class='stat-label'>Model Version</div>"
            f"<div class='stat-value' style='color:{BLUE}'>v{mv}</div>"
            f"<div class='stat-sub'>auto-improves</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    # ── Learning Engine Status ──
    if _learn_summary and _learn_summary.get("stocks_tracked", 0) > 0:
        _ls = _learn_summary
        _le_stocks = _ls.get("stocks_tracked", 0)
        _le_scored = _ls.get("total_scored", 0)
        _le_suppressed = _ls.get("predictions_suppressed", 0)
        _le_avg_adj = _ls.get("avg_confidence_adjustment", 0)
        _adj_color = GREEN if _le_avg_adj > 0 else RED if _le_avg_adj < 0 else TEXT_MUTED
        st.markdown(
            f"<div class='card' style='margin-bottom:24px'>"
            f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:12px'>⬡ Prediqt IQ · Self-Improving</div>"
            f"<div style='display:flex;gap:24px;flex-wrap:wrap'>"
            f"<div style='text-align:center;flex:1'><div style='color:{TEXT_MUTED};font-size:0.65rem'>Stocks Tracked</div><div style='color:{BLUE};font-size:1.3rem;font-weight:800'>{_le_stocks}</div></div>"
            f"<div style='text-align:center;flex:1'><div style='color:{TEXT_MUTED};font-size:0.65rem'>Scored Predictions</div><div style='color:{TEXT_PRIMARY};font-size:1.3rem;font-weight:800'>{_le_scored}</div></div>"
            f"<div style='text-align:center;flex:1'><div style='color:{TEXT_MUTED};font-size:0.65rem'>Signals Suppressed</div><div style='color:{AMBER};font-size:1.3rem;font-weight:800'>{_le_suppressed}</div></div>"
            f"<div style='text-align:center;flex:1'><div style='color:{TEXT_MUTED};font-size:0.65rem'>Avg Confidence Adj</div><div style='color:{_adj_color};font-size:1.3rem;font-weight:800'>{_le_avg_adj:+.1f}%</div></div>"
            f"</div>"
            f"</div>", unsafe_allow_html=True)

    # ── Backtest Accuracy (Instant Feedback) ──
    bt_acc = st.session_state.results.get("backtest_accuracy", {})
    if bt_acc:
        st.markdown(
            f"<div class='card' style='margin-bottom:24px'>"
            f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:4px;overflow:visible'>Instant Backtest Accuracy {info_icon('SYNTHETIC TEST — not real predictions. We take 25 random historical dates, check what direction the model currently predicts vs. what actually happened. This runs every time you click Run Analysis, using the model you just trained. It is NOT the same as live prediction tracking below. Numbers here can differ from Per-Horizon Performance because that section tracks real predictions over time.')}</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.58rem;margin-bottom:12px;font-style:italic'>Simulated on {symbol} historical data — not real predictions</div>"
            f"<div style='display:flex;gap:16px;flex-wrap:wrap'>",
            unsafe_allow_html=True)

        bt_cols = st.columns(len(bt_acc))
        for i, (horizon, bdata) in enumerate(bt_acc.items()):
            bt_color = GREEN if bdata["accuracy"] >= 55 else AMBER if bdata["accuracy"] >= 50 else RED
            with bt_cols[i]:
                st.markdown(
                    f"<div style='text-align:center'>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.7rem;font-weight:600'>{horizon}</div>"
                    f"<div style='color:{bt_color};font-size:1.5rem;font-weight:800'>{bdata['accuracy']:.0f}%</div>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.65rem'>{bdata['correct']}/{bdata['total']} correct</div>"
                    f"</div>", unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    # ── Per-Horizon Performance Chart ──
    ph = pred_analytics.get("per_horizon", {})
    if ph:
        st.markdown(
            f"<div style='color:{TEXT_MUTED};font-size:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:4px;overflow:visible'>Live Prediction Accuracy by Horizon {info_icon('REAL PREDICTIONS — tracked over time. Every time you run analysis, predictions are logged. Once enough time passes (3 days, 1 week, etc.), we check the actual price and grade the prediction. This is the honest scorecard. Numbers may differ from Instant Backtest above because this reflects actual live performance, not a simulation.')}</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.58rem;margin-bottom:12px;font-style:italic'>Graded from real predictions across all stocks</div>",
            unsafe_allow_html=True)

        h_cols = st.columns(len(ph))
        for i, (horizon, hdata) in enumerate(ph.items()):
            with h_cols[i]:
                h_acc = hdata.get("accuracy", 0) or hdata.get("quick_accuracy", 0)
                h_color = GREEN if h_acc >= 55 else AMBER if h_acc >= 50 else RED if h_acc > 0 else TEXT_MUTED
                scored_h = hdata.get("scored", 0) + hdata.get("quick_scored", 0)
                st.markdown(
                    f"<div class='card-sm' style='text-align:center'>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.65rem;font-weight:600;text-transform:uppercase'>{horizon}</div>"
                    f"<div style='color:{h_color};font-size:1.4rem;font-weight:800;margin:8px 0'>{h_acc:.0f}%</div>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.6rem'>{scored_h} scored · avg conf {hdata.get('avg_confidence',0):.0f}%</div>"
                    f"</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    # ── Confidence Calibration ──
    cal_data = pred_analytics.get("confidence_calibration", [])
    if cal_data:
        st.markdown(
            f"<div class='sec-head'><span class='sec-bar'></span>Confidence Calibration {info_icon('Compares predicted confidence to actual accuracy. Perfect calibration = 65% confidence means 65% accuracy.')}</div>",
            unsafe_allow_html=True)

        fig_cal = go.Figure()
        # Perfect calibration line
        fig_cal.add_trace(go.Scatter(
            x=[30, 95], y=[30, 95],
            mode='lines', name='Perfect Calibration',
            line=dict(color=TEXT_MUTED, dash='dash', width=1),
        ))
        # Actual calibration points
        fig_cal.add_trace(go.Scatter(
            x=[c["predicted_confidence"] for c in cal_data],
            y=[c["actual_accuracy"] for c in cal_data],
            mode='markers+lines', name='Actual',
            marker=dict(size=10, color=BLUE),
            line=dict(color=BLUE, width=2),
            text=[f"{c['confidence_range']} ({c['count']} preds)" for c in cal_data],
        ))
        fig_cal.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Predicted Confidence %", gridcolor=CHART_GRID),
            yaxis=dict(title="Actual Accuracy %", gridcolor=CHART_GRID),
            height=350, margin=dict(t=20, b=40, l=50, r=20),
            font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=11),
            showlegend=True,
            legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    # ── Accuracy Over Time ──
    acc_time = pred_analytics.get("accuracy_over_time", [])
    if len(acc_time) >= 3:
        st.markdown(
            f"<div class='sec-head'><span class='sec-bar'></span>Accuracy Trend {info_icon('Rolling 10-prediction accuracy over time. Rising = model improving.')}</div>",
            unsafe_allow_html=True)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=[t["index"] for t in acc_time],
            y=[t["rolling_accuracy"] for t in acc_time],
            mode='lines+markers', name='Rolling Accuracy',
            line=dict(color=CHART_UP, width=2),
            marker=dict(size=4),
            fill='tozeroy', fillcolor='rgba(6,214,160,0.08)',
        ))
        fig_trend.add_hline(y=50, line_dash="dash", line_color=CHART_DOWN, annotation_text="Coin Flip")
        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Prediction #", gridcolor=CHART_GRID),
            yaxis=dict(title="Rolling Accuracy %", gridcolor=CHART_GRID),
            height=300, margin=dict(t=20, b=40, l=50, r=20),
            font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=11),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── Retraining Status (auto-retrain runs above; this just shows status) ──
    try:
        retrain_metrics = get_improvement_metrics()
        _last_rt = st.session_state.get("last_retrain_result")

        if _last_rt:
            # Show the auto-retrain result from this session
            ret_col1, ret_col2, ret_col3 = st.columns([1.5, 1.5, 1])
            with ret_col1:
                st.markdown(
                    f"<div style='background:{GREEN_DIM};border:1.5px solid {GREEN};border-radius:10px;padding:16px;margin-bottom:16px'>"
                    f"<div style='color:{GREEN};font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>✓ Auto-Retrained This Session</div>"
                    f"<div style='color:{TEXT_PRIMARY};font-size:0.95rem;line-height:1.5'>"
                    f"Before: {_last_rt['before_metrics']['accuracy']:.1f}% → Expected: {_last_rt['after_metrics']['expected_accuracy']:.1f}%</div>"
                    f"</div>", unsafe_allow_html=True)
            with ret_col2:
                adj = _last_rt.get("adjustments_applied", {})
                st.markdown(
                    f"<div style='background:{BLUE}20;border:1.5px solid {BLUE};border-radius:10px;padding:16px;margin-bottom:16px'>"
                    f"<div style='color:{BLUE};font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Adjustments Applied</div>"
                    f"<div style='color:{TEXT_PRIMARY};font-size:0.85rem'>"
                    f"{adj.get('feature_boosts_count', 0)} boosts · {adj.get('feature_penalties_count', 0)} penalties · "
                    f"{adj.get('horizon_adjustments_count', 0)} horizon adj</div>"
                    f"</div>", unsafe_allow_html=True)
            with ret_col3:
                st.markdown(
                    f"<div style='background:{BORDER};border-radius:10px;padding:16px;margin-bottom:16px'>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.7rem;text-transform:uppercase;font-weight:600;margin-bottom:8px'>Model Version</div>"
                    f"<div style='color:{TEXT_PRIMARY};font-size:1.6rem;font-weight:800'>v{_last_rt.get('new_model_version', '?')}</div>"
                    f"</div>", unsafe_allow_html=True)

            # Manual retrain button (in case user wants to force it again)
            if st.button("🔄 Force Retrain Again", key="force_retrain_btn"):
                with st.spinner("Retraining…"):
                    try:
                        _rt_mkt = fetch_market_context()
                        _rt_fund = fetch_fundamentals(symbol)
                        _rt_earn = fetch_earnings_data(symbol)
                        _rt_price = float(df["Close"].iloc[-1])
                        _rt_opts = fetch_options_data(symbol, _rt_price)
                        _rt_res = execute_retrain(df, _rt_mkt, _rt_fund, _rt_earn, _rt_opts)
                        st.session_state.last_retrain_result = _rt_res
                        st.success(f"Retrained to v{_rt_res.get('new_model_version', '?')}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Retrain failed: {e}")
        else:
            # No retrain happened this session — show countdown
            _should_rt2, _ = should_retrain_now()
            if not _should_rt2:
                counter = max(0, 20 - (scored_any - (retrain_metrics.get('retrain_count', 0) * 20)))
                st.markdown(
                    f"<div style='background:{BORDER};border:1px solid {BORDER_ACCENT};border-radius:10px;padding:16px;margin-bottom:16px'>"
                    f"<div style='color:{TEXT_SECONDARY};font-size:0.85rem'>"
                    f"Model auto-retrains when enough new scored predictions accumulate. "
                    f"<b>{counter} more</b> until next retrain. Current version: v{retrain_metrics.get('current_version', '1.0')}</div>"
                    f"</div>", unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown(f"<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    # ── Model Improvement Summary ──
    try:
        improver = ModelImprover()
        improvement = improver.get_improvement_summary()

        imp_col1, imp_col2 = st.columns(2)

        with imp_col1:
            st.markdown(
                f"<div class='card'>"
                "<div class='sec-head'><span class='sec-bar'></span>Model Learning Status</div>"
                f"<div class='mrow'><span class='mrow-label'>Version</span><span class='mrow-value' style='color:{BLUE}'>v{improvement['model_version']}</span></div>"
                f"<div class='mrow'><span class='mrow-label'>Trend</span><span class='mrow-value'>{improvement['improvement_trend']}</span></div>"
                f"<div class='mrow'><span class='mrow-label'>Calibration</span><span class='mrow-value'>{improvement['confidence_calibration']}</span></div>"
                f"<div class='mrow'><span class='mrow-label'>Next Retrain</span><span class='mrow-value'>{improvement['next_retrain_in']} predictions away</span></div>"
                f"<div class='mrow'><span class='mrow-label'>Best Horizon</span><span class='mrow-value' style='color:{GREEN}'>{improvement.get('best_horizon', 'N/A')}</span></div>"
                f"<div class='mrow'><span class='mrow-label'>Worst Horizon</span><span class='mrow-value' style='color:{AMBER}'>{improvement.get('worst_horizon', 'N/A')}</span></div>"
                f"</div>", unsafe_allow_html=True)

        with imp_col2:
            # Feature importance ranking
            feat_ranking = get_feature_importance_ranking()
            st.markdown(
                f"<div class='card'>"
                f"<div class='sec-head'><span class='sec-bar'></span>Feature Learning {info_icon('Features ranked by cumulative predictive power. Positive = helps predictions, Negative = hurts.')}</div>",
                unsafe_allow_html=True)

            if feat_ranking:
                for feat in feat_ranking[:8]:
                    score = feat["cumulative_score"]
                    f_color = GREEN if score > 0 else RED
                    acc_str = f"{feat['accuracy_when_top']:.0f}%"
                    st.markdown(
                        f"<div class='mrow'>"
                        f"<span class='mrow-label'>{feat['feature'][:25]}</span>"
                        f"<span style='color:{f_color};font-size:0.75rem;font-weight:700'>{score:+.3f} · {acc_str}</span>"
                        f"</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.75rem'>Run more analyses to build feature importance data.</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
    except Exception:
        pass

    # ── Detailed Prediction History Table ──
    st.markdown(f"<div style='margin-top:32px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sec-head'><span class='sec-bar'></span>Prediction History</div>",
        unsafe_allow_html=True)

    pred_table = pred_analytics.get("predictions_table", [])
    if pred_table:
        display_df = pd.DataFrame(pred_table)

        # ── Filters ───────────────────────────────────────────────────────
        filt_col1, filt_col2, filt_col3, filt_col4 = st.columns(4)

        with filt_col1:
            all_symbols = sorted(display_df["symbol"].unique().tolist()) if "symbol" in display_df.columns else []
            sel_symbols = st.multiselect("Symbol", all_symbols, default=[], key="pred_filt_sym",
                                        placeholder="All symbols")
        with filt_col2:
            all_horizons = sorted(display_df["horizon"].unique().tolist()) if "horizon" in display_df.columns else []
            sel_horizons = st.multiselect("Horizon", all_horizons, default=[], key="pred_filt_hz",
                                         placeholder="All horizons")
        with filt_col3:
            status_options = ["All", "Scored", "Pending"]
            sel_status = st.selectbox("Status", status_options, key="pred_filt_status")
        with filt_col4:
            dir_options = ["All", "Bullish", "Bearish"]
            sel_dir = st.selectbox("Direction", dir_options, key="pred_filt_dir")

        # Apply filters
        filtered_df = display_df.copy()
        if sel_symbols:
            filtered_df = filtered_df[filtered_df["symbol"].isin(sel_symbols)]
        if sel_horizons:
            filtered_df = filtered_df[filtered_df["horizon"].isin(sel_horizons)]
        if sel_status == "Scored" and "final_result" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["final_result"] != "pending"]
        elif sel_status == "Pending" and "final_result" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["final_result"] == "pending"]
        if sel_dir == "Bullish" and "direction" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["direction"].str.upper() == "BULLISH"]
        elif sel_dir == "Bearish" and "direction" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["direction"].str.upper() == "BEARISH"]

        # Select most useful columns for display
        display_cols = ["symbol", "date", "horizon", "direction", "predicted_return",
                        "confidence", "regime", "model_version"]

        # Add score columns that have data
        for interval in ["1d", "3d", "7d", "30d", "90d"]:
            col = f"score_{interval}"
            if col in filtered_df.columns and filtered_df[col].ne("—").any():
                display_cols.append(col)

        if "final_result" in filtered_df.columns:
            display_cols.append("final_result")

        available_cols = [c for c in display_cols if c in filtered_df.columns]
        show_df = filtered_df[available_cols].copy()

        # Rename columns for display
        rename_map = {
            "symbol": "Symbol", "date": "Date", "horizon": "Horizon",
            "direction": "Dir", "predicted_return": "Pred %",
            "confidence": "Conf %", "regime": "Regime",
            "model_version": "Model",
            "score_1d": "1-Day", "score_3d": "3-Day",
            "score_7d": "7-Day", "score_30d": "30-Day",
            "score_90d": "90-Day", "final_result": "Final",
        }
        show_df = show_df.rename(columns={k: v for k, v in rename_map.items() if k in show_df.columns})

        if "Pred %" in show_df.columns:
            show_df["Pred %"] = show_df["Pred %"].apply(lambda x: f"{x:+.1f}%" if isinstance(x, (int, float)) else x)

        st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.7rem;margin-bottom:4px'>"
                    f"Showing {len(show_df)} of {len(pred_table)} predictions</div>",
                    unsafe_allow_html=True)
        st.dataframe(show_df, use_container_width=True, hide_index=True, height=400)

        # Export buttons
        exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])
        with exp_col1:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Export CSV",
                csv_data,
                file_name=f"prediqt_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        with exp_col2:
            st.markdown(
                f"<div style='color:{TEXT_MUTED};font-size:0.7rem;padding-top:8px'>"
                f"{len(pred_table)} predictions total</div>",
                unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='card' style='text-align:center;padding:40px'>"
            f"<div style='color:{TEXT_MUTED};font-size:1.2rem;margin-bottom:8px'>No predictions yet</div>"
            f"<div style='color:{TEXT_MUTED};font-size:0.8rem'>Run an analysis to start logging predictions.</div>"
            f"</div>", unsafe_allow_html=True)

    # ── Per-Symbol Accuracy ──
    per_sym = pred_analytics.get("per_symbol", {})
    if per_sym:
        st.markdown(f"<div style='margin-top:24px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sec-head'><span class='sec-bar'></span>Per-Symbol Performance</div>",
            unsafe_allow_html=True)

        sym_cols = st.columns(min(len(per_sym), 6))
        for i, (sym, sdata) in enumerate(per_sym.items()):
            with sym_cols[i % len(sym_cols)]:
                s_acc = sdata.get("accuracy", 0)
                s_color = GREEN if s_acc >= 55 else AMBER if s_acc >= 50 else RED if s_acc > 0 else TEXT_MUTED
                st.markdown(
                    f"<div class='card-sm' style='text-align:center;margin-bottom:8px'>"
                    f"<div style='color:{TEXT_PRIMARY};font-weight:700;font-size:0.85rem'>{sym}</div>"
                    f"<div style='color:{s_color};font-size:1.2rem;font-weight:800'>{s_acc:.0f}%</div>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.6rem'>{sdata['scored']} scored</div>"
                    f"</div>", unsafe_allow_html=True)

    # ── Regime Accuracy ──
    reg_acc = pred_analytics.get("regime_accuracy", {})
    if reg_acc and any(r["scored"] > 0 for r in reg_acc.values()):
        st.markdown(f"<div style='margin-top:24px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sec-head'><span class='sec-bar'></span>Accuracy by Market Regime</div>",
            unsafe_allow_html=True)

        reg_cols = st.columns(len(reg_acc))
        for i, (regime, rdata) in enumerate(reg_acc.items()):
            r_acc = rdata.get("accuracy", 0)
            regime_c = {"Bull": GREEN, "Bear": RED, "Sideways": AMBER}.get(regime, TEXT_MUTED)
            with reg_cols[i]:
                st.markdown(
                    f"<div class='card-sm' style='text-align:center;overflow:visible !important;position:relative;min-width:80px'>"
                    f"<div style='color:{regime_c};font-weight:700;font-size:0.85rem'>{regime}</div>"
                    f"<div style='color:{TEXT_PRIMARY};font-size:1.4rem;font-weight:800;margin:6px 0'>{r_acc:.0f}%</div>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.6rem'>{rdata['scored']} scored of {rdata['total']}</div>"
                    f"</div>", unsafe_allow_html=True)
