"""
track_record_page.py
────────────────────────────────────────────────────────────────────
Prediqt Track Record — public-facing performance scoreboard.
Designed to build subscriber trust with transparent model performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from prediction_logger_v2 import (
    get_full_analytics, get_current_model_version,
    export_predictions_dataframe, repair_missing_final_scores,
)
from target_hit_analyzer import (
    enrich_predictions_with_target_hit,
    compute_target_hit_aggregates,
    HORIZON_DAYS,
)

# ─── Design tokens (mirror app.py's brand palette — keep in sync) ────────────
BG_PRIMARY    = "#030508"
BG_SURFACE    = "#0a0e14"
BG_CARD       = "#0f1319"
BG_ELEVATED   = "#141a22"
BORDER        = "#1a2233"
BORDER_ACCENT = "#1e3050"
TEXT_PRIMARY   = "#f0f4fa"
TEXT_SECONDARY = "#8898aa"
TEXT_MUTED     = "#4a5a6e"
AMBER          = "#f59e0b"
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
# Directional colors — aliases to brand chart palette so every up/down usage
# in this page renders in brand cyan/muted-red instead of raw green/red.
GREEN          = CHART_UP          # was "#10b981"
RED            = CHART_DOWN        # was "#ef4444"
GREEN_DIM      = "rgba(6,214,160,0.12)"   # brand cyan tint (was "#064e3b")
RED_DIM        = "rgba(224,74,74,0.12)"   # brand red tint (was "#450a0a")
AMBER_DIM      = "rgba(245,158,11,0.12)"  # amber tint (new)


def render_track_record():
    """Render the full Track Record page."""

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        "<div class='page-header'>"
        "    <div class='ph-eyebrow'>Performance · Public Ledger</div>"
        "    <h1>Real predictions.<br><span class='grad'>Real receipts.</span></h1>"
        "    <p class='ph-sub'>Every call the model has made — scored, timestamped, "
        "    and open for inspection. No cherry-picking, no hiding the misses.</p>"
        "</div>",
        unsafe_allow_html=True)

    # ── Repair any gaps in final_scored flags before loading analytics ─────
    # Cached so we only actually re-scan once per hour. This catches the edge
    # case where a prediction's scheduled final checkpoint fetch failed
    # (weekend, transient yfinance outage) and final_scored never flipped.
    @st.cache_data(ttl=3600, show_spinner=False)
    def _run_final_score_repair():
        try:
            return repair_missing_final_scores()
        except Exception:
            return None

    _repair_result = _run_final_score_repair()
    # We don't surface the repair output in the UI — it just quietly fills
    # the gaps. If a user really wants to see it, it's in the return value.

    # ── Load analytics ───────────────────────────────────────────────────────
    try:
        pa = get_full_analytics()
        mv = get_current_model_version()
    except Exception as e:
        st.error(f"Unable to load analytics: {e}")
        return

    # ── Enrich predictions with full-daily target-hit scan ─────────────────
    # This walks the daily OHLC for each symbol and flags target_hit + peak
    # favorable move for every prediction. Cached per-symbol so subsequent
    # loads are near-instant.
    try:
        with st.spinner("Scanning daily prices for target-hit results…",
                        show_time=False):
            pa["predictions_table"] = enrich_predictions_with_target_hit(
                pa.get("predictions_table", []))
        pa.update(compute_target_hit_aggregates(pa["predictions_table"]))
    except Exception as e:
        # Non-fatal: page still renders without the new columns
        st.warning(f"Target-hit scan unavailable: {e}")
        pa.setdefault("target_hit_rate", 0)
        pa.setdefault("target_hit_count", 0)
        pa.setdefault("target_definitive", 0)
        pa.setdefault("target_per_horizon", {})

    total_p = pa.get("total_predictions", 0)
    scored_any = pa.get("scored_any", 0)
    # live_accuracy = full-horizon-matured-only number (Expiration Win Rate)
    # quick_accuracy = 1-day-out direction (kept for the per-horizon "Estimating" fallback)
    live_acc = pa.get("live_accuracy", 0)
    quick_acc = pa.get("quick_accuracy", 0)

    if total_p == 0:
        st.markdown(
            f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});border:1px solid {BORDER};"
            f"border-radius:14px;padding:60px 30px;text-align:center'>"
            f"<div style='font-size:2rem;margin-bottom:16px'>◎</div>"
            f"<div style='color:{TEXT_PRIMARY};font-size:1.1rem;font-weight:700;margin-bottom:8px'>"
            f"    No Predictions Yet</div>"
            f"<div style='color:{TEXT_SECONDARY};font-size:0.82rem;max-width:360px;margin:0 auto;line-height:1.6'>"
            f"    Run your first analysis to start building the track record.</div>"
            f"</div>",
            unsafe_allow_html=True)
        return

    preds_table = pa.get("predictions_table", [])
    win_count = pa.get("direction_correct_any", 0)
    loss_count = scored_any - win_count
    win_rate = (win_count / scored_any * 100) if scored_any > 0 else 0
    pending_count = total_p - scored_any

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: THE SCOREBOARD
    # One glance: "Is this model actually good?"
    # The three honesty bars below (Target Hit / Checkpoint / Expiration) carry
    # the full story — no big cherry-picked headline number up top.
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown(
        f"<div style='background:linear-gradient(145deg, {BG_CARD}, {BG_SURFACE});border:1px solid {BORDER};"
        f"            border-radius:16px;padding:28px 32px;margin-bottom:16px;"
        f"            position:relative;overflow:hidden'>"
        # Brand hairline — matches other cards across the app
        f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
        f"background:{BRAND_GRAD};opacity:0.75'></div>"
        # Top row: scoreboard title + W/L/Pending pills
        f"<div style='display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:16px'>"
        f"<div>"
        f"<div style='color:{CYAN};font-size:0.58rem;font-weight:700;text-transform:uppercase;"
        f"            letter-spacing:0.14em;margin-bottom:8px'>Scoreboard · v{mv}</div>"
        f"<div style='font-family:\"Inter\",sans-serif;color:{TEXT_PRIMARY};"
        f"font-size:1.4rem;font-weight:800;letter-spacing:-0.02em;line-height:1.1'>"
        f"How the model is doing.</div>"
        f"<div style='color:{TEXT_MUTED};font-size:0.7rem;font-weight:500;"
        f"margin-top:6px;line-height:1.5;max-width:440px'>"
        f"Three reads on the same track record — from the <b style='color:{CYAN}'>"
        f"generous</b> (did it ever work?) to the <b style='color:{AMBER}'>strict</b> "
        f"(was it still right at expiration?). Pick whichever honesty level you care about.</div>"
        f"</div>"
        # W / L / Pending compact
        f"<div style='display:flex;gap:10px'>"
        f"<div style='text-align:center;padding:10px 16px;background:{GREEN_DIM};"
        f"            border:1px solid rgba(6,214,160,0.32);border-radius:8px'>"
        f"<div style='color:{GREEN};font-size:1.4rem;font-weight:900;line-height:1;"
        f"font-variant-numeric:tabular-nums'>{win_count}</div>"
        f"<div style='color:{GREEN};font-size:0.5rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.12em;margin-top:3px'>Wins</div>"
        f"</div>"
        f"<div style='text-align:center;padding:10px 16px;background:{RED_DIM};"
        f"            border:1px solid rgba(224,74,74,0.32);border-radius:8px'>"
        f"<div style='color:{RED};font-size:1.4rem;font-weight:900;line-height:1;"
        f"font-variant-numeric:tabular-nums'>{loss_count}</div>"
        f"<div style='color:{RED};font-size:0.5rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.12em;margin-top:3px'>Losses</div>"
        f"</div>"
        f"<div style='text-align:center;padding:10px 16px;background:{BG_SURFACE};"
        f"            border:1px solid {BORDER};border-radius:8px'>"
        f"<div style='color:{TEXT_SECONDARY};font-size:1.4rem;font-weight:900;line-height:1;"
        f"font-variant-numeric:tabular-nums'>{pending_count}</div>"
        f"<div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.12em;margin-top:3px'>Pending</div>"
        f"</div>"
        f"</div>"
        f"</div>"
        # ── Three metric "stat tiles": Target Hit / Checkpoint Win / Expiration Win ──
        # Each one is a different "honesty read" on the model:
        #   Target Hit Rate  → generous  (did the stock EVER hit target?)
        #   Checkpoint Win   → medium    (right at the latest checkpoint reached)
        #   Expiration Win   → strict    (right at scheduled full horizon — only matured)
        + (lambda: (
            lambda thr, thc, td, cwr, sa, ewr, sf: (
                f"<div style='margin-top:24px;display:grid;"
                f"grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px'>"
                # ── Target Hit Rate tile (CYAN · GENEROUS) ─────────────
                f"<div style='background:{BG_SURFACE};border:1px solid {BORDER};"
                f"border-radius:12px;padding:18px 20px;position:relative;overflow:hidden'>"
                f"<div style='position:absolute;top:0;left:0;right:0;height:2px;"
                f"background:{CYAN};opacity:0.85'></div>"
                # Eyebrow: severity pill
                f"<div style='display:flex;align-items:center;justify-content:space-between;"
                f"margin-bottom:10px;gap:8px'>"
                f"<span style='display:inline-block;padding:2px 8px;border-radius:100px;"
                f"background:rgba(6,214,160,0.12);border:1px solid rgba(6,214,160,0.32);"
                f"color:{CYAN};font-size:0.5rem;font-weight:800;letter-spacing:0.14em;"
                f"text-transform:uppercase'>Generous</span>"
                f"<span style='color:{TEXT_MUTED};font-size:0.56rem;font-weight:600;"
                f"font-variant-numeric:tabular-nums'>{thc} / {td} resolved</span>"
                f"</div>"
                # Big headline number
                f"<div style='font-family:\"Inter\",sans-serif;color:{CYAN};"
                f"font-size:2.8rem;font-weight:900;letter-spacing:-0.04em;line-height:0.95;"
                f"font-variant-numeric:tabular-nums;text-shadow:0 0 40px rgba(6,214,160,0.18);"
                f"margin-bottom:2px'>{thr:.0f}%</div>"
                # Metric label
                f"<div style='color:{TEXT_PRIMARY};font-size:0.72rem;font-weight:800;"
                f"letter-spacing:0.02em;margin-bottom:12px'>Target Hit Rate</div>"
                # Bar
                f"<div style='height:4px;background:{BG_CARD};border-radius:100px;"
                f"overflow:hidden;border:1px solid {BORDER};margin-bottom:12px'>"
                f"<div style='height:100%;width:{thr}%;background:{BRAND_GRAD};"
                f"border-radius:100px'></div>"
                f"</div>"
                # Description
                f"<div style='color:{TEXT_SECONDARY};font-size:0.65rem;line-height:1.55'>"
                f"Stock <b style='color:{TEXT_PRIMARY}'>touched the target price</b> "
                f"at any point during the horizon (daily high/low scan)."
                f"</div>"
                f"</div>"
                # ── Checkpoint Win Rate tile (BLUE · MEDIUM) ───────────
                f"<div style='background:{BG_SURFACE};border:1px solid {BORDER};"
                f"border-radius:12px;padding:18px 20px;position:relative;overflow:hidden'>"
                f"<div style='position:absolute;top:0;left:0;right:0;height:2px;"
                f"background:{BLUE};opacity:0.85'></div>"
                f"<div style='display:flex;align-items:center;justify-content:space-between;"
                f"margin-bottom:10px;gap:8px'>"
                f"<span style='display:inline-block;padding:2px 8px;border-radius:100px;"
                f"background:rgba(32,128,229,0.12);border:1px solid rgba(32,128,229,0.35);"
                f"color:{BLUE};font-size:0.5rem;font-weight:800;letter-spacing:0.14em;"
                f"text-transform:uppercase'>Medium</span>"
                f"<span style='color:{TEXT_MUTED};font-size:0.56rem;font-weight:600;"
                f"font-variant-numeric:tabular-nums'>{sa} scored</span>"
                f"</div>"
                f"<div style='font-family:\"Inter\",sans-serif;color:{BLUE};"
                f"font-size:2.8rem;font-weight:900;letter-spacing:-0.04em;line-height:0.95;"
                f"font-variant-numeric:tabular-nums;text-shadow:0 0 40px rgba(32,128,229,0.18);"
                f"margin-bottom:2px'>{cwr:.0f}%</div>"
                f"<div style='color:{TEXT_PRIMARY};font-size:0.72rem;font-weight:800;"
                f"letter-spacing:0.02em;margin-bottom:12px'>Checkpoint Win Rate</div>"
                f"<div style='height:4px;background:{BG_CARD};border-radius:100px;"
                f"overflow:hidden;border:1px solid {BORDER};margin-bottom:12px'>"
                f"<div style='height:100%;width:{cwr}%;background:{BRAND_GRAD};"
                f"border-radius:100px'></div>"
                f"</div>"
                f"<div style='color:{TEXT_SECONDARY};font-size:0.65rem;line-height:1.55'>"
                f"Direction is <b style='color:{TEXT_PRIMARY}'>still correct at the "
                f"latest checkpoint reached</b> — may be interim for in-flight calls."
                f"</div>"
                f"</div>"
                # ── Expiration Win Rate tile (AMBER · STRICT) ──────────
                f"<div style='background:{BG_SURFACE};border:1px solid {BORDER};"
                f"border-radius:12px;padding:18px 20px;position:relative;overflow:hidden'>"
                f"<div style='position:absolute;top:0;left:0;right:0;height:2px;"
                f"background:{AMBER};opacity:0.85'></div>"
                f"<div style='display:flex;align-items:center;justify-content:space-between;"
                f"margin-bottom:10px;gap:8px'>"
                f"<span style='display:inline-block;padding:2px 8px;border-radius:100px;"
                f"background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.38);"
                f"color:{AMBER};font-size:0.5rem;font-weight:800;letter-spacing:0.14em;"
                f"text-transform:uppercase'>Strict</span>"
                f"<span style='color:{TEXT_MUTED};font-size:0.56rem;font-weight:600;"
                f"font-variant-numeric:tabular-nums'>{sf} matured</span>"
                f"</div>"
                + (
                    f"<div style='font-family:\"Inter\",sans-serif;color:{AMBER};"
                    f"font-size:2.8rem;font-weight:900;letter-spacing:-0.04em;line-height:0.95;"
                    f"font-variant-numeric:tabular-nums;text-shadow:0 0 40px rgba(245,158,11,0.18);"
                    f"margin-bottom:2px'>{ewr:.0f}%</div>"
                    if sf > 0 else
                    f"<div style='font-family:\"Inter\",sans-serif;color:{TEXT_MUTED};"
                    f"font-size:2.8rem;font-weight:900;letter-spacing:-0.04em;line-height:0.95;"
                    f"margin-bottom:2px'>—</div>"
                )
                + f"<div style='color:{TEXT_PRIMARY};font-size:0.72rem;font-weight:800;"
                f"letter-spacing:0.02em;margin-bottom:12px'>Expiration Win Rate</div>"
                f"<div style='height:4px;background:{BG_CARD};border-radius:100px;"
                f"overflow:hidden;border:1px solid {BORDER};margin-bottom:12px'>"
                f"<div style='height:100%;width:{ewr if sf > 0 else 0}%;"
                f"background:{BRAND_GRAD};border-radius:100px'></div>"
                f"</div>"
                f"<div style='color:{TEXT_SECONDARY};font-size:0.65rem;line-height:1.55'>"
                f"Direction correct at the <b style='color:{TEXT_PRIMARY}'>scheduled "
                f"full-horizon date</b> — only predictions that have fully matured."
                f"</div>"
                f"</div>"
                f"</div>"
            )
        )(
            pa.get("target_hit_rate", 0),
            pa.get("target_hit_count", 0),
            pa.get("target_definitive", 0),
            win_rate,
            scored_any,
            pa.get("live_accuracy", 0),
            pa.get("scored_final", 0),
        ))()
        +
        # Subtle clarifier explaining the three reads
        f"<div style='color:{TEXT_MUTED};font-size:0.62rem;margin-top:14px;"
        f"line-height:1.55;max-width:760px'>"
        f"<b style='color:{TEXT_SECONDARY}'>Three honesty reads, one track record.</b> "
        f"<b style='color:{CYAN}'>Target Hit Rate</b> asks whether the stock ever "
        f"touched the target at any point during the horizon. "
        f"<b style='color:{BLUE}'>Checkpoint Win Rate</b> scores every prediction at the "
        f"most recent checkpoint reached — so a 1-year call made 30 days ago is judged "
        f"against its 30-day move. "
        f"<b style='color:{AMBER}'>Expiration Win Rate</b> is the strictest — it only "
        f"counts predictions that have actually reached their scheduled full horizon, "
        f"so a 1-year call made 30 days ago doesn't count yet."
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True)

    # ── Per-horizon breakdown ──────────────────────────────────────────────
    per_horizon = pa.get("per_horizon", {})
    target_per_horizon = pa.get("target_per_horizon", {})

    # Compute Checkpoint Win Rate PER HORIZON inline (backend only tracks
    # quick_accuracy and accuracy_final per horizon — not "latest checkpoint").
    # The latest checkpoint = highest-day interval with a recorded score.
    checkpoint_per_horizon: dict = {}
    for _p in preds_table:
        _hz = _p.get("horizon", "")
        d = checkpoint_per_horizon.setdefault(_hz, {"scored": 0, "correct": 0})
        for _int in ["365d", "180d", "90d", "60d", "30d", "14d", "7d", "3d", "1d"]:
            _s = _p.get(f"score_{_int}")
            if _s in ("✓", "✗"):
                d["scored"] += 1
                if _s == "✓":
                    d["correct"] += 1
                break

    if per_horizon:
        # Clarifying header above the cards — tells users what each tile means
        st.markdown(
            f"<div style='display:flex;align-items:baseline;justify-content:space-between;"
            f"gap:12px;margin:28px 0 10px;flex-wrap:wrap'>"
            f"<div class='sec-head' style='display:flex;align-items:center;gap:10px'>"
            f"<div class='sec-bar' style='width:3px;height:14px;background:{BRAND_GRAD};border-radius:2px'></div>"
            f"<span style='color:{TEXT_PRIMARY};font-size:0.7rem;font-weight:800;"
            f"text-transform:uppercase;letter-spacing:0.14em'>Accuracy by Prediction Horizon</span>"
            f"</div>"
            f"<span style='color:{TEXT_MUTED};font-size:0.68rem;font-weight:500;"
            f"line-height:1.5;max-width:600px'>"
            f"Each tile shows all three honesty reads "
            f"(<b style='color:{CYAN}'>Target Hit</b> · "
            f"<b style='color:{BLUE}'>Checkpoint</b> · "
            f"<b style='color:{AMBER}'>Expiration</b>) "
            f"for predictions made with that time window. "
            f"<b style='color:{TEXT_SECONDARY}'>1 Week</b> is the accuracy of every 1-week "
            f"prediction — not accuracy during the last 7 days."
            f"</span>"
            f"</div>",
            unsafe_allow_html=True)

        HORIZON_ORDER = ["3 Day", "1 Week", "1 Month", "1 Quarter", "1 Year"]
        sorted_horizons = sorted(
            per_horizon.items(),
            key=lambda x: HORIZON_ORDER.index(x[0]) if x[0] in HORIZON_ORDER else len(HORIZON_ORDER),
        )
        h_cols = st.columns(len(sorted_horizons))
        for col, (h_name, h_data) in zip(h_cols, sorted_horizons):
            with col:
                h_total = h_data.get("total", 0)

                # ── Target Hit Rate for this horizon ──
                _tph = target_per_horizon.get(h_name, {})
                _t_hit = _tph.get("hit", 0)
                _t_def = _tph.get("definitive", 0)
                _t_rate = _tph.get("rate", 0)
                _t_has = _t_def > 0

                # ── Checkpoint Win Rate for this horizon ──
                _ch_h = checkpoint_per_horizon.get(h_name, {})
                _ch_scored = _ch_h.get("scored", 0)
                _ch_correct = _ch_h.get("correct", 0)
                _ch_rate = (_ch_correct / _ch_scored * 100) if _ch_scored > 0 else 0
                _ch_has = _ch_scored > 0

                # ── Expiration Win Rate for this horizon ──
                _ex_scored = h_data.get("scored", 0)
                _ex_rate = h_data.get("accuracy", 0)
                _ex_has = _ex_scored > 0

                # Mini-bar renderer — compact row with label, %, and a thin bar
                def _mini_row(label: str, pct: float, has_data: bool,
                              sample_text: str, color: str) -> str:
                    _val_txt = f"{pct:.0f}%" if has_data else "—"
                    _val_clr = color if has_data else TEXT_MUTED
                    _bar_w = pct if has_data else 0
                    return (
                        f"<div style='margin-top:10px'>"
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:baseline;margin-bottom:4px;gap:6px'>"
                        f"<span style='color:{color};font-size:0.48rem;font-weight:800;"
                        f"text-transform:uppercase;letter-spacing:0.12em'>{label}</span>"
                        f"<span style='font-family:\"Inter\",sans-serif;color:{_val_clr};"
                        f"font-size:0.95rem;font-weight:900;letter-spacing:-0.02em;"
                        f"line-height:1;font-variant-numeric:tabular-nums'>{_val_txt}</span>"
                        f"</div>"
                        f"<div style='height:3px;background:{BG_CARD};border-radius:100px;"
                        f"overflow:hidden;border:1px solid {BORDER}'>"
                        f"<div style='height:100%;width:{_bar_w}%;background:{color};"
                        f"border-radius:100px;opacity:{0.9 if has_data else 0}'></div>"
                        f"</div>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.48rem;margin-top:2px;"
                        f"font-variant-numeric:tabular-nums;letter-spacing:0.02em'>"
                        f"{sample_text}</div>"
                        f"</div>"
                    )

                _target_row = _mini_row(
                    "Target Hit", _t_rate, _t_has,
                    f"{_t_hit} / {_t_def} resolved" if _t_has else f"0 / {h_total} resolved",
                    CYAN,
                )
                _check_row = _mini_row(
                    "Checkpoint", _ch_rate, _ch_has,
                    f"{_ch_correct} / {_ch_scored} scored" if _ch_has else f"0 / {h_total} scored",
                    BLUE,
                )
                _expir_row = _mini_row(
                    "Expiration", _ex_rate, _ex_has,
                    f"{int(_ex_rate/100*_ex_scored)} / {_ex_scored} matured" if _ex_has
                    else f"0 / {h_total} matured",
                    AMBER,
                )

                st.markdown(
                    f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
                    f"            border:1px solid {BORDER};"
                    f"            border-radius:12px;padding:14px 14px 16px;"
                    f"            position:relative;overflow:hidden'>"
                    # Brand hairline at the top
                    f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
                    f"background:{BRAND_GRAD};opacity:0.55'></div>"
                    # Horizon name header with total count
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:baseline;margin-bottom:2px'>"
                    f"<span style='color:{TEXT_PRIMARY};font-size:0.68rem;font-weight:800;"
                    f"text-transform:uppercase;letter-spacing:0.14em'>{h_name}</span>"
                    f"<span style='color:{TEXT_MUTED};font-size:0.52rem;font-weight:500;"
                    f"font-variant-numeric:tabular-nums'>{h_total} calls</span>"
                    f"</div>"
                    # The three mini rows
                    f"{_target_row}"
                    f"{_check_row}"
                    f"{_expir_row}"
                    f"</div>",
                    unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: SIMULATED PORTFOLIO
    # "If I'd followed every signal, how much would I have made?"
    # ═══════════════════════════════════════════════════════════════════════════

    scored_preds = [p for p in preds_table if p.get("final_result") in ("✓", "✗")]

    st.markdown(f"<div style='height:20px'></div>", unsafe_allow_html=True)

    # Lookup that the Prediction Log will use to show per-trade P&L
    trade_pnl_lookup = {}

    if len(scored_preds) >= 3:
        from datetime import datetime as _dt_sim, timedelta as _td_sim

        # ── Map scored interval → calendar days so we can compute exit dates ──
        _INTERVAL_DAYS = {
            "1d": 1, "3d": 3, "7d": 7, "14d": 14, "30d": 30,
            "60d": 60, "90d": 90, "180d": 180, "365d": 365,
        }

        # ── Build BOTH trade lists in parallel ────────────────────────
        # 1) HOLD — exit at the scheduled checkpoint (current behavior)
        # 2) TAKE-PROFIT — exit at target_hit_date if target was hit,
        #    otherwise fall back to Hold behavior
        _trades = []      # HOLD
        _trades_tp = []   # TAKE-PROFIT

        for pred in scored_preds:
            actual_ret = None
            scored_interval = None
            for interval in ["365d", "180d", "90d", "60d", "30d", "14d", "7d", "3d", "1d"]:
                r = pred.get(f"return_{interval}")
                if r is not None:
                    actual_ret = r / 100
                    scored_interval = interval
                    break
            if actual_ret is None:
                continue

            try:
                _p_date = _dt_sim.strptime(pred["date"], "%Y-%m-%d")
            except Exception:
                continue
            _exit_date = _p_date + _td_sim(days=_INTERVAL_DAYS[scored_interval])

            # Direction-aware: positive predicted_return → long, negative → short
            _pred_ret = pred.get("predicted_return", 0) or 0
            _dir_sign = 1 if _pred_ret >= 0 else -1
            _hold_pnl_pct = actual_ret * _dir_sign

            _common = {
                "pred_date": pred["date"],
                "symbol": pred["symbol"],
                "horizon": pred["horizon"],
                "direction": "LONG" if _dir_sign > 0 else "SHORT",
                "scored_at": scored_interval,
                "actual_ret": actual_ret,
                "predicted_ret": _pred_ret,
            }

            # Hold portfolio record
            _trades.append({
                **_common,
                "exit_date": _exit_date,
                "pnl_pct": _hold_pnl_pct,
            })

            # Take-profit portfolio record
            _th = pred.get("target_hit")
            _th_date_str = pred.get("target_hit_date")
            if _th is True and _th_date_str:
                try:
                    _tp_exit = _dt_sim.strptime(_th_date_str, "%Y-%m-%d")
                except Exception:
                    _tp_exit = _exit_date
                # Capture the full predicted move as the realized % — target
                # was hit, so the thesis's expected gain is locked in.
                _tp_pnl_pct = abs(_pred_ret / 100.0)
                _tp_realized = "target"
            else:
                # Target never hit (or still pending) — fall back to the
                # Hold outcome so the TP curve is never worse than Hold.
                _tp_exit = _exit_date
                _tp_pnl_pct = _hold_pnl_pct
                _tp_realized = "expiration"

            _trades_tp.append({
                **_common,
                "exit_date": _tp_exit,
                "pnl_pct": _tp_pnl_pct,
                "tp_realized": _tp_realized,
            })

        _trades.sort(key=lambda t: t["exit_date"])
        _trades_tp.sort(key=lambda t: t["exit_date"])

        starting_capital = 10000.0

        # ── Hold curve ──────────────────────────────────────────────
        equity = [starting_capital]
        dates_eq = [_trades[0]["exit_date"] if _trades else _dt_sim.now()]
        for _t in _trades:
            position_size = equity[-1] * 0.05
            pnl = position_size * _t["pnl_pct"]
            equity.append(equity[-1] + pnl)
            dates_eq.append(_t["exit_date"])
            _t["position_size"] = round(position_size, 2)
            _t["pnl_dollar"] = round(pnl, 2)
            _t["equity_after"] = round(equity[-1], 2)
            _key = (_t["pred_date"], _t["symbol"], _t["horizon"])
            trade_pnl_lookup[_key] = _t

        # ── Take-Profit curve (independent compounding) ─────────────
        equity_tp = [starting_capital]
        dates_tp = [_trades_tp[0]["exit_date"] if _trades_tp else _dt_sim.now()]
        for _t in _trades_tp:
            pos_tp = equity_tp[-1] * 0.05
            pnl_tp = pos_tp * _t["pnl_pct"]
            equity_tp.append(equity_tp[-1] + pnl_tp)
            dates_tp.append(_t["exit_date"])
            _t["position_size"] = round(pos_tp, 2)
            _t["pnl_dollar"] = round(pnl_tp, 2)
            _t["equity_after"] = round(equity_tp[-1], 2)

        # Keep the original variable name used below so we don't have to
        # rewrite every reference.
        scored_sorted = sorted(scored_preds, key=lambda x: x["date"])

        if len(equity) > 2:
            total_return = (equity[-1] / starting_capital - 1) * 100
            ret_color = GREEN if total_return >= 0 else RED
            _running_max = equity[0]
            max_dd = 0.0
            for _ev in equity:
                _running_max = max(_running_max, _ev)
                _dd = ((_ev - _running_max) / _running_max * 100) if _running_max > 0 else 0
                max_dd = min(max_dd, _dd)

            _n_trades = len(_trades)
            _dd_color = RED if max_dd < -5 else AMBER if max_dd < -1 else GREEN

            # Count wins/losses in the simulator to back up the return number
            _sim_wins = sum(1 for t in _trades if t["pnl_dollar"] > 0)
            _sim_losses = sum(1 for t in _trades if t["pnl_dollar"] < 0)

            # ── Take-Profit portfolio stats (parallel calculations) ──
            _tp_total_return = (equity_tp[-1] / starting_capital - 1) * 100
            _tp_ret_color = GREEN if _tp_total_return >= 0 else RED
            _tp_running_max = equity_tp[0]
            _tp_max_dd = 0.0
            for _ev in equity_tp:
                _tp_running_max = max(_tp_running_max, _ev)
                _dd = ((_ev - _tp_running_max) / _tp_running_max * 100) if _tp_running_max > 0 else 0
                _tp_max_dd = min(_tp_max_dd, _dd)
            _tp_wins = sum(1 for t in _trades_tp if t["pnl_dollar"] > 0)
            _tp_losses = sum(1 for t in _trades_tp if t["pnl_dollar"] < 0)
            _tp_target_hit = sum(1 for t in _trades_tp if t.get("tp_realized") == "target")
            _tp_dd_color = RED if _tp_max_dd < -5 else AMBER if _tp_max_dd < -1 else GREEN

            # How much did take-profit add vs hold?
            _tp_delta = _tp_total_return - total_return
            _tp_delta_color = GREEN if _tp_delta >= 0 else RED

            # ── Card with explanation + stats + chart ─────────────────────
            st.markdown(
                f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});border:1px solid {BORDER};"
                f"            border-radius:16px;padding:24px 28px;position:relative;overflow:hidden'>"
                # Brand hairline
                f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
                f"background:{BRAND_GRAD};opacity:0.75'></div>"
                # Cyan eyebrow + big headline
                f"<div style='color:{CYAN};font-size:0.58rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.14em;margin-bottom:6px'>"
                f"Simulated Portfolio</div>"
                f"<div style='font-family:\"Inter\",sans-serif;color:{TEXT_PRIMARY};"
                f"font-size:1.25rem;font-weight:800;letter-spacing:-0.02em;"
                f"margin-bottom:10px'>"
                f"If you followed every signal</div>"
                # How it works explanation — now with scoring policy
                f"<div style='color:{TEXT_SECONDARY};font-size:0.72rem;line-height:1.65;"
                f"margin-bottom:16px;padding:14px 16px;"
                f"background:rgba(6,214,160,0.04);"
                f"border-left:2px solid rgba(6,214,160,0.35);border-radius:6px'>"
                f"<span style='color:{CYAN};font-size:0.5rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.14em;"
                f"display:block;margin-bottom:6px'>How this works</span>"
                f"Start with a hypothetical $10,000. For each of the {_n_trades} scored predictions, "
                f"we simulate placing 5% of the current portfolio on the call — "
                f"<b style='color:{TEXT_PRIMARY}'>long if the model predicted up, "
                f"short if it predicted down.</b> Two strategies run in parallel:"
                f"<div style='margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:10px'>"
                f"<div style='padding:10px 12px;background:rgba(6,214,160,0.06);"
                f"border:1px solid rgba(6,214,160,0.25);border-radius:6px'>"
                f"<span style='color:{CHART_UP};font-size:0.5rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.14em;"
                f"display:block;margin-bottom:4px'>Hold to expiration</span>"
                f"<span style='color:{TEXT_SECONDARY}'>"
                f"<b style='color:{TEXT_PRIMARY}'>No early exits.</b> "
                f"Held to the scheduled checkpoint regardless of intermediate spikes. "
                f"The stricter read.</span>"
                f"</div>"
                f"<div style='padding:10px 12px;background:rgba(245,158,11,0.06);"
                f"border:1px solid rgba(245,158,11,0.25);border-radius:6px'>"
                f"<span style='color:{AMBER};font-size:0.5rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.14em;"
                f"display:block;margin-bottom:4px'>Take profit at target</span>"
                f"<span style='color:{TEXT_SECONDARY}'>"
                f"Closes the position when the stock touches the target price "
                f"(<b style='color:{TEXT_PRIMARY}'>scanned daily high/low</b>). "
                f"Otherwise falls back to hold.</span>"
                f"</div>"
                f"</div></div>"
                # ── Dual-strategy comparison: Hold vs Take-Profit ──
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>"
                # ── HOLD column ──
                f"<div style='background:{BG_SURFACE};border:1px solid {BORDER};"
                f"border-radius:12px;padding:16px 18px;position:relative;overflow:hidden'>"
                f"<div style='position:absolute;top:0;left:0;right:0;height:2px;"
                f"background:{CHART_UP};opacity:0.8'></div>"
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:12px'>"
                f"<div style='width:12px;height:2px;background:{CHART_UP};"
                f"border-radius:1px'></div>"
                f"<span style='color:{CHART_UP};font-size:0.56rem;font-weight:800;"
                f"text-transform:uppercase;letter-spacing:0.14em'>Hold to expiration</span>"
                f"</div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>"
                f"<div><div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:3px'>Total Return</div>"
                f"<div style='font-family:\"Inter\",sans-serif;color:{ret_color};"
                f"font-size:1.35rem;font-weight:900;letter-spacing:-0.03em;line-height:1;"
                f"font-variant-numeric:tabular-nums'>{total_return:+.1f}%</div></div>"
                f"<div><div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:3px'>Portfolio</div>"
                f"<div style='font-family:\"Inter\",sans-serif;color:{TEXT_PRIMARY};"
                f"font-size:1.35rem;font-weight:900;letter-spacing:-0.03em;line-height:1;"
                f"font-variant-numeric:tabular-nums'>${equity[-1]:,.0f}</div></div>"
                f"<div><div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:3px'>Drawdown</div>"
                f"<div style='font-family:\"Inter\",sans-serif;color:{_dd_color};"
                f"font-size:1rem;font-weight:900;letter-spacing:-0.02em;line-height:1;"
                f"font-variant-numeric:tabular-nums'>{max_dd:+.1f}%</div></div>"
                f"<div><div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:3px'>Record</div>"
                f"<div style='font-family:\"Inter\",sans-serif;font-size:1rem;font-weight:900;"
                f"font-variant-numeric:tabular-nums;line-height:1'>"
                f"<span style='color:{CHART_UP}'>{_sim_wins}W</span>"
                f"<span style='color:{TEXT_MUTED};margin:0 4px'>·</span>"
                f"<span style='color:{CHART_DOWN}'>{_sim_losses}L</span></div></div>"
                f"</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.6rem;margin-top:10px;line-height:1.5'>"
                f"Every position held to the scheduled checkpoint — no early exits.</div>"
                f"</div>"
                # ── TAKE-PROFIT column ──
                f"<div style='background:{BG_SURFACE};border:1px solid {BORDER};"
                f"border-radius:12px;padding:16px 18px;position:relative;overflow:hidden'>"
                f"<div style='position:absolute;top:0;left:0;right:0;height:2px;"
                f"background:{AMBER};opacity:0.85'></div>"
                f"<div style='display:flex;align-items:center;justify-content:space-between;"
                f"gap:8px;margin-bottom:12px;flex-wrap:wrap'>"
                f"<div style='display:flex;align-items:center;gap:8px'>"
                f"<div style='width:12px;height:2px;background:{AMBER};"
                f"border-radius:1px;border-top:1.5px dashed {AMBER}'></div>"
                f"<span style='color:{AMBER};font-size:0.56rem;font-weight:800;"
                f"text-transform:uppercase;letter-spacing:0.14em'>Take profit at target</span>"
                f"</div>"
                f"<span style='color:{_tp_delta_color};font-size:0.62rem;font-weight:800;"
                f"font-variant-numeric:tabular-nums'>{_tp_delta:+.1f}% vs hold</span>"
                f"</div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>"
                f"<div><div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:3px'>Total Return</div>"
                f"<div style='font-family:\"Inter\",sans-serif;color:{_tp_ret_color};"
                f"font-size:1.35rem;font-weight:900;letter-spacing:-0.03em;line-height:1;"
                f"font-variant-numeric:tabular-nums'>{_tp_total_return:+.1f}%</div></div>"
                f"<div><div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:3px'>Portfolio</div>"
                f"<div style='font-family:\"Inter\",sans-serif;color:{TEXT_PRIMARY};"
                f"font-size:1.35rem;font-weight:900;letter-spacing:-0.03em;line-height:1;"
                f"font-variant-numeric:tabular-nums'>${equity_tp[-1]:,.0f}</div></div>"
                f"<div><div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:3px'>Drawdown</div>"
                f"<div style='font-family:\"Inter\",sans-serif;color:{_tp_dd_color};"
                f"font-size:1rem;font-weight:900;letter-spacing:-0.02em;line-height:1;"
                f"font-variant-numeric:tabular-nums'>{_tp_max_dd:+.1f}%</div></div>"
                f"<div><div style='color:{TEXT_MUTED};font-size:0.5rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:3px'>Targets Hit</div>"
                f"<div style='font-family:\"Inter\",sans-serif;color:{CYAN};"
                f"font-size:1rem;font-weight:900;letter-spacing:-0.02em;line-height:1;"
                f"font-variant-numeric:tabular-nums'>{_tp_target_hit}"
                f"<span style='color:{TEXT_MUTED};font-weight:500;font-size:0.72rem'>"
                f" / {_n_trades}</span></div></div>"
                f"</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.6rem;margin-top:10px;line-height:1.5'>"
                f"Exits at target price when hit (daily high/low scan). Otherwise falls back to hold.</div>"
                f"</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True)

            # ── Time-range selector for the equity curve ──────────────────
            _ch_head_l, _ch_head_r = st.columns([3, 2])
            with _ch_head_l:
                st.markdown(
                    f"<div class='sec-head' style='display:flex;align-items:center;"
                    f"gap:10px;margin:16px 0 0'>"
                    f"<div class='sec-bar' style='width:3px;height:14px;background:{BRAND_GRAD};"
                    f"border-radius:2px'></div>"
                    f"<span style='color:{TEXT_PRIMARY};font-size:0.7rem;font-weight:800;"
                    f"text-transform:uppercase;letter-spacing:0.14em'>Equity Curve</span>"
                    f"</div>",
                    unsafe_allow_html=True)
            with _ch_head_r:
                _chart_range = st.segmented_control(
                    "Chart range",
                    options=["7D", "1M", "3M", "1Y", "All"],
                    default="All",
                    selection_mode="single",
                    label_visibility="collapsed",
                    key="track_chart_range",
                )
                if _chart_range is None:
                    _chart_range = "All"

            # ── Premium equity curve chart ─────────────────────────────────
            # Daily-resampled so we have a smooth continuous line instead of
            # jagged per-trade steps. Single-color brand fill (no crossover-
            # zero double-fill that was making it busy).
            _eq_color = CHART_UP if equity[-1] >= starting_capital else CHART_DOWN
            _eq_fill = ("rgba(6,214,160,0.16)" if equity[-1] >= starting_capital
                        else "rgba(224,74,74,0.16)")

            # Resample BOTH curves to daily so same-day events don't stack vertically
            try:
                _eq_df = pd.DataFrame({"date": pd.to_datetime(dates_eq),
                                       "equity": equity})
                _eq_df = _eq_df.sort_values("date").drop_duplicates("date", keep="last")
                _eq_df = _eq_df.set_index("date")
                _full_idx = pd.date_range(_eq_df.index.min(), _eq_df.index.max(), freq="D")
                _eq_df = _eq_df.reindex(_full_idx).ffill()
                _plot_x = _eq_df.index
                _plot_y = _eq_df["equity"].values
            except Exception:
                _plot_x = pd.to_datetime(dates_eq)
                _plot_y = equity

            # Same resample for the take-profit curve
            try:
                _tp_df = pd.DataFrame({"date": pd.to_datetime(dates_tp),
                                       "equity": equity_tp})
                _tp_df = _tp_df.sort_values("date").drop_duplicates("date", keep="last")
                _tp_df = _tp_df.set_index("date")
                _tp_full_idx = pd.date_range(_tp_df.index.min(), _tp_df.index.max(), freq="D")
                _tp_df = _tp_df.reindex(_tp_full_idx).ffill()
                _plot_x_tp = _tp_df.index
                _plot_y_tp = _tp_df["equity"].values
            except Exception:
                _plot_x_tp = pd.to_datetime(dates_tp)
                _plot_y_tp = equity_tp

            # ── Apply chart time-range filter to BOTH curves ──────────────
            _CHART_RANGE_DAYS = {"7D": 7, "1M": 30, "3M": 90, "1Y": 365, "All": None}
            _chart_days = _CHART_RANGE_DAYS.get(_chart_range)
            _range_empty = False
            if _chart_days is not None:
                _chart_cutoff = _dt_sim.now() - _td_sim(days=_chart_days)
                try:
                    _mask = pd.to_datetime(_plot_x) >= pd.Timestamp(_chart_cutoff)
                    _plot_x_f = pd.to_datetime(_plot_x)[_mask]
                    _plot_y_f = pd.Series(_plot_y)[_mask].values
                    if len(_plot_x_f) >= 2:
                        _plot_x = _plot_x_f
                        _plot_y = _plot_y_f
                    else:
                        _range_empty = True
                except Exception:
                    pass
                # Same filter for TP curve
                try:
                    _mask_tp = pd.to_datetime(_plot_x_tp) >= pd.Timestamp(_chart_cutoff)
                    _plot_x_tp_f = pd.to_datetime(_plot_x_tp)[_mask_tp]
                    _plot_y_tp_f = pd.Series(_plot_y_tp)[_mask_tp].values
                    if len(_plot_x_tp_f) >= 2:
                        _plot_x_tp = _plot_x_tp_f
                        _plot_y_tp = _plot_y_tp_f
                except Exception:
                    pass

            if _range_empty:
                st.markdown(
                    f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
                    f"border:1px solid {BORDER};border-radius:12px;padding:40px 24px;"
                    f"text-align:center;position:relative;overflow:hidden;margin-top:8px'>"
                    f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
                    f"background:{BRAND_GRAD};opacity:0.45'></div>"
                    f"<div style='font-size:1.3rem;margin-bottom:8px;opacity:0.5'>📉</div>"
                    f"<div style='color:{TEXT_PRIMARY};font-size:0.88rem;font-weight:700;"
                    f"margin-bottom:4px'>No equity activity in this window</div>"
                    f"<div style='color:{TEXT_SECONDARY};font-size:0.72rem;max-width:380px;"
                    f"margin:0 auto;line-height:1.5'>"
                    f"No trades booked in the last <b>{_chart_range}</b>. "
                    f"Try a longer window or <b style='color:{CYAN}'>All</b>.</div>"
                    f"</div>",
                    unsafe_allow_html=True)
                # Skip rendering the chart below
                _plot_x = []
                _plot_y = []

            if len(_plot_y) >= 2:
                # Range includes BOTH curves so neither gets clipped
                _all_y = list(_plot_y) + list(_plot_y_tp) + [starting_capital]
                _y_min = min(float(min(_all_y)), starting_capital)
                _y_max = max(float(max(_all_y)), starting_capital)
                _y_pad = (_y_max - _y_min) * 0.18 if _y_max > _y_min else 300

                fig_eq = go.Figure()

                # ── HOLD curve (primary — with fill) ──
                fig_eq.add_trace(go.Scatter(
                    x=_plot_x, y=_plot_y, mode="lines",
                    name="Hold to expiration",
                    line=dict(color=_eq_color, width=2.5, shape="spline", smoothing=0.8),
                    fill="tozeroy", fillcolor=_eq_fill,
                    hovertemplate=("<b>%{x|%b %d, %Y}</b><br>"
                                   "Hold: <b>$%{y:,.0f}</b><extra></extra>"),
                ))

                # ── TAKE-PROFIT curve (dashed overlay — no fill) ──
                if len(_plot_y_tp) >= 2:
                    fig_eq.add_trace(go.Scatter(
                        x=_plot_x_tp, y=_plot_y_tp, mode="lines",
                        name="Take profit at target",
                        line=dict(color=AMBER, width=2, shape="spline",
                                  smoothing=0.8, dash="dash"),
                        hovertemplate=("<b>%{x|%b %d, %Y}</b><br>"
                                       "Take-Profit: <b>$%{y:,.0f}</b><extra></extra>"),
                    ))

                # $10K baseline — subtle dotted line + right-edge label
                fig_eq.add_hline(
                    y=starting_capital, line_dash="dot",
                    line_color="rgba(255,255,255,0.22)",
                    line_width=1,
                    annotation_text="$10K start",
                    annotation_position="right",
                    annotation_font_color=TEXT_MUTED,
                    annotation_font_size=10,
                )

                # End-of-line label for HOLD curve
                fig_eq.add_annotation(
                    x=_plot_x[-1], y=float(_plot_y[-1]),
                    text=f"<b>${float(_plot_y[-1]):,.0f}</b>",
                    showarrow=False,
                    xanchor="right", yanchor="bottom",
                    xshift=-4, yshift=8,
                    font=dict(color=_eq_color, size=11,
                              family="Inter, sans-serif"),
                    bgcolor="rgba(10,14,20,0.85)",
                    bordercolor=_eq_color, borderpad=4, borderwidth=1,
                )

                # End-of-line label for TP curve (only if meaningfully different)
                if (len(_plot_y_tp) >= 2
                        and abs(float(_plot_y_tp[-1]) - float(_plot_y[-1])) > 50):
                    fig_eq.add_annotation(
                        x=_plot_x_tp[-1], y=float(_plot_y_tp[-1]),
                        text=f"<b>${float(_plot_y_tp[-1]):,.0f}</b>",
                        showarrow=False,
                        xanchor="right", yanchor="bottom",
                        xshift=-4, yshift=8,
                        font=dict(color=AMBER, size=11,
                                  family="Inter, sans-serif"),
                        bgcolor="rgba(10,14,20,0.85)",
                        bordercolor=AMBER, borderpad=4, borderwidth=1,
                    )

                fig_eq.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    template="plotly_dark",
                    font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=11),
                    height=300, margin=dict(l=8, r=12, t=36, b=30),
                    yaxis=dict(
                        title=None,
                        gridcolor="rgba(255,255,255,0.04)",
                        tickprefix="$", tickformat=",",
                        range=[_y_min - _y_pad, _y_max + _y_pad],
                        zeroline=False,
                        tickfont=dict(color=TEXT_MUTED, size=10),
                    ),
                    xaxis=dict(
                        gridcolor="rgba(255,255,255,0.04)", showgrid=False,
                        tickfont=dict(color=TEXT_MUTED, size=10),
                        type="date",
                    ),
                    showlegend=True,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1,
                        font=dict(color=TEXT_SECONDARY, size=10,
                                  family="Inter, sans-serif"),
                        bgcolor="rgba(0,0,0,0)",
                        itemclick="toggle",
                    ),
                    hovermode="x unified",
                    hoverlabel=dict(bgcolor=BG_ELEVATED, font_color=TEXT_PRIMARY,
                                    bordercolor=BORDER_ACCENT,
                                    font_family="Inter, sans-serif",
                                    font_size=11),
                )
                st.plotly_chart(fig_eq, use_container_width=True,
                                config={"displayModeBar": False})

                # Show range-specific note if filtered
                if _chart_range != "All":
                    _range_return = (float(_plot_y[-1]) / float(_plot_y[0]) - 1) * 100
                    _rng_clr = CHART_UP if _range_return >= 0 else CHART_DOWN
                    _range_note = (
                        f"<span style='color:{TEXT_MUTED}'>"
                        f"Showing last <b style='color:{TEXT_SECONDARY}'>{_chart_range}</b> · "
                        f"<b style='color:{_rng_clr}'>{_range_return:+.2f}%</b> in window</span>"
                    )
                else:
                    _range_note = (
                        f"<span style='color:{TEXT_MUTED}'>Showing "
                        f"<b style='color:{TEXT_SECONDARY}'>all time</b></span>"
                    )
                st.markdown(
                    f"<div style='font-size:0.6rem;text-align:center;margin-top:-4px;"
                    f"font-style:italic;line-height:1.55'>"
                    f"{_range_note}"
                    f"<span style='color:{TEXT_MUTED};display:block;margin-top:2px'>"
                    f"Equity updates on each trade's <b>exit date</b> (prediction date + horizon). "
                    f"Hypothetical simulation only — not a guarantee of future results.</span>"
                    f"</div>",
                    unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: PREDICTION LOG — every call, verified
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown(
        f"<div style='margin-top:28px;padding-top:22px;border-top:1px solid {BORDER}'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-end;"
        f"margin-bottom:14px;gap:14px;flex-wrap:wrap'>"
        f"<div>"
        f"<div style='color:{CYAN};font-size:0.58rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.14em;margin-bottom:5px'>"
        f"Prediction Log</div>"
        f"<div style='font-family:\"Inter\",sans-serif;color:{TEXT_PRIMARY};"
        f"font-size:1.25rem;font-weight:800;letter-spacing:-0.02em'>"
        f"Every call, verified</div>"
        f"<div style='color:{TEXT_MUTED};font-size:0.68rem;font-weight:500;"
        f"margin-top:4px'>"
        f"<span style='color:{CYAN};font-size:0.7rem;vertical-align:-1px'>▸</span> "
        f"Click any row to see full prediction + trade details."
        f"</div>"
        f"</div>"
        f"<div style='color:{TEXT_MUTED};font-size:0.72rem;font-weight:600;"
        f"letter-spacing:0.01em;font-variant-numeric:tabular-nums'>"
        f"<span style='color:{GREEN};font-weight:800'>{win_count}W</span>"
        f"<span style='color:{TEXT_MUTED};margin:0 6px'>·</span>"
        f"<span style='color:{RED};font-weight:800'>{loss_count}L</span>"
        f"<span style='color:{TEXT_MUTED};margin:0 6px'>·</span>"
        f"{pending_count} pending"
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True)

    # ── Style for expandable prediction rows (details/summary) ───────────
    st.markdown(
        f"""<style>
        details.pred-row {{
            margin-bottom: 3px;
        }}
        details.pred-row > summary {{
            list-style: none;
            cursor: pointer;
            outline: none;
            user-select: none;
        }}
        details.pred-row > summary::-webkit-details-marker {{ display: none; }}
        details.pred-row > summary::marker {{ display: none; }}
        details.pred-row > summary:hover .pred-row-card {{
            border-color: {BORDER_ACCENT};
            background: {BG_ELEVATED};
        }}
        details.pred-row[open] > summary .pred-row-card {{
            border-color: {BORDER_ACCENT};
            background: {BG_ELEVATED};
            border-radius: 8px 8px 0 0;
        }}
        details.pred-row[open] > summary .pred-row-chevron {{
            transform: rotate(90deg);
            color: {CYAN};
        }}
        details.pred-row .pred-row-chevron {{
            transition: transform 0.18s ease, color 0.18s ease;
            color: {TEXT_MUTED};
            font-size: 0.75rem;
            line-height: 1;
            margin-left: 4px;
            flex-shrink: 0;
            width: 14px;
            text-align: center;
        }}
        details.pred-row .pred-detail-panel {{
            background: linear-gradient(180deg, {BG_ELEVATED}, {BG_SURFACE});
            border: 1px solid {BORDER_ACCENT};
            border-top: none;
            border-radius: 0 0 8px 8px;
            padding: 16px 20px 18px;
            margin-top: -1px;
        }}
        details.pred-row .pred-detail-col-label {{
            color: {CYAN};
            font-size: 0.5rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        details.pred-row .pred-detail-col-label::before {{
            content: "";
            width: 2px;
            height: 10px;
            background: {BRAND_GRAD};
            border-radius: 1px;
        }}
        details.pred-row .pred-detail-row {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 5px 0;
            border-bottom: 1px dashed {BORDER};
            font-size: 0.72rem;
            gap: 12px;
        }}
        details.pred-row .pred-detail-row:last-child {{ border-bottom: none; }}
        details.pred-row .pred-detail-row .k {{
            color: {TEXT_MUTED};
            font-weight: 500;
            letter-spacing: 0.01em;
            white-space: nowrap;
        }}
        details.pred-row .pred-detail-row .v {{
            color: {TEXT_PRIMARY};
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            text-align: right;
            font-family: "Inter", sans-serif;
        }}
        </style>""",
        unsafe_allow_html=True)

    if preds_table:
        scored_list = [p for p in preds_table if p.get("final_result") in ("✓", "✗")]
        pending_list = [p for p in preds_table if p.get("final_result") not in ("✓", "✗")]

        # ── Filter row: status + time range + sort direction ───────────
        _filt_l, _filt_m, _filt_r = st.columns([2, 2.2, 1.6])
        with _filt_l:
            feed_tab = st.radio(
                "Filter", ["Scored", "Pending", "All"],
                horizontal=True, label_visibility="collapsed",
                key="pred_feed_filter",
            )
        with _filt_m:
            _tbl_range = st.segmented_control(
                "Time range",
                options=["7D", "1M", "3M", "1Y", "All"],
                default="All",
                selection_mode="single",
                label_visibility="collapsed",
                key="pred_log_range",
            )
            if _tbl_range is None:
                _tbl_range = "All"
        with _filt_r:
            _tbl_sort = st.segmented_control(
                "Sort",
                options=["Newest", "Oldest"],
                default="Newest",
                selection_mode="single",
                label_visibility="collapsed",
                key="pred_log_sort",
            )
            if _tbl_sort is None:
                _tbl_sort = "Newest"

        if feed_tab == "Scored":
            display_preds = scored_list
        elif feed_tab == "Pending":
            display_preds = pending_list
        else:
            display_preds = preds_table

        # ── Apply time-range filter (by prediction date) ───────────────
        _TBL_RANGE_DAYS = {"7D": 7, "1M": 30, "3M": 90, "1Y": 365, "All": None}
        _tbl_days = _TBL_RANGE_DAYS.get(_tbl_range)
        _tbl_cutoff_str = None
        if _tbl_days is not None:
            from datetime import datetime as _dt_tbl, timedelta as _td_tbl
            _tbl_cutoff_str = (_dt_tbl.now() - _td_tbl(days=_tbl_days)).strftime("%Y-%m-%d")
            display_preds = [p for p in display_preds
                             if p.get("date", "") >= _tbl_cutoff_str]

        # ── Apply sort direction (by prediction date) ──────────────────
        # Copy before sorting so we don't mutate the source list.
        display_preds = sorted(
            list(display_preds),
            key=lambda p: p.get("date", ""),
            reverse=(_tbl_sort == "Newest"),
        )

        # ── Tiny summary strip showing filter state ───────────────────
        _filter_bits = []
        if feed_tab != "All":
            _filter_bits.append(f"<b style='color:{TEXT_SECONDARY}'>{feed_tab}</b>")
        if _tbl_range != "All":
            _filter_bits.append(
                f"last <b style='color:{TEXT_SECONDARY}'>{_tbl_range}</b>"
            )

        _sort_arrow = "↓" if _tbl_sort == "Newest" else "↑"
        _sort_note = (
            f"<span style='color:{CYAN};font-weight:700'>{_sort_arrow}</span> "
            f"<b style='color:{TEXT_SECONDARY}'>{_tbl_sort}</b> first"
        )

        if _filter_bits:
            _filter_line = (
                f"<span style='color:{CYAN}'>Filtered</span> · "
                + " · ".join(_filter_bits)
                + f" · <b style='color:{TEXT_SECONDARY}'>{len(display_preds)}</b> "
                + ("predictions" if len(display_preds) != 1 else "prediction")
                + f" · {_sort_note}"
            )
        else:
            _filter_line = (
                f"Showing <b style='color:{TEXT_SECONDARY}'>all</b> "
                f"<b style='color:{TEXT_SECONDARY}'>{len(display_preds)}</b> predictions "
                f"· {_sort_note}"
            )
        st.markdown(
            f"<div style='color:{TEXT_MUTED};font-size:0.64rem;margin:6px 2px 10px;"
            f"font-weight:500'>{_filter_line}</div>",
            unsafe_allow_html=True)

        per_page = 20
        total_items = len(display_preds)
        if "pred_page" not in st.session_state:
            st.session_state.pred_page = 0

        # Reset to page 0 if filter change pushed us past the end
        max_page = max(0, (total_items - 1) // per_page)
        if st.session_state.pred_page > max_page:
            st.session_state.pred_page = 0
        start_idx = st.session_state.pred_page * per_page
        page_preds = display_preds[start_idx:start_idx + per_page]

        # ── Empty state if filters killed all rows ─────────────────────
        if total_items == 0:
            st.markdown(
                f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
                f"border:1px solid {BORDER};border-radius:12px;padding:36px 24px;"
                f"text-align:center;position:relative;overflow:hidden;margin:8px 0 12px'>"
                f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
                f"background:{BRAND_GRAD};opacity:0.45'></div>"
                f"<div style='font-size:1.3rem;margin-bottom:8px;opacity:0.5'>🔍</div>"
                f"<div style='color:{TEXT_PRIMARY};font-size:0.9rem;font-weight:700;"
                f"margin-bottom:4px'>No predictions match these filters</div>"
                f"<div style='color:{TEXT_SECONDARY};font-size:0.72rem;max-width:380px;"
                f"margin:0 auto;line-height:1.5'>"
                f"Try widening the time range or switching to "
                f"<b style='color:{CYAN}'>All</b>.</div>"
                f"</div>",
                unsafe_allow_html=True)

        # Column header
        _hdr = (f"color:{TEXT_MUTED};font-size:0.52rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.08em")
        st.markdown(
            f"<div style='display:flex;align-items:center;padding:4px 16px;gap:4px;margin-bottom:4px'>"
            f"<div style='flex:1.0;{_hdr}'>Stock / Date</div>"
            f"<div style='flex:0.65;{_hdr}'>Horizon</div>"
            f"<div style='flex:0.7;{_hdr};text-align:right'>Predicted</div>"
            f"<div style='flex:0.45;{_hdr};text-align:center'>Conf</div>"
            f"<div style='flex:0.7;{_hdr};text-align:right'>Actual</div>"
            f"<div style='flex:0.8;{_hdr};text-align:right'>Trade P&L</div>"
            f"<div style='flex:0.3;{_hdr};text-align:center'>Result</div>"
            f"<div style='width:14px;margin-left:4px'></div>"  # chevron column
            f"</div>",
            unsafe_allow_html=True)

        for pred in page_preds:
            sym = pred.get("symbol", "?")
            date = pred.get("date", "")
            horizon = pred.get("horizon", "")
            pred_ret = pred.get("predicted_return", 0)
            conf = pred.get("confidence", 0)
            final = pred.get("final_result", "pending")

            dir_color = GREEN if pred_ret > 0 else RED if pred_ret < 0 else TEXT_MUTED
            is_scored = final in ("✓", "✗")
            result_color = GREEN if final == "✓" else RED if final == "✗" else TEXT_MUTED

            # Result indicator — colored dot instead of text character
            if final == "✓":
                result_html = (f"<div style='width:22px;height:22px;border-radius:50%;background:{GREEN_DIM};"
                               f"border:2px solid {GREEN};display:flex;align-items:center;justify-content:center;"
                               f"margin:0 auto'>"
                               f"<span style='color:{GREEN};font-size:0.7rem;font-weight:900'>✓</span></div>")
            elif final == "✗":
                result_html = (f"<div style='width:22px;height:22px;border-radius:50%;background:{RED_DIM};"
                               f"border:2px solid {RED};display:flex;align-items:center;justify-content:center;"
                               f"margin:0 auto'>"
                               f"<span style='color:{RED};font-size:0.7rem;font-weight:900'>✗</span></div>")
            else:
                result_html = (f"<div style='width:22px;height:22px;border-radius:50%;background:{BG_SURFACE};"
                               f"border:1px solid {BORDER};display:flex;align-items:center;justify-content:center;"
                               f"margin:0 auto'>"
                               f"<span style='color:{TEXT_MUTED};font-size:0.55rem'>···</span></div>")

            actual_ret = None
            if is_scored:
                for interval in ["365d", "180d", "90d", "60d", "30d", "14d", "7d", "3d", "1d"]:
                    r = pred.get(f"return_{interval}")
                    if r is not None:
                        actual_ret = r
                        break
            actual_str = f"{actual_ret:+.1f}%" if actual_ret is not None else "\u2014"
            actual_color = GREEN if actual_ret and actual_ret > 0 else RED if actual_ret and actual_ret < 0 else TEXT_MUTED

            left_border = f"border-left:3px solid {result_color}" if is_scored else f"border-left:3px solid transparent"

            # Confidence as a mini inline bar
            _conf_bar_color = GREEN if conf >= 65 else AMBER if conf >= 50 else RED

            # Format date to a more compact readable form (e.g. "Apr 22")
            _date_short = date
            try:
                from datetime import datetime as _dt_p
                _dt_obj = _dt_p.strptime(date, "%Y-%m-%d")
                _date_short = _dt_obj.strftime("%b %d")
            except Exception:
                pass

            # ── Pull the simulated trade detail, if any ─────────────────
            _trade = trade_pnl_lookup.get((date, sym, horizon))
            if _trade:
                _pnl_dollar = _trade["pnl_dollar"]
                _pos_size = _trade["position_size"]
                _trade_dir = _trade["direction"]
                _pnl_clr = CHART_UP if _pnl_dollar >= 0 else CHART_DOWN
                _pnl_sign = "+" if _pnl_dollar >= 0 else ""
                _pnl_main = f"{_pnl_sign}${_pnl_dollar:,.2f}"
                _pnl_sub = (f"{_trade_dir} · ${_pos_size:,.0f} pos")
                _pnl_cell = (
                    f"<div style='color:{_pnl_clr};font-weight:800;font-size:0.78rem;"
                    f"font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.1'>"
                    f"{_pnl_main}</div>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.56rem;font-weight:500;"
                    f"margin-top:2px;letter-spacing:0.02em'>{_pnl_sub}</div>"
                )
            else:
                _pnl_cell = (
                    f"<div style='color:{TEXT_MUTED};font-size:0.72rem;"
                    f"font-weight:500'>—</div>"
                )

            # ═══════════════════════════════════════════════════════════
            # BUILD THE EXPANDED DETAIL PANEL
            # Two columns: Prediction Details (left) and Trade Details (right)
            # ═══════════════════════════════════════════════════════════

            # ── Prediction Details ────────────────────────────────────
            _pred_id = pred.get("prediction_id", "—") or "—"
            _pred_id_short = (_pred_id[:12] + "…") if len(_pred_id) > 13 else _pred_id
            _model_v = pred.get("model_version", "—")
            _regime = pred.get("regime", "—") or "—"
            _pred_dir = "UP (long)" if pred_ret >= 0 else "DOWN (short)"
            _pred_dir_color = CHART_UP if pred_ret >= 0 else CHART_DOWN
            _entry_price = pred.get("current_price", 0) or 0
            _target_price = pred.get("predicted_price", 0) or 0
            _top_feats = pred.get("top_features", []) or []

            _pred_rows = [
                ("Prediction ID", _pred_id_short, TEXT_SECONDARY),
                ("Model version",  f"v{_model_v}", TEXT_PRIMARY),
                ("Date made",      date, TEXT_PRIMARY),
                ("Market regime",  _regime, TEXT_PRIMARY),
                ("Horizon",        horizon, TEXT_PRIMARY),
                ("Confidence",     f"{conf:.1f}%", TEXT_PRIMARY),
                ("Direction",      _pred_dir, _pred_dir_color),
                ("Predicted return", f"{pred_ret:+.2f}%", _pred_dir_color),
                ("Entry price",    f"${_entry_price:,.2f}", TEXT_PRIMARY),
                ("Target price",   f"${_target_price:,.2f}", TEXT_PRIMARY),
            ]

            _pred_rows_html = "".join(
                f"<div class='pred-detail-row'>"
                f"<span class='k'>{k}</span>"
                f"<span class='v' style='color:{c}'>{v}</span>"
                f"</div>"
                for k, v, c in _pred_rows
            )

            # Top features block — inline chips
            if _top_feats:
                _feat_chips = "".join(
                    f"<span style='display:inline-block;padding:3px 9px;"
                    f"background:rgba(6,214,160,0.08);border:1px solid rgba(6,214,160,0.24);"
                    f"border-radius:100px;color:{CYAN};font-size:0.62rem;"
                    f"font-weight:600;margin:2px 4px 0 0'>{f}</span>"
                    for f in _top_feats[:3]
                )
                _feats_html = (
                    f"<div style='margin-top:10px;padding-top:10px;"
                    f"border-top:1px dashed {BORDER}'>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.6rem;"
                    f"font-weight:600;margin-bottom:4px'>Top signals that drove this call</div>"
                    f"<div>{_feat_chips}</div>"
                    f"</div>"
                )
            else:
                _feats_html = ""

            # ── Target hit / peak favorable move (pulled from the enrichment) ──
            _th = pred.get("target_hit")
            _th_day = pred.get("day_target_hit")
            _th_date = pred.get("target_hit_date")
            _th_price = pred.get("target_hit_price")
            _pk_move = pred.get("peak_favorable_move_pct")
            _pk_day = pred.get("peak_fav_day")
            _pk_price = pred.get("peak_fav_price")
            _horizon_expired = pred.get("horizon_expired", False)

            if _th is True:
                _th_row_label = "Target hit"
                _th_row_value = (f"✓ at day {_th_day} · ${_th_price:,.2f}"
                                 if _th_day is not None and _th_price
                                 else "✓")
                _th_color = CHART_UP
            elif _th is False and _horizon_expired:
                _th_row_label = "Target hit"
                _th_row_value = "✗ never reached"
                _th_color = CHART_DOWN
            elif _th is False:
                _th_row_label = "Target hit"
                _th_row_value = "Not yet · still in window"
                _th_color = AMBER
            else:
                _th_row_label = "Target hit"
                _th_row_value = "—"
                _th_color = TEXT_MUTED

            if _pk_move is not None:
                _pk_row_value = (f"{_pk_move:+.2f}% at day {_pk_day} "
                                 f"(${_pk_price:,.2f})"
                                 if _pk_day is not None else f"{_pk_move:+.2f}%")
                _pk_color = (CHART_UP if _pk_move > 0
                             else CHART_DOWN if _pk_move < 0 else TEXT_SECONDARY)
            else:
                _pk_row_value = "—"
                _pk_color = TEXT_MUTED

            # Take-profit P&L if target was hit: 5% position sized at entry,
            # gains the predicted_return % (cleanly captures the thesis)
            _tp_pnl_value = "—"
            _tp_pnl_color = TEXT_MUTED
            if _trade and _th is True:
                # Take-profit assumes you exit when target is touched. The
                # realized gain = abs(predicted_return). Direction sign is
                # already baked in because you only profit if you traded the
                # right side.
                _tp_pct = abs(pred.get("predicted_return", 0) or 0) / 100.0
                _tp_pnl_dollar = _trade["position_size"] * _tp_pct
                _tp_pnl_value = f"${_tp_pnl_dollar:+,.2f}"
                _tp_pnl_color = CHART_UP
            elif _trade and _th is False and _horizon_expired:
                # Target never hit — take-profit strategy falls back to
                # hold-to-expiration for this trade (same P&L as Hold)
                _tp_pnl_value = f"${_trade['pnl_dollar']:+,.2f} (fell back to hold)"
                _tp_pnl_color = CHART_UP if _trade["pnl_dollar"] >= 0 else CHART_DOWN

            # ── Trade Details ─────────────────────────────────────────
            if _trade:
                _scored_at = _trade["scored_at"]
                _exit_date_str = _trade["exit_date"].strftime("%b %d, %Y")
                _exit_price = pred.get(f"price_{_scored_at}", 0) or 0
                _actual_ret_pct = _trade["actual_ret"] * 100
                _actual_color = CHART_UP if _actual_ret_pct >= 0 else CHART_DOWN
                _pnl_color = CHART_UP if _trade["pnl_dollar"] >= 0 else CHART_DOWN
                _trade_dir_color = CHART_UP if _trade["direction"] == "LONG" else CHART_DOWN
                _scored_at_pretty = _scored_at.replace("d", "-day")

                # Was full horizon reached, or is this an interim checkpoint?
                _horizon_days = {
                    "3 Day": 3, "1 Week": 7, "1 Month": 30,
                    "1 Quarter": 90, "1 Year": 365,
                }.get(horizon, 0)
                _scored_days = int(_scored_at.replace("d", ""))
                if _horizon_days > 0 and _scored_days < _horizon_days:
                    _checkpoint_note = (
                        f"interim · full horizon ({_horizon_days}d) pending"
                    )
                else:
                    _checkpoint_note = "full horizon reached"

                _trade_rows = [
                    ("Simulated direction", _trade["direction"], _trade_dir_color),
                    ("Position size",       f"${_trade['position_size']:,.2f}", TEXT_PRIMARY),
                    ("Entry date",          date, TEXT_PRIMARY),
                    ("Entry price",         f"${_entry_price:,.2f}", TEXT_PRIMARY),
                    # ── Target-hit / peak block ──
                    (_th_row_label,         _th_row_value, _th_color),
                    ("Peak favorable move", _pk_row_value, _pk_color),
                    # ── Exit block ──
                    ("Scored at",           f"{_scored_at_pretty} checkpoint", TEXT_PRIMARY),
                    ("Checkpoint status",   _checkpoint_note, TEXT_SECONDARY),
                    ("Exit date",           _exit_date_str, TEXT_PRIMARY),
                    ("Exit price",          f"${_exit_price:,.2f}", TEXT_PRIMARY),
                    ("Actual return",       f"{_actual_ret_pct:+.2f}%", _actual_color),
                    # ── P&L block — hold vs take-profit ──
                    ("Hold-to-expiration P&L",    f"${_trade['pnl_dollar']:+,.2f}", _pnl_color),
                    ("Take-profit-at-target P&L", _tp_pnl_value, _tp_pnl_color),
                    ("Portfolio after",     f"${_trade['equity_after']:,.2f}", TEXT_PRIMARY),
                ]
                _trade_rows_html = "".join(
                    f"<div class='pred-detail-row'>"
                    f"<span class='k'>{k}</span>"
                    f"<span class='v' style='color:{c}'>{v}</span>"
                    f"</div>"
                    for k, v, c in _trade_rows
                )
                _trade_footer = ""
            else:
                _trade_rows_html = (
                    f"<div style='color:{TEXT_MUTED};font-size:0.75rem;"
                    f"padding:24px 0;text-align:center;line-height:1.6'>"
                    f"<div style='font-size:1.3rem;opacity:0.4;margin-bottom:8px'>⧖</div>"
                    f"Awaiting first checkpoint.<br>"
                    f"<span style='font-size:0.65rem;color:{TEXT_MUTED}'>"
                    f"The simulator books this trade once the 1-day checkpoint "
                    f"is reached.</span>"
                    f"</div>"
                )
                _trade_footer = ""

            # ── Assemble the details block ───────────────────────────
            st.markdown(
                f"<details class='pred-row'>"
                # ── SUMMARY (clickable compact row) ────────────────
                f"<summary>"
                f"<div class='pred-row-card' style='background:{BG_CARD};"
                f"border:1px solid {BORDER};border-radius:8px;"
                f"padding:10px 16px;{left_border};"
                f"display:flex;align-items:center;gap:4px;"
                f"transition:background 0.18s ease,border-color 0.18s ease'>"
                # Stock + date
                f"<div style='flex:1.0'>"
                f"<div style='color:{TEXT_PRIMARY};font-weight:800;font-size:0.84rem;"
                f"font-family:\"Inter\",sans-serif;letter-spacing:-0.01em'>{sym}</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.62rem;font-weight:500;"
                f"font-variant-numeric:tabular-nums;margin-top:1px'>{_date_short}</div>"
                f"</div>"
                # Horizon
                f"<div style='flex:0.65;color:{TEXT_SECONDARY};font-size:0.7rem;font-weight:600'>{horizon}</div>"
                # Predicted return
                f"<div style='flex:0.7;text-align:right;color:{dir_color};font-weight:800;font-size:0.8rem;"
                f"font-variant-numeric:tabular-nums;white-space:nowrap'>{pred_ret:+.1f}%</div>"
                # Confidence with mini bar
                f"<div style='flex:0.45;text-align:center'>"
                f"<div style='color:{TEXT_SECONDARY};font-size:0.72rem;font-weight:700;"
                f"font-variant-numeric:tabular-nums'>{conf:.0f}%</div>"
                f"<div style='width:32px;height:3px;background:{BG_SURFACE};border-radius:2px;margin:3px auto 0;overflow:hidden'>"
                f"<div style='height:100%;width:{min(conf, 100)}%;background:{_conf_bar_color};border-radius:2px'></div>"
                f"</div>"
                f"</div>"
                # Actual return
                f"<div style='flex:0.7;text-align:right;color:{actual_color};font-weight:800;font-size:0.8rem;"
                f"font-variant-numeric:tabular-nums;white-space:nowrap'>{actual_str}</div>"
                # Trade P&L
                f"<div style='flex:0.8;text-align:right'>{_pnl_cell}</div>"
                # Result dot
                f"<div style='flex:0.3;text-align:center'>{result_html}</div>"
                # Chevron — rotates when open
                f"<span class='pred-row-chevron'>▸</span>"
                f"</div>"
                f"</summary>"
                # ── EXPANDED DETAIL PANEL ──────────────────────────
                f"<div class='pred-detail-panel'>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:28px'>"
                # Prediction column
                f"<div>"
                f"<div class='pred-detail-col-label'>Prediction details</div>"
                f"{_pred_rows_html}"
                f"{_feats_html}"
                f"</div>"
                # Trade column
                f"<div>"
                f"<div class='pred-detail-col-label'>Trade details</div>"
                f"{_trade_rows_html}"
                f"{_trade_footer}"
                f"</div>"
                f"</div>"
                f"</div>"
                f"</details>",
                unsafe_allow_html=True)

        # Pagination
        if total_items > per_page:
            st.markdown(f"<div style='height:6px'></div>", unsafe_allow_html=True)
            p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
            with p_col1:
                if st.session_state.pred_page > 0:
                    if st.button("← Previous", key="pred_prev", use_container_width=True):
                        st.session_state.pred_page -= 1
                        st.rerun()
            with p_col2:
                st.markdown(
                    f"<div style='text-align:center;color:{TEXT_MUTED};font-size:0.65rem;padding:8px 0'>"
                    f"    Page {st.session_state.pred_page + 1} of {max_page + 1}"
                    f"    · {total_items} predictions</div>",
                    unsafe_allow_html=True)
            with p_col3:
                if st.session_state.pred_page < max_page:
                    if st.button("Next →", key="pred_next", use_container_width=True):
                        st.session_state.pred_page += 1
                        st.rerun()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: PERFORMANCE BREAKDOWN (expandable)
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown(f"<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Teaser preview — compute quick stats so users know what's inside ───
    _teaser_scored = [p for p in preds_table if p.get("final_result") in ("✓", "✗")]
    _teaser_curr_streak = 0
    _teaser_streak_type = None
    _teaser_best_streak = 0
    if _teaser_scored:
        _teaser_sorted = sorted(_teaser_scored, key=lambda x: x["date"])
        # Current streak
        for p in reversed(_teaser_sorted):
            is_win = p["final_result"] == "✓"
            if _teaser_streak_type is None:
                _teaser_streak_type = is_win
                _teaser_curr_streak = 1
            elif is_win == _teaser_streak_type:
                _teaser_curr_streak += 1
            else:
                break
        # Best win streak
        curr = 0
        for p in _teaser_sorted:
            if p["final_result"] == "✓":
                curr += 1
                _teaser_best_streak = max(_teaser_best_streak, curr)
            else:
                curr = 0

    _teaser_per_symbol = pa.get("per_symbol", {})
    _teaser_top = None
    if _teaser_per_symbol:
        _teaser_ranked = sorted(
            [(sym, d) for sym, d in _teaser_per_symbol.items() if d.get("scored", 0) >= 3],
            key=lambda x: x[1].get("accuracy", 0), reverse=True,
        )
        if _teaser_ranked:
            _teaser_top = _teaser_ranked[0]

    # Build teaser chips
    _chip_parts = []
    if _teaser_scored:
        _streak_color = CHART_UP if _teaser_streak_type else CHART_DOWN
        _streak_word = "win" if _teaser_streak_type else "loss"
        _chip_parts.append(
            f"<span style='color:{TEXT_SECONDARY};font-size:0.72rem'>"
            f"<span style='color:{_streak_color};font-weight:800;font-variant-numeric:tabular-nums'>"
            f"{_teaser_curr_streak}</span> {_streak_word} streak</span>"
        )
        if _teaser_best_streak > 0:
            _chip_parts.append(
                f"<span style='color:{TEXT_SECONDARY};font-size:0.72rem'>"
                f"best: <span style='color:{CHART_UP};font-weight:800;"
                f"font-variant-numeric:tabular-nums'>{_teaser_best_streak}W</span></span>"
            )
    if _teaser_top:
        _top_sym, _top_data = _teaser_top
        _top_acc = _top_data.get("accuracy", 0)
        _chip_parts.append(
            f"<span style='color:{TEXT_SECONDARY};font-size:0.72rem'>"
            f"top: <span style='color:{TEXT_PRIMARY};font-weight:800'>{_top_sym}</span> "
            f"<span style='color:{CYAN};font-weight:800;font-variant-numeric:tabular-nums'>"
            f"{_top_acc:.0f}%</span></span>"
        )

    if _chip_parts:
        _teaser_html = " <span style='color:" + TEXT_MUTED + "'>·</span> ".join(_chip_parts)
        st.markdown(
            f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
            f"border:1px solid {BORDER};border-radius:10px;padding:10px 16px;"
            f"margin-bottom:8px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;"
            f"position:relative;overflow:hidden'>"
            f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
            f"background:{BRAND_GRAD};opacity:0.45'></div>"
            f"<span style='color:{TEXT_MUTED};font-size:0.55rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.12em'>At a glance</span>"
            f"{_teaser_html}"
            f"</div>",
            unsafe_allow_html=True)

    with st.expander("Performance Breakdown — Streaks, Regimes & Calibration", expanded=False):

        # ── Streaks + Top Performers ─────────────────────────────────────
        lb_col1, lb_col2 = st.columns(2)

        with lb_col1:
            scored_for_streak = sorted(
                [p for p in preds_table if p.get("final_result") in ("✓", "✗")],
                key=lambda x: x["date"]
            )
            if scored_for_streak:
                current_streak = 0
                streak_type = None
                for p in reversed(scored_for_streak):
                    is_win = p["final_result"] == "✓"
                    if streak_type is None:
                        streak_type = is_win
                        current_streak = 1
                    elif is_win == streak_type:
                        current_streak += 1
                    else:
                        break

                best_win_streak = 0
                curr = 0
                for p in scored_for_streak:
                    if p["final_result"] == "✓":
                        curr += 1
                        best_win_streak = max(best_win_streak, curr)
                    else:
                        curr = 0

                streak_color = GREEN if streak_type else RED
                streak_label = "Win" if streak_type else "Loss"

                st.markdown(
                    f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
                    f"            border:1px solid {BORDER};"
                    f"            border-radius:10px;padding:18px;text-align:center;"
                    f"            position:relative;overflow:hidden'>"
                    f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
                    f"background:{BRAND_GRAD};opacity:0.55'></div>"
                    f"<div style='color:{CYAN};font-size:0.56rem;font-weight:700;"
                    f"text-transform:uppercase;letter-spacing:0.14em;margin-bottom:16px'>"
                    f"Streaks</div>"
                    f"<div style='display:flex;justify-content:space-around;align-items:center'>"
                    f"<div>"
                    f"<div style='font-family:\"Inter\",sans-serif;color:{streak_color};"
                    f"font-size:1.9rem;font-weight:900;letter-spacing:-0.03em;line-height:1;"
                    f"font-variant-numeric:tabular-nums'>{current_streak}</div>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.56rem;font-weight:700;"
                    f"text-transform:uppercase;letter-spacing:0.12em;margin-top:6px'>"
                    f"Current {streak_label}</div>"
                    f"</div>"
                    f"<div style='width:1px;height:44px;background:{BORDER}'></div>"
                    f"<div>"
                    f"<div style='font-family:\"Inter\",sans-serif;color:{CHART_UP};"
                    f"font-size:1.9rem;font-weight:900;letter-spacing:-0.03em;line-height:1;"
                    f"font-variant-numeric:tabular-nums'>{best_win_streak}</div>"
                    f"<div style='color:{TEXT_MUTED};font-size:0.56rem;font-weight:700;"
                    f"text-transform:uppercase;letter-spacing:0.12em;margin-top:6px'>"
                    f"Best Win Streak</div>"
                    f"</div>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True)

        with lb_col2:
            per_symbol = pa.get("per_symbol", {})
            if per_symbol:
                ranked = sorted(
                    [(sym, d) for sym, d in per_symbol.items() if d.get("scored", 0) >= 3],
                    key=lambda x: x[1].get("accuracy", 0), reverse=True,
                )
                st.markdown(
                    f"<div style='background:linear-gradient(145deg,{BG_CARD},{BG_SURFACE});"
                    f"            border:1px solid {BORDER};"
                    f"            border-radius:10px;padding:18px;"
                    f"            position:relative;overflow:hidden'>"
                    f"<div style='position:absolute;top:0;left:0;right:0;height:1.5px;"
                    f"background:{BRAND_GRAD};opacity:0.55'></div>"
                    f"<div style='color:{CYAN};font-size:0.56rem;font-weight:700;"
                    f"text-transform:uppercase;letter-spacing:0.14em;margin-bottom:12px'>"
                    f"Top Performers</div>",
                    unsafe_allow_html=True)
                if ranked:
                    for i, (sym, d) in enumerate(ranked[:5]):
                        sym_acc = d.get("accuracy", 0)
                        sym_scored = d.get("scored", 0)
                        sym_correct = d.get("correct", 0)
                        rank_colors = [CYAN, GREEN, AMBER, TEXT_SECONDARY, TEXT_MUTED]
                        rc = rank_colors[min(i, 4)]
                        medals = ["🥇", "🥈", "🥉", "4.", "5."]

                        st.markdown(
                            f"<div style='display:flex;align-items:center;justify-content:space-between;"
                            f"            padding:6px 0;{'border-bottom:1px solid ' + BORDER if i < min(len(ranked)-1,4) else ''}'>"
                            f"<div style='display:flex;align-items:center;gap:7px'>"
                            f"<span style='font-size:0.75rem;min-width:18px'>{medals[min(i,4)]}</span>"
                            f"<span style='color:{TEXT_PRIMARY};font-weight:700;font-size:0.78rem'>{sym}</span>"
                            f"</div>"
                            f"<div style='text-align:right'>"
                            f"<span style='color:{rc};font-weight:800;font-size:0.78rem'>{sym_acc:.0f}%</span>"
                            f"<span style='color:{TEXT_MUTED};font-size:0.5rem;margin-left:4px'>{sym_correct}/{sym_scored}</span>"
                            f"</div>"
                            f"</div>",
                            unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color:{TEXT_MUTED};font-size:0.72rem;padding:8px 0'>Need 3+ scored per symbol.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # ── Regime performance ──────────────────────────────────────────
        regime_acc = pa.get("regime_accuracy", {})
        if regime_acc:
            st.markdown(f"<div style='height:18px'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='sec-head' style='margin-bottom:10px'>"
                "<span class='sec-bar'></span>Accuracy by Market Regime</div>",
                unsafe_allow_html=True)

            regime_cols = st.columns(len(regime_acc))
            for col, (rname, rd) in zip(regime_cols, regime_acc.items()):
                with col:
                    r_acc = rd.get("accuracy", 0)
                    r_scored = rd.get("scored", 0)
                    r_color = GREEN if r_acc >= 55 else AMBER if r_acc >= 50 else RED
                    rbg = {"Bull": GREEN_DIM, "Bear": RED_DIM}.get(rname, BG_CARD)

                    st.markdown(
                        f"<div style='background:linear-gradient(145deg,{rbg},{BG_SURFACE});"
                        f"            border:1px solid {r_color}20;border-radius:8px;"
                        f"            padding:12px 8px;text-align:center;position:relative;overflow:hidden'>"
                        f"<div style='position:absolute;bottom:0;left:0;right:0;height:3px;background:{BG_SURFACE}'>"
                        f"<div style='height:100%;width:{r_acc}%;background:{r_color};opacity:0.6'></div>"
                        f"</div>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.45rem;font-weight:700;text-transform:uppercase;"
                        f"            letter-spacing:1px;margin-bottom:3px'>{rname}</div>"
                        f"<div style='color:{r_color};font-size:1.2rem;font-weight:900'>{r_acc:.0f}%</div>"
                        f"<div style='color:{TEXT_MUTED};font-size:0.42rem;margin-top:2px'>{r_scored} scored</div>"
                        f"</div>",
                        unsafe_allow_html=True)

        # ── Calibration ─────────────────────────────────────────────────
        cal_data = pa.get("confidence_calibration", [])
        if cal_data:
            st.markdown(f"<div style='height:18px'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='sec-head' style='margin-bottom:6px'>"
                "<span class='sec-bar'></span>Confidence Calibration</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.68rem;margin-bottom:12px;"
                f"line-height:1.55;max-width:560px'>"
                f"Is the model's confidence well-calibrated? Bars <b style='color:{TEXT_SECONDARY}'>"
                f"above the line</b> = model is conservative (good). "
                f"<b style='color:{TEXT_SECONDARY}'>Below the line</b> = overconfident.</div>",
                unsafe_allow_html=True)

            fig_cal = go.Figure()
            x_vals = [c["predicted_confidence"] for c in cal_data]
            y_vals = [c["actual_accuracy"] for c in cal_data]

            fig_cal.add_trace(go.Bar(
                x=[c["confidence_range"] for c in cal_data], y=y_vals,
                marker=dict(
                    color=[CYAN if y >= x else AMBER for x, y in zip(x_vals, y_vals)],
                    line=dict(color="rgba(6,214,160,0.25)", width=1),
                ),
                opacity=0.85,
                text=[f"{c['actual_accuracy']:.0f}% (n={c['count']})" for c in cal_data],
                textposition="outside",
                textfont=dict(color=TEXT_SECONDARY, size=9),
                name="Actual", hovertemplate="<b>%{x}</b><br>Actual: %{y:.1f}%<extra></extra>",
            ))
            fig_cal.add_trace(go.Scatter(
                x=[c["confidence_range"] for c in cal_data], y=x_vals,
                mode="lines+markers",
                line=dict(color=AMBER, width=2, dash="dash"),
                marker=dict(size=5, color=AMBER), name="Perfect",
            ))
            fig_cal.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                template="plotly_dark",
                font=dict(family="Inter, sans-serif", color=TEXT_SECONDARY, size=11),
                height=260, margin=dict(l=40, r=20, t=10, b=50),
                yaxis=dict(title=None, range=[0, 105], gridcolor=CHART_GRID, ticksuffix="%"),
                xaxis=dict(gridcolor=CHART_GRID),
                showlegend=True,
                legend=dict(orientation="h", y=-0.2, font=dict(size=9, color=TEXT_MUTED),
                            bgcolor="rgba(0,0,0,0)"),
                hoverlabel=dict(bgcolor=BG_ELEVATED, font_color=TEXT_PRIMARY,
                                bordercolor=BORDER_ACCENT, font_family="Inter, sans-serif"),
            )
            st.plotly_chart(fig_cal, use_container_width=True, config={"displayModeBar": False})

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPORT
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(f"<div style='height:24px'></div>", unsafe_allow_html=True)

    if preds_table:
        _export_df = pd.DataFrame(preds_table)
        _csv = _export_df.to_csv(index=False)
        st.download_button(
            label="Export Track Record as CSV",
            data=_csv,
            file_name="prediqt_track_record.csv",
            mime="text/csv",
            use_container_width=False,
        )
