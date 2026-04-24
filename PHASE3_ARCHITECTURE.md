# Phase 3 Architecture — Auto-Learning System

## System Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                     SIGNALHEDGE ML STOCK SIGNALS                       │
│                   Auto-Learning System Architecture                    │
└───────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ USER INTERFACE (Streamlit)                                          │
│                                                                     │
│  [Stock Input] → [Run Analysis]                                    │
│         ↓              ↓                                            │
│  [📈 Chart Tab] [🎯 Predictions Tab] ← Shows Retrain Status (NEW)  │
│  [📊 Analysis] [📋 Options]          ← Shows Feature Learning      │
│  [🔄 Backtest] [🤖 Model]            ← Shows Model Version         │
└─────────────────────────────────────────────────────────────────────┘
              ↓                              ↓
              ├──────────────────────────────┤
              ↓                              ↓
┌──────────────────────────────┐   ┌──────────────────────────────┐
│   PREDICTION PIPELINE        │   │  ANALYSIS PIPELINE           │
├──────────────────────────────┤   ├──────────────────────────────┤
│ 1. Fetch Stock Data          │   │ 1. Train 16-Model Ensemble  │
│ 2. Engineer Features (75+)   │   │ 2. Generate ML Predictions  │
│ 3. Detect Market Regime      │   │ 3. Calculate Confidence     │
│ 4. Train Ensemble Model      │   │ 4. Run Walk-Forward Test    │
│ 5. Generate Predictions      │   │ 5. Detect Trading Signals   │
│ 6. Apply Confidence Adjust   │   │ 6. Suggest Options Trades   │
│    ↓ (NEW)                   │   └──────────────────────────────┘
└──────────────────────────────┘
      ↓
┌──────────────────────────────┐
│ PREDICTION LOGGING (v2)      │
├──────────────────────────────┤
│ • Log prediction + regime    │
│ • Capture feature importance │
│ • Record model version       │
│ • Store confidence (adjusted)│
│ • Run instant backtest       │
│                              │
│ Output: predictions.jsonl    │
└──────────────────────────────┘
      ↓
┌──────────────────────────────────────────────────────────────────┐
│                  AUTO-LEARNING SYSTEM (PHASE 3)                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. PREDICTION SCORING (prediction_logger_v2.py)        │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ • Score predictions at 1d, 3d, 7d, 30d, 90d intervals  │   │
│  │ • Calculate accuracy per horizon                        │   │
│  │ • Generate confidence calibration data                  │   │
│  │ • Compile analytics across 6 dimensions                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 2. OUTCOME ANALYSIS (model_improvement.py)             │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ • Analyze prediction outcomes                           │   │
│  │ • Identify top/bottom features                          │   │
│  │ • Detect confidence bias                                │   │
│  │ • Compare regime-specific performance                   │   │
│  │ • Calculate expected improvement                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 3. RETRAIN DECISION (auto_retrain.py) ← NEW             │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │ should_retrain_now()                                    │   │
│  │   ├─ Check: 20+ new scored predictions?               │   │
│  │   ├─ Check: Accuracy declining?                       │   │
│  │   ├─ Check: 30+ days since last retrain?              │   │
│  │   └─ Return: (bool, reason)                           │   │
│  │                                                         │   │
│  │ IF should_retrain:                                      │   │
│  │   ├─ get_retrain_adjustments()                         │   │
│  │   │   ├─ Feature boosts (top 5 positive)             │   │
│  │   │   ├─ Feature penalties (bottom 5 negative)       │   │
│  │   │   ├─ Confidence bias corrections                  │   │
│  │   │   └─ Regime-specific multipliers                  │   │
│  │   │                                                    │   │
│  │   ├─ calculate_expected_improvement()                 │   │
│  │   │   ├─ Feature impact: +0-5%                        │   │
│  │   │   ├─ Calibration impact: +0-5%                    │   │
│  │   │   ├─ Regime impact: +0-1.5%                       │   │
│  │   │   └─ Total: 0-10% expected                        │   │
│  │   │                                                    │   │
│  │   └─ execute_retrain()                                │   │
│  │       ├─ Apply adjustments to model                   │   │
│  │       ├─ Bump model version (v1.0→v1.1→v2.0)        │   │
│  │       ├─ Log retrain event                            │   │
│  │       └─ Persist adjustments                          │   │
│  │                                                         │   │
│  │ ELSE (not ready to retrain):                           │   │
│  │   └─ Show countdown: "N predictions until retrain"    │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 4. APPLY LEARNED ADJUSTMENTS (app.py) ← INTEGRATED      │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │ When generating NEW predictions:                        │   │
│  │   apply_confidence_adjustments(horizon, conf, regime)  │   │
│  │     ├─ Apply global bias correction                   │   │
│  │     ├─ Apply horizon-specific adjustment              │   │
│  │     ├─ Apply regime multiplier                        │   │
│  │     └─ Return: adjusted_confidence (30-95%)           │   │
│  │                                                         │   │
│  │ Effect: Predictions use learned calibration            │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 5. MONITORING & DISPLAY (app.py) ← NEW UI               │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │ get_improvement_metrics()                               │   │
│  │   ├─ Current version                                   │   │
│  │   ├─ Retrain count                                     │   │
│  │   ├─ Accuracy metrics (current, best, worst)           │   │
│  │   ├─ Feature importance ranking                        │   │
│  │   ├─ Calibration quality                               │   │
│  │   ├─ Per-horizon performance                           │   │
│  │   ├─ Trend analysis (improving/declining)              │   │
│  │   └─ Next retrain countdown                            │   │
│  │                                                         │   │
│  │ Displays in Predictions Tab:                            │   │
│  │   • Summary Cards (5 columns)                          │   │
│  │   • Retrain Status Alert (when ready)                  │   │
│  │   • Expected Improvement Forecast                      │   │
│  │   • Feature Learning Ranking                           │   │
│  │   • Confidence Calibration Chart                       │   │
│  │   • Accuracy Trend Chart                               │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 6. CONTINUOUS MONITORING (Scheduler) ← NEW              │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │ Daily at 9:07 AM:                                       │   │
│  │   check_and_log_retrain_status()                        │   │
│  │     ├─ Check if retrain is ready                       │   │
│  │     ├─ Log metrics to retrain history                  │   │
│  │     ├─ Track trend (improving/declining/stable)        │   │
│  │     └─ Notify user if status changed                   │   │
│  │                                                         │   │
│  │ Retrain History:                                        │   │
│  │   • Last 100 status checks retained                    │   │
│  │   • Timestamps recorded                                │   │
│  │   • Metrics tracked over time                          │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
      ↓                                           ↓
┌──────────────────────────┐         ┌──────────────────────────┐
│ DATA PERSISTENCE         │         │ USER FEEDBACK           │
├──────────────────────────┤         ├──────────────────────────┤
│ • predictions.jsonl      │         │ Predictions Tab shows:   │
│ • retrain_history.json   │         │ • Retrain readiness     │
│ • model_adjustments.json │         │ • Expected improvement  │
│ • analytics_cache.json   │         │ • Feature learning      │
│ • calibration_data.json  │         │ • Accuracy trends       │
│ • feature_importance.json│         │ • Model version         │
└──────────────────────────┘         └──────────────────────────┘
```

---

## Data Flow: Prediction → Learning → Retraining

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: USER RUNS ANALYSIS                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Stock Symbol (AAPL), Period (5Y)                       │
│      ↓                                                          │
│  • Fetch historical OHLCV data                                │
│  • Calculate 75+ technical indicators                         │
│  • Get market context (SPY, VIX)                              │
│  • Detect market regime (Bull/Bear/Sideways)                  │
│      ↓                                                          │
│  • Train 16-model ensemble (XGBoost, LGB, RF, etc.)           │
│  • Run walk-forward validation                                │
│  • Generate predictions for 4 horizons:                       │
│    - 1 Week                                                   │
│    - 1 Month                                                  │
│    - 1 Quarter                                                │
│    - 1 Year                                                   │
│      ↓                                                          │
│  Output: 4 predictions with confidence, direction, magnitude   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: PREDICTION LOGGING WITH LEARNING                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each prediction:                                          │
│      1. Capture model feature importance (top 10)             │
│      2. Get market regime label (Bull/Bear/Sideways)          │
│      3. Apply learned confidence adjustment:                  │
│         base_conf = 65%                                        │
│         adj_conf = 65% × calibration_factor                    │
│         e.g., 65% → 62% (underconfident in bears)            │
│      4. Log to predictions.jsonl:                              │
│         {                                                      │
│           "symbol": "AAPL",                                   │
│           "date": "2026-04-09",                               │
│           "horizon": "1 Month",                               │
│           "predicted_return": 0.045,                          │
│           "confidence": 62.0,    ← ADJUSTED                   │
│           "regime": "Bull",                                   │
│           "model_version": "1.0",                             │
│           "features": {top 10},   ← NEW in Phase 3           │
│           "scores": {}            ← To be filled in later     │
│         }                                                      │
│      5. Run instant backtest (25 samples):                    │
│         - How would model have done historically?             │
│         - Gives immediate accuracy estimate                   │
│                                                                 │
│  Output: Prediction logged, scored, displayed                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: CONTINUOUS PREDICTION SCORING                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Background scoring (as time passes):                          │
│                                                                 │
│  Day 1: Check if actual price moved in predicted direction    │
│    → score_1d: {correct: true/false, return: 0.02, ...}      │
│                                                                 │
│  Day 3: Check 3-day return                                    │
│    → score_3d: {correct: true/false, return: -0.01, ...}     │
│                                                                 │
│  Day 7: Check 7-day return                                    │
│    → score_7d: {correct: true/false, return: 0.05, ...}      │
│                                                                 │
│  Day 30: Check 30-day return                                  │
│    → score_30d: {correct: true/false, return: 0.12, ...}     │
│                                                                 │
│  Day 90: Check 90-day return                                  │
│    → score_90d: {correct: true/false, return: 0.08, ...}     │
│                                                                 │
│  Day 365: Check 365-day return (final)                        │
│    → score_365d: {correct: true/false, return: 0.35, ...}    │
│                                                                 │
│  Each prediction accumulates scores over time                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: LEARN FROM OUTCOMES (After 20+ Scored)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Analyze all scored predictions:                              │
│                                                                 │
│  A. Feature Importance Learning                               │
│     RSI appears in top 5 features for 80% of accurate preds  │
│       → BOOST RSI in next model (importance += 0.5)           │
│     Stochastic in top 5 for only 20% of accurate preds       │
│       → PENALIZE Stochastic (importance -= 0.3)               │
│                                                                 │
│  B. Confidence Calibration                                    │
│     Predictions with 70% confidence: 64% actual accuracy     │
│       → UNDERCONFIDENT by 6%                                   │
│       → Boost future 70% preds to 75%                         │
│     Predictions with 50% confidence: 58% actual accuracy     │
│       → OVERCONFIDENT by 8%                                    │
│       → Reduce future 50% preds to 45%                        │
│                                                                 │
│  C. Regime-Specific Learning                                  │
│     In Bull markets: 60% accuracy (good)                      │
│     In Bear markets: 48% accuracy (poor)                      │
│       → REDUCE confidence by 15% in bear markets              │
│                                                                 │
│  Output: Adjustments to apply                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: DECIDE IF RETRAIN IS NEEDED                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Check retrain triggers:                                      │
│                                                                 │
│  ✓ 20+ new predictions scored in this batch?                 │
│    → should_retrain = True                                    │
│       reason = "20 predictions with high confidence scored"   │
│                                                                 │
│  ✓ Accuracy declining (last 10 worse than previous 10)?      │
│    → should_retrain = True                                    │
│       reason = "Accuracy declining, model needs refresh"      │
│                                                                 │
│  ✓ 30+ days since last retrain?                              │
│    → should_retrain = True                                    │
│       reason = "Monthly retrain milestone reached"            │
│                                                                 │
│  IF should_retrain:                                            │
│      Calculate expected improvement:                          │
│         • Feature boosts: +2.0% (good features helping)       │
│         • Calibration: +1.5% (fixing overconfidence)          │
│         • Regime adjustments: +0.5% (bear market help)        │
│         → Total: +4.0% expected improvement                   │
│                                                                 │
│      Display alert: "Ready to Retrain"                        │
│      Show expected improvement: "+4.0%"                       │
│      Show retrain count: 1 (first retrain)                    │
│                                                                 │
│  ELSE (not ready):                                             │
│      Show countdown: "5 more predictions until retrain"       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: RETRAIN (When Triggered)                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Execute retraining:                                          │
│                                                                 │
│  1. Apply feature importance updates:                         │
│     - Boost RSI (multiply by 1.05)                            │
│     - Penalize Stochastic (multiply by 0.97)                  │
│                                                                 │
│  2. Apply confidence calibration:                             │
│     - Increase predictions by 5-10% in most cases             │
│     - Decrease in bear markets by 15%                         │
│                                                                 │
│  3. Persist adjustments:                                      │
│     - Save to model_adjustments.json                          │
│     - Increment model version: v1.0 → v1.1                    │
│     - Log retrain event to retrain_history.json               │
│                                                                 │
│  4. Apply conservative improvement:                           │
│     - Expected +4.0%, apply +2.0% (50%)                       │
│     - Reason: Be conservative to avoid overfitting            │
│                                                                 │
│  Output: Model version incremented, adjustments persisted     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: NEXT PREDICTIONS USE ADJUSTMENTS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  When user runs next analysis:                                │
│                                                                 │
│  • Model generates predictions as before                      │
│  • Confidence adjusted using learned calibration:             │
│    apply_confidence_adjustments("1 Month", 65%, "Bull")       │
│      → Returns adjusted confidence (e.g., 68%)                │
│  • Prediction logged with adjusted confidence                 │
│  • New model version (v1.1) is recorded                       │
│                                                                 │
│  Effect: Model is SMARTER and more CALIBRATED                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 8: CONTINUOUS IMPROVEMENT CYCLE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Repeat: Every 20 predictions scored, learn and improve       │
│                                                                 │
│  v1.0 (baseline) → 55% accuracy                               │
│  +10 predictions → 56% accuracy (learning)                    │
│  Retrain #1 (v1.1) → 57% accuracy                             │
│  +15 predictions → 57% accuracy                               │
│  Retrain #2 (v2.0) → 59% accuracy                             │
│  +20 predictions → 58% accuracy (declining)                   │
│  Retrain #3 (v2.1) → 60% accuracy                             │
│                                                                 │
│  Model continuously improves as more data collected            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Interaction Diagram

```
                         USER (Streamlit)
                              │
                    ┌─────────┴─────────┐
                    │                   │
            [Run Analysis]      [View Predictions]
                    │                   │
         ┌──────────┴──────────┐        │
         │                     │        │
    [model.py]             [app.py]     │
   (Ensemble)         (UI + Orchestration)
         │                     │        │
         └─────────┬───────────┘        │
                   │                    │
          ┌────────▼──────────┐         │
          │ log_prediction_v2 │◄─────┐  │
          │                  │       │  │
          │ • Log prediction │       │  │
          │ • Extract features       │  │
          │ • Get regime             │  │
          │ • Backtest score        │  │
          └────────┬──────────┘       │  │
                   │                  │  │
         ┌─────────▼────────┐         │  │
         │ .predictions/    │         │  │
         │ predictions.jsonl│         │  │
         └──────────────────┘         │  │
                                      │  │
              ┌───────────────────────┘  │
              │                          │
    ┌─────────▼──────────────┐           │
    │ score_all_intervals()  │           │
    │                        │           │
    │ • Score 1d, 3d, 7d... │           │
    │ • Calc accuracy        │           │
    │ • Build analytics      │           │
    └─────────┬──────────────┘           │
              │                          │
    ┌─────────▼──────────────┐           │
    │ model_improvement.py   │           │
    │                        │           │
    │ • Analyze outcomes     │           │
    │ • Learn features       │           │
    │ • Get recommendations  │           │
    └─────────┬──────────────┘           │
              │                          │
    ┌─────────▼──────────────┐           │
    │ auto_retrain.py (NEW)  │           │
    │                        │           │
    │ • should_retrain_now() ├───────────┼──► [Display Alert]
    │ • get_improvements()   │           │
    │ • apply_adjustments()  │───────────┘
    └────────────────────────┘

   ┌──────────────────────────────────┐
   │ Daily Scheduler (9 AM)           │
   │                                  │
   │ check_and_log_retrain_status()  │
   │ • Monitor readiness             │
   │ • Log metrics                   │
   │ • Notify user                   │
   └──────────────────────────────────┘
```

---

## File Dependencies

```
app.py
├── Imports: auto_retrain [NEW]
│   ├── should_retrain_now()
│   ├── get_improvement_metrics()
│   ├── execute_retrain()
│   └── apply_confidence_adjustments()
│
├── Imports: model_improvement [EXISTING]
│   ├── get_improvement_summary()
│   └── analyze_prediction_outcomes()
│
├── Imports: prediction_logger_v2 [EXISTING]
│   ├── log_prediction_v2()
│   ├── score_all_intervals()
│   ├── get_full_analytics()
│   ├── backtest_accuracy()
│   └── get_current_model_version()
│
└── Imports: model [EXISTING]
    ├── StockPredictor class
    └── get_feature_importance()

auto_retrain.py [NEW]
├── Imports: prediction_logger_v2
│   ├── get_full_analytics()
│   └── get_current_model_version()
│
├── Imports: model_improvement
│   └── ModelImprover class
│
└── Imports: json, os, datetime [STANDARD]
    └── For file I/O and logging

model_improvement.py [EXISTING]
├── Imports: prediction_logger_v2
│   └── get_full_analytics()
│
└── Imports: json, os [STANDARD]

prediction_logger_v2.py [EXISTING]
├── Imports: model.py
│   └── Uses predictor for feature importance
│
└── Imports: External (yfinance, etc.)
```

---

## State Machine: Retrain Readiness

```
┌──────────────────────┐
│   INITIAL STATE      │
│   (0 predictions)    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐      ┌──────────────────┐
│  LEARNING MODE       │──┐   │ UPDATE COUNTDOWN │
│  (1-19 predictions)  │  │   │ N = 20 - scored  │
└──────────┬───────────┘  │   └──────────────────┘
           │              │
           │ scored ≥ 20? │
           │              ▼
           │         ┌──────────────────────┐
           │         │ CHECK RETRAIN LOGIC  │
           │         │                      │
           │         │ • Accuracy declining?
           │         │ • 30+ days passed?   │
           │         └────────┬─────────────┘
           │                  │
           │         ┌────────┴────────┐
           │         │                 │
           │    YES  │                 │ NO
           │    ┌────▼──────┐     ┌────▼──────┐
           │    │ READY TO  │     │ KEEP      │
           │    │ RETRAIN   │     │ LEARNING  │
           │    │ (SHOW:    │     │ (SHOW:    │
           │    │ "Ready",  │     │ "15 more  │
           │    │ "+2.3%")  │     │ away")    │
           │    └────┬──────┘     └────┬──────┘
           │         │                 │
           │         ▼                 │
           │    ┌──────────────────┐   │
           │    │ RETRAIN EXECUTED │   │
           │    │ Version bumped   │   │
           │    │ (v1.0 → v1.1)    │   │
           │    └────┬─────────────┘   │
           │         │                 │
           │    ┌────▼──────┐     ┌────▼──────┐
           │    │ PREDICTIONS       │ NEW      │
           │    │ USE ADJUSTED      │ BATCH    │
           │    │ CONFIDENCE        │ CONTINUES
           │    │ (APPLY LEARNED)   │ LEARNING │
           │    └──────────────────┘ └─────────┘
           │
           └─────────────────────────────┘
               (Cycle repeats)
```

---

## Summary

**Phase 3 builds a complete auto-learning loop:**

1. **Collect** — Log predictions with features and regime
2. **Score** — Track outcomes at multiple time intervals  
3. **Analyze** — Learn what works, what doesn't
4. **Adjust** — Apply learning to confidence & features
5. **Improve** — Retrain when 20+ scored predictions or decline detected
6. **Monitor** — Daily scheduler checks health
7. **Cycle** — Repeat continuously for improving performance

Each component is modular, testable, and contributes to the auto-improvement system.

---

**Status:** ✅ Ready for testing and deployment
