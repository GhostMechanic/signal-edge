# SignalEdge Full System Architecture

## Overview
Production-grade prediction tracking and model improvement system with accelerated scoring, detailed analytics, and continuous learning.

---

## System Components

### 1. ACCELERATED SCORING (Get results in days, not weeks)

#### A. Quick-Check Scoring (1-2 day results)
- Check stock price after just 1-2 days instead of waiting 7-30 days
- Provides immediate feedback on prediction direction
- Allows faster model iteration

**Implementation:**
- New function: `quick_score_predictions()` 
- Runs daily, checks predictions made 1-2 days ago
- Marks "preliminary" scores (full scores come later)
- Shows in dashboard as "Quick Check: 68% (preliminary)"

#### B. Backtest Synthetic Scoring
- Use historical data to simulate past predictions
- For each stock analysis, run on 6-month historical data
- Ask: "If I had made this prediction 6 months ago, would it have been correct?"
- Provides 20-30 synthetic data points instantly

**Implementation:**
- New function: `backtest_predictions()`
- Analyzes last 252 trading days
- Simulates predictions at 20 random points in past
- Returns synthetic accuracy scores
- Labeled as "Backtested Accuracy" (separate from live)

#### C. Progressive Scoring
- Track predictions at multiple intervals:
  - 1 day check (preliminary)
  - 3 day check (trending)
  - 7 day check (1-week final)
  - 30 day check (1-month final)
  - etc.
- Update confidence calibration with each interval

**Data Structure:**
```python
{
  "prediction_id": "abc123",
  "symbol": "GOOG",
  "date": "2026-04-09",
  "horizon": "1 Month",
  "predicted_return": 0.05,
  "predicted_price": 330.00,
  "current_price": 316.49,
  "confidence": 0.65,
  "direction": "up",
  
  "scores": {
    "1_day": {"actual_price": 318.50, "correct": true, "timestamp": "2026-04-10"},
    "3_day": {"actual_price": 320.00, "correct": true, "timestamp": "2026-04-12"},
    "7_day": {"actual_price": 325.00, "correct": true, "timestamp": "2026-04-16"},
    "30_day": {"actual_price": 335.00, "correct": true, "timestamp": "2026-05-09"}
  },
  "backtest_scores": [
    {"date": "2025-10-09", "correct": true},
    {"date": "2025-11-09", "correct": false},
    ...
  ]
}
```

---

### 2. DETAILED PREDICTION DASHBOARD

#### A. Main Prediction Analytics Page (New Tab)
**Location:** "◎ Predictions" tab (new)

**Displays:**
- Total predictions analyzed
- Live accuracy % (updated daily as predictions age)
- Backtested accuracy % (from synthetic scoring)
- Per-horizon breakdown with charts
- Confidence calibration curve
- Recent predictions table with live scoring

#### B. Prediction History Table
**Columns:**
- Symbol | Date | Horizon | Predicted $ | Predicted % | Current $ | 1-Day | 3-Day | 7-Day | 30-Day | Status
- Color-coded: ✓ correct, ✗ wrong, — pending, ~ preliminary

**Sortable & Filterable:**
- By symbol, date range, horizon, confidence level
- Show only scored predictions, only pending, only backtested

#### C. Analytics Charts
**Chart 1: Accuracy Over Time**
- Line chart showing rolling 20-prediction accuracy
- Separate lines for each horizon
- Identifies if model getting better or worse

**Chart 2: Confidence Calibration**
- Scatter plot: Predicted Confidence % vs Actual Accuracy %
- Perfect calibration = points on diagonal line
- If 70% confidence → 70% accuracy, model is well-calibrated
- Helps identify if confidence scores are trustworthy

**Chart 3: Return Distribution**
- Histogram of predicted returns vs actual returns
- Shows if predictions are too bullish/bearish
- Distribution overlap = good predictions

**Chart 4: Per-Horizon Performance**
- Bar chart: 1-Week accuracy, 1-Month, 1-Quarter, 1-Year
- Shows which horizons work best
- Helps user know which signals to trust

**Chart 5: Win/Loss Ratio**
- Winning predictions vs losing by horizon
- Payoff ratio (avg win / avg loss)
- Win rate % vs breakeven point

#### D. Export Functionality
**CSV Export:**
- All predictions with full data
- Columns: Symbol, Date, Horizon, PredictedPrice, ActualPrice, PredictedReturn, ActualReturn, Correct, Confidence, Features, ModelVersion
- Sortable & filterable before export

**Excel Report:**
- Multiple sheets: Summary, All Predictions, Charts, Analysis
- Professional formatting
- Summary statistics
- Confidence calibration table

---

### 3. MODEL AUTO-IMPROVEMENT SYSTEM

#### A. Outcome Tracking & Analysis
**On each prediction score:**
1. Record the outcome (correct/incorrect)
2. Compare predicted vs actual price
3. Analyze error magnitude
4. Track which features were most predictive

**New fields in prediction log:**
```python
{
  "features_used": {
    "rsi": {"value": 45.2, "rank": 5},      # feature importance rank
    "macd": {"value": 1.2, "rank": 2},
    "momentum_sma": {"value": 0.03, "rank": 1},
    "volatility": {"value": 0.18, "rank": 10},
    ...
  },
  "model_version": "v2.3",
  "outcome": "correct",
  "error_magnitude": 0.02,  # actual vs predicted
  "confidence_vs_outcome": 0.68,  # was confidence well-calibrated?
}
```

#### B. Feature Importance Learning
**Track which features predicted best:**
- For each scored prediction, record feature values
- For correct predictions, boost those feature's importance weights
- For incorrect predictions, reduce those feature's importance weights
- Accumulate learning across all predictions

**Implementation:**
```python
class FeatureImportanceTracker:
    def __init__(self):
        self.feature_importance = {}  # {feature_name: importance_score}
    
    def update_on_outcome(self, features_dict, was_correct):
        """Update feature importance based on prediction outcome"""
        for feature, value in features_dict.items():
            if was_correct:
                self.feature_importance[feature] += 0.1
            else:
                self.feature_importance[feature] -= 0.05
    
    def get_importance_rank(self):
        """Return features ranked by importance"""
        return sorted(self.feature_importance.items(), 
                     key=lambda x: x[1], reverse=True)
```

#### C. Model Versioning & Retraining
**Model versions evolve automatically:**
- **v1.0**: Initial model (baseline)
- **v1.1**: After 20 predictions scored → minor tuning
- **v2.0**: After 50 predictions scored → major retrain
- **v2.1**: Feature weighting adjustments
- **v3.0**: Major architecture changes

**When to retrain:**
- Every 20 new scored predictions
- Monthly automatic retraining
- User-triggered retraining on demand

**What changes during retraining:**
1. Feature weights adjusted based on importance tracking
2. Ensemble member weighting adjusted
3. Confidence calibration updated
4. Ridge alpha (L2 strength) tuned
5. Regime thresholds adjusted

#### D. A/B Testing Framework
**Compare model versions:**
- Keep previous model version
- Deploy new version alongside
- For new predictions, get signals from BOTH models
- Track which version performs better
- Switch to better version when statistically significant

**UI shows:**
```
Model Version: v2.1 (Previous: v2.0)
v2.0 Accuracy: 62.3% (125 predictions)
v2.1 Accuracy: 65.1% (42 predictions)
→ v2.1 leading, more data needed for confidence
```

#### E. Feature Selection Evolution
**Start with top 15 features, evolve over time:**
- Every 30 scored predictions, review feature importance
- Remove features with negative cumulative importance
- Add new feature combinations that appear predictive
- Track which features work best per horizon

**Example log:**
```
Week 1-3: Using RSI, MACD, Momentum, Beta, IV
Week 4: RSI removed (negative importance), Sentiment added
Week 5: IV removed, Volume Profile added
Week 6: Feature set: MACD, Momentum, Sentiment, Volume, Beta
```

---

### 4. PREDICTION LOGGING v2 (Enhanced)

**File: `prediction_logger_v2.py`**

Enhanced tracking with:
- Multiple scoring intervals (1d, 3d, 7d, 30d, etc.)
- Feature tracking for each prediction
- Model version recording
- Outcome analysis
- Confidence calibration data

**New functions:**
```python
def log_prediction_detailed(symbol, predictions, features_dict, model_version)
def quick_score_predictions()
def backtest_predict_accuracy(symbol, lookback_days=252)
def get_prediction_analytics()
def update_feature_importance(prediction_outcome)
def generate_calibration_curve()
def export_predictions_csv()
def export_predictions_excel()
```

---

### 5. DASHBOARD UI UPDATES

**New Tab: "◎ Predictions"**
- Replaces/supplements "◎ Model" tab
- 4 sections:
  1. **Quick Stats**: Total predictions, Live accuracy %, Backtest accuracy %
  2. **Analytics Charts**: 5 charts (accuracy, calibration, returns, per-horizon, win/loss)
  3. **Prediction History**: Sortable, filterable table with live scoring
  4. **Export & Analysis**: CSV/Excel export buttons, detailed stats

**Sidebar Updates:**
- Track Record section expanded
- Shows model version (v2.1)
- Shows feature importance top 5
- Shows next retraining date

**Results Page Updates:**
- "Model learns from outcomes" notice
- Link to view full prediction history
- Feature importance for this prediction shown

---

### 6. CONTINUOUS IMPROVEMENT LOOP

```
1. USER ANALYZES STOCK
   ↓
2. PREDICTION LOGGED
   - Features recorded
   - Model version recorded
   ↓
3. 1 DAY LATER
   - Quick-check score runs
   - Preliminary accuracy shown
   ↓
4. 7/30/90/365 DAYS LATER
   - Full score recorded
   - Feature importance updated
   - Confidence calibration updated
   ↓
5. AFTER 20 SCORED PREDICTIONS
   - Model retrained
   - New version created
   - Better features selected
   ↓
6. LOOP REPEATS
   - User sees improved predictions
   - More data collected
   - Model gets smarter
```

---

## Implementation Timeline

**Phase 1: Accelerated Scoring (1.5 hours)**
- Quick-check scoring system
- Backtest synthetic scoring
- Update prediction_logger.py with intervals

**Phase 2: Prediction Dashboard (1.5 hours)**
- New "Predictions" analytics tab
- History table with live scoring
- 5 analytics charts
- CSV export

**Phase 3: Model Auto-Improvement (1.5 hours)**
- Feature importance tracking
- Model versioning system
- Auto-retraining pipeline
- A/B testing framework

**Phase 4: Polish & Integration (1 hour)**
- Sidebar updates
- Results page integration
- Testing & bug fixes
- Documentation

---

## File Structure

```
Stock Bot/
├── prediction_logger_v2.py         (NEW - enhanced logging)
├── model_improvement.py             (NEW - feature tracking, retraining)
├── prediction_analytics.py          (NEW - dashboard data)
├── app.py                          (UPDATED - new "Predictions" tab)
├── backtester.py                   (UPDATED - synthetic scoring)
└── prediction_log.json             (data file - existing)
```

---

## Key Metrics to Track

1. **Live Accuracy**: % correct predictions as they age
2. **Backtest Accuracy**: % correct synthetic predictions
3. **Confidence Calibration**: Are confidence scores accurate?
4. **Feature Importance**: Which features matter most?
5. **Model Performance Trend**: Is it improving over time?
6. **Per-Horizon Performance**: Which timeframes work best?
7. **Win Rate**: % of winning predictions
8. **Payoff Ratio**: Avg win size / avg loss size

---

## Success Criteria

✅ User can see prediction accuracy within 1-2 days (quick-check)
✅ User has 20+ backtested accuracy points immediately
✅ Model automatically improves version after each 20 predictions
✅ Feature importance tracked and shown to user
✅ Confidence calibration visible in charts
✅ Full prediction history exportable to Excel
✅ Model version shown in UI
✅ A/B testing results visible
✅ User understands which signals are most predictive
✅ System demonstrates clear learning/improvement over time
