# Phase 3 Testing Guide — Auto-Learning System

## Quick Start Test (5 minutes)

### Step 1: Launch the App
```bash
cd "/Sessions/dazzling-quirky-bohr/mnt/Stock Bot"
streamlit run app.py
```

### Step 2: Run Your First Analysis
1. Enter a stock symbol (e.g., `AAPL`, `MSFT`, `GOOGL`)
2. Select a data period (5Y recommended for fastest first run)
3. Click **Run Analysis**
4. Let it complete (takes 1-2 minutes)

### Step 3: Check Predictions Tab
After analysis completes, click the **🎯 Predictions** tab to see:

---

## What You'll See in Phase 3

### 📊 Predictions Tab Layout

```
┌─────────────────────────────────────────────────────────┐
│ 🎯 PREDICTIONS TAB                                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Prediction Intelligence                                 │
│ Track Record & Model Learning                           │
│ Every prediction is logged, scored, and used to         │
│ improve the model.                                      │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 📈 SUMMARY CARDS (5 columns)                            │
│ ┌─────┬─────┬──────────┬──────────┬──────────┐         │
│ │Total│Scored│Quick Chk │Live Acc │ Model    │         │
│ │Pred │   / │(1d) %    │  %      │Version   │         │
│ │ 0   │  0  │ —        │   —     │  v1.0    │         │
│ └─────┴─────┴──────────┴──────────┴──────────┘         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ ✓ RETRAIN STATUS (NEW IN PHASE 3)                       │
│                                                         │
│ When NOT ready to retrain:                              │
│ ┌─────────────────────────────────────────────────┐    │
│ │ Model is learning from predictions.             │    │
│ │ 20 more scored predictions until next           │    │
│ │ retrain opportunity.                            │    │
│ └─────────────────────────────────────────────────┘    │
│                                                         │
│ When READY to retrain (after 20 scored predictions):   │
│ ┌──────────────────┐ ┌──────────────────┐ ┌─────────┐ │
│ │✓ READY TO RETRAIN│ │📈 EXPECTED       │ │RETRAIN  │ │
│ │                  │ │   IMPROVEMENT    │ │COUNT    │ │
│ │20+ predictions   │ │ +2.3% in live    │ │   3     │ │
│ │now scored        │ │ accuracy         │ │         │ │
│ │                  │ │                  │ │         │ │
│ │Recent accuracy   │ │Features helping: │ │New      │ │
│ │trending up       │ │RSI, MACD cross   │ │version: │ │
│ │                  │ │                  │ │ v1.1    │ │
│ └──────────────────┘ └──────────────────┘ └─────────┘ │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 📊 INSTANT BACKTEST ACCURACY (NEW)                      │
│ Shows how model would have performed historically       │
│ ┌─────────┬─────────┬──────────┬──────────┐           │
│ │1 Week   │1 Month  │1 Quarter │1 Year    │           │
│ │ 58%     │ 56%     │ 52%      │ 48%      │           │
│ │ 12/21   │ 14/25   │ 13/25    │ 12/25    │           │
│ └─────────┴─────────┴──────────┴──────────┘           │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 📈 PER-HORIZON PERFORMANCE                              │
│ ┌──────────┬──────────┬──────────┬──────────┐         │
│ │1 Week    │1 Month   │1 Quarter │1 Year    │         │
│ │  58%     │  56%     │   52%    │   48%    │         │
│ │5 scored  │6 scored  │4 scored  │3 scored  │         │
│ │avg 72%   │avg 70%   │avg 68%   │avg 65%   │         │
│ └──────────┴──────────┴──────────┴──────────┘         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 📊 CONFIDENCE CALIBRATION                               │
│ [Chart: Predicted Confidence vs Actual Accuracy]        │
│ • Perfect diagonal line = well-calibrated              │
│ • Above line = overconfident                           │
│ • Below line = underconfident                          │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 📈 ACCURACY TREND                                       │
│ [Chart: Rolling 10-prediction accuracy over time]       │
│ • Rising line = model improving                        │
│ • Falling line = model degrading                       │
│ • Dashed line at 50% = random coin flip                │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 🤖 MODEL LEARNING STATUS (NEW IN PHASE 3)               │
│                                                         │
│ ┌────────────────────────┐ ┌──────────────────────────┐│
│ │VERSION      v1.0       │ │FEATURE LEARNING          ││
│ │TREND        Improving  │ │                          ││
│ │CALIBRATION  Good       │ │1. RSI           +0.125   ││
│ │NEXT RETRAIN 20 away    │ │2. MACD Cross    +0.098   ││
│ │BEST HORIZON 1 Month    │ │3. MA50 Bull     +0.087   ││
│ │WORST        1 Year     │ │...                       ││
│ │             (48%)      │ │                          ││
│ └────────────────────────┘ │Top 3 features helping    ││
│                            │predictions                 ││
│                            └──────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Testing Scenarios

### ✓ Scenario 1: Fresh Start (0 Predictions)
**What to expect:**
- Summary cards show: 0 total, 0 scored
- Retrain section shows: "Run analyses to start logging"
- Learning status blank
- All charts empty

**Result:** ✓ PASS if cards display correctly

---

### ✓ Scenario 2: First Few Analyses (3-5 Predictions)
**Run analyses on 3 different stocks:**
1. Run AAPL analysis (1 prediction)
2. Run MSFT analysis (1 prediction)  
3. Run GOOGL analysis (1 prediction)
4. Back to AAPL analysis (different period/settings)

**What to expect:**
- Total Predictions: 4
- Scored: 0-1 (1-day scores may appear)
- Retrain section: "19 more scored predictions until retrain"
- Per-horizon performance cards start populating
- Feature importance shows top predictive signals

**Result:** ✓ PASS if:
- Prediction count increments
- Countdown updates (e.g., "19 away", "18 away")
- Cards appear in Model Learning Status

---

### ✓ Scenario 3: Multiple Time Horizons (Optional)
**Test predictions across different time periods:**
1. Run AAPL 3-month analysis → 1 Week, 1 Month, 1 Quarter, 1 Year predictions
2. Run AAPL 5-year analysis → Same horizons scored differently
3. Check Per-Horizon Performance cards

**What to expect:**
- Each horizon card shows individual accuracy (if scored)
- Some horizons may have more data than others
- Better horizons show higher accuracy (usually 1-Month > 1-Year)

**Result:** ✓ PASS if:
- Different horizons have different accuracies
- Cards update as you add analyses

---

### ✓ Scenario 4: Confidence Adjustments (Advanced)
**Verify confidence is being adjusted:**

In the Prediction History table (lower section), look at confidence column:
- After 1st analysis: Base confidence ~65-75%
- After 2nd analysis: May be slightly adjusted
- Check if values stay within 30-95% bounds

**What to expect:**
- Confidence never goes below 30%
- Confidence never exceeds 95%
- Adjustments are applied based on regime

**Result:** ✓ PASS if:
- All confidence values in 30-95% range
- Values don't jump unexpectedly (smooth adjustments)

---

### ✓ Scenario 5: Retrain Readiness Alert (20+ Analyses)
**This requires 20+ analyses to trigger:**

Alternative: Run same stock 20+ times with different date ranges to accumulate 20 predictions fast.

**What to expect:**
- After 20th scored prediction:
  - Retrain section changes to GREEN
  - Shows "✓ Ready to Retrain" alert
  - Displays expected improvement (e.g., "+2.3%")
  - Shows retrain count (first retrain = 1)

**Alert displays:**
```
┌──────────────────┐  ┌──────────────────┐  ┌─────────┐
│✓ READY TO RETRAIN│  │📈 EXPECTED       │  │RETRAIN  │
│                  │  │   IMPROVEMENT    │  │COUNT    │
│20+ predictions   │  │ +2.3% in live    │  │   1     │
│now scored        │  │ accuracy         │  │         │
└──────────────────┘  └──────────────────┘  └─────────┘
```

**Result:** ✓ PASS if:
- Alert appears after 20 scored predictions
- Expected improvement is 0.5-10%
- Retrain count increments

---

### ✓ Scenario 6: Confidence Calibration Chart
**When 3+ predictions are scored:**

Navigate to Confidence Calibration section:
- Red data points show actual calibration
- Dashed diagonal line = perfect calibration
- Line above diagonal = overconfident
- Line below diagonal = underconfident

**What to expect:**
- Initially may be scattered (small sample)
- As predictions accumulate, pattern becomes clearer
- Should trend toward diagonal over time

**Result:** ✓ PASS if:
- Chart renders without errors
- Points/lines are visible
- Updates as more predictions score

---

### ✓ Scenario 7: Accuracy Trend Chart
**When 3+ predictions are scored:**

Navigate to Accuracy Trend section:
- Green line shows rolling 10-prediction accuracy
- Should fluctuate initially then stabilize
- Red dashed line at 50% = random

**What to expect:**
- Line starts appearing after 10 predictions
- Mostly above 50% = model working
- Around 55%+ = edge detected
- 60%+ = strong edge

**Result:** ✓ PASS if:
- Line renders correctly
- Values between 0-100%
- Updates as new predictions score

---

### ✓ Scenario 8: Daily Scheduler
**Verify scheduler is monitoring:**

The scheduler runs daily at 9 AM. To verify it's set:
1. Go to Cowork settings → Scheduled Tasks
2. Look for "monitor-model-retrain"
3. Should show: "At 09:07 AM, every day"

**What to expect:**
- Task appears in sidebar
- Shows schedule correctly
- Status shows "Ready" or next run time

**Result:** ✓ PASS if:
- Task exists
- Schedule is correct
- Can be toggled on/off

---

## Manual Retrain Testing (Advanced)

If you want to test actual retraining logic without waiting for 20 predictions:

```python
# In a Python terminal in the Stock Bot folder:
from auto_retrain import should_retrain_now, get_improvement_metrics, execute_retrain

# Check retrain status
should_retrain, reason = should_retrain_now()
print(f"Should retrain: {should_retrain}")
print(f"Reason: {reason}")

# Get metrics
metrics = get_improvement_metrics()
print(f"Version: v{metrics['current_version']}")
print(f"Retrain count: {metrics['retrain_count']}")
print(f"Expected improvement: {metrics['next_retrain'].get('expected_improvement', '0%')}")
```

---

## Debugging Checklist

### If retrain status doesn't appear:
- [ ] Confirm you're in the 🎯 Predictions tab
- [ ] Check browser console (F12) for JavaScript errors
- [ ] Verify at least 1 prediction has been logged
- [ ] Refresh page (Ctrl+R or Cmd+R)

### If confidence adjustments aren't visible:
- [ ] Check prediction history table — confidence column
- [ ] Run 2+ analyses to see adjustments kick in
- [ ] Look at per-regime performance to see adjustments working

### If scheduler doesn't appear:
- [ ] Check Cowork sidebar for "Scheduled" section
- [ ] May need to restart Cowork app
- [ ] Verify task was created: look for "monitor-model-retrain"

### If features don't show in Learning Status:
- [ ] Run 5+ analyses for feature importance to stabilize
- [ ] Features only appear if they're in top 8 cumulative
- [ ] Longer analysis periods (5Y) build feature data faster

---

## Success Criteria

You'll know Phase 3 is working if:

✅ **After 1 analysis:**
- Predictions tab shows 1 prediction logged
- Summary cards update
- Backtest accuracy displays

✅ **After 5 analyses:**
- Retrain countdown shows ~15 remaining
- Per-horizon cards populate
- Calibration chart starts showing data
- Feature importance appears

✅ **After 20+ analyses:**
- "Ready to Retrain" alert appears (GREEN)
- Expected improvement displays
- Retrain count shows 1+

✅ **Overall:**
- No errors in Streamlit output
- Charts render smoothly
- Data updates without refreshing
- Scheduler monitors automatically

---

## Next Steps After Testing

1. **Keep analyzing stocks** to accumulate predictions
2. **Watch Predictions tab** for retrain readiness
3. **Monitor accuracy trend** — should improve over time
4. **Check daily at 9 AM** for scheduler notifications
5. **Report any issues** with confidence adjustments or retrain logic

---

## Support

If you encounter issues:
1. Check console output for error messages
2. Verify all files compiled: `python3 -m py_compile *.py`
3. Check `.predictions/` folder for data files
4. Review PHASE3_IMPLEMENTATION.md for architecture details

---

**Happy testing! The auto-learning system is now live.** 🚀
