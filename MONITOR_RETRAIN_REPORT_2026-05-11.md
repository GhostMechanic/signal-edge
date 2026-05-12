# Daily Retrain Status Check — 2026-05-11

## Status: should_retrain = **True**

```
Reason:                20+ new scored predictions (121 new)
Model Version:         v4.0
Retrain Count:         3
Live Accuracy:         70.1%
Quick Accuracy:        62.0%
Trend:                 declining (-16.0%)
Expected Improvement:  10.0%
Last Retrain:          v4.0 on 2026-04-22T11:57:50 (acc 66% → 71%, 174 scored)
Errors:                none
```

This is the **third consecutive day** the monitor has flagged `should_retrain=True`. Numbers shifted today after several days of byte-identical reports:

| Date | live_acc | trend | new scored | should_retrain |
| --- | --- | --- | --- | --- |
| 2026-05-07 | 68.0% | declining (-50.0%) | 100 | True |
| 2026-05-08 | 68.0% | declining (-50.0%) | 100 | True |
| 2026-05-09 | 68.0% | declining (-50.0%) | 100 | True |
| 2026-05-10 | 68.0% | declining (-50.0%) | 100 | True |
| **2026-05-11** | **70.1%** | **declining (-16.0%)** | **121** | **True** |

## Per-horizon (current model)

```
1 Week     accuracy 66.7%, confidence 57.4%, calibration error  9.3
1 Month    accuracy 52.8%, confidence 67.6%, calibration error 14.8
1 Quarter  accuracy 63.9%, confidence 73.5%, calibration error  9.6
1 Year     accuracy 60.0%, confidence 78.6%, calibration error 18.6  (overconfident: reduce ~19%)
3 Day      accuracy 73.7%, confidence 57.0%, calibration error 16.7  (underconfident: boost ~17%)
```

## What changed today — important context update

The persistence picture documented in memory (prediction_log_v2.json frozen at 2026-04-24, supabase ImportError swallowed by `log_prediction_v2`) is **partially out of date**:

1. **Source of truth has moved to SQLite.** `prediction_logger_v2.py:40` is now annotated `LOG_FILE = ... # legacy; no longer written`, and `_load_log()` is a compat shim that returns `prediction_store.get_all_predictions()` reading from `.predictions/prediction_log.db`. That DB has 64 prediction analyses / 295 horizon predictions / 857 score rows, with `latest scored_at = 2026-05-11T11:18:21` (i.e. scoring is running daily as horizons elapse).
2. **Two analytics functions exist with the same name.** `auto_retrain.py` imports `get_full_analytics` from `prediction_logger_v2` — that path works and produces today's numbers. `db.py:get_full_analytics()` still hits Supabase via `_service_client()` and dies with the same `ImportError: cannot import name 'create_client' from 'supabase'`. Anything that goes through `db.py` (dashboard/UI surfaces) is still broken.
3. **No new prediction *analyses* since 2026-05-07.** SQLite shows the daily-scan pipeline last wrote on 2026-05-07 (5 entries — matches the claude-test smoke run). Scoring of existing predictions has continued (97 new score rows since 5/8), which is what's driving the "new scored predictions" counter up — not new model training data.

## Recommendation

- **Retrain is technically safe to trigger** — the pool driving the count is real (295 horizon predictions, 857 scores, latest scoring 2026-05-11), so v5.0 would not be a no-op fit on stale data the way it would have been if the JSON file were still the source.
- **But upstream data ingestion is still stalled.** The daily-scan pipeline last produced new prediction analyses on 2026-05-07. Retraining now consumes the current scored pool; ongoing learning won't resume until the scanner runs again.
- **Before retrain**, decide whether the SQLite-backed pool is the canonical training set. If `auto_retrain.run_retrain()` reads via `prediction_logger_v2._load_log()` (SQLite shim), it will retrain on the 295-prediction pool that today's metrics describe. If any retrain code path reaches into `db.get_full_analytics()` or directly into Supabase, that path will hit the same ImportError.

## Action items (prioritized)

1. **Fix the `db.py` Supabase import** so the dashboard/UI analytics path stops crashing. Either rename the local `supabase/` (still present, still shadowing the SDK as a namespace package) or pin/install the SDK in the active environment.
2. **Resume the daily-scan pipeline.** Last prediction analysis is 2026-05-07. Without new analyses landing, retrain frequency will exceed data freshness.
3. **Update memory** — the "persistence pipeline silently broken" snapshot from earlier today (2026-05-11) reflected the JSON-file world. SQLite has been the source of truth at least since `_load_log()` was rewritten. (Memory updated as part of this run.)
