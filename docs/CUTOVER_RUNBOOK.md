# Phase 1 Cutover Runbook

This is the procedure for going from `USE_SUPABASE=false` (legacy) to
`USE_SUPABASE=true` (Phase 1 production). Estimated time end-to-end:
**20–30 minutes**, mostly waiting on the Supabase migrations to run.

Every step has a verify command. Don't move on until verify is green.

---

## Pre-flight

Confirm:

- `.env` already has `SUPABASE_URL`, `SUPABASE_ANON_KEY`,
  `SUPABASE_SERVICE_ROLE_KEY`, `DEV_USER_ID` populated (you did this in
  `SUPABASE_SETUP.md`).
- `USE_SUPABASE=false` (we'll flip it at the end of the cutover, not now).
- The Supabase SQL Editor for your project is open in a browser tab.
- Migration `0001_init.sql` was already applied during initial Supabase
  setup (verify by checking that the seven core tables exist in the
  Table Editor: `profiles`, `subscriptions`, `watchlists`,
  `paper_portfolios`, `paper_trades`, `predictions`, `usage_events`).

If `0001_init.sql` hasn't been applied, do that first per `SUPABASE_SETUP.md`
before continuing.

---

## Step 1 — Apply migrations 0002, 0003, 0004 (in order)

In Supabase SQL Editor, run **each migration as a separate query** in
sequence. Don't combine them — if one fails partway, you want to know
which.

### 1a. Apply 0002 — anchor schema deltas

```
File: supabase/migrations/0002_anchor_deltas.sql
```

Open the file, copy the entire contents, paste into a new SQL Editor query,
**Run**.

**Verify:**

In the SQL Editor:

```sql
-- Check the new columns exist on predictions:
select column_name, data_type
  from information_schema.columns
 where table_name = 'predictions'
   and column_name in ('traded', 'verdict', 'rating_target',
                       'horizon_starts_at', 'entry_price')
 order by column_name;
-- Expect 5 rows.

-- Check the new tables exist + the singleton portfolio row was seeded:
select cash, max_open_positions from public.model_paper_portfolio where id = 1;
-- Expect: 10000.00 | 25.

-- Check the three-tier plan check is in force:
select pg_get_constraintdef(oid)
  from pg_constraint
 where conname = 'subscriptions_plan_check';
-- Expect: CHECK ((plan = ANY (ARRAY['free'::text, 'pro_30'::text, 'pro_unlimited'::text])))
```

### 1b. Apply 0003 — open_model_trade RPC

```
File: supabase/migrations/0003_open_model_trade.sql
```

**Verify:**

```sql
select proname from pg_proc where proname = 'open_model_trade';
-- Expect: open_model_trade
```

### 1c. Apply 0004 — close_model_trade RPC + EOD helper

```
File: supabase/migrations/0004_close_model_trade.sql
```

**Verify:**

```sql
select proname from pg_proc
 where proname in ('close_model_trade', 'append_equity_curve_point')
 order by proname;
-- Expect 2 rows.
```

---

## Step 2 — Dry-run the ETL

Confirms the field mapping looks right against your real 56 SQLite
predictions before any data lands in Supabase. **Doesn't write anything**
— prints a sample row + a verdict distribution.

```bash
source .venv/bin/activate
python scripts/etl_sqlite_to_supabase.py --dry-run
```

**Expected output (matches what we just ran from the sandbox):**

```
SQLite rows to migrate: 56
Built 56 rows. Skipped 0.
Verdict distribution: {'OPEN': 56}
Public ledger eligible: 26 / 56
```

All 56 are `OPEN` because none of the 1-Month canonical horizons have
expired yet (the oldest goes back ~30 days from today). 26/56 land on
the public ledger — the rest are small-caps outside the canonical
universe (S&P 500 + Nasdaq 100 + Dow 30 + 6 major ETFs).

If the row count, verdict distribution, or public-ledger ratio looks
wrong, **stop here** and investigate before proceeding.

---

## Step 3 — Flip USE_SUPABASE=true

Edit `.env`:

```diff
- USE_SUPABASE=false
+ USE_SUPABASE=true
```

Save. **Don't restart the API yet** — we want the ETL to run first so the
Track Record isn't briefly empty.

---

## Step 4 — Real ETL run

```bash
python scripts/etl_sqlite_to_supabase.py
```

**Expected output:**

```
SQLite rows to migrate: 56
Built 56 rows. Skipped 0.
Verdict distribution: {'OPEN': 56}
Public ledger eligible: 26 / 56

Inserted: 56
Failed:   0
```

**Verify in Supabase SQL Editor:**

```sql
select count(*) from public.predictions;
-- Expect: 56.

select count(*)
  from public.predictions
 where is_public_ledger = true;
-- Expect: 26.

select verdict, count(*)
  from public.predictions
 group by verdict
 order by 1;
-- Expect: OPEN | 56.

select count(*) from public.model_paper_trades;
-- Expect: 0 (we don't retroactively trade — anchor § 8).
```

---

## Step 5 — Schedule the scoring worker

The worker has two entry points (`tick` for 5-min closes, `eod` for the
4:15 PM mark-to-market). Pick one of these deployment options.

### Option A — Local crontab (simplest)

`crontab -e` and add:

```cron
# Prediqt scoring worker — every 5 min during US market hours, plus EOD pass.
# Paths assume the project lives at ~/Documents/Claude/Projects/Stock\ Bot.
# Adjust to wherever your venv lives.

# 5-min ticks: 9:30 AM ET → 4:00 PM ET, Mon–Fri.
# (Cron runs in UTC by default on most macOS systems; this assumes your
# system clock matches ET. If you're on UTC, adjust the hour ranges.)
*/5 9-15 * * 1-5  cd "$HOME/Documents/Claude/Projects/Stock Bot" && \
                  ./.venv/bin/python -m scoring_worker tick \
                  >> ~/.prediqt-worker.log 2>&1

# EOD pass: 4:15 PM ET, Mon–Fri.
15 16 * * 1-5     cd "$HOME/Documents/Claude/Projects/Stock Bot" && \
                  ./.venv/bin/python -m scoring_worker both \
                  >> ~/.prediqt-worker.log 2>&1
```

**Verify:**

```bash
crontab -l | grep scoring_worker
tail -f ~/.prediqt-worker.log   # next time a tick fires
```

### Option B — GitHub Actions (cloud cron)

Create `.github/workflows/scoring_worker.yml`:

```yaml
name: scoring-worker
on:
  schedule:
    # 5-min ticks during NYSE hours: 9:30 AM – 4:00 PM ET (cron is UTC).
    # 13:30–20:00 UTC (winter EST, UTC-5).
    - cron: '*/5 13-19 * * 1-5'
    - cron: '0,5,10,15,20,25,30 20 * * 1-5'
    # EOD pass: 4:15 PM ET → 21:15 UTC (winter).
    - cron: '15 21 * * 1-5'
jobs:
  scoring:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - env:
          USE_SUPABASE:               'true'
          SUPABASE_URL:               ${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_ROLE_KEY:  ${{ secrets.SUPABASE_SERVICE_ROLE_KEY }}
        run: |
          # Cron schedules can't easily express "tick OR eod"; pick by hour.
          if [[ "$(date -u +%H)" == "21" ]]; then
            python -m scoring_worker both
          else
            python -m scoring_worker tick
          fi
```

Add `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` as repository secrets
under Settings → Secrets and variables → Actions.

Note: cron's UTC hours need adjusting twice a year for EST/EDT; either
change them with daylight saving, or use a Python time-zone aware
gating check at the top of `tick()`.

### Option C — Manual on-demand only

If you don't want to schedule yet, just run the worker manually whenever
you check the dashboard:

```bash
./.venv/bin/python -m scoring_worker tick    # check open positions
./.venv/bin/python -m scoring_worker eod     # MTM the equity curve
```

This is fine for early iteration — there's nothing time-sensitive yet
because all 56 migrated predictions are still OPEN and the model paper
portfolio starts empty.

---

## Step 6 — Restart the API service

```bash
# If running via uvicorn directly:
pkill -f "uvicorn api.main"
cd "$HOME/Documents/Claude/Projects/Stock Bot"
source .venv/bin/activate
uvicorn api.main:app --port 8000 &
```

If you have a `launchd` plist, systemd unit, or process manager (PM2,
forever, supervisord), restart through that instead.

**Verify:**

```bash
curl -s http://localhost:8000/api/portfolio/summary | python -m json.tool
```

Expected response:

```json
{
  "available": true,
  "cash": 10000.0,
  "starting_capital": 10000.0,
  "open_positions": 0,
  "open_value": 0.0,
  "portfolio_value": 10000.0,
  "total_return_pct": 0.0,
  "equity_curve": [],
  "max_open_positions": 25
}
```

```bash
curl -s 'http://localhost:8000/api/analytics/accuracy-bands?public_only=true' \
  | python -m json.tool
```

Expected (with all 56 currently OPEN):

```json
{
  "available": true,
  "all":     {"total": 0, "hit": 0, "partial": 0, "missed": 0,
              "hit_rate": 0.0, "partial_rate": 0.0, "miss_rate": 0.0},
  "traded":  {...same shape, all 0...},
  "passed":  {...same shape, all 0...},
  "conviction_lift": 0.0
}
```

The bands are zeroed because `verdict != 'OPEN'` filters out everything
right now — predictions need to mature before they show up. Once the
first 1-Month prediction expires (around 2026-05-09), the "all" and
"passed" bands will start to populate.

```bash
curl -s 'http://localhost:8000/api/ledger?limit=5' | python -m json.tool
```

Expected: 5 ledger rows with `verdict: OPEN`, the symbols you actually
predicted, no `confidence` / `entry_price` / `stop_price` / `target_price`
fields (those are paid-tier).

---

## Rollback (if anything looks wrong post-cutover)

If new predictions are misbehaving or the dashboard renders weirdly:

1. Flip `.env` back: `USE_SUPABASE=false`.
2. Restart the API.
3. The legacy SQLite path resumes immediately. Supabase rows from the
   ETL stay in place but are dormant until you flip back.

The migrations themselves are NOT reversible without manual SQL — but
they're additive (new columns + new tables), so leaving them in place
while running in legacy mode is harmless.

---

## What changes after cutover

**For you:**
- `python -m scoring_worker tick` is now a thing you can run.
- The Streamlit dashboard's data source for analytics is hybrid: legacy
  per-horizon detail still comes from SQLite; the new accuracy bands +
  portfolio surface come from Supabase.
- Every new prediction goes through the TRADE/PASS rule. Most won't
  trade until you have predictions on canonical-universe symbols with
  confidence ≥ 65 (the threshold from methodology § 2.3).

**For users (when frontend is wired):**
- Track Record gets the new headline: model portfolio value, the three
  accuracy bands, conviction lift, the public ledger.
- Public visitors see ledger rows with verdict + traded but no trade
  details (paid-tier).

---

## What's left after this cutover

These were on the docket for completion but are not blocking:

- Wire `prediqt-web` Track Record components to consume the new
  `/api/analytics/accuracy-bands`, `/api/portfolio/summary`, and
  `/api/ledger` endpoints. Until then, the new shape lives at the
  endpoints but doesn't render in the UI.
- Per-horizon detail in Supabase (a future migration adds a
  `prediction_horizons` table).
- Auth foundation (Phase 2 — currently DEV_USER_ID-acting-as).

Phase 1 backend itself: done.
