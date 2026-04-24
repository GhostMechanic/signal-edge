# Prediqt — Supabase Setup (Phase 1)

This is the one-time setup to move Prediqt's data from local JSON files to
Supabase Postgres. The app keeps working in file-backed mode the whole time
— you flip a single env var when you're ready to test the new backend.

Estimated time: **10–15 minutes**.

---

## Step 1 — Install new dependencies

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

This adds `supabase` and `python-dotenv`. `supabase-py` is only loaded at
runtime when `USE_SUPABASE=true`, so installing it can't break file mode.

## Step 2 — Create a Supabase project

1. Go to <https://supabase.com> and sign up (free tier is fine for dev).
2. **New project** → pick a region close to you → strong DB password.
3. Wait ~2 min for the project to finish provisioning.

Save these from **Project → Settings → API**:

| What | Where it goes |
|---|---|
| Project URL (`https://xxx.supabase.co`) | `SUPABASE_URL` in `.env` |
| `anon` `public` key | `SUPABASE_ANON_KEY` in `.env` |
| `service_role` `secret` key | `SUPABASE_SERVICE_ROLE_KEY` in `.env` (keep secret!) |

## Step 3 — Run the schema migration

1. In Supabase, go to **SQL Editor** → **New query**.
2. Open `supabase/migrations/0001_init.sql` in this repo.
3. Copy the entire file contents into the SQL Editor.
4. Click **Run**. You should see "Success. No rows returned."

Verify: **Table Editor** in the Supabase dashboard should now show the 7 tables:
`profiles`, `subscriptions`, `watchlists`, `paper_portfolios`, `paper_trades`,
`predictions`, `usage_events`.

## Step 4 — Create a dev user

Until we wire up Supabase Auth (Phase 2), your local Streamlit needs to act
as *some* specific user. Easiest way:

1. **Authentication → Users → Add user → Create new user**.
2. Email: `dev@prediqt.local` (or whatever), password anything.
3. Click create. This triggers the `handle_new_user()` function, which
   automatically creates the matching `profiles`, `subscriptions`, and
   `paper_portfolios` rows.
4. Click the user → copy the UUID → paste into `DEV_USER_ID` in `.env`.

## Step 5 — Fill in `.env`

```bash
cp .env.example .env
# then edit .env and paste the values from Steps 2 & 4
```

Your `.env` should now have:

```
USE_SUPABASE=true
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_ANON_KEY=eyJhbGci...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGci...
DEV_USER_ID=11111111-2222-3333-4444-555555555555
```

## Step 6 — Test the wiring

```bash
streamlit run app.py
```

Open the app, add a ticker to the Watchlist, and refresh the page. It should
persist. Then check Supabase → Table Editor → `watchlists` — you should see
the row there with your `DEV_USER_ID`.

## Step 7 — Roll back to file mode (anytime)

Just set `USE_SUPABASE=false` in `.env` and restart Streamlit. The app
immediately reverts to the JSON-file backend. This is your safety net while
we iterate — you can swap between modes freely.

---

## Backend status check

At any time you can inspect what backend the app thinks it's using:

```python
python -c "from db import backend_info; import json; print(json.dumps(backend_info(), indent=2))"
```

Example output when Supabase is wired up correctly:

```json
{
  "backend": "supabase",
  "use_supabase_flag": true,
  "supabase_url_set": true,
  "supabase_anon_set": true,
  "dev_user_id_set": true,
  "client_ok": true
}
```

If `client_ok` is `false`, the `client_error` field will tell you what's
wrong (missing env var, bad URL, etc.).

---

## What Phase 1 *is not*

- **No login screen yet.** `DEV_USER_ID` in `.env` is the user. Everyone who
  runs this local build against your Supabase project is "you."
- **No Stripe / payments yet.** Everyone is "free" plan. Rate limits are
  scaffolded (see `usage_events` + `count_usage_this_month`) but not yet
  enforced in the UI.
- **Only watchlists are wired** through `db.py` right now. `paper_trader` and
  `prediction_logger_v2` still write to local files even with
  `USE_SUPABASE=true`. That's intentional — we migrate one module at a time
  to keep each change reviewable. Next up: paper trading.

## Phase 2 preview

When you're ready: add Supabase Auth (sign up / sign in screens), swap
`DEV_USER_ID` for `auth.uid()`, migrate the remaining modules, and do a one-
time data migration script to copy existing JSON files into the right
per-user rows.

## Cost expectation

Supabase free tier covers:
- 500 MB database
- 1 GB file storage
- 2 GB bandwidth/month
- 50k monthly active users

For Prediqt's early userbase (dozens → low hundreds), free tier is enough.
Supabase Pro ($25/mo) unlocks larger storage, backups, and removes the
project-pause-after-7-days-of-inactivity behavior. Upgrade when you have
real users or you want point-in-time recovery.
