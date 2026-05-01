# Anchor: Data Model

**Status:** Draft v1.1 — TRADE threshold and stop methodology locked in
the companion methodology doc; open-questions section pruned.
**Owner:** Marc.
**Purpose:** The contract that every backend, frontend, and analytics surface
builds against. If a field, flag, or state is not in this doc, it is not real.

This doc supersedes ad-hoc decisions. Changes happen *here first*, then in code.

---

## 1. Source of truth

The forward schema is **Supabase Postgres**, defined in
`supabase/migrations/0001_init.sql`. The local SQLite store
(`prediction_store.py`) is legacy and will be retired once the migration is
flipped on. Anything in this doc refers to the Postgres schema unless
explicitly noted.

The seven tables in 0001_init are the foundation:

```
profiles          — 1:1 with auth.users (Supabase Auth)
subscriptions     — Stripe state per user
watchlists        — per-user tracked tickers
paper_portfolios  — per-user paper-trading header
paper_trades      — individual paper trades
predictions       — universal prediction log (all users)
usage_events      — per-user rate-limit + analytics events
```

Most of what we need already exists. The deltas in this doc are about making
the schema express **the four-tier visibility model**, **the model's own
portfolio**, and **the TRADED/PASSED commitment that prevents
cherry-picking**.

---

## 2. Visibility tiers (the gating model)

Every field on every record is classified as one of three visibility levels.
This classification drives every API response shape and every frontend render.

| Tier | Who sees it |
|------|-------------|
| `public` | Anyone, including unauthenticated visitors and shared-link viewers |
| `member` | Any signed-in user (free or paid) |
| `paid`   | Paid subscribers only ($29.99 or $79.99 plan) |

Special case: a free user always sees the **`paid` tier** of their own
predictions. Free-tier visibility limits apply only to other users' predictions.

The four user/visitor contexts:

```
1. Anonymous / social         → sees public fields only
2. Free, viewing own          → sees paid fields (full detail on own data)
3. Free, viewing others       → sees public fields
4. Paid, viewing anything     → sees paid fields
```

This contract must be enforced **at the API layer**, not by hiding things in
the frontend. Never send a paid field to a non-paid client and let the UI
filter it.

---

## 3. Subscription plans (delta against current schema)

`subscriptions.plan` today is `'free' | 'pro'`. The agreed plans are three:

```
'free'             — 3 predictions / month, ads, own-prediction paper trading
'pro_30'           — $29.99 / month, 30 predictions, paper-trade anyone, no ads
'pro_unlimited'    — $79.99 / month, unlimited predictions, API access
```

**Schema change:** Replace the plan check with `('free', 'pro_30',
'pro_unlimited')`. Add a `prediction_quota` integer column with `null` meaning
unlimited:

```sql
alter table public.subscriptions
    add column prediction_quota integer;  -- 3, 30, or NULL (unlimited)
```

Quota is enforced at request time against `usage_events` of `kind='predict'`
within the current calendar month. Hard limit, no rollover.

---

## 4. Predictions table — deltas

The current `predictions` table is good. Five additions and one rename:

### 4.1 New: `traded` — the commitment flag

```sql
alter table public.predictions
    add column traded boolean not null default false;
```

The most important new column. Set at insertion time, **before the outcome
is known**, by the model's own logic (see Methodology doc § TRADE/PASS rule).
Once written, this column is **immutable** — it is the paper-trail proof that
the model committed (or didn't) before the result was knowable.

If the model decides to trade, an associated `model_paper_trade` row is also
created in the same transaction (see § 5).

Backfill for the existing 55 SQLite predictions: `traded = false`. We can't
retroactively claim the model would have traded them; pretending otherwise
breaks the no-cherry-picking guarantee.

### 4.2 New: `verdict` — the four-state public outcome

```sql
alter table public.predictions
    add column verdict text not null default 'OPEN'
    check (verdict in ('OPEN', 'HIT', 'PARTIAL', 'MISSED'));
```

Replaces the existing `final_result` semantics for public consumption.
`final_result` may stay for internal scoring, but `verdict` is what the public
ledger and share cards render.

| State | Meaning |
|-------|---------|
| `OPEN`    | Prediction window is still active |
| `HIT`     | Target reached before expiration |
| `PARTIAL` | Checkpoint level reached but not target |
| `MISSED`  | Expired without hitting checkpoint or target |

`verdict` is recomputed by the scoring worker on every tick. Once set to
non-`OPEN`, it never reverts.

### 4.3 New: rating breakdown — three numbers per prediction

The Track Record page uses three rating reads (Target / Checkpoint /
Expiration). Today these are reconstructed at query time from
`prediction_scores` rows in SQLite. In Postgres, we materialize them on the
prediction row:

```sql
alter table public.predictions
    add column rating_target     text,  -- 'pending' | 'hit' | 'miss'
    add column rating_checkpoint text,  -- 'pending' | 'hit' | 'miss'
    add column rating_expiration text;  -- 'pending' | 'hit' | 'miss'
```

These are member-tier fields. The public sees only `verdict`, which is
derived from these three:

```
verdict = OPEN     if any rating_* = 'pending'
        = HIT      if rating_target = 'hit'
        = PARTIAL  if rating_checkpoint = 'hit' and rating_target = 'miss'
        = MISSED   otherwise
```

### 4.4 Add explicit horizon endpoints

```sql
alter table public.predictions
    add column horizon_starts_at timestamptz not null default now(),
    add column horizon_ends_at   timestamptz;  -- computed from horizon string
```

Required so the scoring worker doesn't have to re-derive expiration from the
horizon enum every tick, and so `verdict` can flip to `MISSED` deterministically.

### 4.5 Add price targets and stops (paid-tier fields)

The model already produces `predicted_return` and `predicted_price`. Paper
trading needs explicit `entry`, `stop`, and `target` levels:

```sql
alter table public.predictions
    add column entry_price   numeric(14, 4),
    add column stop_price    numeric(14, 4),
    add column target_price  numeric(14, 4);
```

`predicted_price` is the model's price *forecast*; `target_price` is the
*tradeable* level the user (or model) hits to exit. They may be equal or not,
depending on the model's risk shaping.

These three are **paid-tier** fields. The public ledger never returns them.

### 4.6 Rename `final_result` → keep or drop

`final_result` becomes redundant once `verdict` exists. Recommendation: keep
it for one release as a duplicate signal, drop in v2 once dashboards no longer
read it.

---

## 5. New table: `model_paper_portfolio`

The existing `paper_portfolios` is **per-user**. The model needs its own,
publicly viewable, single-row portfolio. Two new tables:

### 5.1 `model_paper_portfolio` (singleton header)

```sql
create table public.model_paper_portfolio (
    id                 integer primary key default 1
                       check (id = 1),                  -- enforce single row
    cash               numeric(14, 2) not null default 10000,
    starting_capital   numeric(14, 2) not null default 10000,
    equity_curve       jsonb not null default '[]'::jsonb,
    max_open_positions integer not null default 25,
    created_at         timestamptz not null default now(),
    updated_at         timestamptz not null default now()
);

-- Row-level public read; writes from service_role only.
alter table public.model_paper_portfolio enable row level security;
create policy "model portfolio: public read"
    on public.model_paper_portfolio for select using (true);
```

### 5.2 `model_paper_trades`

```sql
create table public.model_paper_trades (
    id              uuid primary key default gen_random_uuid(),
    prediction_id   uuid not null references public.predictions(id),
    symbol          text not null,
    direction       text not null check (direction in ('LONG', 'SHORT')),
    entry_price     numeric(14, 4) not null,
    exit_price      numeric(14, 4),
    qty             numeric(14, 4) not null,
    notional        numeric(14, 2) not null,            -- entry_price * qty
    opened_at       timestamptz not null default now(),
    closed_at       timestamptz,
    target_price    numeric(14, 4),
    stop_price      numeric(14, 4),
    status          text not null default 'open'
                    check (status in ('open', 'closed_target',
                                      'closed_stop', 'closed_expiry')),
    realised_pnl    numeric(14, 2),
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

create index model_paper_trades_status_idx
    on public.model_paper_trades (status);
create index model_paper_trades_prediction_idx
    on public.model_paper_trades (prediction_id);

alter table public.model_paper_trades enable row level security;
create policy "model trades: public read"
    on public.model_paper_trades for select using (true);
```

These tables are publicly readable to every tier — the model's portfolio is
the proof surface. Writes happen from the model's own service-role worker.

**Invariant:** for every row in `predictions` where `traded = true`, exactly
one row in `model_paper_trades` exists with that `prediction_id`. Enforced by
the worker, verified by a periodic audit job.

---

## 6. Public ledger row — exact field set

The public (anonymous) view of a prediction returns *only* these fields:

```
id
symbol
direction          ('Bullish' | 'Bearish' | 'Neutral')
horizon            ('3d' | '1w' | '1m' | '1q' | '1y')
verdict            ('OPEN' | 'HIT' | 'PARTIAL' | 'MISSED')
traded             (boolean — model put paper money on this)
created_at
horizon_ends_at
```

Member view adds: `confidence`, `regime`, `model_version`, `rating_target`,
`rating_checkpoint`, `rating_expiration`, `actual_return`, `actual_price`.

Paid view adds: `predicted_return`, `predicted_price`, `entry_price`,
`stop_price`, `target_price`, full feature importance JSON.

Every API endpoint that returns predictions has three shapes. Decide which
to return based on the requester's tier, never both-and-filter-client-side.

---

## 7. The three accuracy bands

The Track Record page reports the model's accuracy three ways. All three are
derived from `predictions`:

| Band | Filter |
|------|--------|
| **All calls** | every row where `verdict != 'OPEN'` |
| **Traded calls** (conviction) | `verdict != 'OPEN' and traded = true` |
| **Passed calls** (watched) | `verdict != 'OPEN' and traded = false` |

For each band: hit rate (`HIT / total`), partial rate, miss rate. These are
public-tier numbers — anyone can see them.

The headline portfolio return is computed only from `model_paper_trades` —
the actual realised P&L of trades the model committed to. It is **not** an
accuracy metric, and it is **not** computed across all predictions. One
number, one source, no recomposition.

---

## 8. Migration for existing SQLite predictions

When we cut over to Supabase (executed 2026-04-25 — see `docs/CUTOVER_RUNBOOK.md`
and `methodology_changes` audit log):

1. Migrate all rows into `predictions`. The ETL initially loaded them with
   `user_id = null` (pre-account era), then attributed them to the dev user
   in a follow-up backfill — every legacy prediction was actually triggered
   by Marc via Streamlit / API / `batch_scanner.py` CLI, so `user_id = null`
   misrepresented reality. Backfill recorded in `methodology_changes`
   under `'predictions.user_id (legacy backfill)'`. The model's outputs
   (`predicted_*`, trade plan, `verdict`) were untouched.
2. Set `traded = false` for all of them. We do not retroactively trade.
3. Compute `verdict` from existing `final_correct` + scores:
   - `final_correct = NULL` → `OPEN`
   - `final_correct = 1` → `HIT`
   - `final_correct = 0` → `MISSED`
   - `PARTIAL` is unavailable historically — accept this as a gap.
4. The model's portfolio starts at $10k cash with no historical trades. The
   accuracy bands include these (in "All" and "Passed"); the portfolio
   return does not.

The 40 Bull / 15 Sideways / 0 Bear regime distribution carries over via the
existing `regime` column. Total rows migrated: **56**.

Going forward, every prediction made via the API or Streamlit
auto-attributes to the requesting user (`db._current_user_id()` falls back
to `DEV_USER_ID` when no explicit user is passed). No new prediction
should ever be written with `user_id = null` once accounts are wired up
in Phase 2.

---

## 9. Open questions (to resolve before code)

Most of the original open questions have been resolved in the companion
methodology doc. What's left:

1. **Public ledger rate of new rows.** With unlimited-tier users firing API
   requests, the public ledger could fill with low-quality calls. Confirm
   the `is_public_ledger` filter (canonical universe + confidence ≥ 55 +
   dedupe-within-the-hour) is sufficient, or tighten further.

### Resolved (linked from methodology doc)

- ~~Confidence cutoff for TRADE~~ → **65**, see methodology § 2.3.
- ~~Re-asks of the same ticker~~ → both predictions written; second is
  `traded = false` while the first is open. See methodology § 8.
- ~~"PARTIAL" definition~~ → checkpoint is the midpoint in log-return space
  between entry and target. See methodology § 5.1.
- ~~Closing on early exit~~ → not allowed in v1. Closes only fire on
  target/stop/expiry. See methodology § 4.2.
- ~~Stop-price methodology~~ → clamped 2× ATR₁₄, bounded 1.5%–8% of entry.
  See methodology § 4.4.

---

## 10. What this enables downstream

When Phase 1 lands against this spec, every other surface gets cheap:

- **Track Record:** three accuracy bands + portfolio return, all derived from
  the same table. No special-case queries.
- **Public ledger row:** one query, one shape, no per-tier branching in SQL.
- **Share card:** every field needed (symbol, direction, verdict, traded,
  date) is on the prediction row.
- **Profile page:** filter by `user_id`, render with the same row shape.
- **Dashboard quota meter:** `usage_events` + `subscriptions.prediction_quota`.
- **Paper trading UI:** existing `paper_trades` table; nothing to add.

The spec is deliberately conservative — most additions are simple columns or
small new tables, not architectural rewrites. The expensive part is the
*decision-locking*: once `traded` is committed, the proof surface holds.
