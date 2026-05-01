-- ─────────────────────────────────────────────────────────────────────────────
-- Prediqt — Phase 1 anchor deltas (migration 0002)
--
-- Implements the data model anchor (docs/anchors/01-data-model.md). This
-- migration is purely structural — schema only, no model logic. Worker code
-- that *acts on* these columns ships separately.
--
-- Sections:
--   1. helper: horizon_to_interval()
--   2. predictions: new columns + backfill
--   3. predictions: check constraints (post-backfill) + indexes
--   4. subscriptions: three-tier plan + prediction_quota
--   5. model_paper_portfolio (singleton)
--   6. model_paper_trades
--   7. methodology_changes (public audit log)
--
-- All operations are written to be idempotent — re-running this file should
-- be a no-op. Adding columns uses `if not exists`; constraints are dropped
-- and re-added; tables use `if not exists`; updates only touch rows where
-- the new field is null.
-- ─────────────────────────────────────────────────────────────────────────────


-- ──────────────────────────────────────────────────────────────────────────
-- 1. Helper — horizon string → interval
--
-- Used by the backfill below and (eventually) by the scoring worker to
-- compute horizon_ends_at deterministically. IMMUTABLE so it can be used
-- in expression-based indexes if ever needed.
-- ──────────────────────────────────────────────────────────────────────────
create or replace function public.horizon_to_interval(h text)
returns interval
language sql
immutable
as $$
    select case h
        when '3d' then interval '3 days'
        when '1w' then interval '7 days'
        when '1m' then interval '30 days'
        when '1q' then interval '90 days'
        when '1y' then interval '365 days'
        else null::interval                  -- unknown horizon → null,
                                             -- so horizon_ends_at stays null
                                             -- rather than collapsing to start.
    end;
$$;


-- ──────────────────────────────────────────────────────────────────────────
-- 2. predictions — new columns
--
-- Order: add nullable columns first, backfill from existing data, then in
-- §3 promote to NOT NULL and add check constraints. This avoids errors if
-- the table already has rows (it shouldn't yet — SQLite is still active —
-- but the migration is safe either way).
-- ──────────────────────────────────────────────────────────────────────────

-- The TRADE/PASS commitment. Locked at insertion time; immutable after.
alter table public.predictions
    add column if not exists traded boolean;

-- The four-state public verdict. Derived from rating_* by the scoring worker.
alter table public.predictions
    add column if not exists verdict text;

-- Per-rating breakdown. Member-tier; verdict is the public projection.
alter table public.predictions
    add column if not exists rating_target     text;
alter table public.predictions
    add column if not exists rating_checkpoint text;
alter table public.predictions
    add column if not exists rating_expiration text;

-- Explicit horizon endpoints. horizon_starts_at = when the call goes live;
-- horizon_ends_at = when verdict flips to MISSED if nothing else has hit.
alter table public.predictions
    add column if not exists horizon_starts_at timestamptz;
alter table public.predictions
    add column if not exists horizon_ends_at   timestamptz;

-- Tradeable price levels. predicted_price is the model's forecast; these
-- are the actual trade plan. Paid-tier visibility.
alter table public.predictions
    add column if not exists entry_price  numeric(14, 4);
alter table public.predictions
    add column if not exists stop_price   numeric(14, 4);
alter table public.predictions
    add column if not exists target_price numeric(14, 4);


-- Backfill — best-effort against any existing rows.
update public.predictions
   set traded = false
 where traded is null;

-- verdict from final_result. PARTIAL is unavailable historically (we don't
-- have the intra-window OHLC trace to reconstruct it), so all settled rows
-- map to HIT or MISSED only. New scoring worker fills PARTIAL going forward.
update public.predictions
   set verdict = case
        when final_result = 'hit'  then 'HIT'
        when final_result = 'miss' then 'MISSED'
        else 'OPEN'
   end
 where verdict is null;

-- rating_* — same lossy mapping. Documented gap: historical rows can't
-- distinguish target-vs-checkpoint hits. Both default to the verdict.
update public.predictions
   set rating_target = case
        when final_result = 'hit'  then 'hit'
        when final_result = 'miss' then 'miss'
        else 'pending'
   end
 where rating_target is null;

update public.predictions
   set rating_checkpoint = case
        when final_result = 'hit'  then 'hit'
        when final_result = 'miss' then 'miss'
        else 'pending'
   end
 where rating_checkpoint is null;

update public.predictions
   set rating_expiration = case
        when final_result = 'hit'  then 'hit'
        when final_result = 'miss' then 'miss'
        else 'pending'
   end
 where rating_expiration is null;

-- Horizon endpoints — start = created_at, end = start + horizon.
update public.predictions
   set horizon_starts_at = created_at
 where horizon_starts_at is null;

update public.predictions
   set horizon_ends_at = horizon_starts_at + public.horizon_to_interval(horizon)
 where horizon_ends_at is null;


-- ──────────────────────────────────────────────────────────────────────────
-- 3. predictions — promote to NOT NULL + add check constraints + indexes
-- ──────────────────────────────────────────────────────────────────────────

alter table public.predictions
    alter column traded set not null,
    alter column traded set default false;

alter table public.predictions
    alter column verdict set not null,
    alter column verdict set default 'OPEN';

alter table public.predictions
    alter column rating_target     set default 'pending',
    alter column rating_checkpoint set default 'pending',
    alter column rating_expiration set default 'pending';

alter table public.predictions
    alter column horizon_starts_at set not null,
    alter column horizon_starts_at set default now();

-- horizon_ends_at intentionally left nullable (a malformed horizon string
-- would otherwise block inserts; better to allow null and surface it).

-- Drop and re-add check constraints (idempotent across re-runs).
alter table public.predictions
    drop constraint if exists predictions_verdict_check;
alter table public.predictions
    add  constraint predictions_verdict_check
    check (verdict in ('OPEN', 'HIT', 'PARTIAL', 'MISSED'));

alter table public.predictions
    drop constraint if exists predictions_rating_target_check;
alter table public.predictions
    add  constraint predictions_rating_target_check
    check (rating_target in ('pending', 'hit', 'miss'));

alter table public.predictions
    drop constraint if exists predictions_rating_checkpoint_check;
alter table public.predictions
    add  constraint predictions_rating_checkpoint_check
    check (rating_checkpoint in ('pending', 'hit', 'miss'));

alter table public.predictions
    drop constraint if exists predictions_rating_expiration_check;
alter table public.predictions
    add  constraint predictions_rating_expiration_check
    check (rating_expiration in ('pending', 'hit', 'miss'));


-- Indexes for the new access patterns.
--
-- (a) Track Record's three accuracy bands all filter verdict != 'OPEN'
--     and pivot on `traded`. Partial index keeps it tight.
create index if not exists predictions_settled_traded_idx
    on public.predictions (traded, verdict)
    where verdict != 'OPEN';

-- (b) Scoring worker scans for OPEN predictions whose window has ended.
create index if not exists predictions_open_horizon_ends_idx
    on public.predictions (horizon_ends_at)
    where verdict = 'OPEN';

-- (c) Public ledger commonly filters is_public_ledger + verdict. The
--     existing predictions_public_created_idx covers ordering; this one
--     supports verdict filters within the public set.
create index if not exists predictions_public_verdict_idx
    on public.predictions (verdict, created_at desc)
    where is_public_ledger = true;


-- ──────────────────────────────────────────────────────────────────────────
-- 4. subscriptions — three-tier plan + prediction_quota
--
-- Day-one tiers:
--   free           → 3 predictions / month
--   pro_30         → 30 predictions / month  ($29.99)
--   pro_unlimited  → unlimited + API         ($79.99)
--
-- prediction_quota is null for unlimited; otherwise the integer monthly cap.
-- A trigger keeps it in sync when plan changes so the Stripe webhook worker
-- only has to write `plan` and the quota follows.
-- ──────────────────────────────────────────────────────────────────────────

-- Migrate any existing 'pro' rows to 'pro_30' before adding the constraint.
update public.subscriptions
   set plan = 'pro_30'
 where plan = 'pro';

alter table public.subscriptions
    drop constraint if exists subscriptions_plan_check;
alter table public.subscriptions
    add  constraint subscriptions_plan_check
    check (plan in ('free', 'pro_30', 'pro_unlimited'));

alter table public.subscriptions
    add column if not exists prediction_quota integer;

-- Backfill quota from plan.
update public.subscriptions
   set prediction_quota = case
        when plan = 'free'          then 3
        when plan = 'pro_30'        then 30
        when plan = 'pro_unlimited' then null
   end
 where true;  -- always overwrite; cheap and keeps things consistent

-- Trigger: whenever plan is inserted or updated, sync prediction_quota.
create or replace function public.sync_prediction_quota()
returns trigger
language plpgsql
as $$
begin
    new.prediction_quota = case
        when new.plan = 'free'          then 3
        when new.plan = 'pro_30'        then 30
        when new.plan = 'pro_unlimited' then null
        else new.prediction_quota
    end;
    return new;
end;
$$;

drop trigger if exists subscriptions_sync_quota on public.subscriptions;
create trigger subscriptions_sync_quota
    before insert or update of plan on public.subscriptions
    for each row execute function public.sync_prediction_quota();


-- ──────────────────────────────────────────────────────────────────────────
-- 5. model_paper_portfolio — singleton header for the model's paper account
--
-- Public-readable to every tier (anonymous included). Writes only from the
-- scoring/portfolio worker running as service_role.
-- ──────────────────────────────────────────────────────────────────────────

create table if not exists public.model_paper_portfolio (
    id                  integer primary key default 1
                        check (id = 1),                   -- enforce single row
    cash                numeric(14, 2) not null default 10000,
    starting_capital    numeric(14, 2) not null default 10000,
    equity_curve        jsonb not null default '[]'::jsonb,  -- [{date, equity}]
    max_open_positions  integer not null default 25,
    created_at          timestamptz not null default now(),
    updated_at          timestamptz not null default now()
);

comment on table public.model_paper_portfolio is
    'Singleton row holding the model''s paper-trading state. id is always 1.';

-- Seed the singleton row.
insert into public.model_paper_portfolio (id)
values (1)
on conflict (id) do nothing;

alter table public.model_paper_portfolio enable row level security;

drop policy if exists "model portfolio: public read"
    on public.model_paper_portfolio;
create policy "model portfolio: public read"
    on public.model_paper_portfolio for select
    using (true);


-- ──────────────────────────────────────────────────────────────────────────
-- 6. model_paper_trades — one row per opened trade
--
-- Mirrors the user-side paper_trades table in shape but is shared/public
-- and tied 1:1 to predictions where traded = true. The on-delete-restrict
-- on prediction_id is intentional: we never want a trade orphaned from its
-- prediction, and predictions are immutable, so the restriction is safe.
-- ──────────────────────────────────────────────────────────────────────────

create table if not exists public.model_paper_trades (
    id              uuid primary key default gen_random_uuid(),
    prediction_id   uuid not null references public.predictions(id) on delete restrict,
    symbol          text not null,
    direction       text not null check (direction in ('LONG', 'SHORT')),
    entry_price     numeric(14, 4) not null,
    exit_price      numeric(14, 4),
    qty             numeric(14, 4) not null,
    notional        numeric(14, 2) not null,             -- entry_price * qty
    opened_at       timestamptz not null default now(),
    closed_at       timestamptz,
    target_price    numeric(14, 4),
    stop_price      numeric(14, 4),
    status          text not null default 'open'
                    check (status in ('open',
                                      'closed_target',
                                      'closed_stop',
                                      'closed_expiry')),
    realised_pnl    numeric(14, 2),
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

-- One open trade per symbol max — enforces methodology § 2.1 rule #5.
-- Partial unique index: only enforces uniqueness while a trade is open.
create unique index if not exists model_paper_trades_one_open_per_symbol_idx
    on public.model_paper_trades (symbol)
    where status = 'open';

create index if not exists model_paper_trades_status_idx
    on public.model_paper_trades (status);
create index if not exists model_paper_trades_prediction_idx
    on public.model_paper_trades (prediction_id);
create index if not exists model_paper_trades_opened_idx
    on public.model_paper_trades (opened_at desc);

alter table public.model_paper_trades enable row level security;

drop policy if exists "model trades: public read"
    on public.model_paper_trades;
create policy "model trades: public read"
    on public.model_paper_trades for select
    using (true);


-- ──────────────────────────────────────────────────────────────────────────
-- 7. methodology_changes — public audit log of methodology tweaks
--
-- Per the methodology anchor (§ 2.3): every time a tunable parameter
-- (e.g. TRADE_CONFIDENCE_THRESHOLD) is changed, a row is written here with
-- the rationale. The Track Record page surfaces these as vertical lines on
-- the equity curve so users can see when rules shifted.
-- ──────────────────────────────────────────────────────────────────────────

create table if not exists public.methodology_changes (
    id          uuid primary key default gen_random_uuid(),
    field       text not null,                    -- e.g. 'TRADE_CONFIDENCE_THRESHOLD'
    old_value   text,
    new_value   text not null,
    rationale   text not null,
    changed_at  timestamptz not null default now()
);

create index if not exists methodology_changes_changed_at_idx
    on public.methodology_changes (changed_at desc);

alter table public.methodology_changes enable row level security;

drop policy if exists "methodology: public read"
    on public.methodology_changes;
create policy "methodology: public read"
    on public.methodology_changes for select
    using (true);


-- ─────────────────────────────────────────────────────────────────────────────
-- End of 0002_anchor_deltas.sql
--
-- After running this in Supabase SQL Editor:
--   • predictions has the full anchor-spec shape (12 new columns, 4 check
--     constraints, 3 new indexes).
--   • subscriptions enforces the three-tier plan and auto-syncs quota.
--   • model_paper_portfolio + model_paper_trades exist and are public-readable.
--   • methodology_changes is ready for the first audit entry.
--
-- Next: backend code changes (insert with traded set; scoring worker writes
-- verdict + rating_*; model portfolio worker writes trades). Those go in
-- application code, not here.
-- ─────────────────────────────────────────────────────────────────────────────
