-- ─────────────────────────────────────────────────────────────────────────────
-- Prediqt — Phase 1 initial schema
--
-- Creates the core tables for multi-user Prediqt:
--   profiles          — app-side user extension (1:1 with auth.users)
--   subscriptions     — Stripe subscription state per user (1:1)
--   watchlists        — per-user tracked tickers
--   paper_portfolios  — per-user paper-trading header (cash, equity curve)
--   paper_trades      — individual paper trades
--   predictions       — UNIVERSAL prediction log (all users write here);
--                       is_public_ledger flag gates Track Record visibility
--   usage_events      — per-user rate-limit tracking (analyses, scans)
--
-- Row Level Security is enabled on every table. The default policy: users can
-- only read/write rows where user_id = auth.uid().  `predictions` gets an
-- additional public-read policy for rows where is_public_ledger = true so
-- Track Record is viewable to anyone (including anonymous visitors).
-- ─────────────────────────────────────────────────────────────────────────────

-- ──────────────────────────────────────────────────────────────────────────
-- 1. profiles
-- ──────────────────────────────────────────────────────────────────────────
create table if not exists public.profiles (
    id                    uuid primary key references auth.users(id) on delete cascade,
    email                 text,
    display_name          text,
    stripe_customer_id    text,
    created_at            timestamptz not null default now(),
    updated_at            timestamptz not null default now()
);

comment on table public.profiles is 'App-side extension of auth.users. 1:1.';

alter table public.profiles enable row level security;

create policy "profiles: read own"
    on public.profiles for select
    using (auth.uid() = id);

create policy "profiles: update own"
    on public.profiles for update
    using (auth.uid() = id);

-- Auto-create a profile row when a new auth user is created.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.profiles (id, email)
    values (new.id, new.email);
    insert into public.subscriptions (user_id, plan, status)
    values (new.id, 'free', 'active');
    insert into public.paper_portfolios (user_id, cash, starting_capital, equity_curve)
    values (new.id, 10000, 10000, '[]'::jsonb);
    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_user();


-- ──────────────────────────────────────────────────────────────────────────
-- 2. subscriptions
-- ──────────────────────────────────────────────────────────────────────────
create table if not exists public.subscriptions (
    user_id                    uuid primary key references public.profiles(id) on delete cascade,
    plan                       text not null default 'free',            -- 'free' | 'pro'
    status                     text not null default 'active',          -- 'active' | 'canceled' | 'past_due' | 'incomplete'
    stripe_subscription_id     text,
    stripe_price_id            text,
    current_period_end         timestamptz,
    cancel_at_period_end       boolean default false,
    updated_at                 timestamptz not null default now()
);

alter table public.subscriptions enable row level security;

create policy "subscriptions: read own"
    on public.subscriptions for select
    using (auth.uid() = user_id);

-- Writes come from the Stripe webhook worker running as service_role;
-- service_role bypasses RLS, so we don't write policies for writes.


-- ──────────────────────────────────────────────────────────────────────────
-- 3. watchlists
-- ──────────────────────────────────────────────────────────────────────────
create table if not exists public.watchlists (
    user_id      uuid not null references public.profiles(id) on delete cascade,
    symbol       text not null,
    added_at     timestamptz not null default now(),
    primary key (user_id, symbol)
);

create index if not exists watchlists_user_added_idx
    on public.watchlists (user_id, added_at desc);

alter table public.watchlists enable row level security;

create policy "watchlists: read own"
    on public.watchlists for select using (auth.uid() = user_id);
create policy "watchlists: insert own"
    on public.watchlists for insert with check (auth.uid() = user_id);
create policy "watchlists: delete own"
    on public.watchlists for delete using (auth.uid() = user_id);


-- ──────────────────────────────────────────────────────────────────────────
-- 4. paper_portfolios (header — one per user)
-- ──────────────────────────────────────────────────────────────────────────
create table if not exists public.paper_portfolios (
    user_id            uuid primary key references public.profiles(id) on delete cascade,
    cash               numeric(14, 2) not null default 10000,
    starting_capital   numeric(14, 2) not null default 10000,
    equity_curve       jsonb not null default '[]'::jsonb,  -- [{date, equity}]
    created_at         timestamptz not null default now(),
    updated_at         timestamptz not null default now()
);

alter table public.paper_portfolios enable row level security;

create policy "portfolios: read own"
    on public.paper_portfolios for select using (auth.uid() = user_id);
create policy "portfolios: update own"
    on public.paper_portfolios for update using (auth.uid() = user_id);
create policy "portfolios: insert own"
    on public.paper_portfolios for insert with check (auth.uid() = user_id);


-- ──────────────────────────────────────────────────────────────────────────
-- 5. paper_trades
-- ──────────────────────────────────────────────────────────────────────────
create table if not exists public.paper_trades (
    id                    uuid primary key default gen_random_uuid(),
    user_id               uuid not null references public.profiles(id) on delete cascade,
    symbol                text not null,
    direction             text not null check (direction in ('LONG', 'SHORT')),
    entry_price           numeric(14, 4) not null,
    exit_price            numeric(14, 4),
    qty                   numeric(14, 4) not null,
    opened_at             timestamptz not null default now(),
    closed_at             timestamptz,
    target_close_date     date,
    stop_loss             numeric(14, 4),
    take_profit           numeric(14, 4),
    confidence            numeric(5, 2),
    horizon               text,
    status                text not null default 'open' check (status in ('open', 'closed', 'expired')),
    unrealised_pnl        numeric(14, 2) default 0,
    realised_pnl          numeric(14, 2) default 0,
    meta                  jsonb default '{}'::jsonb,  -- anything else we want to carry
    created_at            timestamptz not null default now(),
    updated_at            timestamptz not null default now()
);

create index if not exists paper_trades_user_status_idx
    on public.paper_trades (user_id, status);
create index if not exists paper_trades_user_opened_idx
    on public.paper_trades (user_id, opened_at desc);

alter table public.paper_trades enable row level security;

create policy "trades: read own"
    on public.paper_trades for select using (auth.uid() = user_id);
create policy "trades: insert own"
    on public.paper_trades for insert with check (auth.uid() = user_id);
create policy "trades: update own"
    on public.paper_trades for update using (auth.uid() = user_id);
create policy "trades: delete own"
    on public.paper_trades for delete using (auth.uid() = user_id);


-- ──────────────────────────────────────────────────────────────────────────
-- 6. predictions (UNIVERSAL prediction log)
-- ──────────────────────────────────────────────────────────────────────────
create table if not exists public.predictions (
    id                   uuid primary key default gen_random_uuid(),
    user_id              uuid references public.profiles(id) on delete set null,  -- who triggered it; null after account delete
    symbol               text not null,
    horizon              text not null,                         -- '3d' | '1w' | '1m' | '1q' | '1y'
    predicted_return     numeric(10, 4),                        -- stored as pct, e.g. 2.35 = 2.35%
    predicted_price      numeric(14, 4),
    confidence           numeric(5, 2),                         -- 0–100
    direction            text,                                  -- 'Bullish' | 'Bearish' | 'Neutral' ...
    model_version        text,
    regime               text,
    -- Guardrails — is_public_ledger computed in Python per our rules:
    --   • symbol in canonical universe (S&P 500 / Nasdaq 100 / Dow 30 / major ETFs)
    --   • confidence >= 55
    --   • not a duplicate of (user_id, symbol, horizon) within the same hour
    is_public_ledger     boolean not null default false,
    -- Scoring fields (filled when horizon elapses)
    final_result         text,                                  -- null | 'hit' | 'miss' | 'pending'
    actual_return        numeric(10, 4),
    actual_price         numeric(14, 4),
    scored_at            timestamptz,
    -- Bookkeeping
    created_at           timestamptz not null default now()
);

-- Hot-path indexes
create index if not exists predictions_user_created_idx
    on public.predictions (user_id, created_at desc);
create index if not exists predictions_public_created_idx
    on public.predictions (created_at desc) where is_public_ledger = true;
create index if not exists predictions_symbol_horizon_idx
    on public.predictions (symbol, horizon);
-- Dedupe helper — supports fast "has this user made this call in the last
-- hour?" lookups without date_trunc in the expression (which Postgres rejects
-- as non-IMMUTABLE on timestamptz columns). Python-side dedupe does:
--   WHERE user_id = ? AND symbol = ? AND horizon = ?
--     AND created_at >= now() - interval '1 hour'
-- which this composite b-tree index serves efficiently.
create index if not exists predictions_dedupe_idx
    on public.predictions (user_id, symbol, horizon, created_at desc);

alter table public.predictions enable row level security;

-- A user reads every prediction THEY triggered, plus every public-ledger row
-- (regardless of who made it) — this is the Track Record visibility.
create policy "predictions: read own or public"
    on public.predictions for select
    using (auth.uid() = user_id or is_public_ledger = true);

create policy "predictions: insert own"
    on public.predictions for insert with check (auth.uid() = user_id);

-- Scoring updates come from the background worker (service_role). No
-- user-facing update policy — predictions are immutable to users.


-- ──────────────────────────────────────────────────────────────────────────
-- 7. usage_events (rate limiting + analytics)
-- ──────────────────────────────────────────────────────────────────────────
create table if not exists public.usage_events (
    id            bigserial primary key,
    user_id       uuid not null references public.profiles(id) on delete cascade,
    kind          text not null,        -- 'analyze' | 'scan' | ...
    meta          jsonb default '{}'::jsonb,
    created_at    timestamptz not null default now()
);

create index if not exists usage_events_user_kind_created_idx
    on public.usage_events (user_id, kind, created_at desc);

alter table public.usage_events enable row level security;

create policy "usage: read own"
    on public.usage_events for select using (auth.uid() = user_id);
create policy "usage: insert own"
    on public.usage_events for insert with check (auth.uid() = user_id);


-- ──────────────────────────────────────────────────────────────────────────
-- 8. Helper view — monthly analysis count per user (powers quota UI)
-- ──────────────────────────────────────────────────────────────────────────
create or replace view public.v_monthly_usage as
select
    user_id,
    kind,
    date_trunc('month', created_at) as month,
    count(*)::int as n
from public.usage_events
group by user_id, kind, date_trunc('month', created_at);

-- Views inherit RLS from underlying tables — usage_events already scopes by
-- user_id, so v_monthly_usage is automatically per-user safe.
