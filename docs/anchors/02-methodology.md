# Anchor: Methodology

**Status:** Draft v1.2 — all v1 methodology questions resolved.
**Owner:** Marc.
**Purpose:** The locked rules for how the model behaves, scores itself, and
manages its portfolio. Every accuracy stat, every portfolio number, and every
"the model decided to…" event on the site must be explainable by something in
this doc.

This doc is a **public-facing methodology** in spirit. The footnote on the
Track Record page will link here. If a rule isn't written down, the model
isn't allowed to do it.

---

## 1. The integrity contract

Three rules the system never violates. Everything below is in service of these.

1. **No self-generated predictions.** The model never predicts unless a user
   asks. There is no internal scheduler, no auto-suggest, no "it would have
   called this anyway" claim. Every prediction has a `user_id` (or
   `null` for the pre-account era) and a request timestamp.

2. **No look-ahead.** Every decision the model makes about a prediction —
   including whether to trade it — must be made before any future price data
   exists. The TRADE/PASS commitment is locked at insertion time.

3. **No retroactive edits.** Once a prediction is written, its
   `predicted_*`, `confidence`, `entry_price`, `target_price`, `stop_price`,
   and `traded` fields are immutable. Only scoring fields (`verdict`,
   `actual_*`, `rating_*`) update over time, and only by the scoring worker.

Violations of these rules invalidate the Track Record. Treat them as
correctness bugs, not preference issues.

---

## 2. The TRADE / PASS rule

When a prediction is made, the model decides — atomically with the prediction
itself — whether to commit paper money to it. This decision is recorded as
`predictions.traded` (boolean) and is permanent.

### 2.1 The decision rule (v1)

```
TRADE if all of the following hold:
  1. confidence >= TRADE_CONFIDENCE_THRESHOLD          (default: 65)
  2. symbol is in the canonical universe               (S&P 500, Nasdaq 100,
                                                        Dow 30, major ETFs)
  3. model_paper_portfolio.cash >= MIN_POSITION_SIZE   (default: $400)
  4. open positions in model_paper_trades < 25
  5. no existing OPEN position in model_paper_trades
     for this symbol
PASS otherwise.
```

All five must be true. If any is false, `traded = false` and no
`model_paper_trades` row is created.

### 2.2 Why these rules

- **Confidence threshold (65).** Higher than the public-ledger threshold (55)
  because trading is a stronger commitment than visibility. Calibrated against
  the historical accuracy of the model at each confidence band — adjust as
  the model improves.
- **Canonical universe.** Prevents the model from claiming a portfolio of
  obscure tickers where backtest data is thin.
- **Cash constraint.** Honest portfolio mechanics. If we're broke, we don't
  trade.
- **25-position cap.** Prevents the equal-weight portfolio from getting
  diluted past the point of meaningfulness.
- **One position per symbol.** Prevents a single hot ticker from monopolising
  the portfolio when re-asks pile up.

### 2.3 The threshold is a knob, not a guess

**`TRADE_CONFIDENCE_THRESHOLD = 65` (locked, v1).**

Above the public-ledger threshold of 55 (so trading is more selective than
visibility), below 70 (so the 25-slot book actually populates instead of
sitting half-empty in slow weeks). The number is committed publicly here
and lives in config.

A "data-driven" threshold from the existing 55-prediction sample would be
calibration to noise — buckets are too thin. We commit to a defensible
round number, document it, and recalibrate when there's enough data to
recalibrate honestly.

**Recalibration trigger.** At the 200-prediction mark (or sooner if either
of the failure signals below trips), run the calibration:

| Signal | Meaning | Action |
|--------|---------|--------|
| `hit_rate(traded) - hit_rate(all) ≤ 0` | The model isn't picking its winners — the threshold isn't doing work. | Raise it. |
| Average open positions < 5 | Portfolio is too sparse to tell a story. | Lower it. |
| Neither tripped at 200 | Threshold is in the right band. | Hold; reassess at 500. |

When the threshold changes, **a record is written to the
`methodology_changes` audit log** with the old value, new value, timestamp,
and rationale. The public Track Record page surfaces a vertical line on
the equity curve at each change. No silent re-tuning.

```sql
create table public.methodology_changes (
    id          uuid primary key default gen_random_uuid(),
    field       text not null,            -- e.g. 'TRADE_CONFIDENCE_THRESHOLD'
    old_value   text,
    new_value   text not null,
    rationale   text not null,
    changed_at  timestamptz not null default now()
);
```

### 2.4 Continuous retraining

The model gets *retrained* on its own scored predictions, not just
*adjusted* by them. Two loops run in parallel:

**Daily refresh of global adjustments.** The 4:15 PM ET EOD pass
runs `ModelImprover.analyze_prediction_outcomes()` over the full
prediction log. This produces a fresh
`.predictions/model_adjustments.json` with feature_boosts (which
features have been predictive recently), regime_adjustments
(which regimes the model is over- or under-confident in),
confidence_bias (global calibration drift), and horizon_weights
(which horizons need confidence pulled up or down). The model
loads this file at the start of every `train()`, so adjustments
flow into the next retrain automatically.

**Per-stock retrain on fresh evidence.** Each symbol tracks how
many of its predictions have been scored since its last retrain.
When the delta crosses `PER_STOCK_RETRAIN_THRESHOLD = 3` (locked,
v1), the next call to `model.train()` for that symbol bypasses
the 12-hour cache and force-rebuilds the predictor. This means
heavily-asked names retrain at the cadence the data justifies,
without waiting for a global threshold or cache TTL. The marker
lives at `learning_engine.state.retrain_markers[symbol]` and
survives analysis-pass rebuilds.

The 3-prediction threshold prioritizes fresh weights over compute
cost. It's tight enough that heavily-asked names retrain roughly
every other day during active use (a 5-horizon model accumulates
~5 scored predictions per day per actively-asked stock), and
deliberately tighter than the global retrain trigger (which fires
every 20 scored predictions across all stocks) so that per-stock
freshness isn't gated by aggregate volume. Looser settings (5, 10)
save compute but leave weights stale across more new evidence than
is worth the savings under the integrity posture.

A **manual sweep** is available via
`python -m scoring_worker retrain_pending`: walks every stock
above the threshold and rebuilds its predictor in a single batch.
Useful right after a methodology change lands — re-trains
everything pending against the new adjustments JSON without
waiting for users to ask.

Per-prediction integrity holds throughout: every prediction stamps
the `model_version` it was made under, retrains bump the version,
and old predictions stay scored against the version that made them.
A retrain *cannot* rewrite the past — it only changes what the
*next* prediction looks like.

---

## 3. Position sizing

Equal weight across the open book.

```
position_size = min(
    cash,
    starting_capital / max_open_positions
) = min(cash, $10k / 25) = $400 baseline
```

When the cap of 25 isn't full, the model still uses the *baseline* size
($400), not a fraction of available cash. This means cash sits idle when
fewer than 25 positions are open. **Idle cash is not a bug** — it's a
constraint that mirrors how a disciplined trader wouldn't double-down just
because the book is short.

If the model has $400 cash and 24 open positions, it can trade once more.
If it has $250 cash and 24 open positions, it cannot trade — the prediction
gets `traded = false` even if everything else qualifies.

**Open question:** should the position size scale with confidence
(higher confidence = larger position)? Day-one answer: **no**. Equal weight
keeps the size dimension out of the comparison so direction-picking is
isolated. We can add confidence-weighting later as a second variant
("model: equal" vs. "model: confidence-weighted") and let the data say
which lets the user beat it.

---

## 4. Position management — opens and closes

### 4.1 Opens

A trade opens at the moment the prediction is made, at the symbol's last
known price (`entry_price`). No slippage modeling for v1 — the simplification
is documented and stays consistent across the model and user paper portfolios.

The trade record (`model_paper_trades`) is written in the **same transaction**
as the prediction. Either both succeed or neither does.

Direction:
- `Bullish` → LONG
- `Bearish` → SHORT
- `Neutral` → no trade (overrides the TRADE rule)

The trade plan is `(entry_price, stop_price, target_price)`. All three are
locked at open time and recorded on the prediction row. See § 4.4 for how
stop and target are placed.

### 4.2 Closes (model)

This section governs `model_paper_trades` only. User paper trades —
the per-user `paper_trades` table — are not bound by it; see § 4.6.

A model position closes on exactly one of three triggers:

| Trigger | Condition | `status` |
|---------|-----------|----------|
| Target hit | Price crosses `target_price` (any time during window) | `closed_target` |
| Stop hit   | Price crosses `stop_price` (any time during window)   | `closed_stop`   |
| Expiry     | `now() >= horizon_ends_at` and neither above           | `closed_expiry` |

The model **cannot close a position early** for any other reason — no
"changed my mind," no "the regime shifted." Removing close-discretion removes
the second cherry-picking lever after TRADE/PASS.

### 4.3 The scoring worker

A background worker runs on a fixed schedule:

| When | What |
|------|------|
| Every 5 minutes during US market hours (9:30 AM – 4:00 PM ET) | Live close-checks |
| Once at 4:15 PM ET (EOD pass) | Capture close prices, append to equity curve |
| After-hours, overnight, weekends | No runs |

On each tick:

1. Pulls every `model_paper_trades` row where `status = 'open'`.
2. Fetches the latest price for the symbol.
3. Checks: target hit? stop hit? expiry passed?
4. If any: closes the position, computes `realised_pnl`, updates
   `model_paper_portfolio.cash`.
5. Recomputes `predictions.verdict` and `rating_*` for the linked prediction.

The 4:15 PM EOD pass additionally:

6. Computes the day's mark-to-market: `cash + Σ(qty × close_price)` for every
   still-open trade.
7. Appends `{date, equity}` to `model_paper_portfolio.equity_curve`.

Roughly 78 worker invocations per trading day (78 × 5 min = 6.5 hours, plus
the EOD pass). Idempotent: running twice on the same tick produces the same
result. Runs as service_role (bypasses RLS) and is the only writer to
`model_paper_trades` and `model_paper_portfolio`.

### 4.4 Stop and target placement

Both levels are computed at open time and **immutable** for the life of the
trade. They go on the `predictions` row (`entry_price`, `stop_price`,
`target_price`) and on the matching `model_paper_trades` row.

**Target.**

```
target_price = predicted_price
```

The model's own price forecast is the trade's exit-on-success level. No
shaping, no haircut. If the model says NVDA goes to $880, that's the
target. This keeps the model's directional read and the model's tradeable
plan as the same number — there's no second-layer "the model said X but
we'll target Y" softening.

**Stop.** Volatility-aware, with bounds.

```
atr_distance     = 2.0 × ATR₁₄              # 14-day Average True Range
floor_distance   = 0.015 × entry_price      # 1.5% — protects against whipsaw
ceiling_distance = 0.080 × entry_price      # 8% — caps single-trade damage

stop_distance    = clamp(atr_distance, floor_distance, ceiling_distance)

stop_price = entry_price − stop_distance    (LONG)
           = entry_price + stop_distance    (SHORT)
```

Why this rule:

- **Volatility-aware.** A 1.5%-of-entry stop on TSLA gets fired by an
  intraday wick; the same percentage on KO is too wide to do work. ATR
  scales the stop to the symbol's actual movement, so high-vol names get
  more breathing room and low-vol names get tighter stops. This is the
  only single-knob rule that's fair across the canonical universe.
- **Bounded.** The 8% ceiling prevents one absurdly volatile name from
  sinking a single position into a 20% loss when ATR spikes. The 1.5%
  floor stops ultra-low-vol names from getting a stop so tight it fires
  on a quote glitch. The clamp protects the math at both ends.
- **k = 2.0 is the standard choice.** Tighter (k = 1.5) and stops fire on
  intraday noise; wider (k = 3.0) and you're carrying losses too long
  before getting out. k = 2 is the conventional middle.
- **Reproducible from public data.** ATR₁₄ is computable from any OHLC
  feed. Anyone auditing a historical trade can recompute the stop level
  exactly. Methodology footnote becomes one sentence.

**The methodology footnote (verbatim, for the Track Record page):**

> *Stops are placed at 2× the 14-day Average True Range from entry,
> bounded between 1.5% and 8% of entry price, in the opposite direction
> of the call. Targets are the model's predicted price. Both are locked
> at open time.*

**Tunable parameters.** Three knobs, all logged through
`methodology_changes` if they ever move:

| Parameter | v1 value | When it moves |
|-----------|----------|----------------|
| ATR multiplier (`k`) | 2.0 | If stop-hit rate is materially out of line with backtest expectations |
| Floor pct | 1.5% | If we see floor-bound stops getting fired by quote noise |
| Ceiling pct | 8% | If a wider band is needed to give high-vol names a fair window |
| ATR period | 14 | Conventional. Don't move without a reason. |

### 4.5 User-side early close

The no-early-close rule in § 4.2 binds `model_paper_trades` — it's the
guardrail that prevents the model cherry-picking which of its calls to
"keep." Users are not the model, and their paper book is their own. They
may close any open paper trade at any time.

**Equity early close.** Close fills at the price the user submits with
the close request. The book mirrors what they'd see on the asking
flow — same one-price proxy, same no-slippage convention as opens.
Existing behavior, locked here for completeness.

**Option early close.** Closes fill at the live mid-price of the spread,
fetched server-side at the moment of close. Per leg:

```
priced_at = (bid + ask) / 2     when both sides are present and ask >= bid       → source = mid
          = lastPrice           when bid/ask is unusable but lastPrice > 0       → source = last
          = max(0, S − K) CALL  when the chain has the strike but the quote is   → source = intrinsic
          = max(0, K − S) PUT     all-zero (deep OTM with pennies of time value)
          = unavailable         when the chain fetch fails or the strike isn't   → source = unavailable
                                  on the chain
```

Legs net by sign — long legs add, short legs subtract. The position-
level pricing source is the *worst* of any leg's source
(`mid > last > intrinsic > unavailable`). If any leg lands at
`unavailable`, the close is **refused** with a clear error: silent
fallbacks to fictional prices would settle trades at numbers we can't
defend, which the integrity contract (§ 1) forbids.

`source = intrinsic` is reserved for the narrow case where the chain
confirms the strike *exists* but every quote channel reads zero (deep-
OTM strikes with pennies of time value the feed doesn't surface). It's
defensible there because the option really is worth ~zero. **Early close
at `intrinsic` is paused on the user-side UI** — the number ignores time
value, and we don't fill at numbers that ignore time value just because
they're computable. The user sees the position with `—` for "now" and
P&L until the quote feed comes back; they can hold to expiry for
automatic settlement either way.

**Snap-to-chain.** yfinance only carries chains for specific expiry
dates (third-Friday LEAPS plus weeklies). The asking flow's natural
output is `expiry_date = today + N days` which often falls between two
real chain dates. To keep the recorded date honest:

- *At trade-open.* The open path snaps the requested expiry to the
  nearest available chain date with no tolerance — the closest real
  date wins, and the original guess is preserved as
  `instrument_data.original_expiry_date` for audit. The receipt
  surfaces both: "asked for 365d, filled at expiry 2027-04-16 (~351d)."
  If yfinance has no chains for the symbol, the open is **refused**
  rather than persist a fake date the early-close path can't honor.
- *At pricing time.* The pricer tries the recorded expiry first. If it
  isn't on the live chain (legacy trades from before snap-at-open
  landed), the pricer snaps to the closest within ±7 days. Beyond that
  the position lands at `source = unavailable` — the recorded date is
  too far off any real chain for us to honestly stand behind a fill.
  When pricing snaps, the receipt records both the recorded and
  chain-used expiries; the UI calls out the delta.

**Strike snap.** Same class of problem as expiry, one level down. The
asking flow's `_strike_choices` produces theoretical strikes (spot ±
fractional-sigma, ATM rounded, etc.) but real chains only carry
specific strike intervals — typically $5 or $10 around the money,
sometimes $1 for near-dated weeklies. A bear put spread on AMZN at
theoretical $263/$247 won't fill because neither strike exists on the
live chain. The fix mirrors the expiry case in three places:

- *At asking time.* `options_pricer.snap_strategy_to_real_chain` runs
  inside `generate_options_report` once a strategy is built. It
  snaps expiry, snaps every leg's strike to the nearest real strike
  on that chain (within `STRIKE_SNAP_TOLERANCE_PCT = 10%`), fetches
  live mids per leg, replaces theoretical premiums, and recomputes
  numerics via `recompute_strategy_numerics`. The persisted strategy
  carries real strikes and real premiums; the original (asked-for)
  strikes survive on each leg's `original_strike` for audit. The
  user sees real fillable values on /prediqt instead of theoretical
  ones.
- *At trade-open.* `price_legs_at_open` snaps strikes the same way
  as defense in depth, so legacy strategies (or any case where the
  asking-flow snap was skipped or failed) still produce a fillable
  trade. The leg dict gets `snapped_strike` + `original_strike` so
  the receipt surfaces the drift the same way it surfaces premium
  drift today.
- *Refusal beyond tolerance.* If a target strike has no chain match
  within 10%, the system **refuses** rather than fill at a wildly
  different strike. The strategy is marked `tradable = false` with
  an `untradeable_reason` and the open path returns a clear error.

The 10% tolerance is wide enough to absorb the typical $5/$10 strike
intervals (a $260-strike target maps to a $265 chain strike at ~2%
drift) but narrow enough to refuse genuinely incompatible asks. As
with all chain-snap behavior: the rule is in math, not in vibes —
the closest strike within the math wins, and the rest fail closed.

**Live entry premium at open.** The asking flow estimates per-leg
premiums from a Black-Scholes model. At trade-open the system replaces
each leg's estimated premium with the live mid (or last) from the
options chain at the snapped expiry, then `recompute_strategy_numerics`
in `options_analyzer` reshapes the strategy's `cost_per_contract`,
`max_loss_per_contract`, `max_profit_per_contract`, and breakevens
around the live numbers. The asking-flow estimate is preserved on
`instrument_data.planned.*` for audit. Receipt surfaces the drift:
"quoted at $X, filled at $Y (±Z%)". An open is **refused** if any leg
can't fill at mid or last — intrinsic-at-spot is a defensible
valuation for an already-open position with all-zero quotes, but it
isn't an honest fill price for opening one.

**Legacy trades are not re-anchored.** Trades opened before the live
entry-premium path landed keep their original Black-Scholes-based
cost basis. We do not backfill their persisted entry to today's mid
because that would compress their P&L to ~zero on day one and
retroactively claim the system filled them at honest prices it never
actually saw. The honest record is the one the system actually
persisted at open time, even when that record is an estimate; closing
those trades surfaces (live close − persisted entry) as P&L, which is
what that comparison really is. New trades from the live-entry path
forward are clean.

The close receipt persists `bid`, `ask`, `last`, `used_price`, and
`source` per leg on `paper_trades.instrument_data.close_quote`. Anyone
auditing a closed user option trade can reproduce the fill from the
chain at `fetched_at`.

The model is not affected by this section — § 4.2 stands as written.

**Greeks: display, not trade.** The user-facing expansion panel
surfaces Black-Scholes delta, theta, and the chain-reported implied
volatility per leg, plus a payoff-at-expiry diagram. These are
*display only*. Trade fills (open and early-close) are determined by
live mid (or `last` fallback per the source priority above), never
by greek-derived estimates. The two pricing paths are kept separate
so the system can't accidentally start crossing the user at a
modeled number it can't honor: the screen offers context, the books
trade prices.

Greeks are computed via standard Black-Scholes from `(spot, strike,
chain IV, days-to-expiry, risk-free rate ≈ 4.5%)`. They render with
an italic footnote on the live-quote panel ("Greeks shown for
context only. Trade fills cross at the live mid (or last fallback),
never at greek-derived estimates.") so the constraint stays visible
to the user, not just to the doc.

---

## 5. The four-state verdict

How `verdict` is computed from price action over the prediction window:

```
HIT      ← price touched target_price (in the predicted direction)
            before horizon_ends_at
MISSED   ← horizon_ends_at passed without touching target or checkpoint
PARTIAL  ← horizon_ends_at passed; checkpoint touched, target not
OPEN     ← otherwise
```

### 5.1 Defining the checkpoint

The checkpoint is **the midpoint, in log-return space, between entry and
target**:

```
checkpoint_return = (predicted_return) / 2
checkpoint_price  = entry_price * exp(checkpoint_return)
```

Touching `checkpoint_price` in the predicted direction at any time during
the window counts as the checkpoint being hit.

This rule is:
- Symmetric (long and short use the same logic)
- Independent of horizon length
- Computable from existing fields (no extra storage)

### 5.2 What "touched" means

Intraday OHLC granularity. A daily high (for LONG) or low (for SHORT)
crossing the level counts as a touch, even if the close reverts. This
matches how a real stop or limit order would have filled.

### 5.3 What `MISSED` does and doesn't mean

`MISSED` does not mean "wrong direction." It means "didn't reach target or
checkpoint within the window." A LONG prediction where the stock went up but
not enough is `MISSED`, same as one where the stock went down. This is
deliberate — the public verdict measures whether the call *paid off*, not
whether the directional sign was right.

(The `direction_correct` field on `prediction_scores` still tracks pure
directional accuracy for member-level diagnostics. The public surface
intentionally doesn't.)

---

## 6. The three accuracy bands

For the Track Record page. All three are computed from `predictions`:

| Band | Filter | What it answers |
|------|--------|-----------------|
| **All** | `verdict != 'OPEN'` | How accurate is the model overall? |
| **Traded** (conviction) | `verdict != 'OPEN' and traded = true` | Does its commitment pick winners? |
| **Passed** (watched) | `verdict != 'OPEN' and traded = false` | Is it correctly avoiding losers? |

Each band reports `HIT %`, `PARTIAL %`, `MISSED %`. The numbers must add to
100% (every settled prediction has exactly one terminal state).

### 6.1 The meta-judgment metric

A single derived number: **conviction lift** =
`hit_rate(traded) - hit_rate(all)`. Positive means the model's TRADE/PASS
decision is informative. Negative means it's better at picking direction
than picking which of its picks to back.

This number is published. It can be embarrassing. That's the point.

---

## 7. The portfolio return

The headline portfolio number on every public surface is the same number:

```
portfolio_value = model_paper_portfolio.cash
                + Σ(open_position.qty × current_price)
return_pct      = (portfolio_value - starting_capital) / starting_capital
```

One source. Computed live from `model_paper_trades` and the latest prices.
This is the number that ticks intra-day on the dashboard.

**Equity curve.** Per-day mark-to-market only. One point per US trading
day, written by the 4:15 PM EOD pass (see § 4.3):

```
{ date: 2026-04-25, equity: cash + Σ(qty × eod_close_price) }
```

Per-day was chosen over per-close because per-close gaps would tell a
misleading story — sparse closes look like inactivity, dense closes look
like a busy week. Per-day MTM matches how every fund equity curve is read,
and means the curve has a regular rhythm independent of when trades close.

Closed trades' realized P&L hits `cash` immediately on close, but the
curve only persists at EOD. The intraday number on the dashboard reflects
unrealized P&L; the curve itself does not gain a new point until 4:15 PM.

**Benchmarks:** alongside the portfolio number, two reference numbers are
shown:
- `S&P 500 return over same window` — the obvious comparison.
- `Equal-weight all-predictions return` — what the portfolio would be if the
  model had traded *every* prediction (no TRADE/PASS curation). The gap
  between this and the actual portfolio is the value of the model's
  meta-judgment, in dollars.

---

## 8. Cross-user behavior

### 8.1 Re-asks and concurrency

#### Same user, same symbol+horizon, same UTC day → **dedupe**

When the same user asks about the same `(symbol, horizon)` within the same
UTC day, only the first ask is written. Subsequent re-asks return the
existing `prediction_id` and skip writing a duplicate row entirely. The
model still runs fresh inference and the user still sees live numbers in
the response (price, fundamentals, sparkline can drift intraday) — but
the *committed call* on the ledger is the original one.

This prevents scoring imbalance: a single re-asked call would otherwise
land two scored outcomes on a single market move, distorting per-stock
and per-horizon accuracy stats.

Granularity:

- **Supabase mode** (`USE_SUPABASE=true`, production): dedupe key is
  `(user_id, symbol, horizon_code, today_utc)`. Different horizons of the
  same symbol on the same day are still separate predictions — the user
  asked for distinct calls.
- **File mode** (legacy SQLite, dev only): dedupe key is
  `(symbol, today_utc)` because the legacy schema stores all five horizons
  under one `prediction_id` and has no `user_id` column. Slightly broader
  than production semantics; acceptable for dev.

The implementation lives in `db.find_todays_prediction()` (called by
`prediction_logger_v2.log_prediction_v2()` before any compute). The
existing 1-hour dedupe inside `_is_public_ledger` (a visibility-only
gate) remains as defense-in-depth on the ledger flag.

#### Different users, same symbol+horizon, same day → both written

When two **different** users ask about the same symbol within the same
horizon window:

1. Both predictions are written. Both get rows in the public ledger
   (subject to `is_public_ledger`).
2. Only the **first** to be processed becomes a `traded = true` row, *if*
   it passes the TRADE rule. The second is `traded = false` because § 2.1
   rule #5 (no existing open position for this symbol) fails.
3. Both users can paper-trade their own predictions independently in
   *their* `paper_portfolios`. The model's single position doesn't
   constrain user trading.

This means the model's portfolio is more selective than the prediction
log. That's expected — the model has 25 slots; the public ledger doesn't.

(Phase 1 dev mode runs as a single user via `DEV_USER_ID`, so the
multi-user case is currently theoretical. When auth ships and real
accounts arrive, we may revisit whether cross-user duplicates of the same
exact call should also dedupe — for now the per-user rule above is the
locked behavior.)

### 8.2 Deleted users

When a user deletes their account:

| Object | Behavior |
|--------|----------|
| `paper_portfolios` (user's personal) | Cascade-deleted with the user. |
| `paper_trades` (user's personal)     | Cascade-deleted with the user. |
| `predictions` they triggered          | Stay. `user_id` flips to `null` (per `ON DELETE SET NULL` in 0001 schema). |
| `model_paper_trades` from those preds | Untouched. Continue to close on their own rules; contribute to portfolio P&L like any other trade. |

The principle: **the user's personal data leaves with them; the model's
public commitments don't**. Predictions and model trades are model-side
public records, not user-side personal data. The "no deletes, no edits,
no hiding" claim covers the model's track record as much as it covers
individual predictions — and the model's track record is exactly what
breaks if user-side deletions could rewrite it.

In the public ledger, an orphaned prediction (one whose `user_id` is null)
renders the asker as **"User deleted"** — see voice doc § 8 for the
verbatim copy. Honesty over euphemism: the prediction stays visible, the
fact that the user is gone is stated plainly, the model's call is
unaffected.

---

## 9. The model's TRADE/PASS audit

Every prediction has a `traded` boolean. Every closed prediction has a
verdict. The cross-tab is publishable:

```
              | HIT   | PARTIAL | MISSED |
TRADED  (T)   |  T_h  |  T_p    |  T_m   |
PASSED  (F)   |  F_h  |  F_p    |  F_m   |
```

Six cells. The whole "did the model know which of its calls to back" question
lives in this 2×3 grid. We render this somewhere on the Track Record page.

A skeptic should be able to download the raw prediction log (paid feature,
or aggregated for free) and reproduce these six numbers. Methodology only
holds if the data backs it.

---

## 10. What's deliberately not in v1

Listing these explicitly so they don't sneak in:

- **Slippage / commissions modeling.** Out. Real-money realism comes later.
- **Confidence-weighted position sizing.** Out. Equal weight only.
- **Multi-horizon trading.** Each prediction has one horizon, one trade. The
  model does not "scale in" or "average down."
- **Sector / factor neutrality constraints.** Out. The portfolio is what it
  is; no rebalancing rules beyond § 2.
- **Discretionary closes.** Out, by design (see § 4.2).
- **Manual model overrides.** Out. The model does not "skip" predictions
  for any reason not encoded in § 2.1.

When any of these become v2 candidates, they go through `methodology_changes`
with a public rationale. No silent additions.

---

## 11. Open questions to resolve before code

All v1 methodology questions are now resolved. Future revisions will go
through `methodology_changes` per § 2.3.

### Resolved (locked in v1.2)

- ~~Scoring-worker cadence~~ → **every 5 min during US market hours +
  4:15 PM EOD pass.** See § 4.3.
- ~~Equity curve cadence~~ → **per-day mark-to-market only.** See § 7.
- ~~Deleted users~~ → **personal data goes; predictions and model trades
  stay; orphaned predictions render as "User deleted".** See § 8.2.

### Resolved (locked in v1.1)

- ~~TRADE_CONFIDENCE_THRESHOLD initial value~~ → **65**, see § 2.3.
  Recalibration trigger at 200 predictions.
- ~~Stop-price methodology~~ → **clamped 2× ATR₁₄**, see § 4.4. Bounded
  1.5%–8% of entry.

---

## 12. The suppression gate ("No clear read")

Before TRADE/PASS gets a vote (§ 2), the model decides whether to commit
to the prediction at all. When this gate fires, the surface renders a
"No clear read" panel and the call is **not** written to the public
ledger and **does not** count toward the user's asking history. The
intent: don't dilute the track record with calls the model can't stand
behind.

The decision is the disjunction of four triggers. Any one of them
suppresses; all four must be clear to commit.

```
SUPPRESS if any of:
  1. confidence < 50                                            (no-vision floor)
  2. wilson_upper_95(stock × regime accuracy)         < 52     (track record on this combo)
  3. wilson_upper_95(stock accuracy) < 52  AND
       ensemble agreement < 55                                 (low edge + disagreement)
  4. wilson_upper_95(horizon accuracy) < 48  AND
       confidence < 40                                         (low-confidence + bad horizon)
```

The 50-floor is the only trigger that's purely about the *current*
call's confidence. The other three are historical-accuracy gates:
they suppress when the model has demonstrably been below threshold
in this exact (ticker, regime, horizon) context.

**Wilson upper bound, not point estimate.** A naive point-estimate
gate ("accuracy < 52") fires on statistical noise at small samples.
At n=5 with 2 hits, the *observed* accuracy is 40% but the 95%
confidence interval runs from roughly 12% to 77% — the upper bound
is well above the threshold, and we can't reject the hypothesis
that this combo is at-or-above 52%. Suppressing on the point
estimate would fire on data that doesn't support the conclusion.

The Wilson upper bound replaces the point estimate. The gate fires
only when the *upper* end of a 95% confidence interval is below
threshold — i.e., when accumulated evidence rules out (at p ≤ 0.05)
the hypothesis that this combo is at-or-above threshold. This
naturally:

- Stays quiet at low n. CI is wide; gate doesn't fire on n=5 noise.
- Wakes up gradually as evidence accumulates. The gate fires the
  moment the CI is narrow enough to actually support the conclusion.
- Auto-scales forever. No `MIN_SCORED_FOR_ADJUSTMENT` constant to
  re-tune as data volume grows. Same code at n=5 and n=5000.

A 72%-confident MSFT call with 2 hits in 5 sideways predictions
*does not* suppress, because the historical evidence is too thin to
support the suppression. A 65%-confident call on a name with 40
hits in 100 calls *does* suppress, because the evidence supports it.
This is the integrity contract written into math instead of vibes.

### 12.1 The "No clear read" copy

The user-facing surface branches by trigger:

- *Confidence-floor case (#1).* Headline: "The model is sitting this
  one out." Body cites the confidence number and frames it as
  too-close-to-a-coin-flip. Honest because that is what's happening.
- *Track-record cases (#2–#4).* Headline unchanged. Body acknowledges
  the confidence number but pins the suppression on the model's
  history in this context. The italic suppress_reason underneath
  carries the specifics ("Model accuracy for MSFT in Sideways regime
  is 40% (5 predictions) — below 52% threshold").

Saying "72% is too close to a coin flip" when the gate is actually
track-record is a bug, not a copy choice — it contradicts both this
doc and the reason it shows.

### 12.2 Tunable parameters

Three knobs, all logged through `methodology_changes` if they ever move:

| Parameter | v1.3 value | What it gates |
|-----------|------------|---------------|
| `NO_VISION_FLOOR` | 50% | Hard confidence floor; below this, suppress unconditionally |
| `QUALITY_GATE_MIN_ACC` | 52% | Accuracy threshold for triggers #2 and #3 (Wilson upper bound check) |
| `SUPPRESSION_CI` | 0.95 | Significance level for the Wilson upper bound. 95% means the gate fires only when historical data is incompatible (p ≤ 0.05) with "model at-or-above threshold." |

`MIN_SCORED_FOR_ADJUSTMENT` is no longer used by the suppression
gate (v1.3 — see audit log). Wilson handles the small-n case
naturally by widening the CI; an explicit minimum became redundant
and was removing legitimate signal at moderate sample sizes. The
constant still gates the *confidence-adjustment* path in
`apply_learning`, which is independent of suppression.

### 12.3 What suppression doesn't do

- It does not delete or alter the prediction. The row is still
  written to `predictions`; it just gets `is_public_ledger = false`
  and (depending on UI implementation) is excluded from the user's
  asking-history feed.
- It does not block paper trading the prediction. The "Take this
  trade" CTA is gated separately by tradability, not by suppression.
- It does not affect the model's portfolio. `model_paper_trades`
  only opens off non-suppressed predictions in the first place
  (rule § 2.1.1 already requires confidence ≥ 65).

---

## 13. The shape of "no hiding"

This doc exists so that any user, member, or skeptic can read it, then look
at any number on the site, and reconstruct how that number was produced.
That's the operational definition of the product's positioning. If a number
shown on the site can't be derived from rules in this doc, either the
number or the doc is wrong — and we fix it.
