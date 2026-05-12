-- audit_data_integrity.sql
--
-- Run periodically (or on demand from Supabase SQL Editor) to surface
-- predictions whose state is inconsistent. Each query is a distinct
-- "category of suspicious row" — if any query returns rows, those rows
-- need investigation.
--
-- Categories caught by this audit:
--   A. verdict ≠ OPEN but rating_target / _checkpoint / _expiration
--      still 'pending' or NULL.
--      → Scoring half-finished. Re-run score_open_predictions /
--        score_inflight_predictions / score_expiration_for_traded.
--
--   B. verdict = OPEN but horizon_ends_at < now() (overdue).
--      → score_open_predictions hasn't run since the horizon elapsed,
--        or it failed for this symbol (likely yfinance fetch).
--
--   C. rating_expiration in ('hit', 'miss') but horizon_ends_at > now().
--      → Premature endpoint rating. The 0013 migration cleaned the
--        traded case; this catches any new occurrences.
--
--   D. rating_target = 'hit' but rating_checkpoint = 'miss' or 'pending'.
--      → Hitting target implies hitting checkpoint (the midpoint).
--        Atomic update should keep them consistent.
--
--   E. is_public_ledger = true but no entry_price / target_price /
--      stop_price (means the persist path lost the methodology fields).
--      → Receipt UI will render entry/target/stop lines as null.
--
--   F. is_public_ledger = false despite confidence >= 55 AND symbol in
--      canonical universe — i.e. only the dedupe rule could have
--      excluded it. Useful to surface "duplicates within an hour" so
--      the user can see how often the dedupe gate fires.
--
-- Output: each result set is prefixed with a category label comment so
-- you can scan the output and tell which class of inconsistency each
-- row belongs to.


-- ── A. Verdict closed but ratings still pending ────────────────────
SELECT
  'A: verdict_closed_but_ratings_pending' AS category,
  id, symbol, direction, horizon, verdict,
  rating_target, rating_checkpoint, rating_expiration,
  created_at, horizon_ends_at
FROM predictions
WHERE verdict IN ('HIT', 'PARTIAL', 'MISSED')
  AND (
    rating_target IS NULL OR rating_target = 'pending' OR
    rating_checkpoint IS NULL OR rating_checkpoint = 'pending' OR
    rating_expiration IS NULL OR rating_expiration = 'pending'
  )
ORDER BY horizon_ends_at DESC
LIMIT 100;


-- ── B. Open verdict but horizon already elapsed ─────────────────────
SELECT
  'B: open_but_horizon_elapsed' AS category,
  id, symbol, direction, horizon, verdict,
  rating_target, rating_checkpoint, rating_expiration,
  horizon_ends_at,
  EXTRACT(EPOCH FROM (now() - horizon_ends_at))/86400 AS days_overdue
FROM predictions
WHERE verdict = 'OPEN'
  AND horizon_ends_at < now()
ORDER BY horizon_ends_at ASC
LIMIT 100;


-- ── C. Endpoint rating stamped before horizon elapsed ──────────────
SELECT
  'C: premature_expiration_rating' AS category,
  id, symbol, direction, horizon, verdict,
  rating_target, rating_checkpoint, rating_expiration,
  horizon_ends_at,
  EXTRACT(EPOCH FROM (horizon_ends_at - now()))/86400 AS days_remaining
FROM predictions
WHERE rating_expiration IN ('hit', 'miss')
  AND horizon_ends_at > now()
ORDER BY horizon_ends_at ASC
LIMIT 100;


-- ── D. Target hit but checkpoint not hit (logically impossible) ────
SELECT
  'D: target_hit_checkpoint_not' AS category,
  id, symbol, direction, horizon, verdict,
  rating_target, rating_checkpoint, rating_expiration,
  horizon_ends_at
FROM predictions
WHERE rating_target = 'hit'
  AND (rating_checkpoint IS NULL OR rating_checkpoint != 'hit')
ORDER BY created_at DESC
LIMIT 100;


-- ── E. Public ledger row missing methodology fields ────────────────
SELECT
  'E: ledger_missing_methodology' AS category,
  id, symbol, direction, horizon, verdict,
  entry_price, target_price, stop_price,
  confidence
FROM predictions
WHERE is_public_ledger = true
  AND (
    entry_price IS NULL OR
    target_price IS NULL OR
    stop_price IS NULL
  )
ORDER BY created_at DESC
LIMIT 100;


-- ── F. Off-ledger rows that look like they should qualify ──────────
-- (Confidence ≥ 55, but is_public_ledger = false. Usually due to the
--  1-hour dedupe rule. Surfacing helps audit how aggressive the gate
--  has been over the last 30 days.)
SELECT
  'F: high_conf_off_ledger' AS category,
  id, symbol, direction, horizon, verdict, confidence, created_at,
  -- Show the next-most-recent same (symbol, horizon) row for context;
  -- when this row exists within the prior hour, that's the dedupe
  -- ancestor. When it doesn't exist, this row was excluded for some
  -- other reason and probably warrants a manual look.
  (
    SELECT created_at
    FROM predictions AS prev
    WHERE prev.symbol = predictions.symbol
      AND prev.horizon = predictions.horizon
      AND prev.created_at < predictions.created_at
      AND prev.created_at > predictions.created_at - interval '1 hour'
    ORDER BY prev.created_at DESC
    LIMIT 1
  ) AS dedupe_ancestor_created_at
FROM predictions
WHERE confidence >= 55
  AND is_public_ledger = false
  AND created_at > now() - interval '30 days'
ORDER BY created_at DESC
LIMIT 50;
