-- 0014_promote_ttwo_to_public_ledger.sql
--
-- One-shot backfill: promote eligible TTWO predictions to the public
-- ledger that were excluded by the bug where universe.py's hardcoded
-- fallback list didn't include TTWO (Take-Two Interactive — a
-- current S&P 500 + NASDAQ-100 constituent).
--
-- Background
-- ──────────
-- The data-integrity audit (scripts/audit_data_integrity.sql, query F)
-- surfaced TTWO 1-month predictions at 78%+ confidence sitting off
-- the public ledger with no dedupe ancestor. The only other gate that
-- excludes a row is the canonical-universe check (db.py:_is_public_ledger).
-- Wikipedia 403s Render's datacenter IPs, so universe.py falls through
-- to a hardcoded list, and TTWO wasn't in it.
--
-- The going-forward fix is in universe.py:
--   • _fetch_wiki_html() sends a real Chrome User-Agent so the Wiki
--     scrape works from Render too.
--   • TTWO added to the curated fallback as a belt-and-suspenders.
--
-- This migration cleans up the historical mess for TTWO specifically.
-- It promotes any TTWO prediction with confidence ≥ 55 to the public
-- ledger, EXCEPT where an earlier (within 1 hour, same user/symbol/
-- horizon) prediction already exists — the same dedupe rule
-- _is_public_ledger enforces.
--
-- Idempotent: only flips rows currently off-ledger.

UPDATE predictions AS p
SET is_public_ledger = true
WHERE p.symbol = 'TTWO'
  AND p.confidence >= 55
  AND p.is_public_ledger = false
  AND NOT EXISTS (
    SELECT 1
    FROM predictions AS earlier
    WHERE earlier.symbol = p.symbol
      AND earlier.horizon = p.horizon
      AND earlier.user_id = p.user_id
      AND earlier.created_at < p.created_at
      AND earlier.created_at > p.created_at - interval '1 hour'
  );
