-- 0015_promote_etfs_to_public_ledger.sql
--
-- Promote existing high-confidence ETF predictions to the public
-- ledger after universe.py grew an ETFS bucket and
-- _is_public_ledger() learned to consult it.
--
-- Background
-- ──────────
-- Until this commit, _is_public_ledger() only consulted three
-- buckets: sp500, nasdaq100, dow30. ETFs were silently excluded
-- regardless of confidence because the canonical-universe gate
-- returned False for every ticker that wasn't a stock-index member.
-- Surfaced by an IBIT (iShares Bitcoin Trust) prediction at 60%+
-- confidence that the user expected to see on the public ledger.
--
-- This migration backfills the historical mess: any prediction on a
-- ticker that is now in the ETF bucket, with confidence ≥ 55, that's
-- currently off-ledger, gets promoted. The same hourly dedupe rule
-- applies — we don't promote a row if an earlier row exists within
-- the prior hour for (user, symbol, horizon).
--
-- Idempotent. Only flips rows currently off-ledger.

UPDATE predictions AS p
SET is_public_ledger = true
WHERE p.symbol IN (
    -- Mirrors universe.py:ETFS. When that list grows, add the new
    -- ticker to this UNNEST and re-run the migration to backfill.
    'SPY','QQQ','DIA','IWM','VOO','VTI','VEA','VWO',
    'XLK','XLF','XLE','XLU','XLV','XLI','XLY','XLP','XLB','XLRE','XLC',
    'TLT','IEF','SHY','LQD','HYG','AGG','BND','TIP',
    'GLD','SLV','IAU','USO','UNG',
    'EFA','EEM','FXI','INDA','EWZ','EWJ','ASHR',
    'IBIT','FBTC','ARKB','BITB','HODL','BTCO','BRRR','EZBC',
    'ETHA','FETH','ETHE','ETHV','EZET','ETH',
    'VYM','SCHD','DVY','NOBL','SPHD',
    'VUG','VTV','VBR','VBK','VB','VO','VV',
    'ARKK','ARKW','ARKG','ARKQ','ARKF','ICLN','TAN','LIT',
    'TQQQ','SQQQ','UPRO','SPXU','UDOW','SDOW','TMF','TMV',
    'VXX','UVXY','SVXY',
    'VNQ','IYR','REM','SCHH'
  )
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
