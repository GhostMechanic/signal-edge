"""
consensus_check.py
------------------
Horizon-adjusted contrarian guardrail (methodology § 4.3, added after the
MSFT 1m bug on Apr 30 2026 where the public-ledger card said "Model target
$390.62 is 31.6% below analyst consensus $570.72" but the model's
prediction was a one-month horizon while consensus is a 12-month outlook —
two different timeframes silently compared).

Three layered concerns this module addresses:

  1. The 31.6% raw divergence is itself misleading. Analyst consensus is
     normally a 12-month target; the model emits horizon-specific
     predictions (3 days … 1 year). De-annualising the implied analyst
     return down to the prediction's horizon and comparing per-horizon
     expected returns gives an apples-to-apples number.

  2. When the horizon-adjusted divergence is large AND the model and
     consensus are directionally opposed (analysts bullish + model
     bearish, or vice-versa), the call is meaningfully contrarian.
     Surface a flag so the UI can badge it.

  3. Strongly-contrarian directional-opposite calls have higher prior
     risk — analysts have inputs the model doesn't (earnings calls,
     channel checks, sector expertise). Apply a small confidence
     haircut (×0.85) so the displayed conviction reflects the bigger
     bet implied by disagreeing with the room. The PREDICTION still
     ships. We don't suppress contrarian calls — sometimes the model
     IS ahead of the analysts (regime shifts, technical breakdowns
     before fundamentals catch up). Suppression would kill the model's
     edge; awareness + a haircut is the right strength of intervention.

All three pieces live in pure functions here so the orchestrator
(api.main / prediction_logger_v2) can mix and match without copy-
pasting math, and tests can pin the boundaries down without the
Supabase-or-yfinance halo.

Methodology constants are lifted to module-level so they're greppable
and one-stop-changeable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ─── Methodology parameters § 4.3 (locked v1) ─────────────────────────────────

# Yahoo Finance's targetMeanPrice is conventionally a 12-month forward target.
# The implied analyst expected return = (target / current − 1) is therefore
# already annual; we de-annualise it to the prediction's horizon before
# comparing.
ANALYST_TARGET_HORIZON_DAYS: int = 365

# Horizon-adjusted divergence (in expected-return percentage points)
# above which a call is "strongly contrarian." 5pp = a 5 percentage-point
# gap on the same horizon. At a 1-month horizon, that's roughly the
# difference between expecting +3% and expecting -2%, which is a
# meaningful disagreement.
STRONG_DIVERGENCE_PP: float = 5.0

# Confidence multiplier applied when a call is strong-contrarian AND
# directionally opposed. 0.85 on a 70.1% confidence drops it to 59.6% —
# enough to push borderline calls below the 65% TRADE threshold and
# enough to be visually distinct on the confidence bar without nuking
# legitimate contrarian wins.
CONTRARIAN_CONFIDENCE_HAIRCUT: float = 0.85

# Fallback when the prediction horizon is unknown — no haircut applied,
# just return the raw signal.
HORIZON_DAYS_DEFAULT: int = 30


# ─── Math helpers ─────────────────────────────────────────────────────────────

def deannualise_return(
    annual_return: float,
    horizon_days: int,
    *,
    target_horizon_days: int = ANALYST_TARGET_HORIZON_DAYS,
) -> float:
    """De-annualise an `annual_return` (decimal — 0.20 = +20%/yr) down to
    the equivalent compound return over `horizon_days`.

    Uses geometric (compounding) de-annualisation:
        r_h  =  (1 + r_a) ** (h / a)  −  1

    This matches how analyst price targets are commonly interpreted —
    they imply a 12-month total return, which compounds at the same
    annualised rate over any sub-period.

    Defensive on zero / negative inputs:
      • horizon_days <= 0          → return 0 (no time, no return)
      • target_horizon_days <= 0   → return the input unchanged (caller
                                     can opt out by setting target=0)
      • non-finite annual_return   → return 0
    """
    if not math.isfinite(annual_return):
        return 0.0
    if horizon_days <= 0:
        return 0.0
    if target_horizon_days <= 0:
        return float(annual_return)
    # 1 + r_a can dip below 0 for absurd inputs (e.g. -150% target);
    # guard against fractional powers of negatives. Anything below the
    # floor returns -1 (full loss) which is the correct compounding limit.
    base = 1.0 + float(annual_return)
    if base <= 0:
        return -1.0
    exponent = float(horizon_days) / float(target_horizon_days)
    return base ** exponent - 1.0


def directional_opposition(
    model_return: float,
    consensus_return: float,
    *,
    deadband: float = 0.001,  # 0.1% — anything inside this is "no opinion"
) -> bool:
    """True when the model and consensus disagree on direction (one is
    meaningfully positive, the other meaningfully negative). Both inside
    the deadband → not opposed (both are "no opinion"). One inside, one
    outside → not opposed (only one has a directional view)."""
    if not (math.isfinite(model_return) and math.isfinite(consensus_return)):
        return False
    if abs(model_return) <= deadband or abs(consensus_return) <= deadband:
        return False
    return (model_return > 0) != (consensus_return > 0)


# ─── Result type ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConsensusCheck:
    """Output of evaluate(). Flat enough to drop straight onto the
    predict-response per-horizon dict and onto the prediction row."""
    has_consensus:                bool
    horizon_days:                 int
    # All returns expressed as decimals (0.05 = +5%).
    consensus_implied_annual:     Optional[float]
    consensus_implied_horizon:    Optional[float]
    model_return_horizon:         Optional[float]
    # Difference in horizon-adjusted expected returns, in percentage
    # points (model_return − consensus_return) × 100. Sign carries the
    # direction of the gap; magnitude carries the size.
    divergence_pp:                Optional[float]
    is_directionally_opposed:     bool
    is_strong_contrarian:         bool
    # When True, the orchestrator should apply the haircut and surface
    # a "strong contrarian" badge. Equals
    # is_strong_contrarian AND is_directionally_opposed by design — kept
    # explicit so callers can disambiguate the two halves if needed
    # (e.g. for analytics).
    apply_haircut:                bool


def evaluate(
    *,
    consensus_target_price: Optional[float],
    current_price:          Optional[float],
    predicted_price:        Optional[float],
    horizon_days:           int,
    strong_pp:              float = STRONG_DIVERGENCE_PP,
) -> ConsensusCheck:
    """Run the full contrarian check for one horizon.

    Inputs may be None / NaN — yfinance returns None for analyst targets
    on illiquid names, and any of the prices can be missing on first-
    fetch races. In every degenerate case we return a ConsensusCheck
    with `has_consensus=False` and `apply_haircut=False`, leaving the
    caller's existing path unchanged.
    """
    # Guard inputs.
    if (
        consensus_target_price is None
        or current_price is None
        or predicted_price is None
        or current_price <= 0
        or horizon_days <= 0
    ):
        return ConsensusCheck(
            has_consensus=False,
            horizon_days=horizon_days,
            consensus_implied_annual=None,
            consensus_implied_horizon=None,
            model_return_horizon=None,
            divergence_pp=None,
            is_directionally_opposed=False,
            is_strong_contrarian=False,
            apply_haircut=False,
        )

    if not (
        math.isfinite(consensus_target_price)
        and math.isfinite(current_price)
        and math.isfinite(predicted_price)
    ):
        return ConsensusCheck(
            has_consensus=False,
            horizon_days=horizon_days,
            consensus_implied_annual=None,
            consensus_implied_horizon=None,
            model_return_horizon=None,
            divergence_pp=None,
            is_directionally_opposed=False,
            is_strong_contrarian=False,
            apply_haircut=False,
        )

    consensus_annual  = consensus_target_price / current_price - 1.0
    consensus_horizon = deannualise_return(consensus_annual, horizon_days)
    model_horizon     = predicted_price / current_price - 1.0

    divergence_pp = (model_horizon - consensus_horizon) * 100.0

    is_opposed   = directional_opposition(model_horizon, consensus_horizon)
    is_strong    = abs(divergence_pp) >= strong_pp
    apply_h      = is_strong and is_opposed

    return ConsensusCheck(
        has_consensus=True,
        horizon_days=horizon_days,
        consensus_implied_annual=round(consensus_annual, 6),
        consensus_implied_horizon=round(consensus_horizon, 6),
        model_return_horizon=round(model_horizon, 6),
        divergence_pp=round(divergence_pp, 4),
        is_directionally_opposed=is_opposed,
        is_strong_contrarian=is_strong,
        apply_haircut=apply_h,
    )


def haircut_confidence(
    confidence: float,
    check: ConsensusCheck,
    *,
    haircut: float = CONTRARIAN_CONFIDENCE_HAIRCUT,
) -> float:
    """Apply the contrarian haircut to a confidence score (0–100). When
    `check.apply_haircut` is True, multiplies confidence by `haircut`;
    otherwise returns confidence unchanged. Pure — caller is responsible
    for stamping the result back onto the prediction record."""
    if not check.apply_haircut:
        return confidence
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return confidence
    if not math.isfinite(c):
        return confidence
    return round(c * haircut, 4)


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # The MSFT case from the public-ledger card on Apr 30, 2026:
    #   current_price = $407.78
    #   predicted_price (1m) = $390.62  → model_return = -4.21% over 1 month
    #   targetMeanPrice = $570.72       → consensus_annual = +39.95%
    msft = evaluate(
        consensus_target_price=570.72,
        current_price=407.78,
        predicted_price=390.62,
        horizon_days=30,
    )
    print("MSFT 1m contrarian check:")
    print(f"  consensus annual   = {msft.consensus_implied_annual:.4f} ({msft.consensus_implied_annual*100:.2f}%)")
    print(f"  consensus 1m       = {msft.consensus_implied_horizon:.4f} ({msft.consensus_implied_horizon*100:.2f}%)")
    print(f"  model 1m           = {msft.model_return_horizon:.4f} ({msft.model_return_horizon*100:.2f}%)")
    print(f"  divergence (pp)    = {msft.divergence_pp:.2f}")
    print(f"  directionally opp  = {msft.is_directionally_opposed}")
    print(f"  strong contrarian  = {msft.is_strong_contrarian}")
    print(f"  apply haircut      = {msft.apply_haircut}")
    haircut_conf = haircut_confidence(70.1, msft)
    print(f"  haircut: 70.1 → {haircut_conf:.2f}")
    print("consensus_check.py self-test OK")
