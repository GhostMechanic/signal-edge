-- ─────────────────────────────────────────────────────────────────────────────
-- 0009 — methodology_changes: suppression-gate moves from point-estimate to
--                              Wilson upper-bound (95% CI).
--
-- Records the methodology change defined in docs/anchors/02-methodology.md
-- § 12. Required by methodology § 2.3: every behavioral change to the
-- model's gating goes through methodology_changes with a public rationale.
-- The Track Record page surfaces these as vertical lines on the equity
-- curve so users can see exactly when and why the rules moved.
--
-- Behavior delta:
--   Before: gates 2/3/4 fired when the *point estimate* of historical
--           accuracy was below threshold, gated by MIN_SCORED_FOR_ADJUSTMENT=5.
--   After:  same gates fire when the *Wilson 95% upper bound* of historical
--           accuracy is below threshold. The minimum-n knob is gone — the
--           CI naturally widens at small n so sub-significant samples don't
--           fire the gate on noise.
--
-- The rationale is mathematical: with n=5 and 2 hits, observed accuracy
-- is 40% but the 95% CI is roughly [12%, 77%]. The upper bound is well
-- above any threshold we'd realistically choose, so we cannot reject the
-- hypothesis that the model is at-or-above threshold. Suppressing on the
-- point estimate fired on data that doesn't support the conclusion. The
-- Wilson bound fixes this without a tunable minimum.
-- ─────────────────────────────────────────────────────────────────────────────


insert into public.methodology_changes
    (field, old_value, new_value, rationale)
values (
    'should_suppress_prediction.gate_method',
    'point_estimate (accuracy < threshold) with n_scored >= 5',
    'wilson_upper_95ci (upper bound < threshold), no minimum n',
    'Point-estimate gates with n=5 minimum fired on statistical noise — at n=5 with 2 hits the 95% CI is roughly [12%, 77%], so the gate concluded "model below threshold" from data that did not support the conclusion. Wilson upper bound naturally widens at small n and tightens as evidence accumulates, so a single fixed threshold gives sensible behavior across all sample sizes. See methodology § 12.'
);


-- ─────────────────────────────────────────────────────────────────────────────
-- End of 0009_suppression_gate_wilson.sql
-- ─────────────────────────────────────────────────────────────────────────────
