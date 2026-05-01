# Prediqt — Anchor Docs

Three contracts that everything else gets built against. These are
"paper before code" — read them, push on them, get them right, then
the implementation is mostly mechanical.

## Read in this order

1. **[01-data-model.md](./01-data-model.md)** — what shape every prediction,
   trade, and portfolio takes. The Postgres delta against the existing
   Supabase migration. Defines the four-tier visibility contract.

2. **[02-methodology.md](./02-methodology.md)** — how the model decides to
   trade vs. pass, how it sizes, when it closes, and how the public verdict
   is computed. The integrity rules that prevent cherry-picking.

3. **[03-voice-and-positioning.md](./03-voice-and-positioning.md)** — the
   voice the product speaks in, the five non-negotiable lines, the verdict
   labels, the CTA copy, and what we never say. The shift from "tool" to
   "opponent."

## How these get used

- **No shipping copy or schema that contradicts these.** If you want to
  break a rule, change the rule here first.
- **Each doc has an Open Questions section.** Resolve those before writing
  Phase 1 code against the spec.
- **These docs are versioned.** When something changes (e.g. the
  trade-confidence threshold), bump the doc and link to the methodology
  changelog from the Track Record page.

## What's not in here yet

The next anchors, in rough priority order:

- **04-tier-feature-matrix.md** — the precise table of what each plan
  unlocks (predictions/month, paper-trading scope, ledger access, API,
  ads). Spinoff from the data model + voice docs.
- **05-share-card.md** — the visual and copy spec for the prediction share
  card. Tied tightly to the voice doc.
- **06-onboarding.md** — the first-run sequence, from sign-up through
  first prediction. The moment the "opponent" framing has to land in
  under 30 seconds.

These come after the first three are locked.
