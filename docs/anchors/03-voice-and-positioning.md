# Anchor: Voice and Positioning

**Status:** Draft v1.1 — added "User deleted" copy for orphaned predictions.
**Owner:** Marc.
**Purpose:** The locked positioning, voice, and copy primitives for Prediqt.
Every page, button, tooltip, and email pulls from this doc. If the copy on
the site contradicts something here, the copy is wrong.

This is not a brand book. This is a stance document. It exists so the product
sounds like one thing, said by one voice, on every surface.

---

## 1. The one-sentence positioning

> **Prediqt is a stock-prediction model you challenge in public. Every call
> is permanent, dated, and scored against the market.**

Internalize this sentence. It is the answer to "what is this." Every
substitute we write — hero copy, About page, social bio, App Store
description — is a reframing of this sentence, not a different idea.

---

## 2. What the product is, and isn't

| It is | It isn't |
|-------|----------|
| An opponent — something you test | A tool — something you use |
| A public scoreboard for an ML model | A SaaS dashboard for traders |
| A system that learns from being wrong | A black-box with a confidence score |
| A challenge initiated by the user | A signal feed pushed at the user |
| A growing public record | A snapshot of "current picks" |
| A learning artifact for users | Financial advice |

The left column wins every time the right column tries to creep in. When in
doubt, write toward "challenge / scoreboard / record" and away from
"dashboard / signals / advice."

---

## 3. The five non-negotiable lines

These five sentences must appear on the site verbatim, somewhere
discoverable. They are the product's core claims, in its own voice.

1. **Every prediction starts with a question.**
   — The asking mechanic, framed as the engine.

2. **No deletes. No edits. No hiding.**
   — Permanence, in three beats.

3. **It learns from being wrong.**
   — Self-improvement, owned without softening.

4. **The model makes the call. The market decides who's right.**
   — The integrity contract, in one line.

5. **Think you can beat it? Ask for a prediction.**
   — The CTA, with ego baked in.

These five are the seed crystal. New copy grows out from them, doesn't
replace them.

---

## 4. The positioning shift, line by line

The copy currently leans toward "tool." Below is the rewrite map. When
working a surface, scan for the left-column phrasing and flip it.

| Was (tool voice) | Is (opponent voice) |
|------------------|---------------------|
| "Get stock predictions" | "Challenge the model" |
| "Our AI predicts prices" | "The model commits to a call" |
| "Track our predictions over time" | "Watch every call get judged" |
| "Improve your trading with AI" | "Test your read against the model's" |
| "Powered by machine learning" | "Trained on a decade of market data. Tested in public." |
| "Sign up for predictions" | "Start asking" |
| "Disclaimer: not financial advice" | "Not financial advice. A learning tool." (owned, not buried) |
| "Confidence: 78%" | "The model's read: 78% confident" |
| "Prediction made on 4/22" | "Called 4/22 — outcome pending" |
| "View your dashboard" | "Your record" |

Notice the small things: *commits*, *call*, *judged*, *read*, *outcome*.
These verbs do real work. They put weight behind every prediction.

---

## 5. Hero options

Three to choose from, one to put on the landing page. We'll A/B these later;
day one, pick the one that hurts most for skeptics.

**Option A — the conversion-leaning version:**
> The model makes the call. The market decides who's right.
> *Subhead:* Every prediction is public, dated, and scored. No deletes.
> No edits. No hiding.

**Option B — the curiosity hook:**
> Ask it anything. Watch it prove itself.
> *Subhead:* A self-learning prediction model with a public track record.
> Every call you summon goes on the record — pass or fail.

**Option C — the aggressive version:**
> Think you can beat it? Ask for a prediction.
> *Subhead:* Prediqt commits to a call. You decide whether to follow it,
> fight it, or paper-trade it. The market keeps score.

**Recommendation:** A for the marketing site, C as the headline above the
prediction request box (the moment of action). Keep B in the bank for ads.

**Option D — the approved current hero (locked v1.2):**
> Real predictions. Real receipts.
> *Subhead:* The model commits to a call. Every prediction is public,
> dated, and scored. No deletes. No edits. No hiding.

Owner choice for the production landing page. Tighter than A — same
integrity claim in fewer beats — and lets the live ledger underneath
do the proof work that A's "the market decides who's right" leaves to
imagination. A and C remain canonically defined here; D is what
ships. Future copy can pull from any of the four; the doc tracks the
ones in active use so reviewers always know what's authorized.

---

## 6. CTA copy

Buttons are where voice gets tested. These are the approved labels:

| Surface | Primary CTA | Secondary CTA |
|---------|-------------|---------------|
| Landing hero | **Ask for a prediction** | See the track record |
| Pricing | **Start asking** (free) / **Go unlimited** (paid) | Compare plans |
| Empty state on Dashboard | **Make your first call** | Browse the public ledger |
| Prediction page | **Summon a prediction** *or* **Make the call** | — |
| Email confirmation | **See the model's call** | — |

Banned phrasing:
- *Get started* (weak, generic)
- *Sign up* (transactional, uninspired)
- *Try Prediqt* (positions us as a trial, not a confrontation)
- *Submit* (form-flavored)

---

## 7. Public verdict labels — the only four

The four-state outcome from the methodology doc maps to four public-facing
labels. These are short, scannable, and visually distinct enough for the
share card.

| State | Label | Color tone | Notes |
|-------|-------|------------|-------|
| `OPEN`    | **Pending**  | Cyan / brand | Live; clock still running. |
| `HIT`     | **Hit**      | Green        | Target reached. The model wins. |
| `PARTIAL` | **Partial**  | Amber        | Got close. Checkpoint hit, target missed. |
| `MISSED`  | **Missed**   | Red          | Expired without hitting either. |

Don't soften `MISSED`. We don't say "didn't hit," "below target," or "in
review." The whole product is built so the word *Missed* can sit there in
red without anyone needing to spin it.

The TRADE/PASS state has its own label set:

| State | Label | Color tone |
|-------|-------|------------|
| `traded = true`  | **Backed**       | Brand-cyan |
| `traded = false` | **Watched**      | Ink-muted  |

"Backed" reads as commitment. "Watched" reads as caution without apology.

The asker (the user who triggered the prediction) gets one more label state
for the case where the user has deleted their account — the prediction
stays public per methodology § 8.2, but the asker is gone:

| State | Label | Color tone |
|-------|-------|------------|
| Active user      | *(user's display name or "Anonymous" if private)* | Default |
| Deleted user     | **User deleted**  | Ink-muted, italic |

We say *User deleted* — not *Anonymous*, not *Removed*, not *Account
closed*. Honesty over euphemism. The prediction stays in the ledger; the
fact that the asker is no longer here is stated plainly.

---

## 8. The tier-gating copy

Every "you need to upgrade to see this" moment is a chance to either
reinforce the product's voice or undermine it. Default rules:

- Don't apologize for the paywall. The model's track record is *the* free
  thing. Detail is the upgrade.
- Lead with what they unlock, not what they're missing.
- Treat the public ledger snippet as a teaser, not a stub.

Approved phrasings for the three lock states:

**Public viewing a non-public field:**
> *Trade details are visible to members.*
> [Sign up free] [What's included]

**Free user viewing a paid-only field on someone else's prediction:**
> *Full call details — entry, target, stop, and the model's reasoning —
> are part of Pro.*
> [Compare plans]

**Free user hitting their 3-prediction quota:**
> *You've used your three free predictions this month. Pro unlocks 30
> a month, plus the ability to back any prediction in the public ledger.*
> [Go Pro]

Banned phrasings:
- *Sorry, you'll need to…* (don't apologize)
- *Premium feature* (generic, doesn't carry voice)
- *Locked* (doesn't say what they get, only that they don't)

---

## 9. The ask-flow voice

The single most important moment in the product is the user typing a ticker
and clicking the button. The copy in that flow should sound like someone
loading a chamber, not filling in a form.

**Approved microcopy beats:**

- *What ticker should the model call?* (input placeholder)
- *How far out?* (horizon picker label)
- *Make the call →* (submit button)
- *Reading the tape…* (loading state, ~1 second)
- *The call has been recorded. Outcome pending.* (success state)
- *This call is now permanent.* (confirmation tooltip)

**The success state matters.** When the prediction lands, the page should
feel like a record book just got a new entry, not like a form returned a
JSON response. The phrase *"now permanent"* is the moment we earn the "no
hiding" claim.

---

## 10. Confidence and uncertainty — how to talk about them

The model has confidence scores. The temptation is to hedge. Don't.

| Don't say | Do say |
|-----------|--------|
| "The model thinks…" | "The model's call: …" |
| "It might be…" | "The model expects…" |
| "Possibly bullish" | "Bullish — 72% confidence" |
| "There's a chance…" | "The market will decide." |

Confidence scores get rendered as numbers, not adjectives. *72% confident*,
not *quite confident*. Honesty without softening.

When a call MISSES, the page **does not** add language like "but the
fundamentals were strong" or "confidence was on the lower end." The Missed
verdict is allowed to stand. The model doesn't get an excuse column.

The symmetric case is when the model **declines to call** — the
suppression gate fires (methodology § 12) and the surface renders
"No clear read" instead of a prediction. Same posture, different
direction: the model owns its limits without spinning. If the model
has no edge, it says so. Approved framing depends on which gate fired:

- *Confidence-floor case* (raw confidence < 50%): the call is
  acknowledged as a coin flip and the surface says so. Example:
  "Confidence on this horizon came in at **48%**. That's too close to
  a coin flip to commit to a prediqt."
- *Track-record case* (any of the historical-accuracy gates): the
  current confidence number is acknowledged but the suppression is
  pinned on the model's record in this context, not on the read of
  the call. Example: "The current read is **72%** confident, but the
  model's track record in conditions like this hasn't earned the
  right to commit."

Banned in either case: framing the suppression as a system failure
("Sorry," "We couldn't…," "Try a different ticker"). The decision to
sit out is the system working — not failing.

---

## 11. Educational framing — the "learning tool" position

Compliance asks us to say it. The product genuinely is one. Both true.

Approved disclaimer language (use verbatim where required):

> Prediqt is a learning tool, not financial advice. The model makes
> predictions; the market decides outcomes. Don't trade real money based on
> anything you see here — paper-trade, watch, learn.

Notes:
- "Learning tool" appears **before** "not financial advice." The educational
  framing leads.
- "Don't trade real money" is more decisive than "consult a financial
  advisor." It tells the user what *not* to do, which is the actually
  helpful thing.
- The line goes in the footer, in the prediction confirmation, and in the
  paper-trading flow. Three exposures, no buried legalese.

---

## 12. Sharing copy — the viral primitive

Every prediction is a shareable artifact. The copy on the share card and the
auto-generated share text follow a fixed grammar.

**Card title (the moment of asking):**
> Marc challenged Prediqt on $NVDA.

**Card title (after a verdict):**
> Prediqt called $NVDA bullish. Hit. (3-day window, called 4/12.)

**Auto-generated tweet text (when the user clicks share):**

- *On the call:* `I just challenged Prediqt on $NVDA. The model's call:
  Bullish, 3-day window. Outcome pending. → [link]`
- *On a hit:* `Prediqt called $NVDA bullish. Hit. → [link]`
- *On a miss:* `Prediqt called $NVDA bullish. Missed. → [link]`
- *On the user beating the model:* `Prediqt passed on $NVDA. I backed it.
  +6.2%. → [link]`

The grammar: ticker, call, verdict, link. Every share. No exceptions.

---

## 13. Tone — the tight rules

A few formatting and tone constraints that hold across the site.

- **Never use "we."** The product speaks for itself. Refer to "the model"
  or "Prediqt." We is a corporate voice that doesn't belong here.
- **Active voice, present tense.** "The model calls" not "the model has
  called" or "predictions are made." The system is active.
- **No exclamation points.** The product is confident enough to land
  without them. Save the energy for the verbs.
- **No "powered by AI" rhetoric.** It's a model trained on a decade of data;
  if we have to namedrop ML to justify ourselves, we're losing.
- **No greetings in product UI.** No "Welcome back, Marc." It's a record
  book, not a hotel lobby.
- **Numbers are tabular.** Confidence reads "72.4%" not "72%-ish." Returns
  read "+12.8%" not "up about 13."
- **Em dashes are allowed.** They do real work in this voice. Don't
  replace with semicolons.

---

## 14. Anti-examples — copy that violates the voice

Real things the site might say that need to be flagged and rewritten.

| ❌ Off-voice | ✅ Corrected |
|------------|------------|
| "Welcome to Prediqt — your AI-powered stock advisor." | "Prediqt is a stock-prediction model you challenge in public." |
| "Our model has a 63% accuracy rate!" | "The model has hit its target on 63% of settled calls." |
| "Sorry, this prediction is for Pro members only." | "Trade details are part of Pro." |
| "Want unlimited predictions? Upgrade today!" | "Pro Unlimited: ask anything, anytime, plus API access." |
| "Try our free trial." | "Start asking. Three free predictions a month." |
| "Smart, AI-powered insights for serious traders." | "A model that commits to its calls. In public. With receipts." |

---

## 15. Where this doc gets used

This doc is the **single source** for:

- Landing-page hero, subheads, and section copy
- Track Record page header and methodology footnote
- Prediction request page (the "asking" flow)
- Pricing page tier descriptions and CTAs
- Dashboard empty states and quota meters
- Email confirmations and onboarding
- The share card grammar
- Compliance disclaimer language
- Tooltips and microcopy across the app

When a new surface gets designed, the copy comes from here first, gets
adjusted to fit the surface, then gets the adjusted version reflected back
into this doc as approved variants. We don't ship copy that doesn't trace
back to a primitive in §3, §4, or §7.

---

## 16. Open questions

1. **Brand-personality voice for support and emails.** This doc nails the
   product voice; do we want a separate, slightly warmer voice for
   transactional emails and support replies, or hold the same line? Day-one
   recommendation: hold the same line. Warmer-in-product communications
   undercut the integrity claim.

2. **Profanity / casual register.** The doc keeps a confident but PG
   register. Should the site occasionally let a sharper line through
   (e.g. "the model called it. The market called it back.")? Worth a
   conscious decision before a copywriter improvises.

3. **Model "personality" naming.** The Track Record page has a "Model
   Personality" section. With this voice, do we keep that label or shift
   to "How the model thinks" or "What the model leans on"? Personality
   slightly anthropomorphises in a way that nudges toward "tool/companion"
   instead of "opponent." Consider renaming.
