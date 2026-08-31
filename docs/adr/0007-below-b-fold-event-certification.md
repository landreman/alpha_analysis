# ADR 0007: Certify below-b folds as cuttable count-change events

- **Status:** Proposed
- **Date:** 2026-08-31
- **Milestone:** 10.3
- **Design sections:** §5.4 (multiway and degenerate transitions), §10.2
  (companion curve and contact localization), §21.3 dimension 5 (root scan
  resolution), §23 milestones 10.2–10.3; ADR 0003

## Context

ADR 0003 established that a bracketed interior-maximum count change on a
`GAMMA_MAX` curve is one of two §5.4 events: a barrier rising through \(b\)
(an equal-height contact, discontinuous port actions) or an interior
maximum–minimum pair annihilating strictly below \(b\) (a fold, continuous
port actions), with `barrier_margin` separating them only when the count
change belongs to the highest barrier. Milestone 10.2's localization bisects
every bracket, but its only geometric certification is `_solve_equal_height`:
both marginal equations at height \(b\) on one lifted field line. A localized
count change whose event is a below-\(b\) fold has no such solution, so the
event stays uncertified and *every arc incident to it is refused* ("event
geometry remains unresolved").

On the milestone 10.2 real matrix this is not a corner case: 815 of 951
unresolved arcs carry an event-geometry reason, and brackets with
`barrier_margin` far from zero (for example `d23p4` at
\(\lambda_n = 0.5\), margin \(\approx 0.15\)) are folds by ADR 0003's own
discriminator. Milestone 10.3's acceptance (≥95% of 120 cases resolved)
is unreachable while every fold-bearing curve is vetoed whole, and §23
milestone 10.3 explicitly instructs: "diagnose the reasons for unresolved
cases, devise new ideas if needed to remedy the unresolved cases."

The design does not say what certifies a fold or whether its incident arcs
may cut. Two facts shape the options. First, a fold's port actions are
continuous: the annihilating pair lies strictly below \(b\), so
\(A_1\), \(A_3\), \(A_W\) are unchanged as functions of \(u\) across the
event, and the one-sided limits from both sides coincide. Second, near the
fold the pair is shallower than the root scan resolves (§21.3 dimension 5),
so the scanned count flips a *blind zone* away from the true fold — measured
at \(8\times10^{-4}\) in \(u\) on the synthetic fold field against the
\(10^{-5}\) bisection target — and probes inside that zone report the blind
count.

## Options

1. **Keep folds uncertified (status quo)** — no new solve; every
   fold-bearing curve remains an explicit unresolved hyperedge. Costs the
   milestone 10.3 acceptance outright, and refuses cuts whose port data are
   provably continuous — conservatism that no longer encodes a real
   uncertainty once the fold is solved and checked.
2. **Solve and certify the fold, cut up to it, keep the event node
   (implemented)** — a four-equation solve on one lifted field line:
   marginal maximum (\(B=b\), \(D_\parallel B=0\), curvature
   \(< -\)tolerance) and, at lifted offset \(\tau\),
   \(D_\parallel B = D_\parallel^2 B = 0\), accepted only when the fold
   height is at least `fold_height_margin` \(\cdot |b|\) below \(b\)
   (default \(10^{-6}\), a reported control), \(\tau\) lies inside the
   parent well, the solution is local to the original sample bracket, and
   exactly one candidate survives — ambiguity, including a simultaneous
   equal-height solution, stays uncertified (§5.4). A certified fold is a
   `TransitionEvent` with `kind="fold"` and one marginal point; its
   incident arcs split at the event's uncertainty interval, which spans the
   bisected scan flip *and* the solved fold so blind-zone probes stay inside
   the event rather than contaminating an arc with blind counts. Arc
   endpoint limits use the existing `_event_well`/`_event_limit` machinery
   (a one-point event well is the ordinary well around the marginal
   maximum). The event node itself remains an explicit §5.4 hyperedge with
   `unresolved=True`: certifying the geometry does not decide multiway
   connectivity. What this costs: the event's *position along the curve* is
   known only to scan resolution (the blind zone), so the split parameter
   is honest to §21.3 dimension 5, not to the bisection tolerance; and
   equal-height events keep their 10.2 semantics unchanged (arcs split at
   the localized micro-bracket midpoint), so the two kinds are not treated
   symmetrically.
3. **Reclassify a fold bracket as no event at all** — since port actions
   are continuous, drop the event and cut straight through. Cheapest, but it
   erases a detected §5.4 itinerary change ("do not arbitrarily decompose a
   nongeneric event without recording that modeling choice"), loses the
   diagnostic anchor for milestone 11+'s interval transfer near folds, and
   makes a misclassified equal-height contact (blind spot of the margin
   discriminator) silently cuttable — a §21.2 plausible-wrong-number risk.

## Decision

Left for the researcher. Option 2 is what the milestone 10.3 branch
implements; rejecting it reverts fold-bearing curves to explicit uncertified
events, and the 120-case matrix then cannot reach the milestone's 95%
resolved threshold.

## Consequences

- `TransitionEvent` gains `kind` (`"equal_height"`/`"fold"`) and optional
  `fold_points`; `_register_event` never merges events of different kinds.
  `ContactLocalizationConfig` gains `fold_height_margin`. All are appended
  fields with defaults, so existing constructors and the NPZ cut format are
  unchanged (`CutEvent` stores marginal points and incidence only).
- A certified fold's occurrence `u_interval` is the scan-resolution
  uncertainty interval, not the bisection micro-bracket; incident arcs of a
  fold begin at its bounds. Equal-height arcs still split at the
  micro-bracket midpoint, preserving every milestone 10.2 recorded outcome
  (the 10.2 acceptance tests pass unchanged).
- `test_below_b_fold_event_certifies_and_cuts_with_continuous_limits` pins
  the fold field's event locations against an independent count bisection of
  the analytic field and requires continuous one-sided limits;
  `test_contact_localization_budget_escalates_until_events_localize` keeps
  the equal-height control (two marginal points, `equal_height` kind, never
  a fold).
- Milestone 11+ consumers see fold event nodes with one marginal point and
  continuous limiting actions on both incident arcs; connectivity through
  any event node remains explicitly unresolved until a later milestone
  decides it.
