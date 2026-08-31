# ADR 0008: Terminate companion cuts at degenerate curve endpoints

- **Status:** Proposed
- **Date:** 2026-08-31
- **Milestone:** 10.3
- **Design sections:** §5.2 (boundary curves), §5.4 (multiway and degenerate
  transitions), §10.2–§10.3 (companion construction and mesh alignment),
  §21.2, §23 milestone 10.3; ADRs 0006, 0007

## Context

On the real equilibria, `GAMMA_MAX` polylines are frequently *open arms whose
terminal vertices are classified `DEGENERATE`*: milestone 8's junction solves
place a chain endpoint exactly on a `D_parallel^2 B = 0` point (measured
`|D2| ~ 1e-13` on the PCA endpoints), where the marginal maximum annihilates
— the §5.4 first-bullet case, and the generic way a transition curve ends
away from `EDGE`. The transition mapper cannot trace such a vertex: its
sample fails as `source_classification`, and milestone 10.2's arc machinery
then rejects the whole curve ("certification interrupted by mapped sample
failure"). On PCA at \(\lambda_n = 0.1\) all three `GAMMA_MAX` arms (3–63
vertices) fail *only* at their two endpoint vertices; their interiors map
regular and certify. The 10.2 matrix recorded 69 arcs in this class, and
PCA/n3are carry such arms at most levels, so milestone 10.3's ≥95%
threshold is unreachable while a degenerate endpoint vetoes its curve.

The design says a cut must never dangle "away from `EDGE` or an explicit
event" (§10.3, ADR 0006) but does not say what the cut should do where the
curve itself ends at an annihilation.

## Options

1. **Keep the whole-curve veto (status quo)** — every arm with a degenerate
   endpoint stays uncut despite a certified regular interior. Fails the
   milestone acceptance; discards trustworthy port data for a failure that is
   confined to two untraceable vertices.
2. **Register the endpoint as an explicit event and stop the arc one vertex
   short (implemented)** — an open `GAMMA_MAX` polyline whose terminal
   vertex is `DEGENERATE`-classified registers a `TransitionEvent` with
   `kind="degenerate_endpoint"`, one marginal point (the solved degenerate
   vertex itself), `unresolved=True`, and an uncertainty interval equal to
   the terminal authoritative vertex spacing. Incident arcs end at their
   last regular vertex, keep their own payload there (no event-limit well is
   traced — none exists at the annihilation), and the cut terminates on the
   event per the existing ADR 0006 event-endpoint machinery, with the
   degenerate vertex pre-inserted as the shared anchor. Arms shorter than
   two vertex spacings dissolve entirely into their two endpoint events and
   are retained as arc-less explicit hyperedges. Side assignment for a short
   arc whose samples are all event endpoints probes the chain edges'
   flanking triangles with interpolated port actions; a slit that leaves the
   surface globally connected (an under-resolved junction complex) is
   demoted with an explicit reason routed to background refinement. What
   this costs: the cut stops one authoritative vertex spacing short of the
   annihilation, and that terminal strip's measure is inside the event
   rather than cut — resolution-honest (the event interval says exactly
   this) but a real truncation until refinement shortens the spacing.
3. **Trace one-sided limits toward the endpoint and extend the cut to the
   degenerate vertex** — solve the annihilation limit of the ports (child-3
   action → 0) and cut all the way. The limits are singular precisely where
   the trace machinery degenerates; inventing a regularized trace there
   risks the §21.2 plausible-wrong-number failure, and nothing downstream
   needs the last fraction of a vertex spacing before milestone 11.

## Decision

Left for the researcher. Option 2 is what the milestone 10.3 branch
implements; rejecting it restores the whole-curve veto and the 120-case
matrix cannot reach the milestone's 95% resolved threshold.

## Consequences

- `TransitionEvent.kind` gains the `"degenerate_endpoint"` value (joining
  ADR 0007's `"fold"`); registration matches kinds, so two arms ending on
  the same solved junction point share one event hub and their cut chains
  share its anchor vertex.
- `build_transition_arcs` stops arcs at such events' interval bounds and
  keeps the last regular sample as the terminal payload; an arc left with
  fewer than two samples raises the explicit below-resolution reason instead
  of building an uncuttable curve.
- The §23.10.3 matrix report counts curves fully dissolved into endpoint
  events as resolved-with-explicit-events, and the per-case record retains
  every event with its occurrences.
- `test_degenerate_endpoint_arms_certify_their_regular_interior` pins the
  behavior on the PCA \(\lambda_n = 0.1\) arms: every endpoint event must sit
  on a field point with `B = b` and `|D_parallel^2 B| < 1e-6` (checked by
  direct field evaluation), the interior arcs certify with event endpoints,
  the sub-resolution arms dissolve into retained events, and the coarse
  junction slit is demoted with the explicit background-refinement reason
  and its ports kept (§21.2).
