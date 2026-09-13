# R3.5 implementation brief: practical root and contour feasibility

This is an implementation handoff, not completed validation. The normative scope,
named acceptance tests and exit gate are in
[DESIGN §23 R3.5](../DESIGN.md#r35--practical-root-certification-and-contour-feasibility).
[ADR 0011](../adr/0011-pre-r4-feasibility.md) records the requested insertion.
Implement **R3.5 only**; leave R4–R8 unchecked. Follow the repository environment,
scientific mutation checks, PR workflow and STOP rules in [AGENTS](../../AGENTS.md).

## Baseline and first reads

R3 merged as `39c1efd`, with implementation head `45f2d73`. Start from the current
plan-bearing main baseline containing this milestone; check its GitHub Tests
before implementation. Preserve all R1–R3 scientific/public-API regressions.

Read DESIGN §§7–11, §13.4, §20.3, §21 and §22, then:

- `alpha_analysis/j_connectivity/forward_catalogue.py`: reusable scans, root
  completeness, resume and finite-window terminals.
- `alpha_analysis/j_connectivity/branch_atlas.py`: `_certified_roots`, transverse
  bounds, cell classification/refinement, ownership and pointwise transitions.
- `alpha_analysis/j_connectivity/contour_trace.py`: root continuation, step
  correction, `_root_pattern_certified`, supplied-event processing and port tracing.
- `test/test_branch_atlas.py` and `test/test_contour_trace.py`: existing synthetic
  identities and real DMercFail reference.
- [R2 report](../validation/r2-atlas-matrix.md),
  [R2 JSON](../validation/r2-atlas-matrix.json),
  [R3 report](../validation/r3-contour-matrix.md),
  [R3 JSON](../validation/r3-contour-matrix.json) and
  [R3 driver](../validation/r3_contour_matrix.py).

The original R3 coarse/fine controls were `(step=.04, max_steps=80)` and
`(step=.02, max_steps=160)`. Both retained `scan_periods=4`, `min_step=.001`,
`max_certificate_boxes=64` and `action_atol=2e-6`. The certificate also hard-coded
six transverse subdivision levels and two scan phases. Thus the published
comparison refined contour stepping, not all potential failure mechanisms.

The five attempted `(s, alpha)` seed locations were `(0.3,0)`, `(0.5,0)`,
`(0.8,0)`, `(0.5,pi/2)`, `(0.8,pi/2)`. The driver chose the first complete well,
not an unbiased birth-population sample. Preserve its selected root lifts as well
as its transverse points when replaying a seeded query.

## Diagnose before changing defaults

The fine matrix has 13 no-seed cases, seven uncertified numerical closures, one
uncertified edge query and nine ordinary-continuation failures. None of its 17
seeded queries ended solely because `max_steps` was exhausted. Keep distinct
reasons throughout the replacement implementation.

Exploratory diagnosis after R3, before this planning change, found the following.
These are development leads, not committed acceptance evidence: reproduce and
save scripts, exact revision/field/configuration hashes and results in R3.5 before
using them to substantiate the implementation.

- Trying `scan_periods=8,16,32` at the original locations recovered seeds in six
  of the 13 no-seed cases. Independently trying radii `0.95,0.99,0.9,0.1,0.02,0.2`
  and eight equally spaced alpha values with the original four-period window
  recovered 11/13. PCA 0.95 and d23p4 0.95 still had no seed in these probes.
  This establishes search limitations, not empty population or contour success.
- Raising `max_certificate_boxes` from 64 to 256 to 1024 left five tested
  certificate-blocked queries unknown. Instrumenting all eight at 1024 boxes
  found the scan-node envelope restriction and six-level subdivision limit;
  failures occurred after only 7–48 boxes. The d23p4 0.05 fine trace also has
  84 segments, exceeding the original 64-box allowance even without refinement.
- Reducing `min_step` from .001 to .00001 at `step=.02, max_steps=300` left
  PCA 0.5/0.8, TURBO 0.8, d23p4 0.5 and n3are 0.5 unknown. Several traces advanced
  a few points while the weaker endpoint's absolute parallel derivative fell
  roughly 7–15 fold. This suggests approach to marginal roots; it does not prove
  the event type or justify tracing through it with ordinary Newton steps.

Instrument interval failures, root conditioning and event attempts before using
more periods, smaller steps or larger budgets. Profile field evaluation, scan
reuse, quadrature and certification separately, including failed attempts.

## Replace the scan-partition restriction with a proof

At the R3 baseline, `_certified_roots` rejects a transverse cell if any fixed
scan node's B envelope overlaps b. This requires a moving root to remain inside
the same arbitrary longitudinal bin. Ordinary well identity imposes no such
condition. Increasing scan density may make this prerequisite harder to satisfy.

A minimal analytic regression is

\[
B(s,\theta,\zeta)=2+\cos(\theta-\zeta),\quad
\iota=0,\quad C=3,\quad b=2.
\]

For positive C, the owned incoming root is `alpha+pi/2` and its first outgoing
root is `alpha+3*pi/2`. There is one regular owned well throughout the test cells.
Use `s_bounds=(.4,.6)` and alpha cells centered at 0 and .03, with full widths
`1e-3` and `1e-7`. The baseline fails at center 0, where roots align with scan
nodes, and certifies at .03. The replacement must certify all four. Add the
physical reverse-orientation and equivalent lifted-state checks without relying
on root-index proximity. The field can be represented by `SyntheticFourierField`
with modes `(m,n)=(0,0),(1,1)`, cosine coefficients `[[2],[1]]`, zero sine
coefficients, and constant iota/G/I coefficients `[0]`, `[3]`, `[0]`.

Moving root tubes, movable isolating brackets or coalesced scan bins are reasonable
implementation choices. Whichever is used must establish, over the whole
transverse cell or guarded contour segment:

1. Validated opposing endpoint signs and nonvanishing oriented derivative in
   each ordinary root bracket, providing existence, uniqueness and continuation.
2. Ordered incoming and first-outgoing roots in the physical direction, including
   seam/radial lifts and the transition from one certified bracket to the next.
3. Exclusion of every possible B=b barrier between them. A complete catalogue of
   irrelevant below-b extrema is unnecessary; skipping an uncertain interior is
   forbidden. Preserve adversarial hidden-barrier and multiple-crossing tests.

R3 needs this proof for the selected well and guard regions. Its current padded
multi-period scan also certifies unrelated roots outside that well, so unrelated
ambiguity can veto the query. Localize that proof, while separately preserving
R2's complete owned-root census and explicit unknown complement. Certificate
reuse must include field identity, b, lift, domain and error controls.

Tighter local Fourier/radial bounds and recentered charts may reduce pessimism.
For example, bound `m*iota-n` locally without unnecessarily discarding its
cancellation. A sampled derivative/Hessian or a successful Newton residual is
not a validated interval bound. Axis-touching cells may require the existing
regular disk-chart treatment or an explicitly uncertain core under DESIGN §7.3;
never use a small core's local area to dismiss its global connectivity influence.

## Improve bounded seeds and local continuation

Use sampled support information to prioritize the search, with explicit fallback
coverage and budgets; a sampled maximum cannot prove emptiness. Search multiple
transverse locations and distinct owned wells, and reuse/resume R1 catalogues as
windows grow. Preserve the original query cohort. Record supplemental interior
and near-edge seeds separately, with reproducible selection rules. Do not replace
the original interior workload with easier near-edge queries.

Use safeguarded/bracketed ordinary root solves and adaptive contour correction
when the roots are regular. Near a marginal endpoint or an emerging interior
barrier, switch to locating the physical event using B=b, D_parallel B=0 and
the incident contour's action condition. Preserve the radial C and shear terms.
Independently check the solution and incoming branch rather than classifying a
Newton failure itself as a transition.

Automatically discovered generic events must bind all incident root-labelled
ports at the same physical parameter, verify limiting action partition and
one-sided ordinary actions, adopt each port's action and continue away from the
marginal neighborhood into regular states. Preserve missing-branch uncertainty
and positive-witness/negative-closure asymmetry. Respect finite work budgets;
revisiting events or exhausting continuation does not prove inaccessibility.

R3.5 owns these local discoveries and continuations. Full event-curve coverage,
parameter bins/preimages, global finite-bin cycles, weighted bounds, persistence
and outer f integration remain R4–R8. A correctly diagnosed global unresolved
event chain is allowed where the specific R3.5 real gates do not require a
classified terminal; it must remain unknown.

## Fixed real probes and validation outputs

Use the exact file names, b values and initial root pairs in the R3 JSON. Those b
values derive from R0's sampled radially global extrema estimates; preserve that
scope and do not silently recompute pitches in a before/after comparison. The
eight certificate-blocked fine queries are:

| Field | lambda_n | Original seed (s, alpha) | Numerical terminal only |
| --- | ---: | --- | --- |
| PCA | 0.1 | (0.8, 0) | closed |
| TURBO | 0.5 | (0.3, 0) | closed |
| DMercFail | 0.05, 0.1, 0.5, 0.8 | (0.3, 0) | closed |
| d23p4 | 0.05 | (0.3, 0) | closed |
| d23p4 | 0.1 | (0.3, 0) | edge intersection |

DESIGN requires classified terminals for the DMercFail 0.8 and d23p4 0.1
queries at both controlled resolutions. Determine the expected classification
from independent physical/root/action checks: a baseline numerical closure is
not a proof of inaccessibility. Retain the already classified DMercFail near-edge
query at `(0.95,-0.1577992744671513)` as a regression, not a substitute for either
interior query.

For the real automatic-event regression, use the known DMercFail 0.8 generic
event near `(s,alpha)=(0.8,-0.1577992744671513)` as an independent reference.
Construct and record an ordinary incident seed on the appropriate action contour.
The production query receives that seed, not the exact event. Verify discovery,
all incident ports and continued ordinary states. A separate marginal-root solve
and one-sided action evaluation may know the reference event. Do not require a
classified global terminal merely to establish this local event test.

Build real atlas corridors for the classified paths and ordinary segments on
all five fields, with adjacent positive-multiplicity cells and explicit matching.
Use adaptive subdivisions where needed, while reporting known and unknown area
on the same original R2 parent domains. Count owned positive-well measure
separately from certified empty area. An increase in the number of tiny cells is
not evidence of a larger certified region. Sampled K-weight/population estimates
may aid diagnosis, but retain their scope and never call them enclosures.

Create separate R3.5 artifacts, for example
`docs/validation/r3_5_feasibility_matrix.py`, `r3-5-feasibility-matrix.json`,
`r3-5-feasibility-matrix.md` and diagnostic plots. Preserve all original R2/R3
reports and JSON. The new evidence should include:

- All five files times six original radially global pitch levels, at two
  controlled contour/atlas refinements, with exact original-seed comparisons.
- Supplemental seed-search locations, windows, owned root identities, selected
  outcomes and failures; report per-query counts and per-physical-case counts
  separately, without counting multiple seeds as extra physical cases.
- Structured failure reasons and the mechanism-specific disposition of all
  eight probes above; original and revised classifications, action drift/error
  estimates, independent references and scope for every claimed witness/closure.
- Fixed-domain certified positive-well/empty/unknown coverage, adjacent-cell
  root/ownership evidence for corridors and disjoint adaptive descendants.
- Cold setup, stage counters, failed work, wall/CPU times, memory where useful,
  workers/cache state, h=1, code/field/configuration hashes and active 60/300/600 s
  snapshots under DESIGN's R3.5 600-second physical-case guard.

The saved-evidence validator checks the actual scientific gates and provenance;
it does not rerun the expensive matrix inside CI. Keep fast production-path tests
for the underlying invariants, meaningful certificate/event mutations, and all
existing test budgets. Report independently verified obstructions and honest
remaining uncertainty. There is no R3.5 claim of a field-enclosed slice or f result.

## Completion and handoff

Use DESIGN §23's explicit exit gate. A table of 30 recorded failures or successful
seed discovery does not complete this milestone. Do not invent a replacement
success percentage, widen action tolerances, drop hard pitches, or present small
quad residuals as root/field certificates. If a required real result remains
unresolved, leave R3.5 unchecked and follow the draft-PR/ADR process with the
specific failed mechanism and evidence.

Once the named tests, required real evidence, mutations and GitHub Tests pass,
mark only R3.5 complete and leave a concise R4 handoff identifying the certified
atlas corridors, classified oracle queries, controls and remaining error scope.
R4 can then compare its ordinary finite-bin bounds against those real queries.

Suggested implementation request:

> Implement milestone R3.5 from docs/DESIGN.md §23, following
> docs/plans/r3-5-feasibility.md and the milestone skill. Preserve the R2/R3
> evidence and existing scientific acceptance criteria. Complete the named tests,
> real feasibility gates, mutation checks, validation report and PR workflow.
> Implement R3.5 only; do not begin R4 or merge without authorization.
