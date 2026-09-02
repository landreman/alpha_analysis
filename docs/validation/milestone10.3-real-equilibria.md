# Milestone 10.3 unattended matrix convergence

Definitive 120-case matrix (5 boozmn files × {structured, gmsh} × 
{MarchingTetrahedraExtractor, PyVistaSurfaceExtractor} × λₙ ∈ 
{0.05, 0.1, 0.5, 0.8, 0.9, 0.95}) driven by `converge_case` with the
matrix-wide `RefinementBudgets` ladders — no per-case tuning. Companion data:
`docs/validation/milestone10.3-real-equilibria.json` (per-case classification,
terminal reasons, full remediation-attempt log, cut summaries, controls with
`implementation_sha256` of the six library modules).

**Headline: 42/120 cases (35.0%) end resolved or no-transitions; 77 end
unresolved with explicit machine-readable terminals; 1 case crashed (fixed,
disclosed below). The §23 milestone-10.3 acceptance threshold (≥95%
resolved-or-no-transitions) is not met, and
`test_matrix_report_meets_milestone_10_3_acceptance` is deliberately left
red. ADR 0009 puts the decision to the researcher.**

## Controls and reproduction

Library revision `1975444` (all 120 recorded cases; the crash fix and this
report land after it). Per-case wall guard 7200 s (`--max-case-seconds`,
recorded per case as an explicit `wall_budget` terminal).

```bash
source .venv/bin/activate
mkdir -p /tmp/milestone103
for i in 0 1 2 3 4; do for be in structured gmsh; do
  for ex in marching_tetrahedra pyvista; do
    python examples/converge_cut_equilibria.py \
      --output /tmp/milestone103/sub$i-$be-$ex.json \
      --file-index $i --backend $be --extractor $ex --resume &
done; done; wait; done
python examples/merge_matrix_shards.py /tmp/milestone103/sub*.json \
  --output docs/validation/milestone10.3-real-equilibria.json
```

The recorded run sharded exactly this way (20 shards). The first pass ran all
20 shards concurrently on 4 cores; because the wall guard measures wall-clock,
oversubscription can convert compute-heavy cases into `wall_budget` terminals
early. A second, uncontended pass (4 shards at a time, same code, same
controls, same guard; `--resume` retries only `wall_budget`/`timeout` cases)
re-verified the PCA (file 0) λₙ ∈ {0.5, 0.8, 0.9} `wall_budget` terminals:
all twelve re-capped at 7200 s of near-pure compute, so those terminals are
genuine case costs, not scheduling artifacts. The remaining `wall_budget`
retries were cancelled when the researcher capped further per-case compute
(any case not finishing in under 10 minutes is counted failed), which the
recorded 7200 s terminals already satisfy a fortiori.

## Outcomes

`R` resolved (sheets + fluxes), `-` no transitions at this level, `U`
unresolved with explicit terminal reasons, `X` exception (crash; disclosed
below). Columns: structured:MT, structured:PV, gmsh:MT, gmsh:PV.

| file (index) | λₙ=0.05 | 0.1 | 0.5 | 0.8 | 0.9 | 0.95 | dominant unresolved class |
|---|---|---|---|---|---|---|---|
| PCA `…eval000290` (0) | `- - - -` | `U U U U` | `U U U U` | `U U U U` | `U U U U` | `U U U U` | event_geometry (0.1), wall_budget (≥0.5) |
| TURBO `…eval000155` (1) | `- - - -` | `U U U U` | `R U R R` | `U U U U` | `U U U U` | `U U U U` | cut_conflict (0.1), wall_budget (≥0.8) |
| DMercFail `…eval000323` (2) | `R R U U` | `- - - -` | `- - - -` | `R R R R` | `U U U U` | `U U U U` | unresolved_extraction (gmsh 0.05), max_periods (≥0.9) |
| d23p4 (3) | `- - - -` | `- - - -` | `U U X R` | `U U U U` | `U U U U` | `U U U U` | wall_budget (0.5), max_periods (≥0.8) |
| n3are (4) | `- - - -` | `- - - -` | `U U U U` | `U U U U` | `U U U U` | `U U U U` | wall_budget |

Classification totals: 32 `no_transitions`, 10 `resolved`, 77
`unresolved_explicit`, 1 `exception`. Resolved-or-no-transitions: 42/120 =
35.0%. Total recorded case time 112.6 h.

The ten resolved cases: TURBO λₙ=0.5 on 3 of 4 combos (structured:PV demotes
with "companion cut does not separate the surface"), DMercFail λₙ=0.05 both
structured combos and λₙ=0.8 on all four combos (2 sheets, matching the
milestone-9 reference), d23p4 λₙ=0.5 gmsh:PV.

## Failure classes and terminal reasons

Per-case dominant failure classes over the 77 unresolved cases (counts from
`summary.failure_class_counts`; a case can record several):

| class | cases | meaning at the ladder bound |
|---|---|---|
| `wall_budget` | 46 | per-case 7200 s wall guard exhausted before the remediation ladders finished |
| `max_periods` | 20 | field-line scans still capped at 1024 field periods (background-exempt by design; DMercFail ≥0.9 and d23p4 ≥0.8 — the documented long-well physics recorded since milestone 9) |
| `event_geometry` | 6 | localized count-change certifies as neither equal-height (ADR 0003) nor below-b fold (ADR 0007) at root-scan factor 4 (PCA λₙ=0.1 structured) |
| `cut_conflict` | 4 | TURBO λₙ=0.1: critical point 5.154e-02 from the nearest tagged edge of its component after four local-refinement rounds — a genuine component-topology limit, not a resolution limit |
| `empty_arc_interval` | 2 | PCA λₙ=0.1 gmsh: event interval retains no regular arc sample at full source budget |
| `unresolved_extraction` | 2 | DMercFail λₙ=0.05 gmsh: 14 sheet-bridging splits stay unresolved (ADR 0001 thin tube) at every background level |
| `background_geometry` | 2 | non-separating companion slit (TURBO λₙ=0.5 structured:PV) and one PCA λₙ=0.1 curve-geometry demotion at background level 2 |

Remediation-attempt usage across the matrix (every retry recorded per case in
`attempts`): `max_periods`→cap escalation ×40, background escalation ×31
(across five trigger classes), thin-strip local refinement ×15, source-budget
escalation ×13, local projection re-solves ×8, scan-resolution escalation ×4,
contact-localization bisection escalation ×2. Escalations are bounded by the
matrix-wide `RefinementBudgets`; every unresolved case sits at the relevant
ladder bound with its terminal reasons recorded.

## Backend and extractor consistency

For the 19 (file, λₙ, backend) groups with transitions where both extractors
recorded a comparable outcome, 17 agree on the sheet signature
(or on the identical terminal-reason set). The two disagreements are
resolution-marginal, not silent: PCA λₙ=0.1 structured (MT records 4
`event_geometry` terminals, PV records 2 plus a curve-geometry demotion) and
TURBO λₙ=0.5 structured (MT resolves, PV demotes the non-separating slit
explicitly). Resolved cases agree across all four combos wherever more than
one combo resolves (DMercFail λₙ=0.8: 2 sheets, equal fluxes, on all four).

Event kinds localized across the matrix: 48 `degenerate_endpoint` (ADR 0008),
7 `equal_height` (ADR 0003), 3 `fold` (ADR 0007).

## The one crash, and its fix

`3:0.5:gmsh:marching_tetrahedra` crashed after 6455 s with
`ConstrainedCutError: arc 7 vertex 757 … u=0.0 lacks sides (1, 0) … tags=8`:
an arc endpoint pinned to the mesh EDGE whose parent-side triangle fan
collapsed at a corner, detected only in the duplication stage — after the
per-transition demotion loop — where nothing could demote it, so the case
aborted instead of terminating explicitly. The fix
(`_stabilized_side_demotion` in `mesh_cut.py`, regression-tested by
`test_stabilized_side_check_demotes_instead_of_raising`) runs the same
membership predicate as the duplication stage as a last-resort demotion
criterion once the loop stabilizes, so the transition is withdrawn with a
recorded reason (§21.2) and the cut proceeds. Because the check is evaluated
only at the stabilized labels and uses exactly the predicate whose failure
used to raise, any case that completed under the matrix revision is untouched
by construction; the JSON keeps the honest `exception` record for the run as
executed (the fix landed after the matrix revision, and the case would remain
an over-budget failure under the researcher's 10-minute cap regardless).

## Relation to the acceptance criteria

- "Each case either resolves or terminates with a physically meaningful
  recorded reason, with per-class counts": met for 119/120 by the recorded
  terminals above; the 1 exception case is the disclosed gap, fixed and
  regression-tested at HEAD.
- "No per-case parameter tuning": met — one `RefinementBudgets` for all 120
  cases; the controls block would reject drift.
- "≥95% of cases resolved or no-transitions": **not met** (35.0%). The
  shortfall is dominated by genuine per-case compute beyond the recorded
  budget (46), the documented field-period ceilings (20), and 11 genuine
  geometry/certification terminals. `test_matrix_report_meets_milestone_10_3_acceptance`
  encodes the criterion and stays red; ADR 0009 lays out the options.
