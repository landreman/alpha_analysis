# Milestone 10.1 sampling-budget validation

This report records the sampling-robust cut checks from DESIGN.md §23.10.1.
The named acceptance tests are in `test/test_mesh_cut.py` and
`test/test_transitions.py`; the real matrix uses
`examples/validate_cut_equilibria.py`.

## Budget contract and controls

`max_curve_samples` now limits uniquely mapped authoritative vertices. It is
not a uniform-subset geometry control. A bounded run maps midpoint vertices
adaptively and returns `BUDGET_INSUFFICIENT` if any required interval remains
uncertified. Such a curve retains its mapped actions and all three unresolved
ports, but inserts no cut. Numerical or nongeneric sample failures retain
their own statuses instead of being relabeled as budget failures.

The runs below used the defaults: geometry error at most
`1e-8 + 0.2 * interval_u_length`, action interpolation error at most
`1e-8 + 0.02 * max(abs(endpoint/midpoint A))`, edge proximity `1-s <= 0.02`,
and near-self-contact distance `0.1 * interval_u_length`. The action
quadrature starts at order 32 and may refine through 512; the field-period cap
is 128. Root and quadrature tolerances are unchanged from milestone 10.

These checks certify the retained PL representation relative to the existing
critical-curve vertices and detected root-scan itinerary. They do not certify
features below either upstream resolution. Milestones 10.2/10.3 still own
contact localization and failure-directed refinement.

## Named acceptance evidence

- `test_production_synthetic_sheet_graph_is_invariant_across_sample_budgets`
  is parameterized over 8, 10, 16, and full. It uses the original production
  synthetic background `(4,16,12)` and pins the independently known graph:
  three distinct sheets; the parent at `s<0.5` does not touch `EDGE`; both
  children extend to `s=1`. It also rejects triangles spanning the action
  jump. The bounded runs use 8 vertices; full maps all 32.
- `test_dmerc_reference_sheet_graph_is_budget_invariant_or_explicit` sweeps
  the same four budgets on the ADR 0005 DMercFail reference at
  `lambda_n=0.8`. It preserves the reversal-repair and exact-on-mesh port
  action checks from the previous reference test. The 8/10 budgets are
  explicitly insufficient; 16/full have the same two-sheet incidence.
- `test_transition_sampling_budget_is_explicit_when_certification_cannot_finish`
  gives a 16-vertex analytic circle only two mapped vertices, then checks
  `BUDGET_INSUFFICIENT`, retained sample count, explicit intervals/reason, and
  the conspicuous PNG diagnostic.
- `test_transition_sampling_refines_nonlinear_port_actions` uses an analytic
  marginal curve with `G=1+100s^2`, so action curvature alone can require more
  work even when geometry is accepted. The full actions are checked against
  an independent one-dimensional integral.

## DMercFail cross-backend/extractor sweep

The reference file is
`boozmn_20260406-01-262-Ax_nfp4_Garabedian_mpol2_ntor2_minx0_allNfp_aspect10_DMercFail_m0p3_eval000323_low_resolution.nc`.
Radially global refined bounds are `5.040465893072380` and
`12.050343540204448`, giving `b=10.648368010778035` at `lambda_n=0.8`.
Structured resolution is `(6,24,12)`; Gmsh target size is `0.3`.

Reproduce this sweep and its PNG diagnostics with:

```bash
.venv/bin/python examples/validate_transition_sampling.py \
  --output /tmp/milestone10.1-dmerc-budgets.json \
  --plot-dir /tmp/milestone10.1-sampling-plots
```

Each cell below is `mapped/authoritative: outcome`. `limited` means no cut,
three ports with sheet/vertex IDs `-1`, and an explicit budget reason.

| Backend / extractor | Budget 8 | Budget 10 | Budget 16 | Full |
| --- | --- | --- | --- | --- |
| structured / marching tetrahedra | 8/16: limited | 10/16: limited | 16/16: 2 sheets | 16/16: 2 sheets |
| structured / PyVista | 8/17: limited | 10/17: limited | 13/17: 2 sheets | 17/17: 2 sheets |
| Gmsh / marching tetrahedra | 8/9: limited | 9/9: 2 sheets | 9/9: 2 sheets | 9/9: 2 sheets |
| Gmsh / PyVista | 8/11: limited | 10/11: 2 sheets | 11/11: 2 sheets | 11/11: 2 sheets |

Every resolved run has the same role incidence: parent on one sheet,
child-1 and child-3 on the other. Every port action equals the value at its
cut-mesh vertex exactly (maximum absolute difference `0.0`). No backend or
extractor changed the resolved sheet graph. The meshes themselves need not be
identical: for example structured/PyVista has 188 duplicated cut edges at the
certified 13-sample result and 256 at full 17-sample mapping. Both have the
same two-sheet graph; the extra geometry is not claimed to be identical.

The budget-insufficient and full structured/marching transition diagnostics
were rendered and visually inspected. They show the matched `T`/`GAMMA_MAX`
curves, three actions, additivity residual, well profiles, and an explicit
orange budget reason rather than a plausible resolved coarse cut.

## Required five-equilibrium matrix

Reproduction command:

```bash
.venv/bin/python examples/validate_cut_equilibria.py \
  --output /tmp/milestone10.1-cuts.json
```

All 100 combinations were run: the five repository reference files,
`lambda_n = 0.05, 0.1, 0.5, 0.9, 0.95`, structured `(6,24,12)` and Gmsh
target `0.3`, and both surface extractors. The transition work budget was 8.
Summed case time was `673.2 s` (excluding shared field bounds/background
construction).

| Quantity | Result |
| --- | ---: |
| Completed cases / exceptions | 100 / 0 |
| Surface status | 94 regular, 6 unresolved |
| Critical status | 70 regular, 22 degenerate, 8 unresolved |
| Transition status | 6 regular, 46 multiway, 92 unresolved, 20 max-periods |
| Uniquely mapped / authoritative vertices | 579 / 2709 |
| Pickle-free cut-topology round trips | 100 / 100 exact |
| Invalid port incidences | 0 |
| Triangles joining duplicated parent/child ports | 0 |
| Resolved real cuts in this coarse matrix | 0 |

The upstream topology counts and 164 transition curves exactly match the
milestone-10 matrix. Adaptive sampling changes which expensive field lines
are visited: relative to its uniform subset, the aggregate `MAX_PERIODS`
count increases from 9 to 20 and `MULTIWAY` decreases from 57 to 46. No
failure is promoted to a cut. There
are no budget-insufficient curve statuses in this particular matrix: every
bounded, uncertified curve exposes a physical/numerical failure before the
budget becomes its binding reason. Sampling remains explicitly uncertified
on 88 curves, with the failed stage recorded; the other 76 map their complete
small source curves or finish certification.

As before, all 158 nongeneric/failed curves retain unresolved hyperedges.
The six locally regular TURBO curves at `lambda_n=0.1` remain
geometry-unresolved: four structured arms collapse an interior `T` sample
onto `EDGE`, and the two Gmsh curves are `0.1027` logical units off the
incoming surface against a `0.006709` locality allowance. These are the
milestone-10.3 thin-strip/background issues, not permission to manufacture
zero-width sheets. The resolved real-cut evidence is the separate DMercFail
`lambda_n=0.8` sweep above.

## Full-sampling action-stencil regression

The strengthened full-budget synthetic test exposed a latent milestone-10
insertion bug: 48 triangles contained helper vertices whose actions were
interpolated before their parent `T` vertices received branch-specific
limiting values. The sheet graph was correct, but those stale values blended
the two branches (for example `A=0.7321` beside a parent limit `A=1.1459`).

The cut now preserves the insertion interpolation DAG and reevaluates off-cut
descendants after assigning the port limits, selecting every stencil
vertex's copy on the descendant's own sheet. Authoritative port values are
never overwritten. A genuinely cross-sheet stencil remains explicit `NaN`,
not a plausible blend. The same full-budget test now has zero jump-spanning
triangles with its original tolerance. The four-backend/extractor DMercFail
sweep was repeated after this fix and reproduced the table above exactly.
The coarse 100-case matrix never inserts a cut, so it does not traverse this
post-insertion refresh path.

## Mutations

1. Replaced the action-interpolation certification condition with `True`.
   `test_transition_sampling_refines_nonlinear_port_actions` went red: an
   incorrect four-sample result became `REGULAR` instead of budget-limited.
2. Removed the regular/sampling-certified guard from the cut gate.
   `test_dmerc_reference_sheet_graph_is_budget_invariant_or_explicit` went
   red: the under-budget curve actually cut, instead of retaining its
   unresolved hyperedge.

Both mutations were reverted and both tests re-run green. No tolerance was
loosened, no check was removed, no test was marked slow/skip/xfail, and no new
dependency was added.

## Local verification budget

The initial CI run passed, but exposed a timing regression (Python 3.10 fast
tier 176.39 s). Profiling full-budget cutting found a quadratic component
membership scan: every candidate edge rescanned every triangle. A per-pass
edge-membership set preserves the same component filter without those
repeated scans. No mesh resolution, tolerance, or assertion changed.

After this optimization, in the clean Python 3.10.5 venv with three pytest
workers, `make check` passed all 165 tests and formatting in 47.55 seconds
wall (`make test-full`'s pytest portion: 45.88 s). Standalone `make test`
passed in 46.65 seconds wall (slowest individual test 16.35 s). The full-budget
synthetic test fell from 17.11 to 6.90 s. The full durations tails and CI
timings are recorded in the PR body.
