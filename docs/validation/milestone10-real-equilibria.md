# Milestone 10 real-equilibrium validation

This report records the constrained-cut sweep required after Milestone 10. It
is diagnostic and convergence evidence; the named machine-checkable acceptance
tests remain in `test/test_mesh_cut.py`.

The matrix below was regenerated on 2026-08-30 after ADR 0005 (certified
critical-polyline reversal repair) and the review-directed cut hardening
(decisive side-assignment margin, per-transition conflict demotion, component
provenance, post-insertion snap re-check) landed. Every status count, every
geometry-unresolved reason, and every invariant reproduced the original run
exactly, so the tables below stand for both runs; the sections at the end
record what the repair changed outside the matrix grid.

## Matrix and controls

The reproducible driver is `examples/validate_cut_equilibria.py`. The primary
run covered all 100 combinations of five equilibria, five radially global
bounce levels, two background meshes, and two surface extractors:

```text
lambda_n = 0.05, 0.1, 0.5, 0.9, 0.95
background = structured (6, 24, 12), gmsh (target size 0.3)
extractor = MarchingTetrahedraExtractor, PyVistaSurfaceExtractor
transition samples = at most 8 authoritative critical-curve vertices
field-period cap = 128
action order = 32, adaptively doubled through 512
cut anchors per mapped interval = 8
minimum resolved T-to-EDGE width = 0.1 local edge scales
```

The global field bounds and upstream topology reproduced the Milestone 9
validation exactly. A complete run is:

```bash
.venv/bin/python examples/validate_cut_equilibria.py \
  --output /tmp/milestone10-cuts.json
```

## Primary results

All 100 cases completed without an exception in 1,024.6 seconds. They produced
164 transition curves.

| Quantity | Result |
| --- | ---: |
| Surface extraction status | 94 regular, 6 unresolved |
| Critical-curve status | 70 regular, 22 degenerate, 8 unresolved |
| Transition-curve status | 6 regular, 57 multiway, 92 unresolved, 9 max-periods |
| Pickle-free topology round trips | 100 / 100 exact |
| Invalid port-to-sheet incidences | n/a (no valid real ports) |
| Triangles spanning a duplicated parent/child action jump | n/a (no valid real ports) |
| Unhandled exceptions | 0 |

The 158 nongeneric or failed curves retained three unresolved ports with
`sheet_id=-1`, their mapped actions (including `NaN` failures), and an explicit
reason. No missing transition was interpreted as no connection.

The six locally regular curves all occur for the TURBO equilibrium at
`lambda_n=0.1`. The coarse surfaces do not resolve the thin sheet between the
mapped companion curve and the plasma edge:

- the four structured arms (two extractors) place interior `T` samples on the
  `EDGE` boundary;
- the two Gmsh curves (two extractors) place `T` 0.1027 logical units from the
  nearest incoming-surface triangle, while the locality allowance is 0.006709.

Those curves therefore also remain explicit geometry-unresolved hyperedges.
The cut stage creates no zero-width sheet and does not abort the pitch slice.
Consequently the real coarse matrix has no valid duplicated ports; the actual
generic cut is exercised by the production-path analytic test rather than by a
plausible under-resolved real cut.

## Targeted resolution probes

The resolution controls used for the converged Milestone 9 topology were also
tried for TURBO at `lambda_n=0.1`:

- structured `(10, 40, 20)`: both extractors reduce to one locally regular
  curve, but its minimum `T`-to-`EDGE` width is `3.743e-3`, below the explicit
  local requirement `4.376e-3`;
- Gmsh target size `0.2`: both extractors reduce to one locally regular curve,
  but an interior `T` sample still collapses onto `EDGE`.

Thus the transition correspondence converges sooner than the thin geometric
strip needed to represent its cut. This is a background/surface-resolution
control, not evidence that the missing strip has zero area. Production callers
must refine until the cut resolves or carry its weight into later bounds.

## Problems found and fixed

1. Recursive midpoint insertion did not terminate reliably on curved level
   surfaces. The final algorithm inserts authoritative mapped samples, adds
   locally projected anchors, splits crossed edges, uses deterministic local
   edge flips, and retains the DESIGN.md §10.3 mesh-aligned edge path only as a
   finite fallback.
2. Independent local paths could overlap and form a branched cut graph. Closed
   detours are erased before their edge set becomes authoritative.
3. Child-3 critical vertices begin with an ordinary tangent-trace `NaN` action.
   The cut now writes the finite mapped child-3 limiting action instead of
   propagating `NaN` or inventing zero.
4. Standalone refined `GAMMA_MAX` samples can sit slightly off the chordal
   `G_ZERO` boundary. Their insertion is restricted to provenance-tagged edges,
   preventing an interior critical port.
5. Empty incoming surfaces originally lost their required `(0, 3)` array
   shapes during serialization. Empty points, triangles, and provenance arrays
   now retain their schemas and round-trip exactly.
6. A coarse mesh could make `T` appear to coincide with `EDGE` or lie far from
   the surface. Both cases now become explicit geometry-unresolved transitions
   before any mesh mutation.

No tolerance was loosened, no failed trace received zero action or weight, and
no disconnected components were merged by Euclidean proximity.

## 2026-08-30 regeneration after ADR 0005 and cut hardening

The full 100-case matrix was re-run with the same controls on the ADR 0005
branch. Outcomes identical to the original run: surface statuses 94/6,
critical statuses 70/22/8, transition statuses 6 regular / 57 multiway /
92 unresolved / 9 max-periods, zero exceptions, 100/100 exact round trips, no
invalid port incidence, no triangle spanning a parent/child jump, and the same
164 explicit unresolved reasons (158 failed-sample-or-contact gate, the four
structured TURBO interior-`T`-on-`EDGE` cases, the two Gmsh TURBO off-surface
cases). The matrix still resolves no cut: its binding constraints are the
whole-curve resolvability gate (milestone 10.2) and the TURBO thin strips
(milestone 10.3), not the ADR 0005 zigzags.

Certified reversal repair activity inside the matrix, measured over the
structured half (5 files x 5 levels x 2 extractors): 14 reversals repaired
and 29 left unrepaired, with no status changing in either direction. The
unrepaired reversals sit on `GAMMA_MIN` polylines near the thin tubes at
`lambda_n = 0.05` — curves that bear no transitions — except one `GAMMA_MAX`
reversal on the `20260406-01-262` `lambda_n = 0.05` surface whose extraction
is already explicitly `UNRESOLVED` upstream. On the transition-bearing path,
every qualifying reversal certification succeeded.

## The ADR 0005 reference case (outside the matrix grid)

The DMercFail `20260406-01-262` equilibrium at `lambda_n = 0.8` — the case
ADR 0004/0005 were written on, not part of the 25 file/level pairs above —
now cuts end to end with **default** sampling:

- structured `(6, 24, 12)` + marching tetrahedra: 2 sheets, 166 duplicated
  cut edges, all 3 ports on valid sheets, port actions exact on the mesh;
- the same at 8, 16, and full (16-vertex) sampling: the identical 2-sheet
  graph, so the cut is no longer sampling-tuned (`--max-curve-samples 8` was
  required before the repair);
- structured + PyVista extraction (17-vertex repaired curve) and Gmsh
  target 0.3 + marching (9-vertex curve): the same 2-sheet topology with all
  ports valid.

The fast tier pins this case in
`test_dmerc_reference_zigzag_curve_cuts_at_default_sampling`.
