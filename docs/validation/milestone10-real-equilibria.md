# Milestone 10 real-equilibrium validation

This report records the constrained-cut sweep required after Milestone 10. It
is diagnostic and convergence evidence; the named machine-checkable acceptance
tests remain in `test/test_mesh_cut.py`.

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
| Invalid port-to-sheet incidences | 0 |
| Triangles spanning a duplicated parent/child action jump | 0 |
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
