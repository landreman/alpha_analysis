# ADR 0005: Companion cuts inherit sub-resolution zigzags from GAMMA_MAX polylines

- **Status:** Proposed
- **Date:** 2026-08-30
- **Milestone:** 10 (follow-up; discovered while implementing ADR 0004)
- **Design sections:** §8 (critical curves), §10.2 (companion curve), §10.3
  (align \(T\) with the triangular mesh), §21.2 (forbidden numerics)

## Context

Implementing ADR 0004 (open-endpoint snapping) made the open companion cut on
`boozmn_20260406-01-262-…_DMercFail_…_low_resolution.nc` at \(\lambda_n=0.8\)
(structured \((6,24,12)\), marching tetrahedra) geometrically insertable for the
first time — and exposed that the mapped companion polyline **doubles back on
itself**. On the 16-vertex `GAMMA_MAX` curve:

- vertices 0→1→2 run outward 0.0494 and back 0.0956 along the same line
  (turning cosine −0.999): vertex 0 sits at \(s=0.9005\) but vertex 1 at
  \(s=0.9888\) — essentially on the domain edge — so the polyline's *head*
  excursion reaches the boundary and returns, and the "endpoint" \(4.6\times
  10^{-2}\) short of the EDGE polyline is the *second*-outermost vertex;
- vertices 6→7→8 zigzag with a backward step of \(5.8\times10^{-3}\) (cosine
  −1.000), far below the \(\approx 0.2\) local mesh edge.

The **marginal** (`GAMMA_MAX`) polyline shows the same reversals at the same
indices, and parent-port \(s\) equals marginal \(s\) to machine precision, so
the transition map is faithful: the zigzag is inherited from the critical
curve's mesh-edge chain built in milestone 8, not created in milestone 9. All
16 samples are `REGULAR` and additivity holds to \(1.9\times10^{-13}\).

Consequence at cut time: two nearly coincident anti-parallel strands bound a
strip of essentially zero width. Inserting the polyline overlaps its own
constrained chain, the blocked-edge graph grows a degree-4 branch vertex, and
edge-blocked union-find finds three incident sides where the generic split
demands two. Before this ADR's safeguard the cut crashed
(`ConstrainedCutError: … exactly two incident triangle sides; found [0, 1,
2]`); silently splitting or flipping the overlapped chain instead produces a
dangling cut and a plausible, wrong sheet graph — the §21.2 failure mode. Mesh
refinement cannot clear it: the strand separation is \(\sim 0\) at any
resolution.

The problem is sampling-dependent: `max_curve_samples=10` keeps the degenerate
pair (7, 8) and fails; `max_curve_samples=8` happens to skip vertices 1, 7 and
8, and the very same transition then cuts end to end — 2 sheets, 134
duplicated cut edges, all three ports on valid sheets, no triangle spanning
the \(11.1\) parent/child action jump, port actions exact on the mesh.

## Options

1. **Detect the double-back at cut time and report the transition
   geometry-unresolved (implemented as the interim safeguard)** — a
   consecutive-sample triple that reverses direction (negative dot product)
   with strand separation below `min_transition_strip_edge_ratio ×` the local
   edge scale returns an explicit reason naming the samples. Honest and cheap,
   no invented topology; but the reference file's default sampling stays
   unresolved forever, and "refine the background mesh" guidance would be
   wrong here — no refinement clears an exact retrace.
2. **Fix the polyline upstream in `extract_critical_curves`** — the mesh-edge
   chain steps through near-degenerate (sliver) configurations that put an
   interior vertex beyond its neighbor; reorder or straighten where a local
   swap removes a >90° reversal without changing the vertex set, validating
   against the true \(g=0,\,B=b\) curve. The honest end state: the head
   reversal is an ordering artifact (order 1→0→2 is monotone), and fixing it
   would put this curve's true endpoint *on* the EDGE boundary, shrinking the
   ADR 0004 gap at its head to \(\sim 5\times10^{-3}\). Cost: milestone-8
   output (vertex order, hence `u`, `total_u_length`) changes, and everything
   milestone 9 recorded on real equilibria shifts with it.
3. **Collapse sub-resolution zigzags at transition-mapping time** — drop or
   merge samples whose step reverses at below-mesh scale before mapping.
   Keeps the critical curve authoritative but silently edits the transition's
   sample set; without an explicit record this is adjacent to the §21.2
   plausible-number trap.
4. **Choose `max_curve_samples` subsets that avoid degenerate pairs** — works
   today by luck (the 8-sample subset), but sampling-dependent correctness is
   exactly the fragility §21.3 warns about; rejected as a resolution, useful
   only as a demonstration.

## Decision

Left blank until the researcher decides.

## Consequences

- Interim safeguard (option 1) is in `_geometry_resolution_issue` with test
  `test_sub_resolution_double_back_is_not_cut`; the branched-graph crash and
  the silent chain-destruction path are both closed (see also the constrained
  chain protection recorded in ADR 0004's consequences).
- The ADR 0004 reference demonstration currently needs `--max-curve-samples 8`
  on the DMercFail file; with the default 10 samples the transition reports
  `companion T doubles back on itself at samples 3->4->5 with strand
  separation 7.730e-04, below the local resolution requirement 1.227e-02 …`
  explicitly.
- If option 2 is taken, the milestone-9 real-equilibrium sweep in
  `docs/validation/milestone9-real-equilibria.md` must be regenerated, and the
  ADR 0004 endpoint-gap numbers rechecked, since curve ordering feeds both.
