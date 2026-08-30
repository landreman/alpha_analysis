# ADR 0005: Companion cuts inherit sub-resolution zigzags from GAMMA_MAX polylines

- **Status:** Accepted
- **Date:** 2026-08-30 (decided 2026-08-30)
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
   interior vertex beyond its neighbor; reorder where the **true**
   \(g=0,\,B=b\) curve certifies that the reversal is an ordering artifact,
   never on chain geometry alone. `_walk_polylines` follows extracted segment
   adjacency, so replacing order 1→0→2 by 0→1→2 changes which vertices are
   connected — only the implicit curve may arbitrate between connectivity and
   geometry, and a genuine tight fold or self-contact (a §5.4 event, live on
   a DMercFail equilibrium) must never be smoothed into a generic cut. The
   honest end state: the head reversal is an ordering artifact, and fixing it
   puts this curve's outermost vertex (s = 0.9888) at the head, shrinking the
   ADR 0004 gap. Cost: milestone-8 output (vertex order, hence `u`,
   `total_u_length`) changes, and everything milestone 9 recorded on real
   equilibria shifts with it — but the pre-repair `u` carries phantom
   back-and-forth arc length, so that recorded output is the buggy one.
3. **Collapse sub-resolution zigzags at transition-mapping time** — drop or
   merge samples whose step reverses at below-mesh scale before mapping.
   Keeps the critical curve authoritative but silently edits the transition's
   sample set; without an explicit record this is adjacent to the §21.2
   plausible-number trap.
4. **Choose `max_curve_samples` subsets that avoid degenerate pairs** — works
   today by luck (the 8-sample subset), but sampling-dependent correctness is
   exactly the fragility §21.3 warns about; rejected as a resolution, useful
   only as a demonstration.

## Diagnosis (2026-08-30, before deciding)

The context above conjectured the reversals were ordering artifacts but did
not demonstrate it, and its claim that "mesh refinement cannot clear it" was
asserted, not measured. Both were settled empirically before the decision by
walking the true \(B=b,\,g=0\) curve (the milestone-8 predictor–corrector
continuation) through every reversal window, on structured \((6,24,12)\) and
\((10,40,20)\) backgrounds with both extractors:

- every windowed vertex lies **on** the walked true curve to \(\le
  4\times10^{-4}\), two to three orders of magnitude below the local chord
  scale \(\sim 0.05\);
- the walked arcs show **no self-contact** (minimum far-pair separation
  \(\approx 0.1\), the full arc span) and no \(D_\parallel^2 B\) sign change:
  there is no genuine fold here to protect;
- ordering the windowed vertices by true arc length turns each zigzag into a
  single adjacent transposition, and at the head the true order ends at the
  \(s=0.9888\) vertex, past which the walk exits \(s>1\) — the curve really
  terminates on `EDGE`;
- at \((10,40,20)\) the zigzag **reappears** at different indices with a new
  near-duplicate vertex pair \(3.7\times10^{-4}\) apart, so refinement
  relocates the artifact rather than clearing it — the "cannot clear it"
  claim survives, now with evidence rather than assertion.

## Decision

A strengthened option 2, with option 1 retained permanently as the backstop;
options 3 and 4 are rejected. Directed by the researcher on 2026-08-30, who
commissioned the synthesized robustness plan (diagnose against the true
curve first; repair only what certification confirms; keep the guard) after
review of this ADR and an independent assessment.

`extract_critical_curves` repairs each polyline whose chain reverses at
sub-resolution scale, and the **true curve is the only arbiter**: a reversal
triple qualifies only when its strand separation is below
`repair_separation_ratio` of the local chord scale (a wider reversal is
genuine, representable geometry and is never touched), and the surrounding
window is reordered only when a bounded recorded continuation walk certifies
one simple arc — every windowed vertex within `repair_on_curve_ratio` of the
local scale of the walked arc, no self-contact between walk samples far apart
in arc length, no \(D_\parallel^2 B\) sign change — and the reorder both
removes the sub-resolution reversals in its window and preserves the window's
chain attachment points. Any certification failure leaves the chain exactly
as extracted and is counted (`reversal_unrepaired_count`); a genuine fold
fails certification by construction and remains an explicitly reportable
nongeneric feature. Repairs recompute `u` and `total_length` (removing the
phantom back-and-forth length the raw chain carried), keep the vertex set
and `segment_ids` provenance untouched, and are counted
(`reversal_repaired_count`).

Transition-mapping sample budgets must not alter which authoritative curve is
cut: the repair happens before any sampling, so every `max_curve_samples`
subset of a repaired curve is a subset of the same simple polyline. (Full
decoupling of the *inserted cut geometry* from the sample budget is the
follow-on milestone 10.1; this ADR removes the budget's ability to make the
authoritative curve itself self-overlapping.)

## Consequences

- The repair is `_repair_polyline_reversals` /
  `_certify_reversal_window` / `_walk_curve_recorded` in
  `critical_curves.py`, on by default (`repair_reversals`), with
  `test_sub_resolution_zigzag_is_reordered_by_certified_walk`,
  `test_uncertified_reversal_is_left_untouched_and_counted`, and
  `test_wide_reversal_is_not_a_repair_candidate` pinning the certified
  repair, the refusal path, and the sub-resolution gate.
- The cut-time double-back guard (option 1) stays in
  `_geometry_resolution_issue` with test
  `test_sub_resolution_double_back_is_not_cut`; the branched-graph crash and
  the silent chain-destruction path remain closed (see also the constrained
  chain protection recorded in ADR 0004's consequences). Known limit: the
  guard fires only on a strict reversal (negative step dot product); a
  near-90° turn with sub-resolution strand separation bounds the same
  un-representable strip and passes the guard. The upstream repair makes
  that combination rare, and milestone 10.1's budget-invariance checks are
  the place it would surface.
- The ADR 0004 reference demonstration no longer needs
  `--max-curve-samples 8`: on the DMercFail file at \(\lambda_n=0.8\),
  default sampling (10), 16 samples, and the full 16-vertex curve all cut to
  the same 2-sheet graph with port actions exact on the mesh
  (`test_dmerc_reference_zigzag_curve_cuts_at_default_sampling`).
- Making sample order differ from the surface's own (still zigzagged)
  `G_ZERO` chain order exposed a latent defect: a tagged path between two
  samples can pass through a third sample's mesh vertex, and
  `_insert_curve` overwrote that sample's authoritative `u` with the
  passing path's interpolation, shifting its assigned action. The
  authoritative parameter now always wins (`vertex_u.setdefault`).
- The milestone-9/10 real-equilibrium sweeps are regenerated (curve ordering
  feeds `u`, `total_u_length`, and the uniform sample subsets), and the ADR
  0004 endpoint-gap statement is superseded as described there.
