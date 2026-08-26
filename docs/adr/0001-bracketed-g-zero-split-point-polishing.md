# ADR 0001: Polish g=0 split points with bracketed fallbacks, not Newton alone

- **Status:** Proposed
- **Date:** 2026-08-26
- **Milestone:** 5 (robustness follow-up on branch `more_robust_surfaces`)
- **Design sections:** §8.3 (surface extraction), §21.2 (forbidden numerics)

## Context

DESIGN.md §8.3 requires the second marching-triangles pass to split each surface
triangle along \(g = \mathbf b\cdot\nabla B = 0\), but does not specify how the
simultaneous root \(B=b\), \(g=0\) on a surface edge is located. The original
implementation ran an unconstrained two-equation Newton solve (`scipy` `hybr`) in the
affine plane spanned by the edge chord and \(\nabla B\), then rejected non-local or
non-converged results with `ROOT_FAILURE`.

For the W7-X reference file at \(b\) near \(\min B\) this failed reliably. The
\(B=b\) level set is there a thin helical tube whose diameter is smaller than a
background edge; measured on the failing edge at \(b=2.44736\) (structured mesh
6×24×12), the chord between the two surface endpoints dips to \(B-b=-1.6\times10^{-2}\)
mid-edge while \(|\nabla B|\) collapses from 0.67 at the endpoints to 0.095, and the
Newton solve escaped to a genuine but distant root at parameter \(-6.3\) chords with
normal displacement 1.65 (locality limit 0.21). Nothing in §8.3 answers what to do
when the local Newton solve escapes, and §21.2 forbids hiding the failure.

## Options

1. **Keep Newton-only and refine the background mesh near \(\min B\)** — leaves the
   extraction fragile at every resolution (the tube can always be thinner than the
   mesh for \(b\) close enough to \(\min B\)); pushes the burden onto every caller.
2. **Bracketed fallback chain (implemented)** — keep the planar Newton solve as the
   fast path, validated by the same locality and residual checks; on rejection fall
   back to (a) `brentq` on \(g\) along the edge chord with each sample projected onto
   \(B=b\) by a damped, displacement-capped gradient descent (single-sheeted case),
   and if the projection jumps sheets, (b) predictor–corrector continuation of the
   intersection curve of \(B=b\) with the plane through the edge, traced in both
   orientations with orientation continuity, until the bracketed sign change of
   \(g\) is met and polished on a short single-sheeted sub-chord. Every stage is
   bracketed by the opposite-sign endpoint values and bounded by displacement caps
   and an arc-length budget, so a distant root cannot be returned; total failure
   still raises `ROOT_FAILURE`. Costs: more code, and hard edges spend up to a few
   hundred field evaluations (measured: the worst level in a 100-level W7-X sweep
   took 6.8 s total versus ~2 s typical).
3. **Globally subdivide any triangle whose split point fails** — changes mesh
   topology during extraction, complicating parent-edge provenance and the seam
   contract, and still needs a robust root-finder for the subdivided edges.

## Decision

Option 2, implemented at the researcher's direction in the session of 2026-08-26
("I need the pitch surface extraction to be robust, so these failures reliably don't
occur for any value of b between min(B) and max(B)"). Researcher acceptance of this
record is pending.

## Consequences

- A 100-level sweep of \(b\) across \((\min B, \max B)\) for the W7-X reference file
  on the example script's default background mesh extracts every level; the three
  levels reported failing (2.34479, 2.39608, 2.44736) are included in the sweep and
  in `test_w7x_split_points_polish_on_the_thin_tube_near_min_B`.
- Split points on strongly curved sheets may legitimately lie farther from the edge
  chord than the planar solve's normal-displacement limit; the fallback's locality
  guarantee is path-connectedness on the surface within a bounded arc length
  (16 edge lengths), not distance to the chord.
- `test_g_zero_projection_rejects_a_root_beyond_the_local_edge` is replaced: a
  rejected planar root now recovers through the fallback
  (`test_g_zero_planar_runaway_root_is_replaced_by_a_local_bracketed_root`), and
  total failure still raises
  (`test_g_zero_polish_raises_when_no_local_root_exists`).
- The fallback chain lives in `_polish_g_crossing_bracketed`,
  `_brentq_projected_chord`, `_project_to_level_near`, and
  `_trace_plane_curve_to_g_zero` in
  `alpha_analysis/j_connectivity/surface_extract.py`; later refinement milestones
  (§8.4) inherit split points whose residuals still satisfy the §8.3 tolerances.
