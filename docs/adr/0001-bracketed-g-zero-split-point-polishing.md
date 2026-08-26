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
   intersection curve of \(B=b\) with planes through the edge. The continuation
   traces a pencil of cutting planes rotated about the chord (seeded by the
   \(\nabla B\) direction — a single plane can meet a thin tube in two disconnected
   curves, one per endpoint, with no bracket on either), in both directions per
   plane, polishing every sign change of \(g\) on its short single-sheeted
   sub-chord, and returns the crossing **nearest the edge** — the first crossing
   met can be a spurious one many edge lengths away on the long way around the
   tube. Every stage is bracketed and bounded: displacement caps, an arc-length
   budget of 8 locality scales per direction, rejection of any candidate outside
   the unit disk (the surface patch a triangle approximates lies inside its
   background tetrahedron and hence inside the disk; beyond \(x^2+y^2=1\) the
   field is unphysical extrapolation), and a final acceptance radius of
   \(\max(4\,L_{\mathrm{edge}},\ 2\,L_{\mathrm{patch}})\), where
   \(L_{\mathrm{patch}}\) — the largest marching-triangle edge of the full
   surface — proxies the background cell size so that marching edges much
   shorter than their cell are not held to an arbitrarily small radius. Total
   failure still raises `ROOT_FAILURE`. Costs: more code, and hard edges spend
   up to a few thousand field evaluations (measured: the worst level in a
   100-level W7-X sweep took ~20 s total versus ~2 s typical).
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
  chord than the planar solve's normal-displacement limit; the fallback accepts the
  nearest in-disk crossing within \(\max(4\,L_{\mathrm{edge}}, 2\,L_{\mathrm{patch}})\)
  of the edge (measured worst case across resolutions 4×16×8 to 10×32×16 and
  \(b\) sweeps: 3.03 edge lengths on a short edge — 0.46 in logical units, under
  one background cell — with all split points strictly inside the unit disk).
  Vertices many edge lengths from their parent triangle, or outside
  \(x^2+y^2=1\), were a defect of the first first-crossing implementation and are
  regression-tested against.
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
