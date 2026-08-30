# ADR 0004: Snap open companion-cut endpoints onto the EDGE boundary

- **Status:** Accepted
- **Date:** 2026-08-30
- **Milestone:** 10 (follow-up on the constrained-cut machinery merged in PR #19)
- **Design sections:** §10.3 (align \(T\) with the triangular mesh), §10.5 (sheet
  graph), §21.2 (forbidden numerics)

## Context

An open `GAMMA_MAX` curve terminates on the domain `EDGE` boundary, and its companion
curve \(T\) terminates on the same flux surface \(s\) because field lines preserve
\(s\) — so the true \(T\) also ends on the `EDGE` boundary. The mapped endpoint,
however, only lands *on* the piecewise-linear boundary polyline when it happens to
fall slightly outside the PL surface (local projection then clamps it onto a boundary
edge). An endpoint falling slightly inside lands in a triangle interior, the inserted
cut polyline dangles, and edge-blocked union-find finds one incident component
instead of two: before the 2026-08-29 screen, `_branch_components` crashed with
"must have exactly two incident triangle sides; found [0]", and with the screen the
transition becomes a geometry-unresolved hyperedge
(`test_open_T_endpoint_short_of_edge_is_not_cut`).

The numbers make this a decision, not a corner case. On
`boozmn_20260406-01-262-…_DMercFail_…_low_resolution.nc` at \(\lambda_n=0.8\), the
open companion endpoint misses the PL boundary by \(4.6\times10^{-2}\) (logical
units) on the default structured \((6,24,12)\) mesh and by \(2.3\times10^{-2}\) at
\((10,40,20)\): the gap shrinks linearly with resolution and generically never
reaches zero, so on real equilibria an open cut would *almost never* resolve at any
affordable resolution. §10.3 prescribes inserting \(T\) as a constrained polyline and
splitting crossed edges, but is silent on how an open curve's terminal segment meets
the surface boundary.

## Options

1. **Status quo — geometry-unresolved until the endpoint lands exactly on `EDGE`** —
   safe and explicit, but the condition is measure-zero: real open transitions stay
   unresolved hyperedges forever, and milestones 11/13 inherit "unresolved
   connectivity" for cuts whose geometry is actually known to within a fraction of
   one mesh edge. That starves the connectivity metric of physically real
   transitions — a systematic bias, not a conservative bound.
2. **Extend the inserted cut to the nearest `EDGE` boundary edge when the endpoint
   gap is within the resolution-scaled tolerance already used for surface distance
   (`max_surface_distance_ratio * max(local edge scale, snap_tolerance)`)
   (implemented)** — after the authoritative samples are inserted, the closest point
   on the nearest `EDGE`–`EDGE` boundary edge is inserted (splitting that boundary
   edge, the new vertex inheriting the `EDGE` tag), and one extra constrained
   segment connects the endpoint sample to it. Every vertex on the extension takes
   the endpoint's parameter \(u\) (clamped), so its duplicated copies take the
   endpoint sample's parent/child-1 actions — constant, hence still piecewise linear
   and monotone-or-constant per §10.4. The authoritative port samples, their
   \(u\) values, `total_u_length`, and source vertex IDs are untouched; the
   extension is sub-resolution surgery on the PL mesh, of the same character as the
   edge splitting §10.3 already prescribes. An endpoint farther than the tolerance
   remains geometry-unresolved exactly as today. Cost: the reported cut is longer
   than the mapped polyline by up to half a local edge, and the extension's action
   is a clamped constant rather than a mapped value — both are below the resolution
   at which the surrounding mesh represents anything.
3. **Move the authoritative endpoint sample itself onto the boundary** — smaller
   code change, but it mutates mapped geometry: the port sample no longer sits where
   `map_transitions` evaluated its actions, the sample-to-critical-vertex
   correspondence breaks, and the perturbation is silent. Rejected for corrupting
   authoritative data where option 2 only adds explicitly clamped scaffolding.
4. **Map the true terminal segment by extending the critical curve beyond its last
   vertex until \(T\) exits the PL surface** — the physically complete answer, but it
   needs new critical-curve continuation and field-line mapping machinery, costs one
   full transition-mapping trace per endpoint (tens of seconds each on real
   equilibria), and the segment it recovers is shorter than a mesh edge — below the
   resolution of everything that consumes it.

## Decision

Option 2, directed by the researcher when commissioning this change (2026-08-30
instruction: snap within a `max_surface_distance_ratio`-scaled tolerance, split the
boundary edge, clamp \(u\)/action on the extension to the endpoint sample).

## Consequences

- `_geometry_resolution_issue` keeps the open-endpoint screen but only fails an
  endpoint whose boundary gap exceeds
  `max_surface_distance_ratio * max(edge_scale, snap_tolerance)` — the same local
  allowance used for off-surface samples, so no new configuration knob. The
  unresolved message now records both the gap and the tolerance.
- `_insert_curve` gains a `snap_open_ends` mode used only for the companion
  (parent/child-1) polyline, never for the tagged `child_3` curve, and only when the
  curve is open. An endpoint whose inserted vertex already carries the `EDGE` tag is
  left alone. Extension vertices enter `vertex_u` at the clamped endpoint \(u\), so
  vertex duplication assigns them the endpoint's parent/child-1 actions.
- Port sample counts, `polyline_vertex_ids`, action arrays, and `total_u_length` are
  unchanged; `cut_edges` additionally contains the extension segments, which is what
  makes `_branch_components` find two sides and the sheet graph split.
- `test_open_T_endpoint_short_of_edge_is_not_cut` now drives the beyond-tolerance
  branch via a tightened `max_surface_distance_ratio` (the synthetic grid's edges
  are so coarse that every interior point is within the default allowance); a new
  test pins the snapped case: three sheets, a duplicated boundary terminus, and
  clamped actions on the extension.
- Making open cuts insertable exposed two latent defects that this change also
  fixes because the snap cannot stand without them:
  - A later `constrain_edge` could flip or split away an edge an earlier
    constrained chain claimed, leaving the recorded cut referencing a destroyed
    edge — a silent dangling cut. `_MutableMesh` now tracks
    `constrained_edges`; flips refuse them, the split phase leaves them whole
    (falling back to the Dijkstra route around the existing chain, with a
    strong penalty against re-using claimed edges), and
    `cut_surface_at_transitions` verifies every blocked edge still exists
    before component labeling, raising instead of returning a plausible wrong
    sheet graph (`test_inserted_cut_chains_survive_later_constraints`).
  - With no finite pre-cut action anywhere next to the cut (e.g. the caller
    skipped the surface-wide traces, as `--no-actions` does), side assignment
    had nothing to compare. `_branch_components` now traces one well from a
    projected probe point just inside such a side (`_traced_side_action`) —
    the same half-bounce action `evaluate_surface_data` would have supplied —
    so the assignment stays data-driven; if no probe traces, the error remains
    explicit.
- On real equilibria, open companion cuts within half a local edge of the boundary
  now resolve; `examples/plot_cut_surface_pyvista_logical.py --no-actions
  --max-curve-samples 8` on the DMercFail nfp=4 equilibrium at
  \(\lambda_n=0.8\) is the reference demonstration (2 sheets, 134 duplicated
  cut edges, all ports on valid sheets, no triangle spanning the ~11.1 action
  jump). The default 10-sample subset of the same file remains explicitly
  unresolved for a distinct upstream reason — the companion polyline doubles
  back on itself at sub-mesh scale — recorded and decided in ADR 0005.
- This does not touch §21.2: a failed or distant endpoint still yields an explicit
  unresolved hyperedge, never a silent no-connection.
