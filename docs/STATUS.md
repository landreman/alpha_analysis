# Milestone status

The single source of truth for what is done. `docs/DESIGN.md` §23 defines what each
milestone *is* — goal, changes, acceptance criteria. This file records only whether it
has landed.

`[x]` means the milestone's acceptance criteria are satisfied by the pull request that
sets the marker. A milestone's own implementation pull request is what flips its row,
in the same commit range that satisfies the criteria; do not mark a row ahead of the
work, and do not touch another milestone's row.

Milestones are ordered. Unless the researcher says otherwise, the next milestone is the
lowest-numbered unchecked row.

| # | Milestone | Goal | Done | PR |
| --- | --- | --- | --- | --- |
| 0 | Baseline and design scaffolding | Establish package skeleton without changing numerical behavior | [x] | #3 |
| 1 | General Boozer field derivatives and asymmetric modes | Provide the field interface needed by all later work | [x] | #4 |
| 2 | Denominator \(V_h\) and global \(B\) bounds | Implement the independent normalization calculation | [x] | #5 |
| 3 | Deterministic periodic background mesh | Build the axis-regular logical mesh without Gmsh | [x] | #6 |
| 4 | Gmsh background backend | Add the production mesher behind the same interface | [x] | #7 |
| 5 | \(B=b\) surface extraction | Extract all level-surface components and incoming/outgoing halves | [x] | #8 |
| 6 | Regular well tracer | Trace every regular surface vertex, not only the well near \(\pi/N_{\mathrm{fp}}\) | [ ] | |
| 7 | Surface data, refinement, and sheet candidates | Evaluate action data over a whole pitch surface and refine discontinuity candidates | [ ] | |
| 8 | Critical curves | Robustly extract and classify \(\Gamma_{\min}\), \(\Gamma_{\max}\), and degenerate portions | [ ] | |
| 9 | Transition mapping and action additivity | Construct \(T\) and matched parent/child ports without yet cutting the full mesh | [ ] | |
| 10 | Constrained cuts and sheet IDs | Insert \(T\), duplicate vertices, and make \(A\) continuous on each sheet | [ ] | |
| 11 | Direct contour tracer | Build the correctness oracle | [ ] | |
| 12 | Interval primitives and bounded ordinary flood fill | Classify edge-connected action ranges without transitions using a finite algorithm with lower and upper bounds | [ ] | |
| 13 | Transition-aware bounded flood fill | Add common-parameter transfer through hyperedges while preserving finite termination and bounds | [ ] | |
| 14 | Reachable-polygon and surface quadrature | Compute \(Q(b)\) | [ ] | |
| 15 | Pitch-slice pipeline and HDF5 | Produce a restartable `PitchSliceResult` end to end | [ ] | |
| 16 | Adaptive outer quadrature and parallel execution | Compute final \(f\) | [ ] | |
| 17 | Full validation suite | Establish scientific credibility on synthetic and repository data | [ ] | |
| 18 | Profiling and Numba acceleration | Optimize only measured bottlenecks | [ ] | |

## Notes for the next milestone

Anything a later milestone needs to know that is not already in `docs/DESIGN.md` goes
here — a convention settled during implementation, a data file that has to be
regenerated, a known-shaky tolerance. Keep it short; when an entry becomes permanent,
move it into `docs/DESIGN.md` and delete it here.

- Milestone 0 established base-only `j_connectivity` imports; optional features must use `optional_import()` so missing extras provide an install command.
- Milestone 1's `BoozerFieldLike.B()` uses pointwise NumPy broadcasting; legacy `compute_B()` retains its outer-`s` grid semantics.
- Milestone 2's `find_global_B_bounds()` returns refined extrema, a safety-margin bracket, and radial extrema profiles on the configured `s` grid for later background-mesh diagnostics.
- Milestone 3's `BackgroundMesh.boundary_tags` is a point-located bit mask; periodic pairs are explicit `(zeta=0, zeta=L_zeta)` node IDs, and the Gmsh backend must satisfy the same array and orientation invariants.
- Milestone 4's Gmsh backend embeds the logical axis, returns exact lower/upper seam pairs, supports axis/critical/low-gradient sizing, and finalizes its owned Gmsh session before returning plain arrays; full background-resolution convergence remains Milestone 17 validation work.
- Milestone 5's signed-half meshes include the shared `G_ZERO` closure; regular incoming vertices have physical `g<0`, split-created vertices use parent edge `(-1, -1)`, and periodic cells require local zeta unwrapping.
- Milestone 5's common extractor contract runs against both marching tetrahedra and real PyVista/VTK in CI; PyVista seam copies are matched one-to-one, pre-projection boundary tags survive merging, projection is constrained to `s <= 1`, and parent provenance IDs intentionally remain `-1`.
- Milestone 5 uses a temporary centered Cartesian finite difference for the axis gradient only as a projection direction, followed by residual checks; near-axis `dB_dtheta / s` amplification also remains uncontrolled, so §7.3 axis-regular interpolation or an excluded-core bound is required before a production result includes axis topology.

## Accepted deviations

Design decisions taken during implementation that differ from `docs/DESIGN.md` live in
`docs/adr/` and are listed here with one line each.

- (none yet)
