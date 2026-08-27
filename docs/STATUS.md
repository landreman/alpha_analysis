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
| 6 | Regular well tracer | Trace every regular surface vertex, not only the well near \(\pi/N_{\mathrm{fp}}\) | [x] | #10 |
| 7 | Surface data, refinement, and sheet candidates | Evaluate action data over a whole pitch surface and refine discontinuity candidates | [x] | #12 |
| 8 | Critical curves | Robustly extract and classify \(\Gamma_{\min}\), \(\Gamma_{\max}\), and degenerate portions | [x] | #13 |
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
- Milestone 5's common extractor contract runs against both marching tetrahedra and real PyVista/VTK in CI; PyVista empty levels return empty meshes, seam copies are matched one-to-one, `OUTER` provenance uses a dedicated indicator because VTK interpolates packed masks arithmetically, projection is constrained to `s <= 1`, and parent provenance IDs intentionally remain `-1`.
- Milestone 5 uses a temporary centered Cartesian finite difference for the axis gradient only as a projection direction, followed by residual checks; near-axis `dB_dtheta / s` amplification also remains uncontrolled.
- The §7.3 option-1 axis-regular interpolation is implemented: `BoozerField` continues `m != 0` coefficients below the innermost half-grid surface with the `s^{|m|/2}` harmonic scaling (ADR 0002), so `|B|` is single-valued at the axis; values at `s >= s0` are unchanged.
- `g=0` split-point polishing backs the planar Newton solve with bracketed, locality-bounded fallbacks (projected chord solve, plane-curve continuation over a pencil of cutting planes returning the nearest in-disk crossing, and — when the `g=0` curve exits the domain near the outer boundary — a trace of the surface's boundary curve on the `x^2+y^2=1` cylinder; ADR 0001). Thin-tube levels near `min B` extract for the W7-X reference file at every swept `b` on structured and gmsh meshes, with split points local to their parent edge and inside `x^2+y^2<=1`; boundary-exit split points carry `EDGE|G_ZERO` provenance.
- A marching triangle that bridges two sheets of an under-resolved surface (coarse gmsh meshes near `min B`) has no local `g=0` point; its split vertex is placed at the `g` sign discontinuity, tagged `G_JUMP`, excluded from the `g=0` curve, and counted in `SurfaceExtraction.n_unresolved_splits` with status `UNRESOLVED` (ADR 0001). Downstream milestones must treat `UNRESOLVED` extractions as needing background refinement before production use.
- Milestone 6 traces in the physical direction `sign(G + iota I)`, stores authoritative half-bounce `A` and `K` plus lifted exits and extrema itineraries, and leaves `MAX_PERIODS` action/time as `NaN` with an explicit scanned-period count; downstream surface/pipeline stages must carry that status into unresolved-weight bounds rather than `Theta=0`.
- Milestone 6's conservative Fourier-aware scan includes every retained mode regardless of amplitude (about 1125 samples per field period on the W7-X reference surface); Milestone 7 should profile this cost and add a demonstrated amplitude-aware cutoff before batch tracing if needed.
- Milestone 6's `quadrature_error_K` is an adaptive-subdivision estimate, not a floating-point cancellation bound near an internal maximum with `B_max` close to `b`; Milestones 8–9 own special treatment of that `Gamma_max` band, and surface-wide tracing should track its explicit failure fraction.
- Pitch surfaces can be downsampled before Milestone 7's batch traces with topology-preserving shortest-edge collapses. Moved vertices are reprojected to `B=b`; physical, periodic-seam, and `g=0` boundary vertices remain fixed; each face remains close to its original normal; drift in the scalar axis-regular `|ds wedge d alpha|` measure is bounded globally and per connected component; the achieved reduction and rejection reasons are reported; and changed provenance is explicitly invalidated rather than guessed. The scalar budgets do not control local or within-component cancellation or weighted bounce integrals, so later refinement and convergence remain necessary. This is a staged utility rather than a pipeline caller; Milestone 10 must continue to forbid coarsening after transition cuts.
- Milestone 7's `SurfaceRefinementResult.edge_indicators` retains final edge IDs, midpoint `A`/`K` interpolation errors, and unquantized-itinerary candidate flags for Milestone 8; any non-regular endpoint or midpoint stays a candidate, projected refinement invalidates changed provenance, and edges shared by a tagged boundary remain explicitly `refinement_blocked` until Milestone 8 can refine them on the boundary curve. The extrema-height comparison is normalized by `b` using the global itinerary tolerance; Milestone 8 should assess local `b-B_extremum` gap scaling while resolving critical curves.
- Milestone 8's `CriticalCurves` stores point and segment classifications plus ordered, cumulative-arc-length polylines using global vertex IDs. Production seam continuity is established by surface extraction's canonical seam vertex IDs; critical-curve stitching additionally supports uniquely matched lower/upper degree-one endpoints carrying `PERIODIC_SEAM` provenance for staged callers. Vertex and intrinsic-midpoint `D_parallel^2 B` sampling drives direct `B-b=g=D_parallel^2 B=0` junction solves, including a true-fold, nonzero-iota production-path synthetic test with independently derived analytic coordinates. `GAMMA_MAX` polylines are ready for Milestone 9's common-parameter transition mapping, while failed local degenerate-point solves and all other `DEGENERATE` or `UNRESOLVED` results remain explicit rather than being mapped as generic transitions.
- Milestone 8 refines standalone critical-curve arrays. Inserting those vertices into `SurfaceMesh` and clearing Milestone 7's boundary `refinement_blocked` edges is deferred to Milestone 10's constrained cuts; local `b-B_extremum` gap scaling and transition-sampling convergence belong to Milestone 9. The cumulative `u` parameter is currently a piecewise-linear chord-length sum whose convergence is not yet controlled and must be included in that Milestone 9 study. The combined pitch-surface diagnostic must add `EDGE`/`AXIS` overlays when Milestone 15 assembles the pipeline view.

## Accepted deviations

Design decisions taken during implementation that differ from `docs/DESIGN.md` live in
`docs/adr/` and are listed here with one line each.

- (none yet)
