# R2 root-labelled atlas evidence

All 30 physical file/pitch cases were probed at 3×4 and 5×8 transverse grids, two lifted periods, with separate narrow-cell refinements at s=0.4, α=0. The source declaration is h(ρ)=1. Pitches use the R0 radially global extrema **estimates**. These are represented-field geometry and sampled A estimates, not field, K-weight, reachability, slice or f enclosures. An unknown cell is not given zero weight. The exact hashes, controls, errors and wall times are in [the JSON](r2-atlas-matrix.json).

Hardware: `macOS-14.7.6-arm64-arm-64bit`; one worker; new field objects, warm OS cache uncontrolled. Total elapsed: 78.76 s.

Only 0/240 coarse and 3/960 fine cells have certified multiplicity. This broad unresolved coverage is the main feasibility risk for the later reachability and accuracy milestones; a narrower transverse cell is often needed when its B envelope overlaps b at a scan node.

| Field | λn | Coarse owned samples | Coarse known/unknown cells | Fine known/unknown cells | Fixed local counts (wide/narrow) | Owned target counts (wide/narrow) | Grid time s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.05 | 3 | 0/8 | 0/32 | 0 / 0 | None/None | 2.61 |
| 0 | 0.1 | 3 | 0/8 | 0/32 | 0 / 0 | None/1 | 2.55 |
| 0 | 0.5 | 6 | 0/8 | 0/32 | None / None | None/1 | 2.61 |
| 0 | 0.8 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 2.54 |
| 0 | 0.9 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 2.51 |
| 0 | 0.95 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 2.51 |
| 1 | 0.05 | 2 | 0/8 | 0/32 | 0 / 0 | None/1 | 2.51 |
| 1 | 0.1 | 2 | 0/8 | 0/32 | 0 / 0 | None/None | 2.40 |
| 1 | 0.5 | 7 | 0/8 | 0/32 | None / None | None/1 | 2.50 |
| 1 | 0.8 | 4 | 0/8 | 0/32 | None / None | None/1 | 2.61 |
| 1 | 0.9 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 2.36 |
| 1 | 0.95 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 2.39 |
| 2 | 0.05 | 9 | 0/8 | 0/32 | None / None | 1/1 | 2.63 |
| 2 | 0.1 | 9 | 0/8 | 0/32 | None / None | 1/1 | 2.55 |
| 2 | 0.5 | 9 | 0/8 | 0/32 | None / 1 | 1/1 | 2.52 |
| 2 | 0.8 | 8 | 0/8 | 0/32 | None / None | 1/1 | 2.82 |
| 2 | 0.9 | 4 | 0/8 | 0/32 | None / None | 1/1 | 2.61 |
| 2 | 0.95 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 2.48 |
| 3 | 0.05 | 4 | 0/8 | 0/32 | None / None | None/1 | 2.43 |
| 3 | 0.1 | 6 | 0/8 | 0/32 | None / None | None/1 | 2.54 |
| 3 | 0.5 | 6 | 0/8 | 0/32 | None / None | None/1 | 2.46 |
| 3 | 0.8 | 1 | 0/8 | 0/32 | 0 / 0 | None/None | 2.34 |
| 3 | 0.9 | 0 | 0/8 | 1/31 | 0 / 0 | no seed | 2.35 |
| 3 | 0.95 | 0 | 0/8 | 1/31 | 0 / 0 | no seed | 2.32 |
| 4 | 0.05 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 1.71 |
| 4 | 0.1 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 1.75 |
| 4 | 0.5 | 1 | 0/8 | 0/32 | None / None | None/None | 1.70 |
| 4 | 0.8 | 2 | 0/8 | 0/32 | 0 / 0 | None/None | 1.69 |
| 4 | 0.9 | 1 | 0/8 | 0/32 | 0 / 0 | None/None | 1.66 |
| 4 | 0.95 | 0 | 0/8 | 1/31 | 0 / 0 | no seed | 1.64 |

Positive local owned multiplicity was certified in 13/30 cases at one of the two narrow target widths. These patches replace their parent cell with a disjoint partition; the remaining unknown area and all possible links remain unknown. This does not establish a useful global population lower bound.

Synthetic equal-height event: 6 possible limiting wells, status `multiway_unknown`. [Event diagnostic](r2-atlas-plots/synthetic-six-port-atlas.png); [certified multicover diagnostic](r2-atlas-plots/synthetic-multicover-atlas.png).

DMercFail λn=0.8: `generic` at s=0.8 from independent B=b and D∥B=0 solve. [Diagnostic plot](r2-atlas-plots/dmerc-0p8-atlas.png). The old resolved cut is a geometry comparator, not the expected root or port action. Unknown coarse cells and narrow-cell failures require further atlas refinement and field-scope bounds in later milestones.
