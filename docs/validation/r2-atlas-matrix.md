# R2 root-labelled atlas evidence

All 30 physical file/pitch cases were probed at 3×4 and 5×8 transverse grids, two lifted periods, with separate narrow-cell refinements at s=0.4, α=0. The source declaration is h(ρ)=1. Pitches use the R0 radially global extrema **estimates**. These are represented-field geometry and sampled A estimates, not field, K-weight, reachability, slice or f enclosures. An unknown cell is not given zero weight. The exact hashes, controls, errors and wall times are in [the JSON](r2-atlas-matrix.json).

Hardware: `macOS-14.7.6-arm64-arm-64bit`; one worker; new field objects, warm OS cache uncontrolled. Total elapsed: 90.43 s.

| Field | λn | Coarse owned samples | Coarse known/unknown cells | Fine known/unknown cells | Fixed local counts (wide/narrow) | Owned target counts (wide/narrow) | Grid time s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.05 | 3 | 0/8 | 0/32 | 0 / 0 | None/None | 2.82 |
| 0 | 0.1 | 3 | 0/8 | 0/32 | 0 / 0 | None/1 | 2.84 |
| 0 | 0.5 | 6 | 0/8 | 0/32 | None / None | None/1 | 3.43 |
| 0 | 0.8 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 3.27 |
| 0 | 0.9 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 3.43 |
| 0 | 0.95 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 3.39 |
| 1 | 0.05 | 2 | 0/8 | 0/32 | 0 / 0 | None/1 | 3.52 |
| 1 | 0.1 | 2 | 0/8 | 0/32 | 0 / 0 | None/None | 3.20 |
| 1 | 0.5 | 7 | 0/8 | 0/32 | None / None | None/1 | 2.80 |
| 1 | 0.8 | 4 | 0/8 | 0/32 | None / None | None/1 | 2.70 |
| 1 | 0.9 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 2.63 |
| 1 | 0.95 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 2.58 |
| 2 | 0.05 | 9 | 0/8 | 0/32 | None / None | 1/1 | 2.71 |
| 2 | 0.1 | 9 | 0/8 | 0/32 | None / None | 1/1 | 2.76 |
| 2 | 0.5 | 9 | 0/8 | 0/32 | None / 1 | 1/1 | 2.73 |
| 2 | 0.8 | 8 | 0/8 | 0/32 | None / None | 1/1 | 2.90 |
| 2 | 0.9 | 4 | 0/8 | 0/32 | None / None | 1/1 | 2.60 |
| 2 | 0.95 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 2.54 |
| 3 | 0.05 | 4 | 0/8 | 0/32 | None / None | None/1 | 2.53 |
| 3 | 0.1 | 6 | 0/8 | 0/32 | None / None | None/1 | 2.60 |
| 3 | 0.5 | 6 | 0/8 | 0/32 | None / None | None/1 | 2.65 |
| 3 | 0.8 | 1 | 0/8 | 0/32 | 0 / 0 | None/None | 2.53 |
| 3 | 0.9 | 0 | 0/8 | 1/31 | 0 / 0 | no seed | 2.50 |
| 3 | 0.95 | 0 | 0/8 | 1/31 | 0 / 0 | no seed | 2.48 |
| 4 | 0.05 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 1.90 |
| 4 | 0.1 | 0 | 0/8 | 0/32 | 0 / 0 | no seed | 1.99 |
| 4 | 0.5 | 1 | 0/8 | 0/32 | None / None | None/None | 2.02 |
| 4 | 0.8 | 2 | 0/8 | 0/32 | 0 / 0 | None/None | 1.94 |
| 4 | 0.9 | 1 | 0/8 | 0/32 | 0 / 0 | None/None | 2.01 |
| 4 | 0.95 | 0 | 0/8 | 1/31 | 0 / 0 | no seed | 2.32 |

Positive local owned multiplicity was certified in 13/30 cases at one of the two narrow target widths. These patches replace their parent cell with a disjoint partition; the remaining unknown area and all possible links remain unknown. This does not establish a useful global population lower bound.

Synthetic equal-height event: 6 possible limiting wells, status `multiway_unknown`. [Event diagnostic](r2-atlas-plots/synthetic-six-port-atlas.png); [certified multicover diagnostic](r2-atlas-plots/synthetic-multicover-atlas.png).

DMercFail λn=0.8: `generic` at s=0.8 from independent B=b and D∥B=0 solve. [Diagnostic plot](r2-atlas-plots/dmerc-0p8-atlas.png). The old resolved cut is a geometry comparator, not the expected root or port action. Unknown coarse cells and narrow-cell failures require further atlas refinement and field-scope bounds in later milestones.

The analytic height branch `H(s,α)=1.3+0.2s` gives the expected `H=b=1.4` contour at `s=0.5`, independently of α. [Height diagnostic](r2-atlas-plots/synthetic-height-contour.png) is reproduced by `.venv/bin/python examples/plot_r2_height.py`; its maximum analytic residual was `2.22e-16` B units.

The requested difficult probes remain explicit: PCA and TURBO at λn=0.1 had sampled incoming wells but no coarse or fine globally certified cells; DMercFail at λn=0.05 likewise had sampled wells with all broad cells unknown. d23p4 at λn=0.8 had one coarse owned sample well, while λn=0.9 and 0.95 had none within the two-period sample window; these are not passing or empty-slice findings. The local positive certificates are too small to make the full unknown area or possible connectivity negligible. No final `Q` or `f` accuracy claim is made.
