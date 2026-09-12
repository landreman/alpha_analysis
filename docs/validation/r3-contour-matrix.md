# R3 direct-contour oracle evidence

All 30 prescribed field/pitch cases were probed with two contour steps and finite work budgets. The source is declared h(ρ)=1; contours themselves are source-independent. Pitches come from R0 radially global **estimates**. Positive and negative statuses are represented-field numerical queries with finite-window root-pattern checks, not field enclosures or f estimates. Unknown and no-seed outcomes are not classified as unreachable. Exact field hashes, controls, reasons, path lengths and wall times are in [the JSON](r3-contour-matrix.json).

Hardware: `macOS-14.7.6-arm64-arm-64bit`; one worker; new field objects, warm OS cache uncontrolled. Total elapsed: 37.78 s.

Coarse (step .04, 80 steps): {'NO_SEED': 13, 'UNKNOWN': 17}. Fine (step .02, 160 steps): {'NO_SEED': 13, 'UNKNOWN': 17}. The fixed interior seed search gave 0/17 completed seeded queries and 13/30 with no seed at either resolution. The unchanged statuses are an uncertainty result, not a convergence claim. Completion is feasibility evidence, not an unbiased loss estimate.

| Field | λn | Coarse | Fine | Coarse s | Fine s |
| --- | ---: | --- | --- | ---: | ---: |
| 0 | 0.05 | NO_SEED | NO_SEED | 0.30 | 0.26 |
| 0 | 0.1 | UNKNOWN | UNKNOWN | 1.01 | 1.15 |
| 0 | 0.5 | UNKNOWN | UNKNOWN | 0.20 | 0.24 |
| 0 | 0.8 | UNKNOWN | UNKNOWN | 0.43 | 0.44 |
| 0 | 0.9 | NO_SEED | NO_SEED | 0.26 | 0.26 |
| 0 | 0.95 | NO_SEED | NO_SEED | 0.30 | 0.26 |
| 1 | 0.05 | NO_SEED | NO_SEED | 0.26 | 0.26 |
| 1 | 0.1 | NO_SEED | NO_SEED | 0.26 | 0.26 |
| 1 | 0.5 | UNKNOWN | UNKNOWN | 0.81 | 1.14 |
| 1 | 0.8 | UNKNOWN | UNKNOWN | 0.27 | 0.26 |
| 1 | 0.9 | UNKNOWN | UNKNOWN | 0.37 | 0.36 |
| 1 | 0.95 | NO_SEED | NO_SEED | 0.25 | 0.25 |
| 2 | 0.05 | UNKNOWN | UNKNOWN | 1.02 | 1.21 |
| 2 | 0.1 | UNKNOWN | UNKNOWN | 0.98 | 1.79 |
| 2 | 0.5 | UNKNOWN | UNKNOWN | 1.20 | 1.61 |
| 2 | 0.8 | UNKNOWN | UNKNOWN | 0.87 | 2.65 |
| 2 | 0.9 | UNKNOWN | UNKNOWN | 0.18 | 0.17 |
| 2 | 0.95 | NO_SEED | NO_SEED | 0.28 | 0.28 |
| 3 | 0.05 | UNKNOWN | UNKNOWN | 0.95 | 1.43 |
| 3 | 0.1 | UNKNOWN | UNKNOWN | 1.08 | 1.26 |
| 3 | 0.5 | UNKNOWN | UNKNOWN | 0.18 | 0.15 |
| 3 | 0.8 | UNKNOWN | UNKNOWN | 0.31 | 0.31 |
| 3 | 0.9 | NO_SEED | NO_SEED | 0.28 | 0.25 |
| 3 | 0.95 | NO_SEED | NO_SEED | 0.25 | 0.25 |
| 4 | 0.05 | NO_SEED | NO_SEED | 0.20 | 0.20 |
| 4 | 0.1 | NO_SEED | NO_SEED | 0.21 | 0.19 |
| 4 | 0.5 | UNKNOWN | UNKNOWN | 0.20 | 0.23 |
| 4 | 0.8 | UNKNOWN | UNKNOWN | 0.20 | 0.25 |
| 4 | 0.9 | NO_SEED | NO_SEED | 0.19 | 0.19 |
| 4 | 0.95 | NO_SEED | NO_SEED | 0.19 | 0.18 |

DMercFail λn=0.8 probes:

- closed_candidate: `UNKNOWN`; closed trace lacks a root-pattern certificate; 0.76 s. [Plot](r3-contour-plots/dmerc-closed_candidate.png).
- edge_candidate: `ACCESSIBLE`; edge witness; 2.92 s. [Plot](r3-contour-plots/dmerc-edge_candidate.png).
- transition: `UNKNOWN`; all common-parameter ports explored; event-curve or branch uncertainty remains; 2.55 s. [Plot](r3-contour-plots/dmerc-transition.png).

Synthetic [closed](r3-contour-plots/synthetic-closed.png) and [edge-reaching](r3-contour-plots/synthetic-edge.png) diagnostic plots show the two classified reference contours. The targeted DMercFail edge witness is certified within the finite represented-field root window; the numerically closed real path remains unknown because its root-pattern certificate fails. Pointwise transition ports are expanded at the same parameter; unresolved incident continuation remains unknown. This partial oracle can challenge R4 where it classifies, but its zero completion on the fixed interior matrix is a feasibility risk, not an f result.
