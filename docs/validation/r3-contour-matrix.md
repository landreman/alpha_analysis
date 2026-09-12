# R3 direct-contour oracle evidence

All 30 prescribed field/pitch cases were probed with two contour steps and finite work budgets. The source is declared h(ρ)=1; contours themselves are source-independent. Pitches come from R0 radially global **estimates**. Positive and negative statuses are represented-field numerical queries with finite-window root-pattern checks, not field enclosures or f estimates. Unknown and no-seed outcomes are not classified as unreachable. Exact field hashes, controls, reasons, path lengths and wall times are in [the JSON](r3-contour-matrix.json).

Hardware: `macOS-14.7.6-arm64-arm-64bit`; one worker; new field objects, warm OS cache uncontrolled. Total elapsed: 53.78 s.

Coarse (step .04, 80 steps): {'NO_SEED': 13, 'UNKNOWN': 17}. Fine (step .02, 160 steps): {'NO_SEED': 13, 'UNKNOWN': 17}. The fixed interior seed search completed 0/17 seeded coarse queries and 0/17 seeded fine queries; 13/30 cases had no seed at either resolution. Statuses agreed across resolutions in 30/30 cases. Agreement on unknown is not convergence; completion is feasibility evidence, not an unbiased loss estimate.

| Field | λn | Coarse | Fine | Coarse wall (s) | Fine wall (s) |
| --- | ---: | --- | --- | ---: | ---: |
| 0 | 0.05 | NO_SEED | NO_SEED | 0.31 | 0.46 |
| 0 | 0.1 | UNKNOWN | UNKNOWN | 1.34 | 1.56 |
| 0 | 0.5 | UNKNOWN | UNKNOWN | 0.49 | 0.65 |
| 0 | 0.8 | UNKNOWN | UNKNOWN | 0.67 | 0.82 |
| 0 | 0.9 | NO_SEED | NO_SEED | 0.28 | 0.26 |
| 0 | 0.95 | NO_SEED | NO_SEED | 0.27 | 0.27 |
| 1 | 0.05 | NO_SEED | NO_SEED | 0.27 | 0.26 |
| 1 | 0.1 | NO_SEED | NO_SEED | 0.26 | 0.29 |
| 1 | 0.5 | UNKNOWN | UNKNOWN | 1.26 | 1.87 |
| 1 | 0.8 | UNKNOWN | UNKNOWN | 0.39 | 0.41 |
| 1 | 0.9 | UNKNOWN | UNKNOWN | 0.60 | 0.59 |
| 1 | 0.95 | NO_SEED | NO_SEED | 0.27 | 0.25 |
| 2 | 0.05 | UNKNOWN | UNKNOWN | 1.44 | 1.54 |
| 2 | 0.1 | UNKNOWN | UNKNOWN | 1.16 | 2.24 |
| 2 | 0.5 | UNKNOWN | UNKNOWN | 1.99 | 2.79 |
| 2 | 0.8 | UNKNOWN | UNKNOWN | 1.83 | 4.31 |
| 2 | 0.9 | UNKNOWN | UNKNOWN | 0.38 | 0.39 |
| 2 | 0.95 | NO_SEED | NO_SEED | 0.28 | 0.28 |
| 3 | 0.05 | UNKNOWN | UNKNOWN | 1.03 | 1.36 |
| 3 | 0.1 | UNKNOWN | UNKNOWN | 1.22 | 1.31 |
| 3 | 0.5 | UNKNOWN | UNKNOWN | 0.33 | 0.32 |
| 3 | 0.8 | UNKNOWN | UNKNOWN | 0.48 | 0.52 |
| 3 | 0.9 | NO_SEED | NO_SEED | 0.26 | 0.26 |
| 3 | 0.95 | NO_SEED | NO_SEED | 0.26 | 0.26 |
| 4 | 0.05 | NO_SEED | NO_SEED | 0.20 | 0.20 |
| 4 | 0.1 | NO_SEED | NO_SEED | 0.21 | 0.23 |
| 4 | 0.5 | UNKNOWN | UNKNOWN | 0.29 | 0.33 |
| 4 | 0.8 | UNKNOWN | UNKNOWN | 0.35 | 0.47 |
| 4 | 0.9 | NO_SEED | NO_SEED | 0.20 | 0.20 |
| 4 | 0.95 | NO_SEED | NO_SEED | 0.20 | 0.19 |

DMercFail λn=0.8 probes:

- closed_candidate: `UNKNOWN`; closed trace lacks a root-pattern certificate; 1.73 s. [Plot](r3-contour-plots/dmerc-closed_candidate.png).
- edge_candidate: `ACCESSIBLE`; edge witness; 5.87 s. [Plot](r3-contour-plots/dmerc-edge_candidate.png).
- transition: `UNKNOWN`; all common-parameter ports explored; event-curve or branch uncertainty remains; 2.42 s. [Plot](r3-contour-plots/dmerc-transition.png).

Synthetic [closed](r3-contour-plots/synthetic-closed.png) and [edge-reaching](r3-contour-plots/synthetic-edge.png) diagnostic plots show the two classified reference contours. The targeted DMercFail edge witness is certified within the finite represented-field root window; the numerically closed real path remains unknown because its root-pattern certificate fails. Pointwise transition ports are expanded at the same parameter; unresolved incident continuation remains unknown. This partial oracle can challenge R4 where it classifies. The fixed interior matrix's completion rate is a feasibility risk, not an f result.
