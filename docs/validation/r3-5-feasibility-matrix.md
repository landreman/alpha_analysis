# R3.5 feasibility matrix (represented-field local evidence)

Numerical revision: `12bd422557915b984ce481feb2485809724b532a`. Hardware: `macOS-14.7.6-arm64-arm-64bit`; one worker; h(ρ)=1 declared.
Pitches retain R0 sampled radially global extrema estimates. Root/field interpolation and action quadrature errors are not field enclosures.

Cases recorded: 30/30. Total wall: 904.59 s.
The original R2/R3 evidence remains in its separate files. Current results use 4096 certificate boxes, depth 10, and both original contour steps.

The fixed R3 fine cohort had 13 no-seed and 17 unknown cases. The same fixed cohort now has 13 no-seed, 9 unknown, 7 inaccessible and 1 accessible cases. The expanded bounded search recovered seeds in 13/13 original no-seed cases; finding a seed is not a contour classification or an empty-population proof.
Certified local root corridors occur in 22/30 cases and on all five fields. The longest cold physical case took 221.50 s under the 600 s guard. The 60 s active snapshots are recorded for PCA 0.1 and d23p4 0.1; no case reached 300 s.

| Field | λn | Original coarse/fine | Current coarse/fine | Search seeds | Fine atlas positive/unknown area | Local corridor | Full path coarse/fine | Cold wall (s) |
| --- | ---: | --- | --- | ---: | --- | --- | --- | ---: |
| PCA | 0.05 | NO_SEED/NO_SEED | NO_SEED/NO_SEED | 36 | 0/6.28 | CERTIFIED | —/— | 15.62 |
| PCA | 0.1 | UNKNOWN/UNKNOWN | INACCESSIBLE/INACCESSIBLE | 64 | 0/6.28 | CERTIFIED | —/— | 64.35 |
| PCA | 0.5 | UNKNOWN/UNKNOWN | UNKNOWN/UNKNOWN | 64 | 0/6.28 | CERTIFIED | —/— | 11.11 |
| PCA | 0.8 | UNKNOWN/UNKNOWN | UNKNOWN/UNKNOWN | 40 | 0/6.28 | NO_SEED | —/— | 33.81 |
| PCA | 0.9 | NO_SEED/NO_SEED | NO_SEED/NO_SEED | 32 | 0/6.28 | NO_SEED | —/— | 45.59 |
| PCA | 0.95 | NO_SEED/NO_SEED | NO_SEED/NO_SEED | 22 | 0/6.28 | NO_SEED | —/— | 58.69 |
| TURBO | 0.05 | NO_SEED/NO_SEED | NO_SEED/NO_SEED | 33 | 0/6.28 | CERTIFIED | —/— | 14.98 |
| TURBO | 0.1 | NO_SEED/NO_SEED | NO_SEED/NO_SEED | 41 | 0/6.28 | CERTIFIED | —/— | 23.75 |
| TURBO | 0.5 | UNKNOWN/UNKNOWN | INACCESSIBLE/INACCESSIBLE | 64 | 0/6.28 | CERTIFIED | —/— | 15.06 |
| TURBO | 0.8 | UNKNOWN/UNKNOWN | UNKNOWN/UNKNOWN | 61 | 0/6.28 | CERTIFIED | —/— | 15.54 |
| TURBO | 0.9 | UNKNOWN/UNKNOWN | UNKNOWN/UNKNOWN | 35 | 0/6.28 | NO_SEED | —/— | 29.53 |
| TURBO | 0.95 | NO_SEED/NO_SEED | NO_SEED/NO_SEED | 28 | 0/6.28 | NO_SEED | —/— | 31.49 |
| DMercFail | 0.05 | UNKNOWN/UNKNOWN | INACCESSIBLE/INACCESSIBLE | 64 | 0/6.28 | CERTIFIED | —/— | 9.58 |
| DMercFail | 0.1 | UNKNOWN/UNKNOWN | INACCESSIBLE/INACCESSIBLE | 64 | 0/6.28 | CERTIFIED | —/— | 7.75 |
| DMercFail | 0.5 | UNKNOWN/UNKNOWN | INACCESSIBLE/INACCESSIBLE | 64 | 3.14/3.14 | CERTIFIED | —/— | 9.84 |
| DMercFail | 0.8 | UNKNOWN/UNKNOWN | INACCESSIBLE/INACCESSIBLE | 64 | 0/6.28 | CERTIFIED | CERTIFIED/CERTIFIED | 34.13 |
| DMercFail | 0.9 | UNKNOWN/UNKNOWN | UNKNOWN/UNKNOWN | 64 | 0/6.28 | CERTIFIED | —/— | 16.71 |
| DMercFail | 0.95 | NO_SEED/NO_SEED | NO_SEED/NO_SEED | 32 | 0/6.28 | CERTIFIED | —/— | 18.77 |
| d23p4 | 0.05 | UNKNOWN/UNKNOWN | INACCESSIBLE/INACCESSIBLE | 64 | 0/6.28 | CERTIFIED | —/— | 14.38 |
| d23p4 | 0.1 | UNKNOWN/UNKNOWN | ACCESSIBLE/ACCESSIBLE | 64 | 0/6.28 | CERTIFIED | CERTIFIED/CERTIFIED | 221.50 |
| d23p4 | 0.5 | UNKNOWN/UNKNOWN | UNKNOWN/UNKNOWN | 64 | 0/6.28 | CERTIFIED | —/— | 14.25 |
| d23p4 | 0.8 | UNKNOWN/UNKNOWN | UNKNOWN/UNKNOWN | 33 | 0/6.28 | CERTIFIED | —/— | 17.57 |
| d23p4 | 0.9 | NO_SEED/NO_SEED | NO_SEED/NO_SEED | 34 | 0/6.09 | NO_SEED | —/— | 37.13 |
| d23p4 | 0.95 | NO_SEED/NO_SEED | NO_SEED/NO_SEED | 27 | 0/6.09 | NO_SEED | —/— | 48.12 |
| n3are | 0.05 | NO_SEED/NO_SEED | NO_SEED/NO_SEED | 16 | 0/6.28 | CERTIFIED | —/— | 15.04 |
| n3are | 0.1 | NO_SEED/NO_SEED | NO_SEED/NO_SEED | 26 | 0/6.28 | CERTIFIED | —/— | 15.88 |
| n3are | 0.5 | UNKNOWN/UNKNOWN | UNKNOWN/UNKNOWN | 49 | 0/6.28 | CERTIFIED | —/— | 14.72 |
| n3are | 0.8 | UNKNOWN/UNKNOWN | UNKNOWN/UNKNOWN | 42 | 0/6.28 | CERTIFIED | —/— | 10.62 |
| n3are | 0.9 | NO_SEED/NO_SEED | NO_SEED/NO_SEED | 35 | 0/6.28 | CERTIFIED | —/— | 12.09 |
| n3are | 0.95 | NO_SEED/NO_SEED | NO_SEED/NO_SEED | 15 | 0/6.09 | NO_SEED | —/— | 25.24 |

The eight original certificate-blocked fixed probes classify at both resolutions: seven inaccessible and d23p4 0.1 accessible. The named real tests independently check the starting root signs, first-return action and physical terminal. The DMercFail 0.8 event is found from an ordinary seed; its marginal B and D residuals, action partition, one-sided limits and all three port continuations are checked separately.
DMercFail 0.8 has a closed-path atlas corridor with coarse/fine disjoint positive area 0.042505966/0.029237701 ds d-alpha. d23p4 0.1 has an edge-path corridor with 0.00065011537/0.0006507874. Both subtract only their disjoint certified area from the original fixed-domain unknown complement; the much larger unresolved remainder persists.
Disjoint owned tile areas are geometric ds d-alpha, not K-weighted. Root/field representation and action quadrature errors remain outside field enclosures.
No wide or unknown outcome is treated as an f success or an empty trapped population.

Detailed root lifts, terminal reasons, budgets, field hashes, active snapshots and uncertainty scope are in the JSON.
