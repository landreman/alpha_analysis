# Milestone status

[`DESIGN.md` §23](DESIGN.md#23-active-development-milestones) defines the active milestones and
their dependencies. This file records completion. Accepted
[ADR 0010](adr/0010-branch-atlas-and-bounded-f.md) adopts the root-labelled well atlas
and bounded computation of \(f\); it does not claim that the redesigned algorithm
is implemented.

`[x]` means the milestone's acceptance criteria, tests and validation have been
satisfied by its implementation PR. Mark only the milestone being completed, in the
same commit range that provides that evidence. A planning change never marks an
implementation milestone complete.

## Active plan

**Next milestone: R2. R0 and R1 are complete.** Follow the dependencies in `DESIGN.md`
§23; retired milestones are not prerequisites.

| ID | Milestone | Goal | Done | PR |
| --- | --- | --- | --- | --- |
| R0 | Baseline migration and independent population ledger | Establish the accepted code baseline, preserve historical evidence and tests, and independently account for trapped population and unresolved weight | [x] | #26 |
| R1 | Efficient shared forward scans and bounce integrals | Share field-line scans, enumerate maximal wells, and evaluate batched \(A,K\) with error accounting | [x] | #27 |
| R2 | Root-labelled atlas and barrier-height transitions | Build local well charts, explicit seam ownership and certified generic transition relations | [ ] | |
| R3 | Independent continuous contour oracle | Follow constant-action contours and permitted branch transitions without relying on the atlas reachability implementation | [ ] | |
| R4 | Bounded ordinary action-bin accessibility | Compute finite lower and upper reachability sets on regular sheets without transitions | [ ] | |
| R5 | Transitions, cycles and global uncertainty | Transfer through common-parameter ports and propagate uncertain connectivity globally with finite termination | [ ] | |
| R6 | Weighted pitch-slice bounds and persistence | Produce restartable slice bounds with complete weighted-population accounting | [ ] | |
| R7 | Outer fraction and birth-space validation | Compute bounds on \(f\), validate independently from birth space, and optionally estimate within the unresolved gap statistically | [ ] | |
| R8 | Five-equilibrium accuracy and runtime acceptance | Demonstrate the agreed accuracy, coverage and runtime criteria on the five reference equilibria | [ ] | |

The initial acceptance contract uses the existing default source \(h=1\). The full
interval width must satisfy \(f_{\rm upper}-f_{\rm lower}\le0.01\). The physical
slice matrix has five files and six radially global levels, hence 30 cases; more
than 95% means at least 29/30. Each successful slice must meet the §23 normalized
gap criterion, including the independently bounded zero-population case.

The initial final-candidate benchmark guard is 600 seconds per equilibrium,
including loading and shared setup, with a 600-second hard limit on any slice.
Report 60-, 300- and 600-second snapshots where the run remains active. Finishing
with a wide bound is an explicit accuracy failure, not successful completion.
`DESIGN.md` §23 specifies the complete accuracy and timing contract; these values
do not relax the fast/full test-suite budgets.

## Baseline and next-step notes

- R0 verified the green plan-bearing `main` baseline and imported no implementation
  code from PR #24. That PR remains the incomplete milestone-10.3 branch, not an
  accepted redesign baseline, and must not be merged as-is. Its branch, validation
  reports and failure evidence remain preserved. Approval of ADR 0010 does not
  automatically accept PR #24's code or proposed ADRs 0007/0008.
- The recorded 10.3 matrix achieved 42/120 resolved-or-no-transition cases, below
  its required threshold; see [the report](validation/milestone10.3-real-equilibria.md)
  and [ADR 0009](adr/0009-matrix-resolved-fraction-shortfall.md). Its deliberately
  failing acceptance test and branch-specific fixes need explicit treatment in
  R0's gate migration. Preserve the historical failure as evidence and keep the
  scientific regression checks; do not relabel 10.3 complete or claim that
  PR #24 met its acceptance gate.
- R0's independent population evidence covers the five fields and six required
  pitches with declared `h(rho)=1`; see
  [the report](validation/r0-population-ledger.md) and its JSON/PNG companions.
  These are explicitly quadrature estimates under a dense-line assumption, not
  field enclosures or accessibility classifications. The nonsingular whole-band
  fraction changed by at most 3.351e-4 between the recorded grids, but individual
  fixed-b tensor estimates changed by as much as 46.8% near their integrable
  singularity; the diagnostics show that coarse/fine spread explicitly. R1 now
  returns `LinewiseTrappingMasks` from centered scans: definite and possible masks
  with reasons that keep incomplete roots in the population interval. The scalar
  accessor rejects unresolved weight. R6 must separately resolve radial support
  boundaries where `b` crosses `B_max(s)` (§13.2); fixed Gauss nodes can jump across
  them, so the present grid spread is not a bound. Carry source and bound scope for
  both total and owned cell weights; a field-scope total cannot elevate model-scope
  or uncontrolled cell contributions. Dense-line equality also needs its own
  certificate before a positive lower pitch-band bound is reported. The current
  R0 certificate is caller-supplied prose; R6 must replace it with machine-checkable
  linewise evidence or an appropriate dense-line proof before using it for physical
  field enclosures. Preserve precursor estimate errors alongside certified results;
  a ledger's `uncontrolled_errors` is empty by construction after certification.
- Existing public functions, CLI entry points and legacy mesh/extractor tests
  remain compatibility obligations. The old numerical path is also a useful
  reference on cases it resolves; it is not the production prerequisite for R2–R8.
- R1's [forward-scan evidence](validation/r1-forward-scan-matrix.md) covers one
  lifted line per field at all 30 specified pitches and two scan resolutions.
  The Fourier-model envelope excluded hidden \(B=b\) barriers on those lines;
  16/30 pitch probes per resolution still ended at a censored four-period window.
  A certificate for roots inside a window does not establish complete global
  line or atlas coverage. Up to ten cells per line retained unverified extrema,
  so R2 must continue or enclose those local families rather than treating the
  sampled extrema itinerary as a stable branch label. Batched \(A,K\) values
  retain summed numerical estimates and an independent adaptive comparison,
  not rigorous field or quadrature enclosures. The five-field timings are
  representative local probes, not a final \(f\) runtime or accuracy claim.
- Keep the physical trace direction `sign(G + iota I)`, authoritative half-bounce
  action `A` and time length `K`, lifted root/port identities, and explicit
  `MAX_PERIODS`, root, quadrature and topology failures. A capped or clipped well
  is not passing and never receives zero action, zero weight or `Theta=0` by
  default. Unknown connectivity can affect distant resolved wells and must enter
  the global lower/upper reachability bounds.
- Preserve axis-regular field continuation (ADR 0002), pointwise NumPy field
  broadcasting, and explicit periodic ownership. An extremum pair born or lost
  strictly below \(b\) is diagnostic/quadrature information unless a certified
  change of the maximal \(B<b\) well requires a physical transition.
- The [archived pre-redesign status](history/STATUS-pre-redesign.md) retains all
  previous implementation notes verbatim, including thin-surface extraction,
  boundary provenance, sampling budgets, event limits, and unresolved trace
  conventions. Those notes describe the legacy path; their old next-milestone
  instructions and proposed decisions are not active requirements.

## Historical implementation record

The completed rows retain their original historical status. They do not imply
that \(f\) has been computed or that any redesign milestone is complete.

| Legacy ID | Milestone | Historical status | PR |
| --- | --- | --- | --- |
| 0 | Baseline and design scaffolding | Completed | #3 |
| 1 | General Boozer field derivatives and asymmetric modes | Completed | #4 |
| 2 | Denominator \(V_h\) and global \(B\) bounds | Completed | #5 |
| 3 | Deterministic periodic background mesh | Completed | #6 |
| 4 | Gmsh background backend | Completed | #7 |
| 5 | \(B=b\) surface extraction | Completed | #8 |
| 6 | Regular well tracer | Completed | #10 |
| 7 | Surface data, refinement, and sheet candidates | Completed | #12 |
| 8 | Critical curves | Completed | #13 |
| 9 | Transition mapping and action additivity | Completed | #14 |
| 10 | Constrained cuts and sheet IDs | Completed | #19 |
| 10.1 | Sampling-robust cut geometry | Completed | #21 |
| 10.2 | Contact localization and segment-level cutting | Completed | #22 |
| 10.3 | Failure-directed refinement and matrix convergence | **Retired, not completed; acceptance failed** | #24 (draft) |
| 11–18 | Contour tracing through final validation and optimization | **Superseded, not completed** | |

The legacy design, validation reports and ADRs remain scientific history. The
accepted replacement decision is [ADR 0010](adr/0010-branch-atlas-and-bounded-f.md).
