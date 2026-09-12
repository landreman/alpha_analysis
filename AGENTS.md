# Working in this repository

The guiding document is `docs/DESIGN.md`. It defines the physics, the algorithm, the
package layout, the tests, and the milestones. When this file and `docs/DESIGN.md`
disagree, `docs/DESIGN.md` wins — and the disagreement is a bug in one of them, so say
so.

`docs/STATUS.md` records which milestones are done. `docs/adr/` records decisions taken
during implementation that `docs/DESIGN.md` did not settle.

The active plan is the R0–R8 sequence in `docs/DESIGN.md` §23, adopted by accepted
`docs/adr/0010-branch-atlas-and-bounded-f.md`. All redesign milestones are initially
unchecked; R0 is next. Legacy 10.3 is retired **without completion** and legacy
11–18 are superseded. Do not restart the old lowest-numbered unchecked milestone.
Historical notes in `docs/history/` are nonnormative. Approval of the redesign does
not assert that its numerical machinery is implemented or accept PR #24 as-is.

## Environment

* Use the existing conda environment `20220806-03` (Python 3.10.5). Do not create a
  new conda environment. Do not use `20250627-01-libE`: its Python 3.13 `readline`
  extension segfaults during pytest capture.
* Work in the clean virtual environment
  `/Users/mattland/alpha_analysis/alpha_analysis/.venv`, created from that conda
  interpreter without `--system-site-packages`.
* If you need to understand the workings of the `booz_xform` package, look at its
  GitHub repository at https://github.com/hiddenSymmetries/booz_xform, or on the
  researcher's computer at `/Users/mattland/booz_xform/booz_xform`.
* W7-X reference data used by the tests lives under `data/`.

First-time setup, from the repository root:

```bash
/Users/mattland/opt/miniconda3/envs/20220806-03/bin/python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

The clean venv installs the project-declared numerical and development dependencies
itself, preventing unrelated conda packages and pytest plugins from affecting tests.
Activate `.venv` before running `make`.

## Commands

```bash
make test        # fast tier, under 2 min       -- the inner loop
make test-full   # everything, under 5 min      -- before a PR goes ready
make lint        # black --check
make format      # black, applied
make check       # lint + test-full             -- the gate
make smoke       # clean-venv install and import
```

## Code

* Python 3.10+. Format with `black` (`make format`).
* Keep it straightforward. No framework abstractions the physics does not ask for.
* Plain NumPy arrays and integer IDs in the numerical core. Gmsh, PyVista and NetworkX
  objects stay outside it (`docs/DESIGN.md` §19.2).
* Docstrings state conventions, units, and the equation or design section they
  implement.
* New work for the connectivity metric goes in `alpha_analysis/j_connectivity/`.
  Existing public functions and CLI entry points stay backward compatible.

## Test speed

`docs/DESIGN.md` §22.5 is normative:

* `make test` under 2 minutes, no single fast test over about 20 s.
* `make test-full` under 5 minutes, no single `slow` test over about 90 s.

This repository does not chase exhaustive coverage. It wants a small suite that would
actually catch a wrong sign, a dropped term, or a misclassified topology, and that is
fast enough that you run it every time. Make a test cheaper before you mark it `slow`;
lower resolution, shrink the grid, share a fixture, cache the loaded field. Every
scientific claim keeps at least one *fast* test on the production path that fails under
a mutation of the physics it checks.

## Checking new features on real plasma equilibria

When implementing new functionality related to j_connectivity, exercise the new
machinery on these 5 boozmn files in the `data` directory:
~~~~
boozmn_20260402-01-038_Ax_PCA_20dofs_allNfp_aspect6_eval000290_low_resolution.nc
boozmn_20260402-01-178_TURBO_Garabedian_mpol1_xmin0p1_allNfp_aspect6_eval000155.nc
boozmn_20260406-01-262-Ax_nfp4_Garabedian_mpol2_ntor2_minx0_allNfp_aspect10_DMercFail_m0p3_eval000323_low_resolution.nc
boozmn_d23p4_tm_ns51_mbooz16_nbooz16.nc
boozmn_n3are_R7.75B5.7_mbooz18_nbooz12.nc
~~~~
If the new machinery is specific to one value of \(b = B_{bounce}\), then
exercise the functionality for
\(\lambda_n = 0.05, 0.1, 0.5, 0.8, 0.9, 0.95\) where \(\lambda_n\) is defined by
\(b = B_{min} + \lambda_n * (B_{max} - B_{min})\) and \(B_{min}\) and \(B_{max}\) are radially global
(extrema over all radii).

The active atlas validation matrix is these **30 physical cases** (five files ×
six levels), with the resolution and uncertainty comparisons required by the
current milestone. Early milestones exercise the machinery available at that
stage and report limitations; do not claim completed \(f\) acceptance before R8.
The initial acceptance source is \(h=1\), the existing design default. The final
contract in `docs/DESIGN.md` §23 requires full \(f\)-interval width at most 0.01,
at least 29/30 successful slice cases under its normalized gap criterion, and
explicit treatment of independently certified zero trapped population.

For final-candidate benchmarking, the initial guard is 600 seconds per equilibrium
including loading and shared setup, with a 600-second hard slice limit. Record
60-, 300- and 600-second snapshots where the calculation remains active. A bound
that remains too wide when a budget expires is an accuracy failure, not a resolved
case. Use the full accounting and timing rules in `docs/DESIGN.md` §23; these
benchmark guards do not replace the test-suite budgets above.

When touching the **legacy mesh, extraction or cut path**, also exercise the
affected real cases with both structured/Gmsh backends and both
MarchingTetrahedraExtractor/PyVistaSurfaceExtractor extractors. Preserve existing
cross-backend, cross-extractor and public-API regression tests. Those four
combinations are implementation comparisons, not four independent physical
cases in the new 30-case acceptance denominator.

For legacy runs, downsampling before cutting can bound cost. Keep
`TransitionMappingConfig.max_curve_samples` as a unique-vertex work budget:
`BUDGET_INSUFFICIENT` retains all ports and forbids a cut. Use
`map_transitions_budget_sweep()` for the 8, 10, 16 and full comparisons when
sampling is affected. Preserve `total_u_length`, source IDs and certified sheet
graphs; never downsample after cutting. Detailed legacy conventions are archived
in `docs/history/STATUS-pre-redesign.md` and the validation reports.

Inspect the diagnostics, test physical identities, and compare resolutions and
independent algorithms as required by the active milestone. Diagnose discrepancies
and retain honest uncertainty; a missing transition may affect connectivity far
beyond the local cell's own measure.

## Definition of done

A milestone is done when all of these hold:

1. Its active dependencies in `docs/DESIGN.md` §23 are satisfied, and `make check`
   is green inside the test budget above. Retired legacy milestones are not
   prerequisites. R0 owns the explicit baseline and gate migration; this planning
   change does not claim the current PR #24 branch is green.
2. The milestone's acceptance criteria in `docs/DESIGN.md` §23 are each demonstrated by
   a named test — not by a plot, and not by a residual that got small.
3. You verified the tests can fail: applied the one or two mutations that matter for
   this milestone, watched the suite go red, reverted. The PR body names them.
4. Any new geometric object has at least one diagnostic plot (`docs/DESIGN.md` §17).
5. The GitHub Actions `Tests` workflow is green on the branch. The optional
   `Claude Code Review` workflow is not a prerequisite for starting the next
   milestone.
6. `docs/STATUS.md` has the milestone's row marked and any note the next milestone
   needs.
7. Every deviation from `docs/DESIGN.md` is either an accepted ADR or written up in the
   PR body.

For PR #24, retain its branch and failure evidence; do not merge it as-is or
automatically accept its proposed ADRs. Land the planning documentation separately
on `main`. R0 must verify the accepted baseline and selectively carry forward only
needed fixes with regression evidence. Preserve the failed 10.3 matrix result and
scientific tests when migrating its retired acceptance gate; do not relabel the
milestone complete or weaken the new accuracy criteria.

## STOP conditions

Accepted ADR 0010 already authorizes the stated algorithm and milestone migration;
do not ask the researcher to approve that same decision again. Apply the following
STOP conditions to a new ambiguity or deviation beyond that approved scope.

Stop, write an ADR in `docs/adr/`, open the PR as a draft naming it, and end your turn
— do not choose an option and proceed — when:

* `docs/DESIGN.md` is ambiguous or looks wrong about something that changes the physics
  or an acceptance criterion;
* an acceptance criterion cannot be met without loosening a tolerance, widening an
  error bound, or deleting a check;
* you want a new base dependency, or need to cross a §19.2 boundary;
* a topological or numerical failure will not resolve and the only way to green is to
  hide it.

Relaxing a tolerance, marking a test `xfail` or `skip`, or narrowing its inputs to
reach green is never the answer on its own.

## Never

`docs/DESIGN.md` §21.2 is a hard list. Do not:

* replace a failed trace with zero action or zero weight;
* interpret a clipped well as passing;
* drop triangles containing `NaN` without accounting for their measure;
* treat a missing transition as no connection;
* cap a long trace and assign Θ = 0;
* merge disconnected surface components because they are close in Euclidean
  coordinates.

Every one of these produces a plausible number. That is what makes them dangerous.
