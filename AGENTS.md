# Working in this repository

The guiding document is `docs/DESIGN.md`. It defines the physics, the algorithm, the
package layout, the tests, and the milestones. When this file and `docs/DESIGN.md`
disagree, `docs/DESIGN.md` wins — and the disagreement is a bug in one of them, so say
so.

`docs/STATUS.md` records which milestones are done. `docs/adr/` records decisions taken
during implementation that `docs/DESIGN.md` did not settle.

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

## Definition of done

A milestone is done when all of these hold:

1. `make check` is green, inside the test budget above.
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

## STOP conditions

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
