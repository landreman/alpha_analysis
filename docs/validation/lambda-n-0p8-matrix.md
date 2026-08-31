# The new `lambda_n = 0.8` matrix level

The real-equilibrium matrix in `AGENTS.md` now carries six radially global bounce
levels — `0.05, 0.1, 0.5, 0.8, 0.9, 0.95` — so five equilibria, two background
backends, two surface extractors, and six levels make 120 cases.  This report
records what the twenty *new* cases (the `lambda_n = 0.8` slice) do, so that the
level enters the matrix with a known baseline rather than as an untested addition.
It is a diagnostic study, not a replacement for the named acceptance tests.

`lambda_n = 0.8` was already the ADR 0005 reference level for the DMercFail
sampling sweep (`examples/validate_transition_sampling.py`); this makes it part of
the standard matrix.

## Controls

Both reproducible drivers were run with their defaults, restricted to the new
level:

```text
python examples/validate_transition_equilibria.py --lambda-n 0.8 --output ...
python examples/validate_cut_equilibria.py --lambda-n 0.8 --localize-events --output ...

background = structured (6, 24, 12), gmsh (target size 0.3)
extractor  = MarchingTetrahedraExtractor, PyVistaSurfaceExtractor
transition samples = at most 8 authoritative critical-curve vertices
field-period cap = 128, action order 32 adaptively doubled through 512
global bounds = (17, 32, 32) search and refinement
```

The bounce levels these controls select, with the same refined global bounds the
milestone 9 report records:

| File shorthand | NFP | `b` at `lambda_n = 0.8` |
| --- | ---: | ---: |
| PCA aspect6 (`20260402-01-038`) | 4 | 11.167688974652414 |
| TURBO aspect6 (`20260402-01-178`) | 4 | 11.094434030114910 |
| DMercFail nfp4 (`20260406-01-262`) | 4 | 10.648368010778030 |
| `d23p4` | 5 | 3.155082926916041 |
| `n3are` | 3 | 6.637692367842499 |

## Outcomes

All 20 transition cases and all 20 cut cases completed without an exception.
Every surface extraction is `REGULAR`.  Critical-curve extraction is `REGULAR` in
12 cases and `DEGENERATE` in 8 (PCA in all four combinations, `n3are` in all four).

| Transition status | Count |
| --- | ---: |
| `UNRESOLVED` | 22 |
| `MULTIWAY` | 10 |
| `BUDGET_INSUFFICIENT` | 4 |
| `MAX_PERIODS` | 4 |
| `REGULAR` | 2 |

These are the same failure classes milestone 10.3 exists to remediate, in the same
proportions the coarse matrix shows at the other levels; `lambda_n = 0.8`
introduces no new class.  The `d23p4` `MAX_PERIODS` result at this level is the
behaviour the milestone 9 report already records for `d23p4` at `lambda_n = 0.9`
and `0.95`: near `B_max` the wells are long and the default 128-period cap binds.
It stays an explicit cap, never a zero-action or zero-weight substitution.

The cut stage inserted no cut in any of the 20 cases, as in the milestone 10.1 and
10.2 coarse matrices — the bounded budgets leave the transitions uncertified, and
uncertified is not cut.  It produced 191 explicit event nodes, all with unlocalized
geometry (localization budget exhausted, or a source classification or period-cap
failure upstream).  Every serialization round trip passes, and invalid port
incidence is zero in every case.

## Backend and extractor consistency

Transition status counts agree between the two extractors in all 20 cases, on both
backends and every file.  Critical-curve and arc counts agree as well; the two
extractors differ only in how many event intervals localization attempts before its
budget runs out, on the three files that have events at all (24 vs 22, 36 vs 34, 33
vs 32).

Backend differences are the documented background-resolution sensitivity of arm
counts near the global extrema, not a disagreement about the physics:

| File | structured `GAMMA_MAX` / `GAMMA_MIN` | gmsh `GAMMA_MAX` / `GAMMA_MIN` |
| --- | --- | --- |
| PCA aspect6 | 3 / 2 | 3 / 2 |
| TURBO aspect6 | 1 / 0 | 3 / 0 |
| DMercFail nfp4 | 1 / 0 | 1 / 0 |
| `d23p4` | 1 / 0 | 1 / 0 |
| `n3are` | 3 / 2 | 4 / 2 |

On TURBO the gmsh surface has three components where the finer structured surface
has one, and the sheet count follows the component count (3 vs 1).  This is the
same convergence signal `docs/STATUS.md` records for the other levels: counts near
the extrema are background-resolution controlled and belong in a §21.3 report, not
in a tuned knob.

## One number worth carrying into milestone 10.3

The largest relative scalar flux change during event-vertex insertion is `4.13e-3`
(TURBO, gmsh, PyVista; `3.89e-3` for marching tetrahedra on the same mesh).  The
milestone 10.2 matrix recorded `8.77e-4` as its largest.  The cases above it are
exactly the coarse gmsh TURBO surfaces — 201 points carrying 5 event insertions —
so this is the measured drift of insertion on an under-resolved surface, not an
error bound on `K`-weighted volume or `Theta`.  Every structured case at this level
stays at or below `3.81e-5`.  Milestone 10.3's local refinement should shrink it;
if it does not, that is a finding, not a tolerance to widen.
