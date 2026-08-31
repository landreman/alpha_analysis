# Milestone 10.2 five-equilibrium matrix

All **100 cases completed without exceptions**, reproducing the previous matrix's
164 source curves. This is a bounded contact-localization and cut diagnostic,
not a converged connectivity result. No real-matrix arc passed both sampling and
geometric cut certification at these controls. The separate W7-X reference test
uses full source mapping and does cut two regular arcs with explicit events.

## Controls and reproduction

```bash
source .venv/bin/activate
python examples/validate_cut_equilibria.py --localize-events \
  --output /tmp/milestone102-matrix.json
```

The five files are exactly those in `AGENTS.md`; each uses radially global
`lambda_n = 0.05, 0.1, 0.5, 0.9, 0.95`, both background backends, and both extractors.
Bounds use `(17,32,32)` samples with local refinement. Structured resolution is
`(6,24,12)` and Gmsh target size is `0.3`. Source mapping uses an 8-vertex work budget,
the default 128-field-period cap, and quadrature order 32 with cap 512. Contact
localization uses at most 20 midpoint traces per original bracket, shared when
intermediate count levels split it, a `1e-5` final-u target, and density-doubled
endpoint/midpoint probes for scan artifacts. All numerical tolerances are unchanged.

No surface is downsampled after cutting. These diagnostics keep the authoritative
extracted mesh, skip surface-wide action traces, and use production side probes
when an arc reaches side assignment. Every unknown-action cell is retained and
its dimensionless flux measure is recorded. The slowest case took 471.22 seconds;
case runs were parallel, so these are not test-budget timings. Fine scans of long
wells dominate the high-bounce cases; they are not replaced by zero action or weight.

## Outcomes (sums across twenty cases per file)

| File shorthand | Source curves | Event nodes | Localized event geometries | Unresolved arcs |
| --- | ---: | ---: | ---: | ---: |
| PCA aspect6 (`20260402-01-038`) | 51 | 206 | 5 | 256 |
| TURBO aspect6 (`20260402-01-178`) | 28 | 184 | 0 | 204 |
| DMercFail nfp4 (`20260406-01-262`) | 30 | 0 | 0 | 30 |
| d23p4 | 12 | 10 | 4 | 24 |
| n3are | 43 | 394 | 3 | 437 |
| Total | 164 | 794 | 12 | 951 |

An event node with unlocalized geometry represents an unresolved source interval,
not a claim that one physical equal-height event has been identified there. A
bracket can contain several count changes. Counts above repeat physical geometries
across backends/extractors and are not counts of distinct equilibria-wide events.

| Primary unresolved-arc reason | Count |
| --- | ---: |
| Localization or source work budget exhausted | 770 |
| Source critical-point classification | 69 |
| Field-line period cap | 50 |
| No resolved regular span between events | 25 |
| Count change localized but event type unclassified | 19 |
| Cut geometry unresolved | 12 |
| Failed or newly nongeneric arc sample | 6 |

Every arc remains present with all three ports. All serialization round trips
pass. Invalid port incidence and triangles spanning an accepted action jump are
both zero (there are no accepted cuts in this bounded matrix). Unknown-action flux
is never omitted and does not exceed total cut flux. The largest scalar flux
change during insertion is `8.77e-4` relative; this is measured drift, not an error
bound on K-weighted volume or Theta.

## Backend/extractor comparison and diagnosed failures

Source-curve, sheet, event-node, and arc counts agree between extractors in 22/25
structured cases and 16/25 Gmsh cases. The largest structured discrepancy is PCA
at `lambda_n=0.5`: marching tetrahedra reaches 21 candidate event intervals, whereas
PyVista stops with source-classification failures before bracketing any contact.
Both have four source curves and one uncut surface component. That is different
resolved information, not evidence that one equilibrium has no contacts.

Other discrepancies are mostly numbers of unresolved intervals found before the
shared localization budget runs out. Gmsh PCA at `lambda_n=0.1` also has two versus
three source curves, and Gmsh n3are at `lambda_n=0.9` has eight versus seven. Those
are already differences in the input critical curves. Across mesh backends, these
counts remain resolution controlled. No backend/extractor disagreement is promoted
to a resolved physical topology. Milestone 10.3 must coordinate critical/background
refinement, source budgets, contact localization work, and cap convergence.

Three initial d23p4 `lambda_n=0.5` runs raised a field-line-chart insertion error.
The failing point was the known PL `EDGE` snap, whose nonlinear chart need not
contain its Euclidean position. Inserting it through the same component's tagged
boundary fixes the failure without changing the distance allowance. All four
backend/extractor combinations were rerun: the remaining nonseparating arc and
sampling failures are now explicit unresolved outcomes, not exceptions.

Exactly sampled terminal contacts also exposed an open-source wrapping bug.
An open source's final u now stays at its upper endpoint; it does not wrap to zero
or create a zero-length incident arc. The fast synthetic test covers this, and
all TURBO `lambda_n=0.1` combinations were rerun. PCA Gmsh `lambda_n=0.5` was also
rerun because a regular arc reached the changed insertion machinery.

## Provenance and limits

`milestone10.2-real-equilibria.json` contains every case, exact controls, per-run
source hashes, original source certification, localized/uncertain intervals,
per-arc status and reasons, and action/flux diagnostics. It preserves run provenance
because the endpoint fixes were made while the matrix was running. Every case with
an exactly sampled nongeneric event or a regular certified arc was rerun on the
final core; unaffected earlier cases exercised unchanged paths. Nothing was
reclassified merely to remove a failure from the report.

The synthetic 8/10/16/full budget sweep is recorded separately in
`milestone10.2-event-budget-sweep.json`: full mapping gives six sheets, while smaller
budgets retain explicit sampling or geometry failures. Certification remains relative
to source-curve and root-scan resolution. The 100-case matrix is not the failure-
directed convergence acceptance required by milestone 10.3, and no final f or Theta
value is computed here.
