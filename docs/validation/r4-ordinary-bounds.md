# R4 ordinary action-bin accessibility

R4 implements finite lower/upper action-contour reachability on root-labelled
ordinary triangle charts. It preserves action, explicit spatial adjacency,
physical EDGE seeds, finite-area flat actions and saddle point incidence.
Unproved exterior incidence seeds only the upper calculation. This report does
not compute a physical pitch-slice population or f.

## Scope and provenance

The numerical baseline is `f85c0de3336ba845145c92fe626764f1cdf2a178`. Exact hashes of the two added numerical
modules and all five boozmn files are recorded in
[r4-ordinary-bounds.json](r4-ordinary-bounds.json). Its R3.5 evidence hash preserves
that source unchanged. The driver is [r4_ordinary_bounds.py](r4_ordinary_bounds.py).
Every nonempty model stores its mesh inputs and action partitions. CI reconstructs
these inputs and reruns the original/nested bin calculations, comparing bit counts,
geometric areas and seed statuses. Source: `h(rho)=1`; all reported areas use `|ds d-alpha|`, without K weighting.
The 30 pitches retain R0's radially global extrema **estimates**, not certified
extrema. Every case reloads its field; the OS file cache may be warm. The saved R3.5 root-cell certificates are reused as validation inputs; their
original construction time is excluded. No stored numerical scan or action
quadrature cache is reused. These timings measure this R4 replay, and do not
satisfy the final-candidate cold-start accounting contract.

Hardware: Apple M1 MacBook Pro (MacBookPro17,1), 8 CPU cores (4 performance +
4 efficiency), 16 GB memory, `macOS-14.7.6-arm64-arm-64bit`. One worker, BLAS threads set to one,
Python 3.10.5 in the required clean `.venv`.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/alpha-r4-mpl .venv/bin/python docs/validation/r4_ordinary_bounds.py
```

## Controls and interpretation

Coarse/fine atlases use `(n_s,n_alpha)=(3,4)/(5,8)`, two scan periods and eight
cell subdivisions. Their initial action partitions have 8/16 bins over the full
vertex range plus 1% padding; an available seed action is inserted in both.
Flat triangle actions automatically receive singleton atoms. Each mesh is also
recomputed with bisected bins **on identical geometry**, checking nested areas.
No nesting is asserted across the two different atlas geometries.

The fixed original R3 seed is used where available; otherwise the first saved
R3.5 expanded-search seed is attempted, without per-case tuning. A local corridor
uses the existing default step 1e-5 and 12 certificate subdivisions. The accepted
R3.5 local corridors are also overlaid wherever available, including all five
fields, using their saved seeds/step controls and current corner root/action
checks. Those inputs came from R3.5's declared bounded search policy, rather than
new per-case tuning. The original seed attempt and the reused corridor seed are
recorded separately. For DMercFail
0.8 and d23p4 0.1 the complete coarse/fine R3.5 disjoint path corridors are overlaid,
including their full unknown complement. Corner root counts and new action
quadrature estimates are checked again. Adjacency is accepted only with a common
root census and ownership; coordinate coincidence alone supplies no link.

The independent continuous oracle is rerun for those two classified original
queries, at steps 0.04/0.02, 80/160 steps and 4096 root-certificate boxes. DMercFail
remains numerically inaccessible and d23p4 accessible at both settings. The PL
bounds contain those outcomes as **UNKNOWN**; they do not independently classify
either path. In particular the sampled contour chords are never used to seal a
physical inaccessible region. The full corridor input is substantially larger
than the isolated local seed patches.

Vertex action quadrature and PL interpolation errors remain unbounded for these
real models. Consequently these are lower/upper sets for the finite PL model with
numerically integrated geometric areas, not field-level enclosures. No real
model result is promoted to a physical lower/upper weight. Nonconforming interfaces,
unproved periodic identifications and omitted geometry retain upper EDGE wildcard
incidence. Even a certified root cell can therefore remain globally uncertain.
Unknown complement multiplicity and K weight are not inferred from geometric area.

For certified uniform action envelopes the core expands upper transfer images.
It permits a nonzero-error lower transfer only with an independent proof that
all true local level sets are path-connected, and uses inward error margins.
Scalar interpolation or quadrature residuals alone do not provide that proof.
An explicitly uncertified geometry cell invokes a global upper EDGE wildcard
unless its incidence has been established in a certified model. Missing incidence
cannot silently become a closed boundary.
Error-band arithmetic rounds transfer endpoints inward/outward as appropriate;
area clipping remains numerical and is not a weighted-integration certificate.
The affine-envelope test demonstrates both containment and fixed-geometry nesting.

## All 30 physical cases

All 30 cases exercised both atlas levels; nonempty meshes exercised the new
bin backend on all five fields. 22/30 produced nonempty owned
ordinary meshes; the other cases retain unknown coverage, not zero population.
8 saved seeds could not be continued within the oracle's finite window;
their exact reasons remain in JSON, and both atlas calculations still ran.
Total matrix wall time: 520.70 s. Largest field-load-cold case:
d23p4 0.1, 327.16 s.
The maximum time for bin propagation plus geometric integration and its nested
comparison was 0.369 s per mesh. Active 60/300/600-second snapshots are in JSON;
no case reached the 600-second guard. Unresolved accuracy is never a success.

| Field | lambda_n | Triangles coarse/fine | Fine covered area | Fine reachable area L/U | Run wall (s) | Local seed corridor |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PCA | 0.05 | 4/4 | 1.7582628e-09 | 0/1.7582628e-09 | 3.64 | UNKNOWN |
| PCA | 0.1 | 4/4 | 1.6533209e-05 | 0/1.6533209e-05 | 3.58 | UNKNOWN |
| PCA | 0.5 | 4/4 | 0.0017236629 | 0/0.0017236629 | 5.01 | UNKNOWN |
| PCA | 0.8 | 0/0 | 0 | — | 3.27 | UNKNOWN |
| PCA | 0.9 | 0/0 | 0 | — | 3.07 | unknown seed |
| PCA | 0.95 | 0/0 | 0 | — | 3.79 | unknown seed |
| TURBO | 0.05 | 24/24 | 6.1021346e-07 | 0/6.1021346e-07 | 9.00 | CERTIFIED |
| TURBO | 0.1 | 4/4 | 0.00031912637 | 0/0.00031912637 | 3.99 | unknown seed |
| TURBO | 0.5 | 4/4 | 0.0017860761 | 0/0.0017860761 | 3.96 | UNKNOWN |
| TURBO | 0.8 | 4/4 | 4.5495468e-05 | 0/4.5495468e-05 | 3.89 | UNKNOWN |
| TURBO | 0.9 | 0/0 | 0 | — | 3.08 | UNKNOWN |
| TURBO | 0.95 | 0/0 | 0 | — | 2.89 | unknown seed |
| DMercFail | 0.05 | 24/28 | 0.00071195641 | 0/0.00071195641 | 8.91 | CERTIFIED |
| DMercFail | 0.1 | 28/32 | 0.00024165499 | 0/0.00024165499 | 9.81 | CERTIFIED |
| DMercFail | 0.5 | 32/80 | 3.1415927 | 0/3.1415927 | 15.54 | CERTIFIED |
| DMercFail | 0.8 | 140/174 | 0.02956646 | 0/0.02956646 | 34.37 | CERTIFIED |
| DMercFail | 0.9 | 4/4 | 1.7854255e-05 | 0/1.7854255e-05 | 3.98 | UNKNOWN |
| DMercFail | 0.95 | 2/4 | 1.9177683e-05 | 0/1.9177683e-05 | 3.59 | unknown seed |
| d23p4 | 0.05 | 8/8 | 0.00037827126 | 0/0.00037827126 | 4.38 | UNKNOWN |
| d23p4 | 0.1 | 1298/1302 | 0.00065127275 | 0/0.00065127275 | 327.16 | CERTIFIED |
| d23p4 | 0.5 | 8/8 | 0.00081459133 | 0/0.00081459133 | 4.89 | UNKNOWN |
| d23p4 | 0.8 | 2/4 | 0.000256922 | 0/0.000256922 | 3.90 | UNKNOWN |
| d23p4 | 0.9 | 0/0 | 0 | — | 4.30 | unknown seed |
| d23p4 | 0.95 | 0/0 | 0 | — | 7.07 | unknown seed |
| n3are | 0.05 | 2/4 | 1.4832368e-05 | 0/1.4832368e-05 | 6.54 | UNKNOWN |
| n3are | 0.1 | 2/4 | 0.00037460037 | 0/0.00037460037 | 7.09 | unknown seed |
| n3are | 0.5 | 2/4 | 6.7523587e-05 | 0/6.7523587e-05 | 6.59 | UNKNOWN |
| n3are | 0.8 | 4/4 | 3.7536884e-06 | 0/3.7536884e-06 | 8.11 | UNKNOWN |
| n3are | 0.9 | 2/4 | 2.8042571e-07 | 0/2.8042571e-07 | 8.41 | UNKNOWN |
| n3are | 0.95 | 0/0 | 0 | — | 4.92 | UNKNOWN |

The slow d23p4 case spends its time constructing and matching root-labelled
charts, rather than propagating action bins. These coarse experiments leave most
coverage or connectivity uncertain and provide no evidence yet for the final f
accuracy/runtime contract. Root/action error certification and useful global
coverage remain necessary; R5 must not hide them behind small local event weights.

## Acceptance and verification

| DESIGN R4 requirement | Named test |
| --- | --- |
| Bounds enclose independent regular contours | `test_ordinary_bounds_bracket_direct_contours`; `test_production_bounds_bracket_independent_synthetic_contours` |
| Disconnected equal-action branches stay separate | `test_disconnected_equal_action_branches_do_not_connect` (overlapping charts, with the second sheet inside s=1); synthetic two-well adapter test |
| A=s annulus has no interior radial escape | `test_connected_reeb_interval_is_not_radial_escape` |
| Nested bins grow inner/shrink outer sets on fixed geometry | `test_nested_action_bins_tighten_fixed_geometry_bounds`; `test_certified_level_topology_and_uniform_errors_bracket_affine_field` |
| Flat actions and saddles preserve measure | `test_flat_action_and_saddle_cases_preserve_measure` (analytic areas 0.9 and 0.5) |
| Unknown data and actual EDGE seeds are handled conservatively | `test_unknown_data_retains_area_and_remote_influence`; `test_unknown_geometry_never_assumes_missing_incidence_is_closed`; `test_action_error_does_not_become_a_certified_path` |
| Real production comparisons retain unresolved coverage and ownership | `test_real_ordinary_bounds_compare_classified_oracle_paths`; `test_full_real_corridor_preserves_ownership_and_possible_exterior`; `test_r4_real_matrix_retains_coverage_and_classified_oracle_comparisons` |

The tilted affine chart's exact reachable area is `(1-0.1**2)/2=0.495`.
The perturbed affine-envelope control has exact area 0.49095. All worklist bits
change at most once per calculation, verified against `2*triangles*atoms`.

The new tests first failed because the ordinary backend did not exist. Two
mutations were applied to the production backend and reverted after red tests:

- Allowing action changes inside a visited triangle made the annulus regression
  report an interior UNKNOWN instead of its required INACCESSIBLE.
- Linking all triangle pairs through action overlap made the disconnected second
  sheet ACCESSIBLE. The disconnected-branch regression failed.

A further adversarial regression first returned area `(0,0)` for explicitly
uncertified geometry with no supplied incidence; the corrected global upper
wildcard retains `(0,0.9)`, including distant states.

A Python 3.12 CI fast run passed scientifically but took 124.83 s. The local
adapter checks now retain an explicit unresolved background on both original
grids instead of repeating failed global censuses; their owned root/action
patches are still certified afresh. Actual global atlas inputs remain in the
30-case matrix and CI replay. The existing R3.5 two-resolution regression is
parameterized per resolution, preserving every field, resolution, control and
assertion while avoiding a 42-second combined fast case. No scientific check or
tolerance was removed, and no slow/skip/xfail marker was added. The disconnected
sheet control keeps its second sheet strictly inside s=1, with overlapping chart
vertices and actions. Public entry points and legacy geometry are unchanged. No new dependency or DESIGN deviation was introduced. The new
[analytic action-bin diagnostic](r4-ordinary-plots/analytic-action-bins.png) and
[coarse](r4-ordinary-plots/dmerc-coarse.png)/[fine](r4-ordinary-plots/dmerc-fine.png)
real corridor diagnostics label the model scope and unweighted quantities.

`make check`: 245 passed in 98.90s (0:01:38).

`make test`: 244 passed in 74.24s (0:01:14).

## Unresolved test-time acceptance

All Tests jobs on `470e21a` pass, while same-revision fast CI runs range from
89.76 to 189.09 s. The local clean environment meets §22.5 at 74.24 s fast and
98.90 s full. [Proposed ADR 0012](../adr/0012-test-budget-reference-environment.md)
records the missing reference-environment decision and all CI durations. R4 is
not marked complete; its implementation PR remains draft. No budget interpretation,
tolerance relaxation or test removal has been adopted.
