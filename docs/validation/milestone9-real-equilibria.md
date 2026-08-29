# Milestone 9 real-equilibrium validation

This report records the transition-mapping sweep requested after Milestone 9.
It is a diagnostic and convergence study, not a replacement for the named
acceptance tests in `test/test_transitions.py`.

## Matrix and controls

The reproducible driver is `examples/validate_transition_equilibria.py`.  The
primary run covered all 100 combinations of five equilibria, five radially
global bounce levels, two background meshes, and two surface extractors:

```text
lambda_n = 0.05, 0.1, 0.5, 0.9, 0.95
background = structured (6, 24, 12), gmsh (target size 0.3)
extractor = MarchingTetrahedraExtractor, PyVistaSurfaceExtractor
transition samples = at most 8 uniformly selected critical-curve vertices
field-period cap = 128
action order = 32, adaptively doubled through 512
```

The global bounds used a `(17, 32, 32)` search and refinement.  They were:

| File | NFP | Refined global B_min | Refined global B_max |
| --- | ---: | ---: | ---: |
| `20260402-01-038 Ax PCA` | 4 | 7.715684270822196 | 12.03069015060996 |
| `20260402-01-178 TURBO` | 4 | 7.342416966461364 | 12.03243829602829 |
| `20260406-01-262 Ax nfp4` | 4 | 5.040465893072380 | 12.05034354020445 |
| `d23p4` | 5 | 2.293059021646035 | 3.370588903233544 |
| `n3are` | 3 | 4.909260723175189 | 7.069800279009321 |

The driver checkpoints JSON after every case and retains statuses, failure
reasons, topology, action ranges, error estimates, additivity, identities, and
stage timings.  A complete run is:

```bash
.venv/bin/python examples/validate_transition_equilibria.py \
  --output /tmp/milestone9-real-validation.json
```

## Primary results

All 100 cases completed without an exception.  The sweep produced 164
transition curves and 1,017 sampled transition points.

| Quantity | Result |
| --- | ---: |
| Surface extraction status | 94 regular, 6 unresolved |
| Critical-curve status | 70 regular, 22 degenerate, 8 unresolved |
| Transition-curve status | 6 regular, 92 unresolved, 9 max-periods, 57 multiway |
| Transition-sample status | 840 regular, 156 unresolved, 13 max-periods, 8 multiway |
| Nonregular sample reasons | 156 source classification, 13 period cap, 8 duplicate companion |
| Largest additivity residual / tolerance | 7.83e-5 |
| Largest quadrature error / tolerance | 0.924 |
| Largest port radial-identity error | 2.23e-16 |
| Largest port alpha-identity error | 1.43e-14 rad |

The curve-status row was regenerated after the between-sample contact detector
of ADR 0003 landed; every other row in this table is unchanged from the
original sweep.  53 of the 59 previously regular curves are now `MULTIWAY`
because their sampling steps over a nongeneric event: at 8 samples per curve
the parent well's interior-maximum count changes somewhere along nearly every
curve, and only 6 of the 164 curves are free of one.  The four originally
`MULTIWAY` curves are the duplicate-companion cases and are still counted here.
`UNRESOLVED` and `MAX_PERIODS` totals, all 1,017 sample statuses, and every
failure reason are identical to the original run: a bracket lifts only a curve
that is otherwise fully regular, so cap exhaustion remains distinct.

There were no action-quadrature or additivity failures after the fixes below.
The 156 classification failures are samples on critical curves whose source
point is explicitly degenerate or unresolved.  They retain `NaN` actions and a
per-sample reason instead of invalidating regular samples on the same curve.

For each fixed backend, the two extractors agreed on surface status and
component count in all 25 file/level pairs.  They agreed exactly on critical
status and polyline counts in 24/25 structured pairs and 22/25 gmsh pairs, and
on aggregate transition-status counts in 24/25 structured pairs and 23/25
gmsh pairs.  The disagreements all occurred on coarse, resolution-sensitive
critical geometry; none was an action or identity discrepancy.

## Convergence probes

The following targeted reruns distinguish transition behavior from upstream
background-resolution effects:

- The TURBO equilibrium at `lambda_n = 0.1, 0.5, 0.95` converged at structured
  `(10, 40, 20)` / gmsh target `0.2`.  All four backend/extractor combinations
  then had identical critical and transition counts at each level.
- `d23p4`, `lambda_n = 0.1`, was regular with no transition on the structured
  mesh.  Gmsh changed from unresolved at target `0.2` to regular at `0.15` for
  both extractors, confirming an upstream surface-resolution effect.
- `n3are`, `lambda_n = 0.05`, converged at structured `(10, 40, 20)` / gmsh
  `0.2` to one component, one `GAMMA_MIN`, and no transition in all four
  combinations.  At `lambda_n = 0.9`, gmsh target `0.1` converged to the
  structured result: four maxima, two regular transitions, and two explicitly
  endpoint-degenerate transitions for both extractors.
- The thin low-field surface of `20260406-01-262`, `lambda_n = 0.05`, remained
  resolution-sensitive.  Structured `(10, 40, 20)` became a regular surface
  with one maximum; gmsh refinement from `0.2` to `0.15` reduced spurious
  maxima from 8--10 to 2 but remained unresolved.  This is an explicit
  upstream surface status, not a mapped transition.
- Several levels of `20260402-01-038` remain background-resolution controlled,
  especially near the radially global extrema.  The two extractors agree
  within each sufficiently refined backend, but structured and gmsh topology
  has not converged to a common count.  Full mesh convergence remains the
  Milestone 17 responsibility already recorded in `docs/STATUS.md`.

Period-cap probes preserved the distinction between a long trace and no
connection.  The gmsh `20260402-01-038`, `lambda_n = 0.95` trace resolved when
the cap rose from 128/256 to 512.  Structured `d23p4`, `lambda_n = 0.9`, also
resolved at 512, as did the extractor-dependent structured `n3are`,
`lambda_n = 0.95` sample.  The gmsh `n3are`, `lambda_n = 0.5` sample still
exhausted a 512-period cap in both extractors.  Structured
`20260406-01-262`, `lambda_n = 0.9`, and structured `d23p4`,
`lambda_n = 0.95`, remained `MAX_PERIODS` at 1,024 and therefore stay explicitly
unresolved.

Sampling `d23p4`, `lambda_n = 0.5`, at 8, 16, and 32 common-`u` points left all
four combinations regular.  At 32 samples the two extractors agreed exactly on
the gmsh parent-action range; their structured parent-action maxima differed by
about `8.3e-4` relative.  Total critical-curve length differences were about
`1.8e-4` relative on structured and `5.7e-4` on gmsh.

## Problems found and fixed

1. A single degenerate endpoint caused the old curve-wide status gate to erase
   otherwise valid transition samples.  `TransitionCurve` now carries
   per-sample statuses and failure reasons, while its aggregate status remains
   conservative.
2. Fixed-order action quadrature could alias high-mode wells and adaptive
   parent quadrature could exhaust its subdivision limit after hundreds of
   internal extrema.  Every detected extremum is now a breakpoint.  Child
   actions refine their Gauss order to a configured cap; parent action uses
   independent adaptive quadrature with endpoint-regularizing transforms and a
   divided absolute-error budget over extrema-delimited intervals.
3. Duplicate companion detection formerly skipped an entire curve when only
   an endpoint was nongeneric.  It now compares the regular sample masks and
   still marks an overlapping duplicate component `MULTIWAY`.
4. Real-equilibrium validation previously required ad hoc commands.  The new
   checkpointed driver makes the complete matrix, targeted resolution sweeps,
   cap sweeps, and common-`u` sampling sweeps reproducible.
5. A curve whose sampling stepped over an equal-height contact was reported
   `REGULAR` while its port actions jumped across the stepped-over event.  The
   parent well's interior-maximum count and the highest barrier's margin to `b`
   are now recorded per sample, adjacent regular samples whose counts differ are
   bracketed in `contact_sample_pairs`, and such a curve is `MULTIWAY` (ADR
   0003).  The bracket never displaces a sample-level failure.

No tolerance was loosened and no failed trace was assigned zero action or zero
weight.  The remaining unresolved results are retained as convergence data.
