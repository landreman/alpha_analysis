# Milestone 10.2 stopped investigation

This is evidence for **proposed ADR 0006**, not a completed validation report.
Milestone 10.2 remains unchecked. The draft must not be merged in this state.

## Reproduction

```bash
.venv/bin/python examples/investigate_event_junctions.py \
  --output /tmp/milestone102-evidence
.venv/bin/python -m pytest test/test_transition_events.py -q
```

The investigation script saves JSON evidence, a pickle-free cut snapshot, and
the diagnostic below. The checked-in JSON includes hashes of the experimental
implementation files. The script is not a replacement for the failing test.
Its optional `--w7x` path is provided for further investigation; a completed
W7-X cut has **not** been validated in this draft.

## Observations

| Quantity | Observed result |
| --- | --- |
| Synthetic bounce field | `3.001685664207343` |
| Background / extractor | structured `(4,16,36)` / marching tetrahedra |
| Incoming vertices / triangles | 571 / 904 |
| Source extraction / critical status | `REGULAR` / `REGULAR` |
| Source maximum-curve vertices | 71 and 69 |
| Folded input chart triangles | 13, all in incoming component 1 |
| Localized physical events / source occurrences | 2 / 4 |
| Experimental regular arcs | 4 |
| Accepted cuts / explicitly unresolved arcs | 3 / 1 |
| Actual sheets / independently expected sheets | **5 / 6** |
| Unresolved arc | 3, with all three ports retained at sheet ID `-1` |
| Diagnostic run including PNG | 12.64 s |

The two lower maxima and the bounding ordinary crossings define six limiting
well intervals independently of the triangulation; ADR 0006 gives the equations.
The test therefore continues to require six sheets. Replacing that check with
five, deleting it, labeling the accidental path branch an event, or omitting the
unresolved arc would conceal the failure.

The source mesh's 13 reversed chart triangles are genuine folds in its field-line
projection: adjacent triangles have consistent mesh-edge winding. Local cavity
retriangulation and field-line-chart insertion improve the result but do not yet
complete the fourth arc. The current refusal names a branch at vertex 243 and a
dangling sample at vertex 811. IDs are diagnostic to this run, not golden values.

![Unresolved event-junction cut](milestone10.2-event-junctions.png)

The image shows the incoming component carrying the unresolved arc. The blue
curve is the mapped companion curve; it is **not** an accepted cut. The stars
are the independently localized events. No cut is manufactured across the
unresolved strip.

## Tests and limits

- `test_localized_contacts_preserve_the_two_equal_height_maxima` passes. It
  compares against independent scalar equations for a second maximum whose
  height crosses the bounce field between source samples.
- `test_contact_arcs_share_physical_events_and_have_one_sided_actions` passes.
  It checks event sharing, analytic event coordinates, field-line identity, and
  independently integrated limiting action additivity.
- `test_regular_arcs_cut_into_six_wells_without_dangling_event_ends` fails at
  **5 != 6**. Its remaining assertions are not claimed as verification.
- The existing `test_dmerc_reference_sheet_graph_is_budget_invariant_or_explicit`
  also fails: the experimental path-degree guard retains one unresolved
  transition where the existing test requires a cut. This is a regression to
  diagnose and fix before merge, not an accepted change to milestone 10.1.

The baseline full check passed all 166 tests in 55.31 s of pytest time. The final
`make check` passes formatting, then reports **167 passed, 2 failed in 56.11 s**
(57.94 s total command wall time). Its slowest test is the existing bounce-point
plot test at 16.76 s; the new six-sheet test takes 8.05 s, and the shared event
fixture takes 3.83 s. These timings are within the budget, but the gate is red.
The separate fast tier reports the same **167 passed, 2 failed in 48.26 s**
(48.54 s command wall time); its slowest test is the bounce-point plot at 18.39 s,
and the new six-sheet test takes 9.01 s. Both runs use three pytest workers in
the required clean Python 3.10.5 virtual environment.

The tests were first run against unimplemented stubs, then through the prototype.
No test was weakened, skipped, xfailed, deleted, or newly marked slow. Mutation
verification, the required five-equilibrium 100-case matrix, event serialization
round-trip tests, complete nongeneric-sample handling, and final API/diagnostic
integration remain unfinished because the STOP condition was reached.

The W7-X localization probe used
`boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc`, `b=2.7781394`,
and a structured `(6,24,12)` background. Its four contact occurrences pair into
two events at `s=0.1184098663731` and `s=0.318907160215`, with localized parameter
intervals narrower than `1e-5`. That probe took 67.25 s including source mapping.
This is localization evidence only; the real cut acceptance criterion is still
unmet, and no matrix convergence claim is made.
