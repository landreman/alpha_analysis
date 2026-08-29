# ADR 0003: Report equal-height contacts that fall between transition samples

- **Status:** Decided
- **Date:** 2026-08-29
- **Milestone:** 9 (follow-up on branch `fix/multiway-contacts-between-samples`)
- **Design sections:** §5.4 (multiway and degenerate transitions), §10.2 and §10.4
  (companion curve and transition representation), §21.2 (forbidden numerics), §23
  (milestone 9 acceptance)

## Context

`map_transitions` samples a `GAMMA_MAX` polyline at its critical-curve vertices and,
at each sample, traces the parent well and computes \(A_W\), \(A_1\), \(A_3\). A
sample that itself sits on a second maximum of height \(b\) is already detected and
returned as `MULTIWAY` (§5.4). Nothing looked between samples.

On the reference W7-X equilibrium
(`boozmn_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc`, structured
6×24×12 background) this is not hypothetical. At \(b=2.7781394\) the single 65-vertex
`GAMMA_MAX` curve comes back `REGULAR` with every sample additive to \(1.6\times
10^{-13}\), yet 13 of the 65 samples have a second maximum inside the parent well and
the smallest recorded margin is \(b-B_{\mathrm{barrier}}=7.3\times10^{-4}\). At the
four \(u\) values where that barrier enters or leaves the well, \(A_W\) jumps by about
1.75 against a median sample-to-sample step of 0.025:

```
u 1.0823 -> 1.0956 : dA_W = 1.812, barrier margin inf -> 7.3e-4
u 0.6610 -> 0.7109 : dA_W = 1.751, barrier margin 3.8e-3 -> inf
u 1.3005 -> 1.3340 : dA_W = 1.744, barrier margin 1.3e-3 -> inf
u 0.4650 -> 0.4939 : dA_W = 1.736, barrier margin inf -> 1.4e-3
```

A barrier leaves the interior of the parent well in one of two ways: by rising through
\(b\), at which point the field-line crossing lands on it instead, or by annihilating
with its neighboring minimum in a fold (\(D_\parallel^2 B\to0\)) at a height that can be
far below \(b\). The first is §5.4's second bullet and makes \(A_W\) jump; the second is
its first bullet and does not. Both must be reported, and the recorded margin separates
them: at all four brackets above the barrier is within \(1.4\times10^{-3}\) of \(b\)
while the sampled part of the curve runs out to \(3.8\times10^{-3}\), so these are
equal-height contacts, not folds. The margin is that of the *highest* barrier while
the bracket is emitted by a change in the count, which any barrier can cause, so it
discriminates the two only when the count change belongs to that highest barrier. It
does here — only 13 of the 65 samples have a second maximum at all, so the counted
barrier is the one that moves — but not in a well holding dozens of them, where the
highest barrier sits near \(b\) regardless. The same signature appears at \(b=2.8319871\) with a
margin of \(1.0\times10^{-4}\).

§5.4 requires that the code "must refine or report an unresolved nongeneric event"
when it detects "a jump in itinerary larger than the generic one-maximum change", and
§23 lists multiway-event detection among milestone 9's changes. What the design does
not settle is what a curve with a stepped-over contact should *be*: which status it
carries, and whether its port samples remain usable.

This is not covered later by accident. Milestone 10's acceptance ("no triangle spans
an action jump") is a property of the cut surface mesh, which a steep but connected
ramp along \(T\) can satisfy. §10.4 says to "subdivide a transition curve at extrema
of any port function \(A_p(u)\)", after which "every \(A_p(u)\) is piecewise linear and
monotone or constant, which makes interval transfer unambiguous" — a jump is not an
extremum, so an undetected contact stays inside one segment and milestone 13's
transition-aware transfer would carry action intervals across a branch change and
return a plausible, wrong connectivity (§21.2).

## Options

1. **Leave detection to milestone 10's neighbor-itinerary check (status quo)** — costs
   nothing now, but the pre-cut record keeps claiming `REGULAR` for curves whose port
   actions are discontinuous, and the §10.4 subdivision rule provably does not
   recover the contact. Anything built on the hyperedge before that check exists
   inherits the wrong branch correspondence silently.
2. **Detect between samples and label the curve `MULTIWAY`, keeping the ports
   (implemented)** — the scan already computes every interior extremum of each half
   well as an action breakpoint, so the barrier count and its margin to \(b\) are
   free; adjacent regular samples whose counts differ bracket a contact. The curve
   keeps its per-sample statuses (all `REGULAR` here) and its finite actions, gains
   `interior_maximum_count`, `barrier_margin`, and `contact_sample_pairs`, and the
   curve status becomes `MULTIWAY` through the existing conservative aggregation.
   Cost: real W7-X curves that were `REGULAR` now report `MULTIWAY` — at
   \(b=2.7781394\), the one curve on the reference equilibrium. Callers that gate on
   `status is REGULAR` see fewer usable curves until milestone 10 subdivides them,
   although every sample and action they need is still present and marked regular.
   Detection has two resolution limits. Along the curve it is the critical-curve
   vertex spacing, which no transition-side control refines, so a contact between two
   vertices that share a barrier count is still missed. Along the field line it is the
   root-scan step (§21.3 dimension 5, `samples_per_field_period` and
   `samples_per_wavelength`): the scan records one extremum per cell that brackets a
   sign change of `dB/dl`, so a maximum and minimum inside one cell are both missed,
   and the count can change where no event occurred. The bias is toward over-reporting
   a bracket, and a missed barrier that is not the highest one moves the count without
   moving `barrier_margin`, so the fold/contact discriminator is blind to that case.
   A count change that straddles a non-regular sample is not bracketed at all (both
   endpoints must be `REGULAR`) — there the sample's own status is the only signal. On
   a closed curve with only two samples the wrap arc is not compared separately,
   because `(1,0)` would repeat the comparison `(0,1)` already makes; a count change is
   then attributed to `[u[0], u[1]]` even when the event lies in the wrap arc, so
   detection still fires but the reported location can be the wrong arc. Because that spacing comes from the surface mesh and the
   extractor, the bracket count is backend- and extractor-dependent even though the
   field-line trace behind it is not.
3. **Detect, then bisect along the polyline to localize each contact and split the
   hyperedge** — the honest end state, and what §10.4's interval transfer wants. It
   needs a new sample inserted at a located contact, which is the constrained-cut
   machinery milestone 10 is building; doing it here duplicates that work and would
   create transition samples with no corresponding critical-curve vertex.
4. **Detect and report only in diagnostics, leaving the status `REGULAR`** — keeps
   downstream green, but a status that says `REGULAR` while the ports jump is exactly
   the plausible-but-wrong number §21.2 forbids.

## Decision

Option 2 approved.

## Consequences

- `TransitionCurve` gains three optional arrays, appended after the existing fields so
  positional construction stays valid (covered by the existing
  `legacy_positional_curve` assertion): `interior_maximum_count` (`-1` where no well
  was traced), `barrier_margin` (`inf` where the well has no other maximum and `NaN`
  where no well was traced, so a caller filtering on `np.isfinite` can tell the two
  apart), and `contact_sample_pairs` (shape `(n, 2)`). A curve demoted whole after
  mapping — a duplicate companion component — has all three cleared, so every bracket a
  caller sees still names two regular samples.
- `_DirectionalTrace` carries the curvature and \(B-b\) of every recorded extremum.
  These were already evaluated inside the scan's tangent test; nothing new is
  computed, and no extra field evaluations are made.
- Curve status is now `MULTIWAY` when a contact is bracketed even though every sample
  is `REGULAR`. Milestone 10 must subdivide such curves at the located contact rather
  than treat the bracket as an ordinary monotone segment; until it does, a caller that
  needs the actions should read `sample_status`, not `status`.
- A bracket lifts only a curve that is otherwise fully regular (`_curve_status`). It
  never displaces a sample-level failure, so cap exhaustion remains
  `MAX_PERIODS` and a failed action remains `UNRESOLVED` as `docs/STATUS.md` and
  `docs/validation/milestone9-real-equilibria.md` record; the bracket is in
  `contact_sample_pairs` either way.
- The barrier count uses the same `D2_tolerance` gate as the existing tangent tests, so
  an extremum with `|D_parallel^2 B|` below tolerance — §5.4's first bullet — counts as
  neither barrier nor minimum. The bias is conservative: near a fold the count drops
  and a bracket is emitted, which is the outcome §5.4 wants.
- `plot_transition_diagnostics` shades each bracketed \(u\) band on the action panel,
  so a jump is not read as a steep regular slope (§17.5). A closed curve's wraparound
  bracket runs from the last sample through `total_u_length` to the first, so it is
  drawn as the two spans it occupies; shading between its endpoints would shade the
  complement.
- New tests, all on a synthetic field whose barrier height crosses \(b\) strictly
  between two sampled `GAMMA_MAX` vertices, with the interior-maximum counts verified
  independently by direct sampling of \(B\) inside the test:
  `test_equal_height_contact_between_samples_is_multiway_not_regular` (both halves of
  the parent well, via a mirrored field, plus a control field whose barrier never
  crosses \(b\));
  `test_contact_bracket_is_shaded_where_the_contact_actually_lies`, which asserts the
  shaded x-extents against `u[contact_sample_pairs]`, wraparound included, not a legend
  string; `test_a_closed_curve_brackets_a_contact_in_its_wraparound_arc`, which puts
  the count change in the arc from the last sample back to the first and drives the
  plot from the emitted row; `test_a_bracket_never_outranks_a_sample_level_failure`,
  which pins the precedence at the helper and on the production path; and
  `test_interior_maximum_data_reports_the_highest_barrier_inside_the_well`, which pins
  the reduction over several barriers and the exclusion of minima and of extrema beyond
  the crossing, which the field-based tests do not reach.
- Real equilibria make this the common case, not a corner case. Regenerating the
  milestone-9 sweep (`examples/validate_transition_equilibria.py`: five equilibria,
  five levels, structured and gmsh backends, both extractors, 8 samples per curve)
  leaves only 6 of its 164 transition curves free of a stepped-over event: 53 move from
  `REGULAR` to `MULTIWAY`, while the 92 `UNRESOLVED`, the 9 `MAX_PERIODS`, all 1,017
  sample statuses, and every failure reason are unchanged, as are the largest
  additivity ratio (7.83e-5) and identity errors (2.2e-16 in `s`, 1.4e-14 rad in
  `alpha`). Parent wells hold up to 103 interior maxima at these levels. Coarse sampling of a long curve steps
  over many contacts, so `MULTIWAY` will be the usual pre-cut status until milestone
  10 subdivides. Callers must therefore branch on `contact_sample_pairs` and
  `sample_status`, not on `status is REGULAR`, and the useful question for a
  convergence report is how the bracket count behaves as the critical-curve sampling
  is refined.
- `docs/STATUS.md`'s note deferring "neighbor-itinerary jump detection" to milestone
  10 is narrowed: the parent-well count change is detected now; what milestone 10 still
  owns is locating the contact and cutting there.
