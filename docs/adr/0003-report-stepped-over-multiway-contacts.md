# ADR 0003: Report equal-height contacts that fall between transition samples

- **Status:** Proposed
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

A barrier can only leave the interior of the parent well by rising through \(b\), at
which point the field-line crossing lands on it instead. Each of those four brackets
therefore contains a second maximum of exactly the marginal height: the nongeneric
multiway event of §5.4, stepped over by the sampling. The same signature appears at
\(b=2.8319871\) with a margin of \(1.0\times10^{-4}\).

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
   Detection resolution is the critical-curve vertex spacing; no transition-side
   control refines it, so a contact between two vertices that share a barrier count
   is still missed.
3. **Detect, then bisect along the polyline to localize each contact and split the
   hyperedge** — the honest end state, and what §10.4's interval transfer wants. It
   needs a new sample inserted at a located contact, which is the constrained-cut
   machinery milestone 10 is building; doing it here duplicates that work and would
   create transition samples with no corresponding critical-curve vertex.
4. **Detect and report only in diagnostics, leaving the status `REGULAR`** — keeps
   downstream green, but a status that says `REGULAR` while the ports jump is exactly
   the plausible-but-wrong number §21.2 forbids.

## Decision

Left blank until the researcher decides. Option 2 is implemented on this branch so
the evidence and the test are concrete; option 3 remains the milestone 10 follow-up
either way.

## Consequences

- `TransitionCurve` gains three optional arrays, appended after the existing fields so
  positional construction stays valid (covered by the existing
  `legacy_positional_curve` assertion): `interior_maximum_count` (`-1` where no well
  was traced), `barrier_margin` (`inf` where the well has no other maximum), and
  `contact_sample_pairs` (shape `(n, 2)`).
- `_DirectionalTrace` carries the curvature and \(B-b\) of every recorded extremum.
  These were already evaluated inside the scan's tangent test; nothing new is
  computed, and no extra field evaluations are made.
- Curve status is now `MULTIWAY` when a contact is bracketed even though every sample
  is `REGULAR`. Milestone 10 must subdivide such curves at the located contact rather
  than treat the bracket as an ordinary monotone segment; until it does, a caller that
  needs the actions should read `sample_status`, not `status`.
- `plot_transition_diagnostics` shades each bracketed \(u\) band on the action panel,
  so a jump is not read as a steep regular slope (§17.5).
- New tests: `test_equal_height_contact_between_samples_is_multiway_not_regular` and
  `test_contact_bracket_is_shaded_in_the_transition_diagnostic`, both on a synthetic
  field whose barrier height crosses \(b\) strictly between two sampled `GAMMA_MAX`
  vertices, with the interior-maximum counts verified independently by direct
  sampling of \(B\) inside the test.
- Real equilibria make this the common case, not a corner case. With
  `max_curve_samples=10` on a 6x24x12 structured background, every mapped curve on
  `boozmn_20260402-01-038_Ax_PCA_...` (lambda_n = 0.5 and 0.9) and on
  `boozmn_d23p4_tm_ns51_mbooz16_nbooz16.nc` (lambda_n = 0.5) brackets between three
  and six contacts, with parent wells holding up to 103 interior maxima; every
  additivity residual stays at 1e-14 to 1e-12. Coarse sampling of a long curve steps
  over many contacts, so `MULTIWAY` will be the usual pre-cut status until milestone
  10 subdivides. Callers must therefore branch on `contact_sample_pairs` and
  `sample_status`, not on `status is REGULAR`, and the useful question for a
  convergence report is how the bracket count behaves as the critical-curve sampling
  is refined.
- `docs/STATUS.md`'s note deferring "neighbor-itinerary jump detection" to milestone
  10 is narrowed: the parent-well count change is detected now; what milestone 10 still
  owns is locating the contact and cutting there.
