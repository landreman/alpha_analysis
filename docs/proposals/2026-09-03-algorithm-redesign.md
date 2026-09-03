# Proposal: ways forward for computing \(f\) on realistic equilibria

- **Status:** Draft for the researcher's decision (no code changed)
- **Date:** 2026-09-03
- **Scope:** replaces or reorders `docs/DESIGN.md` milestones 3–13 depending on the option chosen; the metric of §3 and the rules of §21.2 are untouched
- **Evidence:** `docs/validation/*.md`, `docs/adr/0001–0009`, the recorded matrices, and the measurements in Appendix A (run on 2026-09-02/03 on the five `data/boozmn_*` files)

## 0. Summary and recommendation

The 120-case matrix resolves 35% of cases after 112.6 h because the pipeline
represents the transition set as a sparse, backward-mapped companion polyline
\(T\) that must be certified event by event and then inserted as a constrained
cut into a PL surface extracted from a 3-D background mesh. On real equilibria
parent wells hold up to 103 interior maxima, only 6 of 164 transition curves are
free of a stepped-over count change, every open \(\Gamma_{\max}\) arm ends on a
point where the backward trace is singular, and the PL insertion does not
converge under refinement (ADRs 0003–0008). Cost and failure coincide: 50 of the
120 cases finish under 10 minutes and 49 of those are the no-transition or
resolved cases; the other 70 are certification ladders that end at a bound.

Three facts change the problem (§2): transitions are visible in the *forward*
trace of every well; the incoming-bounce surface projects diffeomorphically
onto the field-line chart \((s,\alpha)\), where the phase-space measure is
exactly \(ds\,d\alpha\); and the total \(K\)-weighted measure of the whole
surface is a plain volume integral, so every capped, failed or uncut well can
be bounded without being traced.

Seven options are laid out in §3. My recommendation (§5) is:

1. **Adopt the bounds-first framing now (option E).** Redefine success per
   pitch slice as "\(Q(b)\) with a rigorous lower/upper gap below a tolerance
   within the time budget", with the exact total-weight identity supplying the
   upper bound. Nothing in §21.2 is loosened; this is what §11.3 and §12.5
   already ask for. It makes the 20 `max_periods` cases and every event
   certification a matter of measure rather than of veto.
2. **Rebuild milestones 3–10 on the field-line chart (option C)**, with
   per-line well lists, local count-change detection, one field-line scan per
   line reused for every well on it, and event neighbourhoods left as bounded
   cells. It removes every recorded geometric failure class and, combined with
   the measured cost levers (option F), is the only option I can see reaching
   the 10-minute budget on the hard levels.
3. **Keep the direct contour tracer (option D) in the continuous field** as
   the independent oracle and as the estimator of last resort inside cells
   the flood fill leaves unresolved.
4. **Do not finish milestone 10.3 as specified (option A)**: with the current
   definition of "resolved" the 95% threshold is unreachable on these five
   equilibria (ADR 0009's arithmetic), and each further certification kind
   has moved 1–5 cases at hours per case.

Likelihoods I assign to reaching the researcher's goal ("\(f\) computed with a
useful bound for > 95% of cases within 10 minutes per case"): A ≈ 10%,
B ≈ 50–60%, C ≈ 75–85%, D alone ≈ 40%, each conditional on E and F being
adopted with it; without E, no option exceeds ≈ 40% because the `max_periods`
class alone is 16.7% of the matrix.

## 1. Why the current algorithm cannot reach 95%

The pipeline of `docs/DESIGN.md` §§8–10 represents the state space and the
transition set as follows, and every recorded failure sits in steps 2–5:

1. extract \(\Sigma_b^-\) as a PL surface from a tetrahedral background and
   split it by \(g=0\) (milestone 5, ADR 0001);
2. extract \(\Gamma_{\max}\) as chains of marching-triangle edges and classify
   by \(D_\parallel^2 B\) (milestone 8; ADR 0005 zigzags; ADR 0008 endpoints);
3. sample \(\Gamma_{\max}\) at 8–16 vertices, trace backward from each
   marginal point to the companion point \(a(u)\) (milestone 9, 10.1);
4. bracket interior-maximum count changes between samples, bisect along
   \(u\), and certify each as a contact, a fold or a degenerate endpoint before
   any incident arc may cut (ADRs 0003, 0007, 0008; milestones 10.2–10.3);
5. insert \(T\) as a constrained polyline, duplicate vertices, assign sides,
   union-find sheets (milestone 10; ADR 0004 endpoint snap; ADR 0006 corridors).

What the matrix and the ADRs established (details and citations in
`docs/validation/milestone10.3-real-equilibria.md` and the knowledge map
that accompanies this proposal):

- **Nongeneric is the norm.** Only 6 of 164 real transition curves are free of
  a stepped-over count change; parent wells hold up to 103 interior maxima
  (ADR 0003). The certified event census across the whole matrix is 48
  degenerate endpoints, 7 equal-height contacts, 3 folds: most "events" are
  where a curve simply ends.
- **Localization along a curve is arithmetically doomed at the recorded
  controls.** 20 bisections per bracket toward a \(10^{-5}\) target in \(u\),
  with brackets spanning most of a curve because certification stops at the
  first count change (3 of 46–115 vertices mapped), exhausted the budget on
  770 of 951 arcs; 17,044 localization traces localized 12 of 794 event
  nodes. Escalating to 320 bisections moved a handful of cases at 7200 s each.
- **The transition map is seeded at its own singular set.** Open
  \(\Gamma_{\max}\) arms end on \(D_\parallel^2B=0\) points
  (\(|D_2|\sim10^{-13}\)); 156 of 1017 milestone-9 samples and 69 arcs in 10.2
  failed there, vetoing whole curves until ADR 0008.
- **PL insertion does not converge under refinement.** Zigzags relocate
  (ADR 0005), endpoint gaps shrink only linearly (ADR 0004), folded chart
  triangles persist at every grid and the failing arc and rejection reason
  change with the grid (ADR 0006). Thin strips near `EDGE` are judged against
  an allowance set by the longest edge of the nearest sliver (TURBO
  \(\lambda_n=0.1\): \(T\) 0.1027 off the surface against 0.0067 allowed).
- **Long wells are physics, and today one capped backward sample vetoes a
  curve.** DMercFail \(\lambda_n\ge0.9\) and d23p4 \(\ge0.8\) persist at 1024
  periods; each capped scan costs ~45 s at 128 periods; 20 cases spend 40–54
  minutes escalating one vertex's cap. These 20 cases alone are 16.7% of the
  matrix.
- **Cost is the certification, not the physics.** Transition mapping is 91%
  of case time at \(\lambda_n=0.8\); the same cases took 5–471 s under 10.2
  controls and hours under the 10.3 ladders. A resolved DMercFail case costs
  2.1 s of mapping, 15.5 s of surface tracing and 2.2 s of cutting.
- **"Resolved" is topology only.** `unresolved_action_flux` equals essentially
  the whole cut flux in every resolved case, because the matrix skips
  surface-wide traces. No resolved real case yet carries an action field.
- **The backend × extractor axes carry little information.** Extractors agree
  on statuses in 25/25 surface and all \(\lambda_n=0.8\) cases; backends
  disagree on component counts near the global extrema and do not converge.

The conclusion of ADR 0009 is right: no coordinator that keeps §21.2 can reach
95% *resolved cuts* on these equilibria. The way out is to stop requiring a
certified global cut before any weight can be counted.

## 2. Facts that reframe the problem

These follow from the definitions in `docs/DESIGN.md` §§3–5 and §9 and were
checked numerically (Appendix A).

**F1. Transitions are visible from the forward trace of the well itself.**
The parent well \([a,d]\) contains the marginal maximum \(m\) as an interior
maximum whose height reaches \(b\); on the child side the first maximum past
the exit descends to \(b\), and the *count of wells on the field line* changes
by exactly one (the child-3 well \([m,d]\) appears). `WellTrace` already
records the extrema heights (`extrema_B`) and ADR 0003 already stores the
barrier margin. So \(T\) can be located by one-dimensional root finding along
any curve in the state space (a mesh edge or a chart-cell edge) without
constructing it globally or matching it to \(\Gamma_{\max}\) by a common
parameter, and the forward trace at the located point gives \(m\), \(A_W\),
\(A_1\), \(A_3\) and the additivity check. A below-\(b\) fold (ADR 0007)
changes the interior-maximum count but neither the well count nor \(A\): it is
not an event for this indicator. Measured: on DMercFail \(\lambda_n=0.8\), four
count-change edges of a 24×64 chart bisect to points with \(B-b=0\),
\(|D_\parallel B|<10^{-14}\), \(D_\parallel^2B=-63.9\), and the additivity
residual falls as the offset from the crossing: \(3\times10^{-4}\),
\(3\times10^{-5}\), \(3\times10^{-6}\), \(7\times10^{-8}\) at offsets
\(10^{-1}\ldots10^{-5}\) of the edge (Appendix A.2).

**F2. The incoming surface projects diffeomorphically along field lines.**
On \(\Sigma_b^-\), \(g=\mathbf b\cdot\nabla B<0\), so the surface is nowhere
tangent to the field; the map \(\Sigma_b^-\to(s,\alpha)\),
\(\alpha=\theta_--\iota(s)\zeta_-\), is a local diffeomorphism on the open
incoming half. Its only singularities are the boundary curves \(g=0\), which
are exactly the physical events. Each trapping sheet is an honest graph
\(\zeta_-(s,\alpha)\) with boundary, and \(\omega=ds\wedge d\alpha\) pulled
back to the surface is exactly \(ds\,d\alpha\) (§4.3): the measure of a chart
cell is its area times the number of wells over it. (The folds of ADR 0006 are
in the *extracted triangles* near \(g\to0\), where steep triangles straddle the
tangency; they are an artifact of meshing the surface in 3-D first.) The price
is the one-field-period quotient: the well with incoming point
\((\alpha,\zeta_-=L)\) is the well \((\alpha+\iota L,0)\), so sheets carry a
twisted seam not aligned with a grid in \(\alpha\). Decision 1 of §26 chose
the 3-D surface to avoid this; the prototype in Appendix A.2 handles it with
an extended scan window and "ghost" wells and finds it to be bookkeeping, not
a topological problem.

**F3. The total \(K\)-weighted surface integral is a volume integral.** With
\[
V(b)=\int ds\,d\theta\,d\zeta\,\frac{|C|}{B^2}\,h(\sqrt s)\,
\sqrt{\max(0,\,1-B/b)},
\qquad
Q_{\rm total}(b):=\int_{\Sigma_b^-}hK\,|ds\,d\alpha| = 2b^2\,\frac{dV}{db},
\]
because the velocity-space fraction at \(\mathbf x\) with bounce field
\(\le b\) is \(\sqrt{1-B/b}\), and at fixed \(s\) the field-line integral of
\((|C|/B)/\sqrt{1-B/b}\) over all \(B<b\) intervals is the sum of \(K\) over
all wells on the line. Check: \(\Theta\equiv1\) gives \(f=V(B_{\max})/V_h\),
the trapped fraction, as §17's benchmark requires. Consequence: for any set of
wells that failed, hit the period cap, or sit in unresolved cells, the
\(K\)-weighted measure is bounded *exactly* as \(Q_{\rm total}\) minus the
resolved part. Every §21.2 prohibition becomes a rigorous bound:
\(Q_{\rm lower}\) = resolved and reachable, \(Q_{\rm upper}=Q_{\rm total}-\)
(resolved and unreachable). [Numerical check: Appendix A.3.]

**F4. Codimension-two events carry no connectivity of their own.**
Equal-height contacts, degenerate \(\Gamma_{\max}\) endpoints (where
\(A_3\to0\)) and crossings of two \(T\) curves are isolated points of the 2-D
state space. Contours passing exactly through a point have measure zero;
contours passing near it cross the incident \(T\) curves at generic simple
points, which the local rule handles. An event cell can therefore be left
"unresolved with bounds" and shrunk by refinement; its contribution to the
bound gap vanishes with its measure. Certifying the event's *kind* (ADRs
0003/0007/0008) is not needed for the bounds to converge. Caveat: a
symmetry-enforced nongeneric structure (two maxima equal along a whole curve)
would make an event curve rather than a point; its measure is still zero but
the refinement is along a curve.

**F5. The cost levers, measured (Appendix A.1).** Field evaluation is 7–15 µs
per point batched and 34–41 µs per scalar call; the scan step
\(2\pi/(24\,\max|m\iota-n|)\) gives 447–460 samples per period on these files
(≈1125 on the 3200-mode W7-X file). Reducing the scan density is safe but buys
only 1.1–1.4×; amplitude-aware mode truncation is *not* physics-preserving
(wrong topology within \(4\times10^{-3}B_{00}\) of a transition at threshold
\(10^{-4}\)). What is measured to work: (i) 60–80% of a trace is the adaptive
scalar quadrature; a batched composite Gauss–Legendre rule in the tracer's
sine-squared coordinate with breakpoints at the scanned extrema gives \(A\) to
\(10^{-15}\) and \(K\) to \(10^{-9}\) in 1–5 ms instead of 30–134 ms; (ii) a
per-surface quintic tensor spline of \(B(\theta,\zeta)\) at fixed \(s\)
(256×128 per period, built in 0.4–0.7 s) reproduces \(B\) to \(4\times10^{-10}\)
and \(D_\parallel B\) to \(4\times10^{-7}\) relative, with identical itineraries
on every test well including ones \(10^{-7}B_{00}\) from a transition, at
0.4 µs per point (35–70× cheaper per point). Realistic per-trace gain 5–10×
for sub-period wells and 10–20× for multi-period wells; the remaining lever
for long wells is scanning each field line once for all wells on it.

## 3. Options

Each option states what it changes in `docs/DESIGN.md`, how it handles
transitions and events, how unresolved measure enters the bounds, its cost,
strengths, weaknesses, failure modes on the real equilibria, a likelihood, an
effort estimate, and the cheapest experiment that would falsify it.

### Option A — Finish milestone 10.3 as specified (stay the course)

*What it would take.* Junction complexes as first-class §5.4 structures
(crossing \(\Gamma_{\max}\) curves, \(T\)-junctions where one companion curve
terminates on another), event certifications beyond equal-height and fold,
component-topology reconciliation near boundary exits, a faster tracer, and a
coordinator that stops re-tracing (the cap escalation alone is 40–54 minutes
per `max_periods` case). Accepting ADRs 0007 and 0008.

*Transitions and events.* As today: certified geometry per event, cut per arc.

*Bounds.* As today: unresolved hyperedges are retained but no bound on
\(f\) exists until milestones 12–14.

*Cost.* Unchanged in structure; the certification multiplier (20–320 traces per
bracket, source budgets to full, caps to 1024) is the cost. Even with a 10×
faster tracer the 46 wall-budget cases would mostly remain over 10 minutes.

*Strengths.* No redesign; the numerics that exist are excellent where they
apply (additivity \(7.8\times10^{-5}\) of tolerance; port actions exact on the
mesh); the synthetic six-sheet arrangement cuts on six backgrounds.

*Weaknesses and failure modes.* The 20 `max_periods` cases (16.7%) cannot be
counted as resolved under the current definition, so 95% is unreachable
regardless of geometry work. Each new certification kind has moved 1–5 cases.
PL insertion has not converged under refinement on any real case with
interacting transitions; no real cut with more than one transition curve
exists. The extractor/backend axes will keep disagreeing near the extrema.

*Likelihood of the goal.* ≈10% (≈30% if "resolved" is redefined to count
physical terminals, per ADR 0009 option 2, and the 10-minute cap is waived).

*Effort.* Open-ended; ADR 0009 calls the residual "research-scale".

*Decisive experiment.* Already run: the 10.3 matrix.

### Option B — Local, mesh-aligned cut on the existing surface mesh

*Idea.* Keep milestones 3–8 (background, extraction, \(g=0\) split, tracer,
refinement indicators, \(\Gamma_{\max}\) boundary) and replace milestones
9–10.3 by local detection: every surface vertex already has a forward trace
(milestone 7), so an edge whose endpoints have different lifted exits (a jump
in \(\zeta_{\rm out}\) or in the well count on the line) crosses \(T\); locate
the crossing by bisection *along the edge* (points projected to \(B=b\)); at
the crossing the forward trace gives \(m\), \(A_W\), \(A_1\), \(A_3\). Split
each straddling triangle along the chord between its two edge crossings (a
marching-triangles operation on a continuous indicator, not a constrained
insertion); triangles with more than one crossing per edge, three distinct
itineraries, or a crossing whose \(m\) points sit on different
\(\Gamma_{\max}\) components are event triangles: leave them unresolved and
refine. Child-3 ports are the \(m\) points, inserted as boundary vertices on
the \(\Gamma_{\max}\) boundary polyline (a boundary edge split, not an
interior insertion), with actions from their own traces.

*What changes.* §10.2–10.5 replaced; §5.4 events become bounded cells;
ADRs 0003–0008 retired; milestones 9, 10, 10.1–10.3 replaced by one
"local transition detection and cut" milestone; 11–16 unchanged in design.

*Transitions and events.* Generic crossings: exact local hyperedge (edge
crossing ↔ \(m\) on the boundary). Multiway, degenerate endpoints, thin
strips: bounded cells (F4). Seam: the existing periodic mesh adjacency.

*Bounds.* Unresolved triangles and capped vertices carry \(\Theta\in[0,1]\);
the \(K\)-weight of capped vertices is bounded through F3.

*Cost.* One forward trace per vertex (already required) plus a few traces
per straddling edge; no backward traces, no curve certification. DMercFail
\(\lambda_n=0.8\): ~15 s of tracing for 1182 vertices today, 2–3 s with the
batched quadrature.

*Strengths.* Smallest change; reuses the exhaustively tested locator,
measure, persistence; removes failure mechanisms M1–M3, M5–M9, M12 of the
knowledge map.

*Weaknesses and failure modes.* Keeps 3-D extraction and its resolution
problems (thin tubes near \(B_{\min}\), G_JUMP splits, backend disagreement,
sliver-controlled allowances, folded boundary triangles); needs the
\(\Gamma_{\max}\) boundary polyline to be resolved well enough to receive
child-3 ports (projection tolerance is a new control); the edge indicator can
miss two crossings on one edge (detected by the midpoint interpolation error,
otherwise a resolution gap); the \(K|\omega|\) quadrature on triangles stays.

*Likelihood.* ≈50–60% with E and F.

*Effort.* One large milestone (detection + local split + ports) plus
adaptation of the flood fill's port model; roughly the size of milestone 10.

*Decisive experiment.* On DMercFail \(\lambda_n=0.8\) with surface-wide
traces: do the edge crossings reproduce the two-sheet graph and the port
actions of the reference cut, and does the W7-X \(b=2.7781394\) surface show
the two events as isolated event triangles whose measure shrinks under local
refinement?

### Option C — Field-line chart with per-line well lists (recommended)

*Idea.* Replace the 3-D surface by the chart of F2. For each pitch \(b\), lay
a grid in \((s,\alpha)\) (rows of constant \(s\); \(\alpha\) uniform on
\([0,2\pi)\)); on each line scan \(B\) along \(\theta=\alpha+\iota(s)\zeta\)
over an extended window \([-L,2L)\), find every incoming crossing, and trace
every well whose incoming point lies in \([0,L)\). Each grid point carries a
finite list of wells with \(\zeta_-\), \(A\), \(K\), exit, extrema, status.
Match wells between neighbouring grid points by \(\zeta_-\) continuity (ghost
wells outside \([0,L)\) recognise a well drifting through the seam) and by
continuity of \(A\) and of the exit. Classify each cell edge: REGULAR (all
matched, \(A\) continuous), COUNT_CHANGE (one well appears: a
\(\Gamma_{\max}\) split or a \(\Gamma_{\min}\) birth, told apart by the
located point's \(D_\parallel^2B\)), A_JUMP (matched incoming point, exit
jumps: the parent/child-1 side of \(T\)), IRREGULAR (anything else). Locate
crossings by bisection along the edge; build the sheet complex as
cells × wells with the transition hyperedge linking the A_JUMP crossing at
\((s^*,\alpha_a)\) to the COUNT_CHANGE crossing of the child-3 well, which
lives on the line \(\alpha_a-k\iota L\) (the forward trace gives \(m\) and
hence \(k\)). Refine cells touching IRREGULAR edges (quadtree in the chart)
until they are regular or their measure is below tolerance; leave the rest as
bounded cells. The flood fill (§11) runs on the cell complex with
\(\omega\equiv1\); the seam is an identity-action link between the clipped
\(\zeta_-=L\) and \(\zeta_-=0\) edges of the sheet, found by the same
matching.

*What changes.* Decision 1 of §26 reversed by ADR; §4.4, §8 and §10 replaced;
milestones 3, 4, 5, 7 (as surface refinement), 8, 9, 10.x retired; the tracer
(6) becomes a per-line scanner that reuses one scan for all wells on the
line; 11–16 kept in design with triangles → chart cells; the validation
matrix loses the backend × extractor axes (5 files × 6 levels = 30 cases, or
more levels) and gains grid-refinement convergence.

*Transitions and events.* Generic: exact local hyperedges from bisection,
with additivity checked at the crossing (F1). Equal-height contacts,
\(T\)-junctions, degenerate endpoints: cells that stay IRREGULAR under
refinement, bounded (F4). Thin strips near `EDGE`: one-dimensional refinement
in \(s\) (T ends at \(s=1\) exactly because field lines preserve \(s\)).
Thin tubes near \(B_{\min}\): short \(\zeta\)-intervals on a line, no
triangle to bridge sheets. Axis: rows at small \(s\) where wells are nearly
\(\alpha\)-independent; the first row's measure \(s_1\) is bounded if not
resolved. Seam: identity links (above).

*Bounds.* Per slice, \(Q_{\rm lower}\) and \(Q_{\rm upper}\) from the
finite-atom flood fill on regular cells; IRREGULAR cells, capped wells, failed
traces and the axis row enter with \(\Theta\in[0,1]\) and their \(K\)-weight
bounded by F3 (exact measure \(ds\,d\alpha\) is known for every cell;
\(K\) need not be).

*Cost (measured components, Appendix A.2, A.1).* DMercFail \(\lambda_n=0.8\),
24×64 chart with the *current* tracer: 142 s for 1536 lines (2.3 M field
evaluations), all traces regular, four crossings bisected in 4–7 s each.
TURBO \(\lambda_n=0.5\), 24×64: 99 s. With the batched quadrature and a
per-row spline (built once per \(s\) row, 0.4–0.7 s), the per-line cost falls
from ~50 ms to ~5–10 ms; a 1024-period scan of one line at 230 samples per
period on the spline is ~0.1 s instead of ~25 s, and it serves every well on
the line. Projected: 5–20 s per slice at 32×96 for the easy levels, 1–3 min
for the long-well levels; the outer integral over 20–40 slices then fits in
hours on a laptop, in parallel over slices.

*Strengths.* Removes every geometric failure mechanism in the knowledge map
(M1–M12) and the extraction mechanisms (M10, M11); exact measure; transitions
local and self-certifying by additivity; long wells scanned once; the
diagnostic is the physicist's familiar \(J(\psi,\alpha)\) map per well class;
grid refinement is a clean §21.3 convergence dimension; the K-singularity at
\(\Gamma_{\max}\) is on a known curve (the count-change curve) for graded
quadrature.

*Weaknesses and failure modes.* The twisted seam is real bookkeeping (ghost
wells, identity links, child-3 at a shifted label) and must be tested with a
mutation; dense arrangements of count-change curves at high \(\lambda_n\)
(many maxima near \(b\)) need deep local refinement and may leave a larger
bounded measure there; root-scan aliasing of shallow extrema persists
(M13) as §21.3 dimension 5; near-tangent exits trigger `QUADRATURE_FAILURE`
in the current tracer (fix identified, A.1); the α-grid is uniform in a
coordinate whose natural scale varies (adaptive rows/columns mitigate); the
\(K\)-weighted quadrature near \(\Gamma_{\min}\) (\(A\to0\), \(K\) finite) and
\(\Gamma_{\max}\) (\(K\to\infty\)) needs §12.4's graded rules on chart cells;
milestones 11–16 must be re-based on cells and the seam links.

*Likelihood.* ≈75–85% with E and F; the main residual risk is the cost of the
nearly-passing levels (\(\lambda_n\ge0.9\)), which E bounds if not resolved.

*Effort.* Three milestones: (C1) per-line scanner and chart builder with
sheets, edges, crossings, seam links and diagnostics; (C2) cell-complex flood
fill with bounds and the total-weight check (re-based 12–14); (C3) slice
pipeline, outer quadrature, validation matrix (re-based 15–17). C1 is a
week-scale prototype away from a measured answer on all five files.

*Decisive experiment.* Run the chart on all five files at the six levels at
32×96 with the current tracer (cap 128) and report per case: fraction of
IRREGULAR measure after two refinement rounds, additivity residuals at every
crossing, capped measure, and wall time. The option is falsified if the
IRREGULAR measure does not shrink under refinement on the resolved reference
cases, or if seam matching produces spurious count changes.

### Option D — Direct contour following in the continuous field

*Idea.* Do not build a global structure. Sample trapped states in the chart
with weight \(hK\,ds\,d\alpha\) (or uniformly, weighting by \(hK\)); from each,
follow the constant-\(A\) contour as an ODE in \((s,\alpha)\) using the
analytic gradient of \(A\) (bounce integrals of \(\partial_sB\),
\(\partial_\theta B\) with the same endpoint regularization as \(K\)); detect
edge arrival, closure, and transitions (the well's highest interior maximum
reaching \(b\), or the exit becoming tangent) along the way; branch to
child-1 and child-3 (or the parent) using the forward trace at the crossing;
memoize by (well identity, \(A\)). \(\Theta\) per sample; \(Q(b)\) by
stratified Monte Carlo with a statistical error, or by contour bands
(reachability is constant along a contour, so one trace classifies a curve).

*What changes.* Decision 3 of §26 reversed for production; milestones 8–14
replaced by a sampler, a contour integrator, and a variance-reduced estimator;
11 (direct tracer) becomes the production path.

*Transitions and events.* Handled when encountered, locally, by the same
forward-trace rule; nongeneric events are measure-zero for the sampler and
appear as rare branching ambiguities that are reported per sample.

*Bounds.* Failed samples are unresolved measure; the estimator's statistical
error is separate from the bound; F3 bounds capped wells.

*Cost.* One trace per right-hand-side evaluation, ~4 per RK step; contours
on a stellarator sheet need hundreds of steps → ~10³ traces per contour,
1–20 s per contour with the measured levers; \(10^3\)–\(10^4\) contours per
slice for 1% → hours per slice on one core. [Appendix A.4 will report the
measured contour cost when experiment E4 completes.]

*Strengths.* Nothing global to certify; every failure is local and honest;
trivially parallel; the ideal oracle for options B and C and the natural
estimator inside cells they leave unresolved.

*Weaknesses and failure modes.* Cost and statistical error; contour closure
detection under drift; branching trees at dense transition arrangements;
the same long-well cost.

*Likelihood.* ≈40% as the sole production estimator; high value as oracle.

*Effort.* One milestone for the integrator and oracle; one more for the
production sampler if chosen.

*Decisive experiment.* Follow three contours on DMercFail \(\lambda_n=0.8\):
closure, edge arrival, one transition crossing with additivity, and wall time.

### Option E — Bounds-first computation (cross-cutting; recommended with B, C or D)

*Idea.* Make the deliverable of every slice the pair
\((Q_{\rm lower}(b),Q_{\rm upper}(b))\), computed from the finite-atom flood
fill's inner/outer sets on resolved cells and from F3 for everything else, and
of the whole run the pair \((f_{\rm lower},f_{\rm upper})\). Spend refinement
effort where the gap is largest (a cell's gap is its \(hK\,ds\,d\alpha\)
measure if unresolved, plus the action band of contours through it). Stop at
the time budget and report the gap.

*What changes.* §23 milestone 10.3's acceptance ("≥95% resolved") becomes
"≥95% of cases reach a gap below tolerance within the budget; every case
reports its gap"; §11.3/§12.5/§13.4 unchanged; §9.3 satisfied literally.

*Why it is not a loosening.* §21.2 forbids *silent* loss; a bound that
contains the truth by construction is the opposite. ADR 0009's option 2
("exclude physical terminals from the denominator") becomes unnecessary: a
capped well is inside the bound.

*What accuracy is useful.* For optimization use, \(f\) to ±0.01 absolute is
likely sufficient; the nearly-passing band above \(\lambda_n=0.9\) carries
[Appendix A.5: fraction of trapped phase space, to be filled from experiment
E5] and could be left bounded rather than resolved if it is expensive.

*Failure mode.* A large region whose only route to the edge passes through
an unresolved cell inflates the gap; refinement of that cell (not the region)
fixes it, which is why the refinement criterion must be the gap contribution,
not the cell's own measure.

### Option F — Cost engineering (cross-cutting; measured)

In order of measured payoff: (1) replace the adaptive scalar quadrature in
`trace_regular_well` by a batched composite Gauss–Legendre rule in the
existing sine-squared coordinate with breakpoints at the scanned extrema,
64 nodes per segment, 128–256 on segments whose maximum is within
\(10^{-5}B_{00}\) of \(b\), with an \(n\)-vs-\(2n\) error estimate and the
existing endpoint-difference evaluation (30–134 ms → 1–5 ms); (2) cache the
coefficient vector per \(s\) and evaluate \(B\) and \(D_\parallel B\) jointly
(scalar calls 34–41 → 7–9 µs); (3) per-surface quintic spline for the scan
(0.4 µs per point; physics-preserving; bicubic is not); (4) one scan per field
line reused by every well on it (option C gives this for free; option B can
cache by line); (5) vectorize the per-period crossing/extremum bracketing;
(6) Numba last, after profiling. Do *not* truncate the mode set by amplitude.
Also fix the near-tangent-exit `QUADRATURE_FAILURE` (0.5–0.9 s each, at
default settings) by decoupling the endpoint window from the scan step.

### Option G — Physics shortcuts for the nearly-passing band

For \(b\) close to \(B_{\max}(s)\) the wells are ergodically long (about
\(1/p\) periods where \(p\) is the area fraction of \(B>b\) on the surface).
Two honest treatments: (i) bound the whole band \([b^*,B_{\max}]\) by F3's
volume integral (exact, cheap) and report it as unresolved measure; (ii) trace
with the per-line scan and spline, where a 1024-period line costs ~0.1 s. The
band's share of trapped phase space is [Appendix A.5]. A third idea, using
the ergodic average of \(J\) (which makes contours nearly flux-surface-aligned
and hence confined) is *not* admissible as a computation of \(\Theta\) but
is a good diagnostic of what the bound is hiding. Symmetry can halve the
chart (stellarator symmetry maps \((\theta,\zeta)\to(-\theta,-\zeta)\)); use
only after the unsymmetrized code is validated.

## 4. Comparison

| Option | Removes M1–M9, M12 (curve/cut) | Removes M10–M11 (extraction) | Long wells | Bound on \(f\) | Effort | Likelihood of goal |
|---|---|---|---|---|---|---|
| A stay the course | no | no | veto | none until M12–14 | open-ended | ≈10% |
| B local cut on mesh | yes | no | bounded (E) | yes (E) | 1 large milestone | ≈50–60% |
| C field-line chart | yes | yes | one scan per line; bounded (E) | yes (E), exact measure | 3 milestones | ≈75–85% |
| D contour following | n/a | yes | bounded (E) | statistical + bound | 1–2 milestones | ≈40% alone; oracle for B/C |
| E bounds-first | — | — | — | required | small, cross-cutting | prerequisite |
| F cost levers | — | — | 10–20× per trace | — | 1 milestone | prerequisite for 10 min |
| G band shortcuts | — | — | bound or cheap scan | — | small | supporting |

## 5. Recommended path and acceptance criteria

1. **ADR: bounds-first acceptance (E).** Restate milestone 10.3's threshold as
   a bound gap: for each (equilibrium, \(b\)) case,
   \(Q_{\rm upper}-Q_{\rm lower}\le\varepsilon\,Q_{\rm total}(b)\) within 10
   minutes, with \(\varepsilon\) set by the researcher (I suggest 0.02), and
   \(f_{\rm upper}-f_{\rm lower}\le0.01\) for the outer integral. Decide ADRs
   0007/0008 as moot under the new design and ADR 0009 by this restatement.
2. **Milestone F0: tracer cost levers** (batched quadrature, coefficient
   cache, joint \(B\)/\(D_\parallel B\), near-tangent-exit fix), validated by
   bit-level agreement with the current tracer on the milestone-9 sample set.
3. **Milestone C1: chart builder** (per-line scanner, well lists, matching
   with ghosts, edge classes, crossing bisection with additivity, seam links,
   count-map and \(J\)-map diagnostics), validated on the DMercFail
   \(\lambda_n=0.8\) two-sheet reference, the W7-X \(b=2.7781394\) two-event
   reference, and the ADR 0006 six-sheet synthetic field.
4. **Milestone C2: cell-complex flood fill with bounds** (re-based 12–14)
   with F3 as an exact check of the total weight on every slice.
5. **Milestone C3: slice pipeline and outer integral** (re-based 15–17), with
   the direct contour tracer (D) as the oracle on sampled points.
6. Keep option B as the fallback if C1's decisive experiment fails on the
   seam or on dense arrangements; it shares F0, E and the flood-fill re-base.

Early experiments in order (each under an hour): C1's decisive experiment on
all five files; the F3 identity on two files (Appendix A.3); the band
fractions (A.5); D's three contours (A.4).

Open questions for the researcher: the tolerance \(\varepsilon\) and the
\(f\) tolerance; whether the validation matrix should drop the backend and
extractor axes in favour of grid-refinement levels; whether the axis row is
bounded or resolved; disposition of ADRs 0007–0009.

## Appendix A. Measurements (2026-09-02/03)

### A.1 Trace cost levers (experiment E1)

Two equilibria (d23p4 \(\lambda_n=0.5\), DMercFail \(\lambda_n=0.8\)), 30
regular wells each plus 36 constructed near-transition wells, profiled and
re-timed single-process. Field evaluation 7–15 µs per point batched, 34–41 µs
per scalar call; coefficient-spline re-evaluation 28–30% of field time.
Scan density 447–460 samples per period. Per trace: 31–40 ms (sub-period),
78 ms (1 period), 121 ms (2 periods); 60–80% in the adaptive scalar
quadrature. Levers: samples per wavelength 24→8 safe, 1.1–1.4×; mode
truncation at \(10^{-4}\) changes \(A\) by up to \(8\times10^{-4}\), \(K\) by
\(2\times10^{-2}\), itineraries on 1–2 of 30 wells and gives wrong topology
near transitions (rejected); quintic per-surface spline 256×128: \(B\) to
\(4\times10^{-10}\), \(D_\parallel B\) to \(4\times10^{-7}\), identical
itineraries on all tests, 0.4 µs per point; batched composite Gauss–Legendre
with breakpoints at extrema: \(n=64\) gives \(A\) to \(10^{-15}\), \(K\) to
\(10^{-9}\) in 1–5 ms (near-transition segments need 128–256 nodes). Combined
realistic gain 5–10× (sub-period) to 10–20× (multi-period). Eight of 36
near-tangent-exit wells return `QUADRATURE_FAILURE` at default settings.

### A.2 Field-line chart prototype (experiment E2, first run)

DMercFail \(\lambda_n=0.8\), \(b=10.6483680108\), grid 24×64 in
\((s\in[0.05,1],\alpha)\), extended window \([-L,2L)\), cap 128, current
tracer: 142 s wall, 2.27 M field evaluations, 1536 lines, all traces REGULAR;
102 of 1536 lines have an interior maximum, minimum gap \(b-B_{\max,\rm int}=
1.1\times10^{-3}\). Count map: one well per line except a region of two wells
(the child-3 sheet) bounded by COUNT_CHANGE edges and a region where the
parent/child-1 sheet shows A_JUMP edges — the same transition seen at two
chart labels separated by the seam shift. Four count-change edges bisected
(4–7 s each) to bracket width \(10^{-9}\): every located point has
\(B-b=0\), \(|D_\parallel B|\le6\times10^{-15}\), \(D_\parallel^2B\approx-63.9\)
(a \(\Gamma_{\max}\) point). Additivity \(|A_W-A_1-A_3|/A_W\) at offsets
\(10^{-1},10^{-2},10^{-3},10^{-4},10^{-5}\) of the edge: \(3.4\times10^{-4}\),
\(4.2\times10^{-5}\), \(5.0\times10^{-6}\), (quadrature failure),
\(6.6\times10^{-8}\) on one edge; \(1.5\times10^{-4}\ldots2.5\times10^{-7}\) on
another. TURBO \(\lambda_n=0.5\), 24×64: 99 s, all regular, zero
non-regular edges at this grid; with rows down to \(s=0.002\) (16×64), 12
COUNT_CHANGE and 3 IRREGULAR edges near the axis, bisecting to
\(\Gamma_{\min}\) points (\(D_\parallel^2B>0\), well births with \(A\to0\)),
measure fraction 1.6% before refinement. [The second run of E2 and the
hard-case run E2b will be added here.]

### A.3 Total-weight identity (experiment E3)

[Pending: \(2b^2\,dV/db\) versus the chart-side sum of \(hK\) over all wells,
and \(V(B_{\max})/V_h\) versus the trapped fraction, on d23p4 and DMercFail.]

### A.4 Contour following (experiment E4)

[Pending: three contours on DMercFail \(\lambda_n=0.8\).]

### A.5 Long-well measure (experiment E5)

[Pending: capped-well fraction of \(ds\,d\alpha\) and of trapped phase space at
\(\lambda_n\in\{0.5,0.8,0.9,0.95\}\) for d23p4 and DMercFail; ergodic estimate.]

### A.6 Matrix statistics computed from `milestone10.3-real-equilibria.json`

50 of 120 cases finish under 600 s: all 32 no-transition cases, 9 of 10
resolved, 9 unresolved with quick terminals (event geometry, empty interval,
cut conflict, non-separating slit). 68 of the 69 slower cases are unresolved.
Every case with a transition at \(\lambda_n\ge0.8\) is unresolved; no case
at \(\lambda_n\in\{0.9,0.95\}\) finishes under 600 s. Resolved cases: TURBO
0.5 (3–110 s), DMercFail 0.05 (259–261 s), DMercFail 0.8 (30–71 s), d23p4 0.5
gmsh:PV (5490 s). Remediation attempts: cap escalation 40, background 25,
thin-strip 15, source budget 13, local projection 8, scan resolution 4.
