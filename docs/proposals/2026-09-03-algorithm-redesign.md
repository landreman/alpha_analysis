# Proposal: ways forward for computing \(f\) on realistic equilibria

- **Status:** Draft for the researcher's decision (no code changed)
- **Date:** 2026-09-03
- **Scope:** replaces or reorders `docs/DESIGN.md` milestones 3–14 depending on the option chosen; the metric of §3 and the rules of §21.2 are untouched
- **Evidence:** `docs/validation/*.md`, `docs/adr/0001–0009`, the recorded matrices, and the measurements of Appendix A (run 2026-09-02/03 on the five `data/boozmn_*` files, scripts retained under the session scratchpad)

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

Five facts change the problem (§2), all checked numerically this week:

1. transitions are visible in the *forward* trace of every well, and more
   sharply, the transition set at every pitch \(b\) is a level curve of smooth,
   \(b\)-independent "extremum-height" fields on the field-line chart;
2. the incoming-bounce surface projects diffeomorphically onto the field-line
   chart \((s,\alpha)\), where the phase-space measure is exactly
   \(ds\,d\alpha\);
3. the total \(K\)-weighted measure of the whole surface is a plain volume
   integral (restricted to surfaces whose maximum exceeds \(b\)), so every
   capped, failed or uncut well can be bounded without being traced;
4. codimension-two events carry no connectivity of their own, so they can be
   bounded instead of certified;
5. wells longer than 12 field periods carry at most 0.3% of the trapped phase
   space on the two files checked, so the whole `max_periods` class that
   blocks 16.7% of the matrix protects about 0.003 of \(f\).

Options are laid out in §3. The recommendation (§5) is:

1. **Adopt the bounds-first framing now (option E),** with the acceptance
   stated in units of \(f\): the deterministic gap
   \(f_{\rm upper}-f_{\rm lower}\le0.01\) per equilibrium within a
   per-equilibrium time budget, every slice reporting its gap. Nothing in
   §21.2 is loosened; this is what §11.3, §12.5 and §13.4 already ask for.
2. **Rebuild milestones 3–10 on the field-line chart (option C),** with the
   extremum-height fields as the detection core, orbit sampling of \(\alpha\)
   so that one long scan serves many chart lines and the periodic seam is an
   identity, and event neighbourhoods left as bounded cells.
3. **Build reachability as a Reeb complex on the chart (option R)** instead of
   the finite-atom flood fill: exact for the PL interpolant, with unresolved
   cells as explicit wildcard nodes and the quadrature by the coarea formula.
4. **Keep direct contour following (option D)** as the independent oracle, as
   the gap estimator inside cells the chart leaves unresolved (a Monte Carlo
   whose cost falls with the square of the gap), and, in its birth-particle
   form, as the end-to-end validation of the whole calculation.
5. **Do not finish milestone 10.3 as specified (option A).**

Likelihoods for the researcher's goal (a useful bound on \(f\) for > 95% of
cases within budget): A ≈ 10%; B ≈ 50–60%; C with its refinements ≈ 85%
conditional on E and F; D alone ≈ 45–55% (deterministic bounds are not
available from sampling alone); without E no option exceeds ≈ 40%.

## 1. Why the current algorithm cannot reach 95%

The pipeline of `docs/DESIGN.md` §§8–10 represents the state space and the
transition set as follows, and every recorded failure sits in steps 2–5:

1. extract \(\Sigma_b^-\) as a PL surface from a tetrahedral background and
   split it by \(g=0\) (milestone 5, ADR 0001);
2. extract \(\Gamma_{\max}\) as chains of marching-triangle edges and classify
   by \(D_\parallel^2 B\) (milestone 8; ADR 0005 zigzags; ADR 0008 endpoints);
3. sample \(\Gamma_{\max}\) at 8–16 vertices, trace backward from each
   marginal point to the companion point \(a(u)\) (milestones 9, 10.1);
4. bracket interior-maximum count changes between samples, bisect along
   \(u\), and certify each as a contact, a fold or a degenerate endpoint before
   any incident arc may cut (ADRs 0003, 0007, 0008; milestones 10.2–10.3);
5. insert \(T\) as a constrained polyline, duplicate vertices, assign sides,
   union-find sheets (milestone 10; ADR 0004 endpoint snap; ADR 0006 corridors).

What the matrix and the ADRs established (citations in the knowledge map that
accompanies this proposal and in `docs/validation/milestone10.3-real-equilibria.md`):

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
  matrix. (Forward traces from incoming points at the same levels never hit a
  cap of 64 in this week's tests — 79 wells on d23p4 \(\lambda_n\ge0.9\) and
  DMercFail \(\lambda_n=0.9\), longest 15 periods, A.5 — so the binding trace
  was the *backward* scan from a marginal point.)
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
- **The matrix weights cases equally, but \(f\) does not.** The levels that
  always succeed (\(\lambda_n\le0.1\)) carry 0.2–19% of the trapped phase
  space; \(\lambda_n\in[0.1,0.8]\), where almost nothing resolves, carries
  72–92% (Appendix A.5).

The conclusion of ADR 0009 is right: no coordinator that keeps §21.2 can reach
95% *resolved cuts* on these equilibria. The way out is to stop requiring a
certified global cut before any weight can be counted.

## 2. Facts that reframe the problem

**F1. Transitions are visible from the forward trace of the well itself,
and they are level curves of smooth \(b\)-independent fields.** The parent
well \([a,d]\) contains the marginal maximum \(m\) as an interior maximum
whose height reaches \(b\); on the child side the first maximum past the
exit descends to \(b\), and the count of wells on the field line changes by
one (the child-3 well \([m,d]\) appears). `WellTrace` already records the
extrema heights. So \(T\) can be located by one-dimensional root finding along
any curve in the state space, and the forward trace at the located point
gives \(m\), \(A_W\), \(A_1\), \(A_3\) and the additivity check. Measured: on
DMercFail \(\lambda_n=0.8\) five count-change edges of a 24×64 chart bisect to
points with \(B-b=0\), \(|D_\parallel B|\le6\times10^{-15}\),
\(D_\parallel^2B=-63.9\), and the additivity residual falls linearly with the
offset from the crossing, \(3\times10^{-4}\) at \(10^{-1}\) to
\(7\times10^{-8}\) at \(10^{-5}\) (A.2).

The sharper statement: the along-line extrema of \(B\) do not depend on
\(b\). On the chart, the height of the \(j\)-th along-line maximum,
\(B_j(s,\alpha)\), is a smooth function with the envelope-theorem gradient
\(\partial_\alpha B_j=\partial_\theta B|_{\max}\),
\(\partial_sB_j=\partial_sB|_{\max}+\iota'(s)\zeta_j\,\partial_\theta B|_{\max}\)
(checked to six digits against finite differences). At pitch \(b\), the
transition curve and \(\Gamma_{\max}\) together are the level set
\(\bigcup_j\{B_j=b\}\); the child-3 copy of the curve is its rigid shift by
\(k\,\iota(s)L\) in \(\alpha\) (the deck transformation of the one-period
quotient, identity checked to \(4\times10^{-15}\)). Checked on the crossing
E2 had bisected: Newton on \(B_j(s,0)=b\) gives \(s^*=0.6196449011556\) in
2 ms against the bisected \(0.6196449011557605\) in 4.2 s, and the level
curve's width in \(\alpha\) reproduces the chart's count-0 hole and, after
the shift, the parent band (A.2). Below-\(b\) folds (ADR 0007) are where a
field's domain ends; degenerate endpoints (ADR 0008) are where a level curve
meets that boundary (\(A_3\to0\)); equal-height contacts (ADR 0003) are
intersections of two level curves; all are codimension-two points of one
arrangement computed once per equilibrium, not per pitch and not per curve.

**F2. The incoming surface projects diffeomorphically along field lines.**
On \(\Sigma_b^-\), \(g=\mathbf b\cdot\nabla B<0\), so the surface is nowhere
tangent to the field; the map \(\Sigma_b^-\to(s,\alpha)\),
\(\alpha=\theta_--\iota(s)\zeta_-\), is a local diffeomorphism on the open
incoming half, singular only on the boundary curves \(g=0\), which are the
physical events. Each trapping sheet is an honest graph \(\zeta_-(s,\alpha)\)
with boundary, and \(\omega\) pulled back to the surface is exactly
\(ds\,d\alpha\) (§4.3): the measure of a chart cell is its area times the
number of wells over it. (The folds of ADR 0006 are in the *extracted
triangles* near \(g\to0\); they are an artifact of meshing the surface in 3-D
first.) The one-period quotient identifies the well with incoming point
\((\alpha,\zeta_-=L)\) with \((\alpha+\iota L,0)\): a twisted seam. The
prototype handled it with an extended scan window and ghost wells; sampling
\(\alpha\) by orbits of the shift \(\alpha\to\alpha+\iota L\) makes the seam an
identity between consecutive windows of one scan (option C).

**F3. The total \(K\)-weighted surface integral is a volume integral.**
\[
Q_{\rm total}(b):=\int_{\Sigma_b^-}hK\,|ds\,d\alpha|
=\int_{\{s:\,B_{\max}(s)>b\}}ds\int d\theta\,d\zeta\,
\frac{|C|}{B}\,\frac{h(\sqrt s)}{\sqrt{1-B/b}}\,[B<b],
\]
because at fixed \(s\) the map \((\theta,\zeta)\to(\alpha,\zeta)\) has unit
Jacobian and, on a dense field line, the line integral of
\((|C|/B)/\sqrt{1-B/b}\) over all \(B<b\) intervals is the sum of \(K\) over
all wells on the line. The restriction to surfaces with \(B_{\max}(s)>b\) is
essential: a particle with bounce field above its own surface's maximum never
bounces. Verified three ways (A.3): two volume-side routes agree to
0.01–0.13% at all levels; the surface-side sum of traced \(K\) over every
incoming point agrees with the volume side to 0.02–0.05% (DMercFail
\(\lambda_n=0.8\), 1128 wells) and converges as \(O(1/n)\) elsewhere; and
\((1/2V_h)\int Q_{\rm total}/b^2\,db\) reproduces the trapped fraction
\(V_{\rm tr}(B_{\max})/V_h\) to \(1\)–\(4\times10^{-4}\). Consequence: for any
set of wells that failed, hit the period cap, or sit in unresolved cells, the
\(K\)-weighted measure is bounded *exactly* as \(Q_{\rm total}\) minus the
resolved part; the bound is as sharp as the quadrature error of the resolved
part, which therefore needs its own error estimate and a support-aligned
\(s\) integration. \(Q_{\rm total}(b)\) costs 2–3 s per level.

**F4. Codimension-two events carry no connectivity of their own.** Equal-
height contacts, degenerate endpoints and crossings of two \(T\) curves are
isolated points of the 2-D state space. Contours through a point have measure
zero; contours near it cross the incident curves at generic simple points,
which the local rule handles. An event cell can be left "unresolved with
bounds" and shrunk by refinement; certifying its *kind* is not needed for the
bounds to converge. Near a transition curve the divergence of \(K\) has the
analytic form \(K\simeq{\rm const}-(|C|/\sqrt b)\sqrt{2/|D_\parallel^2B|}\,
\ln|B_j-b|\) (checked against four traced parent wells: \(-5.518\) per
\(e\)-fold predicted, \(-5.520\) fitted), so the singular mass of a cell is
bounded in closed form in terms of the smooth field \(B_j-b\).

**F5. Where the mass is, and the cost levers.** Trapped fractions are 0.28–0.63
of all particles; the band above \(\lambda_n=0.9\) carries 1.0–2.7% of the
trapped mass and above 0.95, 0.3–0.7% (A.5). A birth-particle sampler that
draws from the source directly (A.6) shows that wells longer than 12 field
periods carry \(\le0.3\)% of the trapped mass on d23p4 and DMercFail, while
wells of 2–6 periods carry 25% of it on d23p4: long-well cost must be paid for
the mid band, and the very long tail can be bounded. Per trace, 60–80% of the
time is the adaptive scalar quadrature; a batched composite Gauss–Legendre
rule gives \(A\) to \(10^{-15}\) and \(K\) to \(10^{-9}\) in 1–5 ms instead of
30–134 ms; a per-surface quintic spline of \(B(\theta,\zeta)\) is
physics-preserving at 0.4 µs per point; amplitude-aware mode truncation is
not physics-preserving (A.1). Scanning \(B\) along a line does not depend on
\(b\), so one scan per line can serve every pitch slice.

## 3. Options

Each option states what it changes in `docs/DESIGN.md`, how it handles
transitions and events, how unresolved measure enters the bounds, cost,
strengths, weaknesses, failure modes on the real equilibria, a likelihood, an
effort estimate, and the cheapest experiment that would falsify it.

### Option A — Finish milestone 10.3 as specified (stay the course)

*What it would take.* Junction complexes as first-class §5.4 structures,
event certifications beyond equal-height and fold, component-topology
reconciliation near boundary exits, a faster tracer, a coordinator that stops
re-tracing, and accepting ADRs 0007 and 0008.

*Transitions and events.* As today: certified geometry per event, cut per arc.

*Bounds.* None on \(f\) until milestones 12–14; unresolved hyperedges retained.

*Cost.* The certification multiplier (20–320 traces per bracket, source
budgets to full, caps to 1024) is the cost; a 10× faster tracer leaves most of
the 46 wall-budget cases over 10 minutes.

*Strengths.* No redesign; excellent numerics where they apply (additivity
\(7.8\times10^{-5}\) of tolerance); the six-sheet synthetic case cuts.

*Weaknesses and failure modes.* The 20 `max_periods` cases cannot count as
resolved under the current definition, so 95% is unreachable regardless of
geometry work; each new certification kind moved 1–5 cases; no real cut with
interacting transitions exists; PL insertion has not converged on any real
case; the extractor and backend axes keep disagreeing near the extrema.

*Likelihood.* ≈10% (≈30% if "resolved" is redefined per ADR 0009 option 2 and
the 10-minute cap is waived). *Effort.* Open-ended. *Decisive experiment.*
Already run: the 10.3 matrix.

### Option B — Local, mesh-aligned cut on the existing surface mesh

*Idea.* Keep milestones 3–8 and replace 9–10.3 by local detection on the
surface mesh: every vertex has a forward trace, so an edge whose endpoints
have different lifted exits crosses \(T\); locate the crossing by bisection
along the edge (or, better, evaluate the extremum-height fields of F1 at the
vertices, which lie on field lines, and march the level set); split each
straddling triangle along the chord between its two edge crossings (a
marching-triangles operation, not a constrained insertion); leave triangles
with more than one crossing per edge or three distinct itineraries as
unresolved event triangles. Child-3 ports are the \(m\) points, inserted as
boundary vertices on the \(\Gamma_{\max}\) boundary polyline.

*What changes.* §10.2–10.5 replaced; §5.4 events become bounded cells;
ADRs 0003–0008 retired; milestones 9–10.3 replaced by one milestone; 11–16
unchanged in design.

*Bounds.* Unresolved triangles and capped vertices carry \(\Theta\in[0,1]\);
the \(K\)-weight of capped vertices is bounded through F3.

*Cost.* One forward trace per vertex plus a few per straddling edge; no
backward traces, no curve certification (DMercFail \(\lambda_n=0.8\): ~15 s of
tracing today, 2–3 s with the batched quadrature).

*Strengths.* Smallest change; reuses the tested locator, measure and
persistence; removes M1–M9 and M12 of the knowledge map.

*Weaknesses and failure modes.* Keeps 3-D extraction and its resolution
problems (thin tubes near \(B_{\min}\), G_JUMP splits, backend disagreement,
sliver-controlled allowances, folded boundary triangles); needs the
\(\Gamma_{\max}\) boundary polyline resolved well enough to receive child-3
ports; the edge indicator can miss two crossings on one edge; the
\(K|\omega|\) triangle quadrature stays; the extractor/backend matrix axes stay.
The computational-topology review recommends dropping B: its indicator is
strictly weaker than the level-curve predicate of F1 and it gains nothing
over C except code reuse.

*Likelihood.* ≈50–60% with E and F. *Effort.* One large milestone plus the
flood-fill port model. *Decisive experiment.* On DMercFail \(\lambda_n=0.8\)
with surface-wide traces, do the edge crossings reproduce the two-sheet graph
and the reference port actions, and does the W7-X \(b=2.7781394\) surface
show its two events as isolated event triangles whose measure shrinks?

### Option C — Field-line chart with per-line well lists (recommended)

*Idea.* Replace the 3-D surface by the chart of F2. Rows of constant \(s\);
on each row a per-surface quintic spline of \(B(\theta,\zeta)\) (0.4–0.7 s to
build). Sample \(\alpha\) by orbits of the period-shift map
\(\alpha\to\alpha+\iota(s)L\): one long scan of a physical field line over
\(N+M\) periods yields \(N\) chart lines each with \(M\) periods of coverage,
the seam between consecutive windows is an identity, and a well's exit is
read from the same scan however many windows away (near low-order rational
\(\iota\) the orbit clusters, measured at 6–16× the uniform gap on a few rows,
so such rows use several orbits or a uniform fallback). Put the window
boundary away from the stellarator-symmetry planes \(\zeta=0,L/2\): all
located \(\Gamma_{\max}\) points on DMercFail sat exactly on \(\zeta=0\).

Scan each line once, independently of \(b\): list the along-line extrema
(\(\zeta_k\), \(B_k\), kind, \(D_\parallel^2B\)), polish by Newton on
\(D_\parallel B=0\) bracketed between scanned neighbours, and attach the
Lipschitz certificate that no extremum hides between samples. Match extrema
between \(\alpha\)-neighbours by continuity of \((\zeta_k,B_k)\), with fold
detection where a maximum–minimum pair annihilates; this yields the labelled
height fields \(B_j(s,\alpha)\) and \(B_k^{\min}(s,\alpha)\) on domains
bounded by fold curves (about one along-line maximum per field period on
these files; a 103-maximum parent well is a ~100-period well).

Per pitch \(b\): marching squares on \(B_j-b\) over cell edges, root on an
edge by Newton on the smooth \(B_j\) (~1 ms). Classify each cell as REGULAR
if every field satisfies the certified Taylor band
\(|B_j({\rm centre})-b|>|\nabla B_j|\,r+Mr^2/2\) (an interval predicate, no
traces), GENERIC if exactly one level curve crosses transversally, EVENT if
two curves meet, a fold point lies inside, or a curve meets a fold; quadtree-
refine EVENT cells until their measure is below tolerance and leave the rest
bounded. Wells at level \(b\) on each line come from the line's merge tree
(sublevel-set components of \(B\) along the line, which merge at maxima as
\(b\) rises); the crossings are polished between the bracketing extrema and
\(A\), \(K\) are computed by the batched Gauss–Legendre rule with breakpoints
at the extrema. Sheets are unions of cells with a constant merge-tree branch
label; the child-3 copy of a transition is the shift by an integer number of
windows. The transition hyperedge links parent (\(B_j\) slightly below
\(b\)), child-1 (same line, exit before \(m\)) and child-3 (well starting
after \(m\), \(k\) windows downstream), with \(u\) = arc length along
\(\{B_j=b\}\) and additivity checked at the located point. The well count and
exit-displacement signals of the prototype remain as an independent check of
the level-curve predicate, not as the detector (the prototype showed the
count alone is insufficient and \(|\Delta A|/A\) is unusable near the axis).

*What changes.* Decision 1 of §26 reversed by ADR; §4.4, §8 and §10 replaced;
milestones 3, 4, 5, 7 (as surface refinement), 8, 9, 10.x retired; the tracer
(6) becomes a per-line scanner plus a batched quadrature; 11–14 replaced by
option R (or re-based on cells); 15–17 re-based; the validation matrix loses
the backend and extractor axes and gains chart-refinement levels.

*Transitions and events.* Generic crossings: exact, on a smooth function with
a certified gradient; the three ports are the same \((s,\alpha)\) point and
an integer shift, so nothing is matched by a common parameter. Folds,
degenerate endpoints, \(T\)-junctions and equal-height contacts are
codimension-two points of the arrangement (F1, F4), entering only as bounded
EVENT cells. Thin strips near `EDGE`: refinement in \(s\) (T ends at \(s=1\)
exactly). Thin tubes near \(B_{\min}\): short \(\zeta\)-intervals on a line.
Axis: rows at small \(s\) where wells are nearly \(\alpha\)-independent; the
first row's measure is bounded if not resolved. A symmetry-enforced event
curve (two level curves coinciding) is detected by the matching and refined
along the curve.

*Bounds.* Per slice, \(Q_{\rm lower}\) and \(Q_{\rm upper}\) from option R on
resolved cells; EVENT cells, capped wells, failed traces and the axis row
enter with \(\Theta\in[0,1]\), their measure known exactly and their
\(K\)-weight bounded by F3 globally and by the analytic log model of F4 per
cell.

*Cost (measured components in A.1–A.3; combination projected).* Scanning: one
orbit of \(96+128\) periods per row at 460 samples per period on the spline is
~40 ms; \(96+1024\) periods ~0.2 s; 32 rows ~7 s per equilibrium for full
1024-period coverage of every chart line, against ~10 minutes with
independent lines at the same spline speed and hours with the current tracer.
Height-field matching and level-curve extraction: milliseconds per slice.
\(A\), \(K\) for 3000–6000 wells at 1–5 ms: 5–30 s per slice; a 1024-period
well costs ~50 ms. With the *current* tracer the 24×64 prototype took 142 s
(DMercFail \(\lambda_n=0.8\)) and 99 s (TURBO \(\lambda_n=0.5\)) per slice, all
traces regular, five crossings bisected in 4–6 s each. Projected per
equilibrium at 30 slices: 3–15 minutes plus long-well quadrature.

*Strengths.* Removes every geometric failure mechanism in the knowledge map
(M1–M12) and the extraction mechanisms (M10, M11); exact measure; the
transition set is a certified predicate on smooth functions rather than a
tolerance on traced quantities; location cost ~\(10^3\) lower than trace
bisection; all event kinds are geometry computed once per equilibrium; the
seam is an identity; long wells are scanned once per line and once for all
\(b\); the diagnostic is the physicist's \(J(\psi,\alpha)\) map per well
class; chart refinement is a clean §21.3 dimension; the breakpoints of
\(F(b)=Q/b^2\) in \(b\) (critical values of the height fields and the
\(B_{\max}(s)\) crossings) are known in advance for the outer quadrature.

*Weaknesses and failure modes.* Extremum matching across \(\alpha\) is a new
labelling problem: shallow ripples give many fold curves, the 3200-mode W7-X
file has more maxima per period, and a label can retire through a below-\(b\)
fold without any physical event; this bookkeeping needs mutation tests. Newton
on \(D_\parallel B\) must be bracketed on dense ripples. Pairs below the scan
certificate (root-scan aliasing, M13) remain a §21.3 dimension; their effect
is bounded in measure. The near-tangent-exit `QUADRATURE_FAILURE` of the
current tracer (A.1) must be fixed before cut-vertex values are assigned. Rows
near low-order rationals need several orbits. Dense arrangements at high
\(\lambda_n\) still need deep local refinement, now along known smooth
curves. Milestones 11–17 must be re-based.

*Likelihood.* ≈85% with E and F (the prototype's own risks, seam bookkeeping
and dense arrangements, were reduced to integer shifts and refinement along
known curves by the level-curve formulation). The residual is the matching
machinery and the mid-band long wells.

*Effort.* Three milestones: (C1) scanner, height fields, level curves, cells,
sheets, hyperedges, diagnostics; (C2) reachability and bounds (option R) with
the total-weight check; (C3) slice pipeline, outer integral, validation
matrix. C1 is a week-scale prototype away from a measured answer on all five
files.

*Decisive experiment.* Build the height fields on all five files at 32×96
(\(b\)-independent, minutes), extract the level curves at the six levels, and
compare (a) with the prototype's flagged edges on DMercFail 0.8 and TURBO 0.5,
(b) with the milestone-9 \(\Gamma_{\max}\) polylines and companion samples of
the resolved references (DMercFail 0.8, TURBO 0.5, d23p4 0.5, W7-X
\(b=2.7781394\) whose two events at \(s=0.1184\) and \(0.3189\) must appear as
level-curve intersections or fold contacts), (c) on d23p4 0.5, where the
prototype saw interior-maximum counts jump by two across one cell, the two
curves must be resolved individually with EVENT measure shrinking under
refinement. Falsified if a transition seen by the surface pipeline is not a
level curve of some matched field, or if the matching produces spurious folds
on a set of measure that does not shrink.

### Option R — Reeb-complex reachability with coarea quadrature (the engine of C2)

*Idea.* On each sheet \(A\) is continuous and PL; its Reeb graph (each level-
set component contracted to a point) encodes exactly the connectivity the
direct contour tracer explores, for every contour at once. The image of
`EDGE` in the graph is a union of arcs; without transitions \(\Theta\) is the
indicator of the connected component of that image — no atoms, no snapping,
exact for the PL interpolant. A monotone transition segment is three
continuous paths from \([u_0,u_1]\) into the parent, child-1 and child-3
graphs, glued pointwise; the reachable set is the connected component of the
`EDGE` image in the glued 1-complex. \(Q\) follows from the coarea formula:
per arc, the cumulative mass function of the arc's level-set components,
computed once per triangle, with the log-singular boundary triangles handled
by the analytic model of F4. Unresolved cells are holes: the lower bound
treats a hole as boundary, the upper bound as one wildcard node joining every
arc that touches it; the gap contribution of a cell is exactly the mass of
arcs reachable only through the wildcard (two union-finds), which is option
E's refinement criterion, computed exactly.

*What changes.* §11.2–11.4 (interval sets, atoms, bitmasks, inner/outer
snapping, affine preimages) and §12.1 (polygon clipping) replaced; §11.3's
semantics (common parameter \(u\) preserved, action not; periodic adjacency
explicit; no connection by numerical overlap) unchanged; §11.5 handled by
symbolic perturbation on vertex index.

*Transitions and events.* Generic transitions are exact gluings; event cells
and capped or failed wells are wildcard holes; nothing is certified. The one
place discretization enters is a genuine transition cycle (a gluing that
composes to a self-map of an arc interval), where the reachable set is the
orbit of an interval map, iterated with a geometrically decreasing remainder
that is added to the gap; the upper bound runs the same iteration on the
complement with the inverse maps.

*Bounds.* Lower and upper \(Q\) per slice; the gap has exactly two sources,
unresolved-cell mass and the cycle remainder, both reported. A capped well is
a hole with its F3-bounded mass; a missing transition would be a missing
gluing (a mutation test).

*Cost.* Reeb graph \(O(n\log n)\) with a sweep or \(O(nm)\) naively:
milliseconds to a second for \(10^4\) cells; negligible against tracing.
Effort: one milestone merging the re-based 11–14; the direct tracer (11)
becomes a sampled cross-check.

*Strengths.* The oracle and the production path coincide on regular sheets;
the non-termination worry of §11.3 is isolated to genuine cycles and handled
as a convergent series; the refinement criterion is exact; the Reeb graph is
the physicist's diagnostic (which \(J\) contours on which sheet reach the
edge, ranked by persistence); representation-agnostic (works on C's cells,
B's mesh, or as D's memo).

*Weaknesses.* New code on surfaces with boundary, loops and a seam; CLAUDE.md's
"no framework abstractions" argues for the simplest \(O(nm)\) version first;
the cycle remainder needs its own mutation test on the ADR 0006 six-sheet
field; it does not help locate \(T\) or trace wells.

*A lighter cousin (H3).* Enumerate the finite list of critical values of \(A\)
per sheet (saddles and extrema of \(A\), port extrema along \(T\) arcs — the
subdivision points §10.4 already prescribes — and `EDGE` extrema), and classify
each band between consecutive values by one or two direct contour traces;
bisect in \(A\) when they disagree. Cheaper to implement than either the
finite-atom flood fill or the Reeb complex, exact in the continuous field,
but 20 s to 15 min of contour following per slice and only as complete as the
critical-value list.

*Likelihood and role.* Changes bound quality, not the resolution rate; raises
C's chance of meeting a per-slice gap tolerance within budget. *Decisive
experiment.* With \(A\) on the DMercFail \(\lambda_n=0.8\) chart, compute the
Reeb graphs of the two sheets and the glued complex and compare \(\Theta\) on
200 random regular points with direct contour following (exact agreement
required); then on the ADR 0006 six-sheet field compare with a finite-atom
implementation at two atom resolutions (the Reeb result must lie inside the
atom bounds and be tighter; removing one gluing must flip \(\Theta\)).

### Option D — Direct contour following in the continuous field (oracle, gap estimator, validation)

*Idea.* Follow a constant-\(A\) contour as an ODE in \((s,\alpha)\) using the
analytic gradient of \(A\) (bounce integrals of \(\partial_sB\),
\(\partial_\theta B\) with the same endpoint regularization as \(K\)); detect
edge arrival, closure and transitions (the well's highest interior maximum
reaching \(b\), or the exit becoming tangent) along the way; branch at
transitions with the forward-trace rule; memoize by (sheet, \(A\)-band).
Three roles:

*D1, the oracle.* Sampled cross-check of C+R on regular points; exact
agreement required (milestone 11's purpose).

*D2, the gap estimator (H1).* Everything the chart resolves is deterministic;
what remains is the gap set (EVENT cells, capped and failed wells, cycle
remainders), each with exactly known measure and F3-bounded mass. Estimate
\(\Theta\) statistically *only* there, with samples drawn proportional to
\(hK\,ds\,d\alpha\,db\) and the contour follower as oracle. With
\(f=f_{\rm lower}+\sum_c m_c\,p_c\), the variance is at most \(G^2/4n\) for a
gap \(G=f_{\rm upper}-f_{\rm lower}\), so \(\sigma_f=0.005\) needs
\(n=10^4G^2\) contours per equilibrium: 25 at \(G=0.05\), 900 at \(G=0.3\).
The chart only has to make the gap small; sampling closes it at a cost that
falls with \(G^2\). A cost-aware rule per cell (refine while the gap mass
halves per round, else sample) bounds the deep-quadtree risk of dense
arrangements. The deterministic \([f_{\rm lower},f_{\rm upper}]\) is never
widened; \(\hat f\pm2\sigma\) is a point estimate inside it; samples whose
contour fails stay in the deterministic bound, never at \(\Theta=0\).

*D3, birth-particle validation (H2).* Draw birth particles from the source
directly: \(\mathbf x\) with density \(h|C|/B^2\), pitch \(\xi\) uniform,
\(b=B(\mathbf x)/(1-\xi^2)\); the particle is passing iff \(b\ge B_{\max}(s)\).
The sampling density is then exactly \(hK/(2b^2)\,ds\,d\alpha\,db\), so
\(K\), \(\omega\), the surface, the slice and the \(b\)-quadrature are never
computed (verified this week: trapped fraction \(0.633\pm0.012\) against
0.630 on DMercFail; band shares match A.5). \(f\) is one number, so ±0.01 at
two sigma needs \(N\le10^4\) particles in total, of which only the trapped
36–63% need a contour. This is the strongest end-to-end validation available
(it checks \(\Theta\), whereas the \(\Theta\equiv1\) benchmark checks only the
measure) and the fallback if C1's matching fails; it cannot give
deterministic bounds, so it is not the production deliverable.

*Cost.* One trace per right-hand-side evaluation, 2–3 per predictor–corrector
step, hundreds of steps per contour → 1–20 s per contour with the F levers;
D2 needs 25–900 contours per equilibrium; D3 needs 3600–6300 contours per
equilibrium for ±0.01 (3–8 h single core, 1–2 h on four cores; 20–50 min for
±0.03). The draft's earlier estimate of "hours per slice" counted contours
per slice instead of per equilibrium and overstated D by 20–40×. Contour
following itself is unmeasured on real fields (experiment E4 did not
complete); it is the key unknown for D2, D3 and H3.

*Strengths.* Nothing global to certify; every failure local and honest;
trivially parallel; failure modes disjoint from C's, so agreement is a
genuine validation of both; D3's by-product is the per-particle well-length
distribution (F5).

*Weaknesses.* Statistical error inside the gap; contour closure under drift
and branching trees at dense arrangements unproven; contours drift in \(s\),
so the per-surface spline must become a 3-D spline or be rebuilt along the
way; long wells in the mid band dominate cost on d23p4-like files.

*Likelihood.* As sole production estimator under the deterministic-bounds
rule ≈45–55%; ≈60–70% if confidence bounds are acceptable; as D1/D2/D3
alongside C, high value. *Effort.* One milestone for the integrator and the
oracle; one for D2/D3. *Decisive experiment.* Three contours on DMercFail
\(\lambda_n=0.8\): closure, edge arrival, one transition crossing with
additivity, wall time; then 300 birth particles on DMercFail with the
completion rate and median time per contour (pass: ≥90% complete, median
< 20 s).

### Option E — Bounds-first computation (cross-cutting; required)

*Idea.* Make the deliverable of every slice the pair
\((Q_{\rm lower}(b),Q_{\rm upper}(b))\) and of the run the pair
\((f_{\rm lower},f_{\rm upper})\), computed from option R on resolved cells
and from F3 for everything else; add \(\hat f\pm\sigma_f\) from D2 inside the
gap when requested. Spend refinement effort where the gap contribution (not
the cell's own measure) is largest, allocate the time budget across slices in
proportion to mass × gap, stop at the budget and report the gap.

*Acceptance, restated.* Milestone 10.3's "≥95% resolved" becomes: for each
equilibrium, \(\sum_b w_b\,(Q_{\rm upper}-Q_{\rm lower})/(2V_hb^2)\le0.01\)
within the per-equilibrium budget, every slice reporting its gap; ≥95% of the
(equilibrium, level) cases in the matrix reach a per-slice gap below a stated
fraction of \(Q_{\rm total}(b)\). The per-case 10-minute cap becomes a
per-equilibrium budget allocated unevenly (the mid band deserves most of it).

*Why it is not a loosening.* §21.2 forbids *silent* loss; a bound that
contains the truth by construction is the opposite. ADR 0009's option 2
becomes unnecessary: a capped well is inside the bound. ADRs 0007 and 0008
become moot.

*What accuracy is useful.* For optimization use, \(f\) to ±0.01 absolute is
likely sufficient. The band above \(\lambda_n=0.95\) carries 0.3–0.7% of the
trapped mass (≤0.005 on \(f\) if left bounded) and above 0.9, 1.0–2.7%
(≤0.017); wells longer than 12 periods carry ≤0.3% (A.5, A.6).

*Failure mode.* A large region whose only route to the edge passes through an
unresolved cell inflates the gap; option R's wildcard construction finds
exactly that cell.

### Option F — Cost engineering (cross-cutting; measured)

In order of measured payoff: (1) replace the adaptive scalar quadrature in
`trace_regular_well` by a batched composite Gauss–Legendre rule in the
existing sine-squared coordinate with breakpoints at the scanned extrema,
64 nodes per segment, 128–256 on segments whose maximum is within
\(10^{-5}B_{00}\) of \(b\), with an \(n\)-vs-\(2n\) error estimate and the
existing endpoint-difference evaluation (30–134 ms → 1–5 ms); (2) cache the
coefficient vector per \(s\) and evaluate \(B\) and \(D_\parallel B\) jointly
(scalar calls 34–41 → 7–9 µs); (3) a per-surface quintic spline for the scan
(0.4 µs per point; bicubic is not acceptable); (4) one \(b\)-independent scan
per field line, shared by every well and every slice (option C's orbit
sampling gives \(N+M\) instead of \(N\times M\)); (5) vectorize the per-period
crossing/extremum bracketing; (6) Numba last. Do *not* truncate the mode set
by amplitude. Fix the near-tangent-exit `QUADRATURE_FAILURE` (default
settings, 0.5–0.9 s each; a retry with `quadrature_rtol` \(10^{-8}\) or 48
samples per wavelength recovers the identical \(K\)).

### Option G — Physics shortcuts for the nearly-passing band

Bound the band above a chosen \(\lambda_n\) by F3 (exact, cheap) and report
it; trace the mid band with the shared scan, where a 1024-period line costs
~0.2 s; cap the tail at 8–16 periods and bound the remainder (worth ≤0.3% of
the trapped mass). Use the known breakpoints of \(F(b)\) (critical values of
the height fields and the \(B_{\max}(s)\) crossings) to replace much of the
adaptive midpoint tree of §13.3 by Gauss rules between breakpoints, or use
stratified randomized \(b\)-nodes for an unbiased outer integral with an
honest error (§21.3 dimension 12 is otherwise silent). Stellarator symmetry
can halve the chart but conflicts with orbit sampling; use only after the
unsymmetrized code is validated.

## 4. Comparison

| Option | Removes M1–M9, M12 (curve/cut) | Removes M10–M11 (extraction) | Long wells | Bound on \(f\) | Effort | Likelihood of goal |
|---|---|---|---|---|---|---|
| A stay the course | no | no | veto | none until M12–14 | open-ended | ≈10% |
| B local cut on mesh | yes | no | bounded (E) | yes (E) | 1 large milestone | ≈50–60% |
| C field-line chart (height fields, orbit scan) | yes | yes | one scan per line and per \(b\); bounded tail | yes (E, R), exact measure | 3 milestones | ≈85% |
| R Reeb-complex reachability | — | — | — | exact PL bounds; gap = cells + cycle remainder | 1 milestone (replaces 12–14) | raises C's per-slice gap odds |
| D contour following | n/a | yes | bounded (E) | statistical inside the gap | 1–2 milestones | ≈45–55% alone; required as oracle/estimator/validation |
| E bounds-first | — | — | — | required | small, cross-cutting | prerequisite |
| F cost levers | — | — | 10–20× per trace; \(N+M\) scans | — | 1 milestone | prerequisite for the budget |
| G band shortcuts | — | — | bound or cheap scan | — | small | supporting |

## 5. Recommended path and acceptance criteria

1. **ADR: bounds-first acceptance (E)** in \(f\) units, with the per-
   equilibrium budget; decide ADRs 0007/0008 as moot and ADR 0009 by this
   restatement; redefine the validation matrix as five files × levels ×
   chart-refinement levels, keeping the recorded 10.2/10.3 matrices as the
   baseline of what the old pipeline could do.
2. **Milestone F0: tracer cost levers** (batched quadrature, coefficient
   cache, joint \(B\)/\(D_\parallel B\), near-tangent-exit fix), validated by
   agreement with the current tracer on the milestone-9 sample set.
3. **Milestone C1: chart builder** — per-row spline, orbit scanner, extrema
   and height fields with matching and fold detection, level curves per
   \(b\), cell classes, merge-tree well lists, sheets, hyperedges, seam,
   count-map and \(J\)-map diagnostics — validated on DMercFail
   \(\lambda_n=0.8\) (two sheets), the W7-X \(b=2.7781394\) two-event reference,
   the ADR 0006 six-sheet synthetic field and the ADR 0007 fold field, with
   mutation tests on the matching (retire a label wrongly), the seam (drop the
   shift) and the predicate (widen the Taylor band).
4. **Milestone C2: Reeb-complex reachability with bounds (R)**, with F3 as an
   exact check of the total weight on every slice and the direct contour
   tracer (D1) as the sampled oracle.
5. **Milestone C3: slice pipeline, outer integral, D2 gap estimator, D3
   birth-particle validation**, and the validation matrix.
6. Keep option B as the fallback if C1's decisive experiment fails on the
   matching or on dense arrangements; keep D3 as the fallback estimator if
   C1 fails altogether.

Early experiments in order (each under an hour): C1's decisive experiment on
all five files (height fields and level curves versus the recorded
\(\Gamma_{\max}\) and companion samples); D's three contours on DMercFail
\(\lambda_n=0.8\) (the one unmeasured ingredient); R's Reeb-versus-contour
agreement on the DMercFail chart; the orbit-scan check on d23p4
\(\lambda_n=0.8\) (one row, one orbit of \(96+1024\) periods, bit-level
agreement of \(A\), \(K\), exits with the current tracer on 20 windows).

Open questions for the researcher: the tolerance on \(f\) (0.01 assumed) and
whether a probabilistic point estimate inside a deterministic bound is
acceptable; whether the validation matrix should drop the backend and
extractor axes; whether the axis row is bounded or resolved; the per-
equilibrium time budget; disposition of ADRs 0007–0009.

## Appendix A. Measurements (2026-09-02/03)

### A.1 Trace cost levers (experiment E1)

Two equilibria (d23p4 \(\lambda_n=0.5\), DMercFail \(\lambda_n=0.8\)), 30
regular wells each plus 36 constructed near-transition wells, profiled and
re-timed single-process. Field evaluation 7–15 µs per point batched, 34–41 µs
per scalar call; coefficient-spline re-evaluation 28–30% of field time. Scan
density 447–460 samples per period (equivalent to a \(10^{-5}B_{00}\)
interpolation tolerance). Per trace: 31–40 ms (sub-period), 78 ms (1 period),
121 ms (2 periods); 60–80% in the adaptive scalar quadrature. Levers:
samples per wavelength 24→8 safe on all 96 wells but only 1.1–1.4×; mode
truncation at \(10^{-4}\) changes \(A\) by up to \(8\times10^{-4}\), \(K\) by
\(2\times10^{-2}\), itineraries on 1–2 of 30 wells and gives wrong topology
within \(4\times10^{-3}B_{00}\) of a transition (rejected); quintic per-
surface spline 256×128 per period: \(B\) to \(4\times10^{-10}\),
\(D_\parallel B\) to \(4\times10^{-7}\), identical itineraries on every test
including wells \(10^{-7}B_{00}\) from a transition, 0.4 µs per point, 0.4–0.7 s
to build; batched composite Gauss–Legendre with breakpoints at extrema:
\(n=64\) gives \(A\) to \(10^{-15}\), \(K\) to \(10^{-9}\) in 1–5 ms (near-
transition segments need 128–256 nodes). Combined realistic gain 5–10×
(sub-period) to 10–20× (multi-period) per trace. Eight of 36 near-tangent-
exit wells return `QUADRATURE_FAILURE` at default settings.

### A.2 Field-line chart prototype (experiment E2, two runs, bit-identical)

Grid 24×64 in \((s\in[0.05,1],\alpha)\), extended window \([-L,2L)\), ghost
wells for seam matching, cap 128, current tracer. DMercFail \(\lambda_n=0.8\):
142 s wall (33 ms per line for the scan plus ~45 ms per traced well), 2.27 M
field evaluations, 1536 lines, all 1504 traces regular, 0–2 wells per line,
at most one period; 102 lines carry an interior maximum, minimum gap
\(b-B_{\max,\rm int}=1.1\times10^{-3}\). Structure: count 1 for \(s<0.62\); a
count-0 hole at \(\alpha\in[-0.2,0.2]\), \(s\ge0.62\), bounded by COUNT_CHANGE
edges (the new incoming point \(m\) leaves the window); a parent band at
\(\alpha\in[4.52,4.91]\) with doubled \(A\) (22.3–22.5) and an interior maximum
within 0.003–0.056 of \(b\), bounded by EXIT_JUMP edges (the companion curve);
band and hole coincide after the shift \(\alpha\to\alpha+\iota(s)L\) to within
one cell at every \(s\) row; cutting along the band boundary gives exactly the
two sheets of the reference. 45 of 3072 edges flagged, all explained; 3.06%
of the measure touches a flagged edge. Five count-change edges bisected
(4–6 s each) to bracket width \(10^{-9}\): every located point has \(B-b=0\),
\(|D_\parallel B|\le6\times10^{-15}\), \(D_\parallel^2B\approx-63.9\), and all
sit on \(\zeta=0\) (the symmetry plane and the window boundary). Additivity
\(|A_W-A_1-A_3|/A_W\) at offsets \(10^{-1}\ldots10^{-5}\): \(3.4\times10^{-4}\),
\(4.2\times10^{-5}\), \(5.0\times10^{-6}\), (quadrature failure),
\(6.6\times10^{-8}\) on one edge and similar on the others, i.e. linear in the
offset. The extremum-height field \(B_j\) built afterwards reproduces the
located crossing (\(s^*=0.6196449011556\) by Newton in 2 ms versus
\(0.6196449011557605\) by bisection in 4.2 s), the hole's width in \(\alpha\)
(0.008 at \(s=0.62\) to 0.215 at \(s=1\)) and the parent band after the shift
(\([4.48,4.93]\) versus \([4.52,4.91]\) at a grid spacing of 0.098).

TURBO \(\lambda_n=0.5\), 24×64: 99 s, all regular, zero flagged edges; an
independent 3-D search finds no along-line maximum within 0.057 of \(b\) for
\(s\ge0.02\), so any transition sits at \(s<0.02\); the count-0 band (19% of
lines) is a window artifact of the QH-like helical well and is REGULAR under
ghost matching. Rows down to \(s=0.002\): 12 COUNT_CHANGE and 3 IRREGULAR
edges near the axis, bisecting to \(\Gamma_{\min}\) points
(\(D_\parallel^2B>0\), well births with \(A\propto\) offset). d23p4
\(\lambda_n=0.5\), 8×16: 10 s; 31% of cells flagged (grid-limited); interior-
maximum counts jump by two across single edges (two transition curves in one
cell).

### A.3 Total-weight identity (experiment E3)

Volume side: two independent routes (finite differences of \(V_{\rm tr}\) and
the direct singular quadrature over the support \(B_{\max}(s)>b\)) agree to
0.01–0.06% at all five levels on DMercFail and 0.02–0.13% on d23p4 once the
\(V\)-grid has ≥192 \(\zeta\) points per period; the unrestricted integral
overshoots by factors 1.7/3.0/5.7 at d23p4 \(\lambda_n=0.8/0.9/0.95\).
Surface side: the midpoint-rule sum of traced \(K\) over every incoming point
of an \((s,\alpha)\) grid agrees with the volume side to 0.016% (12×24) and
0.048% (24×48) on DMercFail \(\lambda_n=0.8\) (281 and 1128 wells, all
regular), and 0.74% → 0.37% on d23p4 \(\lambda_n=0.5\) (214 → 858 wells,
\(O(1/n)\)); at d23p4 \(\lambda_n=0.8\) the residual (+4.5%, +0.6%, −0.9% at
12, 48, 96 rows) is the unaligned support cut \(s^*=0.456\). The outer
integral \((1/2V_h)\int Q_{\rm total}/b^2\,db\) gives 0.396317 (d23p4) and
0.629512 (DMercFail) against trapped fractions 0.396148 and 0.629588. The
bound in action: at d23p4 \(\lambda_n=0.95\) with a deliberately low cap of 16,
the one failed well (16 periods, 17 interior maxima, closest maximum
\(5\times10^{-4}\) below \(b\); a default-tolerance `QUADRATURE_FAILURE`
recoverable with `quadrature_rtol` \(10^{-8}\)) has true weight 1.800 and the
deficit \(Q_{\rm total}-Q_{\rm resolved}=2.114\) bounds it; the remaining
0.314 is the resolved sum's own discretization error. No `MAX_PERIODS` well
occurred at any level with cap 64 (longest well 22 periods at d23p4
\(\lambda_n=0.8\)). Support at the matrix levels: d23p4 \(s\ge0.46\) at
\(\lambda_n=0.8\), \(\ge0.71\) at 0.9, \(\ge0.85\) at 0.95; DMercFail
\(s\ge0.17\) at 0.9, \(\ge0.51\) at 0.95.

### A.4 Contour following (experiment E4)

Not completed (the agent that would have run it was cut off by usage
limits); the analytic-gradient integrator exists in prototype form. This is
the one unmeasured ingredient of options D, D2, D3 and H3 and is the first
early experiment of §5.

### A.5 Where the trapped phase space sits (experiment E5, band fractions)

From \(V_{\rm tr}(b)\) of F3 (Gauss–Legendre 64 in \(s\), 128×64 periodic in
\(\theta,\zeta\), per-surface maxima on a 256×128 grid; \(V_h\) converged to
\(10^{-9}\)):

| file | trapped fraction | \(\lambda_n<0.1\) | \([0.1,0.5]\) | \([0.5,0.8]\) | \([0.8,0.9]\) | \([0.9,0.95]\) | \(>0.95\) | \(\Delta f\), band \(>0.9\) | \(\Delta f\), band \(>0.95\) |
|---|---|---|---|---|---|---|---|---|---|
| d23p4 | 0.396 | 6.5% | 56.8% | 31.0% | 4.3% | 1.0% | 0.34% | 0.0055 | 0.0014 |
| DMercFail | 0.630 | 19.0% | 50.3% | 21.5% | 6.4% | 2.0% | 0.73% | 0.0174 | 0.0046 |
| PCA | 0.385 | 1.5% | 54.1% | 37.3% | 5.3% | 1.3% | 0.44% | 0.0068 | 0.0017 |
| TURBO | 0.367 | 0.4% | 44.5% | 44.6% | 7.8% | 2.0% | 0.71% | 0.0100 | 0.0026 |
| n3are | 0.278 | 0.2% | 54.9% | 38.9% | 4.8% | 1.0% | 0.29% | 0.0036 | 0.0008 |

The mass per unit \(\lambda_n\) peaks at 0.42–0.56 on four files (0.05 on
DMercFail).

Capped-well measure at the long-well levels (forward traces from every
incoming crossing on a 6×16 chart over the trapped support, cap 64 then 256):

| case | support | wells | statuses | capped | periods median / p90 / max | wall |
|---|---|---|---|---|---|---|
| d23p4 \(\lambda_n=0.9\) | \(s\ge0.71\) | 18 | 18 regular | 0 | 4 / 5 / 10 | 6 s |
| d23p4 \(\lambda_n=0.95\) | \(s\ge0.85\) | 15 | 14 regular, 1 quadrature failure | 0 | 4 / 11 / 15 | 11 s |
| DMercFail \(\lambda_n=0.9\) | \(s\ge0.17\) | 46 | 46 regular | 0 | 1 / 3 / 4 | 8 s |

No forward well reached the 64-period cap at any level where the pipeline
records `max_periods` at 1024: the binding traces there were the backward
scans from marginal points. The one quadrature failure is the near-marginal
bug of A.1.

### A.6 Birth-particle check (from the sampling review)

Sampling \(\mathbf x\) with density \(h|C|/B^2\) and \(\xi\) uniform, with
\(b=B/(1-\xi^2)\) and passing iff \(b\ge B_{\max}(s)\), gives a trapped
fraction \(0.633\pm0.012\) on DMercFail (direct: 0.630) and band shares
19.7/50.2/21.6/6.0/1.8/0.7% (direct: 19.0/50.3/21.5/6.4/2.0/0.73%); on d23p4
\(0.360\pm0.015\) (direct 0.396, 2.4σ low at \(N=1000\) with a crude per-
surface maximum; to be repeated with the milestone-2 extrema). Per-particle
well lengths: wells ≥1 period carry 5.6% (DMercFail) and 30.8% (d23p4) of the
trapped mass, ≥4 periods 0.6% and 11.4%, >12 periods ≤0.3% on both.

### A.7 Matrix statistics computed from `milestone10.3-real-equilibria.json`

50 of 120 cases finish under 600 s: all 32 no-transition cases, 9 of 10
resolved, 9 unresolved with quick terminals. 68 of the 69 slower cases are
unresolved. Every case with a transition at \(\lambda_n\ge0.8\) is unresolved;
no case at \(\lambda_n\in\{0.9,0.95\}\) finishes under 600 s. Resolved cases:
TURBO 0.5 (3–110 s), DMercFail 0.05 (259–261 s), DMercFail 0.8 (30–71 s),
d23p4 0.5 gmsh:PV (5490 s). Remediation attempts: cap escalation 40,
background 25, thin-strip 15, source budget 13, local projection 8, scan
resolution 4.
