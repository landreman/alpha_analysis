# ADR 0010: Compute bounded f with a local atlas of well branches

- **Status:** Accepted by the researcher
- **Date:** 2026-09-11
- **Milestone:** Redesign R0–R8
- **Design sections:** §3–§13, §17–§18, §20–§28
- **Supersedes:** ADRs 0007–0009 as proposals for the active implementation path;
  the requirement to complete global constrained cuts before computing the metric

## Context

The recorded milestone-10.3 matrix contains 32 `no_transitions`, 10 `resolved`,
77 `unresolved_explicit`, and one `exception` outcome: 42/120, or 35%, satisfy
the old resolved-cut criterion. It consumed 112.6 hours. The report in
[`../validation/milestone10.3-real-equilibria.md`](../validation/milestone10.3-real-equilibria.md)
remains the record of that run; neither a new acceptance policy nor a later
crash fix changes its outcomes. Milestone 10.3 did not pass.

The scientific objective is to compute the phase-space fraction defined in
DESIGN §3, rather than to produce a particular global triangulation. The
researcher has approved a new implementation plan based on local well
branches, bounded reachability, and error accounting in units of \(f\).
This ADR records those decisions; they are not open implementation choices.

The exploratory redesign of 2026-09-03 supplied useful hypotheses, but its
claims about Reeb connectivity, zero-measure events, global chart labels,
and deterministic residual bounds required correction. The active contract
is [`../DESIGN.md`](../DESIGN.md), incorporating the decisions below, rather
than the exploratory proposal or an old milestone's implementation recipe.

## Options considered

1. **Continue the global cut coordinator.** Preserve the existing
   representation and add more event certification, surface insertion, and
   remediation. The measured results do not support making this a prerequisite
   for computing \(f\) within the approved budget.
2. **Build a local atlas of well branches and retain bounded action
   reachability.** Reuse field evaluation and trustworthy numerical kernels,
   obtain well branches from forward scans and root continuation, and account
   for every unresolved population and possible connection. This is the
   selected production direction.
3. **Use direct contour following with birth-particle sampling alone.** This
   provides an independent check and a possible statistical estimate, but
   incomplete branch exploration is not proof of unreachability. It does not
   replace the deterministic production bounds.
4. **Replace action reachability with connected components of a Reeb graph.**
   Rejected as a reachability rule: traversing an ordinary Reeb edge changes
   the action. Reeb graphs may later compress the representation while
   preserving the bounded propagation rules.

## Decision

### Preserve the physical metric

DESIGN §3 remains authoritative. At fixed \(b=B_b=W_0/\mu\), ordinary
motion follows a connected constant-\(A\) contour on a continuous well
branch. A split/merge permits every incident branch at the same physical
event parameter, with its own action. Generic limits obey
\(A_W=A_1+A_3\). The relation remains undirected and existential; passing
motion does not bridge trapped states. A failed or capped trace remains
unresolved. This decision does not introduce a loss probability, a finite
orbit-time definition, or a different magnetic moment.

### Represent well branches locally

Replace the mandatory extracted-surface and constrained-cut construction
with overlapping local charts of incoming/first-outgoing root pairs. Preserve
well multiplicity, root order, unwrapped return length, orientation along
\(+\mathbf B\), periodic identifications, and unique ownership of each
integrated population. A local \((s,\alpha)\) chart does not establish a
single-valued global graph or a global integer well label. The twisted seam
must remain explicit even when a shared scan makes a particular match an
integer window shift.

Forward extrema catalogues and their heights may be reused across pitch
slices. A nondegenerate extremum has a smooth height field only on its local
continuation domain; folds, monodromy, and scan-window boundaries require
explicit bookkeeping. Matching must establish root-family continuation,
not Euclidean proximity or equal counts. A maximum–minimum pair appearing
strictly below \(b\) does not itself split a trapped well. Such a change may
be treated as ordinary continuation only after certifying the first-return
roots and absence of an intervening barrier at \(B=b\).

Existing surface backends and cuts remain useful diagnostic and regression
references. The new production path does not depend on resolving them first.
No new base dependency or change to the §19.2 boundaries is approved here.

### Use finite action atoms with global uncertainty propagation

Implement ordinary and transition-aware lower/upper reachability on the
atlas using finite action and event-parameter atoms. Ordinary links preserve
action; transition links preserve the common event parameter. Inner
propagation contains only certified connections and outer propagation
contains every connection consistent with the unresolved data. Refinement
must demonstrate containment and account for action and geometry errors.

A Reeb point represents one connected level contour. Without transitions,
the reachable Reeb set is the image of `EDGE` itself, not the connected
component containing that image. Optional Reeb compression must propagate
reachable subsets of arcs with the same action/parameter rules. Neither two
union-find passes nor an assumed geometric remainder for transition cycles
replaces finite-atom termination. Any later cycle accelerator requires its
own proved enclosure and convergence conditions.

Unknown events and missing branches can change reachability far beyond their
own cells. Their possible connections must be propagated globally before
integrating the uncertain population. Zero geometric area, small local
flux, or a short sampled long-well tail is not grounds for discarding an
event or declaring its effect on \(f\) small. Shrinking an event cell is not
by itself a proof that the resulting global gap shrinks.

### Maintain a population and error ledger

Use an independent total trapped-weight calculation to check coverage and
bound populations not yet traced. In general the volume form includes the
indicator \(\chi_{\mathrm{tr}}(s,\theta,\zeta;b)\) that the allowed
point belongs to a trapped field-line interval:

\[
T(b)=\int ds\,d\theta\,d\zeta\,
 \frac{h(\sqrt{s})|G+\iota I|}{B\sqrt{1-B/b}}
 \,[B<b]\,\chi_{\mathrm{tr}}.
\]

Replacing \(\chi_{\mathrm{tr}}\) by
\([B_{\max}(s)>b]\) requires field-line density for almost every radius
contributing volume, or another demonstrated justification. Isolated
rational surfaces can have zero radial measure; a rational-transform
plateau cannot be ignored. A line on such a plateau can be passing even
when another line on the same surface reaches a larger field strength.

For disjoint covered weight \(R\), a residual upper budget uses
\(T_U-R_L\), including errors in both terms and verifying their consistency.
Subtracting two unconstrained numerical estimates does not give a
deterministic bound. A global residual budget does not assign an exact mass
to each missing cell, and it does not bound the reachability influence of
that cell on already-covered states. Count every population once and retain
separate records of missing weight and uncertain connectivity.

The final enclosure must include field representation, root completeness,
root location, \(A\)/\(K\) quadrature, atlas coverage and interpolation,
reachability atoms, weighted integration, denominator, and outer-pitch
errors. Agreement between two quadrature estimates is supporting evidence,
not automatically a rigorous enclosure. Nondegenerate logarithmic
separatrix asymptotics require remainder control before they are used as
upper bounds and do not apply uniformly where the curvature vanishes.

### Keep independent contour and sampling checks

Implement the continuous-field contour follower before relying on it as an
oracle. It must explore every permitted branch; unsuccessful integration,
uncertain closure, and branch-budget exhaustion remain unresolved. Compare
it with certified atlas classifications outside numerical uncertainty
bands, and measure convergence of discrepancies with refinement.

Birth-particle validation samples positions proportional to
\(h|G+\iota I|/B^2\), with \(\xi\) uniform on \([-1,1]\) and
\(b=B/(1-\xi^2)\). Optional sampling of the globally uncertain population
uses the measure in units of \(f\),
\(hK\,|ds\wedge d\alpha|\,db/(2V_hb^2)\), with conditioning and sampling
weights for the selected uncertain set accounted for explicitly. Failed
classifications remain interval-valued outcomes; dropping them would bias the estimate. Statistical
confidence intervals are reported separately and do not satisfy the
deterministic acceptance criterion by themselves.

### Adopt the initial acceptance and runtime contract

The benchmark source is \(h(\rho)=1\), declared explicitly in every run.
The scientific interface retains support for other nonnegative source
profiles. This default is a new benchmark decision, not a claim about the
source used in an unreproduced exploratory measurement.

- All five reference equilibria must produce a deterministic enclosure with
  **full width** \(f_U-f_L\le0.01\). This is not a half-width of 0.01.
- At least 29 of the 30 physical slices (five equilibria times the six
  radially global levels \(0.05,0.1,0.5,0.8,0.9,0.95\)) must satisfy
  \(Q_U-Q_L\le0.01\,T_U(b)\), using the independently bounded total
  trapped weight as the upper normalization, with independent ledger width
  \(T_U-T_L\le0.001\,T_U\) as required by DESIGN §20.3 so an inflated
  population upper bound cannot make the normalized gap artificially small.
  A certified empty slice has
  zero width. The old backend/extractor combinations are diagnostic
  comparisons, not independent physical successes.
- A complete equilibrium calculation has a 600-second wall-clock guard,
  including cold field loading, normalization, interpolation setup, shared
  scans, all pitch slices, and outer integration. A standalone slice
  calculation also has a 600-second guard including its cold setup; a full
  equilibrium run does not receive 600 seconds per slice.
- Save available bounds and reasons at 60, 300, and 600 seconds, or on earlier
  completion. A budget terminal is reported as a failed acceptance when the
  required enclosure has not been obtained, rather than relabeled resolved.
- Record hardware, worker count, cache state, configuration, provenance, and
  stage costs. Parallel execution does not multiply the wall-clock budget.
  Raising a budget or changing either error target requires a further
  researcher decision.

### Start the active milestone sequence from main

**2026-09-12 amendment:** [ADR 0011](0011-pre-r4-feasibility.md) records the
researcher's insertion of R3.5 between R3 and R4 after PR #29 merged. The original
sequence below is retained as history; current dependencies are in DESIGN §23.

The approved sequence is R0 baseline and population ledger; R1 shared forward
catalogue and batched \(A\)/\(K\); R2 branch atlas and local height fields;
R3 independent continuous contour follower; R4 ordinary finite-atom
reachability; R5 transitions and global uncertainty; R6 weighted slice
quadrature and persistence; R7 outer \(f\) integration and birth-particle
validation; R8 five-equilibrium acceptance. DESIGN §23 supplies each
milestone's detailed acceptance tests. None is complete merely because this
planning ADR is accepted.

The planning changes are to be reviewed as a planning/evidence PR based on
`main`: Markdown plus unchanged historical data such as the 10.3 matrix JSON
when absent from main. No numerical implementation is included. Do not merge draft PR #24 as-is; its reviewed head `c11fb30` has a
failing test gate. Leave that branch and the recorded matrix available as
historical evidence. Do not reset, rewrite, or silently absorb its history.

R0 starts from `main` and establishes a passing baseline. It may selectively
salvage a required fix or numerical kernel from #24 only after identifying
the change, its dependencies, and its relevant tests. Importing the old
failed matrix policy assertion is not required. If that assertion is
salvaged, replace its obsolete policy role with an explicit historical
verification of the recorded 32/10/77/1 outcomes and the fact that the old
criterion failed. Preserve scientific checks, including contextual coverage
of the recorded crash and its proposed fix. This changes the approved
forward acceptance policy; it never makes milestone 10.3 pass retroactively.

## Consequences and disposition of earlier ADRs

ADRs 0007 and 0008 are superseded for the new path. Their implementations and
proposal bodies remain historical; this decision does not retroactively
accept their legacy event/cut semantics. The new path must represent or
bound the same unresolved physical situations under the rules above.

ADR 0009 is superseded by this redesign. Its measured 35% result is retained
as a failure of the old criterion, not accepted as the new target. Its
statements that cap exhaustion proves an algorithmic impossibility are not
assumptions of the new design.

ADRs 0001–0006 retain their recorded decisions and scientific regression
value. Requirements specific to constructing the old global cut apply when
that diagnostic implementation is used, not as prerequisites for the atlas.
The applicable physics, axis conventions, component identity, additivity,
and no-silent-loss requirements remain in force through DESIGN.

This planning change modifies Markdown only. Numerical implementation, test
changes, selective salvage, benchmark runs, and PR #24 disposition are
subsequent work under the active milestones. Existing reports and original
planning documents remain archived with their provenance and limitations.
