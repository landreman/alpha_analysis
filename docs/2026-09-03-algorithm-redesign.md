# Algorithm redesign: review disposition and historical proposal

**Status:** The September 3 proposal is superseded by the accepted redesign in
[DESIGN.md](DESIGN.md) and [ADR 0010](adr/0010-branch-atlas-and-bounded-f.md).
The numbered implementation plan is DESIGN.md §23; [STATUS.md](STATUS.md) records
its progress. This page records the review outcome, not an alternative work queue.

The [original proposal](history/2026-09-03-algorithm-redesign-original.md) is
preserved verbatim after a nonnormative archive banner. Its option probabilities,
implementation instructions, cost projections, and Appendix A claims must not be
used as acceptance evidence for the active plan.

## Ideas retained in the active design

- A **local atlas of well branches**, with explicit lifts and seam maps, replaces
  the requirement to insert every companion curve into a surface extracted from
  a three-dimensional mesh. Projection onto \((s,\alpha)\) is locally regular at
  incoming bounce points; it does not establish a single global chart or a
  globally single-valued well label.
- Along-line extremum heights provide pitch-independent information for locating
  transitions. Their smoothness is local to nondegenerate, correctly matched
  extrema. Shared forward scans, reused across wells and pitches, and batched
  quadrature are the first cost experiments; their accuracy and speed must be
  measured on the production path.
- A **total trapped-weight ledger** provides an independent normalization check
  and accounts for unresolved contributions. Reachability uncertainty is
  propagated globally, and numerical approximation errors enter the reported
  bounds. The goal is an accurate bound on \(f\), with explicit work budgets.

## Corrections that govern interpretation of the archive

**Reachability is not ordinary connectedness of a Reeb graph.** On a sheet with
\(A=s\) and the plasma edge at \(s=1\), the Reeb graph is an interval. Its ordinary
connectedness would incorrectly declare every \(A<1\) contour edge-connected.
Ordinary motion preserves action and follows a single contour component;
transitions transfer between branches only through their common physical
parameter. A Reeb representation may organize those components, but cannot
replace action-aware reachability by an ordinary graph flood fill. General
transition cycles also do not have a demonstrated geometrically decreasing
remainder merely because their maps are piecewise affine.

**Zero local measure does not imply zero connectivity influence.** An isolated
or small event region can mediate the only route from a large regular region to
the edge. Neither a codimension argument nor a small long-well tail justifies
bounding the uncertainty in \(f\) by that region's own mass. Lower and upper
reachability must include every state whose classification depends on unresolved
connections. Events enforced by symmetry need not be isolated.

**The ledger needs hypotheses and numerical error bounds.** Replacing trapped
well sums by a volume integral restricted only by \(B_{\max}(s)>b\) requires
dense field lines on almost every contributing surface, or an explicit
classification of which orbits are trapped. On
rational field lines, line-dependent passing families can coexist with trapped
families even when the surface maximum exceeds \(b\); a rational interval in
\(s\) cannot be dismissed as a measure-zero exception. Unsupported cases must
be classified or bounded explicitly. A continuous identity, agreement between
quadrature rules, an asymptotic logarithmic model, or agreement with a PL
interpolant does not itself certify a numerical enclosure. Subtracting a computed
resolved weight from a computed total needs errors from both quantities.

**The proposed D2 sampler must sample the global ambiguity set.** Its target
mass is proportional to \(hK\,b^{-2}|ds\,d\alpha|\,db\), including the outer-pitch
weight. This includes regular states made uncertain by unresolved connections,
not only local event/capped cells. The stated variance calculation additionally
requires valid normalized mass weights and a correct classification oracle;
failed contours retain uncertainty. Statistical confidence intervals remain
separate from deterministic bounds.

## Verified historical matrix statistics

The tracked [10.3 JSON](validation/milestone10.3-real-equilibria.json) records
42/120 resolved-or-no-transition outcomes: 32 without transitions and 10 marked
resolved. Applying the subsequently documented ten-minute limit gives **41/120**:
32 no-transition and nine resolved cases. Of the 50 cases finishing below 600 s,
nine are unresolved, not one. The remaining 70 comprise 68 unresolved cases, one
exception, and one resolved case taking 5489.94 s.

All four DMercFail \(\lambda_n=0.8\) combinations resolve in 30–71 s, contrary to
the archived blanket claim that every transition-bearing case at or above 0.8
fails. Low levels do not always succeed: PCA and TURBO at 0.1 fail in all four
combinations, and DMercFail at 0.05 fails on both gmsh combinations. There are
31 recorded background-level escalations, not 25. The summary's six
`event_geometry` counts are six terminal reasons in two cases, not six cases.
These corrections leave the original validation reports and raw JSON intact.

The 120 entries repeat 30 physical file/pitch slices over four backend/extractor
combinations. They measure the former cut pipeline, not successful evaluation of
\(f\). The d23p4 0.5 gmsh/PV result has nine arcs, four events, and six sheets;
its unresolved event connectivity still needs treatment. The matrix omits
surface-wide actions, while a separate DMercFail diagnostic did compute them.
Thus neither “no interacting real cut exists” nor “no real action field exists”
is a valid project-wide conclusion.

## Status of Appendix A and projected performance

The Appendix A.1–A.6 pilots have not been independently replicated in this
review. Targeted searches of the repository and plausible session scratch
locations did not recover their scripts or raw outputs; the proposal gives no
reproducible path, commands, seeds, or implementation hashes. Their numerical
source profile \(h(\rho)\) is unspecified. Using \(h=1\) for the current benchmark
does not establish that the historical pilots used it or validate their numbers.
Recovering that evidence or rerunning documented experiments is required before
using those claims in the active plan.

In particular, the observed “at most 0.3% beyond twelve periods” is not a proved
mass bound. With approximately 360 trapped samples in the reported d23p4 pilot,
even zero tail observations would permit 0.829% at a one-sided 95% binomial
confidence level; one observation permits 1.311%. The pilot also reports a
trapped-fraction discrepancy and an approximate surface maximum. Coarse chart
samples with no capped wells provide no probability bound on the missing mass.
A.5's own n3are entries put 93.8%, not at most 92%, in the 0.1–0.8 band.

The proposed minutes-per-equilibrium runtime and 0.01 accuracy are projections,
not a demonstrated runtime/accuracy pair. Direct contour following was not
completed, and its timing remains unmeasured. The original option likelihoods
are subjective estimates, not observed success probabilities. The active design
therefore requires measured intermediate results and end-to-end error accounting
before declaring the redesign successful.
