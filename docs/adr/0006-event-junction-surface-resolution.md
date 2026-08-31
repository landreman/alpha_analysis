# ADR 0006: Set the surface-resolution prerequisite for event-junction cuts

- **Status:** Accepted
- **Date:** 2026-08-30
- **Milestone:** 10.2
- **Design sections:** §5.4, §§8.3–8.4, §§10.2–10.5, §21.2, §21.3, and §23 milestones 10.2–10.3

## Context

This section records the prototype at the time of the decision; follow-up
implementation evidence appears below.

Milestone 10.2 requires that a curve with a bracketed contact "cuts its regular
arcs and carries an explicit event hyperedge at the localized contact" and that
"no dangling cut terminates in a surface interior without an event node."
Milestone 10.3 separately assigns local surface refinement near thin transition
strips and failure-directed background refinement to the next milestone.

The contact prototype localizes the events and evaluates their one-sided action
limits, but a production synthetic case cannot yet meet its cut-topology test.
The decision is whether 10.2 should investigate and implement local surface
refinement as a prerequisite, or whether its acceptance permits a partially cut
arrangement with explicitly unresolved incident arcs. The latter would require
changing the test, so it is not an implementation choice to make silently.

The synthetic field is

\[
B=2+0.2s+\cos(3\zeta)+0.2\cos\zeta
  +0.05s\sin\theta\sin\zeta,
\qquad \iota=0,\quad G=3,\quad I=0.
\]

At \(s=0.5\), \(\theta=0,\pi\), its two lower maxima have
\(\cos\zeta=-\sqrt{7/30}\), with bounce field
\(b=3.001685664207343\). A higher maximum bounds the surrounding ordinary
well. The six limiting branches are independently identifiable:
\([a,m_1]\), \([m_1,m_2]\), \([m_2,d]\), \([a,m_2]\), \([m_1,d]\), and
\([a,d]\). This pins a six-sheet test without using the implementation's output
as the expected answer.

With the structured `(4,16,36)` background and marching tetrahedra:

- The incoming mesh has 571 vertices and 904 triangles. Extraction and critical
  classification both report `REGULAR`; the two maximum curves have 71 and 69
  source vertices.
- Thirteen triangles in incoming component 1 have the opposite signed
  `(x,y)` projection from its other 254 triangles. Since `iota=0`, `(x,y)` is
  the local field-line chart. All shared edges have consistent mesh winding:
  these are folds in the projection, not merely reversed triangle indices.
  The positive doubled chart areas reach `2.18675695863912e-3`.
- Bounded contact bisection and simultaneous marginal-root solves recover two
  physical events, each encountered on both critical curves. Four regular arcs
  have finite, independently integrated one-sided limits satisfying additivity.
- The prototype accepts three arcs, produces **five sheets**, and retains arc 3
  with all three ports explicitly unresolved. Its proposed constrained path
  branches at a non-event vertex near the critical boundary. Accepting that
  path would violate the no-dangling-cut requirement.

Investigation tried joint endpoint insertion, local field-line-chart insertion
and projection, a guard against cyclic edge flips, and bounded cavity
retriangulation preserving all vertices, the directed cavity boundary, and signed
chart area. These did not complete the fourth cut. Those experimental paths are
not a converged surface-refinement algorithm and are not ready to merge.
The folded input chart is evidence for investigating local refinement, not proof
that refinement is necessary or sufficient: a remaining cutter defect has not
been excluded. The next investigation must distinguish those possibilities.

The W7-X reference contact localization independently reproduces the four source
brackets from ADR 0003. They pair into **two physical events**, related across the
periodic seam, at `s=0.1184098663731` and `s=0.318907160215`. The final parameter
bracket widths are below `1e-5`; both marginal equations are solved at each event.
Localization alone does not certify the incident cut geometry or connectivity.

`test_regular_arcs_cut_into_six_wells_without_dangling_event_ends` remains an
ordinary failing test: **5 != 6**. It has not been marked slow, skipped, xfailed,
given a looser tolerance, or changed to expect the incomplete result. The failed
arc is not replaced by zero action, an absent connection, or an invented event
at the accidental branch vertex. Milestone 10.2 remains unchecked.

## Options

1. **Make certified local surface refinement a prerequisite inside 10.2.** Keep
   the six-sheet test and the current acceptance criteria. Specify and implement
   bounded local refinement/repair around event junctions and critical-boundary
   strips, including preservation of component provenance, unresolved-area
   accounting, and a convergence check. Milestone 10.3 would then coordinate
   these remedies across the matrix rather than introduce the prerequisite.
   This expands 10.2 and changes the division of work in §23; it does not weaken
   the topology test.
2. **Accept partially cut arrangements in 10.2 and defer completion to 10.3.**
   Explicitly state that localized events may retain geometrically unresolved
   incident arcs, even on the synthetic acceptance field. Keep those ports and
   their unresolved connectivity. The five-sheet partial result must not be
   described as the field's resolved six-sheet topology. This changes the
   acceptance evidence and needs approval before altering the failing test.
3. **Revise the milestone ordering.** Develop the necessary surface-resolution
   work first on a separately authorized prerequisite. This changes the ordered
   milestone plan and cannot be inferred from "next."

## Decision

**Option 1, scoped by a discriminating resolution sweep run on 2026-08-30.**
The six-sheet test and the no-dangling-cut criterion stay exactly as written,
and milestone 10.2 is not complete until they pass. The prerequisite work
stays inside 10.2. Milestone 10.3 remains the coordinator that automates
remedies across the matrix; it does not inherit the missing capability.

The sweep reran the identical pipeline — same field, extractor, localizer,
arc builder, and cutter, all unchanged — on six structured backgrounds:
`(4,16,36)` (baseline), `(4,16,48)`, `(4,24,36)`, `(4,24,54)`, `(6,24,54)`,
and `(4,16,45)`. Per grid, it counted minority-orientation triangles in each
incoming component's `(x,y)` chart and recorded the cut outcome with its
machine-readable unresolved reasons:

| grid | folded triangles | sheets | unresolved arcs | reason class |
|---|---|---|---|---|
| 4,16,36 | 13 | 5 | 3 | branches away from authorized endpoint |
| 4,16,48 | 19 | 3 | 0,1,2,3 | endpoint / side-count / indecisive assignment |
| 4,24,36 | 4 | 5 | 3 | incident-side count not two |
| 4,24,54 | 10 | 3 | 0,1,2,3 | endpoint / side-count / indecisive assignment |
| 6,24,54 | 9 | 5 | 2 | incident-side count not two |
| 4,16,45 | 15 | 5 | 3 | branches away from authorized endpoint |

Three facts discriminate between the hypotheses left open in Context:

1. **Folds are persistent, not a coarse-grid artifact.** Every resolution
   produces folded chart triangles (4–19); their doubled areas shrink but the
   count does not go to zero. No affordable global refinement yields the
   fold-free input that would have exonerated the cutter.
2. **Fold count does not predict cutting failure.** The 4-fold mesh still
   fails, and the two grids with intermediate fold counts fail on all four
   arcs — strictly worse than the baseline.
3. **The failure is not stable under refinement.** Which arc fails, and which
   of three distinct rejection reasons fires (constrained path branching at a
   non-event vertex, a companion cut with the wrong incident-triangle-side
   count, an indecisive parent/child side assignment), both change with the
   grid. Two of the three signatures are insertion-geometry rejections that a
   robust cutter must handle on meshes of this quality.

Conclusion: global background refinement is demonstrably not sufficient, and
mesh folding is not the proximate cause. The blocked capability is
**robust constrained insertion at event junctions on charts that fold** —
cutter-side work. Milestone 10.2 therefore proceeds by making the insertion
and side-assignment machinery correct on the meshes the extractor actually
produces: certified handling of folded chart regions during companion-curve
insertion, deterministic incident-side recovery, and a decisive or explicitly
unresolved side assignment — with bounded *local* repair used only where a
certified insertion needs it, carrying provenance and unresolved-area
accounting. §23's milestone 10.3 text is unchanged; its "local surface
refinement near thin transition strips" remains a matrix-level remedy, not
the missing prerequisite here.

If this work reveals that the synthetic case genuinely requires an open-ended
remeshing capability rather than bounded insertion robustness, that finding
reopens this ADR in favor of option 3 with a properly scoped prerequisite
milestone; it must not silently expand inside 10.2.

## Consequences

Milestone 10.2 continues with its acceptance criteria unweakened; the
six-sheet test stays a live, honestly failing gate until the cutter-side work
above makes it pass. The draft still contains investigation code and that
failing topology test, not a completed milestone. `docs/DESIGN.md` and the
milestone row must not be revised to imply acceptance before the test is
green. The experimental refinement/retriangulation paths already tried are
not part of the decided scope and should not merge unless the insertion work
independently justifies them.

The final local `make check` also exposes a regression in the existing
`test_dmerc_reference_sheet_graph_is_budget_invariant_or_explicit`: the
experimental path-degree guard retains one transition unresolved where the
existing acceptance test requires a cut. The baseline passed all 166 tests;
this draft passes 167 and fails two. The guard must be diagnosed and the existing
acceptance restored before merging, without disabling its safety check merely
to recover a green suite.

The required 100-case matrix, mutation checks, final diagnostic/persistence
verification, and ready-for-review gate remain incomplete. No Claude review is
requested for this stopped draft. The existing Tests gate is not waived: it must
be green before this milestone is marked complete or the next one starts.

## Implementation follow-up

The decided cutter-side work now completes the six-sheet synthetic arrangement
on all six backgrounds without weakening its acceptance test. A fallback through
actual adjacent triangle faces replaces the branching vertex path; it splits
crossed edges within the existing distance allowance, preserves provenance and
action stencils, and is bounded by 64 faces per insertion. The experimental broad
chart-cavity retriangulation module was removed. The existing DMerc test also
passes: the guard now recognizes physical `EDGE` endpoints in reordered critical
curves while retaining the companion-cut degree check.

The W7-X reference retains two cut regular arcs and both explicit physical events;
coarse `EDGE` strips and a nonseparating incident arc remain unresolved. Distinct
one-sided limits at a shared unresolved event vertex are preserved on their ports;
the vertex action is `NaN`, with its incident measure included in unresolved flux.
No complete real-equilibrium topology or reachability value is claimed.

See `docs/validation/milestone10.2-event-junctions.md` for tests, mutations,
diagnostics, resolution evidence, and the current validation/publication gates.
This follow-up does not change the Decision or milestone 10.3's scope.
