# Architecture decision records

An ADR is what you write **instead of** guessing. Milestone work stops and produces an
ADR when the implementation hits a question `docs/DESIGN.md` does not answer and the
answer would change the physics, the numerics, or an acceptance criterion.

Write one when:

- `docs/DESIGN.md` is ambiguous or self-contradictory on something you must decide;
- the design as written appears to be wrong, or cannot be implemented as specified;
- an acceptance criterion cannot be met without loosening a tolerance, widening an
  error bound, or removing a check;
- you need a new base dependency, or need to cross one of the boundaries in §19.2;
- a topological or numerical failure cannot be resolved and would otherwise have to be
  hidden (§21.2 forbids hiding it).

Relaxing a tolerance, marking a test `xfail`, or narrowing a test's input range to
reach green is never the answer on its own. If that is genuinely the right call, it is
an ADR, and the researcher approves it before the milestone is marked done.

## Process

1. Copy `template.md` to `docs/adr/NNNN-short-slug.md`, next free number.
2. Fill it in. Status starts as `Proposed`.
3. Commit it, push the branch, open the pull request as a **draft**, name the ADR in
   the body, and stop. Do not pick one of the options and carry on.
4. When the researcher accepts it, set Status to `Accepted`, update the
   ADR index and any necessary active handoff notes in `docs/STATUS.md`, and resume.

A rejected ADR stays in the tree with Status `Rejected`; the reasoning is worth as much
as the decision.

An accepted replacement may mark an earlier proposal `Superseded`. Preserve its
original body and add a dated disposition so historical instructions cannot be
mistaken for the active plan. Superseding a proposal does not retroactively accept
its implementation or mark its milestone complete. Do not reopen a decision the
researcher has already accepted.

## Decision index

[`0010-branch-atlas-and-bounded-f.md`](0010-branch-atlas-and-bounded-f.md) is the
accepted basis for the active R0–R8 plan in DESIGN §23. The old cut implementation
and its evidence remain available; completing it is not a prerequisite for the
new production path.

| ADR | Recorded status | Scope and current disposition |
| --- | --- | --- |
| [0001 — Bracketed g=0 split-point polishing](0001-bracketed-g-zero-split-point-polishing.md) | Decided | Old surface extraction; preserve numerical and regression evidence. |
| [0002 — Axis-regular harmonic continuation](0002-axis-regular-harmonic-continuation.md) | Decided | Field representation and axis convention; retained subject to active DESIGN. |
| [0003 — Between-sample contacts](0003-report-stepped-over-multiway-contacts.md) | Decided | Old transition sampling; preserve the physical contact and additivity checks. |
| [0004 — Open cut endpoints at EDGE](0004-snap-open-companion-cut-endpoints-to-edge.md) | Accepted | Old constrained-cut implementation. |
| [0005 — Critical-polyline zigzags](0005-critical-polyline-sub-resolution-zigzags.md) | Accepted | Old critical-curve geometry and sampling diagnostics. |
| [0006 — Event-junction surface resolution](0006-event-junction-surface-resolution.md) | Accepted | Old event cuts; six-well synthetic evidence remains useful. |
| [0007 — Below-b fold certification](0007-below-b-fold-event-certification.md) | Superseded by 0010 | Historical proposal; legacy implementation not retroactively accepted. |
| [0008 — Degenerate cut endpoints](0008-degenerate-endpoint-events.md) | Superseded by 0010 | Historical proposal; legacy implementation not retroactively accepted. |
| [0009 — Matrix resolved-fraction shortfall](0009-matrix-resolved-fraction-shortfall.md) | Superseded by 0010 | Keep 32/10/77/1 outcomes and the failed 35% result as historical evidence. |
| [0010 — Branch atlas and bounded f](0010-branch-atlas-and-bounded-f.md) | Accepted | Active metric, representation, uncertainty, acceptance, runtime, and migration decisions. |
