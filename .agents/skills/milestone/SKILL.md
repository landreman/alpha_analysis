---
name: milestone
description: Implement one active milestone from docs/DESIGN.md §23 through tests, verification, STATUS.md and a draft PR. Use for an explicit milestone implementation or "next milestone" request; planning-only revisions do not start implementation.
---

# Implement an active milestone

Read `AGENTS.md`, `docs/STATUS.md` and the selected entry in `docs/DESIGN.md` §23.
The active sequence is **R0–R8**. For "next", select the first unchecked active
row whose listed dependencies are complete. Do not choose historical 10.3 or
11–18, and do not treat retired work as an unfinished prerequisite.

If the user names an old number, explain its replacement using §23's mapping.
Proceed with the matching active work when intent is clear; clarify only genuinely
ambiguous scope. A request to edit planning documents does not authorize implementing
all milestones or merging a PR.

## Orient and choose the baseline

Read the relevant physics/algorithm sections and the current STATUS notes. The
accepted `docs/adr/0010-branch-atlas-and-bounded-f.md` supersedes the old cut-first
roadmap. Its decision is already authorized; do not reopen its resolved STOP merely
because old ADRs or historical notes contain a contrary instruction.

Active dependencies require completed acceptance and green GitHub `Tests` on the
code being used. R0 is the explicit migration entry: no green 10.3 gate is required.
Follow DESIGN §2.1 when the working tree is on PR #24's draft branch. Prefer the
plan-bearing main baseline, preserve uncommitted work, and selectively port only
needed fixes with regressions. Do not merge #24 as-is or import its whole coordinator
as an implied dependency. The narrow obsolete-test policy migration is specified
in §2.1; unrelated test failures remain real failures.

Inspect `.claude/commands/review-milestone.md` for the scientific review criteria.
Optional Claude review availability does not determine readiness of dependencies.

## Work on one milestone

Use an isolated checkout/worktree when another task or user edits share the current
checkout. Otherwise branch from the verified plan-bearing baseline; default branch
name `codex/<milestone>-<slug>`. Do not blindly switch to main or pull over existing
work. Follow an explicit user branch instruction.

Derive a small set of tests from §23 acceptance and §20, using analytic or independent
expectations. First confirm the relevant tests fail for the intended missing behavior.
Then implement the milestone and preserve existing public APIs. Docstrings state
conventions, units, and the active design equation/section.

The physical metric remains §3. New geometry is a root-labelled local atlas; global
surface cutting is not a prerequisite. An action graph cannot permit changes of J
between transitions. Unknown links affect the whole reachable population, not just
local cell weight. Follow the numerical-bound scopes and no-silent-loss rules in §21.

## Verify and record

Use the clean `.venv` and run `make check`; run `make smoke` when packaging or optional
imports change. Preserve §22.5's fast/full budgets. Apply the one or two meaningful
physics mutations, watch the intended test go red, revert, and record the evidence.
Do not pad the suite with implementation-mirroring checks or hide failures with
markers, looser tolerances, or reduced scientific coverage.

Run the milestone's required bounded real-field experiments under AGENTS and §20.3.
The active matrix has 30 physical file/pitch cases. Backend/extractor comparisons
apply when legacy geometry is changed or supplies reference data. Matrix experiments
are not all rerun inside the five-minute test suite. Record field/source identities,
controls, error scope, failures and wall times; do not call a wide interval a success.

Update only the implemented active row in STATUS when its definition of done is met.
Add brief notes future work actually needs; put permanent decisions in DESIGN or ADRs.
No archived milestone is marked complete to smooth the migration.

## Pull request and CI

Within the user's authorized repository-work scope, push the implementation branch
and open/update a **draft** PR with:

- the active milestone, concrete behavior and baseline;
- each acceptance criterion and its named test;
- measured accuracy, runtime, unresolved limits and real-field provenance;
- mutations verified and the tests that caught them;
- fast/full durations and justified changes to existing tests;
- relevant ADRs and any remaining blockers.

Wait for GitHub `Tests`, fix failures and verify the final code revision. Do not
claim CI passed because an earlier revision passed. Never merge without user
instruction. Mark ready only after applicable gates and desired review are satisfied.

Claude review runs on ready-for-review events in this repository. If a review is
requested/available, address applicable findings; record disagreements with evidence.
Its absence or failure is not a dependency gate. Avoid repeatedly toggling readiness
merely to solicit an unchanged review.

## New STOP conditions

For a new ambiguity affecting physics, an unachievable acceptance requirement,
a new base dependency/core-boundary crossing, or a failure that could only be hidden,
follow AGENTS: record a proposed ADR and leave the implementation PR draft for the
researcher's decision. Distinguish this from executing an already accepted decision.
Do not use the approved redesign as permission to weaken unrelated invariants.
