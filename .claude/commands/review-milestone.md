---
description: Review an active milestone PR against the current design and its scientific invariants
argument-hint: "[PR number, or blank for the current branch]"
allowed-tools: Bash(git diff:*), Bash(git log:*), Bash(gh pr:*), Bash(make*), Bash(pytest*), Bash(python*), Bash(black*), Read, Grep, Glob
---

Review $ARGUMENTS against `docs/DESIGN.md`, active `docs/STATUS.md`, AGENTS and
accepted ADR 0010. The active queue is **R0–R8**. Old 10.3 was retired without
passing; 11–18 were replaced. Historical documents are evidence, not current
instructions. Planning-only changes do not claim any implementation milestone done.

Read §23 acceptance, §3–5 physics, §21.2 prohibitions, §22.5 test budgets and §25
checklist yourself. Do not rely on the PR body's characterization. `[x]` in an active
milestone's implementation PR means its acceptance and definition of done are met
in that PR; it need not already be merged. Report premature completion, not the
mere fact that the marker appears in an unmerged PR.

## 1. Does the change solve the active milestone?

Check the listed dependencies and baseline. R0 may perform the explicit §2.1
migration; it does not depend on a green retired 10.3. PR #24's failed draft is
not a mandatory code dependency. Check that a planning/evidence PR does not import
its implementation diff accidentally.

The obsolete 95%-cut test may become the approved historical census/provenance
regression. Verify the saved crash and failure outcomes remain visible. This is
not permission to delete physics checks, change raw evidence, lower the old
constant, or add `skip`/`xfail`. Do not object to the migration solely because an
obsolete acceptance statement changed under an already accepted ADR.

## 2. Would the tests catch plausible wrong physics?

For each material claim, identify the independent expected value or invariant and
the mutation it detects. Check signs, omitted terms, action/time regularization,
source argument rho rather than s, pitch Jacobian b^-2, periodic lifts, first-root
selection and unique well ownership. Spot-check claimed mutations where warranted.

The repository wants a small effective suite, not exhaustive coverage. Report a
missing test only when it constrains an acceptance criterion, new invariant or
credible failure path. Schema-only checks do not establish a scientific result.

## 3. Are branch identity and accessibility correct?

Check local charts with overlapping well branches, disconnected coverage, periodic
and radial matching, and no silent loss of hidden barriers. A below-b extrema fold
can preserve a physical well; replacing its count veto requires a barrier/root
certificate rather than simply ignoring the change. Height fields are locally
smooth, not global labels through degeneracies.

Ordinary paths preserve J. Transitions preserve the same physical event parameter
and visit all allowed ports at their new actions. A connected sheet or Reeb graph
does not imply accessibility: the no-transition A=s annulus must remain inaccessible
in its interior. Check constant port maps, self-transitions and noncontracting cycles.

## 4. Are bounds and error scope honest?

Check model enclosure versus field enclosure versus numerical estimate versus
statistical confidence interval. A narrow PL result alone is not an R8 success.
Include field/root/coverage, A/K, denominator, support and outer-integration errors.
An extrema solver or n-versus-2n quadrature estimate is not automatically a bound.

Unknown links affect the entire population whose reachability depends on them.
A tiny event/core/long-well mass alone is not a global f-error bound. Check the
upper calculation includes missing branches, unknown edge incidence and permitted
connections; the lower uses only justified paths. Ensure wildcard influences and
chart weights are not double counted.

Validate the total-population identity's dense-line assumption or closed-line
handling. Use total upper minus covered lower, not uncontrolled subtraction.
For gap sampling include the whole global ambiguity set and the b^-2 weight;
failed queries are unknown, not omitted or assigned zero.

## 5. Are failure, compatibility and dependency rules preserved?

Inspect exception handlers, early continues, defaults and statuses that callers
might ignore. No capped/failed well becomes passing, zero weight or Theta=0.
No coordinate-near branches merge without identity. Critical K is not fabricated.

Existing public APIs and relevant legacy scientific regressions stay compatible.
If legacy meshes/cuts change, require their backend/extractor checks. A new atlas
need not pretend to have four legacy backends. Enforce §19.2 array/integer core
boundaries and ADR approval for a new base dependency.

## 6. Do evidence and performance meet the claim?

Check the exact field/source/revision/controls, all case outcomes, numerical scope
and cold-start wall times. R8 means 29/30 physical slices plus all five full f
intervals meeting DESIGN's accuracy and runtime contract, not just a finite result
or an equal-weight count of duplicated backends. Exploratory appendix estimates
and proposed test names are not completed evidence.

Check fast/full test durations against §22.5, and meaningful fast coverage for
scientific claims with slow tests. Do not require full real matrices inside unit
CI. A failed test is not made acceptable by marking it slow. Approved policy
migration is narrow; every other changed tolerance/input/test needs justification.

## 7. Are the plan and diagnostics maintained?

New geometric objects need diagnostic plots plus numerical invariants; a plot or
small residual alone is not acceptance. Ensure STATUS reflects only completed
active work and gives concise necessary handoff notes. New ambiguities require
ADRs; already accepted decisions do not require another approval cycle.

For a documentation-only PR, inspect link/queue/contract consistency and historical
preservation; do not demand fabricated scientific mutation or new runtime evidence.

Return a table of actionable findings: severity (blocking / should-fix / note),
location, problem, and remedy. Then give **merge / fix first / needs ADR** with
its scope. Do not invent findings, and do not call a failed CI run green.
