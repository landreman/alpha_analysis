---
description: Adversarial review of a milestone PR against docs/DESIGN.md
argument-hint: "[PR number, or blank for the current branch]"
allowed-tools: Bash(git diff:*), Bash(git log:*), Bash(gh pr:*), Bash(make*), Bash(pytest*), Bash(python*), Bash(black*), Read, Grep, Glob
---

Review $ARGUMENTS as an adversarial reviewer. Assume the implementation is plausible,
well-formatted, and passing CI, and that if it is wrong the tests are wrong too. Your job
is to find that, not to comment on style — black already did that.

Read the diff, then `docs/DESIGN.md`: §23 for this milestone's acceptance criteria, §21.2
for the things that must never happen, §25 for the validation checklist, §22.5 for the
test budget, and the physics sections the milestone actually implements. Read those
sections yourself; do not take the PR body's characterization of them.

The milestone's `docs/STATUS.md` row MUST be changed to `[x]` in its implementation PR
once the PR satisfies its definition of done. Here `[x]` means complete in the submitted
PR (or merged), not previously merged. Do not report that marker as a finding merely
because the PR is unmerged; report a finding only if the PR fails the documented
definition of done, or if an unrelated milestone's row is changed.

Calibration: this repository deliberately does not chase exhaustive coverage
(`docs/DESIGN.md` §22.5). Do not report a missing test unless it is needed for an
acceptance criterion in §23, an item in the §25 checklist, or an invariant the diff has
just made possible to violate. "Could also test X" is not a finding here.

Answer these, in order, each with a verdict and the evidence you checked:

**1. Does the test actually constrain the physics?**
For each new test: what implementation would pass it that is nevertheless wrong? If you
can construct one, that is a finding. Specifically check that the test would fail if a
sign were flipped, if a term were dropped from a derivative, if an analytic derivative
were replaced by a crude finite difference, if the endpoint regularization in the bounce
quadrature were skipped, if a well were counted twice or not at all, or if two surface
components that should stay separate were merged. Watch for tests that assert only a
shape, a status enum, a schema, or that nothing raised.

Watch in particular for a test whose expected values were produced by the code under
test rather than derived from `docs/DESIGN.md`, an analytic synthetic field (§20.1), or
an independent calculation. A checked-in golden array that nothing else pins is a
finding unless the PR says how it was generated and why that source is trustworthy.

The PR body claims certain mutations were verified — spot-check one yourself by applying
it and running the test.

**2. Do the equations in the code match `docs/DESIGN.md`?**
Term by term, for every formula in the diff: the action and bounce-time integrals (§4.2),
the surface measure and its axis-regular form (§4.3, §4.4), the parallel derivatives
(§5.1), the split/merge relation (§5.3), the Fourier evaluation and its derivatives
(§7.1), the quadrature weights (§12). Check signs, factors of 2, whether a Jacobian is
present, which coordinate each derivative is taken with respect to, and whether any term
present in the design is absent from the code. This is the highest-value part of the
review; spend the most time here.

Check units and conventions in the docstrings against §3.2 and §3.3, and check the
periodicity conventions (θ over 2π, ζ over one field period) at every place the code
wraps an angle.

**3. Is a failure being silently absorbed?**
`docs/DESIGN.md` §21.2 forbids: replacing a failed trace with zero action or zero weight;
interpreting a clipped well as passing; dropping `NaN` triangles without accounting for
their measure; treating a missing transition as no connection; capping a long trace and
assigning Θ = 0; merging disconnected surface components because they are close in
Euclidean coordinates.

Read every `except`, every `if not ...: continue`, every `np.nan_to_num`, every default
value that stands in for a result, and every place a status enum is produced but never
consumed. A status enum that is set and then ignored downstream is the same bug as never
setting it. This is a blocking category, and it is the one most likely to be hiding
behind code that passes CI.

Also check §19.2: no PyVista, Gmsh, or NetworkX object crossing into the numerical core,
and no new base dependency (§19.4) without an ADR.

**4. Are convergence and bound claims honest?**
Lighter than a full rate study — this repository does not require measured convergence
orders except where §23 asks for them explicitly. What it does require:

- A claim that something converges is backed by at least two resolutions with the result
  moving in the right direction, not by one run plus an assertion.
- Where the design promises bounds rather than a value (§13.4, §25), check that
  `f_lower ≤ f ≤ f_upper` is actually asserted and that unresolved weight is carried into
  the bounds rather than dropped.
- Any convergence dimension from §21.3 that this milestone touches is either reported or
  explicitly listed as not yet controlled. Silence is a finding.
- A shrinking residual is not a convergence result. Neither is a plot.

**5. Was an acceptance criterion quietly weakened?**
Diff tolerances, expected values, `xfail`/`skip` markers, test input ranges, and the CI
matrix against `main`. Any loosening without an accepted ADR in `docs/adr/` is a blocking
finding regardless of how reasonable the justification sounds. A tolerance that grew by a
factor of 10 with a comment explaining why is still a blocking finding without the ADR.

Also check §2: existing public functions and CLI entry points
(`plot_bounce_points`, `plot_J_invariant`, `plot_J_invariant_single_lambda`,
`compute_J_invariant`) still behave as before, and the existing tests were not edited to
accommodate the new code.

**6. Did the PR pay its test-time bill?**
`docs/DESIGN.md` §22.5 budgets `make test` under 2 minutes with no single fast test over
about 20 s, and `make test-full` under 5 minutes with no single `slow` test over about
90 s. Check the durations report the PR body should quote; if it is missing, that is a
finding on its own.

A new `slow` marker is a legitimate cost control only if the same physics keeps a live
fast test that goes through the production code path and fails under a mutation. Applying
the marker to a failing test, or using it to remove the only live evidence for an
acceptance criterion, is blocking.

Check that speeding a test up did not quietly cost coverage: a resolution reduction that
drops a mutation check, removes the second point of a two-resolution convergence
comparison, or widens an accuracy tolerance is a finding. A deleted test needs a stated
reason — a code change, an ADR, or a design change that made it irrelevant.

**7. Plausible pictures, undocumented decisions.**
`docs/DESIGN.md` §24 says to reject a pull request that produces plausible pictures but
lacks machine-checkable invariants. If this milestone adds a diagnostic plot (§17), check
that the geometry it draws is also asserted somewhere a test can fail on.

Then: anywhere the implementer resolved an ambiguity in `docs/DESIGN.md` without an ADR
in `docs/adr/` or a note in the PR body. §24 item 9 requires the deviation to be
recorded, not silently chosen. Check `docs/STATUS.md` too — a note the next milestone
needs and will not have is a finding.

Output as a table of findings: severity (blocking / should-fix / note), location, what is
wrong, and what to do about it. Then one line: **merge / fix first / needs ADR**.

If you find nothing blocking, say so plainly. Do not invent findings to seem thorough,
and do not soften a blocking finding into a suggestion.
