---
name: milestone
description: Implement one numbered milestone from docs/DESIGN.md §23 end to end — branch, test-first implementation, verification, STATUS.md update, and PR. Use whenever the user asks to work on a milestone, implement the next milestone, continue the plan, or names a milestone number like 3 or 14.
---

# Implement one milestone

The user will either name a milestone (`3`) or say "next". If they say next, take the
lowest-numbered unchecked row in `docs/STATUS.md`.

Work on exactly one milestone. Follow `AGENTS.md` for environment, commands, definition
of done, test speed, and STOP conditions.

## 1. Orient

Read the milestone's entry in `docs/DESIGN.md` §23, then the sections of
`docs/DESIGN.md` it depends on. §23 gives you goal, changes, and acceptance criteria;
the acceptance criteria are usually one line and the physics behind them is elsewhere in
the document. Read that too — the whole point of the design document is that you do not
have to guess the physics.

Also read the "Notes for the next milestone" section of `docs/STATUS.md`.

If the previous milestone's row in `docs/STATUS.md` is unchecked, or its GitHub Actions
`Tests` workflow was not green, stop and say so. A failed or unavailable `Claude Code
Review` workflow does not block the next milestone.

The pull request you generate will be reviewed by an AI agent following the instructions
in `.claude/commands/review-milestone.md`. So, read that document, and in planning your
implementation, anticipate any objections the review agent might raise, to ensure that
your work will pass review.

## 2. Branch

```bash
git checkout main && git pull
git checkout -b milestone/<number>-<slug>
```

## 3. Write the tests first

Derive them from `docs/DESIGN.md`, not from the implementation you are about to write.
§20 lists what is worth testing for each area; the acceptance criteria in §23 say which
of those this milestone owes. Prefer the tests §20 calls out for this area over tests
you invent, and prefer a test that pins an invariant (§25) or an analytic value on a
synthetic field (§20.1) over one that pins whatever the code happens to produce.

Then confirm the tests fail for the right reason. Run them against the unimplemented
stub and check that the failure is the physics you are about to add, not an import
error or a typo in a fixture.

## 4. Implement

The minimum that satisfies the milestone. Docstrings carry the design section number,
the equation, the conventions, and the units.

Do not refactor code the milestone does not touch. Existing public functions and CLI
entry points stay backward compatible (`docs/DESIGN.md` §2).

## 5. Verify

```bash
make check
```

Then verify the tests can fail. Pick the one or two mutations that matter for this
milestone — the ones that would produce a plausible wrong answer rather than a crash.
Good candidates: flip a sign, drop a term from a derivative, replace an analytic
derivative with a first-order finite difference, skip the endpoint regularization in the
bounce quadrature, treat a failed trace as unreachable, merge two surface components
that should stay separate. Apply each, confirm the suite goes red and that the test that
went red is the one you expected, revert. Record which mutations you checked and which
test caught each; this goes in the PR body.

Re-run `make smoke` if you touched packaging or added an optional dependency.

Then check the budget (`docs/DESIGN.md` §22.5, `AGENTS.md` "Test speed"): `make test`
under 2 minutes, no single fast test over about 20 s in the durations report it prints;
`make test-full` under 5 minutes, no single `slow` test over about 90 s. If your new
tests blow it, make them cheaper before you consider marking anything `slow` — lower the
mesh or Fourier resolution, shrink the pitch or field-line grid, move an expensive
fixture to module scope, cache the loaded field. Copy the durations tail into the PR
body.

A `slow` test must not be the only live evidence for an acceptance criterion. If you
mark one, the same physics keeps a fast test on the production code path that fails
under a mutation. A fast test that only checks a shape, a schema, or that nothing raised
does not count.

This repository does not aim for exhaustive coverage, so do not pad the suite. Adding
tests that cost wall-clock without being able to fail is worse than adding none.

## 6. Run the GitHub Actions Tests workflow

Push the branch to GitHub and let the `Tests` workflow run. Check that it passes. If it
fails, fix the issue, push again, and iterate until the `Tests` workflow is green. The
optional `Claude Code Review` workflow is not part of this gate.

## 7. Record

Update `docs/STATUS.md`: mark the milestone's row, fill in the PR number, and add a line
under "Notes for the next milestone" for anything the next milestone needs to know.

If `docs/DESIGN.md` turned out to be wrong or stale about something you implemented, fix
that section in the same PR and say so in the body. Do not leave the design document
describing code that no longer exists.

## 8. Open the PR

```bash
gh pr create --fill --draft
```

Open it as a **draft**. `claude-code-review.yml` triggers only on `ready_for_review`
(and `reopened`) — not on `opened` or on every push — so a draft PR, and any commits you
push while it stays draft, does not consume a review. Mark it ready only when you want a
Claude Code review pass (step 9).

PR body must contain, in this order:

- milestone number and one-line summary
- each acceptance criterion, and the test that demonstrates it
- measured numbers
- mutations verified to turn the suite red, and which test caught each
- `make test` and `make test-full` wall-clock, and the slowest-test list; any test you
  newly marked `slow`, sped up, or deleted, and why
- open ADRs blocking merge, or "none"
- anything you were unsure about and want the reviewer to look at hardest

## 9. Address Claude review findings

The review only runs when the PR transitions to ready-for-review, so trigger it
explicitly once the PR is in the state you want reviewed, by first converting the
PR to draft if it is not already, and then running

```bash
gh pr ready <number>
```

If the review workflow is configured and runs, periodically check for the
`claude-review` workflow run. Address findings flagged as `blocking` or `should-fix`
when appropriate. A failed, unavailable, or unconfigured Claude review must not block
the milestone or the start of the next one; only the `Tests` workflow is required.
Before pushing more commits that you don't want reviewed immediately (e.g. you're still
iterating on the same round of fixes), convert the PR back to draft:

```bash
gh pr ready <number> --undo
```

Push your fixes, then mark it ready again (after converting it to a draft if it is not
already a draft) to trigger the next review pass.

If the claude review workflow suceeds and recommends "fix-first" instead of "merge", 
then iterate until the review recommends "merge". For items flagged as `note`, it is up to your
judgement whether to address them. If you disagree with a finding, write an ADR and name
it in the PR body.

Once any desired Claude review pass is complete, stop, leaving the PR marked ready for
review if appropriate. Do not merge. A clean `Tests` workflow permits the next
milestone regardless of Claude review status.

## If you hit a STOP condition

The STOP conditions are in `AGENTS.md`. Write the ADR in `docs/adr/` using
`docs/adr/template.md`, commit it, push the branch, open the PR as a draft with the ADR
named in the body, and end your turn. Do not choose an option and proceed. Relaxing a
tolerance, marking a test `xfail`, or narrowing its inputs to get to green is never the
answer.
