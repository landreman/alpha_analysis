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
4. When the researcher accepts it, set Status to `Accepted`, add a line to the
   "Accepted deviations" list in `docs/STATUS.md`, and resume.

A rejected ADR stays in the tree with Status `Rejected`; the reasoning is worth as much
as the decision.
