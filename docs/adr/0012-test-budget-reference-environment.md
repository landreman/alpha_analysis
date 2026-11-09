# ADR 0012: Define the test-budget reference environment

- **Status:** Accepted
- **Date:** 2026-11-09
- **Milestone:** R4
- **Design sections:** §22.5, §23 R4, §28

## Context

R4's finite ordinary action-bin implementation, named scientific tests, mutations,
diagnostics and 30-case validation are in draft
[PR #32](https://github.com/landreman/alpha_analysis/pull/32). Its code/test revision
`470e21aed6744624f85f18e421b613564c6a8104` passes every GitHub Tests job. At proposal,
completion remained unchecked because the test-time reference was unspecified.
This is a test acceptance question, not a request to reapprove ADR 0010's algorithm.

DESIGN §22.5 requires:

> fast: `make test` — under 2 minutes; no single fast test over about 20 s
>
> full: `make test-full` — under 5 minutes; no single slow test over about 90 s

It did not identify a reference machine or state whether the two-minute threshold
must hold on every GitHub-hosted runner as well as the required local environment.
AGENTS requires a green `make check` within the budget and separately requires a
green GitHub Tests workflow. Choosing a local-only or an all-CI interpretation
would change when R4 can be marked complete.

The clean local environment is the specified Python 3.10.5 `.venv` on an Apple M1
MacBook Pro (8 CPU cores, 16 GB RAM), with three pytest workers and numerical library
threads limited to one. Measured commands on the implementation revision passed:

- `make check`: 245 tests, 98.90 s, including lint.
- `make test`: 244 tests, 74.24 s.

The same revision produced these passing CI results. Both workflows use the same
three-worker Makefile commands and numerical thread limits:

| Tier | [Run A](https://github.com/landreman/alpha_analysis/actions/runs/42686708906) | [Run B](https://github.com/landreman/alpha_analysis/actions/runs/42686696692) |
| --- | ---: | ---: |
| Fast, Python 3.10 | 174.71 s | 157.43 s |
| Fast, Python 3.11 | 171.26 s | 166.51 s |
| Fast, Python 3.12 | 89.76 s | 189.09 s |
| Full, Python 3.12 | 138.81 s | 203.54 s |

The Python 3.12 coarse d23p4 contour case alone took 10.12 versus 21.20 s; its fine
case took 9.36 versus 19.81 s. Some other jobs recorded 35–39 s individual cases.
The cause of this timing variability has not been established. The observations
are not evidence that all standard CI runners satisfy §22.5.

Before stopping, redundant global census work was removed from the local-adapter
tests while retaining both real fields, both grids and all owned root/action
assertions. Actual global atlas inputs remain in the full 30-case matrix and are
replayed by fast CI. The existing two-resolution contour regression was split into
per-resolution cases, preserving all scientific inputs and assertions. No slow,
skip or xfail marker, relaxed tolerance or removed resolution was used to reach
green. Both required physics mutations were observed red and reverted.

A local cProfile of the fine d23p4 case attributes 18.78 of its 19.25 instrumented
query seconds to root-pattern certification (1,934 certificate boxes and 164,377
Fourier evaluations). This suggests further optimization work, but does not define
which environment is authoritative for the acceptance deadline. R4's separate
real-model bin calculations take less than 0.4 s per mesh at the recorded sizes.

## Options

1. **Use a declared local reference environment.** Apply the two- and five-minute
   test budgets to the specified M1/clean-venv environment and worker/thread
   configuration. Keep GitHub Tests correctness mandatory and report CI timing
   separately. This permits completion with the current local measurements, but
   does not promise that every GitHub-hosted fast run finishes within two minutes.

2. **Require the budgets in every supported CI job.** Specify the runner/resource
   reference and worker/thread configuration, and continue profiling/optimization
   before R4 completion. A dedicated or otherwise controlled runner may be needed
   if resource variability is decisive. This retains the strict CI interpretation;
   current fast CI timings do not pass it. Scientific checks and tolerances must
   remain intact, and any production optimization needs its scientific validation.

## Decision

Option 1 accepted: Use the local M1 environment.

## Consequences

DESIGN §22.5 identifies the local reference environment and keeps GitHub Tests
correctness mandatory while reporting CI timings separately. The recorded local
fast/full measurements satisfy the accepted timing gate. R4 completion can now
follow the normal final-revision Tests and ready/review workflow.

The physical metric, five-file/six-level cohort, field-error obligations, full f
width 0.01, 29/30 slice criterion, and final 600-second benchmark guards are
unchanged. Reusing saved R3.5 root certificates remains explicit and does not
constitute final cold-start performance evidence. Historical evidence and prior
milestone dispositions are preserved.
