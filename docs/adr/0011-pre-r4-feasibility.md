# ADR 0011: Insert a feasibility milestone before R4

- **Status:** Accepted — researcher-directed milestone insertion
- **Date:** 2026-09-12
- **Milestone:** R3.5
- **Design sections:** §8–§11, §23–§24, §26, §28
- **Amends:** ADR 0010's milestone sequence; retains its algorithm and final contract

## Context

The researcher merged [PR #29](https://github.com/landreman/alpha_analysis/pull/29)
and requested a new R3.5 milestone covering the feasibility work before R4. This
records that instruction; it is not an unresolved choice of physical algorithm.

R3 meets its original milestone requirements, which explicitly require reporting
query completion without setting a minimum rate. Its fixed 30-case probe found
13 cases without a seed and classified none of 17 seeded queries. At the finer
step, seven paths closed numerically but lacked a root-pattern certificate, one
reached the edge numerically without that certificate, and nine failed ordinary
root continuation. The separately targeted DMercFail edge witness remains valid
within its stated represented-field scope. See the preserved
[R3 report](../validation/r3-contour-matrix.md).

R2's [report](../validation/r2-atlas-matrix.md) records only 0/240 coarse and
3/960 fine cell multiplicity certificates. The current fixed-scan-node sign
restriction is stronger than continued-root regularity. Limited seed searches and
missing automatic local event discovery also prevent useful real comparisons.
These are concrete implementation issues; the evidence neither establishes nor
rules out the eventual R8 accuracy/runtime contract.

## Decision

Insert **R3.5 — Practical root certification and contour feasibility** after R3.
Make R3.5 a required dependency of R4. The implementation must improve bounded seed
coverage, certify moving roots and ordinary atlas corridors, and discover and
continue local generic events. DESIGN §23 defines named scientific tests and real
evidence gates; [the implementation brief](../plans/r3-5-feasibility.md) supplies
reproducers and handoff details.

Keep R0–R3 complete, preserve the original matrices and failure outcomes, and leave
R3.5 unchecked until its implementation and definition of done pass. Better seed
counts or another report of all-unknown queries cannot complete this milestone.

Retain the root-labelled atlas, independent oracle and finite-bin accessibility
strategy. Retain the physical metric, all uncertainty obligations, existing
tolerances, dependency boundaries and final accuracy/runtime targets. This
decision authorizes no numerical implementation as correct and grants no new
base dependency. Local generic event work moves forward into R3.5; global
transition preimages, cycles and uncertainty propagation remain R5 responsibilities.

## Consequences

The next active implementation task is R3.5, not R4. Planning documents and agent
instructions point to that dependency. Implementation should first reproduce the
specific failures and then demonstrate improvements at controlled resolution and
measured cost, using the full 30-case matrix and fixed original queries.

If the new real gates cannot be met, leave R3.5 incomplete and record the diagnosed
failure through the existing draft-PR/ADR process. Do not defer the same unexplained
feasibility issue to R8, narrow the benchmark cohort, or change the physics to
obtain a classification. Further strategy changes require a separate decision.
