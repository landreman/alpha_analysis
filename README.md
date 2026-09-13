# alpha_analysis

Tools for magnetic-field and trapped-particle analysis. The connectivity project
aims to compute the source-weighted fraction \(f\) whose trapped wells can reach
the plasma edge through constant-action contours and permitted split/merge
transitions, as defined in [`docs/DESIGN.md` §3](docs/DESIGN.md#3-physical-definition-of-the-metric).

The accepted implementation plan now uses a root-labelled well atlas and lower
and upper bounds on \(f\). This is a **planning decision, not an implemented
end-to-end calculation**. Existing field evaluation, well tracing, mesh and
transition tools remain available. The legacy milestone-10.3 matrix reached
42/120 resolved-or-no-transition cases and failed its acceptance target; it is
retired without being marked complete.

- [Design and active milestones R0–R8, including R3.5 before R4](docs/DESIGN.md)
- [Current status — R0 is next; all redesign milestones are unchecked](docs/STATUS.md)
- [Accepted redesign decision, ADR 0010](docs/adr/0010-branch-atlas-and-bounded-f.md)
- [Development environment, commands and validation rules](AGENTS.md)
- [Historical milestone-10.3 results](docs/validation/milestone10.3-real-equilibria.md)
- [Archived implementation notes](docs/history/STATUS-pre-redesign.md)

PR #24 should not be merged as-is. Preserve its branch and validation evidence;
land the planning documentation separately on `main`, then use R0 to verify the
baseline and selectively salvage needed fixes. Existing public APIs and scientific
regression tests remain compatibility obligations.
