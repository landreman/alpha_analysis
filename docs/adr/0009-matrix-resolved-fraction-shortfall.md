# ADR 0009: The 95% matrix threshold versus the explicit-terminal residual

- **Status:** Proposed
- **Date:** 2026-08-31
- **Milestone:** 10.3
- **Design sections:** §23 milestone 10.3 (goal and acceptance), §21.2, §21.3;
  ADRs 0006–0008

## Context

Milestone 10.3's acceptance requires that "at least 95% of the cases in the
matrix have resolved cuts into sheets or have no transitions; only 5% or fewer
of the cases are unresolved" — at most 6 unresolved cases of 120. The
coordinator implemented on this branch dispatches every recorded failure class
to bounded, matrix-wide remediation ladders (per-sample period caps to 1024,
source budgets to full, localization bisections to 320, root-scan factors to
4, four local refinement rounds, three background levels), and the measured
matrix reaches:

PLACEHOLDER-NUMBERS (filled from `docs/validation/milestone10.3-real-equilibria.json`)

Every remaining unresolved case terminates with a physically meaningful,
machine-readable reason — the first acceptance sentence holds — but the
resolved fraction is below 95%. The residual classes, each already at its
ladder bound:

- genuine field-period cap ceilings (`MAX_PERIODS` persisting at 1024, the
  behaviour `docs/STATUS.md` has recorded for structured DMercFail
  \(\lambda_n=0.9\) and d23p4 \(\lambda_n=0.95\) since milestone 9);
- localized count changes certifying as neither equal-height nor fold at
  root-scan factor 4 (PCA \(\lambda_n=0.1\));
- junction complexes whose companion slits do not separate the surface, or
  whose child-3 return curves straddle sheets, at every background level
  (TURBO, PCA near curve crossings);
- boundary-exit critical points beyond every extracted component's boundary
  allowance (TURBO \(\lambda_n=0.1\)).

Each further remediation this branch invented (fold certification, ADR 0007;
degenerate-endpoint events, ADR 0008; branch-slit bank duplication;
scan-resolution escalation; the outermost background ladder) moved 1–5 cases;
the remaining tail is research-scale physics (multiway junction resolution,
event-geometry certifications beyond the two implemented kinds), not a budget
that can simply be increased. The only in-reach paths to the number would
loosen checks — counting explicit-event terminals as cuts, widening the strip
or side-assignment requirements, or cutting through uncertified events — all
of which `AGENTS.md` and §21.2 forbid.

## Options

1. **Accept the measured fraction and restate the milestone threshold** — the
   researcher re-baselines the acceptance to the achieved value (or to
   "every case terminates explicitly with per-class counts", which is met).
   Cost: the 95% number in §23 changes; nothing else does.
2. **Exclude physically-terminal classes from the denominator** — count a
   genuine cap ceiling and a genuinely unrepresentable strip as
   "resolved-with-a-physical-reason" rather than "unresolved". The
   acceptance's first sentence already names these as legitimate ends. Cost:
   the metric no longer measures cut coverage alone, and the boundary
   between "physical" and "resolution" terminals is a judgment the design
   would need to define.
3. **Fund the residual classes as follow-on milestone work** — junction
   complex resolution (crossing GAMMA_MAX curves as first-class §5.4
   structures), event-geometry certification beyond equal-height/fold, and
   component-topology reconciliation near boundary exits. Cost: milestone
   10.3 stays open until then.

## Decision

Left for the researcher. The branch is opened as a draft naming this ADR; no
tolerance, bound, or check was loosened to move the number.

## Consequences

- `test_matrix_report_meets_milestone_10_3_acceptance` encodes the 95%
  threshold and is red on the measured matrix; it stays red rather than being
  skipped or weakened (`AGENTS.md`).
- The milestone row in `docs/STATUS.md` stays unchecked until the researcher
  decides.
- The matrix report records every residual case's terminal reasons and the
  full remediation attempt log, so any of the three options can proceed from
  the same data.
