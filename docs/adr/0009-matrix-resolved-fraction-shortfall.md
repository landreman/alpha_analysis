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

- 42/120 cases (35.0%) resolved or no-transitions (32 `no_transitions`,
  10 `resolved`), against the ≥114/120 the criterion requires;
- 77 cases unresolved with explicit terminals; 1 case crashed
  (`3:0.5:gmsh:marching_tetrahedra`, the escaped "lacks sides"
  `ConstrainedCutError` now fixed and regression-tested — see the
  validation report);
- 112.6 h of recorded case compute
  (`docs/validation/milestone10.3-real-equilibria.md` has the full grid).

Every remaining unresolved case terminates with a physically meaningful,
machine-readable reason — the first acceptance sentence holds for 119/120 —
but the resolved fraction is far below 95%. The residual classes, each
already at its ladder bound:

- per-case wall-budget exhaustion (46 cases at the recorded 7200 s guard:
  PCA and n3are at every \(\lambda_n \ge 0.5\), TURBO \(\ge 0.8\), d23p4
  structured at 0.5). Twelve of these were re-verified on an uncontended
  second pass and re-capped at 7200 s of near-pure compute, so they are
  genuine case costs under the full remediation ladders, not scheduling
  artifacts; the researcher has since capped acceptable per-case compute at
  10 minutes, which these cases exceed by more than an order of magnitude;
- genuine field-period cap ceilings (20 cases: `MAX_PERIODS` persisting at
  1024 for DMercFail \(\lambda_n \ge 0.9\) and d23p4 \(\lambda_n \ge 0.8\)
  on all four backend×extractor combos — the long-well behaviour
  `docs/STATUS.md` has recorded since milestone 9; correctly
  background-exempt, since no background refinement changes it);
- localized count changes certifying as neither equal-height (ADR 0003) nor
  below-b fold (ADR 0007) at root-scan factor 4, plus empty event intervals
  at full source budget (8 cases, PCA \(\lambda_n=0.1\));
- a critical point 5.15e-02 from the nearest tagged edge of its extracted
  component after four local-refinement rounds (4 cases, TURBO
  \(\lambda_n=0.1\)) — a component-topology limit, not resolution;
- the ADR 0001 thin-tube extraction limit (2 cases, DMercFail
  \(\lambda_n=0.05\) gmsh) and non-separating companion slits /
  curve-geometry demotions at background level 2 (2 cases).

The cap-ceiling arithmetic alone settles the threshold's feasibility for
this equilibrium set: the 20 `max_periods` cases are 16.7% of the matrix —
more than three times the entire 5% unresolved allowance — and they are
documented physics, not a budget anyone can raise (the coordinator already
escalates 128→256→1024 and records every retry). Adding the certified
wall-budget cases, no coordinator that keeps the §21.2 prohibitions can
reach 95% on these five equilibria at these resolutions.

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
