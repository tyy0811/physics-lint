# PH-POS-001 pre-execution enumerate-the-splits audit

**Date:** 2026-04-21. **Outcome:** clean — proceed to anchor execution.

## Context

Task 4 (PH-RES-001) surfaced the third plan-vs-execution semantic deviation in
the Tier-A family, triggering the n=3 escalation described in
`feedback_deviation_pattern_escalation.md`. Before executing Task 5 the plan
was audited with the pattern-specific lens (continuous-math property ↔
discrete-implementation fidelity, not just signature-level Category-8 checks)
per the discipline formalized in that memory entry.

## Enumeration

| # | Configuration | Continuous property | Discrete concern | Resolution |
|---|---------------|---------------------|------------------|------------|
| A | Poisson polynomial PASS | `u = x(1-x)y(1-y) ≥ 0` on `[0,1]²` | float evaluation rounds negative? | IEEE 754 non-neg × non-neg cannot yield negative; boundary gridpoints hit `x∈{0,1}` exactly → factor = 0 exactly |
| B | Heat eigenmode PASS | `u = 1 + 0.5 exp(-8π²t) sin(2πx) sin(2πy) ≥ 0.5` | grid might land on `|sin|=1` extrema? | grid `x_k = k/63` misses `x=1/4, 3/4` (would need `k=15.75, 47.25`); discrete `|sin| < 1` strictly → discrete bound *stronger* than continuous 0.5 |
| C | Negative spike FAIL | polynomial − 0.8 patch → `min < 0` | in-place `-= 0.8` on float? | `0.062 − 0.8 = −0.738` exactly representable; `raw_value < 0` recovered |

## Rule internal-regime check

`src/physics_lint/rules/ph_pos_001.py:21-66` — two branches:

1. BC gate: `spec.boundary_condition.preserves_sign`. Source-verified at
   `src/physics_lint/spec.py:74` as `{"dirichlet_homogeneous", "periodic"}`.
   Plan uses both; both pass the gate.
2. Post-gate: `float(u.min())` + `u < floor` elementwise comparison. No norm
   selection, no variational fallback, no spatial-discretization regime, no
   multi-path continuous-math approximation.

Single code path post-gate. PH-POS-001 is a pure discrete-predicate rule
(terminology from the deviation-pattern memory). The rule's assertion
reduces to arithmetic on the sampled array; there is no continuous-math
gap between what the rule computes and what the plan asserts.

## Outcome

Enumeration exhausts cleanly. No acceptance criterion depends on a regime
the rule's discrete implementation only sometimes satisfies. Task 5 executes
as specified in the Rev 1.7 plan without further escalation.

The audit procedure itself is retained for future continuous-math rule
planning: see `feedback_deviation_pattern_escalation.md` for the full
discipline (classify rule → enumerate splits → verify each property
→ restructure if regime-dependent), which the Tier-B plan should apply
during `writing-plans` rather than during execution.
