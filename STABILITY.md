# API Stability Policy

physics-lint follows [Semantic Versioning](https://semver.org/). This document
defines what is covered by that commitment as of **v1.0.0**.

## Public surface (covered by semver)

- **CLI commands and flags.** `physics-lint check`, `rules`, `config`,
  `self-test` and their documented options (see `physics-lint --help`).
- **Exit codes.** `0` all error-severity rules pass · `1` an error-severity
  rule failed · `2` invalid config/usage · `3` model load failed.
- **Rule IDs.** The `PH-<CATEGORY>-<NNN>` identifiers and their emission
  semantics (the `status` values a rule can return, the meaning of its
  `raw_value`).
- **SARIF output — scalar schema.** The result schema emitted by
  `--format sarif`: one result per non-PASS rule, `ruleId`, `level`, and the
  `properties` block (`raw_value`, `violation_ratio`, `mode`, `reason`).

## Not public (may change in any release)

- Internal rule implementations (`src/physics_lint/rules/ph_*.py` bodies).
- Private helpers, any name prefixed `_`, and rule-specific keyword arguments
  not exposed through the CLI.
- The rollout-anchor harness (`external_validation/_rollout_anchors/_harness/`)
  and everything under `external_validation/`.

## Deprecation policy

- A breaking change to the public surface gets **one minor-version notice**
  before it lands.
- A removed rule continues to emit `SKIPPED` (with a reason) for one minor
  version before its ID disappears.

## Rules deferred to v1.1

`PH-SYM-004` (translation equivariance) and `PH-NUM-001` (quadrature
convergence) ship in v1.0.0 as `SKIPPED`-with-reason. Each needs a design
pass before implementation; both are scheduled for v1.1. Their activation
in v1.1 is an additive change (a minor-version bump) — a rule that emitted
`SKIPPED` beginning to emit a real verdict does not break the contract above.
