# FAQ

Common questions from CS01 / CS02 dogfooding and from external adopters
trying physics-lint for the first time.

## What is an "adapter"?

An adapter is a small Python file (`physics_lint_adapter.py` by
convention) that exposes your model function plus a domain spec to
physics-lint. Rules that need to query the model at multiple points
(e.g., symmetry checks, BC queries) operate via the adapter. See
[Loading models](loading.md) for the full contract.

## Can I lint a model without writing an adapter?

Yes — if you can produce a `.npz` dump of model predictions, you can
lint that directly:

```bash
physics-lint check pred.npz --format text
```

The dump-mode rule set is narrower (no symmetry, no BC re-query), but
covers conservation, positivity, and maximum-principle checks.
Adapters are required for the full rule set.

## What does APPROXIMATE vs FAIL mean?

Each rule has a calibrated noise-floor band. A model whose emitted
quantity is below the band passes (`PASS`); above the band but within
a tolerable factor is `APPROXIMATE` (warning); above the tolerable
factor is `FAIL` (error). Exact thresholds are per-rule and documented
on each rule's page.

## Why does PH-SYM-001 report SKIP on my model?

`SKIP` with a reason means a rule's preconditions aren't met for the
input. PH-SYM-001 checks rotational equivariance; it `SKIP`s if the
domain isn't rotationally symmetric or if the model doesn't expose an
operator the rule can query. See the
[PH-SYM-001 rule page](rules/PH-SYM-001.md) for the exact
preconditions.

## What does a SARIF result in the Security tab mean?

A non-PASS rule firing emits one SARIF result. GitHub code scanning
surfaces it in the repository's Security tab with the rule ID, the
location of the model artifact, and the `properties` block including
raw value, violation ratio, and a one-line reason. See
[SARIF schema](sarif-schema.md) for the field-by-field contract.

## How do I suppress a rule on a known-acceptable case?

Add the rule to the `[tool.physics-lint]` config's exclusion list in
`pyproject.toml`:

```toml
[tool.physics-lint]
exclude = ["PH-NUM-001"]  # SKIP-with-reason expected on FE fields
```

Or pass `--exclude PH-NUM-001` on the CLI. The rule still runs but its
output is downgraded to a `note` and does not fail the run.

## How do I add a new rule?

Three steps:

1. Implement the rule in `src/physics_lint/rules/ph_xxx_nnn.py`,
   following the registry pattern (see existing rules for examples).
2. Add the rule's user-facing description as a `## Rule reference`
   section in `external_validation/PH-XXX-NNN/README.md`. **This is
   mandatory** — the docs build fails loudly if the canonical section
   is missing.
3. Write validation tests under `external_validation/PH-XXX-NNN/` and a
   harness fixture under `external_validation/_harness/` if needed.

See [Contributing](contributing.md) for the full development workflow.

## What's the difference between `physics-lint` and `physics-lint[mesh]`?

`physics-lint[mesh]` adds the `scikit-fem` dependency for unstructured-
mesh field support (used by PhysicsNeMo MGN-style adapters and FE
adapters). Most particle / grid adapters work without it.

## Why "physics-lint" and not "physics-check" or "physics-test"?

Lint means "structural / mechanical static-ish check that catches
problems the test suite doesn't." physics-lint runs on a trained model
artifact in CI; it doesn't validate physics in a research sense (that
requires comparing to held-out data or analytical solutions). It checks
**properties** the model should obey — conservation, symmetry,
positivity, boundary behavior — against calibrated floors. The framing
is the same as `ruff` or `mypy`: cheap mechanical checks that catch
real bugs.
