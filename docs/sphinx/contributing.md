# Contributing

physics-lint welcomes external contributions: new rules, additional
case studies, documentation improvements, and CI integrations.

## Development install

```bash
git clone https://github.com/tyy0811/physics-lint.git
cd physics-lint
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

The repository uses `pre-commit` for ruff + codespell:

```bash
pre-commit install
```

The pre-commit hook runs inside the activated venv; `python` may be
shadowed by a shell function in some environments, so prefer
`.venv/bin/python` directly when troubleshooting.

## Running tests

```bash
.venv/bin/pytest --import-mode=importlib -v
```

## Running the linter on this repo (dogfooding)

```bash
.venv/bin/python -m physics_lint check tests/fixtures/good_adapter.py --format text
```

The dogfood corpus is at `dogfood/laplace_uq_bench/`; six trained
surrogates plus baselines exercise the rule set.

## Adding a new rule

A new rule requires three artifacts:

1. **Rule implementation** at `src/physics_lint/rules/ph_<cat>_<nnn>.py`,
   following the registry pattern. Read existing rules for examples —
   `PH-SYM-001` (continuous symmetry) and `PH-CON-001` (mass
   conservation) are good starting points.

2. **External-validation directory** at
   `external_validation/PH-<CAT>-<NNN>/` containing:

   - `README.md` with a **mandatory** `## Rule reference` section
     (user-facing) and a separate F1/F2/F3-style validation stack below
     it. The docs build fails loudly if the canonical section is
     missing.
   - `CITATION.md` with the function-labeled citation stack (F1
     mathematical legitimacy, F2 correctness fixture, F3 real-world
     anchor).
   - One or more `test_*.py` files exercising the rule on calibrated
     fixtures.

3. **Harness fixture** at `external_validation/_harness/` if the rule
   needs a new field type or analytical solution; otherwise the
   existing harness covers it.

## Adding a case study

Case studies live under `external_validation/_rollout_anchors/`. Each
case study has its own subdirectory with a `README.md` whose canonical
heading is `## Case study reference`. The docs build emits a page from
the canonical section.

## Pull requests

- Open against `master`.
- Pre-commit hooks must pass.
- Test suite must pass.
- New rules require the documentation contract above (canonical
  section + citation stack + fixtures).
- Squash merging is the default.

## Codex review

Every major change (new rule, new case study, new public surface)
benefits from a Codex adversarial-review pass before merge.
Maintainers run this; contributors can request a Codex review on a PR
by tagging the issue.
