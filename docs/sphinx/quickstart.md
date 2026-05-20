# Quick start

After [installation](installation.md), physics-lint provides three command
forms.

## Lint a model dump

The simplest form: lint a `.npz` file containing model predictions.

```bash
physics-lint check pred.npz --format text
```

Output is human-readable; each rule that fired is listed with its verdict
(`PASS`, `APPROXIMATE`, `FAIL`, `SKIP`), raw value, and a reason.

## Lint via an adapter

Adapters give physics-lint access to the model function (not just its
outputs), enabling rules that need to query the model at multiple points
(symmetry checks, boundary-condition queries). Write a
`physics_lint_adapter.py` next to your model code — see
[Loading models](loading.md) for the contract.

Then:

```bash
physics-lint check physics_lint_adapter.py --format text
```

## CI-style SARIF output

For CI integration, emit SARIF instead of text:

```bash
physics-lint check model.py \
    --format sarif \
    --category physics-lint-run \
    --output physics-lint.sarif
```

The SARIF file uploads to GitHub's Security tab via
[`github/codeql-action/upload-sarif@v4`](https://github.com/github/codeql-action).
See the [GitHub Action](action.md) for a drop-in workflow that wraps these
steps, and the [SARIF schema reference](sarif-schema.md) for the field
contract.

## Configuration

physics-lint reads configuration from `pyproject.toml` under
`[tool.physics-lint]`. Generate a starting config for a heat-equation
problem:

```bash
physics-lint config init --pde heat > pyproject-snippet.toml
```

Other supported `--pde` values: `wave`, `darcy`, `navier-stokes-incompressible`.
Run `physics-lint config init --help` for the full list.
