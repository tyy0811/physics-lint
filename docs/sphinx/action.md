# GitHub Action

```{note}
The composite Action described on this page **lands in P3.2** (the
sibling plan to this docs site). The contract documented here is the
authored interface — once P3.2 ships and a release tag is cut,
`uses: tyy0811/physics-lint@v1.X.Y` will resolve. Until then this page
is the design reference; for current CI integration use the inline
`pip install physics-lint && physics-lint check` pattern from the
[README](https://github.com/tyy0811/physics-lint#hero-physics-lint-in-ci).
```

physics-lint ships a composite GitHub Action at the root of
[`tyy0811/physics-lint`](https://github.com/tyy0811/physics-lint) for
drop-in CI integration. Call it as:

```yaml
uses: tyy0811/physics-lint@v1.0.0
with:
  model: physics_lint_adapter.py
```

## Convention note

The dominant convention for major Actions is a separate `<name>-action`
repo (`astral-sh/ruff-action`, `astral-sh/setup-uv`,
`github/codeql-action`). physics-lint ships its Action **in-repo at
root** instead, which trades convention alignment for version-sync
automation: the Action tag and the installed `physics-lint` version are
locked by construction (`uses: …@v1.0.0` installs `physics-lint==1.0.0`).

## Minimal example

```yaml
name: physics-lint
on: [push, pull_request]

permissions:
  contents: read
  security-events: write

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: tyy0811/physics-lint@v1.0.0
        with:
          model: physics_lint_adapter.py
```

## Matrix example (multi-model)

For repositories with multiple model artifacts (e.g.,
[`tyy0811/laplace-uq-bench`](https://github.com/tyy0811/laplace-uq-bench)):

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        model:
          - { name: unet, path: models/unet_adapter.py }
          - { name: fno,  path: models/fno_adapter.py }
          - { name: ddpm, path: models/ddpm_pred.npz }
    steps:
      - uses: actions/checkout@v4
      - uses: tyy0811/physics-lint@v1.0.0
        with:
          model: ${{ matrix.model.path }}
          category: physics-lint-${{ matrix.model.name }}
          output: physics-lint-${{ matrix.model.name }}.sarif
```

## Inputs

| Input | Required | Default | Notes |
|---|---|---|---|
| `model` | yes | — | Path to adapter `.py` or `.npz` dump |
| `config` | no | `pyproject.toml` | Path to config TOML |
| `category` | no | `physics-lint` | SARIF category for code-scanning distinction |
| `output` | no | `physics-lint.sarif` | SARIF output path |
| `version` | no | derived from Action ref | physics-lint PyPI version. See below |
| `python-version` | no | `3.11` | Passed to `actions/setup-python` |
| `mesh` | no | `false` | When `true`, installs `physics-lint[mesh]` |
| `fail-on-error` | no | `true` | When `false`, the CLI's non-zero exit on error-severity rules is suppressed |
| `upload-sarif` | no | `true` | When `false`, skips the `github/codeql-action/upload-sarif` step |

## Version pinning

The `version` input default is derived from `github.action_ref`:

- If the caller pins a tag (e.g., `uses: tyy0811/physics-lint@v1.0.0`),
  the Action installs `physics-lint==1.0.0`. Tag and install are locked
  by construction.
- If the caller pins a branch or SHA, the Action falls back to the
  latest published `physics-lint` (no determinism is possible without an
  explicit input).
- Override at any time with `with: version: 1.0.1` (or any valid PyPI
  version specifier).

## Permissions

physics-lint does not configure workflow-level permissions itself. The
caller must set:

```yaml
permissions:
  contents: read
  security-events: write
```

`security-events: write` is required for the SARIF upload step. Without
it, the upload silently fails (or fails with a permissions error,
depending on the runner). See [security](security.md) for the full
threat model around adapter execution in CI.

## Advanced usage

For CLI flags not exposed as Action inputs (e.g., rule include / exclude,
custom severity overrides), install and run physics-lint directly
instead of using the Action:

```yaml
- run: |
    pip install 'physics-lint==1.*'
    physics-lint check models/fno.py \
      --rules PH-POS-*,PH-SYM-* \
      --format sarif \
      --output physics-lint.sarif

- uses: github/codeql-action/upload-sarif@v4
  with:
    sarif_file: physics-lint.sarif
```

The Action's input surface is intentionally small; expansion happens in
v2 if real usage justifies it.
