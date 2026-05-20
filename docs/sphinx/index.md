# physics-lint

A linter for trained neural PDE surrogates. Runs in CI, catches physics
violations on every model PR, and produces actionable SARIF reports for
GitHub code scanning.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
loading
```

```{toctree}
:maxdepth: 2
:caption: Reference

rules/index
case-studies/index
action
sarif-schema
```

```{toctree}
:maxdepth: 1
:caption: Project

stability
security
faq
contributing
```

## Quick start

```bash
pip install physics-lint
physics-lint check my_model.py --format text
```

For CI integration, see the [GitHub Action](action.md).

For the full rule catalog, see [Rules](rules/index.md).

For the published case studies showing physics-lint detecting real
issues on published checkpoints, see
[Case studies](case-studies/index.md).

## Source

physics-lint is open-source at
[github.com/tyy0811/physics-lint](https://github.com/tyy0811/physics-lint).
