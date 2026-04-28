# physics-lint

A linter for trained neural PDE surrogates. Runs in CI, catches physics
violations on every model PR, and produces actionable SARIF reports for
GitHub code scanning.

```{toctree}
:maxdepth: 2

loading
rules/index
security
```

## Quick start

```bash
pip install physics-lint
physics-lint config init --pde heat > pyproject-snippet.toml
physics-lint check my_model.py --format text
```

For CI integration, see the [README GitHub Actions example](https://github.com/tyy0811/physics-lint#physics-lint).
