# Installation

physics-lint is published on PyPI. Install with pip:

```bash
pip install physics-lint
```

## Python version

physics-lint requires **Python 3.10 or later**.

## Optional dependencies

For unstructured-mesh support (PhysicsNeMo MGN-style adapters, FE field
adapters), install the `mesh` extra:

```bash
pip install physics-lint[mesh]
```

## Pinning a version

For CI use, pin to a specific major version so micro-releases don't change
behavior unexpectedly:

```bash
pip install 'physics-lint==1.*'
```

For full reproducibility, pin exactly:

```bash
pip install physics-lint==1.0.0
```

## Virtualenv

A virtualenv is recommended. The project's dev workflow uses:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

See [Contributing](contributing.md) for the development install.
