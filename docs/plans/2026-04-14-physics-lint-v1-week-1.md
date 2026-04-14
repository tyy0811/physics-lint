# physics-lint V1 — Week 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Week 1 slice of `physics-lint` v0.1.0.dev0 — importable Python package with working Field abstraction, pydantic DomainSpec, hybrid adapter+dump loader, norms module, residual/BC/positivity rules for Laplace and Poisson, analytical battery, hypothesis property tests, and calibrated `floors.toml` — all verified on the 6-job CI matrix.

**Architecture:** Package organized by concern: `field/` (ABC + GridField + CallableField), `spec.py` (pydantic DomainSpec hierarchy), `config.py` (merge path), `loader.py` (extension-dispatched hybrid adapter+dump), `norms.py`, `rules/` (lazy-discovered catalog with one file per rule), `analytical/` (battery), `data/floors.toml`. Rules read `(Field, DomainSpec) -> RuleResult`; never touch raw config. Pydantic v2 owns all validation; rules contain no defensive type checks.

**Tech Stack:** Python 3.10+, NumPy 1.26+, SciPy 1.11+, PyTorch 2.0+, pydantic 2.0+, typer 0.12+, hatchling, pytest + pytest-cov + hypothesis, ruff, Sphinx + MyST + furo, codespell, pre-commit. License Apache-2.0.

**Input spec:** `docs/design/2026-04-14-physics-lint-v1.md` (commit `daa07ee`).

**Scope for this plan:** Week 1 only. Weeks 2–4 (heat/wave + conservation + dogfood; symmetry + FE + broken-model toy; report + CLI + SARIF + release) will be planned separately as each approaches. The Week-1 deliverable is a working but incomplete library — Laplace/Poisson only, no time-dependent PDEs, no CLI yet, programmatic API only.

**Week 1 deliverable (what "done" means):**
1. PyPI namespace `physics-lint` registered under `tyy0811` with `0.0.0.dev0` placeholder.
2. `pip install -e .` works in a clean venv.
3. `pytest` exits zero: all unit tests pass, all hypothesis property tests pass at the `ci` profile, analytical battery for Laplace/Poisson passes at the calibrated floors.
4. `floors.toml` populated for Laplace/Poisson entries with the maximum-observed-across-three-environments multiplicative-tolerance discipline from the spec.
5. Programmatic API: user can import `physics_lint` and run `PH-RES-001/002/003`, `PH-BC-001/002`, `PH-POS-001/002` on a `GridField` or via the hybrid loader.
6. scikit-fem spike decision made and documented in a commit message; `MeshField` either stubbed or deferred to V2 with the relevant rule catalog rows marked conditional.
7. GitHub Actions CI matrix (six jobs) runs green on main.

**Working directory for this plan:** `/Users/zenith/Desktop/physics-lint` (already git-initialized with the design doc committed as `daa07ee`).

---

## Task 0: PyPI namespace verification and placeholder registration

**Files:**
- Create: `pyproject.toml` (minimal placeholder version)
- Create: `README.md` (one-paragraph stub)
- Create: `LICENSE` (Apache-2.0 text)
- Create: `src/physics_lint/__init__.py` (with `__version__ = "0.0.0.dev0"`)

**Rationale:** Claim the namespace before any other work. §9 and §20 of the spec mandate Day-1 AM registration to prevent a name squat.

- [ ] **Step 1: Verify `physics-lint` availability on PyPI**

This is a user-facing action, not an automatable one. The user must run:

```bash
pip index versions physics-lint 2>&1 | head -5
```

Expected: `ERROR: No matching distribution found for physics-lint` (namespace free) or a version listing. If a version listing comes back, pause and use a fallback name — see §0 of the spec for fallbacks: `physicslint`, `pde-lint`, `phylint`.

- [ ] **Step 2: Create `LICENSE` with Apache-2.0 text**

Fetch the canonical Apache-2.0 text:

```bash
curl -s https://www.apache.org/licenses/LICENSE-2.0.txt > LICENSE
head -5 LICENSE
```

Expected: `Apache License\n                           Version 2.0, January 2004\n`

- [ ] **Step 3: Create minimal `pyproject.toml` for placeholder registration**

```toml
[build-system]
requires = ["hatchling>=1.18"]
build-backend = "hatchling.build"

[project]
name = "physics-lint"
version = "0.0.0.dev0"
description = "Linter for trained neural PDE surrogates — placeholder, under development"
readme = "README.md"
license = { text = "Apache-2.0" }
authors = [{ name = "tyy0811" }]
requires-python = ">=3.10"
classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[project.urls]
Homepage = "https://github.com/tyy0811/physics-lint"
Documentation = "https://physics-lint.readthedocs.io"
Repository = "https://github.com/tyy0811/physics-lint"

[tool.hatch.build.targets.wheel]
packages = ["src/physics_lint"]
```

- [ ] **Step 4: Create `README.md` stub**

```markdown
# physics-lint

Linter for trained neural PDE surrogates. Under development; V1 target ships Q2 2026.

See `docs/design/2026-04-14-physics-lint-v1.md` for the full design document.
```

- [ ] **Step 5: Create `src/physics_lint/__init__.py`**

```python
"""physics-lint — linter for trained neural PDE surrogates.

See docs/design/2026-04-14-physics-lint-v1.md for the V1 design.
"""

__version__ = "0.0.0.dev0"
__all__ = ["__version__"]
```

- [ ] **Step 6: Build and upload the placeholder to PyPI**

The user must run this (requires PyPI credentials):

```bash
pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/physics_lint-0.0.0.dev0* --repository pypi
```

Expected: `View at:\nhttps://pypi.org/project/physics-lint/0.0.0.dev0/`

If the name is already taken, stop and use a fallback from §0 of the spec. The rest of this plan is name-agnostic — all file-level references are to the import name `physics_lint`, which stays the same regardless of fallback choice.

- [ ] **Step 7: Verify `pip install physics-lint==0.0.0.dev0` works from a clean venv**

```bash
python -m venv /tmp/physics-lint-smoke
/tmp/physics-lint-smoke/bin/pip install physics-lint==0.0.0.dev0
/tmp/physics-lint-smoke/bin/python -c "import physics_lint; print(physics_lint.__version__)"
rm -rf /tmp/physics-lint-smoke
```

Expected: `0.0.0.dev0`

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml README.md LICENSE src/physics_lint/__init__.py
git commit -m "chore: register physics-lint 0.0.0.dev0 placeholder on PyPI

Claims the namespace per design doc §0 and §20 Next Actions item 2.
Apache-2.0 license per §2.1.
Minimal pyproject.toml; dependencies added in later tasks."
```

---

## Task 1: Repository scaffolding — dev tooling, CI matrix, pre-commit

**Files:**
- Modify: `pyproject.toml` (add dependencies + dev tools + tool configs)
- Create: `.github/workflows/ci.yml`
- Create: `.pre-commit-config.yaml`
- Create: `.codespellrc`
- Create: `.gitignore`
- Create: `src/physics_lint/data/.gitkeep` (empty; populated later)

**Rationale:** Lock in the toolchain before writing code so every subsequent commit runs through ruff, codespell, and pytest via pre-commit and CI. Saves retrofitting later.

- [ ] **Step 1: Extend `pyproject.toml` with dependencies and tool configs**

Replace the `pyproject.toml` from Task 0 Step 3 with this full version:

```toml
[build-system]
requires = ["hatchling>=1.18"]
build-backend = "hatchling.build"

[project]
name = "physics-lint"
version = "0.0.0.dev0"
description = "Linter for trained neural PDE surrogates"
readme = "README.md"
license = { text = "Apache-2.0" }
authors = [{ name = "tyy0811" }]
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "torch>=2.0",
    "pydantic>=2.0",
    "typer>=0.12",
    "rich>=13.0",
    "tomli>=2.0; python_version<'3.11'",
]
classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[project.optional-dependencies]
mesh = ["scikit-fem>=10"]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "hypothesis>=6.100",
    "ruff>=0.5",
    "codespell>=2.2",
    "pre-commit>=3.5",
    "sphinx>=7.0",
    "myst-parser>=3.0",
    "furo>=2024.5",
]

[project.urls]
Homepage = "https://github.com/tyy0811/physics-lint"
Documentation = "https://physics-lint.readthedocs.io"
Repository = "https://github.com/tyy0811/physics-lint"

[tool.hatch.build.targets.wheel]
packages = ["src/physics_lint"]

[tool.hatch.build.targets.wheel.force-include]
"src/physics_lint/data" = "physics_lint/data"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "N", "SIM", "RET", "RUF"]
ignore = ["E501"]  # line-length handled by formatter

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["B", "N802"]  # allow non-snake-case test names (test_PH_RES_001)

[tool.ruff.format]
quote-style = "double"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers --hypothesis-show-statistics"

[tool.coverage.run]
source = ["physics_lint"]
branch = true

[tool.coverage.report]
fail_under = 85
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "\\.\\.\\.",
]

[tool.codespell]
skip = "*.lock,*.pyc,.git,dist,build,*.egg-info,docs/_build,src/physics_lint/data/floors.toml"
ignore-words-list = "bochner,fosls,riesz,sobolev,dirichlet,neumann,laplace,poisson,helmholtz,ernst,fornberg,trefethen,jekel,hansen,gruver,brandstetter,virkkunen,bachmayr,dahmen,oster,qiu,kovachki,welling,worrall,helwig,ot,cfm,ddpm,dps,fno,pinn,pino,pde,pdes,nd,ue,nin"
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.coverage
.coverage.*
htmlcov/
dist/
build/
*.so
.hypothesis/

# Venvs
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
.DS_Store

# Docs
docs/_build/

# Worktrees (used by superpowers workflows)
worktrees/
```

- [ ] **Step 3: Create `.codespellrc` for CLI invocation**

codespell reads the `[tool.codespell]` section from `pyproject.toml` (set up in Step 1), but a standalone `.codespellrc` is also valid. Create:

```ini
[codespell]
skip = *.lock,*.pyc,.git,dist,build,*.egg-info,docs/_build,src/physics_lint/data/floors.toml
ignore-words-list = bochner,fosls,riesz,sobolev,dirichlet,neumann,laplace,poisson,helmholtz,ernst,fornberg,trefethen,jekel,hansen,gruver,brandstetter,virkkunen,bachmayr,dahmen,oster,qiu,kovachki,welling,worrall,helwig,ot,cfm,ddpm,dps,fno,pinn,pino,pde,pdes,nd,ue,nin
```

- [ ] **Step 4: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/codespell-project/codespell
    rev: v2.3.0
    hooks:
      - id: codespell
        additional_dependencies:
          - tomli

  - repo: local
    hooks:
      - id: field-property-tests
        name: Field property tests (hypothesis, ci-quick profile)
        entry: pytest tests/test_field_properties.py -q --hypothesis-profile=ci-quick
        language: system
        files: '^src/physics_lint/(field/.*|norms)\.py$'
        pass_filenames: false
```

- [ ] **Step 5: Create the six-job CI matrix workflow**

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main, master]
  pull_request:

permissions:
  contents: read

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        include:
          - { os: ubuntu-latest, python: "3.10", numpy: "1.26.4", torch: "2.0.1", role: "floor-of-floors" }
          - { os: ubuntu-latest, python: "3.11", numpy: "1.26.4", torch: "2.2.2", role: "middle" }
          - { os: ubuntu-latest, python: "3.12", numpy: "2.0.2", torch: "2.5.0", role: "latest-everything" }
          - { os: ubuntu-latest, python: "3.12", numpy: "1.26.4", torch: "2.5.0", role: "numpy-1x-ceiling" }
          - { os: ubuntu-latest, python: "3.10", numpy: "2.0.2", torch: "2.2.2", role: "numpy-2x-floor" }
          - { os: macos-14,      python: "3.11", numpy: "2.0.2", torch: "2.5.0", role: "apple-silicon-smoke" }
    runs-on: ${{ matrix.os }}
    name: ${{ matrix.role }} (py${{ matrix.python }} np${{ matrix.numpy }} torch${{ matrix.torch }})

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}

      - name: Install pinned dependencies
        run: |
          pip install --upgrade pip
          pip install "numpy==${{ matrix.numpy }}" "torch==${{ matrix.torch }}"
          pip install -e ".[dev]"

      - name: Ruff check
        run: ruff check src tests

      - name: Ruff format --check
        run: ruff format --check src tests

      - name: Codespell
        run: codespell

      - name: Pytest + coverage
        run: pytest --cov=physics_lint --cov-report=term-missing --hypothesis-profile=ci
```

- [ ] **Step 6: Install dev deps locally and run pre-commit for the first time**

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
```

Expected: ruff passes (nothing to check yet beyond the init files), codespell passes, hypothesis hook is skipped (no changes to `field/` or `norms.py` yet).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml .pre-commit-config.yaml .codespellrc .gitignore
git commit -m "chore: scaffold dev tooling, pre-commit hooks, and 6-job CI matrix

- pyproject.toml with runtime + [dev] + [mesh] extras
- ruff lint+format with 100-char line length
- codespell with scientific-terminology allowlist
- pre-commit: ruff, codespell, hypothesis smoke check on field/norms edits
- CI matrix per design doc §2.4: 6 boundary cells across
  Python 3.10-3.12, NumPy 1.26/2.0, PyTorch 2.0/2.2/2.5,
  ubuntu-latest + macos-14 (arm64)"
```

---

## Task 2: Field ABC and norm module scaffolding

**Files:**
- Create: `src/physics_lint/field/__init__.py`
- Create: `src/physics_lint/field/_base.py`
- Create: `src/physics_lint/norms.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py` (hypothesis profiles)
- Create: `tests/test_field_base.py`

**Rationale:** The Field ABC is the load-bearing interface every rule reads from. Establishing the abstract surface first, with a failing test, anchors the GridField implementation in Task 3. The norms module is empty scaffold now; populated in Task 4 alongside GridField's derivative routines.

- [ ] **Step 1: Write the failing test for Field ABC**

`tests/test_field_base.py`:

```python
"""Field ABC contract tests.

The ABC is not instantiable on its own; this test asserts the required
abstract methods are declared so subclasses that forget one fail at
instantiation time rather than at first use.
"""

import pytest
from physics_lint.field import Field


def test_field_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        Field()  # type: ignore[abstract]


def test_field_abstract_method_names():
    expected = {"values", "at", "grad", "laplacian", "integrate", "values_on_boundary"}
    assert set(Field.__abstractmethods__) == expected
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_field_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'physics_lint.field'`

- [ ] **Step 3: Create `tests/__init__.py` and `tests/conftest.py`**

`tests/__init__.py`: empty file.

`tests/conftest.py`:

```python
"""Shared test fixtures and hypothesis profile registration."""

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci-quick",
    max_examples=25,
    deadline=500,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "dev",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("dev")
```

- [ ] **Step 4: Create Field ABC at `src/physics_lint/field/_base.py`**

```python
"""Field abstract base class.

A `Field` represents a discretized scalar or vector field over a PDE domain,
with a uniform interface for values, evaluation, differentiation, integration,
and boundary trace extraction. Subclasses (GridField, CallableField, MeshField)
implement these methods against their specific storage backends.

Smoothness caveat: physics-lint cannot introspect arbitrary torch.nn.Modules for
non-C2 activations. PH-NUM-003's detection is a best-effort scan of named
submodules (nn.ReLU, nn.LeakyReLU, nn.ELU, etc.) and does NOT detect F.relu
functional calls inside a forward method. For second-order PDEs with callable
fields, treat PH-NUM-003 as a check against a common class of footguns, not
a proof of smoothness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Field(ABC):
    """Abstract field over a discretized domain."""

    @abstractmethod
    def values(self) -> np.ndarray:
        """Return the underlying stored values on the native discretization."""

    @abstractmethod
    def at(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the field at arbitrary coordinates x via interpolation or AD."""

    @abstractmethod
    def grad(self) -> "Field":
        """Return the gradient as a new Field (vector-valued)."""

    @abstractmethod
    def laplacian(self) -> "Field":
        """Return the Laplacian as a new Field (scalar-valued)."""

    @abstractmethod
    def integrate(self, weight: "Field | None" = None) -> float:
        """Integrate the field (optionally weighted by another Field) over the domain."""

    @abstractmethod
    def values_on_boundary(self) -> np.ndarray:
        """Return the field's trace on the domain boundary for BC checking."""
```

- [ ] **Step 5: Create `src/physics_lint/field/__init__.py`**

```python
"""Field abstraction: ABC + concrete subclasses (GridField, CallableField, MeshField).

GridField, CallableField, and MeshField are added as they land (Tasks 3, 5, 6).
"""

from physics_lint.field._base import Field

__all__ = ["Field"]
```

- [ ] **Step 6: Create `src/physics_lint/norms.py` scaffold**

```python
"""Norm computations for residuals and field differences.

Populated incrementally:
- l2_grid: Task 4 (GridField derivatives + trapezoidal L2)
- h_minus_one_spectral: Task 4
- h_minus_one_fe: Task 8 (conditional on scikit-fem spike)
- bochner_l2_h_minus_one: Week 2 (heat/wave)
"""

from __future__ import annotations

import numpy as np


def l2_grid(u: np.ndarray, h: float | tuple[float, ...]) -> float:
    """Trapezoidal L2 norm on a uniform Cartesian grid.

    Half-weights boundary points per the trapezoidal rule.

    Args:
        u: Field values on the grid. Shape (Nx,), (Nx, Ny), or (Nx, Ny, Nz).
        h: Uniform spacing. Scalar if isotropic, tuple of (hx, hy, ...) otherwise.

    Returns:
        sqrt(integral of |u|^2 dx) over the grid.
    """
    raise NotImplementedError("Populated in Task 4.")
```

- [ ] **Step 7: Run the ABC test and verify it passes**

```bash
pytest tests/test_field_base.py -v
```

Expected: both tests PASS.

- [ ] **Step 8: Run ruff to catch any issues**

```bash
ruff check src tests && ruff format --check src tests
```

Expected: no errors (clean code).

- [ ] **Step 9: Commit**

```bash
git add src/physics_lint/field/ src/physics_lint/norms.py tests/__init__.py tests/conftest.py tests/test_field_base.py
git commit -m "feat(field): Field ABC with 6 abstract methods + norms scaffold

Per design doc §3.1 and §7.4.

Field ABC declares: values(), at(x), grad(), laplacian(), integrate(weight=None),
values_on_boundary(). Smoothness caveat documented in the module docstring
to match PH-NUM-003's best-effort submodule scan discipline.

norms.py currently stubs l2_grid only; spectral and FE H^-1 variants land
in Task 4. Bochner variants deferred to Week 2.

Tests assert ABC is non-instantiable and abstract methods match the spec.
Hypothesis profiles (ci-quick, ci, dev) registered in conftest.py."
```

---

## Task 3: GridField — FD backend

**Files:**
- Create: `src/physics_lint/field/grid.py`
- Create: `tests/test_gridfield_fd.py`
- Modify: `src/physics_lint/field/__init__.py` (export GridField)

**Rationale:** GridField is the primary Field subclass — used by dump mode unconditionally and by adapter mode when the user returns a tensor. Implementing the 4th-order FD backend first establishes the derivative machinery the spectral backend (Task 4) and CallableField (Task 5) will mirror.

- [ ] **Step 1: Write failing test — constructor stores values and spacing**

`tests/test_gridfield_fd.py`:

```python
"""GridField 4th-order central FD backend tests."""

import numpy as np
import pytest

from physics_lint.field import GridField


def test_gridfield_stores_values_and_spacing():
    u = np.zeros((8, 8))
    f = GridField(u, h=0.125, periodic=False)
    assert np.array_equal(f.values(), u)
    assert f.h == (0.125, 0.125)
    assert f.periodic is False
    assert f.backend == "fd"


def test_gridfield_rejects_periodic_and_fd_together_if_forced():
    # periodic=True auto-selects spectral; forcing backend="fd" with
    # periodic=True is legal (user override) and should still work.
    u = np.zeros((8, 8))
    f = GridField(u, h=0.125, periodic=True, backend="fd")
    assert f.backend == "fd"


def test_gridfield_scalar_h_expands_to_tuple():
    f = GridField(np.zeros((4, 4, 4)), h=0.5, periodic=False)
    assert f.h == (0.5, 0.5, 0.5)


def test_gridfield_tuple_h_must_match_ndim():
    with pytest.raises(ValueError, match="ndim"):
        GridField(np.zeros((4, 4)), h=(0.5, 0.5, 0.5), periodic=False)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_gridfield_fd.py -v
```

Expected: `ImportError: cannot import name 'GridField'`

- [ ] **Step 3: Create `src/physics_lint/field/grid.py` with constructor only**

```python
"""GridField: regular Cartesian grid with FD or spectral derivative backends.

4th-order central FD per Fornberg 1988 (design doc §3.2). Spectral branch
selected automatically when periodic=True unless user forces backend="fd".
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from physics_lint.field._base import Field


class GridField(Field):
    """Field stored as a NumPy array on a uniform Cartesian grid."""

    def __init__(
        self,
        values: np.ndarray,
        h: float | tuple[float, ...],
        *,
        periodic: bool,
        backend: Literal["fd", "spectral", "auto"] = "auto",
    ) -> None:
        self._values = np.ascontiguousarray(values)
        ndim = self._values.ndim
        if isinstance(h, (int, float)):
            self.h: tuple[float, ...] = (float(h),) * ndim
        else:
            if len(h) != ndim:
                raise ValueError(
                    f"h tuple length ({len(h)}) must match values.ndim ({ndim})"
                )
            self.h = tuple(float(hi) for hi in h)
        self.periodic = bool(periodic)
        if backend == "auto":
            self.backend: Literal["fd", "spectral"] = "spectral" if self.periodic else "fd"
        else:
            self.backend = backend

    def values(self) -> np.ndarray:
        return self._values

    def at(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("GridField.at() lands in a later task.")

    def grad(self) -> "GridField":
        raise NotImplementedError("Lands in Step 5 of this task.")

    def laplacian(self) -> "GridField":
        raise NotImplementedError("Lands in Step 5 of this task.")

    def integrate(self, weight: "Field | None" = None) -> float:
        raise NotImplementedError("Lands in Task 4 alongside l2_grid.")

    def values_on_boundary(self) -> np.ndarray:
        raise NotImplementedError("Lands in Task 6.")
```

- [ ] **Step 4: Export `GridField` from the package**

Update `src/physics_lint/field/__init__.py`:

```python
"""Field abstraction: ABC + concrete subclasses."""

from physics_lint.field._base import Field
from physics_lint.field.grid import GridField

__all__ = ["Field", "GridField"]
```

- [ ] **Step 5: Run the test and verify the constructor tests pass**

```bash
pytest tests/test_gridfield_fd.py::test_gridfield_stores_values_and_spacing tests/test_gridfield_fd.py::test_gridfield_rejects_periodic_and_fd_together_if_forced tests/test_gridfield_fd.py::test_gridfield_scalar_h_expands_to_tuple tests/test_gridfield_fd.py::test_gridfield_tuple_h_must_match_ndim -v
```

Expected: all four PASS.

- [ ] **Step 6: Write failing test — 4th-order FD derivative on polynomial is exact**

Append to `tests/test_gridfield_fd.py`:

```python
def test_fd_derivative_exact_on_cubic():
    # A cubic polynomial's fourth derivative is zero, so a 4th-order
    # FD stencil should reproduce the second derivative exactly to
    # machine precision at interior points.
    N = 32
    x = np.linspace(0.0, 1.0, N)
    h = x[1] - x[0]
    u_1d = x ** 3                                      # 1D, length N
    u_2d = u_1d[:, None] * np.ones((N, N))             # 2D, uniform in y

    f = GridField(u_2d, h=h, periodic=False, backend="fd")
    lap = f.laplacian().values()

    # interior second derivative wrt x of x^3 is 6x; wrt y of const is 0
    expected = 6.0 * x[:, None] * np.ones((N, N))
    assert np.allclose(lap[3:-3, 3:-3], expected[3:-3, 3:-3], atol=1e-10)


def test_fd_laplacian_periodic_sine_converges():
    # Laplacian of sin(2*pi*x)*sin(2*pi*y) is -8*pi^2 * sin*sin.
    # With periodic wraps, 4th-order FD should be accurate everywhere,
    # not just interior.
    for N in (32, 64, 128):
        x = np.linspace(0.0, 1.0, N, endpoint=False)
        y = np.linspace(0.0, 1.0, N, endpoint=False)
        h = 1.0 / N
        X, Y = np.meshgrid(x, y, indexing="ij")
        u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        f = GridField(u, h=h, periodic=True, backend="fd")
        lap = f.laplacian().values()
        expected = -8 * np.pi ** 2 * u
        # 4th-order FD on smooth periodic input: error ~ h^4
        tol = 200 * h ** 4
        assert np.max(np.abs(lap - expected)) < tol, (
            f"N={N}: max err {np.max(np.abs(lap - expected)):.3e}, tol {tol:.3e}"
        )
```

- [ ] **Step 7: Run the new tests to verify they fail**

```bash
pytest tests/test_gridfield_fd.py::test_fd_derivative_exact_on_cubic tests/test_gridfield_fd.py::test_fd_laplacian_periodic_sine_converges -v
```

Expected: `NotImplementedError: Lands in Step 5 of this task.`

- [ ] **Step 8: Implement the FD laplacian in `grid.py`**

Add at the top of `grid.py` (after the imports):

```python
# 4th-order central finite difference stencil for the second derivative.
# f''(x_i) ≈ (-f[i-2] + 16 f[i-1] - 30 f[i] + 16 f[i+1] - f[i+2]) / (12 h^2)
# One-sided 3rd-order variant for non-periodic boundaries: interior-only at
# depth 2; edges use numpy.gradient as a 2nd-order fallback for the outermost
# two layers (the rules that consume the Laplacian exclude the outer band via
# half-weight trapezoidal integration, so residual contributions there are
# suppressed but not zero).

_FD4_STENCIL = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0
```

Replace the `laplacian` method body:

```python
    def laplacian(self) -> "GridField":
        if self.backend == "spectral":
            raise NotImplementedError("Spectral Laplacian lands in Task 4.")
        u = self._values
        out = np.zeros_like(u)
        for axis, h_ax in enumerate(self.h):
            out = out + _fd4_second_derivative(u, axis=axis, h=h_ax, periodic=self.periodic)
        return GridField(out, h=self.h, periodic=self.periodic, backend=self.backend)
```

Add a module-level helper `_fd4_second_derivative` below the class (still inside `grid.py`):

```python
def _fd4_second_derivative(u: np.ndarray, *, axis: int, h: float, periodic: bool) -> np.ndarray:
    """4th-order central FD second derivative along a single axis.

    Periodic: np.roll wraps the stencil around the boundary — exact on
    smooth periodic inputs to ~h^4.

    Non-periodic: interior uses the central stencil; outer 2 layers fall
    back to a 2nd-order one-sided form via numpy.gradient twice. This is
    a degradation — the 4th-order rate only holds in the interior.
    """
    n = u.shape[axis]
    if n < 5:
        raise ValueError(f"4th-order FD requires at least 5 points along axis {axis}; got {n}")

    if periodic:
        out = np.zeros_like(u)
        for offset, coef in zip((-2, -1, 0, 1, 2), _FD4_STENCIL):
            out = out + coef * np.roll(u, -offset, axis=axis)
        return out / (h ** 2)

    # Non-periodic: central stencil in interior, 2nd-order fallback at edges.
    out = np.zeros_like(u)
    # Interior: slice [2:-2] along the target axis
    slicers_out = [slice(None)] * u.ndim
    slicers_out[axis] = slice(2, -2)
    for offset, coef in zip((-2, -1, 0, 1, 2), _FD4_STENCIL):
        slicers_in = [slice(None)] * u.ndim
        slicers_in[axis] = slice(2 + offset, n - 2 + offset if n - 2 + offset != 0 else None)
        out[tuple(slicers_out)] = out[tuple(slicers_out)] + coef * u[tuple(slicers_in)]
    out[tuple(slicers_out)] = out[tuple(slicers_out)] / (h ** 2)
    # Edge fallback: numpy.gradient twice (2nd-order, still converges)
    first = np.gradient(u, h, axis=axis, edge_order=2)
    second = np.gradient(first, h, axis=axis, edge_order=2)
    edge_front = [slice(None)] * u.ndim
    edge_front[axis] = slice(0, 2)
    edge_back = [slice(None)] * u.ndim
    edge_back[axis] = slice(-2, None)
    out[tuple(edge_front)] = second[tuple(edge_front)]
    out[tuple(edge_back)] = second[tuple(edge_back)]
    return out
```

- [ ] **Step 9: Run all GridField tests and verify they pass**

```bash
pytest tests/test_gridfield_fd.py -v
```

Expected: all 6 tests PASS. If `test_fd_laplacian_periodic_sine_converges` fails with a tolerance violation, debug by printing the max-error-as-function-of-N — the rate should be ~h^4 (error drops ~16x when N doubles).

- [ ] **Step 10: Run ruff and commit**

```bash
ruff check src tests && ruff format src tests
git add src/physics_lint/field/grid.py src/physics_lint/field/__init__.py tests/test_gridfield_fd.py
git commit -m "feat(field): GridField with 4th-order central FD Laplacian

Per design doc §3.2. Fornberg 1988 stencil [-1, 16, -30, 16, -1]/12 along
each axis, summed over axes to give the d-dimensional Laplacian.

- Periodic boundary: np.roll wraps the stencil (exact to ~h^4 on smooth periodic)
- Non-periodic interior: 4th-order central stencil in slice [2:-2]
- Non-periodic edges: numpy.gradient twice (2nd-order fallback, outer 2 layers)

grad(), at(), integrate(), values_on_boundary() still NotImplementedError
pending Tasks 4, 5, 6. Spectral branch raises; lands in Task 4."
```

---

## Task 4: GridField spectral backend + norms module

**Files:**
- Modify: `src/physics_lint/field/grid.py` (add spectral branch; add `grad`, `integrate`, `values_on_boundary`)
- Modify: `src/physics_lint/norms.py` (fill in `l2_grid` and `h_minus_one_spectral`)
- Create: `tests/test_gridfield_spectral.py`
- Create: `tests/test_norms.py`

**Rationale:** Spectral derivatives are the variationally-correct path for periodic Poisson at machine precision (floor ~1e-14), which is the primary `PH-RES-001` case for Week 1. `h_minus_one_spectral` is the norm `PH-RES-001` reports. `l2_grid` underlies every other norm. Finishing the remaining Field methods here (`grad`, `integrate`, `values_on_boundary`) lets Task 5 wire up CallableField against the same contract.

- [ ] **Step 1: Write failing spectral test**

`tests/test_gridfield_spectral.py`:

```python
"""GridField Fourier spectral backend tests.

Spectral derivatives on smooth periodic inputs are exact to roughly
machine precision; the tolerance here reflects floating-point noise
plus FFT backend drift (see design doc §6.3).
"""

import numpy as np

from physics_lint.field import GridField


def test_spectral_laplacian_sine_machine_precision():
    # -∆ of sin(2*pi*x)*sin(2*pi*y) = 8*pi^2 * sin*sin at machine precision.
    N = 64
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    h = 1.0 / N
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    f = GridField(u, h=h, periodic=True)   # auto-selects spectral
    lap = f.laplacian().values()
    expected = -8 * np.pi ** 2 * u

    assert np.max(np.abs(lap - expected)) < 1e-11


def test_spectral_laplacian_multimode():
    # Higher modes: -∆ sin(k*pi*x) sin(k*pi*y) = 2*(k*pi)^2 * sin*sin
    N = 128
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    for k in (1, 3, 7):
        u = np.sin(2 * k * np.pi * X) * np.sin(2 * k * np.pi * Y)
        f = GridField(u, h=h, periodic=True)
        lap = f.laplacian().values()
        expected = -2 * (2 * k * np.pi) ** 2 * u
        assert np.max(np.abs(lap - expected)) < 1e-10, f"k={k} failed"


def test_spectral_laplacian_3d():
    N = 32
    h = 1.0 / N
    axis = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    u = (
        np.sin(2 * np.pi * X)
        * np.sin(2 * np.pi * Y)
        * np.sin(2 * np.pi * Z)
    )
    f = GridField(u, h=h, periodic=True)
    lap = f.laplacian().values()
    expected = -12 * np.pi ** 2 * u
    assert np.max(np.abs(lap - expected)) < 1e-10
```

- [ ] **Step 2: Run spectral tests to verify they fail**

```bash
pytest tests/test_gridfield_spectral.py -v
```

Expected: `NotImplementedError: Spectral Laplacian lands in Task 4.`

- [ ] **Step 3: Implement spectral Laplacian in `grid.py`**

Add a module-level helper below `_fd4_second_derivative`:

```python
def _spectral_laplacian(u: np.ndarray, h: tuple[float, ...]) -> np.ndarray:
    """Fourier spectral Laplacian on a uniform periodic grid.

    For a d-dimensional grid with spacing h = (h_0, h_1, ...) and shape
    (N_0, N_1, ...), the physical length along axis i is L_i = N_i * h_i
    (endpoint=False convention). Wavenumbers k_i = 2*pi*fftfreq(N_i, d=h_i).

    Laplacian transform: -(k_0^2 + k_1^2 + ...) * u_hat; zero out Nyquist
    per Trefethen 2000 advice for first-derivative consistency (harmless
    for the Laplacian too since it preserves symmetry).
    """
    shape = u.shape
    ndim = u.ndim
    k_grids = []
    for axis in range(ndim):
        k = np.fft.fftfreq(shape[axis], d=h[axis]) * (2.0 * np.pi)
        # zero out Nyquist bin
        if shape[axis] % 2 == 0:
            k[shape[axis] // 2] = 0.0
        shape_broadcast = [1] * ndim
        shape_broadcast[axis] = shape[axis]
        k_grids.append(k.reshape(shape_broadcast))
    k_sq_total = sum(k ** 2 for k in k_grids)
    u_hat = np.fft.fftn(u)
    return np.real(np.fft.ifftn(-k_sq_total * u_hat))
```

Update `GridField.laplacian` to branch on backend:

```python
    def laplacian(self) -> "GridField":
        u = self._values
        if self.backend == "spectral":
            if not self.periodic:
                raise ValueError("spectral backend requires periodic=True")
            out = _spectral_laplacian(u, self.h)
        else:
            out = np.zeros_like(u)
            for axis, h_ax in enumerate(self.h):
                out = out + _fd4_second_derivative(u, axis=axis, h=h_ax, periodic=self.periodic)
        return GridField(out, h=self.h, periodic=self.periodic, backend=self.backend)
```

- [ ] **Step 4: Run spectral tests and verify they pass**

```bash
pytest tests/test_gridfield_spectral.py -v
```

Expected: all 3 PASS with max errors ~1e-13 to 1e-12.

- [ ] **Step 5: Implement `grad`, `integrate`, `values_on_boundary`**

Append to `GridField` (replacing the NotImplementedError stubs):

```python
    def grad(self) -> list["GridField"]:
        """Return a list of per-axis partial derivatives, each a GridField.

        Note: unlike laplacian(), grad() returns a list rather than a single
        Field because physics-lint never materializes vector Fields directly;
        the FD-vs-AD cross-check in PH-RES-002 consumes components separately,
        and boundary flux computations in PH-BC-002 dot with the outward normal
        component-wise.
        """
        u = self._values
        parts: list[GridField] = []
        for axis, h_ax in enumerate(self.h):
            if self.backend == "spectral":
                parts.append(
                    GridField(
                        _spectral_first_derivative(u, axis=axis, h=h_ax),
                        h=self.h, periodic=self.periodic, backend=self.backend,
                    )
                )
            else:
                parts.append(
                    GridField(
                        _fd4_first_derivative(u, axis=axis, h=h_ax, periodic=self.periodic),
                        h=self.h, periodic=self.periodic, backend=self.backend,
                    )
                )
        return parts  # type: ignore[return-value]  # grad signature refined in Task 6

    def integrate(self, weight: "Field | None" = None) -> float:
        from physics_lint.norms import trapezoidal_integral
        u = self._values
        if weight is None:
            return trapezoidal_integral(u, self.h)
        if not isinstance(weight, GridField):
            raise TypeError("GridField.integrate currently supports only GridField weights")
        if weight.values().shape != u.shape:
            raise ValueError("weight shape must match values shape")
        return trapezoidal_integral(u * weight.values(), self.h)

    def values_on_boundary(self) -> np.ndarray:
        """Return a flat array of values on the d-dimensional boundary faces.

        For a 2D grid of shape (Nx, Ny), the boundary is the concatenation of
        the four edges: left (x=0), right (x=Nx-1), bottom (y=0 excluding
        corners already in left/right), top (y=Ny-1 excluding corners).
        Output order is deterministic and reproducible; rules that compare
        boundary values must pass boundary targets in the same ordering
        (the analytical battery provides matched pairs).
        """
        u = self._values
        if u.ndim == 1:
            return np.concatenate([u[:1], u[-1:]])
        if u.ndim == 2:
            left = u[0, :]
            right = u[-1, :]
            bottom = u[1:-1, 0]
            top = u[1:-1, -1]
            return np.concatenate([left, right, bottom, top])
        if u.ndim == 3:
            faces = [
                u[0, :, :].ravel(),
                u[-1, :, :].ravel(),
                u[1:-1, 0, :].ravel(),
                u[1:-1, -1, :].ravel(),
                u[1:-1, 1:-1, 0].ravel(),
                u[1:-1, 1:-1, -1].ravel(),
            ]
            return np.concatenate(faces)
        raise ValueError(f"values_on_boundary: unsupported ndim {u.ndim}")
```

Add the first-derivative helpers near the second-derivative ones:

```python
# First-derivative stencil: (1/12)(f[i-2] - 8 f[i-1] + 8 f[i+1] - f[i+2]) / h
_FD4_FIRST_STENCIL = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0


def _fd4_first_derivative(u: np.ndarray, *, axis: int, h: float, periodic: bool) -> np.ndarray:
    n = u.shape[axis]
    if n < 5:
        raise ValueError(f"4th-order first derivative requires at least 5 points; got {n}")
    if periodic:
        out = np.zeros_like(u)
        for offset, coef in zip((-2, -1, 0, 1, 2), _FD4_FIRST_STENCIL):
            out = out + coef * np.roll(u, -offset, axis=axis)
        return out / h
    # Non-periodic fallback: numpy.gradient (2nd-order)
    return np.gradient(u, h, axis=axis, edge_order=2)


def _spectral_first_derivative(u: np.ndarray, *, axis: int, h: float) -> np.ndarray:
    n = u.shape[axis]
    k = np.fft.fftfreq(n, d=h) * (2.0 * np.pi)
    if n % 2 == 0:
        k[n // 2] = 0.0  # zero Nyquist
    shape_broadcast = [1] * u.ndim
    shape_broadcast[axis] = n
    k_b = k.reshape(shape_broadcast)
    u_hat = np.fft.fft(u, axis=axis)
    return np.real(np.fft.ifft(1j * k_b * u_hat, axis=axis))
```

- [ ] **Step 6: Write failing test for `l2_grid` and `trapezoidal_integral`**

`tests/test_norms.py`:

```python
"""Norms module tests — L^2 trapezoidal, H^-1 spectral."""

import numpy as np
import pytest

from physics_lint.norms import (
    h_minus_one_spectral,
    l2_grid,
    trapezoidal_integral,
)


def test_trapezoidal_integral_constant_1d():
    N = 17
    u = np.ones(N)
    h = 1.0 / (N - 1)   # endpoint-inclusive grid
    result = trapezoidal_integral(u, (h,))
    assert abs(result - 1.0) < 1e-12


def test_trapezoidal_integral_linear_2d():
    N = 65
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = X + Y                                  # integral over [0,1]^2 = 1.0
    h = 1.0 / (N - 1)
    result = trapezoidal_integral(u, (h, h))
    assert abs(result - 1.0) < 1e-12


def test_l2_grid_sine():
    # integral of sin^2(pi x) sin^2(pi y) dx dy over [0,1]^2 = 1/4
    # so sqrt(1/4) = 0.5
    N = 129
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)
    h = 1.0 / (N - 1)
    assert abs(l2_grid(u, (h, h)) - 0.5) < 1e-5


def test_h_minus_one_spectral_sine_mode():
    # u = sin(2 pi x) sin(2 pi y) on periodic [0,1]^2
    # ||u||_{H^-1}^2 = sum_{k!=0} |u_hat_k|^2 / |k|^2
    # The only nonzero modes are (+/-1, +/-1) with |k|^2 = (2 pi)^2 * 2
    # and |u_hat|^2 = 1/16 each for N^2 = 1 samples after FFT normalization
    # This test asserts the function runs and returns a positive value
    # of the expected order of magnitude.
    N = 64
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    h = 1.0 / N
    val = h_minus_one_spectral(u, (h, h))
    # Expected: ||u||^2 / (8 pi^2)  for this single-mode case
    expected_sq = 0.25 / (8 * np.pi ** 2)
    assert abs(val - np.sqrt(expected_sq)) < 1e-8


def test_h_minus_one_spectral_zero_mean_required():
    # The DC mode (k=0) has no inverse; h_minus_one_spectral must either
    # ignore it or raise. We choose: ignore it silently (sum over k != 0).
    N = 32
    u = np.ones((N, N)) * 3.5   # pure DC
    h = 1.0 / N
    val = h_minus_one_spectral(u, (h, h))
    assert val == 0.0
```

- [ ] **Step 7: Run norm tests to verify they fail**

```bash
pytest tests/test_norms.py -v
```

Expected: `ImportError: cannot import name 'h_minus_one_spectral'` and `NotImplementedError: Populated in Task 4.`

- [ ] **Step 8: Implement norms in `src/physics_lint/norms.py`**

Replace the file contents with:

```python
"""Norm computations for residuals and field differences.

- l2_grid: trapezoidal L^2 on a uniform Cartesian grid (half-weights at edges)
- trapezoidal_integral: weighted L^1 integral via the same trapezoidal rule
- h_minus_one_spectral: sqrt(sum_{k != 0} |u_hat|^2 / |k|^2) on periodic grids
- h_minus_one_fe: (Task 8, conditional on scikit-fem spike)
- bochner_l2_h_minus_one: (Week 2, heat/wave)
"""

from __future__ import annotations

import numpy as np


def trapezoidal_integral(u: np.ndarray, h: tuple[float, ...]) -> float:
    """Multi-dimensional trapezoidal integral on a uniform grid.

    Half-weights all boundary points (1D: 0.5 at i=0 and i=N-1; 2D: quarter
    at the four corners; and so on). The grid is assumed endpoint-inclusive:
    u has shape (N_0, N_1, ...) with physical lengths L_i = (N_i - 1) * h_i.

    Args:
        u: Values on the grid.
        h: Spacing tuple, len == u.ndim.

    Returns:
        sum_i w_i u_i, where w_i are the trapezoidal weights scaled by prod(h).
    """
    if len(h) != u.ndim:
        raise ValueError(f"h length {len(h)} must match u.ndim {u.ndim}")
    weights = np.ones_like(u)
    for axis in range(u.ndim):
        slicer_front = [slice(None)] * u.ndim
        slicer_back = [slice(None)] * u.ndim
        slicer_front[axis] = 0
        slicer_back[axis] = -1
        weights[tuple(slicer_front)] *= 0.5
        weights[tuple(slicer_back)] *= 0.5
    cell_volume = float(np.prod(h))
    return float(np.sum(weights * u) * cell_volume)


def l2_grid(u: np.ndarray, h: float | tuple[float, ...]) -> float:
    """sqrt(integral |u|^2 dx) on a uniform Cartesian grid.

    Uses trapezoidal_integral with half-weights at boundaries. Assumes the
    grid is endpoint-inclusive (physical lengths L_i = (N_i - 1) * h_i).
    """
    if isinstance(h, (int, float)):
        h_tuple = (float(h),) * u.ndim
    else:
        h_tuple = tuple(float(hi) for hi in h)
    return float(np.sqrt(trapezoidal_integral(u * u, h_tuple)))


def h_minus_one_spectral(r: np.ndarray, h: float | tuple[float, ...]) -> float:
    """sqrt(sum_{k != 0} |r_hat_k|^2 / |k|^2) on a periodic grid.

    Design doc §3.4 and §7.4. Only valid for periodic boundary conditions;
    the caller must ensure periodicity. The k=0 (DC) mode is excluded because
    the H^-1 norm is not defined on constants (Poincaré argument).

    Scaling note: numpy FFT is unnormalized, so |r_hat_k|^2 carries a factor
    of N^2 relative to the continuous Fourier coefficient. The division by
    N^2 converts back to the normalized spectrum, and the multiplication by
    the physical volume L^d = prod(L_i) = prod(N_i * h_i) converts the
    discrete sum to a Riemann approximation of the spectral integral.
    """
    if isinstance(h, (int, float)):
        h_tuple = (float(h),) * r.ndim
    else:
        h_tuple = tuple(float(hi) for hi in h)

    shape = r.shape
    r_hat = np.fft.fftn(r)
    # Build squared wavenumber grid
    k_sq_total = np.zeros(shape, dtype=float)
    for axis in range(r.ndim):
        k = np.fft.fftfreq(shape[axis], d=h_tuple[axis]) * (2.0 * np.pi)
        shape_broadcast = [1] * r.ndim
        shape_broadcast[axis] = shape[axis]
        k_sq_total = k_sq_total + (k.reshape(shape_broadcast)) ** 2
    # Exclude DC mode
    mask = k_sq_total > 0
    if not np.any(mask):
        return 0.0
    n_total = float(np.prod(shape))
    volume = float(np.prod([s * hi for s, hi in zip(shape, h_tuple)]))
    # |r_hat|^2 / N^2 gives the normalized spectrum; dividing by |k|^2 gives
    # the H^-1 weight; summing and multiplying by the physical volume gives
    # the H^-1 squared norm as a Riemann approximation.
    h_minus_one_sq = float(
        np.sum(np.abs(r_hat[mask]) ** 2 / k_sq_total[mask]) / (n_total ** 2) * volume
    )
    return float(np.sqrt(h_minus_one_sq))
```

- [ ] **Step 9: Run norm tests and verify they pass**

```bash
pytest tests/test_norms.py -v
```

Expected: all 5 PASS. If `test_h_minus_one_spectral_sine_mode` fails by a constant factor, the normalization in `h_minus_one_spectral` needs a FFT-scaling review — print `|r_hat|^2` sum and compare to analytical `||u||^2` as a sanity cross-check.

- [ ] **Step 10: Run the whole test suite to make sure nothing regressed**

```bash
pytest -q
```

Expected: all tests pass (ABC contract + GridField FD + GridField spectral + norms).

- [ ] **Step 11: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/physics_lint/field/grid.py src/physics_lint/norms.py tests/test_gridfield_spectral.py tests/test_norms.py
git commit -m "feat(field,norms): spectral Laplacian + grad + L2 and H^-1 norms

Per design doc §3.2 (spectral branch) and §7.4 (norm module).

GridField:
- spectral Laplacian via fftn with Nyquist zeroed (Trefethen 2000)
- spectral and FD first derivatives -> grad() returns list of component Fields
- integrate() via trapezoidal_integral (half-weights at boundaries)
- values_on_boundary() returns deterministic concatenation of faces for 1D/2D/3D

norms:
- trapezoidal_integral: half-weight edge rule, endpoint-inclusive grid
- l2_grid: sqrt(trapezoidal_integral(u^2))
- h_minus_one_spectral: sum over k != 0 of |r_hat|^2 / |k|^2, FFT-normalized

Tests: machine-precision on sin*sin Laplacian, multi-mode, 3D, linear trapezoidal
exact to 1e-12, l2_grid matches analytical sin^2 integral to 1e-5, and H^-1
single-mode reproduces the analytical coefficient."
```

---

## Task 5: CallableField — PyTorch autograd-based Field

**Files:**
- Create: `src/physics_lint/field/callable.py`
- Create: `tests/test_callable_field.py`
- Modify: `src/physics_lint/field/__init__.py` (export CallableField)

**Rationale:** CallableField is the Field subclass for adapter-mode with autograd. Required by `PH-RES-002` (FD-vs-AD cross-check) and `PH-SYM-003` (SO(2) LEE). For Week 1 we need at least `values()` (materializes on a sampling grid) and `laplacian()` (via `torch.autograd.functional.hessian` summed over diagonals). The detailed AD paths for LEE and FD-vs-AD land in Weeks 3-4; the Week 1 scope is a working Laplace/Poisson residual path through a callable.

- [ ] **Step 1: Write the failing CallableField test**

`tests/test_callable_field.py`:

```python
"""CallableField — wrap a PyTorch callable as a Field.

Week 1 scope: values() materializes on a user-provided sampling grid;
laplacian() via AD; grad() via AD. at() and integrate() delegate to
the materialized GridField.
"""

import numpy as np
import pytest
import torch

from physics_lint.field import CallableField, GridField


def _quadratic_model(x: torch.Tensor) -> torch.Tensor:
    # u(x, y) = x^2 + y^2 => Laplacian = 4
    return (x[..., 0] ** 2 + x[..., 1] ** 2).unsqueeze(-1)


def test_callable_field_values_on_grid():
    grid = torch.stack(torch.meshgrid(
        torch.linspace(0.0, 1.0, 16),
        torch.linspace(0.0, 1.0, 16),
        indexing="ij",
    ), dim=-1)
    f = CallableField(_quadratic_model, sampling_grid=grid, h=(1.0 / 15, 1.0 / 15))
    vals = f.values()
    assert vals.shape == (16, 16)
    expected = grid[..., 0].numpy() ** 2 + grid[..., 1].numpy() ** 2
    assert np.allclose(vals, expected, atol=1e-6)


def test_callable_field_laplacian_quadratic():
    grid = torch.stack(torch.meshgrid(
        torch.linspace(0.0, 1.0, 8),
        torch.linspace(0.0, 1.0, 8),
        indexing="ij",
    ), dim=-1)
    f = CallableField(_quadratic_model, sampling_grid=grid, h=(1.0 / 7, 1.0 / 7))
    lap = f.laplacian().values()
    # Exact: Laplacian of x^2 + y^2 is 4 everywhere
    assert np.allclose(lap, 4.0, atol=1e-5)


def test_callable_field_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        CallableField("not a callable", sampling_grid=torch.zeros(4, 4, 2), h=(0.25, 0.25))
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_callable_field.py -v
```

Expected: `ImportError: cannot import name 'CallableField'`

- [ ] **Step 3: Implement CallableField**

`src/physics_lint/field/callable.py`:

```python
"""CallableField — Field wrapping a PyTorch callable via autograd.

The callable accepts a Tensor of shape (..., d) representing d-dimensional
points and returns a Tensor of shape (..., 1) (scalar field). values() and
laplacian() materialize the field and its Laplacian on a user-provided
sampling grid, then return GridField-compatible numpy arrays.

Week 1 scope: values + laplacian + grad + values_on_boundary via AD.
at() and integrate() delegate to an internally-materialized GridField.

Limitations noted in the design doc §3.3:
- Requires C2 activations for second-order PDEs (PH-NUM-003 warns on
  best-effort submodule scan; does not detect F.relu in forward).
- For performance, the Week 1 implementation uses torch.autograd.functional.hessian
  which is O(d^2) per point; future work can specialize for 2D/3D and use vmap.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from physics_lint.field._base import Field
from physics_lint.field.grid import GridField


class CallableField(Field):
    """Field backed by a torch-callable mapping coords to scalar predictions."""

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        *,
        sampling_grid: torch.Tensor,
        h: float | tuple[float, ...],
        periodic: bool = False,
    ) -> None:
        if not callable(model):
            raise TypeError(f"model must be callable, got {type(model).__name__}")
        self._model = model
        # sampling_grid shape: (..., d) where d = number of spatial dims
        self._grid = sampling_grid
        self.ndim = sampling_grid.shape[-1]
        if isinstance(h, (int, float)):
            self.h: tuple[float, ...] = (float(h),) * self.ndim
        else:
            self.h = tuple(float(hi) for hi in h)
        self.periodic = bool(periodic)
        self._cached_values: np.ndarray | None = None
        self._cached_grid_field: GridField | None = None

    def _materialize(self) -> GridField:
        if self._cached_grid_field is None:
            with torch.no_grad():
                out = self._model(self._grid)
                if out.dim() == self._grid.dim():
                    out = out.squeeze(-1)
                vals = out.detach().cpu().numpy()
            self._cached_values = vals
            self._cached_grid_field = GridField(
                vals, h=self.h, periodic=self.periodic,
            )
        return self._cached_grid_field

    def values(self) -> np.ndarray:
        return self._materialize().values()

    def at(self, x: np.ndarray) -> np.ndarray:
        # Delegate to the materialized GridField (no interpolation yet; Week 2+).
        raise NotImplementedError("CallableField.at() lands in Week 2 if needed by rules.")

    def grad(self) -> list["CallableField"]:
        raise NotImplementedError("CallableField.grad lands in Week 3 with PH-SYM-003.")

    def laplacian(self) -> GridField:
        """Compute the Laplacian via torch autograd at each sampling-grid point.

        For a 2D grid of shape (Nx, Ny, 2), flatten to (Nx*Ny, 2), compute
        the Hessian per point with torch.func.hessian composed with vmap,
        and sum the diagonal. Fallback to per-point torch.autograd.functional.hessian
        when torch.func is unavailable (PyTorch < 2.0).
        """
        pts = self._grid.reshape(-1, self.ndim).clone().requires_grad_(True)

        def _scalar(p: torch.Tensor) -> torch.Tensor:
            y = self._model(p.unsqueeze(0))
            return y.reshape(())

        try:
            from torch.func import hessian, vmap
            hess_fn = vmap(hessian(_scalar))
            hess_all = hess_fn(pts)   # shape (N, d, d)
        except ImportError:
            hess_rows = [
                torch.autograd.functional.hessian(_scalar, p)
                for p in pts
            ]
            hess_all = torch.stack(hess_rows)

        trace = hess_all.diagonal(dim1=-2, dim2=-1).sum(-1)  # (N,)
        lap = trace.reshape(self._grid.shape[:-1]).detach().cpu().numpy()
        return GridField(lap, h=self.h, periodic=self.periodic)

    def integrate(self, weight: "Field | None" = None) -> float:
        return self._materialize().integrate(weight)

    def values_on_boundary(self) -> np.ndarray:
        return self._materialize().values_on_boundary()
```

- [ ] **Step 4: Export CallableField**

Update `src/physics_lint/field/__init__.py`:

```python
"""Field abstraction: ABC + concrete subclasses."""

from physics_lint.field._base import Field
from physics_lint.field.callable import CallableField
from physics_lint.field.grid import GridField

__all__ = ["Field", "GridField", "CallableField"]
```

- [ ] **Step 5: Run the CallableField tests**

```bash
pytest tests/test_callable_field.py -v
```

Expected: all 3 PASS. `test_callable_field_laplacian_quadratic` should give exactly 4.0 everywhere — a quadratic's Hessian is constant, so AD is exact.

- [ ] **Step 6: Run the full test suite**

```bash
pytest -q
```

Expected: all tests (ABC + GridField FD + spectral + norms + CallableField) pass.

- [ ] **Step 7: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/physics_lint/field/callable.py src/physics_lint/field/__init__.py tests/test_callable_field.py
git commit -m "feat(field): CallableField with autograd Laplacian

Per design doc §3.3.

Wraps torch.nn.Module or any Callable[[Tensor], Tensor]. Materializes onto
a user-provided sampling grid for values, uses torch.func.hessian + vmap
(fallback to torch.autograd.functional.hessian) for the Laplacian.

grad() and at() stubbed with NotImplementedError pointing at later tasks
(PH-SYM-003 Week 3 for grad via autograd, at() needs-driven in Week 2).
integrate() and values_on_boundary() delegate to the internally materialized
GridField.

Tests: quadratic model reproduces Laplacian exactly (= 4 at every point on
x^2 + y^2), values() materializes correctly, rejects non-callable inputs."
```

---

## Task 6: scikit-fem spike and Week-1 Day-2 decision gate

**Files:**
- Create: `scripts/spike_scikit_fem.py` (throwaway, not committed)
- Modify: design-doc status or `docs/design/spike-notes-2026-04-14.md` (commit the decision)

**Rationale:** The MeshField / FE H^-1 path is conditional on a half-day spike per §2.4 of the design doc. The spike asks one question: can we assemble a Poisson stiffness matrix on a `MeshTri` basis with `ElementTriP2`, solve for a known analytical solution, and measure $O(h^2)$ convergence in under half a day? The outcome determines whether Week 3 Day 4 implements `MeshField`/`PH-CON-004`/`PH-NUM-001` or defers them to V2.

- [ ] **Step 1: Install scikit-fem**

```bash
pip install "scikit-fem>=10"
```

Expected: clean install (pure-Python package, no native deps).

- [ ] **Step 2: Write the spike script**

`scripts/spike_scikit_fem.py` (add to `.gitignore` via a `scripts/*.py` wildcard — this script is not committed):

```python
"""Timeboxed spike: can we assemble + solve Poisson on MeshTri in under 4h?

Decision criteria (design doc §2.4):
  - PASS if (a) stiffness assembly works, (b) solve runs, (c) L2 error on
    u = sin(pi x) sin(pi y) converges as O(h^2) across 3 refinements,
    and (d) total time-to-first-correct-result < 4 hours.
  - If any criterion fails, MeshField / PH-CON-004 / PH-NUM-001 ship in V2.
"""

import time

import numpy as np
from skfem import Basis, MeshTri, ElementTriP2, condense, solve
from skfem.helpers import dot, grad
from skfem.models.poisson import laplace, unit_load


def run_single(n_refinements: int) -> float:
    mesh = MeshTri().refined(n_refinements)
    basis = Basis(mesh, ElementTriP2())
    K = laplace.assemble(basis)
    # f = 2 pi^2 sin(pi x) sin(pi y)
    @__import__("skfem").Functional
    def f_functional(w):
        x, y = w.x
        return 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y) * w.v

    # Use unit_load as a placeholder; swap for manufactured f with LinearForm
    from skfem import LinearForm
    from skfem.helpers import grad as _

    @LinearForm
    def load(v, w):
        x, y = w.x
        return 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y) * v

    b = load.assemble(basis)
    D = basis.get_dofs().flatten()
    x_h = solve(*condense(K, b, D=D))

    # Compare to analytical u = sin(pi x) sin(pi y)
    x_coords = basis.doflocs[0]
    y_coords = basis.doflocs[1]
    u_exact = np.sin(np.pi * x_coords) * np.sin(np.pi * y_coords)
    err = np.linalg.norm(x_h - u_exact) / np.sqrt(len(x_h))
    return err


if __name__ == "__main__":
    t0 = time.time()
    errs = [run_single(r) for r in (2, 3, 4)]
    t_elapsed = time.time() - t0
    print(f"errors by refinement: {errs}")
    if errs[0] > 0 and errs[1] > 0:
        rate = np.log2(errs[0] / errs[1])
        print(f"observed rate (refinement 2->3): {rate:.3f}  (expect ~2.0)")
    print(f"spike total time: {t_elapsed:.1f}s")
```

- [ ] **Step 3: Run the spike**

```bash
python scripts/spike_scikit_fem.py
```

**Pass criteria:** no exceptions; errors decrease by ~4x per refinement (rate ~ 2); total time < 4 hours (including debugging time).

**If it runs clean on first try** (expected; scikit-fem is well-documented): the spike has "passed" and we continue on this plan.

**If it fails** — LinearForm assembly errors, scikit-fem API changed, mesh type issues, or other obstacles that look like they'll take more than half a day total — stop, document the blocker, and invoke the rollback below.

- [ ] **Step 4: Document the outcome in a commit message and a notes file**

**If spike passed:**

```bash
cat > docs/design/spike-notes-2026-04-14.md << 'EOF'
# scikit-fem spike — Week 1 Day 2

**Outcome:** PASSED on YYYY-MM-DD

**Measured:**
- Assembly of Poisson stiffness via `laplace.assemble(Basis(MeshTri, ElementTriP2))`: OK
- Solve of Dirichlet-homogeneous Poisson for $u = \sin(\pi x) \sin(\pi y)$: OK
- L^2 errors at refinements 2, 3, 4: [record actual values]
- Measured convergence rate: [record — should be ~2.0]
- Total elapsed time: [record — must be < 4h]

**Decision:** MeshField, PH-CON-004, PH-NUM-001 ship in V1. Week 3 Day 4
will implement scikit-fem-backed path per design doc §3.4 and §8.2.
EOF
git add docs/design/spike-notes-2026-04-14.md
git commit -m "docs: record scikit-fem spike outcome (PASSED) — MeshField ships in V1"
```

**If spike failed or overshot the timebox:**

```bash
cat > docs/design/spike-notes-2026-04-14.md << 'EOF'
# scikit-fem spike — Week 1 Day 2

**Outcome:** DEFERRED on YYYY-MM-DD

**Blocker:** [what went wrong — API error, excessive time, unexpected behavior]

**Decision:** MeshField, PH-CON-004, PH-NUM-001 defer to V2. V1 ships GridField
only. The rule catalog entries for PH-CON-004 and PH-NUM-001 are marked as
"conditional, not shipped in V1" in the rule catalog and docs.

**Reclaimed time:** Week 3 Day 4 becomes a buffer day or extends the
broken-model toy gallery (Week 4 Day 4). Week-4 LOC budget drops from
~4050 to ~3650 (design doc §17 reconciliation).
EOF
git add docs/design/spike-notes-2026-04-14.md
git commit -m "docs: record scikit-fem spike outcome (DEFERRED) — MeshField -> V2"
```

- [ ] **Step 5: Remove the throwaway spike script (not committed)**

```bash
rm scripts/spike_scikit_fem.py
# scripts/ directory stays; used later for floor calibration
```

- [ ] **Step 6: If spike failed, update the Week 1 plan (this document) to skip MeshField references**

Not applicable if spike passed. If failed, the only downstream effect on Week 1 is that Task 9 (analytical battery) only needs GridField-compatible analytical solutions, which it already does — no changes required. Week 3 Day 4 changes from "MeshField + PH-CON-004 + PH-NUM-001" to "buffer day" in the separate Week 3 plan when it's written.

No commit needed for this step; it's a planning-only observation.

---

## Task 7: DomainSpec pydantic hierarchy

**Files:**
- Create: `src/physics_lint/spec.py`
- Create: `tests/test_spec.py`
- Modify: `src/physics_lint/__init__.py` (export `DomainSpec`)

**Rationale:** Every rule reads `DomainSpec`, never raw config. Establishing the type — and its cross-validators — before wiring up the config loader (Task 8) or the rules (Tasks 10–12) guarantees that validation lives in exactly one place.

- [ ] **Step 1: Write the failing DomainSpec tests**

`tests/test_spec.py`:

```python
"""DomainSpec — pydantic v2 hierarchy tests."""

import warnings

import pytest
from pydantic import ValidationError

from physics_lint import DomainSpec
from physics_lint.spec import BCSpec, FieldSourceSpec, GridDomain, SymmetrySpec


def _valid_heat_dict() -> dict:
    return {
        "pde": "heat",
        "grid_shape": [64, 64, 32],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "symmetries": {"declared": ["D4"]},
        "field": {"type": "grid", "backend": "fd", "dump_path": "pred.npz"},
        "diffusivity": 0.01,
    }


def test_heat_valid_config_roundtrips():
    spec = DomainSpec.model_validate(_valid_heat_dict())
    assert spec.pde == "heat"
    assert spec.diffusivity == 0.01
    assert spec.domain.is_time_dependent is True


def test_heat_without_diffusivity_raises():
    cfg = _valid_heat_dict()
    cfg["diffusivity"] = None
    with pytest.raises(ValidationError, match="diffusivity"):
        DomainSpec.model_validate(cfg)


def test_heat_without_time_domain_raises():
    cfg = _valid_heat_dict()
    cfg["domain"] = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    with pytest.raises(ValidationError, match="time domain"):
        DomainSpec.model_validate(cfg)


def test_wave_requires_wave_speed():
    cfg = _valid_heat_dict()
    cfg["pde"] = "wave"
    cfg["diffusivity"] = None
    cfg["wave_speed"] = None
    with pytest.raises(ValidationError, match="wave_speed"):
        DomainSpec.model_validate(cfg)


def test_bcspec_computed_properties():
    assert BCSpec(kind="periodic").conserves_mass is True
    assert BCSpec(kind="periodic").conserves_energy is True
    assert BCSpec(kind="periodic").preserves_sign is True
    assert BCSpec(kind="dirichlet_homogeneous").conserves_mass is False
    assert BCSpec(kind="dirichlet_homogeneous").conserves_energy is True
    assert BCSpec(kind="dirichlet_homogeneous").preserves_sign is True
    assert BCSpec(kind="dirichlet").conserves_mass is False
    assert BCSpec(kind="dirichlet").preserves_sign is False
    assert BCSpec(kind="neumann_homogeneous").conserves_mass is True


def test_symmetry_spec_accepts_both_c4_and_d4():
    ss = SymmetrySpec(declared=["C4", "D4", "reflection_x"])
    assert "C4" in ss.declared
    assert "D4" in ss.declared


def test_field_source_exactly_one_source():
    with pytest.raises(ValidationError, match="Exactly one"):
        FieldSourceSpec(type="grid", adapter_path=None, dump_path=None)
    with pytest.raises(ValidationError, match="Exactly one"):
        FieldSourceSpec(type="grid", adapter_path="a.py", dump_path="b.npz")
    # These should succeed:
    FieldSourceSpec(type="grid", adapter_path="a.py")
    FieldSourceSpec(type="grid", dump_path="b.npz")


def test_d4_on_non_square_domain_warns():
    cfg = _valid_heat_dict()
    cfg["domain"]["y"] = [0.0, 2.0]   # non-square
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        DomainSpec.model_validate(cfg)
        assert any("non-square" in str(warning.message).lower() or
                   "not square" in str(warning.message).lower()
                   for warning in w), f"Expected a D4-on-non-square warning; got {[str(x.message) for x in w]}"


def test_bcspec_unknown_kind_rejected():
    with pytest.raises(ValidationError):
        BCSpec(kind="wibble")   # type: ignore[arg-type]
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
pytest tests/test_spec.py -v
```

Expected: `ImportError: cannot import name 'DomainSpec'`

- [ ] **Step 3: Create `src/physics_lint/spec.py`**

```python
"""DomainSpec pydantic hierarchy — validated config for rules to consume.

Design doc §4. This is the single source of truth for physics-lint config;
rules read DomainSpec, never raw TOML. The hierarchy is a superset of the
user-writable config schema: computed properties (BCSpec.conserves_mass,
GridDomain.spatial_lengths, etc.) and derived fields (adapter_path /
dump_path on FieldSourceSpec) are populated during the merge path in
physics_lint.config.load_spec().
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

BCKind = Literal[
    "periodic",
    "dirichlet_homogeneous",
    "dirichlet",
    "neumann_homogeneous",
    "neumann",
]

PDEKind = Literal["laplace", "poisson", "heat", "wave"]

SymmetryLiteral = Literal[
    "D4",
    "C4",
    "reflection_x",
    "reflection_y",
    "translation_x",
    "translation_y",
    "SO2",
]

FieldType = Literal["grid", "callable", "mesh"]
FieldBackend = Literal["fd", "spectral", "auto"]


class GridDomain(BaseModel):
    """Spatial (and optionally temporal) domain extents."""

    x: tuple[float, float]
    y: tuple[float, float]
    t: Optional[tuple[float, float]] = None

    @property
    def spatial_lengths(self) -> tuple[float, ...]:
        return tuple(hi - lo for lo, hi in (self.x, self.y))

    @property
    def is_time_dependent(self) -> bool:
        return self.t is not None


class BCSpec(BaseModel):
    """Boundary condition with computed properties replacing rule-side BC taxonomy.

    The computed properties are the design doc §4.2 deduplication mechanism:
    per-rule conservation and sign-preservation checks read these booleans
    instead of re-encoding the PER/hN/hD logic in each rule.
    """

    kind: BCKind

    @property
    def preserves_sign(self) -> bool:
        return self.kind in {"dirichlet_homogeneous", "periodic"}

    @property
    def conserves_mass(self) -> bool:
        return self.kind in {"periodic", "neumann_homogeneous"}

    @property
    def conserves_energy(self) -> bool:
        return self.kind in {"periodic", "neumann_homogeneous", "dirichlet_homogeneous"}


class SymmetrySpec(BaseModel):
    """User-declared problem-instance symmetries.

    Not auto-detected. Design doc §9.4 explains why: operator-level admissibility
    is PDE-class-dependent but problem-instance symmetry depends on domain,
    source/IC, and BCs in ways physics-lint cannot mechanically verify.
    """

    declared: list[SymmetryLiteral] = Field(default_factory=list)


class FieldSourceSpec(BaseModel):
    """Field source: adapter module or dump file. Exactly one must be set."""

    type: FieldType
    backend: FieldBackend = "auto"
    adapter_path: Optional[str] = None
    dump_path: Optional[str] = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> "FieldSourceSpec":
        if (self.adapter_path is None) == (self.dump_path is None):
            raise ValueError(
                "Exactly one of adapter_path or dump_path must be set; "
                f"got adapter_path={self.adapter_path!r}, dump_path={self.dump_path!r}"
            )
        return self


class SARIFSpec(BaseModel):
    """Optional source-mapping config for SARIF Tier 2 (PR-check surfacing)."""

    source_file: Optional[str] = None
    pde_line: Optional[int] = None
    bc_line: Optional[int] = None
    symmetry_line: Optional[int] = None


class DomainSpec(BaseModel):
    """Top-level validated spec consumed by every rule."""

    pde: PDEKind
    grid_shape: tuple[int, ...] = Field(min_length=2, max_length=3)
    domain: GridDomain
    periodic: bool = False
    boundary_condition: BCSpec
    symmetries: SymmetrySpec = Field(default_factory=SymmetrySpec)
    field: FieldSourceSpec

    diffusivity: Optional[float] = None
    wave_speed: Optional[float] = None
    source_term: Optional[str] = None
    sarif: Optional[SARIFSpec] = None

    @model_validator(mode="after")
    def pde_params_consistent(self) -> "DomainSpec":
        if self.pde == "heat" and self.diffusivity is None:
            raise ValueError("PDE 'heat' requires 'diffusivity'")
        if self.pde == "wave" and self.wave_speed is None:
            raise ValueError("PDE 'wave' requires 'wave_speed'")
        if self.pde in {"heat", "wave"} and not self.domain.is_time_dependent:
            raise ValueError(
                f"PDE '{self.pde}' requires a time domain 't'; "
                "add 't = [t_start, t_end]' to [tool.physics-lint.domain]"
            )
        return self

    @model_validator(mode="after")
    def symmetries_compatible_with_domain(self) -> "DomainSpec":
        if any(s in self.symmetries.declared for s in ("D4", "C4")):
            lx, ly = self.domain.spatial_lengths[:2]
            if abs(lx - ly) / max(lx, ly) > 1e-6:
                warnings.warn(
                    f"D4/C4 symmetry declared but domain is not square "
                    f"({lx} × {ly}); symmetry rules may produce artifacts",
                    stacklevel=2,
                )
        return self
```

- [ ] **Step 4: Export from the top-level package**

Update `src/physics_lint/__init__.py`:

```python
"""physics-lint — linter for trained neural PDE surrogates."""

from physics_lint.field import CallableField, Field, GridField
from physics_lint.spec import (
    BCSpec,
    DomainSpec,
    FieldSourceSpec,
    GridDomain,
    SARIFSpec,
    SymmetrySpec,
)

__version__ = "0.0.0.dev0"
__all__ = [
    "__version__",
    "Field",
    "GridField",
    "CallableField",
    "DomainSpec",
    "GridDomain",
    "BCSpec",
    "SymmetrySpec",
    "FieldSourceSpec",
    "SARIFSpec",
]
```

- [ ] **Step 5: Run the DomainSpec tests and verify they pass**

```bash
pytest tests/test_spec.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 6: Export the JSON Schema for IDE autocomplete**

Create `scripts/generate_config_schema.py`:

```python
"""Regenerate src/physics_lint/data/config_schema.json from DomainSpec.

Run this in CI whenever DomainSpec changes so the committed schema stays
in sync with the runtime definition.
"""

import json
from pathlib import Path

from physics_lint.spec import DomainSpec


def main() -> None:
    schema = DomainSpec.model_json_schema()
    out = Path(__file__).parent.parent / "src" / "physics_lint" / "data" / "config_schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python scripts/generate_config_schema.py
```

Expected: `wrote .../config_schema.json`

- [ ] **Step 7: Commit**

```bash
ruff check src tests scripts && ruff format src tests scripts
git add src/physics_lint/spec.py src/physics_lint/__init__.py src/physics_lint/data/config_schema.json tests/test_spec.py scripts/generate_config_schema.py
git commit -m "feat(spec): DomainSpec pydantic v2 hierarchy with cross-validators

Per design doc §4. Five sub-models:

- GridDomain: x, y, optional t + computed spatial_lengths, is_time_dependent
- BCSpec: kind literal + computed preserves_sign, conserves_mass, conserves_energy
  (replaces per-rule BC taxonomy duplication per §4.2)
- SymmetrySpec: user-declared list of C4, D4, reflections, translations, SO2
- FieldSourceSpec: type/backend/adapter_path/dump_path + exactly_one_source validator
- SARIFSpec: optional source_file/pde_line/bc_line/symmetry_line
- DomainSpec top-level with pde_params_consistent and symmetries_compatible_with_domain
  cross-validators

JSON Schema regeneration script at scripts/generate_config_schema.py;
output committed to src/physics_lint/data/config_schema.json for IDE
autocomplete via evenBetterToml."
```

---

## Task 8: Config merge path + hybrid adapter+dump loader

**Files:**
- Create: `src/physics_lint/config.py`
- Create: `src/physics_lint/loader.py`
- Create: `tests/test_config.py`
- Create: `tests/test_loader.py`
- Create: `tests/fixtures/good_adapter.py`
- Create: `tests/fixtures/good_dump.py` (script to generate the test .npz on demand)

**Rationale:** The loader + merge path is the on-ramp from user input (TOML, adapter, dump, CLI) to the validated `DomainSpec` that rules consume. It's the single most concentrated piece of integration code in Week 1 — and per the Week 1 Day 3 rollback plan, the riskiest.

- [ ] **Step 1: Write failing tests for the config merge**

`tests/test_config.py`:

```python
"""Config merge path — TOML + adapter + CLI flags -> validated DomainSpec."""

from pathlib import Path

import pytest

from physics_lint import DomainSpec
from physics_lint.config import load_spec_from_toml, merge_into_spec


def _write_toml(tmp_path: Path, contents: str) -> Path:
    p = tmp_path / "pyproject.toml"
    p.write_text(contents)
    return p


_MINIMAL_LAPLACE_TOML = """
[tool.physics-lint]
pde = "laplace"
grid_shape = [64, 64]
domain = { x = [0.0, 1.0], y = [0.0, 1.0] }
periodic = false
boundary_condition = "dirichlet_homogeneous"

[tool.physics-lint.field]
type = "grid"
backend = "fd"
dump_path = "pred.npz"
"""


def test_load_toml_minimal_laplace(tmp_path: Path):
    path = _write_toml(tmp_path, _MINIMAL_LAPLACE_TOML)
    raw = load_spec_from_toml(path)
    assert raw["pde"] == "laplace"
    assert raw["boundary_condition"]["kind"] == "dirichlet_homogeneous"


def test_merge_cli_overrides(tmp_path: Path):
    path = _write_toml(tmp_path, _MINIMAL_LAPLACE_TOML)
    raw = load_spec_from_toml(path)
    merged = merge_into_spec(raw, adapter_spec=None, cli_overrides={"periodic": True})
    spec = DomainSpec.model_validate(merged)
    assert spec.periodic is True


def test_merge_adapter_overrides_toml(tmp_path: Path):
    path = _write_toml(tmp_path, _MINIMAL_LAPLACE_TOML)
    raw = load_spec_from_toml(path)
    adapter_spec = {
        "pde": "laplace",
        "grid_shape": [32, 32],     # override
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "field": {"type": "grid", "backend": "fd", "dump_path": "pred.npz"},
    }
    merged = merge_into_spec(raw, adapter_spec=adapter_spec, cli_overrides={})
    spec = DomainSpec.model_validate(merged)
    assert spec.grid_shape == (32, 32)


def test_missing_toml_falls_back_to_physics_lint_toml(tmp_path: Path):
    standalone = tmp_path / "physics-lint.toml"
    standalone.write_text(
        _MINIMAL_LAPLACE_TOML.replace("[tool.physics-lint]", "").replace(
            "[tool.physics-lint.field]", "[field]"
        )
    )
    raw = load_spec_from_toml(standalone)
    assert raw["pde"] == "laplace"


def test_invalid_toml_raises_config_error(tmp_path: Path):
    path = _write_toml(
        tmp_path,
        """
        [tool.physics-lint]
        pde = "heat"
        grid_shape = [64, 64]
        domain = { x = [0.0, 1.0], y = [0.0, 1.0] }
        boundary_condition = "dirichlet_homogeneous"
        [tool.physics-lint.field]
        type = "grid"
        dump_path = "p.npz"
        # heat requires diffusivity and a time domain -- missing both
        """,
    )
    raw = load_spec_from_toml(path)
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        DomainSpec.model_validate(merge_into_spec(raw, adapter_spec=None, cli_overrides={}))
```

- [ ] **Step 2: Run config tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: `ImportError: cannot import name 'load_spec_from_toml'`

- [ ] **Step 3: Implement `src/physics_lint/config.py`**

```python
"""Config loading and merge path: TOML + adapter + CLI -> dict (validated by DomainSpec).

Design doc §12.4. Load order:

    1. Shipped defaults (live in DomainSpec field defaults)
    2. pyproject.toml [tool.physics-lint]  (or standalone physics-lint.toml fallback)
    3. Adapter domain_spec() return value
    4. CLI flag overrides

The merge returns a plain dict; DomainSpec.model_validate() is called at the
end of load_spec() in loader.py as the single validation point.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def load_spec_from_toml(path: Path) -> dict[str, Any]:
    """Read [tool.physics-lint] from pyproject.toml OR the top table of physics-lint.toml.

    Raises FileNotFoundError if the file doesn't exist.
    Returns the raw dict with nested `boundary_condition` wrapped as a dict
    so it matches BCSpec's shape (users write `boundary_condition = "periodic"`;
    we expand to `{"kind": "periodic"}`).
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    if "tool" in data and "physics-lint" in data["tool"]:
        raw = data["tool"]["physics-lint"]
    else:
        raw = data   # physics-lint.toml fallback: whole file is the spec

    return _normalize_config_shape(raw)


def _normalize_config_shape(raw: dict) -> dict:
    """Expand user-friendly config shapes into the shapes DomainSpec expects."""
    raw = dict(raw)  # shallow copy

    bc = raw.get("boundary_condition")
    if isinstance(bc, str):
        raw["boundary_condition"] = {"kind": bc}

    sym = raw.get("symmetries")
    if isinstance(sym, list):
        raw["symmetries"] = {"declared": sym}

    return raw


def merge_into_spec(
    toml_spec: dict,
    *,
    adapter_spec: dict | None,
    cli_overrides: dict,
) -> dict:
    """Merge four sources into a single dict ready for DomainSpec.model_validate().

    Precedence (later overrides earlier):
        1. toml_spec (from load_spec_from_toml)
        2. adapter_spec (from adapter.domain_spec().model_dump())
        3. cli_overrides (from CLI flags)

    Top-level keys override wholesale; nested dicts are merged one level deep
    so that `[tool.physics-lint.field]` can be partially overridden by an
    adapter's `field` return without clobbering the other keys.
    """
    merged: dict[str, Any] = dict(toml_spec)

    if adapter_spec is not None:
        merged = _deep_merge_one_level(merged, _normalize_config_shape(adapter_spec))

    if cli_overrides:
        merged = _deep_merge_one_level(merged, _normalize_config_shape(cli_overrides))

    return merged


def _deep_merge_one_level(base: dict, override: dict) -> dict:
    """Merge `override` into `base`. For dict-valued keys, merge one level deep."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out
```

- [ ] **Step 4: Run config tests and verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Write failing loader tests**

Create `tests/fixtures/__init__.py` as empty file, then `tests/fixtures/good_adapter.py`:

```python
"""Test fixture: a valid adapter with load_model() and domain_spec()."""

import numpy as np
import torch

from physics_lint import DomainSpec


def load_model():
    # Trivial "model": returns zero everywhere; quadratic field for Laplacian test
    def _model(x: torch.Tensor) -> torch.Tensor:
        return (x[..., 0] ** 2 - x[..., 1] ** 2).unsqueeze(-1)
    return _model


def domain_spec() -> DomainSpec:
    return DomainSpec.model_validate({
        "pde": "laplace",
        "grid_shape": [32, 32],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet"},
        "field": {"type": "callable", "adapter_path": __file__},
    })
```

Create `tests/fixtures/good_dump.py`:

```python
"""Test fixture: generate a valid .npz dump file on demand.

Called by test_loader.py via `_write_good_dump(tmp_path)`. Not committed as
a .npz artifact because the dump file should be regenerated fresh for each
test run to catch silent drift.
"""

from pathlib import Path

import numpy as np


def write_good_dump(path: Path) -> Path:
    N = 32
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = X ** 2 - Y ** 2    # harmonic on [0,1]^2
    metadata = {
        "pde": "laplace",
        "grid_shape": [N, N],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": "dirichlet",
        "field": {"type": "grid", "backend": "fd"},
    }
    np.savez(path, prediction=u, metadata=metadata)
    return path
```

`tests/test_loader.py`:

```python
"""Hybrid adapter+dump loader tests.

Extension dispatch per design doc §5.1:
    .py         -> adapter path (exec + load_model + domain_spec)
    .npz / .npy -> dump path (np.load + metadata dict -> DomainSpec)
    .pt / .pth  -> error
"""

from pathlib import Path

import pytest

from physics_lint import DomainSpec, GridField
from physics_lint.loader import LoaderError, load_target


def test_load_adapter(tmp_path: Path, monkeypatch):
    import shutil
    fixture_src = Path(__file__).parent / "fixtures" / "good_adapter.py"
    adapter_copy = tmp_path / "physics_lint_adapter.py"
    shutil.copy(fixture_src, adapter_copy)

    loaded = load_target(adapter_copy, cli_overrides={}, toml_path=None)
    assert isinstance(loaded.spec, DomainSpec)
    assert loaded.spec.pde == "laplace"
    assert loaded.model is not None  # adapter's load_model() returned a callable
    assert callable(loaded.model)


def test_load_dump(tmp_path: Path):
    from tests.fixtures.good_dump import write_good_dump
    dump_path = write_good_dump(tmp_path / "pred.npz")

    loaded = load_target(dump_path, cli_overrides={}, toml_path=None)
    assert isinstance(loaded.spec, DomainSpec)
    assert loaded.spec.pde == "laplace"
    assert loaded.model is None                        # dump mode: no callable
    assert isinstance(loaded.field, GridField)


def test_load_pt_file_errors(tmp_path: Path):
    p = tmp_path / "model.pt"
    p.write_bytes(b"\x80\x04")   # fake torch pickle header
    with pytest.raises(LoaderError, match="adapter or convert to .npz"):
        load_target(p, cli_overrides={}, toml_path=None)


def test_load_unknown_extension_errors(tmp_path: Path):
    p = tmp_path / "model.bin"
    p.write_bytes(b"")
    with pytest.raises(LoaderError, match="unsupported"):
        load_target(p, cli_overrides={}, toml_path=None)
```

- [ ] **Step 6: Run loader tests to verify they fail**

```bash
pytest tests/test_loader.py -v
```

Expected: `ImportError: cannot import name 'load_target'`

- [ ] **Step 7: Implement `src/physics_lint/loader.py`**

```python
"""Hybrid adapter+dump loader with extension-based dispatch.

Design doc §5. Entry point: load_target(path, cli_overrides, toml_path)
returns a LoadedTarget(spec, field, model) where:

- spec: validated DomainSpec
- field: a Field instance materialized from the target (GridField for dumps,
  CallableField for adapter with model, or None if the rule will create it later)
- model: the adapter's load_model() return (None in dump mode)

The loader is the single place where user-supplied Python (`exec`) or user-
supplied data files (`np.load`) enter physics-lint. All subsequent code
assumes inputs have been validated.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from physics_lint import DomainSpec
from physics_lint.config import load_spec_from_toml, merge_into_spec
from physics_lint.field import CallableField, Field, GridField


class LoaderError(RuntimeError):
    """Raised when physics-lint cannot load the user's target."""


@dataclass
class LoadedTarget:
    spec: DomainSpec
    field: Field
    model: Optional[Callable[..., Any]]


def load_target(
    path: Path,
    *,
    cli_overrides: dict[str, Any],
    toml_path: Optional[Path],
) -> LoadedTarget:
    """Load a target file, merge config, and return a LoadedTarget."""
    path = Path(path)
    suffix = path.suffix.lower()

    toml_spec: dict[str, Any] = {}
    if toml_path is not None:
        toml_spec = load_spec_from_toml(toml_path)

    if suffix == ".py":
        return _load_adapter(path, toml_spec=toml_spec, cli_overrides=cli_overrides)
    if suffix in (".npz", ".npy"):
        return _load_dump(path, toml_spec=toml_spec, cli_overrides=cli_overrides)
    if suffix in (".pt", ".pth"):
        raise LoaderError(
            f"{path.name}: .pt/.pth files are not supported directly. "
            "Please use an adapter or convert to .npz; see docs/loading.html"
        )
    raise LoaderError(f"{path.name}: unsupported file extension {suffix}")


def _load_adapter(
    path: Path,
    *,
    toml_spec: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> LoadedTarget:
    """Load a user adapter module: exec + call load_model() and domain_spec()."""
    if not path.is_file():
        raise LoaderError(f"adapter file not found: {path}")

    module_name = f"_physics_lint_adapter_{path.stem}_{id(path)}"
    spec_obj = importlib.util.spec_from_file_location(module_name, str(path))
    if spec_obj is None or spec_obj.loader is None:
        raise LoaderError(f"cannot import adapter from {path}")
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    try:
        spec_obj.loader.exec_module(module)
    except Exception as e:
        raise LoaderError(f"adapter {path} raised during import: {e}") from e

    if not hasattr(module, "load_model"):
        raise LoaderError(f"adapter {path} missing required load_model() function")
    if not hasattr(module, "domain_spec"):
        raise LoaderError(f"adapter {path} missing required domain_spec() function")

    try:
        model = module.load_model()
    except Exception as e:
        raise LoaderError(f"adapter {path}.load_model() raised: {e}") from e
    try:
        adapter_spec_obj = module.domain_spec()
    except Exception as e:
        raise LoaderError(f"adapter {path}.domain_spec() raised: {e}") from e

    adapter_spec_dict: dict[str, Any]
    if isinstance(adapter_spec_obj, DomainSpec):
        adapter_spec_dict = adapter_spec_obj.model_dump()
    elif isinstance(adapter_spec_obj, dict):
        adapter_spec_dict = adapter_spec_obj
    else:
        raise LoaderError(
            f"adapter {path}.domain_spec() must return DomainSpec or dict; "
            f"got {type(adapter_spec_obj).__name__}"
        )
    # Make sure the adapter source path is recorded
    adapter_spec_dict.setdefault("field", {})
    adapter_spec_dict["field"]["adapter_path"] = str(path)
    adapter_spec_dict["field"].pop("dump_path", None)

    merged = merge_into_spec(toml_spec, adapter_spec=adapter_spec_dict, cli_overrides=cli_overrides)
    spec = DomainSpec.model_validate(merged)

    # For Week 1 we materialize the callable onto a GridField via CallableField.
    # Build a sampling grid from the spec.
    import torch

    grid_tensor = _build_sampling_grid(spec)
    field = CallableField(
        model=model,
        sampling_grid=grid_tensor,
        h=_compute_h_from_spec(spec),
        periodic=spec.periodic,
    )
    return LoadedTarget(spec=spec, field=field, model=model)


def _load_dump(
    path: Path,
    *,
    toml_spec: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> LoadedTarget:
    """Load a .npz dump: read prediction + metadata and wrap in GridField."""
    if not path.is_file():
        raise LoaderError(f"dump file not found: {path}")

    loaded = np.load(path, allow_pickle=True)
    if "prediction" not in loaded.files:
        raise LoaderError(f"{path}: .npz must contain a 'prediction' array")
    prediction = loaded["prediction"]
    metadata_raw = loaded.get("metadata") if "metadata" in loaded.files else None
    if metadata_raw is None:
        adapter_spec_dict = {}
    else:
        # np.savez wraps dicts in 0-dim object arrays
        adapter_spec_dict = metadata_raw.item() if metadata_raw.shape == () else dict(metadata_raw)
    if not isinstance(adapter_spec_dict, dict):
        raise LoaderError(f"{path}: metadata must be a dict")

    adapter_spec_dict.setdefault("field", {})
    adapter_spec_dict["field"]["dump_path"] = str(path)
    adapter_spec_dict["field"].pop("adapter_path", None)

    merged = merge_into_spec(toml_spec, adapter_spec=adapter_spec_dict, cli_overrides=cli_overrides)
    spec = DomainSpec.model_validate(merged)

    h = _compute_h_from_spec(spec)
    field = GridField(
        prediction,
        h=h,
        periodic=spec.periodic,
        backend=spec.field.backend if spec.field.backend != "auto" else (
            "spectral" if spec.periodic else "fd"
        ),
    )
    return LoadedTarget(spec=spec, field=field, model=None)


def _compute_h_from_spec(spec: DomainSpec) -> tuple[float, ...]:
    """Derive uniform grid spacings from domain extents and grid_shape."""
    lengths = spec.domain.spatial_lengths
    # endpoint-inclusive for non-periodic, endpoint-exclusive for periodic
    shape = spec.grid_shape
    if spec.periodic:
        return tuple(L / N for L, N in zip(lengths, shape[: len(lengths)]))
    return tuple(L / (N - 1) for L, N in zip(lengths, shape[: len(lengths)]))


def _build_sampling_grid(spec: DomainSpec) -> "torch.Tensor":
    import torch
    lengths = spec.domain.spatial_lengths
    shape = spec.grid_shape[: len(lengths)]
    axes = []
    for L, N in zip(lengths, shape):
        if spec.periodic:
            axes.append(torch.linspace(0.0, L, N + 1)[:-1])
        else:
            axes.append(torch.linspace(0.0, L, N))
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    return grid
```

- [ ] **Step 8: Run loader tests and verify they pass**

```bash
pytest tests/test_loader.py -v
```

Expected: all 4 PASS.

- [ ] **Step 9: Run the full suite**

```bash
pytest -q
```

Expected: everything green.

- [ ] **Step 10: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/physics_lint/config.py src/physics_lint/loader.py tests/test_config.py tests/test_loader.py tests/fixtures/
git commit -m "feat(loader,config): hybrid adapter+dump loader with TOML merge path

Per design doc §5 and §12.4.

config.py:
- load_spec_from_toml() reads pyproject.toml [tool.physics-lint] or
  standalone physics-lint.toml fallback
- merge_into_spec() composes TOML + adapter_spec + cli_overrides with
  precedence (later overrides earlier), one-level-deep merge for nested dicts
- _normalize_config_shape expands user-friendly forms (string BC, list
  symmetries) into the shapes DomainSpec expects

loader.py:
- load_target() dispatches on file extension (.py adapter, .npz/.npy dump,
  .pt/.pth -> LoaderError with conversion guidance)
- _load_adapter exec's the user module, calls load_model() + domain_spec(),
  builds a CallableField against the materialized sampling grid
- _load_dump reads prediction + metadata from np.load, wraps in GridField
- Single DomainSpec.model_validate() call at the end of each branch —
  pydantic owns all validation per the 'single validation point' discipline

Tests cover minimal Laplace TOML, CLI override, adapter override, TOML
fallback, invalid-config error path, .py/.npz/.pt dispatch, and two
fixture files (good_adapter.py, good_dump.py)."
```

---

## Task 9: Report schema + rule registry scaffolding + analytical battery

**Files:**
- Create: `src/physics_lint/report.py`
- Create: `src/physics_lint/rules/__init__.py`
- Create: `src/physics_lint/rules/_registry.py`
- Create: `src/physics_lint/analytical/__init__.py`
- Create: `src/physics_lint/analytical/laplace.py`
- Create: `src/physics_lint/analytical/poisson.py`
- Create: `tests/test_report.py`
- Create: `tests/test_registry.py`
- Create: `tests/test_analytical.py`

**Rationale:** Three thin but load-bearing pieces: the `RuleResult`/`PhysicsLintReport` dataclasses (needed by every rule), the lazy rule registry (needed by `rules list` later but also by the executor in Task 10), and the analytical battery (needed by `floors.toml` in Task 14). Consolidating them reduces the task count without increasing the risk since all three are small and independent.

- [ ] **Step 1: Write failing tests for RuleResult and PhysicsLintReport**

`tests/test_report.py`:

```python
"""Report schema tests — RuleResult + PhysicsLintReport with SKIPPED handling."""

from physics_lint.report import PhysicsLintReport, RuleResult


def _rr(rule_id: str, status: str, severity: str = "error", **kw) -> RuleResult:
    return RuleResult(
        rule_id=rule_id,
        rule_name=f"{rule_id} name",
        severity=severity,
        status=status,
        raw_value=kw.get("raw_value"),
        violation_ratio=kw.get("violation_ratio"),
        mode=kw.get("mode"),
        reason=kw.get("reason"),
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="H^-1",
        citation="Week 1 plan Task 9",
        doc_url="https://physics-lint.readthedocs.io/rules/",
    )


def test_overall_status_all_pass():
    r = PhysicsLintReport(
        pde="laplace", grid_shape=(64, 64), metadata={},
        rules=[_rr("PH-RES-001", "PASS"), _rr("PH-BC-001", "PASS")],
    )
    assert r.overall_status == "PASS"
    assert r.exit_code == 0


def test_overall_status_with_skipped_is_not_warn():
    r = PhysicsLintReport(
        pde="laplace", grid_shape=(64, 64), metadata={},
        rules=[
            _rr("PH-RES-001", "PASS"),
            _rr("PH-SYM-003", "SKIPPED", reason="dump mode"),
            _rr("PH-BC-001", "PASS"),
        ],
    )
    assert r.overall_status == "PASS"     # SKIPPED rank == PASS rank
    assert r.exit_code == 0
    counts = r.status_counts
    assert counts == {"PASS": 2, "WARN": 0, "FAIL": 0, "SKIPPED": 1}


def test_overall_status_warn_beats_pass():
    r = PhysicsLintReport(
        pde="laplace", grid_shape=(64, 64), metadata={},
        rules=[_rr("PH-RES-001", "PASS"), _rr("PH-SYM-001", "WARN", severity="warning")],
    )
    assert r.overall_status == "WARN"
    assert r.exit_code == 0    # WARN does not trigger non-zero exit


def test_overall_status_fail_beats_warn():
    r = PhysicsLintReport(
        pde="laplace", grid_shape=(64, 64), metadata={},
        rules=[
            _rr("PH-RES-001", "PASS"),
            _rr("PH-SYM-001", "WARN", severity="warning"),
            _rr("PH-POS-001", "FAIL"),
        ],
    )
    assert r.overall_status == "FAIL"
    assert r.exit_code == 1


def test_fail_of_warning_severity_does_not_trigger_exit_code():
    r = PhysicsLintReport(
        pde="laplace", grid_shape=(64, 64), metadata={},
        rules=[_rr("PH-SYM-001", "FAIL", severity="warning")],
    )
    assert r.overall_status == "FAIL"
    assert r.exit_code == 0
```

- [ ] **Step 2: Run the report tests to verify they fail**

```bash
pytest tests/test_report.py -v
```

Expected: `ImportError: cannot import name 'PhysicsLintReport'`

- [ ] **Step 3: Implement `src/physics_lint/report.py`**

```python
"""Report schema: RuleResult and PhysicsLintReport dataclasses.

Design doc §11. _STATUS_RANK is module-level (not inline) so overall_status
and any future sort/filter operations share one source of truth. SKIPPED
has rank 0 (same as PASS) — a skipped rule never moves overall status,
and status_counts always reports all four keys with explicit zeros.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

Status = Literal["PASS", "WARN", "FAIL", "SKIPPED"]
Severity = Literal["error", "warning", "info"]

_STATUS_RANK: dict[Status, int] = {"SKIPPED": 0, "PASS": 0, "WARN": 1, "FAIL": 2}
_STATUS_ORDER: tuple[Status, ...] = ("PASS", "WARN", "FAIL", "SKIPPED")


@dataclass
class RuleResult:
    rule_id: str
    rule_name: str
    severity: Severity
    status: Status
    raw_value: Optional[float]
    violation_ratio: Optional[float]
    mode: Optional[str]
    reason: Optional[str]
    refinement_rate: Optional[float]
    spatial_map: Optional[np.ndarray]
    recommended_norm: str
    citation: str
    doc_url: str


@dataclass
class PhysicsLintReport:
    pde: str
    grid_shape: tuple[int, ...]
    rules: list[RuleResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_status(self) -> Status:
        if not self.rules:
            return "PASS"
        return max(self.rules, key=lambda r: _STATUS_RANK[r.status]).status

    @property
    def status_counts(self) -> dict[Status, int]:
        counts = Counter(r.status for r in self.rules)
        return {s: counts.get(s, 0) for s in _STATUS_ORDER}

    @property
    def exit_code(self) -> int:
        """Non-zero iff any error-severity rule has status FAIL. SKIPPED is ignored."""
        return int(any(r.status == "FAIL" and r.severity == "error" for r in self.rules))
```

- [ ] **Step 4: Run report tests and verify they pass**

```bash
pytest tests/test_report.py -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Write failing test for the lazy rule registry**

`tests/test_registry.py`:

```python
"""Rule registry tests — lazy metadata discovery.

The registry reads __rule_id__, __rule_name__, __default_severity__, and
__input_modes__ from rule modules WITHOUT importing their check() functions,
so `physics-lint rules list` stays under 50 ms.
"""

from physics_lint.rules import _registry


def test_registry_discovers_at_least_placeholder():
    # After Task 10 this asserts the full Week-1 rule set; for now we just
    # ensure the registry can iterate and find the placeholder module.
    rules = _registry.list_rules()
    ids = {r.rule_id for r in rules}
    assert "PH-PLACEHOLDER-000" in ids, f"expected placeholder module; found {ids}"


def test_registry_lazy_check_not_imported():
    # Iterating metadata should NOT import the check() functions.
    rules = _registry.list_rules()
    for r in rules:
        assert r.check_fn is None, f"{r.rule_id} check was eagerly imported"


def test_registry_materialize_check():
    rules = _registry.list_rules()
    placeholder = next(r for r in rules if r.rule_id == "PH-PLACEHOLDER-000")
    check = _registry.load_check(placeholder)
    assert callable(check)
```

- [ ] **Step 6: Run registry tests to verify they fail**

```bash
pytest tests/test_registry.py -v
```

Expected: `ImportError: No module named 'physics_lint.rules._registry'`

- [ ] **Step 7: Implement the rule registry and a placeholder rule module**

`src/physics_lint/rules/__init__.py`:

```python
"""Rule catalog — lazy-discovered at import time.

Do NOT eagerly import rule check() functions here; the _registry module
scans for __rule_id__, __rule_name__, __default_severity__, __input_modes__
at the module level and defers check() imports until a rule is actually
invoked. This keeps `physics-lint rules list` fast.
"""

from physics_lint.rules._registry import list_rules, load_check

__all__ = ["list_rules", "load_check"]
```

`src/physics_lint/rules/_registry.py`:

```python
"""Lazy rule discovery.

Each rule module at physics_lint/rules/ph_xxx_nnn.py exports four module-level
attributes: __rule_id__, __rule_name__, __default_severity__, __input_modes__
(a frozenset of {"adapter", "dump"}). The registry walks the rules package,
reads ONLY those four attributes (no check() import), and returns a list of
RegistryEntry. The check() function is loaded on demand via load_check().

This is the Section-2-review rollback pattern: if Day 3 converges lazy
discovery, rules list runs in <50 ms; if not, Week 4 Day 2 can switch to
eager discovery (import every module), at the cost of ~500 ms on
rules list, without changing any rule-module code.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class RegistryEntry:
    rule_id: str
    rule_name: str
    default_severity: str
    input_modes: frozenset[str]
    module_name: str
    check_fn: Optional[Callable[..., Any]] = None


def list_rules() -> list[RegistryEntry]:
    """Scan physics_lint.rules for rule modules; return metadata-only entries."""
    import physics_lint.rules as rules_pkg

    entries: list[RegistryEntry] = []
    for mod_info in pkgutil.iter_modules(rules_pkg.__path__):
        name = mod_info.name
        if name.startswith("_"):
            continue
        full_name = f"physics_lint.rules.{name}"
        # Read metadata without executing check(): use importlib.util to
        # load the module, then discard any non-metadata attributes.
        spec = importlib.util.find_spec(full_name)
        if spec is None:
            continue
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        rule_id = getattr(module, "__rule_id__", None)
        if rule_id is None:
            continue
        entries.append(
            RegistryEntry(
                rule_id=rule_id,
                rule_name=getattr(module, "__rule_name__", ""),
                default_severity=getattr(module, "__default_severity__", "warning"),
                input_modes=frozenset(getattr(module, "__input_modes__", ())),
                module_name=full_name,
                check_fn=None,     # NOT loaded here
            )
        )
    return sorted(entries, key=lambda e: e.rule_id)


def load_check(entry: RegistryEntry) -> Callable[..., Any]:
    """Import the rule module for real and return its check function."""
    if entry.check_fn is not None:
        return entry.check_fn
    module = importlib.import_module(entry.module_name)
    check = getattr(module, "check", None)
    if check is None:
        raise AttributeError(f"{entry.module_name} has no `check` function")
    entry.check_fn = check
    return check
```

`src/physics_lint/rules/ph_placeholder_000.py`:

```python
"""Placeholder rule used by the Week 1 Day 3 registry tests.

Deleted in Task 10 once the first real rule (PH-RES-001) lands. Kept here
only so the registry has something to discover before any real rule exists.
"""

__rule_id__ = "PH-PLACEHOLDER-000"
__rule_name__ = "Placeholder (Week 1 Day 3 only)"
__default_severity__ = "info"
__input_modes__ = frozenset({"adapter", "dump"})


def check(field, spec):
    from physics_lint.report import RuleResult
    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status="PASS",
        raw_value=0.0,
        violation_ratio=0.0,
        mode=None,
        reason=None,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="",
        citation="",
        doc_url="",
    )
```

- [ ] **Step 8: Run registry tests and verify they pass**

```bash
pytest tests/test_registry.py -v
```

Expected: all 3 PASS.

- [ ] **Step 9: Write failing test for the analytical battery**

`tests/test_analytical.py`:

```python
"""Analytical battery tests — each returns a function + the exact Laplacian."""

import numpy as np

from physics_lint.analytical import laplace as laplace_sols
from physics_lint.analytical import poisson as poisson_sols


def test_laplace_harmonic_polynomial_satisfies_pde():
    sol = laplace_sols.harmonic_polynomial_square()
    N = 64
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = sol.u(X, Y)
    lap = sol.laplacian(X, Y)
    # Laplace: -Delta u = 0 so the analytical Laplacian should be zero
    assert np.max(np.abs(lap)) < 1e-15
    # Value sanity: u = x^2 - y^2
    assert np.allclose(u, X ** 2 - Y ** 2)


def test_laplace_eigen_trace_satisfies_pde():
    sol = laplace_sols.eigen_trace_square(n=1)
    N = 64
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    lap = sol.laplacian(X, Y)
    assert np.max(np.abs(lap)) < 1e-14


def test_poisson_sin_sin_mms_residual_matches_source():
    sol = poisson_sols.sin_sin_mms_square()
    N = 64
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = sol.u(X, Y)
    f = sol.source(X, Y)
    # -Delta u = 2 pi^2 sin(pi x) sin(pi y), which is f
    expected = 2 * np.pi ** 2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    assert np.allclose(f, expected)
    # Analytical Laplacian
    lap = sol.laplacian(X, Y)
    assert np.allclose(-lap, f)


def test_poisson_periodic_mms_roundtrip():
    sol = poisson_sols.periodic_sin_sin()
    N = 64
    x = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = sol.u(X, Y)
    f = sol.source(X, Y)
    lap = sol.laplacian(X, Y)
    assert np.allclose(-lap, f)
```

- [ ] **Step 10: Run analytical tests to verify they fail**

```bash
pytest tests/test_analytical.py -v
```

Expected: `ModuleNotFoundError: No module named 'physics_lint.analytical'`

- [ ] **Step 11: Implement the analytical battery**

`src/physics_lint/analytical/__init__.py`:

```python
"""Analytical solutions for the self-test battery.

Each solution module exposes factory functions returning an AnalyticalSolution
dataclass with callable `u(X, Y)`, `laplacian(X, Y)`, and optional `source(X, Y)`
(for Poisson). Heat/wave variants land in Week 2.
"""
```

`src/physics_lint/analytical/laplace.py`:

```python
"""Analytical solutions for Laplace's equation (Delta u = 0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class LaplaceSolution:
    name: str
    u: Callable[[np.ndarray, np.ndarray], np.ndarray]
    laplacian: Callable[[np.ndarray, np.ndarray], np.ndarray]


def harmonic_polynomial_square() -> LaplaceSolution:
    """u(x,y) = x^2 - y^2 on [0,1]^2 with Dirichlet trace. Harmonic."""
    return LaplaceSolution(
        name="harmonic_polynomial_square",
        u=lambda X, Y: X ** 2 - Y ** 2,
        laplacian=lambda X, Y: np.zeros_like(X),
    )


def eigen_trace_square(n: int = 1) -> LaplaceSolution:
    """u = sin(n pi x) sinh(n pi y) / sinh(n pi) on [0,1]^2 with inhomogeneous
    Dirichlet trace (0 on three sides, sin(n pi x) on y=1). Still harmonic."""
    def u(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.sin(n * np.pi * X) * np.sinh(n * np.pi * Y) / np.sinh(n * np.pi)

    def lap(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.zeros_like(X)

    return LaplaceSolution(name=f"eigen_trace_square_n{n}", u=u, laplacian=lap)
```

`src/physics_lint/analytical/poisson.py`:

```python
"""Analytical solutions for Poisson's equation (-Delta u = f)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PoissonSolution:
    name: str
    u: Callable[[np.ndarray, np.ndarray], np.ndarray]
    laplacian: Callable[[np.ndarray, np.ndarray], np.ndarray]
    source: Callable[[np.ndarray, np.ndarray], np.ndarray]  # f = -Delta u


def sin_sin_mms_square() -> PoissonSolution:
    """u(x,y) = sin(pi x) sin(pi y) on [0,1]^2 with hD. -Delta u = 2 pi^2 u."""
    def u(X, Y):
        return np.sin(np.pi * X) * np.sin(np.pi * Y)

    def lap(X, Y):
        return -2 * np.pi ** 2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

    def source(X, Y):
        return 2 * np.pi ** 2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

    return PoissonSolution(name="sin_sin_mms_square", u=u, laplacian=lap, source=source)


def periodic_sin_sin() -> PoissonSolution:
    """u = sin(x) sin(y) on [0, 2 pi]^2 periodic. -Delta u = 2 sin(x) sin(y)."""
    def u(X, Y):
        return np.sin(X) * np.sin(Y)

    def lap(X, Y):
        return -2 * np.sin(X) * np.sin(Y)

    def source(X, Y):
        return 2 * np.sin(X) * np.sin(Y)

    return PoissonSolution(name="periodic_sin_sin", u=u, laplacian=lap, source=source)
```

- [ ] **Step 12: Run analytical tests and verify they pass**

```bash
pytest tests/test_analytical.py -v
```

Expected: all 4 PASS.

- [ ] **Step 13: Run the full test suite and commit**

```bash
pytest -q
ruff check src tests && ruff format src tests
git add src/physics_lint/report.py src/physics_lint/rules/ src/physics_lint/analytical/ tests/test_report.py tests/test_registry.py tests/test_analytical.py
git commit -m "feat(report,rules,analytical): report schema, lazy rule registry, Laplace/Poisson battery

Per design doc §11 (report schema) and §10.4 (lazy registry) and §6.1 (battery).

report.py:
- RuleResult dataclass with all fields from §11 including mode + reason
- PhysicsLintReport with _STATUS_RANK (SKIPPED=0=PASS) module-level
- status_counts always reports all four keys with explicit zeros
- exit_code ignores SKIPPED (prevents the Section-2-review KeyError bug)

rules/_registry.py:
- Lazy discovery via pkgutil.iter_modules + importlib.util.find_spec
- Reads only __rule_id__, __rule_name__, __default_severity__, __input_modes__
- load_check() defers the real check() import until first use
- Placeholder rule PH-PLACEHOLDER-000 for registry tests; deleted in Task 10

analytical/:
- Laplace: harmonic polynomial (x^2 - y^2), eigen trace (sin sinh)
- Poisson: sin*sin MMS on [0,1]^2 hD, periodic sin*sin on [0, 2 pi]^2

All analytical Laplacians verified to machine precision against the
manufactured sources."
```

---

## Task 10: Residual rules — PH-RES-001, PH-RES-002, PH-RES-003 for Laplace/Poisson

**Files:**
- Delete: `src/physics_lint/rules/ph_placeholder_000.py`
- Create: `src/physics_lint/rules/ph_res_001.py`
- Create: `src/physics_lint/rules/ph_res_002.py`
- Create: `src/physics_lint/rules/ph_res_003.py`
- Create: `src/physics_lint/rules/_helpers.py` (shared `violation_ratio` + `_tristate` + `_load_floor`)
- Create: `tests/rules/test_ph_res_001.py`
- Create: `tests/rules/test_ph_res_002.py`
- Create: `tests/rules/test_ph_res_003.py`
- Modify: `tests/test_registry.py` (switch the placeholder assertion to assert real rules)

**Rationale:** `PH-RES-001` is the headline residual rule — variationally-correct $H^{-1}$ norm for Laplace/Poisson, tri-state against the calibrated floor. `PH-RES-002` is the FD-vs-AD cross-check (adapter-only, SKIPs in dump mode). `PH-RES-003` is the spectral-vs-FD cross-check on periodic grids. All three share a common structure (compute residual, compute norm, compute ratio, emit `RuleResult`).

- [ ] **Step 1: Write a shared helper test (floor loader + tristate)**

`tests/rules/__init__.py`: empty file.

`tests/rules/test_helpers.py`:

```python
"""Shared rule helpers: _tristate and _load_floor."""

import pytest

from physics_lint.rules._helpers import _load_floor, _tristate


def test_tristate_pass():
    assert _tristate(ratio=5.0, pass_=10.0, fail_=100.0) == "PASS"


def test_tristate_warn():
    assert _tristate(ratio=50.0, pass_=10.0, fail_=100.0) == "WARN"


def test_tristate_fail():
    assert _tristate(ratio=500.0, pass_=10.0, fail_=100.0) == "FAIL"


def test_tristate_boundary_inclusive():
    # ratio == pass_ is still PASS; ratio == fail_ is WARN (not FAIL)
    assert _tristate(10.0, 10.0, 100.0) == "PASS"
    assert _tristate(100.0, 10.0, 100.0) == "WARN"


def test_load_floor_returns_shipped_defaults_before_calibration():
    # Week 1 Day 4: floors.toml is empty; helper should return a conservative
    # default so rules still compute violation_ratio without KeyError.
    floor = _load_floor(rule="PH-RES-001", pde="laplace", grid_shape=(64, 64),
                        method="fd4", norm="H-1")
    assert floor.value > 0
    assert floor.tolerance >= 1.0
```

- [ ] **Step 2: Run helper tests to verify they fail**

```bash
pytest tests/rules/test_helpers.py -v
```

Expected: `ImportError: cannot import name '_load_floor'`

- [ ] **Step 3: Implement `src/physics_lint/rules/_helpers.py`**

```python
"""Shared rule helpers: tristate thresholds, floor loading, safe ratios.

Floors are loaded from physics_lint/data/floors.toml; until Task 14
populates that file, a conservative shipped default is returned so
rules can still compute a violation_ratio. The default is intentionally
pessimistic (large) so that real floor calibration later produces
violation_ratios < the shipped values — avoids spurious PASS on uncalibrated
installs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

Status = Literal["PASS", "WARN", "FAIL", "SKIPPED"]


@dataclass
class Floor:
    value: float
    tolerance: float
    source: str    # "shipped" | "calibrated"


_SHIPPED_DEFAULTS: dict[tuple[str, str, str, str], float] = {
    # Conservative defaults; refined by Task 14 floors.toml
    ("PH-RES-001", "laplace", "fd4", "H-1"): 1e-5,
    ("PH-RES-001", "laplace", "spectral", "H-1"): 1e-13,
    ("PH-RES-001", "poisson", "fd4", "H-1"): 1e-5,
    ("PH-RES-001", "poisson", "spectral", "H-1"): 1e-13,
    ("PH-BC-001",  "laplace", "fd4", "L2-rel"): 1e-11,
    ("PH-BC-001",  "poisson", "fd4", "L2-rel"): 1e-11,
}

_TOLERANCE_DEFAULTS: dict[str, float] = {
    "spectral": 3.0,
    "fd4": 2.0,
}


def _tristate(ratio: float, pass_: float, fail_: float) -> Status:
    """Tri-state classification against the calibrated floor.

    ratio <= pass_: PASS
    pass_ < ratio <= fail_: WARN
    ratio > fail_: FAIL
    """
    if ratio <= pass_:
        return "PASS"
    if ratio <= fail_:
        return "WARN"
    return "FAIL"


def _load_floor(
    *,
    rule: str,
    pde: str,
    grid_shape: tuple[int, ...],
    method: str,
    norm: str,
) -> Floor:
    """Load a floor entry from floors.toml, falling back to shipped defaults.

    Task 14 populates physics_lint/data/floors.toml with calibrated values;
    the shipped-default fallback is exercised in CI and in the Week 1 test
    suite so rules never raise KeyError on first install.
    """
    floors_path = Path(__file__).parent.parent / "data" / "floors.toml"
    if floors_path.is_file():
        with open(floors_path, "rb") as f:
            data = tomllib.load(f)
        for entry in data.get("floor", []):
            if (
                entry.get("rule") == rule
                and entry.get("pde") == pde
                and tuple(entry.get("grid_shape", ())) == tuple(grid_shape)
                and entry.get("method") == method
                and entry.get("norm") == norm
            ):
                return Floor(
                    value=float(entry["value"]),
                    tolerance=float(entry.get("tolerance", _TOLERANCE_DEFAULTS.get(method, 2.0))),
                    source="calibrated",
                )

    default = _SHIPPED_DEFAULTS.get((rule, pde, method, norm))
    if default is None:
        default = 1e-5
    return Floor(
        value=default,
        tolerance=_TOLERANCE_DEFAULTS.get(method, 2.0),
        source="shipped",
    )
```

- [ ] **Step 4: Run helper tests and verify they pass**

```bash
pytest tests/rules/test_helpers.py -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Write failing test for PH-RES-001**

`tests/rules/test_ph_res_001.py`:

```python
"""PH-RES-001 — Residual exceeds variationally-correct norm threshold.

Laplace/Poisson path: compute the strong-form residual of the Field against
the configured PDE, take its H^-1 norm via the spectral formula (for periodic
inputs) or a Riesz-lift surrogate (for non-periodic, Week-1 falls back to L2),
divide by the calibrated floor, emit tri-state.
"""

from pathlib import Path

import numpy as np
import pytest

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_res_001


def _laplace_periodic_spec() -> DomainSpec:
    return DomainSpec.model_validate({
        "pde": "laplace",
        "grid_shape": [64, 64],
        "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi]},
        "periodic": True,
        "boundary_condition": {"kind": "periodic"},
        "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
    })


def test_ph_res_001_exact_harmonic_is_pass():
    spec = _laplace_periodic_spec()
    # Harmonic: Laplacian is identically zero, so residual norm is ~0
    N = 64
    x = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.zeros_like(X)                      # the trivial harmonic
    field = GridField(u, h=(2 * np.pi / N, 2 * np.pi / N), periodic=True)

    result = ph_res_001.check(field, spec)
    assert result.rule_id == "PH-RES-001"
    assert result.status == "PASS"
    assert result.raw_value is not None
    assert result.raw_value < 1e-12


def test_ph_res_001_nonzero_residual_is_warn_or_fail():
    spec = _laplace_periodic_spec()
    N = 64
    x = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    # u = cos(x) cos(y) — Laplacian is -2 cos(x) cos(y), so this is NOT
    # a Laplace solution (residual is nonzero and large).
    u = np.cos(X) * np.cos(Y)
    field = GridField(u, h=(2 * np.pi / N, 2 * np.pi / N), periodic=True)

    result = ph_res_001.check(field, spec)
    assert result.status in {"WARN", "FAIL"}
    assert result.violation_ratio is not None
    assert result.violation_ratio > 1.0


def test_ph_res_001_metadata():
    assert ph_res_001.__rule_id__ == "PH-RES-001"
    assert ph_res_001.__default_severity__ == "error"
    assert "adapter" in ph_res_001.__input_modes__
    assert "dump" in ph_res_001.__input_modes__
```

- [ ] **Step 6: Run the test to verify it fails**

```bash
pytest tests/rules/test_ph_res_001.py -v
```

Expected: `ModuleNotFoundError: No module named 'physics_lint.rules.ph_res_001'`

- [ ] **Step 7: Implement `src/physics_lint/rules/ph_res_001.py`**

```python
"""PH-RES-001: Residual exceeds variationally-correct norm threshold.

Design doc §7.5. For Laplace/Poisson the recommended norm is H^-1;
periodic grids use the spectral formula in physics_lint.norms.
Non-periodic Laplace/Poisson on Week 1 falls back to L^2 with a
PH-VAR-001-style caveat (implemented later; Week 1 only emits the
result, not the caveat).
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field, GridField
from physics_lint.norms import h_minus_one_spectral, l2_grid
from physics_lint.report import RuleResult
from physics_lint.rules._helpers import Floor, _load_floor, _tristate
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-RES-001"
__rule_name__ = "Residual exceeds variationally-correct norm threshold"
__default_severity__ = "error"
__input_modes__ = frozenset({"adapter", "dump"})

__default_thresholds__ = {"tol_pass": 10.0, "tol_fail": 100.0}

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-RES-001"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    """Compute strong-form residual, measure in H^-1 (or L^2 fallback), report."""
    if not isinstance(field, GridField):
        # Callable fields are handled by materializing via the loader before
        # rule dispatch; if we still see one here, raise to surface the bug.
        raise TypeError(
            f"PH-RES-001 requires a GridField; got {type(field).__name__}. "
            "Adapter-mode callables must be materialized by the loader."
        )

    # Compute the residual: R = -Delta u for Laplace; R = -Delta u - f for Poisson
    lap = field.laplacian().values()
    if spec.pde == "laplace":
        residual = -lap
    elif spec.pde == "poisson":
        # The spec should carry the source term as a function or array.
        # Week 1: the loader doesn't yet plumb the source; for now require it
        # on spec.metadata, and error otherwise.
        source = getattr(spec, "_source_array", None)
        if source is None:
            raise NotImplementedError(
                "PH-RES-001 for Poisson requires a source term on spec; Week 1 scope "
                "covers Laplace (automatic source=0); Poisson source wiring lands in Week 2."
            )
        residual = -lap - source
    elif spec.pde in {"heat", "wave"}:
        raise NotImplementedError(
            f"PH-RES-001 for {spec.pde} lands in Week 2 with the Bochner norm."
        )
    else:
        raise ValueError(f"unknown PDE {spec.pde}")

    # Norm selection
    method = field.backend                              # "fd" or "spectral"
    if spec.periodic and field.backend == "spectral":
        raw_value = h_minus_one_spectral(residual, field.h)
        norm_name = "H-1"
    else:
        # Non-periodic Week 1: L^2 fallback with PH-VAR-001 documented caveat
        raw_value = l2_grid(residual, field.h)
        norm_name = "L2"

    floor: Floor = _load_floor(
        rule="PH-RES-001",
        pde=spec.pde,
        grid_shape=spec.grid_shape,
        method=("fd4" if method == "fd" else method),
        norm=norm_name,
    )
    ratio = raw_value / floor.value if floor.value > 0 else float("inf")
    status = _tristate(ratio, pass_=floor.tolerance * 10, fail_=floor.tolerance * 100)

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=raw_value,
        violation_ratio=ratio,
        mode=None,
        reason=None,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm=norm_name,
        citation="Bachmayr et al. 2024; Ernst et al. 2025 v3",
        doc_url=_DOC_URL,
    )
```

- [ ] **Step 8: Run PH-RES-001 tests and verify they pass**

```bash
pytest tests/rules/test_ph_res_001.py -v
```

Expected: 3 PASS. If the first test hits a spectral-mode PASS at `raw_value ~ 1e-15`, that's correct (exact harmonic input).

- [ ] **Step 9: Write failing test + implementation for PH-RES-002**

`tests/rules/test_ph_res_002.py`:

```python
"""PH-RES-002 — FD-vs-AD cross-check. Adapter-only; dump mode SKIPs."""

import numpy as np
import torch

from physics_lint import CallableField, DomainSpec, GridField
from physics_lint.rules import ph_res_002


def _laplace_spec(periodic: bool = False) -> DomainSpec:
    return DomainSpec.model_validate({
        "pde": "laplace",
        "grid_shape": [16, 16],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": periodic,
        "boundary_condition": {"kind": "dirichlet" if not periodic else "periodic"},
        "field": {"type": "callable" if not periodic else "grid",
                  "backend": "fd",
                  "adapter_path" if not periodic else "dump_path": "x"},
    })


def test_ph_res_002_dump_mode_skipped():
    spec = _laplace_spec(periodic=True)
    u = np.zeros((16, 16))
    field = GridField(u, h=1.0 / 16, periodic=True)
    result = ph_res_002.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason and "callable" in result.reason.lower()


def test_ph_res_002_adapter_mode_quadratic_zero_discrepancy():
    spec = _laplace_spec(periodic=False)
    grid = torch.stack(torch.meshgrid(
        torch.linspace(0.0, 1.0, 16),
        torch.linspace(0.0, 1.0, 16),
        indexing="ij",
    ), dim=-1)
    def model(x):
        return (x[..., 0] ** 2 + x[..., 1] ** 2).unsqueeze(-1)
    field = CallableField(model, sampling_grid=grid, h=(1.0 / 15, 1.0 / 15))
    result = ph_res_002.check(field, spec)
    assert result.status in {"PASS", "WARN"}   # small FD error at edges
    assert result.raw_value is not None
    assert result.raw_value < 0.1              # 10% discrepancy tolerance
```

`src/physics_lint/rules/ph_res_002.py`:

```python
"""PH-RES-002: FD-vs-AD residual cross-check.

Adapter-only — requires an AD-capable model. Dump mode emits
SKIPPED with an explicit reason string.

Discrepancy formula (design doc §7.2):
    |R_FD - R_AD| / max(|R_FD|, |R_AD|, epsilon_floor)

Default threshold 0.01. Above: status = "WARN" (severity "warning").
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import CallableField, Field, GridField
from physics_lint.report import RuleResult
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-RES-002"
__rule_name__ = "FD-vs-AD residual cross-check discrepancy"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter"})   # dump emits SKIPPED

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-RES-002"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    if not isinstance(field, CallableField):
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status="SKIPPED",
            raw_value=None,
            violation_ratio=None,
            mode=None,
            reason="FD-vs-AD cross-check requires a callable model; dump mode provides only a frozen tensor",
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="",
            citation="PhysicsNeMo Sym multi-backend pattern",
            doc_url=_DOC_URL,
        )

    # AD path: Laplacian via torch autograd (already done in CallableField.laplacian)
    lap_ad = field.laplacian().values()

    # FD path: materialize the field values on the grid, wrap in a plain GridField
    # with backend="fd", and compute its Laplacian via the FD stencil.
    vals = field.values()
    h = field.h
    fd_field = GridField(vals, h=h, periodic=spec.periodic, backend="fd")
    lap_fd = fd_field.laplacian().values()

    epsilon = 1e-12
    denom = np.maximum(np.maximum(np.abs(lap_fd), np.abs(lap_ad)), epsilon)
    discrepancy_map = np.abs(lap_fd - lap_ad) / denom
    # Interior-only comparison (exclude the outer 2 layers where FD is 2nd-order)
    if vals.ndim == 2:
        interior = discrepancy_map[2:-2, 2:-2]
    elif vals.ndim == 3:
        interior = discrepancy_map[2:-2, 2:-2, 2:-2]
    else:
        interior = discrepancy_map
    max_discrepancy = float(np.max(interior)) if interior.size > 0 else 0.0

    status = "PASS" if max_discrepancy < 0.01 else "WARN"
    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=max_discrepancy,
        violation_ratio=max_discrepancy / 0.01 if max_discrepancy > 0 else 0.0,
        mode=None,
        reason=None,
        refinement_rate=None,
        spatial_map=discrepancy_map,
        recommended_norm="max discrepancy ratio",
        citation="PhysicsNeMo Sym multi-backend pattern",
        doc_url=_DOC_URL,
    )
```

- [ ] **Step 10: Run PH-RES-002 tests**

```bash
pytest tests/rules/test_ph_res_002.py -v
```

Expected: both tests PASS.

- [ ] **Step 11: Implement PH-RES-003 (spectral-vs-FD cross-check on periodic grids)**

`tests/rules/test_ph_res_003.py`:

```python
"""PH-RES-003 — Spectral-vs-FD discrepancy on periodic grids."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_res_003


def _periodic_spec() -> DomainSpec:
    return DomainSpec.model_validate({
        "pde": "laplace",
        "grid_shape": [64, 64],
        "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi]},
        "periodic": True,
        "boundary_condition": {"kind": "periodic"},
        "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
    })


def test_ph_res_003_smooth_periodic_passes():
    N = 64
    h = 2 * np.pi / N
    x = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = np.sin(X) * np.sin(Y)
    field = GridField(u, h=h, periodic=True)
    result = ph_res_003.check(field, _periodic_spec())
    assert result.status == "PASS"


def test_ph_res_003_skipped_on_nonperiodic():
    spec = DomainSpec.model_validate({
        "pde": "laplace",
        "grid_shape": [32, 32],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet"},
        "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
    })
    u = np.zeros((32, 32))
    field = GridField(u, h=1.0 / 31, periodic=False)
    result = ph_res_003.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason and "periodic" in result.reason.lower()
```

`src/physics_lint/rules/ph_res_003.py`:

```python
"""PH-RES-003: Spectral-vs-FD residual cross-check on periodic grids."""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field, GridField
from physics_lint.report import RuleResult
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-RES-003"
__rule_name__ = "Spectral-vs-FD residual discrepancy on periodic grid"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-RES-003"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    if not spec.periodic:
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status="SKIPPED",
            raw_value=None,
            violation_ratio=None,
            mode=None,
            reason="PH-RES-003 applies only to periodic domains",
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="",
            citation="Trefethen 2000; Fornberg 1988",
            doc_url=_DOC_URL,
        )
    if not isinstance(field, GridField):
        raise TypeError(f"PH-RES-003 requires GridField; got {type(field).__name__}")

    vals = field.values()
    spectral_f = GridField(vals, h=field.h, periodic=True, backend="spectral")
    fd_f = GridField(vals, h=field.h, periodic=True, backend="fd")
    lap_spectral = spectral_f.laplacian().values()
    lap_fd = fd_f.laplacian().values()
    diff = lap_spectral - lap_fd
    denom = float(np.max(np.abs(lap_spectral))) or 1.0
    max_rel = float(np.max(np.abs(diff))) / denom

    status = "PASS" if max_rel < 0.01 else "WARN"
    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=max_rel,
        violation_ratio=max_rel / 0.01 if max_rel > 0 else 0.0,
        mode=None,
        reason=None,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="max relative difference",
        citation="Trefethen 2000; Fornberg 1988",
        doc_url=_DOC_URL,
    )
```

- [ ] **Step 12: Delete the placeholder rule and update the registry test**

```bash
rm src/physics_lint/rules/ph_placeholder_000.py
```

Update `tests/test_registry.py` (replace the placeholder assertion):

```python
"""Rule registry tests — lazy metadata discovery."""

from physics_lint.rules import _registry


def test_registry_discovers_week_1_rules():
    rules = _registry.list_rules()
    ids = {r.rule_id for r in rules}
    expected = {"PH-RES-001", "PH-RES-002", "PH-RES-003"}
    assert expected.issubset(ids), f"missing rules: {expected - ids}"


def test_registry_lazy_check_not_imported():
    rules = _registry.list_rules()
    for r in rules:
        assert r.check_fn is None, f"{r.rule_id} check was eagerly imported"


def test_registry_materialize_check():
    rules = _registry.list_rules()
    first = next(iter(rules))
    check = _registry.load_check(first)
    assert callable(check)
```

- [ ] **Step 13: Run the full suite**

```bash
pytest -q
```

Expected: all tests pass. Registry discovers 3 real rules; all three rule tests pass.

- [ ] **Step 14: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/physics_lint/rules/ tests/rules/ tests/test_registry.py
git commit -m "feat(rules): PH-RES-001/002/003 with shared _helpers and floors integration

Per design doc §7.2, §7.4, §7.5, §10.2 (rule catalog rows) and §10.4 (lazy
registry pattern). Removes ph_placeholder_000 introduced in Task 9.

_helpers.py:
- _tristate(ratio, pass_, fail_) -> Status
- Floor dataclass + _load_floor() reads physics_lint/data/floors.toml with
  a conservative shipped-default fallback for pre-calibration (Task 14)

PH-RES-001: H^-1 spectral norm on periodic Laplace/Poisson; L^2 fallback on
non-periodic (PH-VAR-001 caveat lands when the meta-warning rules are added).
Tri-state against floor * tolerance.

PH-RES-002: FD-vs-AD cross-check (adapter-only). Dump mode emits
RuleResult(status='SKIPPED', reason='requires callable model').
Max discrepancy < 0.01 = PASS; else WARN.

PH-RES-003: spectral-vs-FD cross-check on periodic grids. Non-periodic
emits SKIPPED. PASS below 1% max relative difference."
```

---

## Task 11: BC rules — PH-BC-001 (relative/absolute branching) and PH-BC-002

**Files:**
- Create: `src/physics_lint/rules/ph_bc_001.py`
- Create: `src/physics_lint/rules/ph_bc_002.py`
- Create: `tests/rules/test_ph_bc_001.py`
- Create: `tests/rules/test_ph_bc_002.py`

**Rationale:** `PH-BC-001` is the most subtle Week-1 rule — mode-branched on `||g||`, with absolute mode being deliberately binary PASS/FAIL while relative mode is tri-state. This branching is the Rev 4.1 fix for the homogeneous-Dirichlet footgun; the test coverage must prove both branches fire correctly on the right inputs.

- [ ] **Step 1: Write failing test for PH-BC-001 both branches**

`tests/rules/test_ph_bc_001.py`:

```python
"""PH-BC-001 — relative vs absolute mode branching."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_bc_001


def _laplace_hd_spec() -> DomainSpec:
    return DomainSpec.model_validate({
        "pde": "laplace",
        "grid_shape": [32, 32],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
    })


def _laplace_dirichlet_spec() -> DomainSpec:
    return DomainSpec.model_validate({
        "pde": "laplace",
        "grid_shape": [32, 32],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet"},
        "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
    })


def _zeros_on_hd_boundary() -> GridField:
    N = 32
    u = np.zeros((N, N))
    # Create a bump in the interior: u = sin(pi x) sin(pi y)
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)
    # Boundary is zero by construction of sin(pi*0)=sin(pi*1)=0
    return GridField(u, h=(1.0 / 31, 1.0 / 31), periodic=False)


def test_ph_bc_001_homogeneous_dirichlet_is_absolute_pass():
    spec = _laplace_hd_spec()
    field = _zeros_on_hd_boundary()
    # For hD, boundary targets are zero.
    boundary_target = np.zeros_like(field.values_on_boundary())
    result = ph_bc_001.check(field, spec, boundary_target=boundary_target)
    assert result.rule_id == "PH-BC-001"
    assert result.mode == "absolute"
    assert result.status == "PASS"   # boundary values are ~0 exactly


def test_ph_bc_001_homogeneous_dirichlet_violation_is_fail():
    spec = _laplace_hd_spec()
    N = 32
    u = np.ones((N, N))   # violates u=0 on boundary
    field = GridField(u, h=(1.0 / 31, 1.0 / 31), periodic=False)
    boundary_target = np.zeros_like(field.values_on_boundary())
    result = ph_bc_001.check(field, spec, boundary_target=boundary_target)
    assert result.mode == "absolute"
    assert result.status == "FAIL"
    assert result.raw_value is not None and result.raw_value > 0


def test_ph_bc_001_inhomogeneous_is_relative_pass():
    spec = _laplace_dirichlet_spec()
    N = 32
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = X ** 2 - Y ** 2   # harmonic polynomial, trace is nonzero on boundary
    field = GridField(u, h=(1.0 / 31, 1.0 / 31), periodic=False)
    # Match the boundary exactly: compute the analytical trace from u
    boundary_target = field.values_on_boundary().copy()
    result = ph_bc_001.check(field, spec, boundary_target=boundary_target)
    assert result.mode == "relative"
    assert result.status == "PASS"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/rules/test_ph_bc_001.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/physics_lint/rules/ph_bc_001.py`**

```python
"""PH-BC-001: Boundary condition violation with mode-branched normalization.

Design doc §8.5.

If ||g|| < abs_threshold: absolute mode (binary PASS/FAIL against abs_tol_fail)
Otherwise: relative mode (tri-state against the calibrated relative floor)
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field
from physics_lint.norms import l2_grid
from physics_lint.report import RuleResult
from physics_lint.rules._helpers import _load_floor, _tristate
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-BC-001"
__rule_name__ = "Boundary condition violation (relative or absolute mode)"
__default_severity__ = "error"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-BC-001"

_DEFAULT_ABS_THRESHOLD = 1e-10
_DEFAULT_ABS_TOL_FAIL = 1e-3


def check(
    field: Field,
    spec: DomainSpec,
    *,
    boundary_target: np.ndarray,
    abs_threshold: float = _DEFAULT_ABS_THRESHOLD,
    abs_tol_fail: float = _DEFAULT_ABS_TOL_FAIL,
) -> RuleResult:
    """Compute ||u - g|| on the boundary; mode-branch on ||g||."""
    u_boundary = field.values_on_boundary()
    if u_boundary.shape != boundary_target.shape:
        raise ValueError(
            f"boundary_target shape {boundary_target.shape} does not match "
            f"field.values_on_boundary() shape {u_boundary.shape}"
        )
    err_values = u_boundary - boundary_target
    # 1D L2 norm on the boundary trace (points are ordered; use discrete L2)
    err_norm = float(np.linalg.norm(err_values) / np.sqrt(max(len(err_values), 1)))
    g_norm = float(np.linalg.norm(boundary_target) / np.sqrt(max(len(boundary_target), 1)))

    if g_norm < abs_threshold:
        status = "PASS" if err_norm < abs_tol_fail else "FAIL"
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status=status,
            raw_value=err_norm,
            violation_ratio=None,
            mode="absolute",
            reason=None,
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="boundary L2 (absolute)",
            citation="design doc §8.5",
            doc_url=_DOC_URL,
        )

    relative_value = err_norm / g_norm
    floor = _load_floor(
        rule="PH-BC-001",
        pde=spec.pde,
        grid_shape=spec.grid_shape,
        method="fd4",
        norm="L2-rel",
    )
    ratio = relative_value / floor.value if floor.value > 0 else float("inf")
    status = _tristate(ratio, pass_=floor.tolerance * 10, fail_=floor.tolerance * 100)

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=relative_value,
        violation_ratio=ratio,
        mode="relative",
        reason=None,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="boundary L2 (relative)",
        citation="design doc §8.5",
        doc_url=_DOC_URL,
    )
```

- [ ] **Step 4: Run PH-BC-001 tests and verify they pass**

```bash
pytest tests/rules/test_ph_bc_001.py -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Implement PH-BC-002 (boundary flux imbalance) — test first**

`tests/rules/test_ph_bc_002.py`:

```python
"""PH-BC-002 — boundary flux imbalance via divergence theorem."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_bc_002


def _laplace_spec() -> DomainSpec:
    return DomainSpec.model_validate({
        "pde": "laplace",
        "grid_shape": [64, 64],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
    })


def test_ph_bc_002_harmonic_has_zero_net_flux():
    # u = x^2 - y^2, harmonic. Net flux around the boundary = 0 (up to FD error).
    spec = _laplace_spec()
    N = 64
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = X ** 2 - Y ** 2
    field = GridField(u, h=(1.0 / (N - 1), 1.0 / (N - 1)), periodic=False)
    result = ph_bc_002.check(field, spec)
    assert result.rule_id == "PH-BC-002"
    assert result.status == "PASS"
    assert result.raw_value is not None
    assert abs(result.raw_value) < 0.01    # small FD edge contribution


def test_ph_bc_002_non_harmonic_has_nonzero_net_flux():
    spec = _laplace_spec()
    N = 64
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.exp(X) * np.sin(Y)   # not harmonic — Laplacian is nonzero
    field = GridField(u, h=(1.0 / (N - 1), 1.0 / (N - 1)), periodic=False)
    result = ph_bc_002.check(field, spec)
    # Under Laplace we expect net flux ~ integral of Laplacian (nonzero here)
    assert result.status in {"WARN", "FAIL"}
    assert result.raw_value is not None and abs(result.raw_value) > 0.01
```

- [ ] **Step 6: Implement `src/physics_lint/rules/ph_bc_002.py`**

```python
"""PH-BC-002: Boundary flux imbalance (divergence theorem).

For Laplace/Poisson: integral of -Delta u over the domain equals the net
outward boundary flux integral. Violation of this identity is a sign that
the learned field is inconsistent with the PDE at a weak-form level even if
the pointwise residual is small.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field, GridField
from physics_lint.norms import trapezoidal_integral
from physics_lint.report import RuleResult
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-BC-002"
__rule_name__ = "Boundary flux imbalance (divergence theorem)"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-BC-002"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    if spec.pde not in {"laplace", "poisson"}:
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status="SKIPPED",
            raw_value=None,
            violation_ratio=None,
            mode=None,
            reason=f"PH-BC-002 applies to laplace/poisson only; got {spec.pde}",
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="",
            citation="divergence theorem",
            doc_url=_DOC_URL,
        )
    if not isinstance(field, GridField):
        raise TypeError(f"PH-BC-002 requires GridField; got {type(field).__name__}")

    lap = field.laplacian().values()
    u_vol_integral_of_lap = trapezoidal_integral(lap, field.h)
    # The expected net boundary flux under -Delta u = f is -integral(f)
    # For Laplace (f=0) it's zero; for Poisson we'd subtract integral(f).
    # Week 1: Laplace only (Poisson source is plumbed in Week 2).
    expected = 0.0 if spec.pde == "laplace" else -u_vol_integral_of_lap
    imbalance = float(u_vol_integral_of_lap - expected)
    # Threshold is scale-dependent; use relative to the field's L2 norm
    from physics_lint.norms import l2_grid
    scale = max(l2_grid(field.values(), field.h), 1e-12)
    ratio = abs(imbalance) / scale
    status = "PASS" if ratio < 0.01 else ("WARN" if ratio < 0.1 else "FAIL")
    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=imbalance,
        violation_ratio=ratio,
        mode=None,
        reason=None,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="integral of Laplacian (divergence theorem)",
        citation="classical divergence theorem",
        doc_url=_DOC_URL,
    )
```

- [ ] **Step 7: Run PH-BC-002 tests**

```bash
pytest tests/rules/test_ph_bc_002.py -v
```

Expected: both PASS.

- [ ] **Step 8: Run the full suite and commit**

```bash
pytest -q
ruff check src tests && ruff format src tests
git add src/physics_lint/rules/ph_bc_001.py src/physics_lint/rules/ph_bc_002.py tests/rules/test_ph_bc_001.py tests/rules/test_ph_bc_002.py
git commit -m "feat(rules): PH-BC-001 with mode branching + PH-BC-002 divergence theorem

Per design doc §8.5 (mode-branched BC rule) and §8.1 / §10.2 (PH-BC-002).

PH-BC-001: Compute ||u - g|| on the boundary trace; if ||g|| < abs_threshold
(default 1e-10) branch to absolute mode (binary PASS/FAIL vs abs_tol_fail);
otherwise branch to relative mode (tri-state vs calibrated floor). The
RuleResult.mode field is set so the text reporter can surface [absolute mode]
vs [relative mode] per §11.1.

Absolute mode is deliberately binary in V1 per §8.5 — no abs_tol_warn tier
because the problem-specific scale for intermediate deviation cannot be
calibrated without domain knowledge.

PH-BC-002: Integral of Laplacian over the volume vs expected boundary flux
(zero for Laplace, -integral(f) for Poisson). Relative threshold vs L^2 of
the field; PASS < 1%, WARN < 10%, FAIL above."
```

---

## Task 12: Positivity rules — PH-POS-001 and PH-POS-002

**Files:**
- Create: `src/physics_lint/rules/ph_pos_001.py`
- Create: `src/physics_lint/rules/ph_pos_002.py`
- Create: `tests/rules/test_ph_pos_001.py`
- Create: `tests/rules/test_ph_pos_002.py`

**Rationale:** Positivity rules are straightforward but demonstrate the `BCSpec.preserves_sign` dividend — a single boolean read replaces what would otherwise be per-PDE BC taxonomy duplication. Both rules are simple implementations; both tests exercise the SKIPPED path as well as the PASS/FAIL paths.

- [ ] **Step 1: Write failing tests for PH-POS-001**

`tests/rules/test_ph_pos_001.py`:

```python
"""PH-POS-001 — Positivity violation."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_pos_001


def _heat_per_spec() -> DomainSpec:
    return DomainSpec.model_validate({
        "pde": "heat",
        "grid_shape": [32, 32, 4],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.1]},
        "periodic": True,
        "boundary_condition": {"kind": "periodic"},
        "diffusivity": 0.01,
        "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
    })


def _poisson_hd_spec() -> DomainSpec:
    return DomainSpec.model_validate({
        "pde": "poisson",
        "grid_shape": [32, 32],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
    })


def test_ph_pos_001_nonneg_passes():
    spec = _poisson_hd_spec()
    u = np.ones((32, 32)) * 0.5
    field = GridField(u, h=1.0 / 31, periodic=False)
    result = ph_pos_001.check(field, spec)
    assert result.status == "PASS"


def test_ph_pos_001_negative_values_fail():
    spec = _poisson_hd_spec()
    u = np.ones((32, 32)) * 0.5
    u[10:20, 10:20] = -0.1
    field = GridField(u, h=1.0 / 31, periodic=False)
    result = ph_pos_001.check(field, spec)
    assert result.status == "FAIL"
    assert result.raw_value == -0.1


def test_ph_pos_001_skipped_on_non_sign_preserving_bc():
    spec = DomainSpec.model_validate({
        "pde": "laplace",
        "grid_shape": [32, 32],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet"},
        "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
    })
    u = np.ones((32, 32))
    field = GridField(u, h=1.0 / 31, periodic=False)
    result = ph_pos_001.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason and "sign" in result.reason.lower()
```

- [ ] **Step 2: Implement `src/physics_lint/rules/ph_pos_001.py`**

```python
"""PH-POS-001: Positivity violation.

Applies when the BC preserves sign (read via spec.boundary_condition.preserves_sign).
Otherwise emits SKIPPED.
"""

from __future__ import annotations

from physics_lint.field import Field, GridField
from physics_lint.report import RuleResult
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-POS-001"
__rule_name__ = "Positivity violation"
__default_severity__ = "error"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-POS-001"


def check(field: Field, spec: DomainSpec, *, floor: float = 0.0) -> RuleResult:
    if not spec.boundary_condition.preserves_sign:
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status="SKIPPED",
            raw_value=None,
            violation_ratio=None,
            mode=None,
            reason=(
                f"Configured BC '{spec.boundary_condition.kind}' does not preserve sign; "
                "PH-POS-001 does not apply"
            ),
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="",
            citation="heat eigenfunction decay; Poisson positivity under hD with f >= 0",
            doc_url=_DOC_URL,
        )

    u = field.values()
    min_val = float(u.min())
    violation_map = u < floor
    n_violations = int(violation_map.sum())
    violation_fraction = float(violation_map.mean())

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status="PASS" if n_violations == 0 else "FAIL",
        raw_value=min_val,
        violation_ratio=violation_fraction if n_violations > 0 else 0.0,
        mode=None,
        reason=(
            None if n_violations == 0
            else f"{n_violations} cells below {floor} (fraction {violation_fraction:.3f})"
        ),
        refinement_rate=None,
        spatial_map=violation_map,
        recommended_norm="min pointwise value",
        citation="design doc §8.6",
        doc_url=_DOC_URL,
    )
```

- [ ] **Step 3: Run PH-POS-001 tests**

```bash
pytest tests/rules/test_ph_pos_001.py -v
```

Expected: all 3 PASS.

- [ ] **Step 4: Write failing tests for PH-POS-002**

`tests/rules/test_ph_pos_002.py`:

```python
"""PH-POS-002 — Maximum principle violation for Laplace."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_pos_002


def _laplace_dirichlet_spec() -> DomainSpec:
    return DomainSpec.model_validate({
        "pde": "laplace",
        "grid_shape": [32, 32],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet"},
        "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
    })


def test_ph_pos_002_harmonic_passes():
    # u = x^2 - y^2, harmonic. Boundary values span [-1, 1], interior is within.
    spec = _laplace_dirichlet_spec()
    N = 32
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = X ** 2 - Y ** 2
    field = GridField(u, h=(1.0 / 31, 1.0 / 31), periodic=False)
    result = ph_pos_002.check(field, spec, boundary_values=field.values_on_boundary())
    assert result.status == "PASS"


def test_ph_pos_002_interior_overshoot_fails():
    spec = _laplace_dirichlet_spec()
    N = 32
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.zeros_like(X)
    u[15, 15] = 5.0   # wild interior value above all boundary values
    field = GridField(u, h=(1.0 / 31, 1.0 / 31), periodic=False)
    boundary_vals = field.values_on_boundary()   # all zeros
    result = ph_pos_002.check(field, spec, boundary_values=boundary_vals)
    assert result.status == "FAIL"
    assert result.raw_value is not None and result.raw_value > 0
```

- [ ] **Step 5: Implement `src/physics_lint/rules/ph_pos_002.py`**

```python
"""PH-POS-002: Maximum principle violation for Laplace.

Under a well-posed Dirichlet problem for -Delta u = 0, min/max of u are
attained on the boundary. Violation indicates a spurious interior extremum.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field
from physics_lint.report import RuleResult
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-POS-002"
__rule_name__ = "Maximum principle violation"
__default_severity__ = "error"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-POS-002"


def check(
    field: Field,
    spec: DomainSpec,
    *,
    boundary_values: np.ndarray,
) -> RuleResult:
    if spec.pde != "laplace":
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status="SKIPPED",
            raw_value=None,
            violation_ratio=None,
            mode=None,
            reason=f"max principle applies to Laplace only; got {spec.pde}",
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="",
            citation="maximum principle for harmonic functions",
            doc_url=_DOC_URL,
        )

    u = field.values()
    bmin = float(boundary_values.min())
    bmax = float(boundary_values.max())
    below = max(0.0, bmin - float(u.min()))
    above = max(0.0, float(u.max()) - bmax)
    overshoot = max(below, above)

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status="PASS" if overshoot <= 1e-10 else "FAIL",
        raw_value=overshoot,
        violation_ratio=None,
        mode=None,
        reason=(None if overshoot <= 1e-10 else
                f"interior extremum beyond boundary extrema by {overshoot:.3e}"),
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="interior extremum overshoot",
        citation="maximum principle for harmonic functions",
        doc_url=_DOC_URL,
    )
```

- [ ] **Step 6: Run PH-POS-002 tests and the full suite**

```bash
pytest tests/rules/test_ph_pos_002.py -v
pytest -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
ruff check src tests && ruff format src tests
git add src/physics_lint/rules/ph_pos_001.py src/physics_lint/rules/ph_pos_002.py tests/rules/test_ph_pos_001.py tests/rules/test_ph_pos_002.py
git commit -m "feat(rules): PH-POS-001 and PH-POS-002 positivity + maximum principle

Per design doc §8.6 and §10.2.

PH-POS-001: Reads spec.boundary_condition.preserves_sign (the BCSpec
computed-property dividend from Task 7). If False -> SKIPPED with explicit
reason; otherwise checks min pointwise value vs floor=0.0 and reports the
violation count + fraction + spatial_map.

PH-POS-002: Interior extremum overshoot vs boundary extremum, applies to
Laplace only (any well-posed Dirichlet). Non-Laplace PDEs emit SKIPPED."
```

---

## Task 13: Hypothesis property-based tests for Field

**Files:**
- Create: `tests/test_field_properties.py` (five property tests)
- Modify: `tests/conftest.py` (already has profiles; no change)

**Rationale:** Hypothesis is Week 1, not deferred — the Field ABC is too load-bearing to ship without property-based coverage. Five properties per design doc §15: polynomial exactness, sine-mode correctness, rotation commutativity, refinement convergence, and integration-by-parts on periodic domains.

- [ ] **Step 1: Write the five property tests**

`tests/test_field_properties.py`:

```python
"""Hypothesis property-based tests for the Field abstraction.

Five properties (design doc §15):
1. Polynomial of degree <= stencil order -> 4th-order FD derivative exact
2. sin(kx) on periodic grid -> spectral derivative is k*cos(kx) to ~1e-12
3. Laplacian commutes with np.rot90 on square grids
4. Residual of analytical solution converges at expected rate under refinement
5. Integration by parts on periodic domains: integral(u*Lap(v)) == integral(v*Lap(u))
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from physics_lint.field import GridField
from physics_lint.norms import trapezoidal_integral


# Strategy helpers
_GRID_SIZES = st.integers(min_value=8, max_value=128).filter(lambda n: n % 2 == 0)
_SMALL_K = st.integers(min_value=1, max_value=4)


@given(N=_GRID_SIZES, degree=st.integers(min_value=0, max_value=3))
@settings(max_examples=25, deadline=None)
def test_polynomial_fd_exact_interior(N: int, degree: int):
    """4th-order central FD on a polynomial of degree <= 3 is exact in the interior."""
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    h = 1.0 / (N - 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    # u = X^degree + Y^degree so Laplacian = d(d-1)*(X^(d-2) + Y^(d-2))
    u = X ** degree + Y ** degree
    f = GridField(u, h=h, periodic=False, backend="fd")
    lap = f.laplacian().values()
    if degree >= 2:
        expected = degree * (degree - 1) * (X ** (degree - 2) + Y ** (degree - 2))
    else:
        expected = np.zeros_like(X)
    interior = lap[3:-3, 3:-3]
    expected_interior = expected[3:-3, 3:-3]
    assert np.max(np.abs(interior - expected_interior)) < 1e-9


@given(N=_GRID_SIZES, k=_SMALL_K)
@settings(max_examples=25, deadline=None)
def test_spectral_sine_derivative(N: int, k: int):
    """Spectral first derivative of sin(2*pi*k*x)*(1 in y) matches 2*pi*k*cos(...)."""
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.sin(2 * np.pi * k * X)
    f = GridField(u, h=h, periodic=True, backend="spectral")
    # Spectral Laplacian of sin(2*pi*k*x) is -(2*pi*k)^2 * sin(2*pi*k*x)
    lap = f.laplacian().values()
    expected = -((2 * np.pi * k) ** 2) * u
    assert np.max(np.abs(lap - expected)) < 1e-10


@given(N=_GRID_SIZES)
@settings(max_examples=20, deadline=None)
def test_rotation_commutes_with_laplacian(N: int):
    """rot90(Laplacian(u)) == Laplacian(rot90(u)) on a square periodic grid."""
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    f = GridField(u, h=h, periodic=True)
    u_rot = np.rot90(u)
    f_rot = GridField(u_rot, h=h, periodic=True)

    lhs = np.rot90(f.laplacian().values())
    rhs = f_rot.laplacian().values()
    assert np.max(np.abs(lhs - rhs)) < 1e-10


@given(k=_SMALL_K)
@settings(max_examples=10, deadline=None)
def test_refinement_rate_fd_periodic(k: int):
    """4th-order FD should converge like h^4 on a smooth periodic sine."""
    errs: list[float] = []
    for N in (32, 64, 128):
        h = 2 * np.pi / N
        x = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        y = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u = np.sin(k * X) * np.sin(k * Y)
        f = GridField(u, h=h, periodic=True, backend="fd")
        lap = f.laplacian().values()
        expected = -2 * k ** 2 * u
        errs.append(float(np.max(np.abs(lap - expected))))
    # Coarse-to-fine ratio >= 2^3.5 ~ 11.3 between N=32 and N=64
    if errs[0] > 0:
        ratio_coarse = errs[0] / max(errs[1], 1e-300)
        assert ratio_coarse > 8.0, f"refinement rate collapsed: {errs}"


@given(k=_SMALL_K)
@settings(max_examples=10, deadline=None)
def test_integration_by_parts_periodic(k: int):
    """integral(u * Lap(v)) == integral(v * Lap(u)) on a periodic domain."""
    N = 64
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.sin(2 * np.pi * k * X) * np.cos(2 * np.pi * k * Y)
    v = np.cos(2 * np.pi * k * X) * np.sin(2 * np.pi * k * Y)
    fu = GridField(u, h=h, periodic=True)
    fv = GridField(v, h=h, periodic=True)
    lap_u = fu.laplacian().values()
    lap_v = fv.laplacian().values()
    lhs = trapezoidal_integral(u * lap_v, (h, h))
    rhs = trapezoidal_integral(v * lap_u, (h, h))
    assert abs(lhs - rhs) < 1e-10
```

- [ ] **Step 2: Run the hypothesis tests**

```bash
pytest tests/test_field_properties.py --hypothesis-profile=dev -v
```

Expected: all 5 properties PASS. If `test_refinement_rate_fd_periodic` flakes near the tolerance (e.g., rate of 7.5 instead of 8), bump the grid sizes — but investigate first: a consistent below-8 rate would indicate a subtle bug in the FD stencil.

- [ ] **Step 3: Run under the `ci` profile to match CI coverage**

```bash
pytest tests/test_field_properties.py --hypothesis-profile=ci -v
```

Expected: all pass with more examples per property.

- [ ] **Step 4: Commit**

```bash
ruff check tests && ruff format tests
git add tests/test_field_properties.py
git commit -m "test(field): hypothesis property-based tests for Field abstraction

Per design doc §15. Five properties:

1. Polynomial fd exact interior: 4th-order FD on a polynomial of degree <= 3
   is exact to machine precision in the interior.
2. Spectral sine derivative: spectral Laplacian of sin(2 pi k x) matches
   -(2 pi k)^2 * sin(...) to 1e-10.
3. Rotation commutes with Laplacian: rot90(Lap(u)) == Lap(rot90(u)) on
   square periodic grids — cross-check between spectral and FD backends.
4. Refinement rate: 4th-order FD coarse-to-fine error ratio >= 8 between
   N=32 and N=64 on smooth periodic sines.
5. Integration by parts on periodic domains: integral(u Lap(v)) ==
   integral(v Lap(u)) — catches sign errors in the Laplacian that
   rotation-commutativity misses.

Registered ci-quick / ci / dev hypothesis profiles are in tests/conftest.py
from Task 2; CI runs --hypothesis-profile=ci and pre-commit runs ci-quick
via the field-property-tests local hook."
```

---

## Task 14: Floor calibration protocol + `floors.toml`

**Files:**
- Create: `scripts/calibrate_floors.py`
- Create: `src/physics_lint/data/floors.toml`
- Create: `.github/workflows/floor-calibration.yml` (throwaway, on a branch)
- Modify: `src/physics_lint/rules/_helpers.py` (tighten the shipped defaults now that calibration is real)

**Rationale:** The floor calibration half-day closes Week 1. The protocol (§6.4) is two-step on purpose: (a) measure on multiple environments, (b) commit with validation CI. This task implements both.

- [ ] **Step 1: Write the calibration script**

`scripts/calibrate_floors.py`:

```python
"""Floor calibration script for physics_lint/data/floors.toml.

Runs the analytical battery against every Week-1 (rule, pde, grid_shape,
method, norm) tuple, records the measured residual/error, and writes a
machine-readable summary to stdout. Run in three environments (macOS
arm64 local, ubuntu Docker, throwaway GHA worker) and take the MAXIMUM
observed value across environments as the floors.toml 'value'. Multiply
by the per-method tolerance multiplier (2x for fd4, 3x for spectral).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from physics_lint.analytical import laplace as laplace_sols
from physics_lint.analytical import poisson as poisson_sols
from physics_lint.field import GridField
from physics_lint.norms import h_minus_one_spectral, l2_grid


@dataclass
class FloorEntry:
    rule: str
    pde: str
    grid_shape: tuple[int, ...]
    method: str
    norm: str
    measured: float
    analytical_solution: str


def _measure_laplace_fd(N: int) -> float:
    sol = laplace_sols.harmonic_polynomial_square()
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = sol.u(X, Y)
    h = 1.0 / (N - 1)
    f = GridField(u, h=h, periodic=False, backend="fd")
    lap = f.laplacian().values()
    # Residual: -Delta u should equal zero; L^2 norm of -lap is our raw value
    residual = -lap
    return l2_grid(residual, (h, h))


def _measure_poisson_periodic_spectral(N: int) -> float:
    sol = poisson_sols.periodic_sin_sin()
    x = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = sol.u(X, Y)
    f_arr = sol.source(X, Y)
    h = 2 * np.pi / N
    field = GridField(u, h=h, periodic=True, backend="spectral")
    lap = field.laplacian().values()
    residual = -lap - f_arr
    return h_minus_one_spectral(residual, (h, h))


def _measure_bc_l2_rel_fd(N: int) -> float:
    sol = laplace_sols.harmonic_polynomial_square()
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = sol.u(X, Y)
    h = 1.0 / (N - 1)
    field = GridField(u, h=h, periodic=False, backend="fd")
    boundary = field.values_on_boundary()
    # Self-check: error against itself is zero (to floating-point noise)
    err = np.linalg.norm(boundary - boundary) / np.sqrt(max(len(boundary), 1))
    gnorm = np.linalg.norm(boundary) / np.sqrt(max(len(boundary), 1))
    return float(err / gnorm) if gnorm > 0 else float(err)


def main() -> None:
    entries: list[FloorEntry] = []

    entries.append(FloorEntry(
        rule="PH-RES-001", pde="laplace", grid_shape=(64, 64),
        method="fd4", norm="L2",
        measured=_measure_laplace_fd(64),
        analytical_solution="harmonic_polynomial_square",
    ))

    entries.append(FloorEntry(
        rule="PH-RES-001", pde="poisson", grid_shape=(64, 64),
        method="spectral", norm="H-1",
        measured=_measure_poisson_periodic_spectral(64),
        analytical_solution="periodic_sin_sin",
    ))

    entries.append(FloorEntry(
        rule="PH-BC-001", pde="laplace", grid_shape=(64, 64),
        method="fd4", norm="L2-rel",
        measured=_measure_bc_l2_rel_fd(64),
        analytical_solution="harmonic_polynomial_square",
    ))

    # Emit JSON to stdout; a driver script collects results across three
    # environments and writes the final floors.toml.
    print(json.dumps([asdict(e) for e in entries], indent=2, default=str))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run calibration locally and record the values**

```bash
python scripts/calibrate_floors.py > /tmp/floors-local.json
cat /tmp/floors-local.json
```

Expected: three entries with small measured values (e.g., `1e-6` for FD Laplace, `1e-14` for spectral Poisson, `0.0` for BC self-check).

- [ ] **Step 3: Run in a fresh Ubuntu Docker container**

```bash
docker run --rm -v "$PWD":/repo -w /repo python:3.11-slim bash -c "
  pip install -q -e '.[dev]' && python scripts/calibrate_floors.py
" > /tmp/floors-docker.json
cat /tmp/floors-docker.json
```

Expected: similar values, possibly differing by a few ULP in the spectral case due to pocketfft-vs-platform variations.

- [ ] **Step 4: Run on throwaway GHA workflow**

Create `.github/workflows/floor-calibration.yml` on a branch `chore/floor-calibration`:

```yaml
name: floor-calibration (throwaway)

on:
  workflow_dispatch:
  push:
    branches: [chore/floor-calibration]

jobs:
  calibrate:
    strategy:
      matrix:
        include:
          - { os: ubuntu-latest, python: "3.11" }
          - { os: macos-14, python: "3.11" }
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: pip install -e ".[dev]"
      - run: python scripts/calibrate_floors.py > floors-${{ matrix.os }}.json
      - uses: actions/upload-artifact@v4
        with:
          name: floors-${{ matrix.os }}
          path: floors-${{ matrix.os }}.json
```

Push the branch, trigger the workflow, download the artifacts. The user must do this manually via the GitHub UI or `gh run`:

```bash
git checkout -b chore/floor-calibration
git add .github/workflows/floor-calibration.yml scripts/calibrate_floors.py
git commit -m "chore: throwaway calibration workflow (branch only; not for main)"
git push -u origin chore/floor-calibration
gh workflow run floor-calibration --ref chore/floor-calibration
# wait, then:
gh run download --name floors-ubuntu-latest --dir /tmp/gha-floors
gh run download --name floors-macos-14 --dir /tmp/gha-floors
```

- [ ] **Step 5: Compute the maximum across environments and write `floors.toml`**

```bash
python - <<'PY'
import json
from pathlib import Path

sources = [
    Path("/tmp/floors-local.json"),
    Path("/tmp/floors-docker.json"),
    Path("/tmp/gha-floors/floors-ubuntu-latest.json"),
    Path("/tmp/gha-floors/floors-macos-14.json"),
]

all_entries = []
for src in sources:
    if src.exists():
        all_entries.append(json.loads(src.read_text()))

# Group by (rule, pde, grid_shape, method, norm) and take max
from collections import defaultdict
groups = defaultdict(list)
for run in all_entries:
    for e in run:
        key = (e["rule"], e["pde"], tuple(e["grid_shape"]), e["method"], e["norm"])
        groups[key].append(e["measured"])

toml_out = ["schema_version = 1", ""]
tolerance = {"spectral": 3.0, "fd4": 2.0}
for (rule, pde, shape, method, norm), values in sorted(groups.items()):
    max_val = max(values)
    # Apply safety floor of 1e-16 to avoid literal zero
    value = max(max_val, 1e-16)
    tol = tolerance.get(method, 2.0)
    toml_out.append("[[floor]]")
    toml_out.append(f'rule = "{rule}"')
    toml_out.append(f'pde = "{pde}"')
    toml_out.append(f"grid_shape = {list(shape)}")
    toml_out.append(f'method = "{method}"')
    toml_out.append(f'norm = "{norm}"')
    toml_out.append(f"value = {value:.3e}")
    toml_out.append(f"tolerance = {tol}")
    toml_out.append('analytical_solution = "see scripts/calibrate_floors.py"')
    toml_out.append('citation = "Week 1 floor calibration PR"')
    toml_out.append("")

Path("src/physics_lint/data/floors.toml").write_text("\n".join(toml_out))
print("wrote src/physics_lint/data/floors.toml")
PY
cat src/physics_lint/data/floors.toml
```

- [ ] **Step 6: Switch back to main and open a PR against main with just `floors.toml`**

```bash
git checkout master
git checkout -b chore/populate-floors
git add src/physics_lint/data/floors.toml
git commit -m "chore: populate floors.toml from 3-environment calibration

Per design doc §6.4 two-step calibration protocol:
Step 1: calibration on chore/floor-calibration branch + local + Docker
        (values recorded; branch NOT merged)
Step 2: this PR against main with the maximum-across-environments values

Values derived by taking the MAX across local laptop, Ubuntu Docker,
and the throwaway GHA matrix (ubuntu-latest + macos-14). Per-floor
multiplicative tolerance: 3.0 for spectral (near FFT backend noise),
2.0 for fd4 (truncation dominates).

The floor-calibration workflow on chore/floor-calibration is NOT merged
to main; it's ephemeral and the branch stays as historical record."
```

- [ ] **Step 7: Run the full suite with floors.toml in place**

```bash
pytest -q
```

Expected: all tests pass. The shipped-default fallback in `_helpers.py` is no longer exercised for the calibrated rules, but it should still be there for any uncalibrated tuples.

- [ ] **Step 8: Merge the PR and confirm CI is green**

```bash
git push -u origin chore/populate-floors
gh pr create --title "chore: populate floors.toml" --body "$(cat <<'EOF'
Populates src/physics_lint/data/floors.toml from the 3-environment floor
calibration (design doc §6.4).

## Test plan
- [x] All pytest tests pass locally
- [ ] CI matrix (6 jobs) passes on this PR
- [ ] Security tab on the repo still clean
EOF
)"
# Wait for CI; merge when green
```

**If the CI fails on any job**, the root cause is almost always that one platform measured a higher value than was captured in the 3-environment sample. Fix: bump the `value` on the failing floor entry by a factor of 2 or 3, re-run the CI, and note the discrepancy in the PR body. Do NOT lower the tolerance multiplier — that defeats the calibration discipline.

- [ ] **Step 9: Delete the throwaway `chore/floor-calibration` branch (local + remote)**

```bash
git branch -D chore/floor-calibration
git push origin :chore/floor-calibration    # delete remote branch
```

The calibration workflow and its YAML only ever existed on the throwaway branch — main never sees it, which is what keeps the validation non-circular (§6.4).

---

## End-of-Week-1 sanity check

- [ ] **Step 1: Run everything**

```bash
pytest --cov=physics_lint --cov-report=term-missing -q
```

Expected:
- All tests pass
- Coverage >= 85% (the coverage gate from pyproject.toml)
- No ruff, codespell, or pre-commit failures

- [ ] **Step 2: Verify the public API is importable and works programmatically**

```bash
python - <<'PY'
import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_res_001, ph_bc_001, ph_pos_001, ph_pos_002

# Build a Laplace problem with a harmonic input
N = 64
x = np.linspace(0.0, 1.0, N)
y = np.linspace(0.0, 1.0, N)
X, Y = np.meshgrid(x, y, indexing="ij")
u = X ** 2 - Y ** 2
field = GridField(u, h=(1.0 / (N - 1), 1.0 / (N - 1)), periodic=False)

spec = DomainSpec.model_validate({
    "pde": "laplace",
    "grid_shape": [N, N],
    "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
    "periodic": False,
    "boundary_condition": {"kind": "dirichlet"},
    "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
})

# Run four rules
print("PH-RES-001:", ph_res_001.check(field, spec).status)
print("PH-BC-001:",  ph_bc_001.check(field, spec, boundary_target=field.values_on_boundary()).status)
print("PH-POS-001:", ph_pos_001.check(field, spec).status)
print("PH-POS-002:", ph_pos_002.check(field, spec, boundary_values=field.values_on_boundary()).status)
PY
```

Expected: four lines, all showing `PASS` or `SKIPPED` (PH-POS-001 skips because `dirichlet` is not sign-preserving; the other three pass because `x^2 - y^2` is harmonic).

- [ ] **Step 3: Confirm CI green on main**

```bash
gh run list --branch master --limit 3
```

Expected: the latest run on main is green across all six matrix cells.

- [ ] **Step 4: Tag the Week-1 deliverable**

```bash
git tag -a v0.1.0.dev1 -m "Week 1 deliverable: Laplace/Poisson + BC + positivity

Closes Week 1 of the physics-lint V1 plan (docs/plans/2026-04-14-physics-lint-v1-week-1.md).

Deliverable:
- Field ABC + GridField (FD + spectral) + CallableField
- DomainSpec pydantic v2 hierarchy
- Hybrid adapter+dump loader
- Norms: l2_grid, h_minus_one_spectral, trapezoidal_integral
- Rules: PH-RES-001/002/003, PH-BC-001/002, PH-POS-001/002
- Analytical battery: Laplace (2), Poisson (2)
- Hypothesis property tests: 5 properties
- Lazy rule registry
- Calibrated floors.toml across 3 environments
- Six-job CI matrix green

Week 2: heat + wave + conservation + dogfood prep
Week 3: symmetry + broken-model toy + (conditional) MeshField
Week 4: report + CLI + SARIF + release v1.0.0"
git push origin v0.1.0.dev1
```

---

## Self-review

I ran the writing-plans skill self-review checklist after drafting:

**Spec coverage** — Week 1 spec requirements mapped to tasks:
- §1.1–§1.2 Laplace/Poisson math → Task 9 analytical battery
- §2 tooling stack → Task 1 scaffolding (ruff, codespell, pytest, hypothesis, CI matrix, pyproject.toml)
- §3.1–§3.3 Field ABC + GridField + CallableField → Tasks 2, 3, 4, 5
- §3.4 MeshField conditional → Task 6 spike
- §4 DomainSpec pydantic hierarchy → Task 7
- §5 hybrid loader → Task 8
- §6.1–§6.4 self-test battery + floors.toml + calibration protocol → Tasks 9, 14
- §7.1–§7.4 residual formulas + norms → Tasks 4 (norms), 10 (PH-RES rules)
- §7.5 variational-correctness rules → Task 10 (PH-RES-001 uses H^-1 / L^2 branching)
- §8.5 PH-BC-001 mode branching → Task 11
- §8.6 positivity rules → Task 12
- §10.2 rule catalog rows (Week-1 subset) → covered in Tasks 10, 11, 12
- §10.4 lazy registry → Task 9
- §11 report schema → Task 9
- §15 hypothesis properties → Task 13

**Gaps (knowingly deferred to later weeks, NOT Week 1):**
- PH-RES-001 Poisson source-term handling (Poisson residual = -Δu - f) — the Week 1 test uses Laplace only; `check()` raises `NotImplementedError` for Poisson. Week 2 plumbs the source term through the loader.
- Heat + wave rules (PH-CON-001/002/003, Bochner norm) — Week 2.
- Symmetry rules (PH-SYM-001/002/003/004) — Week 3.
- Meta-warning rules (PH-VAR-001/002, PH-NUM-001-004) — scattered across Weeks 2-4.
- CLI (`physics-lint` binary) — Week 4 Day 2.
- SARIF output — Week 4 Day 3.
- Full report module with `plot()` and `summary()` — Week 4 Day 1. Week 1 only ships the dataclass skeleton enough to satisfy `exit_code` and `overall_status`.
- README, docs pages, Sphinx theming — Week 4 Day 5.

**Placeholder scan:** No TBD / TODO / "add error handling" / "similar to Task N" patterns. Every code block is complete.

**Type consistency:** `RuleResult.status` is `Literal["PASS", "WARN", "FAIL", "SKIPPED"]` everywhere. `DomainSpec.boundary_condition` is a `BCSpec`, never a string at the rule-consumer level (the string form is only the user-facing TOML shorthand, normalized in `_normalize_config_shape`). `check(field, spec)` is the consistent rule signature across all rule modules.

**Scope:** Week 1 only. Weeks 2–4 will get their own plans as each week approaches.

---

## Execution handoff

Plan complete and saved to `docs/plans/2026-04-14-physics-lint-v1-week-1.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. For this plan, task boundaries are the natural checkpoint and each task is self-contained (tests + impl + commit), so subagent-driven mode fits well.

2. **Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints for review. Appropriate if you'd rather watch each task land in real time.

**Which approach?** Note: per your original framing, the subagent (or inline executor) has explicit license to flag if the design doc is missing implementation-level detail it needs — e.g., if the PH-RES-002 interior comparison ends up needing a tolerance that wasn't specified, flag it as a design-doc gap rather than guess.

