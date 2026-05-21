"""Lock physics_lint.__version__ to the package version, end to end.

Regression guards against the drift that shipped the v1.0.0 wheel with
`__version__ == "0.0.0.dev0"`:

- `__version__` was a hand-maintained literal in `__init__.py` that
  diverged from `pyproject.toml`. It now derives from
  `importlib.metadata`; `test_version_matches_package_metadata` fails if
  a future change re-introduces a hand-maintained literal.
- `test_package_metadata_matches_pyproject` closes the remaining gap:
  since `__version__` derives from `importlib.metadata`, the first test
  alone cannot detect installed metadata that is stale relative to
  `pyproject.toml`. This second test pins metadata to the source of
  truth.
"""

import sys
from importlib.metadata import version
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import physics_lint


def test_version_matches_package_metadata():
    assert physics_lint.__version__ == version("physics-lint")


def test_package_metadata_matches_pyproject():
    """Installed metadata must match pyproject.toml's declared version.

    Catches a stale editable install: if pyproject.toml's version was
    bumped but the package not reinstalled, `importlib.metadata` returns
    the old `.dist-info` value. CI reinstalls on every run so this only
    bites a long-lived local venv — where the failure is the fix
    ("reinstall the package").
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert version("physics-lint") == declared
