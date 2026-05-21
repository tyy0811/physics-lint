"""physics-lint — linter for trained neural PDE surrogates.

See docs/design/2026-04-14-physics-lint-v1.md for the V1 design.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from physics_lint.field import CallableField, Field, GridField, MeshField
from physics_lint.spec import (
    BCSpec,
    DomainSpec,
    FieldSourceSpec,
    GridDomain,
    SARIFSpec,
    SymmetrySpec,
)

try:
    __version__ = _version("physics-lint")
except _PackageNotFoundError:  # source tree, not installed
    __version__ = "0.0.0+unknown"
__all__ = [
    "BCSpec",
    "CallableField",
    "DomainSpec",
    "Field",
    "FieldSourceSpec",
    "GridDomain",
    "GridField",
    "MeshField",
    "SARIFSpec",
    "SymmetrySpec",
    "__version__",
]
