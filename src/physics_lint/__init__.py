"""physics-lint — linter for trained neural PDE surrogates.

See docs/design/2026-04-14-physics-lint-v1.md for the V1 design.
"""

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
    "BCSpec",
    "CallableField",
    "DomainSpec",
    "Field",
    "FieldSourceSpec",
    "GridDomain",
    "GridField",
    "SARIFSpec",
    "SymmetrySpec",
    "__version__",
]
