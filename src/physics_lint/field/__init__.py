"""Field abstraction: ABC + concrete subclasses (GridField, CallableField, MeshField).

GridField, CallableField, and MeshField are added as they land (Tasks 3, 5, 6).
"""

from physics_lint.field._base import Field
from physics_lint.field.callable import CallableField
from physics_lint.field.grid import GridField

try:
    from physics_lint.field.mesh import MeshField

    _HAS_MESH = True
except ImportError:
    _HAS_MESH = False
    MeshField = None  # type: ignore[assignment,misc]

__all__ = ["CallableField", "Field", "GridField", "MeshField"]
