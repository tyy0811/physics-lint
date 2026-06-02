"""MeshVectorField — a 2D vector (velocity) field on an unstructured triangular mesh.

Subclasses the Field ABC directly (NOT MeshField — that would inherit MeshField's
scalar `integrate` / `laplacian_l2_projected_zero_trace`). Holds a scikit-fem
MeshTri + ElementTriP1 basis (built from node_positions + cells) and a (T, N, 2)
velocity array. The scalar-ABC methods (`at`, `grad`, `laplacian`, `integrate`,
`values_on_boundary`) raise NotImplementedError — a velocity-on-mesh is not a
scalar field; PH-CON-005 reads `.values()` + `.basis` directly.

2D only in v1.2.0 (MeshTri / ElementTriP1 / cells (M, 3)). A 3D path needs
MeshTet / ElementTetP1 / cells (M, 4) and is a follow-on; with triangles and 3D
positions the divergence would be a surface divergence, not the volumetric div v.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field._base import Field


class MeshVectorField(Field):
    def __init__(
        self,
        *,
        node_positions: np.ndarray,
        cells: np.ndarray,
        velocity: np.ndarray,
    ) -> None:
        nodes = np.asarray(node_positions, dtype=np.float64)
        cells_arr = np.asarray(cells)
        vel = np.asarray(velocity)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError(
                f"node_positions must be (N, 2) for 2D MeshVectorField; got {nodes.shape}"
            )
        if cells_arr.ndim != 2 or cells_arr.shape[1] != 3:
            raise ValueError(
                f"cells must be (M, 3) triangles (MeshTri/ElementTriP1); got "
                f"{cells_arr.shape}. 3D MeshTet (cells (M, 4)) is a v1.2.x follow-on."
            )
        if vel.ndim != 3 or vel.shape[2] != 2:
            raise ValueError(
                f"velocity must be (T, N, 2) for a 2D field; got {vel.shape}. "
                f"3D velocity is a v1.2.x follow-on."
            )
        if vel.shape[1] != nodes.shape[0]:
            raise ValueError(
                f"velocity N={vel.shape[1]} must match node_positions N={nodes.shape[0]}"
            )
        self._node_positions = nodes
        self._cells = cells_arr.astype(np.int64)
        self._velocity = vel
        self._basis = self._build_basis()

    def _build_basis(self):  # type: ignore[no-untyped-def]
        import skfem

        mesh = skfem.MeshTri(
            np.ascontiguousarray(self._node_positions.T),
            np.ascontiguousarray(self._cells.T),
        )
        return skfem.Basis(mesh, skfem.ElementTriP1())

    @property
    def basis(self):  # type: ignore[no-untyped-def]
        return self._basis

    def values(self) -> np.ndarray:
        """Return the (T, N, 2) velocity array."""
        return self._velocity

    def at(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "MeshVectorField has no scalar point-evaluation; PH-CON-005 reads "
            ".values() + .basis directly."
        )

    def grad(self) -> list[Field]:
        raise NotImplementedError(
            "MeshVectorField is a vector field; per-axis scalar grad is undefined."
        )

    def laplacian(self) -> Field:
        raise NotImplementedError("MeshVectorField does not implement a scalar Laplacian.")

    def integrate(self, weight: Field | None = None) -> float:
        raise NotImplementedError("MeshVectorField does not implement scalar integration.")

    def values_on_boundary(self) -> np.ndarray:
        raise NotImplementedError("MeshVectorField does not implement a scalar boundary trace.")
