"""MeshField — Field backed by a scikit-fem Basis + DOF vector.

Design doc §3.4. The Week-1 Day-2 scikit-fem spike passed on 2026-04-14
(commit 941658d); MeshField ships unconditionally in V1.

**Laplacian via Galerkin projection.** The continuous Laplacian is
projected onto the FE space via ``M lap = -K u`` where ``K`` is the
stiffness matrix (``∫ grad(u) · grad(v) dx``) and ``M`` is the
consistent mass matrix (``∫ u * v dx``). Both are assembled from
scikit-fem's built-in BilinearForm helpers and returned as scipy
sparse matrices.

The raw ``-M^{-1} K u`` projection is contaminated at the boundary
whenever ``∂u/∂n ≠ 0``: the missing boundary-flux term from
integration by parts (``∫ (∂u/∂n) v ds``) pollutes rows touching the
boundary DOFs and blows up the nodal values there. To keep the
projected Laplacian well-behaved the system is solved with
*homogeneous Dirichlet* on the boundary DOFs of the result via
``skfem.condense/solve`` — i.e. the returned Laplacian lives in the
zero-trace subspace. This matches the convention the scikit-fem spike
(commit 941658d) used for the Poisson manufactured-solution check and
is the right scope for V1: PH-CON-004 / PH-NUM-001 only consume the
Laplacian on interior DOFs, where this projection is consistent and
superconverges on P2 meshes for smooth inputs. Fields whose
analytical Laplacian does *not* vanish on the boundary will still be
correct in the interior; the boundary trace of the returned Laplacian
should not be interpreted as a pointwise value of Δu.

**Scope.** V1 exposes ``values``, ``laplacian``, ``integrate``, and
``values_on_boundary``. ``at`` and ``grad`` are deferred to V1.1:
point-evaluation and per-component gradients require an interpolation
path the Week-3 rules don't consume.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from physics_lint.field._base import Field


class MeshField(Field):
    def __init__(self, *, basis: Any, dofs: np.ndarray) -> None:
        self._basis = basis
        self._dofs = np.asarray(dofs)
        self._cached_lap: np.ndarray | None = None

    def values(self) -> np.ndarray:
        return self._dofs

    def at(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("MeshField.at() lands in V1.1 if needed")

    def grad(self) -> list[Field]:
        raise NotImplementedError("MeshField.grad() lands in V1.1 if needed")

    def laplacian(self) -> MeshField:
        if self._cached_lap is None:
            from skfem import BilinearForm, condense, solve
            from skfem.helpers import dot, grad

            @BilinearForm
            def _laplace_form(u, v, _):
                return dot(grad(u), grad(v))

            @BilinearForm
            def _mass_form(u, v, _):
                return u * v

            k_mat = _laplace_form.assemble(self._basis)
            m_mat = _mass_form.assemble(self._basis)

            # Galerkin Laplacian with homogeneous Dirichlet on boundary DOFs:
            #   M lap = -K u   on interior,   lap = 0 on the boundary trace.
            # See module docstring for why the boundary is pinned to 0 rather
            # than left to the raw L^2 projection (which is contaminated by
            # the missing ∫ (∂u/∂n) v ds flux term when ∂u/∂n ≠ 0).
            rhs = -(k_mat @ self._dofs)
            boundary_dofs = self._basis.get_dofs().flatten()
            lap_dofs = solve(*condense(m_mat, rhs, D=boundary_dofs))
            self._cached_lap = np.asarray(lap_dofs)

        return MeshField(basis=self._basis, dofs=self._cached_lap)

    def integrate(self, weight: Field | None = None) -> float:
        from skfem import LinearForm

        @LinearForm
        def _integrand(v, _):
            return v

        unit_form = _integrand.assemble(self._basis)

        if weight is None:
            return float(unit_form @ self._dofs)

        if not isinstance(weight, MeshField):
            raise TypeError("MeshField.integrate weight must be a MeshField")

        # Weighted integral approximation using diagonal mass.
        return float((self._dofs * weight._dofs) @ unit_form)

    def values_on_boundary(self) -> np.ndarray:
        boundary_dofs = self._basis.get_dofs().flatten()
        return self._dofs[boundary_dofs]
