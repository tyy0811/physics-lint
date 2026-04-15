"""MeshField — Field backed by a scikit-fem Basis + DOF vector.

Design doc §3.4. The Week-1 Day-2 scikit-fem spike passed on 2026-04-14
(commit 941658d); MeshField ships unconditionally in V1.

V1 scope and the ``.laplacian()`` stub
--------------------------------------

The ``Field`` ABC contract for ``.laplacian()`` says the returned Field's
``values()`` has shape ``(*spatial,)``, with per-point values interpreted
pointwise. ``GridField`` and ``CallableField`` satisfy this contract by
returning an FD/spectral or autograd-computed pointwise approximation at
every DOF.

**MeshField does NOT implement ``.laplacian()`` in V1** because the V1
finite-element operator does not satisfy the pointwise-approximation
contract that the ABC implies. ``.laplacian()`` raises
``NotImplementedError`` and points callers at
``laplacian_l2_projected_zero_trace()``, which is the V1 operator
physics-lint's mesh rules actually consume.

V1.1 may add a true pointwise ``.laplacian()`` via superconvergent patch
recovery (SPR) of gradients followed by a second recovery pass, or via
another technique that satisfies the ABC contract at every DOF. That is
tracked as a V1.1 backlog item; it is deliberately out of Week-3 scope.

The V1 operator: ``laplacian_l2_projected_zero_trace``
------------------------------------------------------

Given a MeshField ``u`` on a basis, this method returns a new MeshField
whose DOFs are the L² projection of ``Δu`` onto the **zero-trace FE
subspace** ``V_{h,0}`` — that is, the subset of the FE space whose
members vanish on the domain boundary. Concretely, the returned DOFs
satisfy::

    M_II lap_I = -(K u)_I          on interior DOFs
    lap_B      = 0                  on boundary DOFs

where ``K = ∫ ∇u · ∇v dx`` is the stiffness matrix, ``M = ∫ u v dx`` is
the consistent mass matrix, and ``I``/``B`` index interior / boundary
DOFs. The implementation uses ``skfem.condense`` and ``skfem.solve`` to
enforce ``lap_B = 0`` as a hard Dirichlet boundary condition on the
projection.

**Why not the plan's raw ``-M⁻¹ K u``.** The plan scaffolded
``lap = -M⁻¹ K u`` as a Galerkin projection of the pointwise Laplacian.
This formulation is mathematically unsound for fields with non-zero
``∂u/∂n`` on the boundary: integration by parts leaves a boundary flux
term ``∫_{∂Ω} (∂u/∂n) v dS`` that the stiffness matrix alone does not
capture. Worse, the error does not stay at the boundary — the mass
matrix ``M`` couples interior and boundary rows, so the missing flux
term pollutes ``M⁻¹ K u`` globally. Numerical verification on
``u = sin(πx) sin(πy)`` at P2 refine=4 shows interior relative error
~260% for the raw formula, vs ~1.2% for the zero-trace projection.

**Boundary semantics.** The boundary trace of the returned MeshField is
**not** a pointwise approximation to the true boundary value of ``Δu``;
it is hard-pinned to zero by construction. Downstream rules must either
restrict to interior DOFs/elements by construction or explicitly
acknowledge the artifact. PH-CON-004 (the V1 consumer) takes the
structural approach: its per-element residual excludes boundary
elements so boundary-DOF values never reach the rule output.

**Convergence.** The refinement test in ``tests/test_meshfield.py``
observed O(h²) interior convergence on the smooth analytical solution
``sin(πx) sin(πy)`` with P2 elements. The rate may be lower for
non-smooth fields (piecewise H¹ but not H² inputs, discontinuous
sources, kinked solutions) where L² projection is at best O(h). This
is a property of the inputs, not a regression.

**Scope of V1 public methods.** V1 exposes ``values``,
``laplacian_l2_projected_zero_trace``, ``integrate``, and
``values_on_boundary``. ``at`` (point evaluation) and ``grad`` (per-axis
partials) raise ``NotImplementedError``; they are not consumed by Week-3
rules and adding them needs an interpolation path that is V1.1 work.
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

    def laplacian(self) -> Field:
        raise NotImplementedError(
            "MeshField does not implement a pointwise Laplacian in V1. The V1 "
            "finite-element operator is an L² projection of Δu onto the zero-"
            "trace FE subspace and does not satisfy the pointwise-approximation "
            "contract that the Field ABC's .laplacian() implies (boundary DOFs "
            "are hard-pinned to 0 rather than approximating the true boundary "
            "trace of Δu). Use MeshField.laplacian_l2_projected_zero_trace() "
            "for the V1 operator; a true pointwise .laplacian() via SPR or "
            "similar is tracked as V1.1 backlog."
        )

    def laplacian_l2_projected_zero_trace(self) -> MeshField:
        """Return the L² projection of Δu onto the zero-trace FE subspace.

        See module docstring for full semantics. Briefly: this is NOT a
        pointwise Laplacian; it is an FE projection with boundary DOFs
        hard-pinned to 0. Interior convergence is O(h²) on smooth inputs
        (empirically observed; see ``tests/test_meshfield.py``).
        """
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
