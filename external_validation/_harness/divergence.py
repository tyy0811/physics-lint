"""Gauss-Green / divergence-theorem helpers (harness-level, authoritative F2).

Populated for Task 5 (PH-BC-002). This module implements the general
Gauss-Green identity verification on the unit square for `F=(x,y)`,
independent of the production rule PH-BC-002's V1 Laplace-only emitted
quantity.

**Scope-separation discipline** (per complete-v1.0 plan §13 and the
2026-04-24 V1-stub CRITICAL-task pattern): this module is the
**authoritative** harness-level Function 2 correctness fixture for
Task 5. The production rule PH-BC-002 is Laplace-scope only
(`src/physics_lint/rules/ph_bc_002.py:8`: "Week 1 scope: Laplace only
(expected imbalance is zero). The Poisson arm raises
`NotImplementedError` ..."), so it does not and cannot validate general
vector-flux Gauss-Green. Task 5's external validation separates:

- (F2 harness-level, here) a free-standing Gauss-Green correctness
  fixture for `F=(x,y)` on the unit square where `∫_Ω div F dV = 2`
  and `∫_{∂Ω} F·n dS = 2` analytically.
- (rule-verdict contract, in `external_validation/PH-BC-002/test_anchor.py`)
  the production rule's actual V1 emitted quantity (`∫Δu dV` on a
  Laplace-harmonic fixture) — Laplace scope only.

CITATION.md, README, and docstrings must not imply broader production
coverage than the rule provides.

References:
- Evans 2010 Appendix C.2 Gauss-Green Theorem 1 (section-level per
  `TEXTBOOK_AVAILABILITY.md` ⚠)
- Gilbarg-Trudinger 2001 §2.4 (section-level per `TEXTBOOK_AVAILABILITY.md` ⚠)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

MeshType = Literal["tri", "quad"]


def gauss_green_on_unit_square(
    *,
    mesh_type: MeshType,
    n_refine: int = 8,
) -> tuple[float, float]:
    """Compute (LHS, RHS) of Gauss-Green on unit square for F=(x, y).

    ``LHS = ∫_Ω (∂F_x/∂x + ∂F_y/∂y) dV = ∫_Ω 2 dV = 2``
    ``RHS = ∫_{∂Ω} F·n dS                = 2``

    Parameters
    ----------
    mesh_type : "tri" or "quad"
        "tri" uses `skfem.MeshTri.init_tensor` with P1 elements;
        "quad" uses `skfem.MeshQuad.init_tensor` with Q1 elements.
    n_refine : int, default 8
        Number of mesh subdivisions per axis. Both LHS and RHS should
        equal 2 within float64 roundoff (~1e-15) because F=(x,y) is a
        polynomial of degree 1 and scikit-fem's default Gaussian
        quadrature exactly integrates polynomials up to the element's
        order.

    Returns
    -------
    (lhs, rhs) : tuple[float, float]
        The volume-integrated divergence and boundary-integrated flux
        computed independently via scikit-fem quadrature.

    Notes
    -----
    Requires `scikit-fem` (optional dep under `[project.optional-
    dependencies].mesh` in `pyproject.toml`; currently pinned >=10).
    """
    from skfem import (
        Basis,
        ElementQuad1,
        ElementTriP1,
        FacetBasis,
        Functional,
        MeshQuad,
        MeshTri,
    )

    xs = np.linspace(0.0, 1.0, n_refine + 1)
    if mesh_type == "tri":
        mesh = MeshTri.init_tensor(xs, xs)
        element = ElementTriP1()
    elif mesh_type == "quad":
        mesh = MeshQuad.init_tensor(xs, xs)
        element = ElementQuad1()
    else:
        raise ValueError(f"unknown mesh_type={mesh_type!r}; expected 'tri' or 'quad'")

    basis = Basis(mesh, element)
    fbasis = FacetBasis(mesh, element)

    @Functional
    def div_f(w):
        # F = (x, y), div F = ∂F_x/∂x + ∂F_y/∂y = 1 + 1 = 2 (constant).
        return 2.0 * np.ones_like(w.x[0])

    @Functional
    def flux(w):
        # F·n = x · n_x + y · n_y on the outward boundary normal.
        return w.x[0] * w.n[0] + w.x[1] * w.n[1]

    lhs = float(div_f.assemble(basis))
    rhs = float(flux.assemble(fbasis))
    return lhs, rhs
