"""FEM quadrature-order exactness helpers for Task 11 (PH-NUM-001).

Consumers: ``external_validation/PH-NUM-001/test_anchor.py``.

Scope (2026-04-24 user-revised Task 11 contract): *simple polynomial integrals
over a full FEM assembly first*. The rule PH-NUM-001 is a V1 structural stub
that emits ``PASS`` with a pass-through baseline integral; a full MMS FEM
h-refinement sweep with the Ciarlet `p + 1` rate is **not** what the rule
emits in V1, so this harness stays close to the mathematical statement:

    The n-point Gauss-Legendre quadrature on a 1D interval exactly
    integrates polynomials of degree <= 2n - 1. scikit-fem's ``intorder``
    parameter selects a quadrature rule whose exactness degree is at
    least ``intorder`` (the library rounds up to a standard rule;
    ``intorder=k`` gives exactness of degree k or higher, depending on
    the element).

The harness integrates monomials ``x**d`` and 2D products ``x**dx * y**dy``
over a scikit-fem mesh, compares to analytical values, and demonstrates
three regimes:

    Case A  (exact)      : degree <= intorder -> error at float64 roundoff
    Case B  (under-int)  : degree >  intorder -> error bounded away from 0
    Case C  (convergence): increase intorder -> error decreases and
                           eventually reaches roundoff once intorder
                           matches or exceeds the polynomial degree.

No full FEM assembly is performed here; no `a(u, v) vs a_h(u, v)` bilinear
form is computed. The variational-crime framing in the CITATION.md F1
proof-sketch is cited at chapter-level via Ciarlet 2002 section 4.1 +
Strang 1972 + Brenner-Scott section 10.3; the harness layer anchors the
core mathematical claim of quadrature-exactness + quadrature-under-
integration at the level the V1 rule's emitted quantity
(``field.integrate()``) actually exercises.
"""

from __future__ import annotations

import numpy as np


def _mesh_and_basis(n_refine: int, intorder: int):
    """Unit-square triangular mesh + P2 basis with explicit intorder.

    Uses ``MeshTri.init_sqsymmetric()`` as the base mesh (4 triangles in a
    unit square, symmetric; avoids the asymmetry of the default
    ``init_refdom`` which is a single right triangle). ``n_refine`` uniform
    refinements give ~``4 * 4**n_refine`` triangles.
    """
    from skfem import Basis, ElementTriP2, MeshTri

    mesh = MeshTri.init_sqsymmetric().refined(n_refine)
    basis = Basis(mesh, ElementTriP2(), intorder=intorder)
    return mesh, basis


def analytical_monomial_1d_on_unit_square(degree: int) -> float:
    """Closed-form integral of ``x**degree`` over the unit square [0,1]^2.

    ``int_0^1 int_0^1 x^d dy dx = 1/(d + 1)``.
    """
    return 1.0 / (degree + 1)


def analytical_product_monomial_on_unit_square(dx: int, dy: int) -> float:
    """Closed-form integral of ``x**dx * y**dy`` over the unit square [0,1]^2.

    ``(int_0^1 x^dx dx) * (int_0^1 y^dy dy) = 1/((dx + 1)(dy + 1))``.
    """
    return 1.0 / ((dx + 1) * (dy + 1))


def integrate_monomial_1d(degree: int, *, intorder: int, n_refine: int = 1) -> float:
    """Integrate ``x**degree`` over [0, 1]^2 via scikit-fem at given intorder."""
    from skfem import Functional

    _mesh, basis = _mesh_and_basis(n_refine, intorder)

    @Functional
    def poly(w):  # type: ignore[no-untyped-def]
        return w.x[0] ** degree

    return float(poly.assemble(basis))


def integrate_product_monomial(dx: int, dy: int, *, intorder: int, n_refine: int = 1) -> float:
    """Integrate ``x**dx * y**dy`` over [0, 1]^2 via scikit-fem at given intorder."""
    from skfem import Functional

    _mesh, basis = _mesh_and_basis(n_refine, intorder)

    @Functional
    def poly(w):  # type: ignore[no-untyped-def]
        return (w.x[0] ** dx) * (w.x[1] ** dy)

    return float(poly.assemble(basis))


def quadrature_error_monomial_1d(degree: int, *, intorder: int, n_refine: int = 1) -> float:
    """Absolute error between numerical and analytical integral of ``x**degree``."""
    num = integrate_monomial_1d(degree, intorder=intorder, n_refine=n_refine)
    ana = analytical_monomial_1d_on_unit_square(degree)
    return abs(num - ana)


def quadrature_error_product_monomial(
    dx: int, dy: int, *, intorder: int, n_refine: int = 1
) -> float:
    """Absolute error between numerical and analytical integral of ``x**dx * y**dy``."""
    num = integrate_product_monomial(dx, dy, intorder=intorder, n_refine=n_refine)
    ana = analytical_product_monomial_on_unit_square(dx, dy)
    return abs(num - ana)


def convergence_sweep_over_intorder(
    degree: int,
    *,
    intorders: tuple[int, ...],
    n_refine: int = 1,
) -> np.ndarray:
    """Return an array of absolute errors for integrating ``x**degree`` across
    the given ``intorders``. Expected behavior (Case C):

    - Monotonically non-increasing as ``intorder`` rises (scikit-fem rounds
      ``intorder`` up to the next quadrature rule, so adjacent ``intorder``
      values may give identical errors; strictly-decreasing is NOT required).
    - Error at ``intorder >= degree`` drops to float64 roundoff.
    """
    return np.array(
        [quadrature_error_monomial_1d(degree, intorder=io, n_refine=n_refine) for io in intorders]
    )


def is_non_increasing(values: np.ndarray, *, slack: float = 1e-15) -> bool:
    """Returns True iff ``values[i+1] <= values[i] + slack`` for all i.

    ``slack`` allows for float-roundoff flatness between already-converged
    entries; adjacent ``intorder`` that both sit at roundoff (e.g. 1e-17
    and 3e-16) should still count as non-increasing.
    """
    return all(values[i + 1] <= values[i] + slack for i in range(len(values) - 1))
