"""PH-BC-002 external-validation anchor - Gauss-Green / divergence theorem.

Task 5 of the complete-v1.0 plan. Applies the V1-stub CRITICAL-task
pattern (2026-04-24 `feedback_critical_rule_stub_three_layer_contract.md`):
the production rule PH-BC-002 is Laplace-scope only (Week 1 scope per
src/physics_lint/rules/ph_bc_002.py line 8), strictly narrower than the
general Gauss-Green mathematical anchor. External validation separates:

    (F1)  Mathematical-legitimacy anchor: Gauss-Green / divergence
          theorem (Evans 2010 App C.2 Thm 1, section-level ⚠;
          Gilbarg-Trudinger 2001 §2.4, section-level ⚠).
    (F2)  Harness-level correctness fixture (authoritative):
          external_validation/_harness/divergence.py
          `gauss_green_on_unit_square(mesh_type, n_refine)`.
          Verifies LHS = ∫_Ω div F dV = 2 and RHS = ∫_{∂Ω} F·n dS = 2
          on F=(x,y) unit square, both triangulation and
          quadrilateralization. Independent of the production rule's
          emitted quantity.
    (F3)  Borrowed-credibility: absent-with-justification per plan §13
          rationale (Gauss-Green reproduction on MMS fixtures is
          tautological under the theorem's stated preconditions, so
          borrowed-credibility via published numerical baseline is not
          applicable). LeVeque 2002 FVM §2.1 in Supplementary
          calibration context (pedagogical framing flag).
    (RVC) Rule-verdict contract: exercises the rule's V1 Laplace-scope
          emitted quantity (∫Δu dV for a harmonic field). u = x² - y²
          gives exact 0 (FD4 is exact on polynomials of degree ≤ 4, on
          both interior and boundary stencils). u = x⁵ - 10x³y² + 5xy⁴
          (5th-degree harmonic) exercises the boundary-stencil O(h²)
          regime — rule WARNs at N=16, PASSes at N ≥ 32. SKIPPED paths
          on Poisson (Week 2 source wiring) and non-laplace/poisson
          PDEs (out of V1 rule scope).

**Wording discipline.** Do not write "PH-BC-002 validates the
divergence theorem generally." The production rule does not. Write:
"PH-BC-002's external validation separates (i) a harness-level Gauss-
Green correctness fixture from (ii) the production rule's currently
supported Laplace-scope verdict behavior."

Plan-diffs logged with the commit (plan-vs-committed-state drift, plan
section 7.4):
    6. (Task 5) Plan §13 F2 fixture "F=(x,y) ... yield LHS = RHS = 2
       within tolerance" recast as harness-level (F2) only. The
       production rule PH-BC-002 is Laplace-scope-only (ph_bc_002.py
       line 8), so the rule's emitted quantity does not exercise the
       F=(x,y) arbitrary-F fixture. Scope separation per 2026-04-24
       Path C approval + V1-stub CRITICAL-task pattern: F2 is
       authoritative via external_validation/_harness/divergence.py;
       rule-verdict contract layer added on Laplace-harmonic fixture
       (u = x² - y², u = x⁵ - 10x³y² + 5xy⁴) to exercise actual V1
       rule scope. CITATION.md + README carry the scope separation
       explicitly.

Plan-diffs 1-5 are from Tasks 2 + 3 (commits 30baf3e, 0cedc7b).
"""

from __future__ import annotations

import numpy as np
import pytest

from external_validation._harness.divergence import gauss_green_on_unit_square
from physics_lint import DomainSpec
from physics_lint.field import GridField
from physics_lint.rules import ph_bc_002

GAUSS_GREEN_TOL = 1e-12
REFINEMENT_LEVELS = (4, 8, 16)
MESH_TYPES = ("tri", "quad")


def _harmonic_poly_deg2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """u = x² - y². Harmonic (Δu ≡ 0); FD4 exact on interior + boundary stencils."""
    return x**2 - y**2


def _harmonic_poly_deg5(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """u = x⁵ - 10 x³ y² + 5 x y⁴. Harmonic (Re of z⁵); FD4 interior exact, boundary O(h²)."""
    return x**5 - 10.0 * x**3 * y**2 + 5.0 * x * y**4


def _run_rule(
    n: int,
    *,
    u_fn=_harmonic_poly_deg2,
    pde: str = "laplace",
    diffusivity: float | None = None,
):
    xs = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    u = u_fn(mesh_x, mesh_y)
    h = 1.0 / (n - 1)
    field = GridField(u, h=(h, h), periodic=False)
    domain = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    if pde in {"heat", "wave"}:
        domain["t"] = [0.0, 1.0]
    spec_dict = {
        "pde": pde,
        "grid_shape": [n, n],
        "domain": domain,
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
    }
    if diffusivity is not None:
        spec_dict["diffusivity"] = diffusivity
    spec = DomainSpec.model_validate(spec_dict)
    return ph_bc_002.check(field, spec)


# =========================================================================
# Layer 2: F2 harness-level Gauss-Green on F=(x,y), unit square.
#
# LHS = ∫_Ω div F dV = 2; RHS = ∫_{∂Ω} F·n dS = 2. Both exact at the
# analytical level for F = (x, y) (div F = 2 constant; boundary flux
# splits as 0 + 1 + 0 + 1 across bottom/top/left/right). Numerical
# integration via scikit-fem Gauss quadrature is exact on polynomials up
# to the element order, so both values equal 2 within float64 roundoff.
#
# This layer is the AUTHORITATIVE F2 correctness fixture for Task 5. It
# is independent of the production rule PH-BC-002 and does not imply
# rule coverage of arbitrary-F Gauss-Green. See module docstring.
# =========================================================================


@pytest.mark.parametrize("mesh_type", MESH_TYPES)
@pytest.mark.parametrize("n_refine", REFINEMENT_LEVELS)
def test_layer2_harness_gauss_green_both_sides_equal_two(mesh_type, n_refine):
    """Harness-level Gauss-Green: LHS and RHS both equal 2 within 1e-12."""
    lhs, rhs = gauss_green_on_unit_square(mesh_type=mesh_type, n_refine=n_refine)
    assert abs(lhs - 2.0) < GAUSS_GREEN_TOL, (
        f"LHS ∫div F dV = {lhs!r} differs from analytical 2 by more than "
        f"{GAUSS_GREEN_TOL:.0e} on {mesh_type} mesh at n_refine={n_refine}"
    )
    assert abs(rhs - 2.0) < GAUSS_GREEN_TOL, (
        f"RHS ∫F·n dS = {rhs!r} differs from analytical 2 by more than "
        f"{GAUSS_GREEN_TOL:.0e} on {mesh_type} mesh at n_refine={n_refine}"
    )
    assert abs(lhs - rhs) < GAUSS_GREEN_TOL, (
        f"Gauss-Green identity violated: LHS={lhs!r}, RHS={rhs!r} on "
        f"{mesh_type} mesh at n_refine={n_refine}"
    )


def test_layer2_harness_identity_invariant_under_refinement():
    """Across triangulation and quadrilateralization at multiple refinements,
    LHS = RHS = 2 (roundoff) — the Gauss-Green identity is mesh-type and
    refinement-level invariant for F=(x,y) on the unit square.
    """
    for mesh_type in MESH_TYPES:
        for n_refine in REFINEMENT_LEVELS:
            lhs, rhs = gauss_green_on_unit_square(mesh_type=mesh_type, n_refine=n_refine)
            assert abs(lhs - 2.0) < GAUSS_GREEN_TOL
            assert abs(rhs - 2.0) < GAUSS_GREEN_TOL


# =========================================================================
# Rule-verdict contract (RVC): exercises the production rule PH-BC-002's
# ACTUAL V1 Laplace-scope emitted quantity. Does NOT claim the rule
# validates general Gauss-Green; that's the harness layer above.
# =========================================================================


def test_rvc_rule_passes_on_degree2_harmonic():
    """Degree-2 harmonic u = x² - y²: FD4 Δu = 0 exactly (stencil exact on
    polynomials of degree ≤ 4, both interior and boundary). Rule emits
    raw_value = 0 exactly → PASS at every refinement level.
    """
    for n in (16, 32, 64):
        result = _run_rule(n, u_fn=_harmonic_poly_deg2)
        assert result.status == "PASS", (
            f"rule status at n={n} expected PASS on degree-2 harmonic, got {result.status!r}"
        )
        assert result.raw_value == pytest.approx(0.0, abs=1e-14), (
            f"rule raw_value at n={n} expected 0 on degree-2 harmonic, got {result.raw_value!r}"
        )


def test_rvc_rule_converges_on_degree5_harmonic():
    """Degree-5 harmonic u = x⁵ - 10x³y² + 5xy⁴: FD4 interior is exact on
    polynomials of degree ≤ 4 but boundary stencils are 2nd-order (error
    ∝ 4th derivative of u, which is nonzero for degree 5). Rule WARNs at
    N=16, PASSes at N ∈ {32, 64} as boundary-layer error shrinks.
    """
    prior = None
    for n in (32, 64):
        result = _run_rule(n, u_fn=_harmonic_poly_deg5)
        assert result.status == "PASS", (
            f"rule status at n={n} expected PASS on degree-5 harmonic, got {result.status!r}"
        )
        assert abs(result.raw_value) < 0.01, (
            f"rule raw_value at n={n} expected < 0.01 on degree-5 harmonic, got {result.raw_value!r}"
        )
        if prior is not None:
            # Raw values should decrease (boundary FD4 O(h²) convergence).
            assert abs(result.raw_value) < abs(prior), (
                f"raw_value did not decrease from n={n // 2} → n={n}: "
                f"prior={prior!r}, current={result.raw_value!r}"
            )
        prior = result.raw_value


def test_rvc_rule_warns_on_degree5_harmonic_at_coarse_grid():
    """At N=16 the boundary FD4 O(h²) error exceeds the rule's 0.01
    threshold on degree-5 harmonic. This is documented rule behavior on
    smooth non-polynomial-degree-≤4 harmonics at coarse resolution, not
    a rule bug. If this flips to PASS, the rule's boundary stencil or
    threshold changed — audit grid.py / ph_bc_002.py before softening.
    """
    result = _run_rule(16, u_fn=_harmonic_poly_deg5)
    assert result.status == "WARN", (
        f"rule status at n=16 expected WARN on degree-5 harmonic due to "
        f"boundary FD4 O(h²) error at coarse grid; got {result.status!r}"
    )


def test_rvc_rule_skipped_on_poisson_week1_scope():
    """Week 1 rule scope excludes Poisson (source-term integration lands
    in Week 2 per ph_bc_002.py:8). Rule SKIPs with the documented reason.
    """
    result = _run_rule(16, pde="poisson")
    assert result.status == "SKIPPED", (
        f"rule status on Poisson expected SKIPPED (Week 2 wiring), got {result.status!r}"
    )
    assert result.reason is not None and "Week 2" in result.reason, (
        f"SKIPPED reason on Poisson expected to mention Week 2 scope, got {result.reason!r}"
    )


def test_rvc_rule_skipped_on_non_laplace_non_poisson():
    """Rule only applies to laplace/poisson (ph_bc_002.py:31-46). Heat,
    wave, and other PDEs SKIP with the scope-guard reason.
    """
    result = _run_rule(16, pde="heat", diffusivity=1.0)
    assert result.status == "SKIPPED", (
        f"rule status on heat expected SKIPPED, got {result.status!r}"
    )
    assert result.reason is not None and "laplace/poisson only" in result.reason, (
        f"SKIPPED reason on heat expected scope-guard, got {result.reason!r}"
    )
