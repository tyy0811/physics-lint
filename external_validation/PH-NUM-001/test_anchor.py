"""PH-NUM-001 external-validation anchor - FEM quadrature exactness (V1 stub).

Task 11 of the complete-v1.0 plan. Applies the V1-stub CRITICAL three-layer
pattern (2026-04-24 `feedback_critical_rule_stub_three_layer_contract.md`,
Task 5 / Task 7 precedents): the production rule `ph_num_001.py` is a V1
structural stub that emits PASS with the reason "qorder convergence check
is a stub until V1.1" on any MeshField, and SKIPPED on non-MeshField
(ph_num_001.py:31-57). The rule does NOT compute a convergence rate, does
NOT compare quadrature at orders q and 2q, and does NOT measure any
variational crime in V1. V1.1 can plug in the real q-vs-2q check without
breaking the public API.

External validation separates:

    F1   Mathematical-legitimacy anchor: classical Gauss-Legendre exactness
         (an n-point rule integrates polynomials of degree <= 2n - 1
         exactly) + variational-crime framing (Strang 1972). Ciarlet 2002
         §4.1 + Strang 1972 + Brenner-Scott 2008 §10.3, all chapter-level
         per TEXTBOOK_AVAILABILITY.md WARN flag. Seven-step structural
         proof-sketch with explicit bilinear-form separation
         (a(u,v) = integral grad u . grad v dx vs a_h(u,v) = sum_K sum_q
         w_q grad u(x_q) . grad v(x_q)) and explicit "does NOT prove the
         full p+1 rate" disclaimer.

    F2   Harness-level correctness fixture (authoritative). Three scoped
         cases per 2026-04-24 user-revised contract:

         - Case A (exact): degree <= intorder -> error at float64 roundoff.
           5 parametrized 1D pairs + 2 2D product-monomial pairs.
         - Case B (under-integrated): degree > intorder with gap >= 3 ->
           error bounded away from 0 at >= 1e-6. 5 parametrized 1D pairs +
           1 2D product-monomial pair.
         - Case C (convergence): fix polynomial degree, sweep intorder.
           Error decreases monotonically and reaches roundoff once intorder
           matches the polynomial degree. At degree=10, intorders
           (2, 4, 6, 8, 10) give errors (4.3e-4, 2.9e-6, 2.8e-9, 1.3e-13,
           5.6e-17); drop factor 7.8e+12.

    F3   Borrowed-credibility: absent-with-justification per plan §19
         rationale + 2026-04-24 user-revised F3 contract ("Ciarlet /
         Strang / Brenner-Scott can support F1, not F3 reproduction").
         Task 0 F3-hunt confirmed no CI-executable reproduction target
         exists for the rule's emitted quantity (pass-through baseline
         integral). Ern-Guermond 2021 §8.3 + MOOSE FEM-convergence
         tutorial moved to Supplementary calibration context with
         pedagogical / methodology framing flags.

    RVC  Rule-verdict contract: exercises the rule's V1 PASS-with-reason
         behavior:
         - MeshField on any basis / any DOFs -> status="PASS",
           reason="qorder convergence check is a stub until V1.1",
           raw_value = field.integrate() (pass-through baseline).
         - GridField (non-mesh) -> status="SKIPPED",
           reason="PH-NUM-001 requires MeshField".
         Anchor does NOT assert the rule computes a convergence rate; it
         explicitly asserts the rule's V1 pass-through behavior.

Wording discipline (CITATION.md + README + tests):
    "PH-NUM-001 validates the mathematical and harness-level quadrature-
    error contract for controlled weak-form fixtures. The v1.0 production
    rule validates only its implemented diagnostic behavior."

Plan-diffs logged (plan-vs-committed-state drift, section 7.4):
    27. (Task 11) Plan §19 step 4 "MMS fixture with analytical solution;
        vary intorder in {1,2,3,4}; measure convergence rate against
        element order p in {1,2,3}; assert rate matches Ciarlet's
        theoretical prediction within 10%" -> simpler polynomial-
        exactness fixtures (Gauss-Legendre exactness / under-integration
        / convergence sweep) per 2026-04-24 user-revised Task 11
        contract ("simple polynomial integrals over a full FEM assembly
        first"). Full MMS h-refinement with p+1 rate deferred to V1.1
        when rule has qorder kwarg.
    28. (Task 11) CRITICAL three-layer pattern applied (Task 5 / Task 7
        precedents): rule-verdict contract verifies the V1-stub PASS-
        with-reason behavior on MeshField + SKIP on non-MeshField.
        Anchor docs explicitly state the rule does NOT compute a
        convergence rate or catch quadrature pathologies in V1.
    29. (Task 11) Plan §19 F3 already-absent status reinforced per 2026-
        04-24 user-revised F3 contract: Ciarlet / Strang / Brenner-
        Scott support F1, not F3 reproduction. Ern-Guermond 2021 §8.3 +
        MOOSE tutorial in Supplementary calibration context with
        pedagogical / methodology framing flags.

Plan-diffs 1-26 are from Tasks 2, 3, 4, 5, 7, 8, 9, 10, 12 (commits
30baf3e, 0cedc7b, 18312b9, 6800d6f, 1112da3, 26ed3bd, 84c7163, 87e8a3e,
ae1f9a9).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skfem")

from external_validation._harness.quadrature import (
    convergence_sweep_over_intorder,
    is_non_increasing,
    quadrature_error_monomial_1d,
    quadrature_error_product_monomial,
)
from physics_lint import DomainSpec
from physics_lint.field import GridField, MeshField
from physics_lint.rules import ph_num_001

# ---------------------------------------------------------------------------
# Acceptance bands (2026-04-24 precheck-calibrated per user's revised contract)
# ---------------------------------------------------------------------------

EXACT_ROUNDOFF_TOL = 1e-14
UNDER_INTEGRATED_FLOOR = 1e-6
CASE_C_DEGREE = 10
CASE_C_INTORDERS = (2, 4, 6, 8, 10)
CASE_C_FINAL_TOL = 1e-14
CASE_C_DROP_FACTOR_MIN = 1e6


SPEC = DomainSpec.model_validate(
    {
        "pde": "poisson",
        "grid_shape": [32, 32],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "field": {"type": "mesh", "backend": "fd", "adapter_path": "x"},
    }
)


# =========================================================================
# Case A: exact quadrature (degree <= intorder -> error at float64 roundoff).
# =========================================================================


@pytest.mark.parametrize("degree,intorder", [(0, 0), (1, 1), (3, 3), (5, 5), (7, 7)])
def test_case_a_1d_monomial_exact_at_matched_intorder(degree, intorder):
    """Gauss-Legendre quadrature integrates x**degree exactly when
    intorder >= degree. Error sits at float64 roundoff (~5e-16).
    """
    err = quadrature_error_monomial_1d(degree, intorder=intorder)
    assert err <= EXACT_ROUNDOFF_TOL, (
        f"Case A exact: x^{degree} at intorder={intorder} should integrate "
        f"to roundoff; got error {err!r} (tolerance {EXACT_ROUNDOFF_TOL:.0e})"
    )


@pytest.mark.parametrize("dx,dy,intorder", [(2, 3, 5), (3, 3, 6)])
def test_case_a_2d_product_monomial_exact_at_sufficient_intorder(dx, dy, intorder):
    """2D product monomial x^dx * y^dy integrates exactly when intorder
    covers the total polynomial degree dx + dy (scikit-fem rule
    exactness is at least intorder).
    """
    err = quadrature_error_product_monomial(dx, dy, intorder=intorder)
    assert err <= EXACT_ROUNDOFF_TOL, (
        f"Case A 2D: x^{dx}*y^{dy} at intorder={intorder} should integrate "
        f"to roundoff; got error {err!r} (tolerance {EXACT_ROUNDOFF_TOL:.0e})"
    )


# =========================================================================
# Case B: under-integrated quadrature (degree > intorder -> nonzero error).
#
# Required by 2026-04-24 user-revised Task 11 F2 contract: "include under-
# integrated high-degree case." Gap >= 3 between degree and intorder keeps
# the error safely above the numerical floor even on moderately refined
# meshes.
# =========================================================================


@pytest.mark.parametrize(
    "degree,intorder",
    [(4, 1), (6, 2), (8, 3), (10, 4), (12, 5)],
)
def test_case_b_1d_under_integrated_has_bounded_nonzero_error(degree, intorder):
    """Under-integration with gap >= 3 produces error bounded away from 0.
    Measured floor across these pairs is 2.9e-6; assertion uses 1e-6 for
    safety margin.
    """
    err = quadrature_error_monomial_1d(degree, intorder=intorder)
    assert err >= UNDER_INTEGRATED_FLOOR, (
        f"Case B under-integrated: x^{degree} at intorder={intorder} "
        f"(gap={degree - intorder}) should produce error >= "
        f"{UNDER_INTEGRATED_FLOOR:.0e}; got {err!r}"
    )


def test_case_b_2d_product_monomial_under_integrated():
    """2D under-integration cross-check: x^5 * y^5 (total degree 10) at
    intorder=3 (gap 7) gives error ~1e-4, well above the floor.
    """
    err = quadrature_error_product_monomial(5, 5, intorder=3)
    assert err >= UNDER_INTEGRATED_FLOOR, (
        f"Case B 2D under-integrated: x^5 * y^5 at intorder=3 should give "
        f"error >= {UNDER_INTEGRATED_FLOOR:.0e}; got {err!r}"
    )


# =========================================================================
# Case C: convergence (increase intorder -> error decreases to roundoff).
# =========================================================================


def test_case_c_convergence_reaches_roundoff_at_sufficient_intorder():
    """At polynomial degree 10, sweeping intorder in {2, 4, 6, 8, 10}: the
    final error (intorder=10) must hit float64 roundoff -- the quadrature
    is exact once intorder matches the polynomial degree.
    """
    errs = convergence_sweep_over_intorder(CASE_C_DEGREE, intorders=CASE_C_INTORDERS)
    final = float(errs[-1])
    assert final <= CASE_C_FINAL_TOL, (
        f"Case C convergence: final error at intorder={CASE_C_INTORDERS[-1]} "
        f"(= polynomial degree {CASE_C_DEGREE}) should be <= "
        f"{CASE_C_FINAL_TOL:.0e}; got {final!r}. Full sweep: "
        f"{[float(e) for e in errs]!r}"
    )


def test_case_c_convergence_is_monotonically_non_increasing():
    """The error sequence across ascending intorder must be non-increasing.
    Strict monotonicity is NOT required -- scikit-fem may round adjacent
    intorder values up to the same quadrature rule, giving identical errors.
    But later intorder MUST NOT yield a larger error.
    """
    errs = convergence_sweep_over_intorder(CASE_C_DEGREE, intorders=CASE_C_INTORDERS)
    assert is_non_increasing(errs, slack=1e-15), (
        f"Case C convergence: error sequence is not non-increasing across "
        f"intorders={CASE_C_INTORDERS}: errors = {[float(e) for e in errs]!r}"
    )


def test_case_c_convergence_drop_factor_is_large():
    """Ratio of first (under-integrated) to last (exact) error must exceed
    1e6. Measured 2026-04-24: 7.8e+12 at degree=10. This demonstrates
    that increasing quadrature order progressively eliminates the
    variational-crime regime.
    """
    errs = convergence_sweep_over_intorder(CASE_C_DEGREE, intorders=CASE_C_INTORDERS)
    first = float(errs[0])
    last = float(errs[-1])
    assert last > 0.0, f"last error must be positive for ratio; got {last!r}"
    drop_factor = first / last
    assert drop_factor >= CASE_C_DROP_FACTOR_MIN, (
        f"Case C drop factor: errs[0]={first!r} / errs[-1]={last!r} = "
        f"{drop_factor:.2e} < {CASE_C_DROP_FACTOR_MIN:.0e}"
    )


# =========================================================================
# Rule-verdict contract: V1-stub PASS on MeshField + SKIP on non-MeshField.
# =========================================================================


def _build_meshfield_smooth(n_refine: int = 2) -> MeshField:
    """Smooth MeshField on the unit square for rule-verdict testing. The
    DOFs are sin(pi x) sin(pi y) sampled at the P2 nodes -- arbitrary
    smooth field; the rule only computes field.integrate() as the
    pass-through raw_value.
    """
    from skfem import Basis, ElementTriP2, MeshTri

    mesh = MeshTri.init_sqsymmetric().refined(n_refine)
    basis = Basis(mesh, ElementTriP2())
    x = basis.doflocs[0]
    y = basis.doflocs[1]
    dofs = np.sin(np.pi * x) * np.sin(np.pi * y)
    return MeshField(basis=basis, dofs=dofs)


def test_rvc_rule_passes_on_meshfield_with_stub_reason():
    """Rule returns PASS with the V1.1-stub reason on any MeshField. The
    anchor does NOT assert the rule computes a convergence rate -- it
    asserts the rule's V1 pass-through behavior matches the docstring
    contract at ph_num_001.py:31-57.
    """
    field = _build_meshfield_smooth()
    result = ph_num_001.check(field, SPEC)
    assert result.status == "PASS", (
        f"Rule-verdict: MeshField expected PASS, got {result.status!r} (reason={result.reason!r})"
    )
    assert result.reason is not None
    assert "qorder convergence check is a stub until V1.1" in result.reason, (
        f"Rule-verdict: expected V1.1-stub reason substring; got reason={result.reason!r}"
    )


def test_rvc_rule_raw_value_equals_baseline_integral():
    """Rule's raw_value is a pass-through field.integrate() (baseline
    integral over the mesh domain). Verify numerical equality -- any
    future V1.x change to the rule's raw_value semantic must update this
    anchor in the same commit.
    """
    field = _build_meshfield_smooth()
    result = ph_num_001.check(field, SPEC)
    expected_baseline = field.integrate()
    assert result.raw_value is not None
    assert abs(result.raw_value - expected_baseline) < 1e-14, (
        f"Rule-verdict: raw_value {result.raw_value!r} should equal "
        f"field.integrate() = {expected_baseline!r}"
    )


def test_rvc_rule_skips_on_gridfield():
    """Rule SKIPs when given a GridField (not a MeshField). The reason
    string must point at the MeshField requirement (ph_num_001.py:34-39).
    """
    field = GridField(
        np.zeros((16, 16)),
        h=(1 / 15, 1 / 15),
        periodic=False,
        backend="fd",
    )
    result = ph_num_001.check(field, SPEC)
    assert result.status == "SKIPPED", (
        f"Rule-verdict: GridField expected SKIPPED, got {result.status!r}"
    )
    assert result.reason is not None
    assert "PH-NUM-001 requires MeshField" in result.reason, (
        f"Rule-verdict: expected MeshField-required reason substring; got reason={result.reason!r}"
    )


def test_rvc_rule_has_no_convergence_rate_in_v1():
    """V1 contract: rule does NOT emit a convergence rate or a refinement
    rate. Both fields in the result must be None -- if a future V1.1
    change wires a qorder sweep in, this test must be updated in the
    same commit.
    """
    field = _build_meshfield_smooth()
    result = ph_num_001.check(field, SPEC)
    assert result.refinement_rate is None, (
        f"V1 rule should not emit refinement_rate; got {result.refinement_rate!r}"
    )
    assert result.violation_ratio is None, (
        f"V1 rule should not emit violation_ratio; got {result.violation_ratio!r}"
    )
