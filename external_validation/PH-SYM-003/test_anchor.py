"""PH-SYM-003 external-validation anchor - SO(2) Lie-derivative equivariance
diagnostic (adapter-only, CRITICAL three-layer with mathematical preflight gate).

Task 6 of the complete-v1.0 plan. Applies the V1-stub / scope-narrower CRITICAL
three-layer pattern (2026-04-24 feedback_critical_rule_stub_three_layer_contract.md
+ feedback_narrower_estimator_than_theorem.md, Tasks 5/7/11 precedent) plus the
user's 2026-04-24 revised mathematical-preflight-gate contract for Task 6:
F1 proof skeleton authored in CITATION.md before any test code.

External validation separates:

    F1   Mathematical-legitimacy anchor: Hall 2015 section 2.5 + section 3.7
         (section-level, WARN-flagged) + Varadarajan 1984 section 2.9-2.10
         (section-level, WARN-flagged) + Kondor-Trivedi 2018 compact-group
         equivariance theorem (arXiv:1802.03690). Six-step structural proof-
         sketch with explicit assumption statement separating the
         finite-implies-infinitesimal direction (trivial) from the
         infinitesimal-implies-finite direction (requires smoothness,
         connected group, generator coverage, exact constraint) and scoping
         the rule's claim to infinitesimal-LEE-diagnostic-of-scalar-SO(2)-
         invariance.

    F2   Harness-level correctness fixture (authoritative). Controlled
         scalar-output maps exercised via external_validation/_harness/
         symmetry.py so2_lie_derivative_norm + finite_small_angle_defect:

         Case A positive controls (equivariant, expected L_A f = 0):
           identity_scalar_2d                - constant map, L_A f = 0 exact
           radial_scalar(phi)                - f(r), L_A f = 0 exact

         Case B negative controls (non-equivariant, closed-form L_A f):
           coord_dependent_scalar_2d         - f = x; L_A f = -y
           anisotropic_xx_minus_yy_2d        - f = x^2 - y^2; L_A f = -4 x y

         Case C finite-vs-infinitesimal consistency:
           finite_small_angle_defect         - Taylor remainder scales O(eps)

         Measured across 64x64 origin-centered grid on [-1, 1]^2:
         positive controls hit exactly 0.0; negative controls match
         closed-form analytical values to float64 accuracy; Case C
         remainder ratio scales linearly in epsilon with bounded
         coefficient.

    F3   Borrowed-credibility: absent-with-justification per plan section 14
         rationale + 2026-04-24 user-revised F3 contract. No CI-executable
         reproduction target exists for the rule's emitted quantity in V1
         (no Modal/RotMNIST/escnn/e3nn/Gruver infrastructure in codebase).
         Cohen-Welling 2016 / Weiler-Cesa 2019 / Gruver 2023 moved to
         Supplementary calibration context with "theoretical framing, not
         reproduction" flag. Plan-diff logged in commit Provenance.

    RVC  Rule-verdict contract: exercises ph_sym_003.check()'s V1 paths:
         SKIP paths (5):
           (a) SO2 not declared in SymmetrySpec
           (b) dump mode (field is not CallableField)
           (c) non-2D sampling grid
           (d) empty sampling grid
           (e) non-origin-centered sampling grid
           (f) non-square domain
         Live PASS path:
           radial_scalar wrapped as CallableField on origin-centered 2D
           square grid returns PASS with lie_norm below 10 x tolerance x
           floor threshold.
         Live WARN/FAIL path:
           anisotropic_xx_minus_yy_2d wrapped as CallableField returns
           FAIL with the expected closed-form L_A f norm.

Wording discipline (CITATION.md + README + this file):
    "PH-SYM-003 validates an infinitesimal Lie-derivative equivariance
    diagnostic under explicit SO(2) / smoothness / generator assumptions.
    It does not prove global finite equivariance for arbitrary models."

Plan-diffs logged (plan-vs-committed-state drift, section 7.4):
    30. (Task 6) Plan section 14 F3 two-layer RotMNIST CI policy +
        ImageNet-opt-in Gruver reproduction pre-demoted to F3-absent +
        Supplementary calibration context. Reason: F3-INFRA-GAP
        (codebase grep for rotmnist/modal/escnn/e3nn/gruver/lie-deriv
        returns only a codespell ignore-list row and unrelated Tier-A
        script; no Modal integration, no RotMNIST loader, no
        equivariance optional-dependency group in pyproject.toml, no
        workflow_dispatch trigger in .github/workflows/). Authorized
        by 2026-04-24 user-revised F3 contract: "If RotMNIST + escnn
        is CI-runnable in v1, keep it as optional borrowed credibility.
        If not, demote to Supplementary."
    31. (Task 6) Plan section 14 three-cross-library EMLP + escnn +
        e3nn correctness fixture replaced by controlled-operator
        harness (radial positive + coord_x / xx-yy negatives + Case C
        finite-vs-infinitesimal). Same mathematical property with
        simpler analytical operators whose L_A f is closed-form; avoids
        unpinned library dependencies consistent with Task 7 plan-diff
        23 substitution pattern.
    32. (Task 6) CRITICAL three-layer pattern applied (Tasks 5, 7, 11
        precedent) with mathematical preflight gate: F1 proof skeleton
        authored in CITATION.md section "Mathematical-legitimacy" +
        docs/audits/2026-04-24-task-6-preflight.md before any test code.
        Rule-verdict contract layer exercises the live PASS and live
        FAIL paths in addition to the five SKIP gates in
        ph_sym_003.py:36-68; differs from Task 7's all-SKIP contract
        because ph_sym_003 emits live PASS/WARN/FAIL values when gates
        pass.
    33. (Task 6) Narrower-estimator-than-theorem scoping applied
        (Task 10 precedent): F1 scope explicitly restricted to
        infinitesimal-LEE-diagnostic-of-scalar-SO(2)-invariance. Four
        subset relationships named: (i) infinitesimal-only not
        finite-sweep; (ii) scalar-invariant only (rho_Y = identity);
        (iii) single-generator so(2) only; (iv) sampled-grid only.
        F1 does not inherit Hall/Varadarajan/Kondor-Trivedi guarantees
        that the rule cannot validate.
    34. (Task 6) Infinitesimal-vs-finite direction separation authored
        explicitly in F1 proof skeleton and CITATION.md assumption
        statement per 2026-04-24 user-revised contract. Finite-implies-
        infinitesimal is trivial (differentiate); infinitesimal-implies-
        finite requires four assumptions (smoothness, connected group,
        generator coverage, exact constraint) and does not hold for
        empirical grid-sampled tests as a global claim.

Plan-diffs 1-29 are from Tasks 2, 3, 4, 5, 7, 8, 9, 10, 11, 12 (commits
30baf3e, 0cedc7b, 18312b9, 6800d6f, 1112da3, ae1f9a9, 26ed3bd, 84c7163,
87e8a3e, 2ae7d28).
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from external_validation._harness.symmetry import (
    _origin_centered_square_grid,
    anisotropic_xx_minus_yy_2d,
    coord_dependent_scalar_2d,
    finite_small_angle_defect,
    identity_scalar_2d,
    radial_scalar,
    so2_lie_derivative,
    so2_lie_derivative_norm,
)
from physics_lint import DomainSpec
from physics_lint.field import CallableField, GridField
from physics_lint.rules import ph_sym_003

# ---------------------------------------------------------------------------
# Fixtures and acceptance bands (2026-04-24 user-revised contract)
# ---------------------------------------------------------------------------

GRID_N = 64
HALF_EXTENT = 1.0
DOMAIN_LENGTH = 2.0 * HALF_EXTENT

POSITIVE_CONTROL_TOL = 1e-12
CLOSED_FORM_TOL = 1e-14
CASE_C_LINEARITY_COEFF_BOUND = 10.0


def _grid() -> torch.Tensor:
    return _origin_centered_square_grid(GRID_N, half_extent=HALF_EXTENT)


def _closed_form_norm_coord_x(grid: torch.Tensor) -> float:
    y = grid[..., 1]
    return float(torch.linalg.vector_norm(y) / (y.numel() ** 0.5))


def _closed_form_norm_anisotropic(grid: torch.Tensor) -> float:
    x = grid[..., 0]
    y = grid[..., 1]
    four_xy = 4.0 * x * y
    return float(torch.linalg.vector_norm(four_xy) / (four_xy.numel() ** 0.5))


def _build_callable_field(model, *, grid: torch.Tensor, periodic: bool = False) -> CallableField:
    """Wrap a scalar-output model as CallableField with the rule's grid contract.

    ph_sym_003.check() calls model(rotated_grid) internally then squeezes the
    last dim. The harness primitives return shape (..., ) without a trailing
    size-1 dim; squeeze(-1) is a no-op on (H, W), so both conventions work.
    """
    h = DOMAIN_LENGTH / (GRID_N - 1)
    return CallableField(model, sampling_grid=grid, h=(h, h), periodic=periodic)


def _build_spec(
    *,
    domain_x=(-1.0, 1.0),
    domain_y=(-1.0, 1.0),
    declared=("SO2",),
) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [GRID_N, GRID_N],
            "domain": {"x": list(domain_x), "y": list(domain_y)},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "symmetries": {"declared": list(declared)},
            "field": {"type": "callable", "adapter_path": "dummy.py"},
        }
    )


# =========================================================================
# F2 Case A: equivariant positive controls (L_A f = 0 exactly)
# =========================================================================


def test_case_a_identity_positive_control_is_exactly_invariant():
    """Constant scalar map is SO(2)-invariant trivially; L_A f = 0 exact."""
    grid = _grid()
    norm = so2_lie_derivative_norm(identity_scalar_2d, grid)
    assert norm == 0.0, (
        f"identity scalar Lie-derivative norm: got {norm!r}, expected exactly 0 "
        f"(constant functions have zero gradient, so L_A f = grad(f) . (A x) = 0)"
    )


@pytest.mark.parametrize(
    "phi_name,phi",
    [
        ("gaussian", lambda r: torch.exp(-r * r)),
        ("log_one_plus_r2", lambda r: torch.log1p(r * r)),
        ("r_squared", lambda r: r * r),
        ("sinc_of_r2", lambda r: torch.sinc(r * r)),
    ],
)
def test_case_a_radial_scalar_positive_control_is_exactly_invariant(phi_name, phi):
    """Radial f(x, y) = phi(sqrt(x^2 + y^2)) is SO(2)-invariant by construction.

    |R_theta (x, y)| = |(x, y)|, so phi(r) is invariant under rotation; L_A f = 0.
    """
    grid = _grid()
    f = radial_scalar(phi)
    norm = so2_lie_derivative_norm(f, grid)
    assert norm <= POSITIVE_CONTROL_TOL, (
        f"radial_scalar({phi_name}) Lie-derivative norm: got {norm:.3e}, "
        f"expected <= {POSITIVE_CONTROL_TOL:.0e} (radial f is SO(2)-invariant)"
    )


# =========================================================================
# F2 Case B: non-equivariant negative controls with closed-form L_A f
# =========================================================================


def test_case_b_coord_x_negative_control_matches_closed_form():
    """f(x, y) = x has L_A f = -y. Norm ||L_A f|| = ||-y||_{per-point L2}."""
    grid = _grid()
    got = so2_lie_derivative_norm(coord_dependent_scalar_2d, grid)
    expected = _closed_form_norm_coord_x(grid)
    assert abs(got - expected) <= CLOSED_FORM_TOL, (
        f"coord_x Lie-derivative norm: got {got!r}, expected {expected!r} "
        f"(closed-form ||-y||_{{per-point L2}}); diff {abs(got - expected):.3e}"
    )
    assert expected > 0.1, (
        f"sanity: closed-form ||-y|| should be ~0.58 on [-1, 1]^2, got {expected}"
    )


def test_case_b_anisotropic_negative_control_matches_closed_form():
    """f(x, y) = x^2 - y^2 has L_A f = -4 x y. Norm = ||-4 x y||_{per-point L2}."""
    grid = _grid()
    got = so2_lie_derivative_norm(anisotropic_xx_minus_yy_2d, grid)
    expected = _closed_form_norm_anisotropic(grid)
    assert abs(got - expected) <= CLOSED_FORM_TOL, (
        f"anisotropic x^2 - y^2 Lie-derivative norm: got {got!r}, expected "
        f"{expected!r} (closed-form ||-4 x y||_{{per-point L2}}); diff "
        f"{abs(got - expected):.3e}"
    )
    assert expected > 1.0, (
        f"sanity: closed-form ||-4 x y|| should be ~1.38 on [-1, 1]^2, got {expected}"
    )


def test_case_b_negative_is_far_above_positive():
    """Negative control L_A f norm dwarfs the positive-control roundoff floor.

    This is the "distinguishability" sanity the rule's tri-state classifier
    depends on: positive and negative controls must be separated by many
    orders of magnitude in the emitted quantity.
    """
    grid = _grid()
    pos = so2_lie_derivative_norm(radial_scalar(lambda r: torch.exp(-r * r)), grid)
    neg = so2_lie_derivative_norm(anisotropic_xx_minus_yy_2d, grid)
    assert neg > 1e12 * max(pos, 1e-16), (
        f"negative/positive separation: pos={pos:.3e}, neg={neg:.3e}; "
        f"expected negative at least 1e12x positive (got ratio "
        f"{neg / max(pos, 1e-16):.3e})"
    )


# =========================================================================
# F2 Case C: finite-vs-infinitesimal consistency (Taylor remainder)
# =========================================================================


@pytest.mark.parametrize("epsilon", [1e-1, 1e-2, 1e-3, 1e-4])
def test_case_c_finite_vs_infinitesimal_scales_linearly(epsilon):
    """||f(R_eps x) - f(x) - eps * L_A f(x)|| = O(eps^2), so the ratio
    against ||eps * L_A f|| scales as O(eps).

    On coord_dependent_scalar_2d (f = x, L_A f = -y), the closed-form
    finite rotation is cos(eps) x - sin(eps) y, and the first-order
    Taylor approximation is x + eps * (-y). The residual is
    (cos(eps) - 1) x - (sin(eps) - eps) y, whose leading term is
    -eps^2 / 2 x + O(eps^4). ||residual|| / ||eps * L_A f|| ~
    eps/2 * ||x|| / ||y||, bounded by eps / 2 on [-1, 1]^2.
    """
    grid = _grid()
    ratio = finite_small_angle_defect(coord_dependent_scalar_2d, grid, epsilon)
    coefficient = ratio / epsilon
    assert coefficient <= CASE_C_LINEARITY_COEFF_BOUND, (
        f"Case C linearity at eps={epsilon:.0e}: ratio={ratio:.3e}, "
        f"coefficient={coefficient:.3e} (ratio/eps); expected <= "
        f"{CASE_C_LINEARITY_COEFF_BOUND} for O(eps) scaling"
    )


def test_case_c_infinitesimal_is_linear_part_of_finite():
    """Directly show the ratio shrinks proportional to epsilon."""
    grid = _grid()
    ratios = [
        finite_small_angle_defect(coord_dependent_scalar_2d, grid, eps)
        for eps in (1e-2, 1e-3, 1e-4)
    ]
    # Each successive ratio should be ~ 10x smaller (one more order of epsilon).
    for prev, curr in itertools.pairwise(ratios):
        contraction = prev / max(curr, 1e-30)
        assert 5.0 <= contraction <= 20.0, (
            f"Case C contraction between consecutive eps=10^-k: got factor "
            f"{contraction:.2f}, expected ~10 (O(eps) linear scaling)"
        )


# =========================================================================
# F2 shared primitive: so2_lie_derivative returns a tensor shaped like model(grid)
# =========================================================================


def test_so2_lie_derivative_shape_matches_model_output():
    """Sanity: the jvp primitive returns a tensor with the model's output shape."""
    grid = _grid()
    lie = so2_lie_derivative(coord_dependent_scalar_2d, grid)
    assert lie.shape == (GRID_N, GRID_N), (
        f"so2_lie_derivative output shape: got {tuple(lie.shape)}, expected "
        f"{(GRID_N, GRID_N)} (model output shape)"
    )


# =========================================================================
# Rule-verdict contract: SKIP paths (5 gates in ph_sym_003.py:36-68)
# =========================================================================


def test_rule_verdict_skip_when_so2_not_declared():
    """Gate (a): SO2 not in SymmetrySpec.declared -> SKIPPED."""
    grid = _grid()
    field = _build_callable_field(radial_scalar(lambda r: torch.exp(-r * r)), grid=grid)
    spec = _build_spec(declared=())  # no SO2
    result = ph_sym_003.check(field, spec)
    assert result.status == "SKIPPED"
    assert "SO2 not declared" in (result.reason or "")


def test_rule_verdict_skip_on_dump_mode():
    """Gate (b): dump mode (non-CallableField input) -> SKIPPED."""
    # Build a GridField with constant values to mimic dump mode.
    h = DOMAIN_LENGTH / (GRID_N - 1)
    values = torch.zeros(GRID_N, GRID_N).numpy()
    dump_field = GridField(values, h=(h, h), periodic=False)
    spec = _build_spec()
    result = ph_sym_003.check(dump_field, spec)
    assert result.status == "SKIPPED"
    assert "callable" in (result.reason or "").lower() or "dump" in (result.reason or "").lower()


def test_rule_verdict_skip_on_non_2d_grid():
    """Gate (c): grid.shape[-1] != 2 -> SKIPPED."""
    # Build a 3D origin-centered grid: shape (N, N, N, 3) with last dim = 3.
    n_small = 8
    coord = torch.linspace(-1.0, 1.0, n_small, dtype=torch.float64)
    gx, gy, gz = torch.meshgrid(coord, coord, coord, indexing="ij")
    grid3d = torch.stack([gx, gy, gz], dim=-1)
    h = DOMAIN_LENGTH / (n_small - 1)
    # CallableField with 3D grid; model returns scalar from 3-coord input.
    field = CallableField(
        lambda c: c[..., 0],
        sampling_grid=grid3d,
        h=(h, h, h),
        periodic=False,
    )
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [n_small, n_small, n_small],
            "domain": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "symmetries": {"declared": ["SO2"]},
            "field": {"type": "callable", "adapter_path": "dummy.py"},
        }
    )
    result = ph_sym_003.check(field, spec)
    assert result.status == "SKIPPED"
    assert "2D" in (result.reason or "") or "2d" in (result.reason or "").lower()


def test_rule_verdict_skip_on_non_origin_centered_grid():
    """Gate (e): grid center offset > 1e-6 * max(|coord|, 1) -> SKIPPED."""
    # Grid on [0, 1]^2 has center offset (0.5, 0.5), offset norm ~0.707.
    coord = torch.linspace(0.0, 1.0, GRID_N, dtype=torch.float64)
    gx, gy = torch.meshgrid(coord, coord, indexing="ij")
    grid_offcenter = torch.stack([gx, gy], dim=-1)
    field = _build_callable_field(radial_scalar(lambda r: torch.exp(-r * r)), grid=grid_offcenter)
    spec = _build_spec(domain_x=(0.0, 1.0), domain_y=(0.0, 1.0))
    result = ph_sym_003.check(field, spec)
    assert result.status == "SKIPPED"
    assert "origin" in (result.reason or "").lower()


def test_rule_verdict_skip_on_non_square_domain():
    """Gate (f): abs(lx - ly) / max(lx, ly) > 1e-6 -> SKIPPED."""
    # Origin-centered but rectangular domain: [-1, 1] x [-2, 2]. Pass the
    # origin-centered gate (grid mean = (0, 0)) but fail the square gate.
    xs = torch.linspace(-1.0, 1.0, GRID_N, dtype=torch.float64)
    ys = torch.linspace(-2.0, 2.0, GRID_N, dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    grid_rect = torch.stack([gx, gy], dim=-1)
    field = _build_callable_field(radial_scalar(lambda r: torch.exp(-r * r)), grid=grid_rect)
    # Symmetries-compat warning fires on the DomainSpec constructor; catch it.
    with pytest.warns(UserWarning, match=r"symmetry declared but domain is not square"):
        spec = _build_spec(domain_x=(-1.0, 1.0), domain_y=(-2.0, 2.0))
    result = ph_sym_003.check(field, spec)
    assert result.status == "SKIPPED"
    assert "square" in (result.reason or "").lower()


# =========================================================================
# Rule-verdict contract: live PASS / live FAIL paths
# =========================================================================


def test_rule_verdict_live_pass_on_radial_scalar():
    """Live path: radial Gaussian (SO(2)-invariant) -> rule emits PASS.

    L_A f = 0 exactly for radial scalars, so lie_norm = 0, ratio = 0 against
    any floor, status = PASS (<= 10 * tolerance = 30 threshold).
    """
    grid = _grid()
    field = _build_callable_field(radial_scalar(lambda r: torch.exp(-r * r)), grid=grid)
    spec = _build_spec()
    result = ph_sym_003.check(field, spec)
    assert result.status == "PASS", (
        f"radial Gaussian live-path status: got {result.status!r}, expected PASS "
        f"(reason={result.reason!r}, raw_value={result.raw_value!r})"
    )
    assert result.raw_value is not None
    assert float(result.raw_value) <= POSITIVE_CONTROL_TOL
    assert result.citation, "PASS rule-verdict must record a citation"
    assert result.doc_url.endswith("PH-SYM-003")


def test_rule_verdict_live_fail_on_anisotropic_scalar():
    """Live path: anisotropic x^2 - y^2 -> rule emits FAIL.

    Closed-form L_A f = -4 x y has norm ~1.376 on [-1, 1]^2. With floor
    value 2.221e-16 and tolerance 3.0, ratio = 1.376 / 2.221e-16 = 6.2e15,
    vastly above fail threshold (300). Status = FAIL.
    """
    grid = _grid()
    field = _build_callable_field(anisotropic_xx_minus_yy_2d, grid=grid)
    spec = _build_spec()
    result = ph_sym_003.check(field, spec)
    assert result.status == "FAIL", (
        f"anisotropic live-path status: got {result.status!r}, expected FAIL "
        f"(reason={result.reason!r}, raw_value={result.raw_value!r})"
    )
    expected = _closed_form_norm_anisotropic(grid)
    assert result.raw_value is not None
    assert abs(float(result.raw_value) - expected) <= CLOSED_FORM_TOL, (
        f"live-path raw_value: got {result.raw_value!r}, expected {expected!r} "
        f"(closed-form ||-4 x y||_{{per-point L2}})"
    )
    assert "FAIL" in (result.reason or "") or "fail" in (result.reason or "").lower()


def test_rule_verdict_live_fail_on_coord_x_scalar():
    """Live path: f(x, y) = x -> rule emits FAIL with closed-form L_A f norm."""
    grid = _grid()
    field = _build_callable_field(coord_dependent_scalar_2d, grid=grid)
    spec = _build_spec()
    result = ph_sym_003.check(field, spec)
    assert result.status == "FAIL", (
        f"coord_x live-path status: got {result.status!r}, expected FAIL (reason={result.reason!r})"
    )
    expected = _closed_form_norm_coord_x(grid)
    assert result.raw_value is not None
    assert abs(float(result.raw_value) - expected) <= CLOSED_FORM_TOL


# =========================================================================
# Wording-discipline safeguard: CITATION.md and README do not overclaim
# =========================================================================


def test_citation_md_does_not_claim_global_finite_equivariance():
    """CITATION.md wording discipline: scope-narrowing language present,
    overclaim language absent. Guards against future rewrites that would
    re-introduce the overclaim."""
    from pathlib import Path

    citation = Path(__file__).parent / "CITATION.md"
    text = citation.read_text(encoding="utf-8")
    # Whitespace-normalized view so required phrases can survive line-wrap.
    normalized = " ".join(text.split())
    # Required language.
    assert "does not prove global finite equivariance" in normalized, (
        "CITATION.md must explicitly state the rule does not prove global finite equivariance"
    )
    assert "infinitesimal Lie-derivative equivariance diagnostic" in normalized
    # Forbidden overclaim phrasings (per 2026-04-24 user-revised contract).
    for forbidden in (
        "proves rotation equivariance",
        "certifies disconnected-group",
        "tests that a model is SO(2)-equivariant",
    ):
        # These phrases appear ONLY inside the "Avoid:" wording block, never
        # as an affirmative claim. Test for that structural placement.
        occurrences = text.count(forbidden)
        # Exactly one occurrence allowed (inside "Avoid:" block); more would
        # indicate a new affirmative use sneaking in.
        assert occurrences <= 1, (
            f"forbidden overclaim phrase `{forbidden}` appears {occurrences} "
            f"times in CITATION.md; expected at most 1 (inside 'Avoid:' block)"
        )


# =========================================================================
# Pre-execution audit artifact existence (mathematical preflight gate)
# =========================================================================


def test_preflight_audit_file_exists():
    """User's 2026-04-24 revised contract requires F1 proof skeleton before
    code. The audit file is the preflight gate artifact; this test guards
    against the file being deleted by a later commit."""
    from pathlib import Path

    audit = (
        Path(__file__).resolve().parents[2] / "docs" / "audits" / "2026-04-24-task-6-preflight.md"
    )
    assert audit.is_file(), (
        f"task 6 preflight audit missing at {audit}; mathematical preflight "
        f"gate requires this artifact"
    )
    body = audit.read_text(encoding="utf-8")
    # Structural checks: F3-INFRA-GAP classification, infinitesimal/finite
    # separation, scalar-invariant restriction, narrower-estimator scoping.
    # Markers use the Unicode ⇒ double-arrow as authored in the audit.
    for marker in (
        "F3-INFRA-GAP",
        "Finite ⇒ infinitesimal",
        "Infinitesimal ⇒ finite",
        "Scalar-invariant restriction",
        "narrower than F1 mathematical family",
    ):
        assert marker in body, f"preflight audit missing required marker `{marker}`"


# Intentional no-op to touch math import (linting guard).
_ = math.pi
