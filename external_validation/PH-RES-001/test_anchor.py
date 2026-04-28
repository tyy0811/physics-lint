"""PH-RES-001 external-validation anchor - Layer 1 Fornberg + Layer 2 norm characterization.

Rev 1.7.2 (Path A'): Task 4 characterizes both PH-RES-001 code paths across
both spatial-discretization and norm-selection dimensions.

- **Layer 1a (Fornberg interior O(h^4)):** the rule's FD4 Laplacian uses
  Fornberg 1988 coefficients (-1, 16, -30, 16, -1)/12 on the interior
  5-point stencil. MMS residual on the interior band [2:-2, 2:-2]
  converges at the expected O(h^4) rate at N in {16, 32, 64, 128}.
- **Layer 1b (full-domain characterization, non-periodic+FD):** the full-
  domain L^2 residual is boundary-dominated because the outer two layers
  use one-sided/off-center stencils with documented O(h^2) truncation
  error (grid.py:22-31). Empirical slope ~3.5 — positive characterization
  of the boundary-influenced full-domain rate, not a Fornberg
  reproduction. The reproduction lives in Layer 1a.
- **Layer 2a (BDO norm-equivalence, periodic+spectral H^-1):** the rule's
  variationally-correct path satisfies ||r||_{H^-1} proportional to
  ||u_pert - u_exact||_{H^1} within a bounded ratio (C_max/c_min < 10).
- **Layer 2b (L^2-fallback characterization, non-periodic+FD):** the rule's
  L^2 fallback path (per its module docstring at ph_res_001.py:14-20)
  produces rho that scales linearly with perturbation wavenumber, NOT as a
  bounded ratio. Tests assert the k-linear scaling; this is a positive
  characterization of the L^2 path's known non-equivalence, not a
  softening of the BDO claim.

The Layer 1a/1b + Layer 2a/2b symmetry is intentional: each layer pairs an
external-reference reproduction on the rule's clean regime with a positive
characterization of the rule's documented fallback regime. Both layers
surface the rule's two-regime structure honestly rather than pinning a
single number that would only hold on one regime.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from external_validation._harness.assertions import assert_slope_in_range
from external_validation._harness.mms import mms_perturbation_h1_error
from physics_lint import DomainSpec, GridField
from physics_lint.analytical.poisson import periodic_sin_sin, sin_sin_mms_square
from physics_lint.rules import ph_res_001

_FIXTURES = Path(__file__).parent / "fixtures" / "norm_equivalence_bounds.json"


def _nonperiodic_spec_with_source(n: int, source_array: np.ndarray) -> DomainSpec:
    spec = DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [n, n],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    object.__setattr__(spec, "_source_array", source_array)
    return spec


# =========================================================================
# Layer 1: Fornberg interior O(h^4) + full-domain characterization
# =========================================================================

LAYER_1_NS = [16, 32, 64, 128]


def _mms_residuals_at(n: int) -> tuple[float, float]:
    """Return (interior L^2, full-domain L^2) residual norms on MMS at grid N.

    Interior norm: L^2 of the residual on the [2:-2, 2:-2] sub-domain where
    the FD4 central stencil (Fornberg 1988, coefficients (-1, 16, -30, 16,
    -1)/12) applies uniformly. Pure interior convergence rate is O(h^4).

    Full-domain norm: the rule's raw_value (L^2 over [0, 1]^2 with half-
    weighted boundary trapezoidal weights). Includes boundary contributions
    from the one-sided/off-center O(h^2) stencils at the outer two layers
    (per grid.py:22-31) and is therefore not a Fornberg reproduction.
    """
    sol = sin_sin_mms_square()
    h = 1.0 / (n - 1)
    xs = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    u = sol.u(mesh_x, mesh_y)
    source = sol.source(mesh_x, mesh_y)
    field = GridField(u, h=(h, h), periodic=False)

    lap = field.laplacian().values()
    residual = -lap - source
    interior = residual[2:-2, 2:-2]
    interior_l2 = float(np.sqrt((interior * interior).sum() * h * h))

    spec = _nonperiodic_spec_with_source(n, source)
    result = ph_res_001.check(field, spec)
    assert result.status != "SKIPPED", f"rule SKIPPED at n={n}; reason={result.reason!r}"
    full_l2 = float(result.raw_value or 0.0)
    return interior_l2, full_l2


def _collect_layer1_residuals() -> tuple[list[float], list[float], list[float]]:
    hs = [1.0 / (n - 1) for n in LAYER_1_NS]
    interior, full = [], []
    for n in LAYER_1_NS:
        i, f = _mms_residuals_at(n)
        interior.append(i)
        full.append(f)
    return hs, interior, full


def test_layer1a_interior_fornberg_slope_is_4():
    """Fornberg 1988 reproduction: interior [2:-2, 2:-2] residual converges O(h^4)."""
    hs, interior, _ = _collect_layer1_residuals()
    if interior[-1] < 1e-14:
        pytest.skip(
            f"N=128 interior residual={interior[-1]:.3e} at float64 noise floor; "
            "slope regression would be dominated by numerical noise."
        )
    assert_slope_in_range(hs=hs, errs=interior, expected_slope=4.0, tolerance=0.2)


def test_layer1a_interior_monotonically_decreases():
    _, interior, _ = _collect_layer1_residuals()
    for k in range(len(interior) - 1):
        assert interior[k + 1] < interior[k], (
            f"interior[{LAYER_1_NS[k + 1]}]={interior[k + 1]:.3e} "
            f"not below interior[{LAYER_1_NS[k]}]={interior[k]:.3e}"
        )


def test_layer1a_interior_regression_r_squared_above_0_99():
    hs, interior, _ = _collect_layer1_residuals()
    if interior[-1] < 1e-14:
        pytest.skip("noise-floor saturation at N=128")
    log_h = np.log(np.array(hs))
    log_e = np.log(np.array(interior))
    slope, intercept = np.polyfit(log_h, log_e, 1)
    predicted = slope * log_h + intercept
    ss_res = np.sum((log_e - predicted) ** 2)
    ss_tot = np.sum((log_e - log_e.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot
    assert r_squared >= 0.99, f"R^2={r_squared:.4f} below 0.99 for interior Fornberg fit"


def test_layer1b_full_domain_slope_characterizes_boundary_contribution():
    """Characterization: full-domain residual slope is ~3.5, not 4.

    Measured: 3.49. The gap between interior O(h^4) (Layer 1a) and full-
    domain O(h^{~3.5}) (here) is the boundary contribution: outer two
    layers use one-sided/off-center stencils with O(h^2) truncation error
    (documented at grid.py:22-31). Boundary L^2 contribution scales as
    O(h^{2.5}); mixed with interior O(h^4) on log-log regression over the
    N=[16, 128] range it produces an effective slope near 3.5.

    This test asserts the characterization range [3.3, 3.7], NOT an
    external-reference reproduction. The reproduction is in Layer 1a.
    Widening this range would require investigating why the rule's
    full-domain L^2 slope moved - the widening itself would be the
    finding.
    """
    hs, _, full = _collect_layer1_residuals()
    log_h = np.log(np.array(hs))
    slope, _ = np.polyfit(log_h, np.log(full), 1)
    assert 3.3 <= slope <= 3.7, (
        f"full-domain slope = {slope:.3f}, expected [3.3, 3.7]. "
        "If the slope moved, the boundary-stencil regime changed - "
        "audit grid.py FD4 boundary logic before tightening the window."
    )


def test_layer1b_full_domain_monotonically_decreases():
    _, _, full = _collect_layer1_residuals()
    for k in range(len(full) - 1):
        assert full[k + 1] < full[k], (
            f"full[{LAYER_1_NS[k + 1]}]={full[k + 1]:.3e} "
            f"not below full[{LAYER_1_NS[k]}]={full[k]:.3e}"
        )


# =========================================================================
# Layer 2a: BDO norm-equivalence on periodic + spectral (H^-1 path)
# =========================================================================

LAYER_2_N = 64
LAYER_2_TWO_PI = 2 * math.pi
LAYER_2_H_PER = LAYER_2_TWO_PI / LAYER_2_N


def _periodic_spec_with_source(source_array: np.ndarray) -> DomainSpec:
    spec = DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [LAYER_2_N, LAYER_2_N],
            "domain": {"x": [0.0, LAYER_2_TWO_PI], "y": [0.0, LAYER_2_TWO_PI]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )
    object.__setattr__(spec, "_source_array", source_array)
    return spec


def _per_p1(x, y):
    return 0.01 * np.sin(x) * np.sin(y)


def _per_p2(x, y):
    return 0.01 * np.sin(2 * x) * np.sin(2 * y)


def _per_p3(x, y):
    return 0.01 * np.sin(3 * x) * np.sin(3 * y)


def _periodic_rule_residual_norm(pert) -> tuple[float, float, str]:
    sol = periodic_sin_sin()
    xs = np.linspace(0.0, LAYER_2_TWO_PI, LAYER_2_N, endpoint=False)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    u_pert = sol.u(mesh_x, mesh_y) + pert(mesh_x, mesh_y)
    source = sol.source(mesh_x, mesh_y)
    field = GridField(u_pert, h=(LAYER_2_H_PER, LAYER_2_H_PER), periodic=True, backend="spectral")
    spec = _periodic_spec_with_source(source)
    result = ph_res_001.check(field, spec)
    assert result.status != "SKIPPED", (
        f"rule SKIPPED on periodic+spectral; reason={result.reason!r}"
    )
    h1 = mms_perturbation_h1_error(mesh_x, mesh_y, perturbation=pert, periodic=True)
    return float(result.raw_value or 0.0), h1, result.recommended_norm


def _load_bounds() -> dict:
    if not _FIXTURES.exists():
        pytest.skip(
            f"Layer 2 bounds not calibrated yet; run calibrate_bounds.py. Missing: {_FIXTURES}"
        )
    return json.loads(_FIXTURES.read_text())


def test_layer2a_periodic_spectral_emits_h_minus_one():
    _, _, norm = _periodic_rule_residual_norm(_per_p1)
    assert norm == "H-1", (
        f"Layer 2a prerequisite failed: rule emitted norm={norm!r}, "
        "expected 'H-1' on periodic+spectral. If this assertion fails, the "
        "BDO norm-equivalence claim cannot be tested; re-audit "
        "ph_res_001.py:123-131."
    )


def test_layer2a_rho_k1_within_bounds():
    bounds = _load_bounds()["periodic_spectral"]
    r_norm, h1, _ = _periodic_rule_residual_norm(_per_p1)
    rho = r_norm / h1
    assert bounds["c_min"] <= rho <= bounds["C_max"], (
        f"rho(k=1)={rho:.3e} outside [{bounds['c_min']:.3e}, {bounds['C_max']:.3e}]"
    )


def test_layer2a_rho_k2_within_bounds():
    bounds = _load_bounds()["periodic_spectral"]
    r_norm, h1, _ = _periodic_rule_residual_norm(_per_p2)
    rho = r_norm / h1
    assert bounds["c_min"] <= rho <= bounds["C_max"], (
        f"rho(k=2)={rho:.3e} outside [{bounds['c_min']:.3e}, {bounds['C_max']:.3e}]"
    )


def test_layer2a_rho_k3_within_bounds():
    bounds = _load_bounds()["periodic_spectral"]
    r_norm, h1, _ = _periodic_rule_residual_norm(_per_p3)
    rho = r_norm / h1
    assert bounds["c_min"] <= rho <= bounds["C_max"], (
        f"rho(k=3)={rho:.3e} outside [{bounds['c_min']:.3e}, {bounds['C_max']:.3e}]"
    )


def test_layer2a_c_max_over_c_min_below_10():
    bounds = _load_bounds()["periodic_spectral"]
    ratio = bounds["C_max"] / bounds["c_min"]
    assert ratio < 10.0, (
        f"C_max/c_min = {ratio:.2f} >= 10 on the periodic+spectral H^-1 path. "
        "BDO norm-equivalence claim failing on the variationally-correct path "
        "is a plan-integrity escalation, not an in-session amendment."
    )


# =========================================================================
# Layer 2b: L^2 fallback characterization (non-periodic + FD)
# =========================================================================


def _nonperiodic_rule_residual_norm(pert) -> tuple[float, float, str]:
    sol = sin_sin_mms_square()
    h = 1.0 / (LAYER_2_N - 1)
    xs = np.linspace(0.0, 1.0, LAYER_2_N)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    u_pert = sol.u(mesh_x, mesh_y) + pert(mesh_x, mesh_y)
    source = sol.source(mesh_x, mesh_y)
    field = GridField(u_pert, h=(h, h), periodic=False)
    spec = _nonperiodic_spec_with_source(LAYER_2_N, source)
    result = ph_res_001.check(field, spec)
    assert result.status != "SKIPPED", f"rule SKIPPED on non-periodic+FD; reason={result.reason!r}"
    h1 = mms_perturbation_h1_error(mesh_x, mesh_y, perturbation=pert, periodic=False)
    return float(result.raw_value or 0.0), h1, result.recommended_norm


def _np_p_k1(x, y):
    return 0.01 * np.sin(math.pi * x) * np.sin(math.pi * y)


def _np_p_k4(x, y):
    return 0.01 * np.sin(4 * math.pi * x) * np.sin(4 * math.pi * y)


def test_layer2b_nonperiodic_fd_emits_l2_fallback():
    _, _, norm = _nonperiodic_rule_residual_norm(_np_p_k1)
    assert norm == "L2", (
        f"Layer 2b prerequisite failed: rule emitted norm={norm!r}, "
        "expected 'L2' on non-periodic+FD. This characterization assumes the "
        "rule has fallen back to L^2 per ph_res_001.py:123-131."
    )


def test_layer2b_rho_scales_linearly_with_wavenumber():
    """Characterization: rho(k=4) / rho(k=1) is approximately 4 on the L^2 fallback path.

    For perturbation p = sin(k*pi*x)sin(k*pi*y) on [0,1]^2:
        ||Laplacian p||_{L^2} scales as k^2 * constant
        ||p||_{H^1} scales as k * constant (gradient-dominated)
        rho = ||Laplacian p||_{L^2} / ||p||_{H^1} scales as k

    Predicted rho(k=4)/rho(k=1) = 4. Measured on the rule: ~4.12 (3% error).
    This is a POSITIVE assertion about the rule's documented L^2 fallback
    behavior - it is not a test of norm-equivalence. Norm-equivalence is
    tested in Layer 2a on the H^-1 path.
    """
    r_norm_k1, h1_k1, _ = _nonperiodic_rule_residual_norm(_np_p_k1)
    r_norm_k4, h1_k4, _ = _nonperiodic_rule_residual_norm(_np_p_k4)
    rho_k1 = r_norm_k1 / h1_k1
    rho_k4 = r_norm_k4 / h1_k4
    ratio = rho_k4 / rho_k1
    predicted = 4.0
    assert 3.5 <= ratio <= 4.5, (
        f"rho(k=4)/rho(k=1) = {ratio:.3f}, expected ~{predicted} "
        "(within +/-12.5%); if ratio departs significantly, the rule's L^2 "
        "fallback behavior differs from the documented ph_res_001.py:14-20 "
        "specification."
    )
