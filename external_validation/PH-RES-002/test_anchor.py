"""PH-RES-002 external-validation anchor - AD vs FD residual cross-check.

The rule PH-RES-002 (src/physics_lint/rules/ph_res_002.py) compares the
Laplacian computed via torch autograd (AD path, exact to machine precision
on a smooth callable) against the 4th-order central FD stencil (FD path,
Fornberg 1988 coefficients per src/physics_lint/field/grid.py). The emitted
quantity is the max relative discrepancy on the interior [2:-2, 2:-2] band
where the FD4 stencil applies uniformly.

Structural-equivalence anchor (Function 1):
    - Griewank-Walther 2008 Chapter 3 (section-level per
      external_validation/_harness/TEXTBOOK_AVAILABILITY.md WARN): reverse-
      mode AD computes derivatives to machine-precision accuracy, bounded
      by a small constant times unit roundoff eps approx 1e-16 for float64.
    - LeVeque 2007 (section-level WARN): the 4th-order central FD stencil
      has truncation error O(h^4) in the interior.
    - Therefore |Lap_AD - Lap_FD| is bounded by C * h^4 + O(eps), and as
      h -> 0 the gap shrinks at O(h^4) until the float64 noise floor is
      reached.

Correctness-fixture layer (Function 2):
    - MMS sin(pi x) sin(pi y) on [0, 1]^2 (reuses physics_lint.analytical.
      poisson.sin_sin_mms_square for the analytical Laplacian form).
    - Refinement sweep at N in {16, 32, 64, 128}; log-log slope of max
      interior discrepancy vs h is asserted within 4.0 +/- 0.4.
    - Sanity: rule PASSes at every N in the sweep (max ratio < 0.01
      rule threshold), monotone decrease in h.

F3 (borrowed-credibility) is absent with justification - see CITATION.md.
Plan Task 2 Step 4 acceptance-criteria fallback: Task 0 literature-pin
pass produced no directly-comparable CAN-PINN Chiu 2022 CMAME row, so
CAN-PINN moves to Supplementary calibration context per plan section 10.

Plan-diff notes:
    - Plan section 10 acceptance criteria mention "FD stencil order p in
      {2, 4}"; in practice src/physics_lint/field/grid.py uses 4th-order
      FD only on the interior [2:-2] band that PH-RES-002 checks, so the
      refinement test asserts p=4 only. The outer 2 boundary layers are
      2nd-order but are excluded by the rule's interior cut.
    - Plan section 10 risks mention "CAN-PINN Table numbers not
      reproducible"; because Task 0 did not pin a CAN-PINN row, the
      rule-side F3 remains absent by structure rather than by
      reproduction failure.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from external_validation._harness.assertions import assert_slope_in_range
from physics_lint import DomainSpec
from physics_lint.field import CallableField
from physics_lint.rules import ph_res_002

REFINEMENT_NS = [16, 32, 64, 128]


def _torch_mms(coords: torch.Tensor) -> torch.Tensor:
    """u(x, y) = sin(pi x) sin(pi y). Smooth, matches sin_sin_mms_square.

    coords shape: (..., 2). Returns shape (..., 1) scalar field.
    """
    x = coords[..., 0]
    y = coords[..., 1]
    return (torch.sin(math.pi * x) * torch.sin(math.pi * y)).unsqueeze(-1)


def _build_callable_field(n: int) -> tuple[CallableField, DomainSpec]:
    """Build CallableField + DomainSpec on [0, 1]^2 Dirichlet MMS at grid N."""
    xs = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    mesh_x, mesh_y = torch.meshgrid(xs, xs, indexing="ij")
    grid = torch.stack([mesh_x, mesh_y], dim=-1)  # (N, N, 2)
    h = 1.0 / (n - 1)
    field = CallableField(_torch_mms, sampling_grid=grid, h=(h, h), periodic=False)
    spec = DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [n, n],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "callable", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    return field, spec


def _max_discrepancy_at(n: int) -> float:
    """Return the rule's max interior AD-vs-FD relative discrepancy at grid N."""
    field, spec = _build_callable_field(n)
    result = ph_res_002.check(field, spec)
    assert result.status != "SKIPPED", (
        f"PH-RES-002 returned SKIPPED at n={n}; reason={result.reason!r}"
    )
    assert result.raw_value is not None, (
        f"PH-RES-002 returned raw_value=None at n={n} (status={result.status!r})"
    )
    return float(result.raw_value)


def _collect_refinement_series() -> tuple[list[float], list[float]]:
    hs = [1.0 / (n - 1) for n in REFINEMENT_NS]
    errs = [_max_discrepancy_at(n) for n in REFINEMENT_NS]
    return hs, errs


# =========================================================================
# Layer 1: structural-equivalence reproduction (Fornberg O(h^4))
# =========================================================================


def test_layer1_refinement_slope_is_4():
    """AD-vs-FD gap shrinks at O(h^4) on the interior FD4 band.

    Griewank-Walther 2008 Chapter 3 (section-level) + LeVeque 2007
    (section-level): AD Laplacian is machine-precision exact on a smooth
    torch callable; FD4 Laplacian has O(h^4) truncation on the interior
    [2:-2] band (Fornberg 1988 coefficients). Therefore the rule's max
    interior relative discrepancy scales as O(h^4) until the float64
    noise floor is reached.
    """
    hs, errs = _collect_refinement_series()
    if errs[-1] < 1e-14:
        pytest.skip(
            f"N=128 max discrepancy={errs[-1]:.3e} at float64 noise floor; "
            "slope regression would be dominated by numerical noise."
        )
    assert_slope_in_range(hs=hs, errs=errs, expected_slope=4.0, tolerance=0.4)


def test_layer1_refinement_monotonically_decreases():
    _, errs = _collect_refinement_series()
    for k in range(len(errs) - 1):
        assert errs[k + 1] < errs[k], (
            f"errs[N={REFINEMENT_NS[k + 1]}]={errs[k + 1]:.3e} "
            f"not below errs[N={REFINEMENT_NS[k]}]={errs[k]:.3e}"
        )


def test_layer1_regression_r_squared_above_0_99():
    """log-log fit of max discrepancy vs h has R^2 >= 0.99 on the FD4 band.

    If R^2 drops below 0.99 the AD-FD gap has stopped scaling as a single
    power of h - likely the noise floor is being reached mid-range, or the
    FD4 stencil region has changed. Audit grid.py before widening.
    """
    hs, errs = _collect_refinement_series()
    if errs[-1] < 1e-14:
        pytest.skip("noise-floor saturation at N=128")
    log_h = np.log(np.array(hs))
    log_e = np.log(np.array(errs))
    slope, intercept = np.polyfit(log_h, log_e, 1)
    predicted = slope * log_h + intercept
    ss_res = np.sum((log_e - predicted) ** 2)
    ss_tot = np.sum((log_e - log_e.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot
    assert r_squared >= 0.99, f"R^2={r_squared:.4f} below 0.99 for FD4 refinement fit"


# =========================================================================
# Layer 2: rule-verdict sanity (every N passes the 0.01 threshold)
# =========================================================================


def test_rule_passes_at_every_refinement_level():
    """Rule's PASS/WARN boundary at max ratio 0.01 is comfortably clear at
    N in {16, 32, 64, 128} for the smooth sin-sin MMS fixture.

    This is a Layer 2 correctness-fixture check: the rule correctly
    classifies a smooth AD-capable model on a refined grid as PASS. If any
    N flips to WARN, either the FD4 stencil region changed or the MMS
    fixture lost smoothness - audit before softening the fixture.
    """
    _, errs = _collect_refinement_series()
    for n, err in zip(REFINEMENT_NS, errs, strict=True):
        assert err < 0.01, (
            f"max discrepancy ratio at N={n} = {err:.3e} >= 0.01 rule threshold; "
            "rule would WARN on a smooth AD-capable MMS. Root-cause before "
            "adjusting the threshold or the fixture."
        )


def test_rule_skipped_on_non_callable_field():
    """PH-RES-002 emits SKIPPED when handed a GridField (dump mode).

    Category 8 semantic-compatibility check: the rule's __input_modes__ is
    {"adapter"} only; dump mode (GridField) must emit SKIPPED with an
    explicit reason string, never silently PASS.
    """
    from physics_lint.field import GridField

    n = 16
    h = 1.0 / (n - 1)
    xs = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    u = np.sin(math.pi * mesh_x) * np.sin(math.pi * mesh_y)
    field = GridField(u, h=(h, h), periodic=False)
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
    result = ph_res_002.check(field, spec)
    assert result.status == "SKIPPED", (
        f"expected SKIPPED on GridField, got status={result.status!r}"
    )
    assert result.reason is not None, "SKIPPED result must carry a reason string"
