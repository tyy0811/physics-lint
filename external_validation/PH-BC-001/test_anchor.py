"""PH-BC-001 external-validation anchor - Dirichlet boundary-trace violation.

Task 4 of the complete-v1.0 plan. PH-BC-001 is a FLAG-level task per the
2026-04-24 forward-look precheck (docs/audits/2026-04-24-plan-smoke-
precheck.md): the production rule's `values_on_boundary()` semantic is
Dirichlet-type (value on boundary), not Neumann-type (normal derivative
on boundary). External validation scopes F2 to Dirichlet-trace behavior
only; Neumann fixture handling is deferred.

Wording discipline (CITATION.md + README + tests): do not write
"PH-BC-001 validates boundary conditions." The production rule does not
validate Neumann flux semantics. Write: "PH-BC-001 validates Dirichlet-
type boundary trace behavior in the production rule; Neumann/flux
semantics are outside the production validation scope for v1.0."

Three-function-labeled stack per complete-v1.0 plan section 1.3:

    F1  Trace theorem (Evans 2010 section 5.5 Theorem 1, section-level
        per ../_harness/TEXTBOOK_AVAILABILITY.md WARN): the trace
        operator gamma: H^1(Omega) -> H^{1/2}(partial Omega) is
        bounded on Lipschitz-boundary domains.

    F2  Correctness-fixture layer with three analytic Dirichlet
        fixtures on the unit square (two explicit per the Task 4
        acceptance contract, plus a third for absolute-mode coverage):
          - u = x^2 - y^2 (polynomial, relative mode, nonzero
            Dirichlet)
          - u = sin(pi x) sin(pi y) (trigonometric, absolute mode,
            zero-Dirichlet)
          - u = cos(pi x) cos(pi y) (trigonometric, relative mode,
            nonzero Dirichlet)
        Boundary target construction via
        external_validation/_harness/trace.py
        dirichlet_trace_on_unit_square_grid. Scope is Dirichlet-trace
        value mismatch only; Neumann flux is not exercised.

    F3  Absent with justification. Plan section 12 + Task 0 PDEBench
        pin (docs/audits/2026-04-22-pdebench-hansen-pins.md section
        "Task 4") identified Diffusion-sorption (Dirichlet-dominant)
        and 2D diffusion-reaction (Neumann no-flow) as semantically-
        equivalent bRMSE reproduction targets. Live CI-runnable
        reproduction would require a PDEBench dataset loader
        (adapter-mode plumbing for the external benchmark), which
        V1 physics-lint does not yet ship. Semantic-equivalence
        derivation is documented in CITATION.md Supplementary
        calibration context; live reproduction deferred.

Rule-verdict contract:
    Rule correctly PASSes with raw_value = 0 exactly when the field's
    boundary trace matches the target. When boundary is perturbed by a
    known amount, raw_value scales with the perturbation magnitude in
    the expected discrete-L^2 sense, demonstrating the rule is live.
    Mode branches correctly: absolute when ||g|| < 1e-8, relative
    otherwise. Shape mismatch between field boundary and target raises
    ValueError with diagnostic.

Plan-diffs logged (plan-vs-committed-state drift, plan section 7.4):
    7. (Task 4) Plan section 12 F2 fixtures listed as
       "Dirichlet, Neumann, periodic". Production rule PH-BC-001's
       values_on_boundary() extracts Dirichlet-type trace (value on
       boundary, not normal derivative). Fixture scope tightened to
       Dirichlet-only per 2026-04-24 Path C precheck + FLAG
       disposition + user-approved revised Task 4 contract; Neumann
       semantics explicitly deferred; periodic is vacuous on a torus
       (no boundary) and omitted.
    8. (Task 4) Plan section 12 F3 "If F3-present: PDEBench
       reproduction per pinned row" recast as F3-absent + PDEBench
       rows in Supplementary calibration context with semantic-
       equivalence derivation. Reason: V1 physics-lint lacks a
       PDEBench dataset loader; live CI-runnable reproduction of
       U-Net / FNO bRMSE numbers on the pinned PDEBench rows would
       exceed Task 4's 1.5 ED budget and was not identified in the
       2026-04-24 precheck (precheck missed the loader-infrastructure
       gap). Per user's revised Task 4 contract: "Use PDEBench pins
       only for boundary-condition credibility context. Do not
       overstate this as a direct theorem reproduction unless the
       fixture exactly supports that claim."

Plan-diffs 1-6 are from Tasks 2 + 3 + 5 (commits 30baf3e, 0cedc7b,
18312b9).
"""

from __future__ import annotations

import numpy as np
import pytest

from external_validation._harness.trace import dirichlet_trace_on_unit_square_grid
from physics_lint import DomainSpec
from physics_lint.field import GridField
from physics_lint.rules import ph_bc_001

REFINEMENT_NS = (16, 32, 64)


def _poly_deg2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """u = x^2 - y^2. Polynomial Dirichlet field; nonzero trace on unit square."""
    return x**2 - y**2


def _trig_zero_dirichlet(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """u = sin(pi x) sin(pi y). Homogeneous Dirichlet on unit square; absolute mode."""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def _trig_nonzero_dirichlet(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """u = cos(pi x) cos(pi y). Nonzero Dirichlet on unit square; relative mode."""
    return np.cos(np.pi * x) * np.cos(np.pi * y)


def _build_field_and_spec(u_fn, n):
    xs = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    u = u_fn(mesh_x, mesh_y)
    h = 1.0 / (n - 1)
    field = GridField(u, h=(h, h), periodic=False)
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [n, n],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    return field, spec, u


# =========================================================================
# F2 correctness-fixture layer: three Dirichlet analytic fixtures.
#
# Per the user-approved Task 4 revised contract, at least two analytic
# functions with known boundary values. We use three (polynomial +
# zero-Dirichlet trig + nonzero-Dirichlet trig) so the rule's absolute-
# and relative-mode branches are both exercised, plus a third fixture
# in case one library-level change makes one of the two flip mode.
# =========================================================================


@pytest.mark.parametrize(
    ("u_fn", "fixture_label", "expected_mode"),
    [
        (_poly_deg2, "x^2 - y^2 (polynomial)", "relative"),
        (_trig_zero_dirichlet, "sin(pi x) sin(pi y) (zero Dirichlet)", "absolute"),
        (_trig_nonzero_dirichlet, "cos(pi x) cos(pi y) (nonzero Dirichlet)", "relative"),
    ],
)
@pytest.mark.parametrize("n", REFINEMENT_NS)
def test_f2_dirichlet_fixture_passes_when_boundary_exact(u_fn, fixture_label, expected_mode, n):
    """Field matches analytical Dirichlet target exactly -> raw = 0, PASS.

    Covers the user-approved Task 4 contract's "at least two analytic
    functions" requirement (three fixtures here for mode-branch coverage).
    """
    field, spec, _ = _build_field_and_spec(u_fn, n)
    target = dirichlet_trace_on_unit_square_grid(u_fn, n)
    result = ph_bc_001.check(field, spec, boundary_target=target)
    assert result.status == "PASS", (
        f"rule status at n={n} on {fixture_label} expected PASS, got {result.status!r}"
    )
    assert result.raw_value == pytest.approx(0.0, abs=1e-14), (
        f"rule raw_value at n={n} on {fixture_label} expected 0 on exact "
        f"boundary, got {result.raw_value!r}"
    )
    assert result.mode == expected_mode, (
        f"rule mode at n={n} on {fixture_label} expected {expected_mode!r} "
        f"(per ||g||-based branch), got {result.mode!r}"
    )


def test_f2_rule_detects_known_boundary_perturbation():
    """Perturbing a field's boundary by a known delta produces a raw_value
    that scales with the discrete-L2 magnitude of the perturbation.

    For u = x^2 - y^2 on N=32: add delta=0.001 to the left edge (Ny=32
    entries out of total boundary length 4N-4=124). Unperturbed err = 0;
    perturbed err_norm = 0.001 * sqrt(Ny) / sqrt(total_boundary_len)
    = 0.001 * sqrt(32) / sqrt(124) ~= 5.08e-4. Rule's raw_value in
    relative mode equals err_norm / g_norm where g_norm > 0, so
    raw_value > 0 and within roughly an order of magnitude of err_norm.
    """
    n = 32
    delta = 1e-3
    field_clean, spec, u = _build_field_and_spec(_poly_deg2, n)
    target = dirichlet_trace_on_unit_square_grid(_poly_deg2, n)

    result_clean = ph_bc_001.check(field_clean, spec, boundary_target=target)
    assert result_clean.raw_value == pytest.approx(0.0, abs=1e-14)

    u_pert = u.copy()
    u_pert[0, :] += delta
    field_pert = GridField(u_pert, h=field_clean.h, periodic=False)
    result_pert = ph_bc_001.check(field_pert, spec, boundary_target=target)

    # Rule must detect the perturbation (nonzero raw).
    assert result_pert.raw_value > 1e-5, (
        f"rule failed to detect boundary perturbation delta={delta}: "
        f"raw_value={result_pert.raw_value!r}"
    )
    # Raw-value magnitude tracks the perturbation at roughly the discrete-L2
    # scale (relative mode adds the /g_norm factor, which is O(1) on this
    # fixture). Bound loosely to [1e-4, 5e-3] for a 1e-3 perturbation on
    # 32/124 boundary entries.
    assert 1e-4 < result_pert.raw_value < 5e-3, (
        f"perturbation raw_value={result_pert.raw_value!r} outside plausible "
        f"[1e-4, 5e-3] range for delta={delta} on N={n}"
    )


# =========================================================================
# Rule-verdict contract: exercises the production rule's mode-branch and
# API-contract behavior. Does NOT claim general boundary-condition
# validation coverage (Neumann flux is explicitly out of scope).
# =========================================================================


def test_rvc_rule_mode_branches_on_g_norm_threshold():
    """Rule's mode-branch key: absolute when ||g|| < 1e-8, relative
    otherwise. Exercised via the zero-Dirichlet (||g|| = roundoff) and
    nonzero-Dirichlet (||g|| ~= O(1)) fixtures.
    """
    field_zero, spec_zero, _ = _build_field_and_spec(_trig_zero_dirichlet, 32)
    target_zero = dirichlet_trace_on_unit_square_grid(_trig_zero_dirichlet, 32)
    g_norm_zero = float(np.linalg.norm(target_zero) / np.sqrt(max(len(target_zero), 1)))
    assert g_norm_zero < 1e-8, (
        f"zero-Dirichlet fixture's boundary norm {g_norm_zero!r} should "
        "be below the rule's absolute-mode threshold 1e-8"
    )
    result_zero = ph_bc_001.check(field_zero, spec_zero, boundary_target=target_zero)
    assert result_zero.mode == "absolute"

    field_nz, spec_nz, _ = _build_field_and_spec(_trig_nonzero_dirichlet, 32)
    target_nz = dirichlet_trace_on_unit_square_grid(_trig_nonzero_dirichlet, 32)
    g_norm_nz = float(np.linalg.norm(target_nz) / np.sqrt(max(len(target_nz), 1)))
    assert g_norm_nz > 1e-2, (
        f"nonzero-Dirichlet fixture's boundary norm {g_norm_nz!r} should "
        "be well above the rule's absolute-mode threshold 1e-8"
    )
    result_nz = ph_bc_001.check(field_nz, spec_nz, boundary_target=target_nz)
    assert result_nz.mode == "relative"


def test_rvc_rule_raises_on_shape_mismatch():
    """Category 8 API-contract: rule raises ValueError with diagnostic
    when boundary_target shape doesn't match field boundary length.
    """
    field, spec, _ = _build_field_and_spec(_poly_deg2, 16)
    wrong_target = np.zeros(5)  # intentionally wrong length
    with pytest.raises(ValueError, match="boundary_target shape"):
        ph_bc_001.check(field, spec, boundary_target=wrong_target)


def test_rvc_rule_absolute_mode_fails_above_abs_tol():
    """Absolute mode: when ||g|| < 1e-8 and err_norm >= abs_tol_fail (1e-3
    default), rule emits FAIL (binary PASS/FAIL, no tri-state in this
    branch per ph_bc_001.py Rev 4.1 fix).
    """
    n = 32
    field, spec, u = _build_field_and_spec(_trig_zero_dirichlet, n)
    target = dirichlet_trace_on_unit_square_grid(_trig_zero_dirichlet, n)
    u_bad = u.copy()
    u_bad[0, :] += 1e-2  # well above abs_tol_fail=1e-3 in discrete-L2 sense
    field_bad = GridField(u_bad, h=field.h, periodic=False)
    result = ph_bc_001.check(field_bad, spec, boundary_target=target)
    assert result.mode == "absolute"
    assert result.status == "FAIL", (
        f"rule status on absolute-mode violation above abs_tol_fail expected "
        f"FAIL, got {result.status!r} (raw_value={result.raw_value!r})"
    )
