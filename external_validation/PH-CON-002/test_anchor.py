"""PH-CON-002 external-validation anchor - wave energy conservation (analytic snapshots).

Task 9 of the complete-v1.0 plan. PH-CON-002 is a FLAG-level task per
the 2026-04-24 forward-look precheck: plan's "1e-8 energy over 1000
leapfrog steps" tolerance is tight by ~5 orders of magnitude versus
leapfrog's bounded O(Delta t^2 * E_max) oscillation. Task 9 revised
contract (2026-04-24) resolves this by splitting the validation into
two distinct layers:

  F2 harness-level (authoritative): compute E(t) directly from
      analytical u_t, u_x, u_y field components. Drift is roundoff-only
      because no numerical derivative is taken.

  Rule-verdict contract: feed analytical u snapshots through the
      production rule. Rule computes u_t internally via 2nd-order
      central FD (ph_con_002.py:65), introducing O(Delta t^2)
      truncation error. Tolerance is method-dependent per user's
      revised contract "if a leapfrog/numerically evolved test is
      retained, label it supplementary liveness and give it a
      method-dependent tolerance."

Wording discipline (CITATION.md + README + tests): PH-CON-002 validates
the production rule's ability to measure wave-energy drift on
analytically controlled conservative snapshots. It does not certify the
accuracy of a wave-equation time integrator.

Three-function-labeled stack per complete-v1.0 plan section 1.3:

    F1  Wave-equation energy identity. For u_tt = c^2 u_xx under
        periodic or compatible homogeneous boundary conditions,
            E(t) = 1/2 integral (u_t^2 + c^2 |grad u|^2) dV
        is constant (Evans 2010 section 2.4.3, section-level WARN;
        Strauss 2007 section 2.2, section-level WARN; Hairer-Lubich-
        Wanner 2006 Chapter IX on symplectic-integrator conservation,
        section-level WARN).

    F2  Analytical-snapshot correctness fixture (authoritative). Uses
        external_validation/_harness/energy.py
        analytical_wave_snapshot_2d_yindep to build the exact solution
        u(x, y, t) = sin(k x) cos(c k t) on [0, 2 pi]^2 periodic, with
        closed-form u_t, u_x, u_y. wave_energy_from_analytical_fields
        computes E(t) directly from the analytical components; drift
        across time snapshots is roundoff-bound (observed ~5e-16
        relative). Scope is energy-invariant validation on analytic
        snapshots, NOT time-stepper accuracy validation.

    F3  Absent with justification (pre-recorded by Task 0 Step 4 pin
        audit, docs/audits/2026-04-22-pdebench-hansen-pins.md line
        207). PDEBench lacks a standalone wave-equation dataset; its
        shallow-water and compressible-NS rows report mass-only
        conservation metrics that are semantically incompatible with
        wave-energy. Hansen ProbConserv CE metric is defined for
        first-order-in-time integral laws, not second-order-in-time
        energy functionals. No F3-INFRA-GAP risk (F3-absent is
        structural, not a loader-infrastructure gap).

Rule-verdict contract:
    Production rule computes u_t via 2nd-order central FD
    (ph_con_002.py:65 np.gradient(u, dt, axis=-1, edge_order=2)), so
    feeding analytical u snapshots still leaves O(Delta t^2) truncation
    error in the rule's emitted drift. Measured at 2026-04-24: rule
    drift raw_value = alpha * Delta t^2 with alpha approx 1/3 * (ck)^2
    for c = k = 1 (dominant-order analysis: central-FD error on u_t is
    (dt^2/6) * u_ttt, and u_ttt = (ck)^3 * sin(kx) sin(ckt) on this
    fixture). The rule PASSes (raw < 0.01) for Delta t <= pi / 25 on
    one full wave period and exhibits log-log slope 1.94 (approximately
    2) across the tested Delta t range. Refinement-independent on
    spatial nx (error is time-FD-bound).

Plan-diffs logged (plan-vs-committed-state drift, section 7.4):
    12. (Task 9) Plan section 17 F2 "u = sin(kx) cos(ckt) fixture
        with leapfrog time-stepper; assert E(t) bounded oscillation
        within 1e-8 over 1000 steps" replaced with two-layer
        analytical-snapshot F2:
        - Layer A (authoritative harness-level): E(t) from analytical
          u_t, u_x, u_y directly -- roundoff-only drift (observed
          ~5e-16 relative; tolerance 1e-14 with ~20x safety over
          observed).
        - Rule-verdict contract: rule drift = O(Delta t^2) method-
          dependent tolerance (measured 1/3 * (ck)^2 * Delta t^2 at
          c = k = 1). Rule PASSes at Delta t <= pi/25; drift
          converges as Delta t^2 (slope ~1.94).
        Plan's "leapfrog time-stepper with 1000 steps" is NOT
        implemented in V1: PH-CON-002 rule internally computes u_t
        via central FD, not a leapfrog stepper; the rule's scope is
        energy-drift DETECTION, not time-stepper accuracy. Per user-
        approved Task 9 revised contract, any leapfrog-evolved
        fixture would be labeled supplementary liveness with method-
        dependent tolerance; none is included in V1 anchor.
    13. (Task 9) Plan section 17 F2 fixture "u = sin(kx) cos(ckt)"
        (1D) extended to 2D y-independent analog u(x, y, t) = sin(k x)
        cos(c k t) to satisfy the rule's ndim >= 3 + nt >= 3
        precondition (ph_con_002.py:51-57). The y-independent 2D
        extension preserves the energy-invariance property (E factors
        as E_x * 2pi from y integration); E(0) = pi^2 c^2 k^2.

Plan-diffs 1-11 are from Tasks 2, 3, 4, 5, 8 (commits 30baf3e,
0cedc7b, 18312b9, 6800d6f, 1112da3).
"""

from __future__ import annotations

import numpy as np
import pytest

from external_validation._harness.energy import (
    analytical_wave_snapshot_2d_yindep,
    wave_energy_from_analytical_fields,
)
from physics_lint import DomainSpec
from physics_lint.field import GridField
from physics_lint.rules import ph_con_002

HARNESS_ENERGY_TOL = 1e-14
RULE_PASS_NT = 51  # at one full period with c=k=1, dt <= pi/25 ~= 0.126 => rule PASSes
REFINEMENT_NXS = (16, 32, 64)
SNAPSHOT_NTS = (11, 21, 51)


def _build_spec(*, nx: int, nt: int, c: float, t_end: float) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "wave",
            "grid_shape": [nx, nx, nt],
            "domain": {
                "x": [0.0, 2.0 * np.pi],
                "y": [0.0, 2.0 * np.pi],
                "t": [0.0, t_end],
            },
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "field": {
                "type": "grid",
                "backend": "fd",
                "dump_path": "p.npz",
            },
            "wave_speed": c,
        }
    )


# =========================================================================
# F2 Layer A: authoritative harness-level E(t) from analytical fields.
#
# Computes the energy functional directly from analytical u_t, u_x, u_y
# -- no numerical derivative. Drift across time snapshots is roundoff-
# bound (~5e-16 relative), confirming the F1 identity dE/dt = 0 to
# float64 precision on the spectrally-accurate rectangle quadrature.
# This layer is INDEPENDENT of the production rule.
# =========================================================================


@pytest.mark.parametrize("nt", SNAPSHOT_NTS)
@pytest.mark.parametrize("nx", REFINEMENT_NXS)
def test_f2_harness_energy_is_invariant_at_roundoff(nx, nt):
    """E(t) computed from analytical u_t, u_x, u_y is constant at roundoff.

    Relative drift |E(t) - E(0)| / |E(0)| stays below HARNESS_ENERGY_TOL
    = 1e-14 across (nx, nt). Refinement-invariant because the rectangle
    rule is spectrally accurate on the analytic periodic integrands.
    """
    _u, u_t, u_x, u_y, h = analytical_wave_snapshot_2d_yindep(nx=nx, nt=nt, c=1.0, k=1)
    energies = wave_energy_from_analytical_fields(
        u_t=u_t, u_x=u_x, u_y=u_y, c=1.0, h=h, periodic=True
    )
    drift_rel = float(np.max(np.abs(energies - energies[0])) / abs(energies[0]))
    assert drift_rel < HARNESS_ENERGY_TOL, (
        f"harness-level relative energy drift {drift_rel!r} at "
        f"(nx={nx}, nt={nt}) exceeds tolerance {HARNESS_ENERGY_TOL:.0e}; "
        f"observed floor is ~5e-16 with ~20x margin at the tolerance"
    )


def test_f2_harness_energy_matches_analytical_value():
    """E(0) = pi^2 c^2 k^2 on [0, 2 pi]^2 y-independent fixture.

    Derivation (see _harness/energy.py analytical_wave_snapshot_2d_yindep):
    integral_0^{2pi} sin^2(k x) dx = integral_0^{2pi} cos^2(k x) dx = pi
    for integer k; sin^2(ckt) + cos^2(ckt) = 1; y integral contributes
    factor 2 pi. So E = pi^2 c^2 k^2. For c = k = 1, E = pi^2 ~= 9.8696.
    """
    _u, u_t, u_x, u_y, h = analytical_wave_snapshot_2d_yindep(nx=32, nt=11, c=1.0, k=1)
    energies = wave_energy_from_analytical_fields(
        u_t=u_t, u_x=u_x, u_y=u_y, c=1.0, h=h, periodic=True
    )
    expected_e0 = np.pi**2
    assert energies[0] == pytest.approx(expected_e0, rel=1e-14, abs=1e-14), (
        f"harness-level E(0)={energies[0]!r} does not match analytical "
        f"pi^2 = {expected_e0!r} for c=k=1 y-independent fixture"
    )


def test_f2_harness_energy_c_and_k_scaling():
    """E(0) scales as c^2 k^2. Verify on (c=2, k=1) and (c=1, k=2)."""
    for c, k in [(2.0, 1), (1.0, 2), (2.0, 2)]:
        _u, u_t, u_x, u_y, h = analytical_wave_snapshot_2d_yindep(nx=32, nt=11, c=c, k=k)
        energies = wave_energy_from_analytical_fields(
            u_t=u_t, u_x=u_x, u_y=u_y, c=c, h=h, periodic=True
        )
        expected = (np.pi * c * k) ** 2
        assert energies[0] == pytest.approx(expected, rel=1e-14, abs=1e-14), (
            f"harness E(0)={energies[0]!r} at (c={c}, k={k}) does not "
            f"match analytical pi^2 c^2 k^2 = {expected!r}"
        )


# =========================================================================
# Rule-verdict contract: rule's raw_value on analytical u-snapshots.
# The rule computes u_t via 2nd-order central FD internally, so it emits
# O(Delta t^2) drift on analytical snapshots. Tolerance is method-
# dependent per user's revised contract.
# =========================================================================


def _run_rule_on_analytic_snapshots(nx: int, nt: int, c: float = 1.0, k: int = 1):
    u, _u_t, _u_x, _u_y, h = analytical_wave_snapshot_2d_yindep(nx=nx, nt=nt, c=c, k=k)
    field = GridField(u, h=h, periodic=True)
    period = 2.0 * np.pi / (c * k)
    spec = _build_spec(nx=nx, nt=nt, c=c, t_end=period)
    return ph_con_002.check(field, spec)


def test_rvc_rule_passes_on_fine_time_resolution():
    """At nt = 51 (Delta t = pi/25 ~= 0.126 over one period at c = k = 1),
    rule drift (1/3) * (c k)^2 * Delta t^2 ~= 5.2e-3, below the 0.01
    shipped threshold -> rule PASSes.
    """
    result = _run_rule_on_analytic_snapshots(nx=32, nt=RULE_PASS_NT)
    assert result.status == "PASS", (
        f"rule status at nt={RULE_PASS_NT} expected PASS (theory drift "
        f"~5.2e-3 below 0.01 threshold), got {result.status!r} "
        f"(raw_value={result.raw_value!r})"
    )
    assert result.raw_value is not None
    assert 1e-3 < result.raw_value < 1e-2, (
        f"rule drift raw_value={result.raw_value!r} at nt={RULE_PASS_NT} "
        "outside expected [1e-3, 1e-2] range for O(Delta t^2) rule error"
    )


def test_rvc_rule_drift_converges_as_dt_squared():
    """Rule drift scales as O(Delta t^2). Measured log-log slope ~2 on
    nt in {11, 21, 51, 101} at fixed nx = 32.

    2026-04-24 empirical measurement: drifts (1.16e-1, 3.25e-2, 5.23e-3,
    1.32e-3) at Delta t (0.628, 0.314, 0.126, 0.063) give log-log slope
    1.94, matching the theoretical O(Delta t^2) rate for the rule's
    internal central-FD on u_t.
    """
    nts = [11, 21, 51, 101]
    drifts = []
    dts = []
    for nt in nts:
        result = _run_rule_on_analytic_snapshots(nx=32, nt=nt)
        assert result.raw_value is not None
        drifts.append(result.raw_value)
        period = 2.0 * np.pi
        dts.append(period / (nt - 1))
    log_dt = np.log(np.array(dts))
    log_d = np.log(np.array(drifts))
    slope, _intercept = np.polyfit(log_dt, log_d, 1)
    assert 1.7 < slope < 2.3, (
        f"rule-drift log-log slope vs Delta t = {slope:.4f} outside "
        f"[1.7, 2.3]; rule's internal FD should give O(Delta t^2) drift. "
        f"Drifts: {drifts!r}; dts: {dts!r}"
    )


def test_rvc_rule_refinement_independent_on_spatial_nx():
    """Rule drift is dominated by time-FD error and is ~independent of
    spatial nx at fixed nt (spatial Laplacian is spectral-exact on
    sin(kx) cos(...) well below Nyquist).
    """
    drifts_by_nx = {}
    for nx in REFINEMENT_NXS:
        result = _run_rule_on_analytic_snapshots(nx=nx, nt=RULE_PASS_NT)
        assert result.raw_value is not None
        drifts_by_nx[nx] = result.raw_value
    values = list(drifts_by_nx.values())
    # All measurements should agree to 1e-6 (well below the ~5.2e-3 drift).
    spread = max(values) - min(values)
    assert spread < 1e-6, (
        f"rule drift varies across nx beyond time-FD expectation: "
        f"drifts={drifts_by_nx!r}, spread={spread:.3e}; spatial-FFT is "
        f"supposed to be refinement-independent on sin(kx) fixture"
    )


# =========================================================================
# Rule SKIP paths (Category 8 API contract).
# =========================================================================


def test_rvc_rule_skipped_on_non_wave_pde():
    """Rule is wave-only (ph_con_002.py:40). Other PDEs SKIP."""
    u, *_rest, h = analytical_wave_snapshot_2d_yindep(nx=16, nt=11)
    field = GridField(u, h=h, periodic=True)
    spec = DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [16, 16, 11],
            "domain": {"x": [0.0, 2.0 * np.pi], "y": [0.0, 2.0 * np.pi], "t": [0.0, 2 * np.pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
            "diffusivity": 1.0,
        }
    )
    result = ph_con_002.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason is not None and "wave only" in result.reason


def test_rvc_rule_skipped_on_non_conserving_bc():
    """Rule SKIPs when boundary_condition does not conserve energy
    (ph_con_002.py:42). dirichlet_inhomogeneous is non-conserving.
    """
    u, *_rest, h = analytical_wave_snapshot_2d_yindep(nx=16, nt=11)
    field = GridField(u, h=h, periodic=False)
    # Use a BC that does not conserve energy. Per spec.py:82,
    # conserves_energy = {"periodic", "neumann_homogeneous",
    # "dirichlet_homogeneous"}; so "dirichlet" (inhomogeneous) is
    # non-conserving.
    spec = DomainSpec.model_validate(
        {
            "pde": "wave",
            "grid_shape": [16, 16, 11],
            "domain": {"x": [0.0, 2.0 * np.pi], "y": [0.0, 2.0 * np.pi], "t": [0.0, 2 * np.pi]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
            "wave_speed": 1.0,
        }
    )
    result = ph_con_002.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason is not None and "does not conserve wave energy" in result.reason


def test_rvc_rule_skipped_on_insufficient_time_samples():
    """Rule needs nt >= 3 for 2nd-order central time derivative
    (ph_con_002.py:54).
    """
    u, *_rest, h = analytical_wave_snapshot_2d_yindep(nx=16, nt=3)
    u_short = u[..., :2]
    field = GridField(u_short, h=h, periodic=True)
    spec = _build_spec(nx=16, nt=2, c=1.0, t_end=2.0 * np.pi)
    result = ph_con_002.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason is not None and "3 time" in result.reason
