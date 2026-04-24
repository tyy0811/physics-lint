"""PH-CON-001 external-validation anchor - heat mass conservation (analytic snapshots).

Task 8 of the complete-v1.0 plan. PH-CON-001 is a FLAG-level task per
the 2026-04-24 forward-look precheck: "1e-14 cosine-IC conservation"
tolerance is fixture-mode-dependent. Task 8 revised contract
(2026-04-24) resolves this by scoping F2 explicitly to
**analytical-snapshot** mode.

Wording discipline (CITATION.md + README + tests): PH-CON-001 validates
the production rule's ability to measure integral conservation drift on
analytically controlled source-free snapshots. It does not certify the
accuracy of a heat-equation time integrator.

Three-function-labeled stack per complete-v1.0 plan section 1.3:

    F1  Balance-law / mass-conservation identity (Evans 2010 section
        2.3, section-level WARN; Dafermos 2016 Chapter I balance-law
        framing, section-level WARN). For source-free heat on periodic
        or no-flux domain:
            d/dt int_Omega u(t, x) dV = 0.

    F2  Analytical-snapshot correctness fixture (authoritative). Uses
        external_validation/_harness/energy.py
        analytical_heat_snapshot_2d to build the exact solution
        u(x, y, t) = cos(2 pi x) cos(2 pi y) * exp(-8 pi^2 kappa t) on
        [0, 1]^2 periodic. Spatial integral is zero at every t. Rule's
        emitted mass-drift metric should land at numerical roundoff.
        Scope is conservation-drift detection on analytic snapshots,
        NOT time-stepper accuracy validation.

    F3  Absent with justification. Live Hansen ProbConserv reproduction
        (docs/audits/2026-04-22-pdebench-hansen-pins.md line 166, ANP
        row CE = 4.68e-3 +/- 0.10) requires integration with the
        github.com/amazon-science/probconserv repository -- a
        pre-trained checkpoint loader and inference pipeline -- which
        V1 physics-lint does not ship. external_validation/README.md
        already scopes "PH-CON-001 Hansen ProbConserv mass CE" to v1.1.
        Hansen Table 1 ANP row retained in Supplementary calibration
        context with semantic-equivalence derivation preserved.

Rule-verdict contract:
    Production verdict validates conservation-drift detection on
    analytical snapshots. It does not validate the numerical accuracy
    of a heat-equation solver. An optional numerically-evolved fixture
    would use a method-dependent tolerance (O(delta_t) for explicit
    FTCS, etc.) and is NOT included in V1 -- Task 8 stays
    analytical-snapshot-only per the revised contract.

Observed analytical-snapshot mass-drift floor (2026-04-24, float64):
    ~1e-18 absolute (relative to L^1 scale ~1.0) across
    (Nx, Nt) in {16, 32, 64, 128} x {5, 11, 21}. Max observed:
    3.57e-18. Tolerance for the PASS assertion set at 1e-15 to give a
    ~300x safety factor over the observed maximum -- well below the
    rule's shipped-default relative-mass floor of 1e-4, so rule
    classification is comfortably PASS.

Plan-diffs logged (plan-vs-committed-state drift, section 7.4):
    9. (Task 8) Plan section 16 F3 "Per Task 0 pin outcome: Hansen +
       PDEBench reproduction on pinned rows" recast as F3-absent +
       Hansen in Supplementary calibration context. Reason: F3
       executability infrastructure check (per
       feedback_precheck_f3_executability_category.md) confirmed V1
       physics-lint does not ship a ProbConserv loader; a mid-plan
       introduction of this infrastructure exceeds Task 8's 1.5 ED
       budget. The pre-existing external_validation/README.md entry
       already scopes Hansen reproduction to v1.1. Semantic-equivalence
       derivation (Hansen CE measures prediction-vs-exact-conservation;
       PH-CON-001 measures prediction-mass-vs-IC-mass; equivalent for
       zero-mean IC) preserved from Task 0 audit.
    10. (Task 8) Plan section 16 acceptance criterion "Cosine-IC
        conservation to 1e-14" replaced with empirically-measured
        tolerance 1e-15 (observed floor 1e-18, 1000x safety factor).
        Reason: per user-approved revised Task 8 contract, "avoid
        hard-coding 1e-14 unless measured robustly across N and
        platforms." Measured floor is 3 orders of magnitude below
        1e-14, so 1e-14 would be loose; 1e-15 lands between observed
        floor and the rule's 1e-4 shipped default with large safety
        margins on both sides.
    11. (Task 8) Plan section 16 F2 fixture "Cosine-IC fixture:
        u_0(x) = cos(2 pi x) on periodic domain" extended to the 2D
        analog u(x, y, t) = cos(2 pi x) cos(2 pi y) * exp(-8 pi^2
        kappa t) to satisfy the rule's ndim >= 3 + nt >= 3
        precondition (ph_con_001.py:52-59 requires time-dependent
        field with at least 3 time samples for 2nd-order central time
        derivative). The 2D spatial extension preserves the zero-
        spatial-mean property at every t.

Plan-diffs 1-8 are from Tasks 2, 3, 4, 5 (commits 30baf3e, 0cedc7b,
18312b9, 6800d6f).
"""

from __future__ import annotations

import pytest

from external_validation._harness.energy import analytical_heat_snapshot_2d
from physics_lint import DomainSpec
from physics_lint.field import GridField
from physics_lint.rules import ph_con_001

ANALYTIC_SNAPSHOT_TOL = 1e-15
REFINEMENT_NS = (16, 32, 64, 128)
TIME_SAMPLES = (5, 11, 21)


def _build_spec(nx: int, nt: int, kappa: float, t_end: float) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [nx, nx, nt],
            "domain": {
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
                "t": [0.0, t_end],
            },
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "field": {
                "type": "grid",
                "backend": "fd",
                "dump_path": "p.npz",
            },
            "diffusivity": kappa,
        }
    )


def _run_rule_on_analytic_snapshot(nx: int, nt: int, kappa: float = 1.0, t_end: float = 0.1):
    u, h = analytical_heat_snapshot_2d(fixture="cos_2pi_2d", nx=nx, nt=nt, kappa=kappa, t_end=t_end)
    field = GridField(u, h=h, periodic=True)
    spec = _build_spec(nx=nx, nt=nt, kappa=kappa, t_end=t_end)
    return ph_con_001.check(field, spec)


# =========================================================================
# F2 analytical-snapshot correctness: rule's mass-drift metric should
# land at numerical roundoff on source-free analytically controlled
# periodic snapshots. This is the AUTHORITATIVE F2 layer per user-
# approved Task 8 revised contract. Does NOT exercise a numerical
# solver; does NOT claim solver accuracy coverage.
# =========================================================================


@pytest.mark.parametrize("nt", TIME_SAMPLES)
@pytest.mark.parametrize("nx", REFINEMENT_NS)
def test_f2_analytical_snapshot_mass_drift_is_at_roundoff(nx, nt):
    """Analytic heat snapshot has zero spatial mean exactly; rule emits
    mass-drift at or below 1e-15 (300x margin over observed 1e-18 floor).
    """
    result = _run_rule_on_analytic_snapshot(nx=nx, nt=nt)
    assert result.status == "PASS", (
        f"rule status at (nx={nx}, nt={nt}) expected PASS on analytic "
        f"snapshot, got {result.status!r}"
    )
    assert result.mode == "exact-mass", (
        f"rule mode at (nx={nx}, nt={nt}) expected 'exact-mass' (periodic "
        f"-> conserves_mass branch), got {result.mode!r}"
    )
    assert result.raw_value is not None
    assert abs(result.raw_value) < ANALYTIC_SNAPSHOT_TOL, (
        f"mass-drift raw_value={result.raw_value!r} at (nx={nx}, nt={nt}) "
        f"exceeds analytical-snapshot tolerance {ANALYTIC_SNAPSHOT_TOL:.0e} "
        f"(observed floor is ~1e-18; tolerance has ~1000x margin)"
    )


def test_f2_rule_detects_injected_mass_drift():
    """Liveness test: inject a known mass perturbation at a late time
    snapshot; rule raw_value should rise above roundoff in proportion.

    Add delta=1e-6 uniformly to u[..., -1]. Mass perturbation =
    delta * area = delta * 1.0 = 1e-6 (constant integrand over unit
    square). Rule's raw_value is max|M(t) - M(0)| / max(|M(0)|, ||u_0||_1).
    M(0) ~= 0 (zero-mean periodic); ||cos(2 pi x) cos(2 pi y)||_1 on
    [0, 1]^2 = (2/pi)^2 ~= 0.4053. So expected raw ~= 1e-6 / 0.4053
    ~= 2.47e-6 (empirically measured 2.48e-6 at 2026-04-24).
    """
    nx, nt = 32, 11
    delta = 1e-6
    u, h = analytical_heat_snapshot_2d(nx=nx, nt=nt)
    u_perturbed = u.copy()
    u_perturbed[..., -1] += delta

    field = GridField(u_perturbed, h=h, periodic=True)
    spec = _build_spec(nx=nx, nt=nt, kappa=1.0, t_end=0.1)
    result = ph_con_001.check(field, spec)

    assert result.raw_value is not None
    assert result.raw_value > 1e-6, (
        f"rule raw_value={result.raw_value!r} should rise to ~2.5e-6 after "
        f"injecting delta={delta} mass perturbation (expected "
        f"delta / (2/pi)^2); rule may not be live"
    )
    # Upper bound: envelope around the theoretical 2.47e-6 magnitude.
    assert result.raw_value < 5e-6, (
        f"rule raw_value={result.raw_value!r} on delta={delta} exceeds "
        f"theoretical estimate ~2.47e-6 by more than 2x; suggests metric "
        f"is misscaled"
    )


def test_f2_refinement_invariance_of_analytic_snapshot_floor():
    """Analytical-snapshot floor is refinement-invariant (no N-scaling).

    Unlike numerically-evolved fixtures (where O(h^p) time-integrator
    error scales with N), the analytical-snapshot fixture's drift is
    roundoff-bound across all tested N. Confirmed by the parametrized
    test above landing below 1e-15 at every (nx, nt); this summary test
    asserts the max over the sweep is still below tolerance.
    """
    max_raw = 0.0
    for nx in REFINEMENT_NS:
        for nt in TIME_SAMPLES:
            result = _run_rule_on_analytic_snapshot(nx=nx, nt=nt)
            assert result.raw_value is not None
            max_raw = max(max_raw, abs(result.raw_value))
    assert max_raw < ANALYTIC_SNAPSHOT_TOL, (
        f"max analytical-snapshot mass drift across sweep = {max_raw!r} "
        f"exceeds tolerance {ANALYTIC_SNAPSHOT_TOL:.0e}"
    )


# =========================================================================
# Rule-verdict contract (RVC): production rule's V1 emitted quantity and
# mode-branch behavior. Does NOT claim solver-accuracy validation.
# =========================================================================


def test_rvc_rule_skipped_on_non_heat_pde():
    """PH-CON-001 is heat-only in V1 (ph_con_001.py:47). Other PDEs SKIP."""
    u, h = analytical_heat_snapshot_2d(nx=16, nt=5)
    field = GridField(u, h=h, periodic=True)
    spec = DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [16, 16, 5],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.1]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    result = ph_con_001.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason is not None and "heat-only" in result.reason


def test_rvc_rule_skipped_on_insufficient_time_samples():
    """Rule needs nt >= 3 for 2nd-order central time derivative
    (ph_con_001.py:42, _MIN_TIME_STEPS_FOR_GRADIENT)."""
    # Construct a minimal 2-timestep fixture (below threshold).
    u, h = analytical_heat_snapshot_2d(nx=16, nt=3)
    u_short = u[..., :2]  # drop to 2 time samples
    h_short = h
    field = GridField(u_short, h=h_short, periodic=True)
    spec = _build_spec(nx=16, nt=2, kappa=1.0, t_end=0.1)
    result = ph_con_001.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason is not None and "3 time" in result.reason


def test_rvc_exact_mass_branch_activates_on_periodic_bc():
    """Periodic BC -> conserves_mass=True -> exact-mass branch (vs
    rate-consistency branch on Dirichlet)."""
    result = _run_rule_on_analytic_snapshot(nx=32, nt=11)
    assert result.mode == "exact-mass", (
        f"expected exact-mass branch on periodic BC, got mode={result.mode!r}"
    )
