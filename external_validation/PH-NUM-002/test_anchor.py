"""PH-NUM-002 external-validation anchor — refinement-convergence observed order.

Task 12 of the complete-v1.0 plan. PH-NUM-002 is a FLAG-level task per the
2026-04-24 forward-look precheck: plan's "p_obs matches expected within
0.1" single tolerance is ambiguous once the expected rate is recognized to
be case-specific across (PDE, backend, BC) triples. Task 12 revised contract
(2026-04-24) resolves this by splitting the validation into three scoped
cases with per-case tolerance bands + one authoritative harness-level
methodology anchor:

  Case A (F2 harness-level, authoritative): compute p_obs on a pure-
      interior 2nd-order central-difference Laplacian applied to smooth
      harmonic fixtures. Asymptote p_obs -> 2.00; tolerance [1.9, 2.1].

  Case B (rule-verdict, fd + non-periodic): rule refinement_rate on the
      boundary-dominated FD4 path. Asymptote ~2.50 (4N-cell 2nd-order
      boundary band dominates N^2-cell 4th-order interior at O(h^2.5)
      on 2D). Tolerance [2.3, 2.8]; rule PASSes at all measured pairs.

  Case C (rule-verdict, saturation floor): spectral+periodic on a
      period-compatible harmonic (Liouville: = constant) or fd+
      non-periodic on a harmonic polynomial (2nd-order FD exact). Both
      residuals below _SATURATION_FLOOR = 1e-11 -> rule rate=inf PASS.
      No algebraic rate asserted.

Wording discipline (CITATION.md + README + tests): PH-NUM-002 validates
observed-order detection on explicitly declared manufactured-solution
cases. The expected rate is case-specific and depends on PDE, backend,
boundary treatment, and asymptotic regime. The anchor does not certify
convergence for arbitrary PDE/backend/BC triples.

Three-function-labeled stack per complete-v1.0 plan section 1.3:

    F1  Lax-equivalence theorem (Strikwerda 2004 Chapter 10, section-
        level WARN) + Roy 2005 observed-order formula p_obs =
        log2(e_h / e_{h/2}) (DOI 10.1016/j.jcp.2004.10.036) + Cea's
        lemma (Ciarlet 2002 section 3.2, section-level WARN) + Oberkampf-
        Roy 2010 Chapters 5-6 verification procedure (section-level
        WARN). Five-step proof-sketch with L^2-scaling derivation of
        Case A p=2 and Case B p=2.5 boundary-dominance.

    F2  Three-case correctness fixture (authoritative harness layer +
        two rule-verdict layers):
        - Case A: harness 2nd-order FD MMS. mms_observed_order_fd2 on
          exp(x)cos(y) and sin(pi x)sinh(pi y). p_obs asymptote 2.00.
        - Case B: rule on fd+non-periodic, boundary-dominated. Same
          fixtures. refinement_rate asymptote 2.50.
        - Case C: rule on spectral+periodic u=0, or fd+non-periodic
          on x^2-y^2 polynomial. refinement_rate = inf PASS.
        SKIP-path contracts for Poisson and heat as V1 scope boundary.

    F3  Absent with justification. No live external MMS benchmark
        dataset publishes a reproducible p_obs for a specific PDE+
        backend+BC triple matching physics-lint's rule path. Oberkampf-
        Roy 2010 and Roy 2005 remain in Mathematical-legitimacy as
        methodology + theoretical framing (2026-04-24 user-revised
        Task 12 contract: "published MMS / verification literature can
        remain mathematical or supplementary context"). No F3-INFRA-GAP
        (F3-absent is structural, not a loader gap).

Rule-verdict contract:
    Rule computes rate = log2(r_coarse / r_fine) where r = L2 norm of
    -Delta_h u (ph_num_002.py:127). FD4 interior is 4th-order
    (Fornberg 1988) but the outer boundary band is 2nd-order; on 2D
    with N ~ 1/h the 4N-cell boundary term dominates at O(h^2.5) in
    L2, giving measured p_obs asymptote 2.50. Saturation floor at
    1e-11 (ph_num_002.py:115-121) clamps to rate=inf when both
    residuals fall below floor.

Liouville scope-truth observation (plan-diff 18):
    Rule docstring claims "fd4 interior-dominated (periodic): ~4 per
    doubling" but this regime is structurally unreachable. The only
    harmonic functions on the 2-torus are constants (Liouville), so
    periodic + harmonic fixtures trivially saturate; non-harmonic
    periodic fixtures correctly WARN at rate ~0. The rule's shipped
    behavior is correct; V1 anchor restricts F2 to reachable cases.

Plan-diffs logged (plan-vs-committed-state drift, section 7.4):
    14. (Task 12) Plan section 20 F2 "SymPy MMS for Laplace/Poisson/
        heat" narrowed to Laplace-only F2 (rule V1 scope per
        ph_num_002.py:92). Poisson / heat covered only by SKIP-path
        contracts.
    15. (Task 12) Plan section 20 "p_obs matches expected within 0.1"
        single-tolerance replaced with per-case tolerance bands per
        2026-04-24 user-revised contract: Case A +/- 0.1, Case B
        +/- 0.25, Case C exact inf.
    16. (Task 12) Plan section 20 "three-level vs four-level Richardson
        extrapolation" not exercised. Rule's shipped path is two-level
        log2 ratio (ph_num_002.py:127); Richardson extrapolation does
        not enter the rule's emitted quantity. Logged as Supplementary
        methodology reference.
    17. (Task 12) Plan section 20 "Oberkampf-Roy Chs 5-6 p_obs
        reproduced" borrowed-credibility claim -> F3 absent-with-
        justification. No live MMS benchmark dataset. Oberkampf-Roy
        and Roy 2005 remain in Mathematical-legitimacy + Supplementary
        calibration context. Per user-revised F3 contract.
    18. (Task 12) Rule docstring "fd4 interior-dominated (periodic):
        ~4 per doubling" is structurally unreachable (Liouville on T^2
        forces periodic harmonics to constants -> saturation). V1 F2
        scope restricted to Case A (harness 2nd-order MMS), Case B
        (rule fd+non-periodic boundary-dominated ~2.5), Case C
        (saturation floor). Rule's shipped behavior correct; no code
        change. Scope-truth observation documented in CITATION.md.

Plan-diffs 1-13 are from Tasks 2, 3, 4, 5, 8, 9 (commits 30baf3e,
0cedc7b, 18312b9, 6800d6f, 1112da3, 26ed3bd).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from external_validation._harness.mms import (
    interior_l2_norm,
    laplacian_fd2_interior,
    mms_observed_order_fd2,
)
from physics_lint import DomainSpec, GridField
from physics_lint.norms import l2_grid
from physics_lint.rules import ph_num_002

# ---------------------------------------------------------------------------
# Acceptance bands (2026-04-24 precheck-calibrated per user's revised contract)
# ---------------------------------------------------------------------------

CASE_A_BAND = (1.9, 2.1)
CASE_B_BAND = (2.3, 2.8)
CASE_A_ASYMPTOTE_BAND = (1.95, 2.05)  # log-log slope across N sweep
CASE_B_ASYMPTOTE_BAND = (2.40, 2.65)  # log-log slope across N sweep

CASE_A_N_PAIRS = ((16, 32), (32, 64), (64, 128), (128, 256))
CASE_B_N_PAIRS = ((16, 32), (32, 64), (64, 128), (128, 256))
CASE_A_LOGLOG_NS = (32, 64, 128, 256)  # skip pre-asymptotic N=16
CASE_B_LOGLOG_NS = (16, 32, 64, 128, 256)


# ---------------------------------------------------------------------------
# Case A fixtures: smooth non-periodic harmonic u on [0,1]^2
# for the interior-only 2nd-order-FD MMS harness-layer anchor.
# ---------------------------------------------------------------------------


def harmonic_expcos_interior(n: int) -> tuple[np.ndarray, float, float]:
    """u = exp(x) cos(y) on [0,1]^2; Delta u = 0 (harmonic)."""
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    u = np.exp(grid_x) * np.cos(grid_y)
    return u, 1.0 / (n - 1), 1.0 / (n - 1)


def harmonic_sin_sinh_interior(n: int) -> tuple[np.ndarray, float, float]:
    """u = sin(pi x) sinh(pi y) on [0,1]^2; Delta u = 0 (harmonic)."""
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    u = np.sin(np.pi * grid_x) * np.sinh(np.pi * grid_y)
    return u, 1.0 / (n - 1), 1.0 / (n - 1)


_CASE_A_FIXTURES: tuple[tuple[str, Callable[[int], tuple[np.ndarray, float, float]]], ...] = (
    ("exp(x)*cos(y)", harmonic_expcos_interior),
    ("sin(pi x)*sinh(pi y)", harmonic_sin_sinh_interior),
)


# ---------------------------------------------------------------------------
# Case B fixtures: smooth non-periodic harmonic u on [0,1]^2, fed to the rule
# on fd+non-periodic boundary-dominated path.
# ---------------------------------------------------------------------------


def _build_non_periodic_spec(n: int) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [n, n],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def harmonic_expcos_rule(n: int) -> GridField:
    u, hx, hy = harmonic_expcos_interior(n)
    return GridField(u, h=(hx, hy), periodic=False, backend="fd")


def harmonic_sin_sinh_rule(n: int) -> GridField:
    u, hx, hy = harmonic_sin_sinh_interior(n)
    return GridField(u, h=(hx, hy), periodic=False, backend="fd")


_CASE_B_FIXTURES: tuple[tuple[str, Callable[[int], GridField]], ...] = (
    ("exp(x)*cos(y)", harmonic_expcos_rule),
    ("sin(pi x)*sinh(pi y)", harmonic_sin_sinh_rule),
)


def _run_rule_fd_nonperiodic(fixture: Callable[[int], GridField], nc: int, nf: int):
    spec = _build_non_periodic_spec(nc)
    return ph_num_002.check(fixture(nc), spec, refined_field=fixture(nf))


# =========================================================================
# Case A: F2 harness-level authoritative p_obs -> 2 observed-order anchor.
#
# Interior-only 2nd-order central-difference Laplacian on smooth harmonic
# u. Pure O(h^2) truncation -> p_obs asymptote 2. Independent of the
# rule's FD4 stencil.
# =========================================================================


@pytest.mark.parametrize("fixture_name,fixture_fn", _CASE_A_FIXTURES)
@pytest.mark.parametrize("nc,nf", [(32, 64), (64, 128), (128, 256)])
def test_case_a_harness_pobs_is_second_order(fixture_name, fixture_fn, nc, nf):
    """Case A (harness-layer authoritative): p_obs in [1.9, 2.1] on
    smooth harmonic at N >= 32->64.

    Pure O(h^2) interior truncation, no boundary-stencil mixing. This is
    the textbook observed-order formula p_obs = log2(r_h / r_{h/2}) at
    its theoretical rate p = 2.
    """
    p_obs, r_c, r_f = mms_observed_order_fd2(fixture_fn, n_coarse=nc, n_fine=nf)
    lo, hi = CASE_A_BAND
    assert lo <= p_obs <= hi, (
        f"Case A harness p_obs={p_obs!r} at ({fixture_name}, N {nc}->{nf}) "
        f"outside [{lo}, {hi}]; expected p_obs -> 2 on 2nd-order interior "
        f"FD. r_coarse={r_c:.3e}, r_fine={r_f:.3e}"
    )


@pytest.mark.parametrize("fixture_name,fixture_fn", _CASE_A_FIXTURES)
def test_case_a_harness_loglog_slope_is_2(fixture_name, fixture_fn):
    """Case A log-log slope across the full asymptotic N sweep is in
    [1.95, 2.05]. Rate asymptotes from above (2.04 at coarse, 2.002 at
    fine) so an average slope fits closer to 2 than any single pair.
    """
    hs: list[float] = []
    rs: list[float] = []
    for n in CASE_A_LOGLOG_NS:
        u, hx, _ = fixture_fn(n)
        lap = laplacian_fd2_interior(u, hx, hx)
        hs.append(hx)
        rs.append(interior_l2_norm(lap, hx, hx))
    log_h = np.log(np.array(hs))
    log_r = np.log(np.array(rs))
    slope, _ = np.polyfit(log_h, log_r, 1)
    lo, hi = CASE_A_ASYMPTOTE_BAND
    assert lo <= slope <= hi, (
        f"Case A log-log slope {slope:.4f} on {fixture_name} across N "
        f"{CASE_A_LOGLOG_NS} outside [{lo}, {hi}]; residuals {rs!r}, "
        f"spacings {hs!r}"
    )


# =========================================================================
# Case B: rule-verdict, fd+non-periodic, boundary-dominated (p_obs ~2.5).
#
# Rule on the FD4 + Dirichlet BC path. The 4N-cell 2nd-order boundary band
# dominates the N^2-cell 4th-order interior at O(h^2.5) in L^2 (see
# CITATION.md proof-sketch step 4). Measured asymptote 2.50.
# =========================================================================


@pytest.mark.parametrize("fixture_name,fixture_fn", _CASE_B_FIXTURES)
@pytest.mark.parametrize("nc,nf", CASE_B_N_PAIRS)
def test_case_b_rule_pobs_is_boundary_dominated(fixture_name, fixture_fn, nc, nf):
    """Case B (rule-verdict, fd+non-periodic): refinement_rate in [2.3,
    2.8] on smooth harmonic fixtures across N pairs 16->32 ... 128->256.

    Rule PASSes (threshold 1.8). Measured asymptote 2.50 matches
    theoretical O(h^2.5) boundary-dominance scaling in 2D.
    """
    result = _run_rule_fd_nonperiodic(fixture_fn, nc, nf)
    assert result.status == "PASS", (
        f"Case B status at ({fixture_name}, N {nc}->{nf}) expected PASS, "
        f"got {result.status!r} (raw_value={result.raw_value!r})"
    )
    assert result.refinement_rate is not None
    lo, hi = CASE_B_BAND
    assert lo <= result.refinement_rate <= hi, (
        f"Case B refinement_rate={result.refinement_rate!r} at "
        f"({fixture_name}, N {nc}->{nf}) outside [{lo}, {hi}]; expected "
        "boundary-dominated asymptote 2.50"
    )


@pytest.mark.parametrize("fixture_name,fixture_fn", _CASE_B_FIXTURES)
def test_case_b_rule_loglog_slope_is_boundary_dominated(fixture_name, fixture_fn):
    """Case B log-log slope across N sweep is in [2.40, 2.65]. Fits the
    boundary-dominated O(h^2.5) scaling on 2D non-periodic grids.
    """
    hs: list[float] = []
    rs: list[float] = []
    for n in CASE_B_LOGLOG_NS:
        field = fixture_fn(n)
        lap = field.laplacian().values()
        rs.append(l2_grid(lap, field.h))
        hs.append(field.h[0])
    log_h = np.log(np.array(hs))
    log_r = np.log(np.array(rs))
    slope, _ = np.polyfit(log_h, log_r, 1)
    lo, hi = CASE_B_ASYMPTOTE_BAND
    assert lo <= slope <= hi, (
        f"Case B log-log slope {slope:.4f} on {fixture_name} across N "
        f"{CASE_B_LOGLOG_NS} outside [{lo}, {hi}]; residuals {rs!r}, "
        f"spacings {hs!r}"
    )


# =========================================================================
# Case C: rule-verdict, saturation floor (rate = inf, no algebraic rate).
#
# Both residuals fall below _SATURATION_FLOOR = 1e-11 (ph_num_002.py:
# 115-121), so the rule clamps to rate=inf and PASSes. Two triggering
# regimes: (i) spectral+periodic on a period-compatible harmonic — by
# Liouville on T^2 this is necessarily a constant, here u=0; (ii) 2nd-
# order FD exact on polynomials of degree <= 2 with the rule's FD4 being
# exact on degree <= 4.
# =========================================================================


def test_case_c_spectral_periodic_constant_saturates():
    """spectral+periodic on u=0 constant: both residuals at roundoff ->
    rate=inf PASS. Liouville implies the only harmonic periodic function
    is a constant, so the anchor tests the only reachable spectral+
    periodic+harmonic case.
    """
    n_coarse, n_fine = 32, 64

    def _const_zero(n: int) -> GridField:
        u = np.zeros((n, n), dtype=np.float64)
        return GridField(
            u,
            h=(2.0 * np.pi / n, 2.0 * np.pi / n),
            periodic=True,
            backend="spectral",
        )

    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [n_coarse, n_coarse],
            "domain": {"x": [0.0, 2.0 * np.pi], "y": [0.0, 2.0 * np.pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )
    result = ph_num_002.check(_const_zero(n_coarse), spec, refined_field=_const_zero(n_fine))
    assert result.status == "PASS", (
        f"Case C spectral+periodic constant expected PASS (saturation), "
        f"got {result.status!r}; raw_value={result.raw_value!r}"
    )
    assert result.refinement_rate == float("inf"), (
        f"Case C spectral+periodic constant expected rate=inf, got {result.refinement_rate!r}"
    )


def test_case_c_fd_nonperiodic_polynomial_saturates():
    """fd+non-periodic on u = x^2 - y^2 harmonic polynomial: FD4
    stencil is exact on polynomials of degree <= 4, so both residuals
    are at roundoff (~1e-12) and fall below _SATURATION_FLOOR -> rate=inf
    PASS.
    """
    n_coarse, n_fine = 32, 64

    def _xsq_ysq(n: int) -> GridField:
        x = np.linspace(0.0, 1.0, n)
        y = np.linspace(0.0, 1.0, n)
        grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
        u = grid_x * grid_x - grid_y * grid_y
        return GridField(
            u,
            h=(1.0 / (n - 1), 1.0 / (n - 1)),
            periodic=False,
            backend="fd",
        )

    spec = _build_non_periodic_spec(n_coarse)
    result = ph_num_002.check(_xsq_ysq(n_coarse), spec, refined_field=_xsq_ysq(n_fine))
    assert result.status == "PASS", (
        f"Case C fd+non-periodic polynomial expected PASS (saturation), "
        f"got {result.status!r}; raw_value={result.raw_value!r}"
    )
    assert result.refinement_rate == float("inf"), (
        f"Case C fd+non-periodic polynomial expected rate=inf, got {result.refinement_rate!r}"
    )


# =========================================================================
# SKIP-path contracts: V1 scope boundary (Laplace-only).
# =========================================================================


def test_skipped_on_poisson_pde():
    """Rule is Laplace-only per ph_num_002.py:92; Poisson SKIPs with the
    source-subtraction justification. Anchor records this as a V1 scope
    boundary contract — not a reproduction claim.
    """
    n = 32

    def _poisson_field(n: int) -> GridField:
        x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        y = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
        u = np.sin(grid_x) * np.sin(grid_y)
        return GridField(
            u,
            h=(2.0 * np.pi / n, 2.0 * np.pi / n),
            periodic=True,
            backend="spectral",
        )

    spec = DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [n, n],
            "domain": {"x": [0.0, 2.0 * np.pi], "y": [0.0, 2.0 * np.pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )
    result = ph_num_002.check(_poisson_field(n), spec, refined_field=_poisson_field(n * 2))
    assert result.status == "SKIPPED"
    assert result.reason is not None and "laplace" in result.reason.lower()
    assert "poisson" in result.reason.lower()


def test_skipped_on_heat_pde():
    """Rule is Laplace-only; heat SKIPs because field.laplacian() would
    differentiate the time axis. Anchor records the V1 scope-boundary
    contract.
    """
    n, nt = 16, 11

    def _heat_field(n: int, nt: int) -> GridField:
        x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        y = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        t = np.linspace(0.0, 0.5, nt)
        grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
        u = np.stack(
            [np.cos(grid_x) * np.cos(grid_y) * np.exp(-2.0 * 0.01 * ti) for ti in t],
            axis=-1,
        )
        return GridField(
            u,
            h=(2.0 * np.pi / n, 2.0 * np.pi / n, 0.5 / (nt - 1)),
            periodic=True,
            backend="spectral",
        )

    spec = DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [n, n, nt],
            "domain": {
                "x": [0.0, 2.0 * np.pi],
                "y": [0.0, 2.0 * np.pi],
                "t": [0.0, 0.5],
            },
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "diffusivity": 0.01,
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )
    result = ph_num_002.check(_heat_field(n, nt), spec, refined_field=_heat_field(n * 2, nt))
    assert result.status == "SKIPPED"
    assert result.reason is not None and "heat" in result.reason.lower()
