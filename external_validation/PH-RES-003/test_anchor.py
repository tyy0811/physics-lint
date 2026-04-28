"""PH-RES-003 external-validation anchor - spectral-vs-FD residual on periodic grids.

The rule PH-RES-003 (src/physics_lint/rules/ph_res_003.py) computes the
Laplacian of a periodic field via two backends (spectral FFT-based and
4th-order central FD), then emits the max relative discrepancy
max|Lap_spectral - Lap_FD| / max|Lap_spectral|. PASS threshold is 0.01.

Structural-equivalence anchor (Function 1):
    - Trefethen 2000 Chapters 3-4 (section-level per
      external_validation/_harness/TEXTBOOK_AVAILABILITY.md WARN): Fourier
      spectral methods on analytic periodic functions have
      super-algebraic (exponential-in-N) accuracy. Theorem number pending
      local copy per plan section 6.4.
    - Fornberg 1988 (via LeVeque 2007, section-level WARN): the 4th-order
      central FD stencil has truncation error O(h^4) on periodic grids.
    - Therefore spectral residual decays as O(exp(-cN)) and FD residual
      as O(N^-4), with the rule's emitted quantity bounded by the larger
      (FD) term for moderate N.

Correctness-fixture layer (Function 2), three sub-layers:

    Layer 1 (spectral convergence) - tightened per plan-diff 3. Uses
        SPECTRAL_NS = [8, 10, 12, 14, 16] (pre-floor grid) with documented
        float64 noise floor FLOAT64_FLOOR = 1e-13. Fits log-linear
        log(err) vs N on points above floor; if four or more points remain
        above floor, asserts slope < 0 and R^2 > 0.99. If fewer than four
        remain, reports observed rapid convergence-to-floor behavior
        instead of forcing an R^2 criterion (first-point-large +
        last-point-at-or-below-floor). Empirical on the 2D exp(sin(x) +
        sin(y)) fixture at 2026-04-24: all 5 points above floor, slope
        approx -1.27, R^2 approx 0.995.

    Layer 2 (FD polynomial rate) - unchanged from plan. FD_NS = [16, 32,
        64] on the same periodic fixture; log-log slope in [3.6, 4.4],
        R^2 > 0.99. Empirical at 2026-04-24: residuals 1.03e-1, 7.26e-3,
        4.68e-4 -> slope 3.89, R^2 0.9999.

    Layer 3 (rule verdict) - unchanged from plan. Rule PASSes at
        RULE_NS = [16, 32, 64] with max ratio < 0.01 at each N. Empirical
        at 2026-04-24: 6.97e-3, 4.91e-4, 3.17e-5.

F3 (borrowed-credibility) is absent with justification - see CITATION.md.
Plan section 11 Step 3 + Task 0 Step 5 F3-hunt outcome (docs/audits/
2026-04-22-f3-hunt-results.md line 123-168) confirm: Trefethen's canonical
exp(sin x) demonstration is a plot, not a tabulated reproduction target.
Canuto-Hussaini-Quarteroni-Zang 2006 section 2.3 convergence curves and
Trefethen 2000 Program 5 plot land in Supplementary calibration context
with explicit "curve-shape framing, not reproduction" flags.

Plan-diffs logged with the commit (plan-vs-committed-state drift, plan
section 7.4):
    3. SPECTRAL_NS widened from plan section 11 Step 3's N = {16, 32, 64}
       to [8, 10, 12, 14, 16] for Layer 1. Reason: spectral residual on
       analytic periodic exp(sin x + sin y) saturates at float64
       machine precision (approx 1e-13) by N approx 32, leaving only the
       N=16 point above floor. A three-point log-linear fit with two
       noise-floor points would be mathematically cosmetic rather than
       evidential. Plan-level re-audit (2026-04-24) approved Path C
       with this tightened Layer 1 criterion; FD_NS and RULE_NS remain
       at {16, 32, 64} per plan.
    4. Fixture dimension: plan section 11 pre-execution enumerate-the-
       splits audit states "V1: 1D only". In practice
       physics_lint.spec.DomainSpec enforces grid_shape tuple length
       in [2, 3] (spec.py line 127), so 1D is not constructible through
       the rule's DomainSpec contract. Fixture uses the 2D analog
       u(x,y) = exp(sin(x) + sin(y)) on [0, 2 pi]^2 periodic. The 2D
       fixture preserves the analytic-periodic property the plan's
       F1 anchor presupposes; the test's reported spectral and FD
       residuals are dim-agnostic at the Laplacian level.
    5. Audit file docs/audits/2026-04-22-f3-hunt-results.md lines 129
       and 160-161 carry the phrase "Trefethen 2000 Chs 3-4 Thm 4" as
       a provisional Function 1 anchor. Plan section 11 Step 1 mandates
       chapter-level framing without tight theorem-number citations
       (enforced by scripts/check_theorem_number_framing.py against
       the WARN row in TEXTBOOK_AVAILABILITY.md). This test's
       CITATION.md drops "Thm 4" and uses "Chapters 3-4, theorem
       number pending local copy" phrasing per section 6.4.

Plan-diffs 1 and 2 are from Task 2 (PH-RES-002) commit 30baf3e.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics_lint import DomainSpec
from physics_lint.field import GridField
from physics_lint.rules import ph_res_003

SPECTRAL_NS = [8, 10, 12, 14, 16]
FD_NS = [16, 32, 64]
RULE_NS = [16, 32, 64]
FLOAT64_FLOOR = 1e-13


def _fixture_2d(n: int) -> tuple[np.ndarray, float, np.ndarray]:
    """u(x, y) = exp(sin(x) + sin(y)) on [0, 2 pi]^2 periodic, grid N x N.

    Returns (u, h, lap_exact). The closed-form Laplacian is
    Delta u = [(cos^2 x - sin x) + (cos^2 y - sin y)] * u.
    """
    xs = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    h = 2.0 * np.pi / n
    x_grid, y_grid = np.meshgrid(xs, xs, indexing="ij")
    u = np.exp(np.sin(x_grid) + np.sin(y_grid))
    lap_exact = (
        (np.cos(x_grid) ** 2 - np.sin(x_grid)) + (np.cos(y_grid) ** 2 - np.sin(y_grid))
    ) * u
    return u, h, lap_exact


def _spectral_residual(n: int) -> float:
    """max |Lap_spectral - Lap_exact| on the periodic fixture at grid N."""
    u, h, lap_exact = _fixture_2d(n)
    lap = GridField(u, h=(h, h), periodic=True, backend="spectral").laplacian().values()
    return float(np.max(np.abs(lap - lap_exact)))


def _fd_residual(n: int) -> float:
    """max |Lap_FD - Lap_exact| on the periodic fixture at grid N."""
    u, h, lap_exact = _fixture_2d(n)
    lap = GridField(u, h=(h, h), periodic=True, backend="fd").laplacian().values()
    return float(np.max(np.abs(lap - lap_exact)))


def _rule_ratio(n: int) -> float:
    """Invoke PH-RES-003 on the periodic fixture at grid N; return raw_value."""
    u, h, _ = _fixture_2d(n)
    field = GridField(u, h=(h, h), periodic=True)
    spec = DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [n, n],
            "domain": {"x": [0.0, 2.0 * np.pi], "y": [0.0, 2.0 * np.pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "field": {"type": "grid", "backend": "auto", "dump_path": "p.npz"},
        }
    )
    result = ph_res_003.check(field, spec)
    assert result.status != "SKIPPED", (
        f"PH-RES-003 returned SKIPPED at n={n}; reason={result.reason!r}"
    )
    assert result.raw_value is not None, (
        f"PH-RES-003 returned raw_value=None at n={n} (status={result.status!r})"
    )
    return float(result.raw_value)


# =========================================================================
# Layer 1: spectral convergence on the pre-floor grid (tightened per
# plan-diff 3: SPECTRAL_NS = [8, 10, 12, 14, 16]; noise-floor clip; R^2
# threshold applies only if 4+ points remain above floor, else rapid-
# collapse characterization).
# =========================================================================


def test_layer1_spectral_residual_above_floor_has_r2_above_0_99():
    """Trefethen-spectral-accuracy reproduction: exp-decay fit on pre-floor N.

    Trefethen 2000 Chapters 3-4 (section-level per
    ../_harness/TEXTBOOK_AVAILABILITY.md WARN) establishes super-algebraic
    spectral accuracy for analytic periodic functions; on exp(sin(x) +
    sin(y)) the residual decays as O(exp(-cN)) with c approx 1.27.

    If at least four of SPECTRAL_NS remain above FLOAT64_FLOOR, fit
    log(err) vs N linearly and assert slope < 0, R^2 > 0.99. Otherwise
    enter the rapid-collapse-to-floor pathway (separate test).
    """
    residuals = [_spectral_residual(n) for n in SPECTRAL_NS]
    above = [(n, r) for n, r in zip(SPECTRAL_NS, residuals, strict=True) if r > FLOAT64_FLOOR]
    if len(above) < 4:
        pytest.skip(
            f"only {len(above)} of {len(SPECTRAL_NS)} spectral residuals above "
            f"FLOAT64_FLOOR={FLOAT64_FLOOR:.0e}; rapid-collapse pathway handled by "
            "test_layer1_spectral_residual_rapid_collapse_characterization."
        )
    ns_fit = np.array([n for n, _ in above], dtype=float)
    log_e = np.log(np.array([r for _, r in above]))
    slope, intercept = np.polyfit(ns_fit, log_e, 1)
    assert slope < 0, (
        f"spectral residual log-linear slope={slope:.4f} should be < 0 (decay), "
        f"got increase; residuals={residuals!r}"
    )
    predicted = slope * ns_fit + intercept
    ss_res = float(np.sum((log_e - predicted) ** 2))
    ss_tot = float(np.sum((log_e - log_e.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot
    assert r_squared > 0.99, (
        f"spectral exp-decay fit R^2={r_squared:.6f} below 0.99 on "
        f"SPECTRAL_NS={SPECTRAL_NS!r} above-floor subset; residuals={residuals!r}"
    )


def test_layer1_spectral_residual_rapid_collapse_characterization():
    """Fallback pathway: if fewer than 4 points above floor, assert the
    physical pattern (large first point; last point at or below floor).

    Reports the observed rapid convergence-to-floor behavior when the
    R^2 criterion is not meaningful. Plan-diff 3 fallback.
    """
    residuals = [_spectral_residual(n) for n in SPECTRAL_NS]
    above = [(n, r) for n, r in zip(SPECTRAL_NS, residuals, strict=True) if r > FLOAT64_FLOOR]
    if len(above) >= 4:
        pytest.skip(
            f"{len(above)} of {len(SPECTRAL_NS)} spectral residuals above floor; "
            "R^2 pathway handled by test_layer1_spectral_residual_above_floor_has_r2_above_0_99."
        )
    # At the smallest N, spectral residual should still be macroscopically
    # large (> 1e-3) - otherwise SPECTRAL_NS was chosen wrong. At the
    # largest N, it should be at or below the documented float64 floor.
    assert residuals[0] > 1e-3, (
        f"residual at N={SPECTRAL_NS[0]} = {residuals[0]:.3e} is too small to "
        "characterize rapid collapse; pick smaller SPECTRAL_NS[0]."
    )
    assert residuals[-1] <= 5 * FLOAT64_FLOOR, (
        f"residual at N={SPECTRAL_NS[-1]} = {residuals[-1]:.3e} has not reached "
        f"5x float64 floor ({5 * FLOAT64_FLOOR:.0e}); spectral didn't collapse."
    )


def test_layer1_spectral_residual_monotonically_decreases_in_exp_regime():
    """On SPECTRAL_NS, residual decreases until it hits float64 floor."""
    residuals = [_spectral_residual(n) for n in SPECTRAL_NS]
    for k in range(len(residuals) - 1):
        # Allow equality at/below the noise floor (residual may plateau
        # or wobble by factors of 2-3 around float64 precision).
        if residuals[k] <= 5 * FLOAT64_FLOOR:
            continue
        assert residuals[k + 1] < residuals[k], (
            f"spectral residual non-monotone above floor: "
            f"err[N={SPECTRAL_NS[k + 1]}]={residuals[k + 1]:.3e} "
            f">= err[N={SPECTRAL_NS[k]}]={residuals[k]:.3e}"
        )


# =========================================================================
# Layer 2: FD polynomial rate (unchanged from plan: FD_NS = [16, 32, 64],
# slope approx 4, R^2 > 0.99).
# =========================================================================


def test_layer2_fd_residual_polynomial_order_4():
    """LeVeque/Fornberg-FD4 reproduction: log-log slope approx 4 on FD_NS."""
    residuals = [_fd_residual(n) for n in FD_NS]
    hs = [2.0 * np.pi / n for n in FD_NS]
    log_h = np.log(np.array(hs))
    log_e = np.log(np.array(residuals))
    slope, intercept = np.polyfit(log_h, log_e, 1)
    assert 3.6 <= slope <= 4.4, (
        f"FD log-log slope={slope:.4f} outside [3.6, 4.4] on FD_NS={FD_NS!r}; "
        f"residuals={residuals!r}"
    )
    predicted = slope * log_h + intercept
    ss_res = float(np.sum((log_e - predicted) ** 2))
    ss_tot = float(np.sum((log_e - log_e.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot
    assert r_squared > 0.99, f"FD poly-fit R^2={r_squared:.6f} below 0.99 on FD_NS={FD_NS!r}"


def test_layer2_fd_residual_monotonically_decreases():
    residuals = [_fd_residual(n) for n in FD_NS]
    for k in range(len(residuals) - 1):
        assert residuals[k + 1] < residuals[k], (
            f"FD residual non-monotone: "
            f"err[N={FD_NS[k + 1]}]={residuals[k + 1]:.3e} "
            f">= err[N={FD_NS[k]}]={residuals[k]:.3e}"
        )


# =========================================================================
# Layer 3: rule verdict sanity (unchanged from plan: rule PASSes at
# RULE_NS = [16, 32, 64] with max ratio < 0.01).
# =========================================================================


def test_layer3_rule_passes_at_every_refinement_level():
    """Rule's PASS/WARN boundary at max ratio 0.01 is cleared at every N.

    If any N flips to WARN on the analytic-periodic exp(sin x + sin y)
    fixture, either the spectral backend changed or the FD4 interior
    stencil changed; audit GridField before softening the fixture.
    """
    for n in RULE_NS:
        ratio = _rule_ratio(n)
        assert ratio < 0.01, (
            f"rule raw_value at N={n} = {ratio:.3e} >= 0.01 threshold; "
            "rule would WARN on analytic-periodic exp(sin x + sin y). "
            "Root-cause in GridField before adjusting threshold or fixture."
        )


def test_rule_skipped_on_non_periodic_domain():
    """PH-RES-003 emits SKIPPED when handed a non-periodic spec.

    Category 8 semantic-compatibility check: the rule's docstring states
    "applies only to periodic domains" (ph_res_003.py line 22-30). Any
    non-periodic spec must SKIP cleanly, not silently PASS or crash.
    """
    n = 16
    u, h, _ = _fixture_2d(n)
    field = GridField(u, h=(h, h), periodic=False)
    spec = DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [n, n],
            "domain": {"x": [0.0, 2.0 * np.pi], "y": [0.0, 2.0 * np.pi]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    result = ph_res_003.check(field, spec)
    assert result.status == "SKIPPED", (
        f"expected SKIPPED on non-periodic spec, got status={result.status!r}"
    )
    assert result.reason is not None, "SKIPPED result must carry a reason string"
