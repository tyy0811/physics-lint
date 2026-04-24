"""PH-SYM-004 external-validation anchor - translation equivariance (V1 stub).

Task 7 of the complete-v1.0 plan. Applies the V1-stub CRITICAL three-layer
pattern (2026-04-24 `feedback_critical_rule_stub_three_layer_contract.md`,
Task 5 precedent): the production rule `ph_sym_004.py` is a V1 structural
stub that always emits SKIPPED once past its declared-symmetry + periodicity
gates (ph_sym_004.py:36-52). True translation equivariance is a model
property -- comparing f(roll(x)) against roll(f(x)) on a live callable --
and requires adapter-mode plumbing deferred to V1.1.

External validation separates:

    F1   Mathematical-legitimacy anchor: Kondor-Trivedi 2018
         compact-group equivariance theorem (arXiv:1802.03690) + Li et
         al. 2021 FNO section 2 convolution theorem (arXiv:2010.08895).
         Five-step structural proof-sketch with explicit assumption
         statement: periodic domain, grid-aligned shifts, same input/
         output grid, consistent translation action, no boundary
         artifacts unless deliberately tested.

    F2   Harness-level correctness fixture (authoritative). Controlled
         operators exercised via external_validation/_harness/
         symmetry.py shift_commutation_error:

         - identity_op: trivially equivariant (error = 0 exactly)
         - circular_convolution_1d / 2d: equivariant by convolution
           theorem (error ~3e-16 in float64)
         - fourier_multiplier_1d / 2d: equivariant by Fourier
           multiplier theorem (error ~4e-16 in float64)
         - coord_dependent_multiply_1d / 2d: NON-equivariant negative
           control (error 9e-02 to 9e-01)

         Measured across 100 random 2D trials: equivariant max error
         <= 3.75e-16; non-equivariant min error >= 9.17e-02.

    F3   Borrowed-credibility: absent-with-justification per plan §15
         rationale + 2026-04-24 user-revised F3 contract. No CI-
         executable reproduction target exists for the rule's emitted
         quantity (SKIPPED in V1). Helwig 2023 §2.2 Lemma 3.1 moved to
         Supplementary calibration context with "theoretical framing,
         not reproduction" flag. Cohen-Welling 2016 G-CNN also in
         Supplementary as pedagogical framing.

    RVC  Rule-verdict contract: exercises the rule's V1 SKIP path on
         all three code branches:
         (a) no translation_x/y declared -> SKIP "no translation_x or
             translation_y declared"
         (b) non-periodic + symmetry declared -> SKIP "periodic-only
             in V1"
         (c) periodic + symmetry declared + past gates -> SKIP "V1
             structural stub"
         plus invariance check: rule SKIPs regardless of whether the
         input is equivariant or non-equivariant (V1 stub doesn't
         measure).

Wording discipline (CITATION.md + README + tests):
    "PH-SYM-004 validates the mathematical and harness-level translation-
    equivariance contract for controlled operators. The v1.0 production
    rule validates only its implemented rule-verdict behavior."

Plan-diffs logged (plan-vs-committed-state drift, section 7.4):
    23. (Task 7) Plan §15 step 3 "random FNO-layer + random input +
        random grid-shift fixture; assert commutation error < 1e-5" ->
        controlled-operator harness (identity + circular conv + Fourier
        multiplier + coord-dependent-mul negative control) per 2026-04-24
        user-revised F2 contract. Avoids neuraloperator / pytorch_fno
        dependency; same mathematical property with simpler operators.
    24. (Task 7) Plan §15 step 3 tolerance "< 1e-5" -> "<= 1e-14" across
        100 random 2D trials on controlled harness operators. Measured
        max 3.75e-16 in float64; 1e-14 gives ~30x safety over observed.
    25. (Task 7) CRITICAL three-layer pattern applied (Task 5 precedent):
        rule-verdict contract layer added to verify all three V1-stub
        SKIP branches matching ph_sym_004.py:36-52 reason strings.
    26. (Task 7) Plan §15 F3 already-absent status reinforced per 2026-
        04-24 user-revised F3 contract: Helwig 2023 §2.2 Lemma 3.1 moved
        to Supplementary calibration context with explicit "theoretical
        framing, not reproduction" flag.

Plan-diffs 1-22 are from Tasks 2, 3, 4, 5, 8, 9, 10, 12 (commits 30baf3e,
0cedc7b, 18312b9, 6800d6f, 1112da3, 26ed3bd, 84c7163, 87e8a3e).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from external_validation._harness.symmetry import (
    circular_convolution_1d,
    circular_convolution_2d,
    coord_dependent_multiply_1d,
    coord_dependent_multiply_2d,
    fourier_multiplier_1d,
    fourier_multiplier_2d,
    identity_op,
    shift_commutation_error,
)
from physics_lint import DomainSpec
from physics_lint.field import GridField
from physics_lint.rules import ph_sym_004

# ---------------------------------------------------------------------------
# Acceptance bands (2026-04-24 precheck-calibrated per user's revised contract)
# ---------------------------------------------------------------------------

EQUIVARIANT_TOL = 1e-14
NON_EQUIVARIANT_FLOOR = 0.05
N_RANDOM_TRIALS = 100


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _x1(n: int, seed: int = 0) -> torch.Tensor:
    return torch.from_numpy(_rng(seed).standard_normal(n)).to(torch.float64)


def _x2(nx: int, ny: int, seed: int = 0) -> torch.Tensor:
    return torch.from_numpy(_rng(seed).standard_normal((nx, ny))).to(torch.float64)


def _kernel(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64)


# =========================================================================
# Case A: F2 harness-level equivariant operators.
# =========================================================================


@pytest.mark.parametrize("shift", [1, 5, 17, 31])
def test_case_a_1d_identity_is_exactly_equivariant(shift):
    """identity_op commutes with any shift at float precision (error = 0)."""
    x = _x1(64)
    err = shift_commutation_error(identity_op, x, shifts=shift, dims=-1)
    assert err == 0.0, (
        f"1D identity shift-commutation error at shift={shift}: got {err!r}, "
        f"expected exactly 0 for the identity operator"
    )


@pytest.mark.parametrize("shift", [1, 5, 17, 31])
def test_case_a_1d_circular_convolution_is_equivariant(shift):
    """Circular conv with a fixed kernel commutes with periodic shift
    to within float64 roundoff (~3e-16)."""
    x = _x1(64)
    kernel = _kernel((7,))
    err = shift_commutation_error(circular_convolution_1d(kernel), x, shifts=shift, dims=-1)
    assert err <= EQUIVARIANT_TOL, (
        f"1D circular convolution shift-commutation error at shift={shift}: "
        f"got {err!r}, expected <= {EQUIVARIANT_TOL:.0e}"
    )


@pytest.mark.parametrize("shift", [1, 5, 17, 31])
def test_case_a_1d_fourier_multiplier_is_equivariant(shift):
    """Fourier-multiplier operator commutes with periodic shift to within
    float64 roundoff. Equivariance is the shift theorem:
    F(T_s u)[k] = e^(-2 pi i k s / N) F(u)[k]; multiplication by a fixed
    m(k) commutes with phase rotation."""
    x = _x1(64)
    err = shift_commutation_error(fourier_multiplier_1d(64, seed=42), x, shifts=shift, dims=-1)
    assert err <= EQUIVARIANT_TOL, (
        f"1D Fourier multiplier shift-commutation error at shift={shift}: "
        f"got {err!r}, expected <= {EQUIVARIANT_TOL:.0e}"
    )


@pytest.mark.parametrize("shifts", [(1, 0), (0, 3), (5, 7), (11, 13)])
def test_case_a_2d_identity_is_exactly_equivariant(shifts):
    x = _x2(32, 32)
    err = shift_commutation_error(identity_op, x, shifts=shifts, dims=(-2, -1))
    assert err == 0.0, (
        f"2D identity shift-commutation error at shifts={shifts}: got {err!r}, expected exactly 0"
    )


@pytest.mark.parametrize("shifts", [(1, 0), (0, 3), (5, 7), (11, 13)])
def test_case_a_2d_circular_convolution_is_equivariant(shifts):
    x = _x2(32, 32)
    kernel = _kernel((5, 5))
    err = shift_commutation_error(circular_convolution_2d(kernel), x, shifts=shifts, dims=(-2, -1))
    assert err <= EQUIVARIANT_TOL, (
        f"2D circular convolution shift-commutation error at shifts={shifts}: "
        f"got {err!r}, expected <= {EQUIVARIANT_TOL:.0e}"
    )


@pytest.mark.parametrize("shifts", [(1, 0), (0, 3), (5, 7), (11, 13)])
def test_case_a_2d_fourier_multiplier_is_equivariant(shifts):
    x = _x2(32, 32)
    err = shift_commutation_error(
        fourier_multiplier_2d(32, 32, seed=42), x, shifts=shifts, dims=(-2, -1)
    )
    assert err <= EQUIVARIANT_TOL, (
        f"2D Fourier multiplier shift-commutation error at shifts={shifts}: "
        f"got {err!r}, expected <= {EQUIVARIANT_TOL:.0e}"
    )


def test_case_a_2d_stability_across_100_random_trials():
    """100-trial stability sweep: max shift-commutation error across
    random (input, shift, kernel, multiplier) trials must stay below
    1e-14 for every equivariant operator.
    """
    nx = ny = 32
    max_id = 0.0
    max_cv = 0.0
    max_fm = 0.0
    for trial in range(N_RANDOM_TRIALS):
        rng = _rng(trial)
        x = torch.from_numpy(rng.standard_normal((nx, ny))).to(torch.float64)
        sx = int(rng.integers(1, nx))
        sy = int(rng.integers(1, ny))
        kernel = _kernel((5, 5), seed=trial)
        max_id = max(
            max_id,
            shift_commutation_error(identity_op, x, shifts=(sx, sy), dims=(-2, -1)),
        )
        max_cv = max(
            max_cv,
            shift_commutation_error(
                circular_convolution_2d(kernel), x, shifts=(sx, sy), dims=(-2, -1)
            ),
        )
        max_fm = max(
            max_fm,
            shift_commutation_error(
                fourier_multiplier_2d(nx, ny, seed=trial),
                x,
                shifts=(sx, sy),
                dims=(-2, -1),
            ),
        )
    assert max_id <= EQUIVARIANT_TOL, (
        f"2D identity max error across {N_RANDOM_TRIALS} trials: {max_id!r} > {EQUIVARIANT_TOL:.0e}"
    )
    assert max_cv <= EQUIVARIANT_TOL, (
        f"2D circular_convolution max error across {N_RANDOM_TRIALS} trials: "
        f"{max_cv!r} > {EQUIVARIANT_TOL:.0e}"
    )
    assert max_fm <= EQUIVARIANT_TOL, (
        f"2D fourier_multiplier max error across {N_RANDOM_TRIALS} trials: "
        f"{max_fm!r} > {EQUIVARIANT_TOL:.0e}"
    )


# =========================================================================
# Case B: F2 harness-level NON-equivariant negative control.
#
# Required by 2026-04-24 user-revised Task 7 contract: "also include a
# deliberately non-equivariant operator: coordinate-dependent
# multiplication or boundary-sensitive operator." Validates the
# measurement framework by showing shift_commutation_error actually
# distinguishes equivariant from non-equivariant operators.
# =========================================================================


@pytest.mark.parametrize("shift", [5, 17, 31])
def test_case_b_1d_coord_dep_mul_breaks_equivariance(shift):
    """Coordinate-dependent multiplication u(x) -> w(x) u(x) is NOT
    translation-equivariant because the mask w is fixed in space while
    the shift moves the signal under it.
    """
    x = _x1(64)
    err = shift_commutation_error(coord_dependent_multiply_1d(64), x, shifts=shift, dims=-1)
    assert err > NON_EQUIVARIANT_FLOOR, (
        f"1D coord-dependent multiplication at shift={shift}: error "
        f"{err!r} <= {NON_EQUIVARIANT_FLOOR}; non-equivariant control "
        f"should register substantial commutation error"
    )


@pytest.mark.parametrize("shifts", [(5, 7), (11, 13), (3, 17)])
def test_case_b_2d_coord_dep_mul_breaks_equivariance(shifts):
    x = _x2(32, 32)
    err = shift_commutation_error(
        coord_dependent_multiply_2d(32, 32), x, shifts=shifts, dims=(-2, -1)
    )
    assert err > NON_EQUIVARIANT_FLOOR, (
        f"2D coord-dependent multiplication at shifts={shifts}: error "
        f"{err!r} <= {NON_EQUIVARIANT_FLOOR}"
    )


def test_case_b_2d_non_equivariant_stability_across_100_trials():
    """100-trial stability sweep on the negative control: MIN error must
    stay above 0.05 (i.e. every trial must show substantial non-
    equivariance; no accidental near-commutation).
    """
    nx = ny = 32
    min_err = math.inf
    for trial in range(N_RANDOM_TRIALS):
        rng = _rng(trial)
        x = torch.from_numpy(rng.standard_normal((nx, ny))).to(torch.float64)
        sx = int(rng.integers(1, nx))
        sy = int(rng.integers(1, ny))
        err = shift_commutation_error(
            coord_dependent_multiply_2d(nx, ny),
            x,
            shifts=(sx, sy),
            dims=(-2, -1),
        )
        min_err = min(min_err, err)
    assert min_err > NON_EQUIVARIANT_FLOOR, (
        f"Negative-control min error across {N_RANDOM_TRIALS} trials: "
        f"{min_err!r} <= {NON_EQUIVARIANT_FLOOR}; non-equivariant floor "
        f"breached -- measurement framework would fail to distinguish "
        f"equivariant from non-equivariant in some trials"
    )


# =========================================================================
# Rule-verdict contract: V1-stub SKIP on all three code paths.
# =========================================================================


def _field_for_spec(spec: DomainSpec) -> GridField:
    """Minimal GridField matching a given spec. Rule doesn't look at the
    field's values in any SKIP path; shape + backend + periodic match
    the spec's declared grid.
    """
    nx = spec.grid_shape[0]
    ny = spec.grid_shape[1] if len(spec.grid_shape) > 1 else nx
    domain_x = spec.domain.x
    domain_y = spec.domain.y
    hx = (domain_x[1] - domain_x[0]) / (nx if spec.periodic else nx - 1)
    hy = (domain_y[1] - domain_y[0]) / (ny if spec.periodic else ny - 1)
    return GridField(
        np.zeros((nx, ny)),
        h=(hx, hy),
        periodic=spec.periodic,
        backend="fd" if not spec.periodic else "spectral",
    )


def test_rvc_skips_when_no_translation_symmetry_declared():
    """Rule SKIPs when neither translation_x nor translation_y is in
    spec.symmetries (ph_sym_004.py:37-40).
    """
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [32, 32],
            "domain": {"x": [0.0, 2 * math.pi], "y": [0.0, 2 * math.pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "symmetries": {"declared": []},
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )
    result = ph_sym_004.check(_field_for_spec(spec), spec)
    assert result.status == "SKIPPED"
    assert result.reason is not None
    assert "no translation_x or translation_y declared" in result.reason


def test_rvc_skips_when_non_periodic():
    """Rule SKIPs when translation symmetry is declared but domain is not
    periodic (ph_sym_004.py:41-45).
    """
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [32, 32],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "symmetries": {"declared": ["translation_x"]},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    result = ph_sym_004.check(_field_for_spec(spec), spec)
    assert result.status == "SKIPPED"
    assert result.reason is not None
    assert "periodic-only in V1" in result.reason


def test_rvc_skips_with_v1_stub_reason_on_valid_spec():
    """Rule SKIPs with the V1-structural-stub reason when symmetry is
    declared AND domain is periodic (the third SKIP branch at
    ph_sym_004.py:46-52). This is the path a V1.1 adapter-mode
    implementation will eventually replace with a numerical equivariance
    check; the SKIP reason must point forward to that work.
    """
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [32, 32],
            "domain": {"x": [0.0, 2 * math.pi], "y": [0.0, 2 * math.pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "symmetries": {"declared": ["translation_x", "translation_y"]},
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )
    result = ph_sym_004.check(_field_for_spec(spec), spec)
    assert result.status == "SKIPPED"
    assert result.reason is not None
    assert "V1 structural stub" in result.reason


def test_rvc_rule_skips_regardless_of_input_equivariance():
    """The V1 stub does not compute shift_commutation_error, so the rule's
    SKIP verdict is invariant to whether the input field is genuinely
    equivariant (e.g., a constant) or non-equivariant (e.g., a
    coord-dependent bump). This test asserts both cases SKIP with the
    same V1-stub reason -- the rule does not accidentally start
    measuring something in V1.
    """
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [32, 32],
            "domain": {"x": [0.0, 2 * math.pi], "y": [0.0, 2 * math.pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "symmetries": {"declared": ["translation_x"]},
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )
    # Equivariant input (constant)
    const_field = GridField(
        np.ones((32, 32)),
        h=(2 * math.pi / 32, 2 * math.pi / 32),
        periodic=True,
        backend="spectral",
    )
    # Non-equivariant input (position-dependent bump)
    bump_field = GridField(
        np.exp(
            -(
                (np.linspace(0, 1, 32)[:, None] - 0.5) ** 2
                + (np.linspace(0, 1, 32)[None, :] - 0.5) ** 2
            )
            / 0.05
        ),
        h=(2 * math.pi / 32, 2 * math.pi / 32),
        periodic=True,
        backend="spectral",
    )
    r_const = ph_sym_004.check(const_field, spec)
    r_bump = ph_sym_004.check(bump_field, spec)
    assert r_const.status == "SKIPPED"
    assert r_bump.status == "SKIPPED"
    assert r_const.reason is not None and "V1 structural stub" in r_const.reason
    assert r_bump.reason is not None and "V1 structural stub" in r_bump.reason
