"""Bochner L^2(0,T; H^-1) norm tests — trapezoidal rule over endpoint-inclusive
time samples (the convention _compute_h_from_spec produces).
"""

import numpy as np

from physics_lint.norms import (
    bochner_l2_fallback,
    bochner_l2_h_minus_one,
    h_minus_one_spectral,
    l2_grid,
)


def test_bochner_constant_in_time():
    """Endpoint-inclusive constant-in-t residual integrates exactly to T * H^-1^2."""
    n_spatial = 64
    total_time = 1.0
    n_steps = 32
    dt = total_time / (n_steps - 1)  # endpoint-inclusive: n_steps samples span [0, T]
    h = (1.0 / n_spatial, 1.0 / n_spatial)
    x = np.linspace(0.0, 1.0, n_spatial, endpoint=False)
    y = np.linspace(0.0, 1.0, n_spatial, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    r_spatial = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    r_series = np.stack([r_spatial] * n_steps, axis=-1)

    h_sq_per_slice = h_minus_one_spectral(r_spatial, h) ** 2
    expected_sq = total_time * h_sq_per_slice
    bochner = bochner_l2_h_minus_one(r_series, spatial_h=h, dt=dt)
    # Trapezoidal weights sum to (n_steps - 1) * dt = total_time exactly, so
    # the constant-in-t integral is exact to floating-point.
    assert abs(bochner**2 - expected_sq) < 1e-12


def test_bochner_overintegration_regression():
    """Regression guard for the midpoint-vs-trapezoidal bug.

    With nt=8 endpoint-inclusive samples and the old midpoint-style weight
    (dt per slice), a constant-in-t residual of amplitude C^2 would produce
    Bochner^2 = 8 * dt * C^2 = 8 * (T/7) * C^2 = (8/7) * T * C^2, an
    ~1.069x overintegration. With trapezoidal weights the answer is
    exactly T * C^2.
    """
    nt = 8
    total_time = 1.0
    dt = total_time / (nt - 1)
    # Pure spatial sin in a periodic 16x16 grid so H^-1 is nonzero.
    n_spatial = 16
    x = np.linspace(0.0, 1.0, n_spatial, endpoint=False)
    X, _ = np.meshgrid(x, x, indexing="ij")  # noqa: N806
    r_spatial = np.sin(2 * np.pi * X)
    r_series = np.stack([r_spatial] * nt, axis=-1)

    h = (1.0 / n_spatial, 1.0 / n_spatial)
    h_sq = h_minus_one_spectral(r_spatial, h) ** 2
    expected_sq = total_time * h_sq
    bochner_sq = bochner_l2_h_minus_one(r_series, spatial_h=h, dt=dt) ** 2
    rel = abs(bochner_sq - expected_sq) / expected_sq
    assert rel < 1e-12, f"relative error {rel} (pre-fix bug would give ~0.069)"


def test_bochner_zero_for_zero_residual():
    n_steps = 16
    r_series = np.zeros((32, 32, n_steps))
    assert bochner_l2_h_minus_one(r_series, spatial_h=(1 / 32, 1 / 32), dt=0.1) == 0.0


def test_bochner_nonconstant_in_time():
    """r(x, y, t) = sin(pi t) * sin(2 pi x) sin(2 pi y).

    Integral_0^1 sin^2(pi t) dt = 0.5, so Bochner^2 = 0.5 * H^-1(base)^2.
    Trapezoidal on endpoint-inclusive samples converges at O(dt^2); with
    n_steps=128 the relative error is well below 1e-3.
    """
    n_spatial = 64
    total_time = 1.0
    n_steps = 128
    dt = total_time / (n_steps - 1)
    h = (1.0 / n_spatial, 1.0 / n_spatial)
    x = np.linspace(0.0, 1.0, n_spatial, endpoint=False)
    y = np.linspace(0.0, 1.0, n_spatial, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    base = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    times = np.linspace(0.0, total_time, n_steps)
    r_series = np.stack([np.sin(np.pi * ti) * base for ti in times], axis=-1)

    bochner_sq = bochner_l2_h_minus_one(r_series, spatial_h=h, dt=dt) ** 2
    expected_sq = 0.5 * h_minus_one_spectral(base, h) ** 2
    assert abs(bochner_sq - expected_sq) / expected_sq < 1e-3


def test_bochner_rejects_too_few_steps():
    import pytest

    r = np.zeros((8, 8, 1))
    with pytest.raises(ValueError, match="n_steps >= 2"):
        bochner_l2_h_minus_one(r, spatial_h=(1 / 8, 1 / 8), dt=0.1)


def test_bochner_l2_fallback_catches_dc_residual():
    """Regression guard for the non-periodic DC false-pass.

    A constant-in-space residual (u_t = 1 everywhere on hD u=t) has all
    its mass in the k=0 mode, which h_minus_one_spectral drops. The L^2
    fallback keeps the DC content, so a constant residual of amplitude 1
    on [0,1]^2 x [0,1] gives Bochner = sqrt(T * L_x * L_y * 1^2) = 1.
    """
    n_spatial = 16
    n_steps = 8
    dt = 1.0 / (n_steps - 1)
    r_series = np.ones((n_spatial, n_spatial, n_steps))
    h = (1.0 / (n_spatial - 1), 1.0 / (n_spatial - 1))

    hmin1 = bochner_l2_h_minus_one(r_series, spatial_h=h, dt=dt)
    l2 = bochner_l2_fallback(r_series, spatial_h=h, dt=dt)
    # Spectral H^-1 drops DC -> 0
    assert hmin1 == 0.0
    # L^2 fallback sees the full residual. L2(slice) = 1 on the unit
    # square exactly; Bochner^2 = T * 1^2 = 1. Trapezoidal endpoint weights
    # on constant data are exact, so equality holds to fp.
    assert abs(l2**2 - 1.0) < 1e-10


def test_bochner_l2_fallback_matches_l2_grid_on_single_slice():
    # When n_steps=2, Bochner^2 with trapezoidal weights is
    # dt * (0.5 * l2[0]^2 + 0.5 * l2[-1]^2). Use an identical pair of
    # slices so the answer is dt * l2^2, and we can check against l2_grid.
    n = 8
    slice_val = (
        np.sin(np.pi * np.linspace(0, 1, n))[:, None]
        * np.sin(np.pi * np.linspace(0, 1, n))[None, :]
    )
    r_series = np.stack([slice_val, slice_val], axis=-1)
    h = (1.0 / (n - 1), 1.0 / (n - 1))
    dt = 0.25
    l2_per_slice_sq = l2_grid(slice_val, h) ** 2
    expected = dt * l2_per_slice_sq
    got = bochner_l2_fallback(r_series, spatial_h=h, dt=dt) ** 2
    assert abs(got - expected) < 1e-12
