"""Bochner L^2(0,T; H^-1) norm tests — midpoint rule over time slices."""

import numpy as np

from physics_lint.norms import bochner_l2_h_minus_one, h_minus_one_spectral


def test_bochner_constant_in_time():
    """If the spatial residual is constant in t, Bochner norm = sqrt(T) * H^-1 norm."""
    N = 64  # noqa: N806
    T = 1.0  # noqa: N806
    n_steps = 32
    dt = T / n_steps
    h = (1.0 / N, 1.0 / N)
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    # Spatial residual is sin(2 pi x) sin(2 pi y) for all t
    r_spatial = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    r_series = np.stack([r_spatial] * n_steps, axis=-1)

    h_sq_per_slice = h_minus_one_spectral(r_spatial, h) ** 2
    expected_sq = T * h_sq_per_slice
    bochner = bochner_l2_h_minus_one(r_series, spatial_h=h, dt=dt)
    assert abs(bochner**2 - expected_sq) < 1e-10


def test_bochner_zero_for_zero_residual():
    n_steps = 16
    r_series = np.zeros((32, 32, n_steps))
    assert bochner_l2_h_minus_one(r_series, spatial_h=(1 / 32, 1 / 32), dt=0.1) == 0.0


def test_bochner_nonconstant_in_time():
    """r(x, y, t) = sin(pi t) * sin(2 pi x) sin(2 pi y).

    Bochner-squared = integral_0^1 sin^2(pi t) dt * H^-1(sin sin)^2 = 0.5 * H^-1^2.
    Sampling at midpoint t = (k + 0.5) * dt gives a 2nd-order-accurate midpoint
    rule; a 128-step grid takes the relative error well below 1e-3.
    """
    N = 64  # noqa: N806
    T = 1.0  # noqa: N806
    n_steps = 128
    dt = T / n_steps
    h = (1.0 / N, 1.0 / N)
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    base = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    times = (np.arange(n_steps) + 0.5) * dt
    r_series = np.stack([np.sin(np.pi * t) * base for t in times], axis=-1)

    bochner_sq = bochner_l2_h_minus_one(r_series, spatial_h=h, dt=dt) ** 2
    expected_sq = 0.5 * h_minus_one_spectral(base, h) ** 2
    assert abs(bochner_sq - expected_sq) / expected_sq < 1e-3
