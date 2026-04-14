"""Norms module tests — L^2 trapezoidal, H^-1 spectral."""

import numpy as np

from physics_lint.norms import (
    h_minus_one_spectral,
    l2_grid,
    trapezoidal_integral,
)


def test_trapezoidal_integral_constant_1d():
    N = 17  # noqa: N806  (N is grid resolution; math convention)
    u = np.ones(N)
    h = 1.0 / (N - 1)  # endpoint-inclusive grid
    result = trapezoidal_integral(u, (h,))
    assert abs(result - 1.0) < 1e-12


def test_trapezoidal_integral_linear_2d():
    N = 65  # noqa: N806  (N is grid resolution; math convention)
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    u = X + Y  # integral over [0,1]^2 = 1.0
    h = 1.0 / (N - 1)
    result = trapezoidal_integral(u, (h, h))
    assert abs(result - 1.0) < 1e-12


def test_l2_grid_sine():
    # integral of sin^2(pi x) sin^2(pi y) dx dy over [0,1]^2 = 1/4
    # so sqrt(1/4) = 0.5
    N = 129  # noqa: N806  (N is grid resolution; math convention)
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)
    h = 1.0 / (N - 1)
    assert abs(l2_grid(u, (h, h)) - 0.5) < 1e-5


def test_h_minus_one_spectral_sine_mode():
    # u = sin(2 pi x) sin(2 pi y) on periodic [0,1]^2
    # ||u||_{H^-1}^2 = sum_{k!=0} |u_hat_k|^2 / |k|^2
    # The only nonzero modes are (+/-1, +/-1) with |k|^2 = (2 pi)^2 * 2
    # and |u_hat|^2 = 1/16 each for N^2 = 1 samples after FFT normalization
    # This test asserts the function runs and returns a positive value
    # of the expected order of magnitude.
    N = 64  # noqa: N806  (N is grid resolution; math convention)
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    h = 1.0 / N
    val = h_minus_one_spectral(u, (h, h))
    # Expected: ||u||^2 / (8 pi^2)  for this single-mode case
    expected_sq = 0.25 / (8 * np.pi**2)
    assert abs(val - np.sqrt(expected_sq)) < 1e-8


def test_h_minus_one_spectral_zero_mean_required():
    # The DC mode (k=0) has no inverse; h_minus_one_spectral must either
    # ignore it or raise. We choose: ignore it silently (sum over k != 0).
    N = 32  # noqa: N806  (N is grid resolution; math convention)
    u = np.ones((N, N)) * 3.5  # pure DC
    h = 1.0 / N
    val = h_minus_one_spectral(u, (h, h))
    assert val == 0.0
