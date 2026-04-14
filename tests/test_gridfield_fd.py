"""GridField 4th-order central FD backend tests."""

import numpy as np
import pytest

from physics_lint.field import GridField


def test_gridfield_stores_values_and_spacing():
    u = np.zeros((8, 8))
    f = GridField(u, h=0.125, periodic=False)
    assert np.array_equal(f.values(), u)
    assert f.h == (0.125, 0.125)
    assert f.periodic is False
    assert f.backend == "fd"


def test_gridfield_rejects_periodic_and_fd_together_if_forced():
    # periodic=True auto-selects spectral; forcing backend="fd" with
    # periodic=True is legal (user override) and should still work.
    u = np.zeros((8, 8))
    f = GridField(u, h=0.125, periodic=True, backend="fd")
    assert f.backend == "fd"


def test_gridfield_scalar_h_expands_to_tuple():
    f = GridField(np.zeros((4, 4, 4)), h=0.5, periodic=False)
    assert f.h == (0.5, 0.5, 0.5)


def test_gridfield_tuple_h_must_match_ndim():
    with pytest.raises(ValueError, match="ndim"):
        GridField(np.zeros((4, 4)), h=(0.5, 0.5, 0.5), periodic=False)


def test_fd_derivative_exact_on_cubic():
    # A cubic polynomial's fourth derivative is zero, so a 4th-order
    # FD stencil should reproduce the second derivative exactly to
    # machine precision at interior points.
    N = 32  # noqa: N806  (N is grid resolution; math convention)
    x = np.linspace(0.0, 1.0, N)
    h = x[1] - x[0]
    u_1d = x**3  # 1D, length N
    u_2d = u_1d[:, None] * np.ones((N, N))  # 2D, uniform in y

    f = GridField(u_2d, h=h, periodic=False, backend="fd")
    lap = f.laplacian().values()

    # interior second derivative wrt x of x^3 is 6x; wrt y of const is 0
    expected = 6.0 * x[:, None] * np.ones((N, N))
    assert np.allclose(lap[3:-3, 3:-3], expected[3:-3, 3:-3], atol=1e-10)


def test_fd_laplacian_periodic_sine_converges():
    # Laplacian of sin(2*pi*x)*sin(2*pi*y) is -8*pi^2 * sin*sin.
    # With periodic wraps, 4th-order FD should be accurate everywhere,
    # not just interior.
    for N in (32, 64, 128):  # noqa: N806  (N is grid resolution; math convention)
        x = np.linspace(0.0, 1.0, N, endpoint=False)
        y = np.linspace(0.0, 1.0, N, endpoint=False)
        h = 1.0 / N
        X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
        u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        f = GridField(u, h=h, periodic=True, backend="fd")
        lap = f.laplacian().values()
        expected = -8 * np.pi**2 * u
        # 4th-order FD on smooth periodic input: error ~ C * h^4.
        # Empirical prefactor for sin(2pi x)*sin(2pi y) at the (2pi)
        # wavenumber is ~1367 (stable across N=32..256); 2000 gives CI
        # headroom while still catching a regression to O(h^2).
        tol = 2000 * h**4
        assert (
            np.max(np.abs(lap - expected)) < tol
        ), f"N={N}: max err {np.max(np.abs(lap - expected)):.3e}, tol {tol:.3e}"
