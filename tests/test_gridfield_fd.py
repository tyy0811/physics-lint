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


def test_fd_laplacian_nonperiodic_edge_converges_second_order():
    # Pins the O(h^2) promise of the non-periodic outer-band fallback.
    # u(x,y) = x^4 + y^4 has Laplacian 12(x^2 + y^2). The 4th-order central
    # stencil is exact for quartics up to a truncation involving u^(6) = 0,
    # so interior error is ~machine precision. The outer band uses the
    # one-sided 4-point + 3-point central formulas which are O(h^2) with
    # a non-zero u^(4) = 24, so err ~ C*h^2 with C ~ O(1).
    prefactors = []
    for N in (32, 64, 128, 256):  # noqa: N806  (N is grid resolution; math convention)
        x = np.linspace(0.0, 1.0, N)
        h = x[1] - x[0]
        X, Y = np.meshgrid(x, x, indexing="ij")  # noqa: N806  (meshgrid coords)
        u = X**4 + Y**4
        f = GridField(u, h=h, periodic=False, backend="fd")
        lap = f.laplacian().values()
        expected = 12.0 * (X**2 + Y**2)
        edge_err = float(np.max(np.abs(lap[:2, :] - expected[:2, :])))
        edge_err = max(edge_err, float(np.max(np.abs(lap[-2:, :] - expected[-2:, :]))))
        edge_err = max(edge_err, float(np.max(np.abs(lap[:, :2] - expected[:, :2]))))
        edge_err = max(edge_err, float(np.max(np.abs(lap[:, -2:] - expected[:, -2:]))))
        prefactors.append(edge_err / h**2)
    # O(h^2) means err/h^2 is bounded by a constant; check the ratio is
    # stable (no first-order creep) across N. Allow a 2x drift to tolerate
    # finite-N effects.
    assert (
        max(prefactors) / min(prefactors) < 2.0
    ), f"Non-periodic edge prefactors drift: {prefactors}"
    # And pin an absolute ceiling so a regression to O(h) would blow up.
    # With u''''=24 and a 4-point one-sided formula (error ~ -11/12 h^2 u''''),
    # the expected prefactor is ~22 per axis; with both axes contributing,
    # bound at ~80 for headroom.
    assert max(prefactors) < 80.0, f"Non-periodic edge prefactor too large: {prefactors}"


def test_gridfield_numpy_scalar_h_accepted():
    f = GridField(np.zeros((4, 4)), h=np.float32(0.25), periodic=False)
    assert f.h == (0.25, 0.25)


def test_gridfield_invalid_h_type_raises_typeerror():
    with pytest.raises(TypeError, match="scalar or an iterable"):
        GridField(np.zeros((4, 4)), h="not-a-number", periodic=False)
