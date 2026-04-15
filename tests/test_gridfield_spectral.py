"""GridField Fourier spectral backend tests.

Spectral derivatives on smooth periodic inputs are exact to roughly
machine precision; the tolerance here reflects floating-point noise
plus FFT backend drift (see design doc §6.3).
"""

import numpy as np
import pytest

from physics_lint.field import GridField


def test_spectral_laplacian_sine_machine_precision():
    # -∆ of sin(2*pi*x)*sin(2*pi*y) = 8*pi^2 * sin*sin at machine precision.
    N = 64  # noqa: N806  (N is grid resolution; math convention)
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    h = 1.0 / N
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    f = GridField(u, h=h, periodic=True)  # auto-selects spectral
    lap = f.laplacian().values()
    expected = -8 * np.pi**2 * u

    # Tolerance note: FFT roundoff varies slightly across platforms.
    # Local macOS/x86_64/torch2.2.2 measures ~9.7e-12; CI ubuntu-latest/py3.11
    # measures ~1.0e-11 (just over the original 1e-11 threshold). Bumped to
    # 5e-11 for ~5x cross-platform headroom while still catching a regression
    # from spectral-accurate to any finite-difference-like degradation.
    assert np.max(np.abs(lap - expected)) < 5e-11


def test_spectral_laplacian_multimode():
    # Higher modes: -∆ sin(k*pi*x) sin(k*pi*y) = 2*(k*pi)^2 * sin*sin
    #
    # Tolerance note: FFT floating-point noise scales with the expected
    # signal amplitude, which grows as k^2 (from the Laplacian eigenvalue).
    # An *absolute* tolerance therefore tightens against the highest mode.
    # We use a *relative* tolerance pegged a few orders of magnitude above
    # machine epsilon (empirically ~1e-13 rel). This checks the intrinsic
    # invariant — "spectral Laplacian is exact up to FFT roundoff" — rather
    # than an amplitude-coupled bound.
    N = 128  # noqa: N806  (N is grid resolution; math convention)
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    for k in (1, 3, 7):
        u = np.sin(2 * k * np.pi * X) * np.sin(2 * k * np.pi * Y)
        f = GridField(u, h=h, periodic=True)
        lap = f.laplacian().values()
        expected = -2 * (2 * k * np.pi) ** 2 * u
        rel_err = np.max(np.abs(lap - expected)) / np.max(np.abs(expected))
        assert rel_err < 1e-12, f"k={k} failed rel_err={rel_err:.3e}"


def test_spectral_laplacian_3d():
    N = 32  # noqa: N806  (N is grid resolution; math convention)
    h = 1.0 / N
    axis = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")  # noqa: N806  (meshgrid coords)
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) * np.sin(2 * np.pi * Z)
    f = GridField(u, h=h, periodic=True)
    lap = f.laplacian().values()
    expected = -12 * np.pi**2 * u
    assert np.max(np.abs(lap - expected)) < 1e-10


def test_fd_grad_nonperiodic_converges_second_order():
    # Pins the contract of _fd4_first_derivative's non-periodic branch
    # (numpy.gradient(edge_order=2) fallback). This branch is not exercised
    # by any other Task 3 or Task 4 test but will be hit by residual rules
    # consuming non-periodic gradients in Task 10.
    #
    # u(x, y) = x**3 + y**3 has grad = (3 x**2, 3 y**2). The 4th-order
    # central stencil nails the interior exactly (4th-deriv = 0), but the
    # edge fallback is only 2nd-order, so we measure the edge rate.
    prefactors = []
    for N in (33, 65, 129):  # noqa: N806  (N is grid resolution; math convention)
        x = np.linspace(0.0, 1.0, N)
        h = x[1] - x[0]
        X, Y = np.meshgrid(x, x, indexing="ij")  # noqa: N806  (meshgrid coords)
        u = X**3 + Y**3
        f = GridField(u, h=h, periodic=False, backend="fd")
        gx, gy = f.grad()
        expected_gx = 3.0 * X**2
        expected_gy = 3.0 * Y**2
        edge_err = 0.0
        edge_err = max(edge_err, float(np.max(np.abs(gx.values()[:2, :] - expected_gx[:2, :]))))
        edge_err = max(edge_err, float(np.max(np.abs(gx.values()[-2:, :] - expected_gx[-2:, :]))))
        edge_err = max(edge_err, float(np.max(np.abs(gy.values()[:, :2] - expected_gy[:, :2]))))
        edge_err = max(edge_err, float(np.max(np.abs(gy.values()[:, -2:] - expected_gy[:, -2:]))))
        prefactors.append(edge_err / h**2)
    # O(h^2) means err/h^2 is bounded by a constant; check the ratio
    # across N is stable within 2x to tolerate finite-N effects.
    assert max(prefactors) / min(prefactors) < 2.0, (
        f"Non-periodic grad edge prefactors drift: {prefactors}"
    )
    # Absolute ceiling to catch a regression to O(h).
    assert max(prefactors) < 100.0, f"Non-periodic grad edge prefactor too large: {prefactors}"


def test_integrate_rejects_mismatched_weight_grid():
    u = GridField(np.ones((4, 4)), h=0.1, periodic=False)
    w = GridField(np.ones((4, 4)), h=0.2, periodic=False)  # different h
    with pytest.raises(ValueError, match="spacing"):
        u.integrate(weight=w)


def test_spectral_grad_sine_machine_precision():
    # d/dx [sin(2pi x) sin(2pi y)] = 2pi cos(2pi x) sin(2pi y)
    # d/dy [sin(2pi x) sin(2pi y)] = 2pi sin(2pi x) cos(2pi y)
    # Both should be exact to FFT roundoff on periodic grids. Exercises
    # _spectral_first_derivative (grid.py:294-302).
    N = 64  # noqa: N806  (N is grid resolution; math convention)
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    f = GridField(u, h=h, periodic=True)  # auto-selects spectral
    gx, gy = f.grad()
    expected_gx = 2 * np.pi * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    expected_gy = 2 * np.pi * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    assert np.max(np.abs(gx.values() - expected_gx)) < 1e-11
    assert np.max(np.abs(gy.values() - expected_gy)) < 1e-11


def test_spectral_grad_requires_periodic():
    # Force spectral backend on a non-periodic GridField; grad() must raise.
    # Exercises the ValueError branch at grid.py:101-103.
    u = np.zeros((8, 8))
    f = GridField(u, h=0.125, periodic=False, backend="spectral")
    with pytest.raises(ValueError, match="spectral backend requires periodic=True"):
        f.grad()


def test_spectral_laplacian_requires_periodic():
    # Force spectral backend on a non-periodic GridField; laplacian() must raise.
    # Exercises the ValueError branch at grid.py:120.
    u = np.zeros((8, 8))
    f = GridField(u, h=0.125, periodic=False, backend="spectral")
    with pytest.raises(ValueError, match="spectral backend requires periodic=True"):
        f.laplacian()


def test_fd_grad_periodic_stencil_hits_fd_branch():
    # Force FD backend on a periodic GridField; grad() should take the
    # _fd4_first_derivative(periodic=True) branch (not spectral). Exercises
    # the periodic np.roll stencil path in _fd4_first_derivative (grid.py:253-256).
    N = 64  # noqa: N806  (N is grid resolution; math convention)
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    f = GridField(u, h=h, periodic=True, backend="fd")  # forced FD
    gx, gy = f.grad()
    expected_gx = 2 * np.pi * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    expected_gy = 2 * np.pi * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    # 4th-order FD periodic: err ~ C * h^4 per axis. Empirically the
    # prefactor for sin(2pi x)*sin(2pi y) grad at the (2pi) wavenumber is
    # O(a few hundred); 2000 * h^4 is the same headroom the Laplacian test
    # uses and comfortably holds.
    tol = 2000 * h**4
    assert np.max(np.abs(gx.values() - expected_gx)) < tol
    assert np.max(np.abs(gy.values() - expected_gy)) < tol
