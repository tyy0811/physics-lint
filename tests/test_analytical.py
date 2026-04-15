"""Analytical battery tests — each returns a function + the exact Laplacian."""

import numpy as np

from physics_lint.analytical import laplace as laplace_sols
from physics_lint.analytical import poisson as poisson_sols


def test_laplace_harmonic_polynomial_satisfies_pde():
    sol = laplace_sols.harmonic_polynomial_square()
    size = 64
    x = np.linspace(0.0, 1.0, size)
    y = np.linspace(0.0, 1.0, size)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    u = sol.u(xg, yg)
    lap = sol.laplacian(xg, yg)
    # Laplace: -Delta u = 0 so the analytical Laplacian should be zero
    assert np.max(np.abs(lap)) < 1e-15
    # Value sanity: u = x^2 - y^2
    assert np.allclose(u, xg**2 - yg**2)


def test_laplace_eigen_trace_satisfies_pde():
    sol = laplace_sols.eigen_trace_square(n=1)
    size = 64
    x = np.linspace(0.0, 1.0, size)
    y = np.linspace(0.0, 1.0, size)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    lap = sol.laplacian(xg, yg)
    assert np.max(np.abs(lap)) < 1e-14
    # Value sanity: u at y=1 should equal sin(n pi x)
    u = sol.u(xg, yg)
    assert np.allclose(u[:, -1], np.sin(np.pi * x))


def test_poisson_sin_sin_mms_residual_matches_source():
    sol = poisson_sols.sin_sin_mms_square()
    size = 64
    x = np.linspace(0.0, 1.0, size)
    y = np.linspace(0.0, 1.0, size)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    u = sol.u(xg, yg)
    f = sol.source(xg, yg)
    # -Delta u = 2 pi^2 sin(pi x) sin(pi y), which is f
    expected = 2 * np.pi**2 * np.sin(np.pi * xg) * np.sin(np.pi * yg)
    assert np.allclose(f, expected)
    # Analytical Laplacian
    lap = sol.laplacian(xg, yg)
    assert np.allclose(-lap, f)
    # Value sanity: u zero on the boundary
    assert np.allclose(u[0, :], 0.0)
    assert np.allclose(u[-1, :], 0.0)


def test_poisson_periodic_mms_roundtrip():
    sol = poisson_sols.periodic_sin_sin()
    size = 64
    x = np.linspace(0.0, 2 * np.pi, size, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, size, endpoint=False)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    u = sol.u(xg, yg)
    f = sol.source(xg, yg)
    lap = sol.laplacian(xg, yg)
    assert np.allclose(-lap, f)
    # Value sanity
    assert np.allclose(u, np.sin(xg) * np.sin(yg))
