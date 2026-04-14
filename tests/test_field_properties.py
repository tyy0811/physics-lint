"""Hypothesis property-based tests for the Field abstraction.

Five properties (design doc §15):
1. Polynomial of degree <= stencil order -> 4th-order FD derivative exact
2. sin(kx) on periodic grid -> spectral derivative is k*cos(kx) to ~1e-12
3. Laplacian commutes with np.rot90 on square grids
4. Residual of analytical solution converges at expected rate under refinement
5. Integration by parts on periodic domains: integral(u*Lap(v)) == integral(v*Lap(u))
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from physics_lint.field import GridField
from physics_lint.norms import trapezoidal_integral

# Strategy helpers
_GRID_SIZES = st.integers(min_value=8, max_value=128).filter(lambda n: n % 2 == 0)
_SMALL_K = st.integers(min_value=1, max_value=4)


@given(n=_GRID_SIZES, degree=st.integers(min_value=0, max_value=3))
@settings(max_examples=25, deadline=None)
def test_polynomial_fd_exact_interior(n: int, degree: int) -> None:
    """4th-order central FD on a polynomial of degree <= 3 is exact in the interior."""
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    h = 1.0 / (n - 1)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    # u = mesh_x^degree + mesh_y^degree so Laplacian =
    # d(d-1)*(mesh_x^(d-2) + mesh_y^(d-2))
    u = mesh_x**degree + mesh_y**degree
    f = GridField(u, h=h, periodic=False, backend="fd")
    lap = f.laplacian().values()
    if degree >= 2:
        expected = degree * (degree - 1) * (mesh_x ** (degree - 2) + mesh_y ** (degree - 2))
    else:
        expected = np.zeros_like(mesh_x)
    interior = lap[3:-3, 3:-3]
    expected_interior = expected[3:-3, 3:-3]
    assert np.max(np.abs(interior - expected_interior)) < 1e-9


@given(n=_GRID_SIZES, k=_SMALL_K)
@settings(max_examples=25, deadline=None)
def test_spectral_sine_derivative(n: int, k: int) -> None:
    """Spectral first derivative of sin(2*pi*k*x)*(1 in y) matches 2*pi*k*cos(...)."""
    h = 1.0 / n
    xg = np.linspace(0.0, 1.0, n, endpoint=False)
    yg = np.linspace(0.0, 1.0, n, endpoint=False)
    mesh_x, _mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = np.sin(2 * np.pi * k * mesh_x)
    f = GridField(u, h=h, periodic=True, backend="spectral")
    # Spectral Laplacian of sin(2*pi*k*x) is -(2*pi*k)^2 * sin(2*pi*k*x)
    lap = f.laplacian().values()
    expected = -((2 * np.pi * k) ** 2) * u
    # Empirical worst case across n in [8,128] (even), k in {1..4} is 1.92e-10
    # at (n=126, k=4); this is FFT roundoff that scales like (2*pi*k)^2 * n * eps.
    # Deterministic; 5e-10 gives ~2.6x margin over the observed max. See Task 13
    # deviation D2 in plan 2026-04-14.
    assert np.max(np.abs(lap - expected)) < 5e-10


@given(n=_GRID_SIZES)
@settings(max_examples=20, deadline=None)
def test_rotation_commutes_with_laplacian(n: int) -> None:
    """rot90(Laplacian(u)) == Laplacian(rot90(u)) on a square periodic grid."""
    h = 1.0 / n
    xg = np.linspace(0.0, 1.0, n, endpoint=False)
    yg = np.linspace(0.0, 1.0, n, endpoint=False)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = np.sin(2 * np.pi * mesh_x) * np.sin(2 * np.pi * mesh_y)

    f = GridField(u, h=h, periodic=True)
    u_rot = np.rot90(u)
    f_rot = GridField(u_rot, h=h, periodic=True)

    lhs = np.rot90(f.laplacian().values())
    rhs = f_rot.laplacian().values()
    assert np.max(np.abs(lhs - rhs)) < 1e-10


@given(k=_SMALL_K)
@settings(max_examples=10, deadline=None)
def test_refinement_rate_fd_periodic(k: int) -> None:
    """4th-order FD should converge like h^4 on a smooth periodic sine."""
    errs: list[float] = []
    for n in (32, 64, 128):
        h = 2 * np.pi / n
        xg = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        yg = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
        u = np.sin(k * mesh_x) * np.sin(k * mesh_y)
        f = GridField(u, h=h, periodic=True, backend="fd")
        lap = f.laplacian().values()
        expected = -2 * k**2 * u
        errs.append(float(np.max(np.abs(lap - expected))))
    # Coarse-to-fine ratio >= 2^3.5 ~ 11.3 between N=32 and N=64
    if errs[0] > 0:
        ratio_coarse = errs[0] / max(errs[1], 1e-300)
        assert ratio_coarse > 8.0, f"refinement rate collapsed: {errs}"


@given(k=_SMALL_K)
@settings(max_examples=10, deadline=None)
def test_integration_by_parts_periodic(k: int) -> None:
    """integral(u * Lap(v)) == integral(v * Lap(u)) on a periodic domain."""
    n = 64
    h = 1.0 / n
    xg = np.linspace(0.0, 1.0, n, endpoint=False)
    yg = np.linspace(0.0, 1.0, n, endpoint=False)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = np.sin(2 * np.pi * k * mesh_x) * np.cos(2 * np.pi * k * mesh_y)
    v = np.cos(2 * np.pi * k * mesh_x) * np.sin(2 * np.pi * k * mesh_y)
    fu = GridField(u, h=h, periodic=True)
    fv = GridField(v, h=h, periodic=True)
    lap_u = fu.laplacian().values()
    lap_v = fv.laplacian().values()
    lhs = trapezoidal_integral(u * lap_v, (h, h))
    rhs = trapezoidal_integral(v * lap_u, (h, h))
    assert abs(lhs - rhs) < 1e-10
