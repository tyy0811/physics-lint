"""Heat and wave analytical solution tests — Laplacian and time derivative exact."""

import numpy as np

from physics_lint.analytical import heat as heat_sols
from physics_lint.analytical import wave as wave_sols

# numpy 2.0 removed np.trapz; prefer np.trapezoid, fall back for numpy 1.26.x.
# Ternary (not getattr default) because getattr evaluates the default eagerly.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def test_heat_eigenfunction_hd_square_pde_satisfied():
    sol = heat_sols.eigenfunction_decay_square(kappa=0.01)
    N = 64  # noqa: N806  (grid resolution)
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    for t in (0.0, 0.1, 0.5, 1.0):
        u = sol.u(X, Y, t)
        u_t = sol.time_derivative(X, Y, t)
        lap = sol.laplacian(X, Y, t)
        # Heat PDE: u_t - kappa * lap u = 0
        residual = u_t - sol.kappa * lap
        assert np.max(np.abs(residual)) < 1e-12, f"t={t}: max |r|={np.max(np.abs(residual)):.2e}"
        # Boundary is zero for hD eigenfunction
        assert abs(u[0, :].max()) < 1e-15
        assert abs(u[-1, :].max()) < 1e-15
        assert abs(u[:, 0].max()) < 1e-15
        assert abs(u[:, -1].max()) < 1e-15


def test_heat_periodic_cos_cos_pde_satisfied():
    sol = heat_sols.periodic_cos_cos(kappa=0.01)
    N = 64  # noqa: N806
    x = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    for t in (0.0, 0.1, 0.5):
        u_t = sol.time_derivative(X, Y, t)
        lap = sol.laplacian(X, Y, t)
        residual = u_t - sol.kappa * lap
        assert np.max(np.abs(residual)) < 1e-12


def test_heat_mass_conservation_periodic():
    # Under periodic BC, integral of u over the domain should be constant in time.
    sol = heat_sols.periodic_cos_cos(kappa=0.01)
    N = 128  # noqa: N806
    x = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    h = 2 * np.pi / N
    masses = []
    for t in (0.0, 0.1, 0.5, 1.0):
        u = sol.u(X, Y, t)
        masses.append(np.sum(u) * h * h)
    # cos(x)*cos(y) integrates to zero over the full period; decay preserves zero
    for m in masses:
        assert abs(m) < 1e-10


def test_wave_standing_wave_pde_satisfied():
    sol = wave_sols.standing_wave_square(c=1.0)
    N = 64  # noqa: N806
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    for t in (0.0, 0.25, 0.5):
        u_tt = sol.second_time_derivative(X, Y, t)
        lap = sol.laplacian(X, Y, t)
        residual = u_tt - sol.c**2 * lap
        assert np.max(np.abs(residual)) < 1e-12


def test_wave_energy_conserved_standing():
    """E(t) = 0.5 * integral(u_t^2 + c^2 |grad u|^2) — const under hD.

    Plain sum*h*h is left-Riemann (O(h)) and drifts across timesteps as the
    integrand redistributes between the two cos/sin components — even though
    the sum of both is analytically constant. Trapezoidal over both axes is
    the right tool here: sin^2 vanishes at the [0,1] boundary so the
    Euler-Maclaurin tail for this integrand is smooth, giving quadrature
    error comfortably below the 1e-4 conservation tolerance.
    """
    sol = wave_sols.standing_wave_square(c=1.0)
    N = 128  # noqa: N806
    h = 1.0 / (N - 1)
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    energies = []
    for t in (0.0, 0.1, 0.3, 0.5):
        u_t = sol.time_derivative(X, Y, t)
        gx = sol.grad_x(X, Y, t)
        gy = sol.grad_y(X, Y, t)
        energy_density = 0.5 * (u_t**2 + sol.c**2 * (gx**2 + gy**2))
        energies.append(float(_trapz(_trapz(energy_density, dx=h), dx=h)))
    for e in energies[1:]:
        assert abs(e - energies[0]) / energies[0] < 1e-4
