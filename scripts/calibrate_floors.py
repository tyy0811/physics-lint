"""Floor calibration script for physics_lint/data/floors.toml.

Runs the analytical battery against every Week-1 (rule, pde, grid_shape,
method, norm) tuple, records the measured residual/error, and writes a
machine-readable JSON summary to stdout. Run in multiple environments
(macOS arm64 local, ubuntu Docker, throwaway GHA matrix) and take the
MAXIMUM observed value across environments as the floors.toml ``value``.
The per-method tolerance multiplier (2x for fd4, 3x for spectral) is
applied when writing floors.toml, not here.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass

import numpy as np

from physics_lint.analytical import heat as heat_sols
from physics_lint.analytical import laplace as laplace_sols
from physics_lint.analytical import poisson as poisson_sols
from physics_lint.analytical import wave as wave_sols
from physics_lint.field import GridField
from physics_lint.field.grid import _fd4_second_derivative, _spectral_laplacian
from physics_lint.norms import (
    bochner_l2_fallback,
    bochner_l2_h_minus_one,
    h_minus_one_spectral,
    integrate_over_domain,
    l2_grid,
)


@dataclass
class FloorEntry:
    rule: str
    pde: str
    grid_shape: tuple[int, ...]
    method: str
    norm: str
    measured: float
    analytical_solution: str


def _measure_laplace_fd_l2(n: int) -> float:
    """PH-RES-001 floor for non-periodic FD Laplace via harmonic polynomial.

    u = x^2 - y^2 is harmonic; the exact Laplacian is zero. The FD
    Laplacian's L^2 norm is purely numerical noise (4th-order in interior,
    2nd-order at edges).
    """
    sol = laplace_sols.harmonic_polynomial_square()
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = sol.u(mesh_x, mesh_y)
    h = 1.0 / (n - 1)
    field = GridField(u, h=h, periodic=False, backend="fd")
    residual = -field.laplacian().values()
    return l2_grid(residual, (h, h))


def _measure_poisson_spectral_h_minus_one(n: int) -> float:
    """PH-RES-001 floor for periodic spectral Poisson via sin(x)sin(y) MMS.

    u = sin(x)sin(y) on [0, 2 pi]^2, f = 2 sin(x)sin(y). The spectral
    Laplacian is exact to FFT roundoff, so residual = -lap - f is purely
    roundoff.
    """
    sol = poisson_sols.periodic_sin_sin()
    xg = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    yg = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = sol.u(mesh_x, mesh_y)
    f_arr = sol.source(mesh_x, mesh_y)
    h = 2 * np.pi / n
    field = GridField(u, h=h, periodic=True, backend="spectral")
    residual = -field.laplacian().values() - f_arr
    return h_minus_one_spectral(residual, (h, h))


def _measure_laplace_spectral_h_minus_one(n: int) -> float:
    """PH-RES-001 floor for periodic spectral Laplace via sin(x)sin(y) eigenfunction.

    u = sin(x)sin(y) is NOT harmonic, but -Delta u = 2 sin(x)sin(y).
    For Laplace (no source), residual = -lap. We instead calibrate using
    u that IS a Laplace solution on the periodic torus: the only such
    periodic Laplace solutions are constants. Use u = const; residual is
    literally zero, H^-1 norm is zero — record machine-epsilon as the
    floor since we cannot distinguish below that.
    """
    u = np.full((n, n), 0.25, dtype=np.float64)
    h = 2 * np.pi / n
    field = GridField(u, h=h, periodic=True, backend="spectral")
    residual = -field.laplacian().values()
    value = h_minus_one_spectral(residual, (h, h))
    # Safety floor: the true value is analytically zero, so we record
    # machine-epsilon as the observable floor. Prevents 0.0 from collapsing
    # tri-state division later.
    return max(value, float(np.finfo(np.float64).eps))


def _measure_heat_periodic_spectral_bochner(n: int, nt: int) -> float:
    """PH-RES-001 floor for heat + periodic + spectral: Bochner-H-1.

    Residual = u_t - kappa * Lap u, where u_t is computed via np.gradient
    (2nd-order central FD) and Lap u via the spectral backend. The
    Bochner-H-1 norm drops the DC mode per slice — valid on periodic grids.
    """
    kappa = 0.01
    sol = heat_sols.periodic_cos_cos(kappa=kappa)
    xg = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    yg = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    tg = np.linspace(0.0, 0.5, nt)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = np.stack([sol.u(mesh_x, mesh_y, ti) for ti in tg], axis=-1)
    h_spatial = (2 * np.pi / n, 2 * np.pi / n)
    dt = 0.5 / (nt - 1)
    u_t = np.gradient(u, dt, axis=-1, edge_order=2)
    residual = np.empty_like(u)
    for k in range(nt):
        slice_k = np.take(u, k, axis=-1)
        lap = _spectral_laplacian(slice_k, h_spatial)
        residual[..., k] = np.take(u_t, k, axis=-1) - kappa * lap
    return bochner_l2_h_minus_one(residual, spatial_h=h_spatial, dt=dt)


def _measure_heat_con_003_periodic_spectral(n: int, nt: int) -> float:
    """PH-CON-003 floor for periodic+spectral heat.

    cos(x)cos(y) energy decays as e^(-4 kappa t); the analytical derivative
    is strictly negative, so the floor is just numerical noise from the
    endpoint-order central time derivative near t=0 and t=T.
    """
    kappa = 0.01
    sol = heat_sols.periodic_cos_cos(kappa=kappa)
    xg = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    yg = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    tg = np.linspace(0.0, 0.5, nt)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = np.stack([sol.u(mesh_x, mesh_y, ti) for ti in tg], axis=-1)
    h_spatial = (2 * np.pi / n, 2 * np.pi / n)
    dt = 0.5 / (nt - 1)
    energy = np.array(
        [
            integrate_over_domain(np.take(u, k, axis=-1) ** 2, h_spatial, periodic=True)
            for k in range(nt)
        ]
    )
    de_dt = np.gradient(energy, dt, edge_order=2)
    max_growth = max(0.0, float(np.max(de_dt)))
    energy_scale = max(float(np.max(energy)), 1e-12)
    return max_growth / energy_scale


def _measure_wave_hd_fd_bochner(n: int, nt: int) -> float:
    """PH-RES-001 floor for wave + hD + fd4: Bochner-L2 fallback.

    Standing wave eigenfunction on [0,1]^2. Non-periodic so the
    h_minus_one_spectral path would drop the DC mode and silently hide a
    constant-in-space residual — the rule falls back to Bochner-L2 and so
    does this calibration.
    """
    c = 1.0
    sol = wave_sols.standing_wave_square(c=c)
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    tg = np.linspace(0.0, 0.5, nt)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = np.stack([sol.u(mesh_x, mesh_y, ti) for ti in tg], axis=-1)
    h_spatial = (1.0 / (n - 1), 1.0 / (n - 1))
    dt = 0.5 / (nt - 1)
    u_t = np.gradient(u, dt, axis=-1, edge_order=2)
    u_tt = np.gradient(u_t, dt, axis=-1, edge_order=2)
    residual = np.empty_like(u)
    for k in range(nt):
        slice_k = np.take(u, k, axis=-1)
        lap = _fd4_second_derivative(
            slice_k, axis=0, h=h_spatial[0], periodic=False
        ) + _fd4_second_derivative(slice_k, axis=1, h=h_spatial[1], periodic=False)
        residual[..., k] = np.take(u_tt, k, axis=-1) - (c**2) * lap
    return bochner_l2_fallback(residual, spatial_h=h_spatial, dt=dt)


def _measure_heat_con_001_periodic_spectral(n: int, nt: int) -> float:
    """PH-CON-001 exact-mass floor for periodic+spectral heat.

    Matches the rule: max |M(t) - M(0)| / max(|M(0)|, ||u_0||_1) using
    ``integrate_over_domain(..., periodic=True)`` (rectangle rule) for
    both numerator and scale. The analytical solution cos(x)cos(y) has
    zero analytic mass and rectangle quadrature reproduces that to
    machine precision, so the reported floor is time-derivative noise
    rather than a quadrature bias.
    """
    kappa = 0.01
    sol = heat_sols.periodic_cos_cos(kappa=kappa)
    xg = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    yg = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    tg = np.linspace(0.0, 0.5, nt)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = np.stack([sol.u(mesh_x, mesh_y, ti) for ti in tg], axis=-1)
    h_spatial = (2 * np.pi / n, 2 * np.pi / n)
    mass = np.array(
        [integrate_over_domain(np.take(u, k, axis=-1), h_spatial, periodic=True) for k in range(nt)]
    )
    m0 = float(mass[0])
    l1 = float(integrate_over_domain(np.abs(np.take(u, 0, axis=-1)), h_spatial, periodic=True))
    scale = max(abs(m0), l1, 1e-12)
    return float(np.max(np.abs(mass - m0))) / scale


def _measure_heat_con_001_hd_fd(n: int, nt: int) -> float:
    """PH-CON-001 rate-consistency floor for hD+fd heat.

    Uses the eigenfunction decay solution; observed dM/dt compared against
    the divergence-theorem expected value kappa * integral(lap u), reduced
    via the relative L^2 over [0, T] as in the rule.
    """
    kappa = 0.01
    sol = heat_sols.eigenfunction_decay_square(kappa=kappa)
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    tg = np.linspace(0.0, 0.5, nt)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = np.stack([sol.u(mesh_x, mesh_y, ti) for ti in tg], axis=-1)
    h_spatial = (1.0 / (n - 1), 1.0 / (n - 1))
    dt = 0.5 / (nt - 1)
    mass = np.array(
        [
            integrate_over_domain(np.take(u, k, axis=-1), h_spatial, periodic=False)
            for k in range(nt)
        ]
    )
    dm_dt = np.gradient(mass, dt, edge_order=2)
    expected = np.zeros(nt)
    for k in range(nt):
        slice_k = np.take(u, k, axis=-1)
        sub = GridField(slice_k, h=h_spatial, periodic=False, backend="fd")
        lap = sub.laplacian().values()
        expected[k] = kappa * integrate_over_domain(lap, h_spatial, periodic=False)
    err = float(np.sqrt(np.sum((dm_dt - expected) ** 2) * dt))
    denom = max(float(np.sqrt(np.sum(expected**2) * dt)), 1e-12)
    return err / denom


def _measure_wave_con_002_hd_fd(n: int, nt: int) -> float:
    """PH-CON-002 floor for wave + hD + fd4: relative energy drift.

    Standing wave eigenfunction on [0,1]^2. Matches the rule's IBP-based
    energy formulation: 0.5 * (u_t^2 - c^2 * u * Laplacian u) integrated
    via trapezoidal quadrature (endpoint-inclusive for hD).
    """
    c = 1.0
    sol = wave_sols.standing_wave_square(c=c)
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    tg = np.linspace(0.0, 0.5, nt)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = np.stack([sol.u(mesh_x, mesh_y, ti) for ti in tg], axis=-1)
    h_spatial = (1.0 / (n - 1), 1.0 / (n - 1))
    dt = 0.5 / (nt - 1)
    u_t = np.gradient(u, dt, axis=-1, edge_order=2)
    energies = np.empty(nt)
    for k in range(nt):
        slice_k = np.take(u, k, axis=-1)
        slice_ut = np.take(u_t, k, axis=-1)
        sub = GridField(slice_k, h=h_spatial, periodic=False, backend="fd")
        lap = sub.laplacian().values()
        density = 0.5 * (slice_ut**2) - 0.5 * (c**2) * slice_k * lap
        energies[k] = integrate_over_domain(density, h_spatial, periodic=False)
    e0 = float(energies[0])
    denom = max(abs(e0), 1e-12)
    return float(np.max(np.abs(energies - e0)) / denom)


def _measure_wave_con_002_periodic_spectral(n: int, nt: int) -> float:
    """PH-CON-002 floor for a periodic traveling wave + spectral.

    Reviewer regression: a correct periodic traveling wave must not show
    seam drift under PH-CON-002. Uses the IBP identity plus rectangle
    quadrature end-to-end.
    """
    c = 1.0
    length = 2 * np.pi
    sol = wave_sols.periodic_traveling(c=c, length=length)
    xg = np.linspace(0.0, length, n, endpoint=False)
    yg = np.linspace(0.0, length, n, endpoint=False)
    tg = np.linspace(0.0, 0.5, nt)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = np.stack([sol.u(mesh_x, mesh_y, ti) for ti in tg], axis=-1)
    h_spatial = (length / n, length / n)
    dt = 0.5 / (nt - 1)
    u_t = np.gradient(u, dt, axis=-1, edge_order=2)
    energies = np.empty(nt)
    for k in range(nt):
        slice_k = np.take(u, k, axis=-1)
        slice_ut = np.take(u_t, k, axis=-1)
        sub = GridField(slice_k, h=h_spatial, periodic=True, backend="spectral")
        lap = sub.laplacian().values()
        density = 0.5 * (slice_ut**2) - 0.5 * (c**2) * slice_k * lap
        energies[k] = integrate_over_domain(density, h_spatial, periodic=True)
    e0 = float(energies[0])
    denom = max(abs(e0), 1e-12)
    return float(np.max(np.abs(energies - e0)) / denom)


def _measure_heat_con_003_hd_fd(n: int, nt: int) -> float:
    """PH-CON-003 floor for hD+fd heat: positive dE/dt / max E.

    Eigenfunction decay: energy is strictly decreasing analytically; the
    floor is whatever positive dE/dt the numerical derivative leaks.
    """
    kappa = 0.01
    sol = heat_sols.eigenfunction_decay_square(kappa=kappa)
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    tg = np.linspace(0.0, 0.5, nt)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = np.stack([sol.u(mesh_x, mesh_y, ti) for ti in tg], axis=-1)
    h_spatial = (1.0 / (n - 1), 1.0 / (n - 1))
    dt = 0.5 / (nt - 1)
    energy = np.array(
        [
            integrate_over_domain(np.take(u, k, axis=-1) ** 2, h_spatial, periodic=False)
            for k in range(nt)
        ]
    )
    de_dt = np.gradient(energy, dt, edge_order=2)
    max_growth = max(0.0, float(np.max(de_dt)))
    energy_scale = max(float(np.max(energy)), 1e-12)
    return max_growth / energy_scale


def _measure_bc_l2_rel_self_check(n: int) -> float:
    """PH-BC-001 floor for boundary L2-rel self-check.

    The field's boundary trace is compared against itself, so the error
    is literally zero. Record machine epsilon as the floor."""
    sol = laplace_sols.harmonic_polynomial_square()
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = sol.u(mesh_x, mesh_y)
    h = 1.0 / (n - 1)
    field = GridField(u, h=h, periodic=False, backend="fd")
    boundary = field.values_on_boundary()
    err = float(np.linalg.norm(boundary - boundary) / np.sqrt(max(len(boundary), 1)))
    gnorm = float(np.linalg.norm(boundary) / np.sqrt(max(len(boundary), 1)))
    ratio = err / gnorm if gnorm > 0 else err
    return max(ratio, float(np.finfo(np.float64).eps))


def main() -> None:
    entries: list[FloorEntry] = [
        FloorEntry(
            rule="PH-RES-001",
            pde="laplace",
            grid_shape=(64, 64),
            method="fd4",
            norm="L2",
            measured=_measure_laplace_fd_l2(64),
            analytical_solution="harmonic_polynomial_square",
        ),
        FloorEntry(
            rule="PH-RES-001",
            pde="laplace",
            grid_shape=(64, 64),
            method="spectral",
            norm="H-1",
            measured=_measure_laplace_spectral_h_minus_one(64),
            analytical_solution="constant_on_torus",
        ),
        FloorEntry(
            rule="PH-RES-001",
            pde="poisson",
            grid_shape=(64, 64),
            method="spectral",
            norm="H-1",
            measured=_measure_poisson_spectral_h_minus_one(64),
            analytical_solution="periodic_sin_sin",
        ),
        FloorEntry(
            rule="PH-BC-001",
            pde="laplace",
            grid_shape=(64, 64),
            method="fd4",
            norm="L2-rel",
            measured=_measure_bc_l2_rel_self_check(64),
            analytical_solution="harmonic_polynomial_square",
        ),
        FloorEntry(
            rule="PH-RES-001",
            pde="heat",
            grid_shape=(64, 64, 16),
            method="spectral",
            norm="Bochner-H-1",
            measured=_measure_heat_periodic_spectral_bochner(64, 16),
            analytical_solution="periodic_cos_cos",
        ),
        FloorEntry(
            rule="PH-RES-001",
            pde="wave",
            grid_shape=(64, 64, 32),
            method="fd4",
            norm="Bochner-L2",
            measured=_measure_wave_hd_fd_bochner(64, 32),
            analytical_solution="standing_wave_square",
        ),
        FloorEntry(
            rule="PH-CON-001",
            pde="heat",
            grid_shape=(64, 64, 16),
            method="spectral",
            norm="relative",
            measured=_measure_heat_con_001_periodic_spectral(64, 16),
            analytical_solution="periodic_cos_cos",
        ),
        FloorEntry(
            rule="PH-CON-001",
            pde="heat",
            grid_shape=(64, 64, 32),
            method="fd4",
            norm="relative_L2_over_T",
            measured=_measure_heat_con_001_hd_fd(64, 32),
            analytical_solution="eigenfunction_decay_square",
        ),
        FloorEntry(
            rule="PH-CON-002",
            pde="wave",
            grid_shape=(64, 64, 32),
            method="fd4",
            norm="relative",
            measured=_measure_wave_con_002_hd_fd(64, 32),
            analytical_solution="standing_wave_square",
        ),
        FloorEntry(
            rule="PH-CON-002",
            pde="wave",
            grid_shape=(64, 64, 32),
            method="spectral",
            norm="relative",
            measured=_measure_wave_con_002_periodic_spectral(64, 32),
            analytical_solution="periodic_traveling",
        ),
        FloorEntry(
            rule="PH-CON-003",
            pde="heat",
            grid_shape=(32, 32, 16),
            method="fd4",
            norm="relative",
            measured=_measure_heat_con_003_hd_fd(32, 16),
            analytical_solution="eigenfunction_decay_square",
        ),
        FloorEntry(
            rule="PH-CON-003",
            pde="heat",
            grid_shape=(64, 64, 16),
            method="spectral",
            norm="relative",
            measured=_measure_heat_con_003_periodic_spectral(64, 16),
            analytical_solution="periodic_cos_cos",
        ),
    ]

    env = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    try:
        import torch

        env["torch"] = torch.__version__
    except ImportError:
        env["torch"] = "absent"

    print(
        json.dumps(
            {"environment": env, "entries": [asdict(e) for e in entries]},
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
