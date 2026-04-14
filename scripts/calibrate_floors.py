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

from physics_lint.analytical import laplace as laplace_sols
from physics_lint.analytical import poisson as poisson_sols
from physics_lint.field import GridField
from physics_lint.norms import h_minus_one_spectral, l2_grid


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
