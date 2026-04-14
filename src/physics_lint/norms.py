"""Norm computations for residuals and field differences.

Populated incrementally:
- l2_grid: Task 4 (GridField derivatives + trapezoidal L2)
- h_minus_one_spectral: Task 4
- h_minus_one_fe: Task 8 (conditional on scikit-fem spike)
- bochner_l2_h_minus_one: Week 2 (heat/wave)
"""

from __future__ import annotations

import numpy as np


def l2_grid(u: np.ndarray, h: float | tuple[float, ...]) -> float:
    """Trapezoidal L2 norm on a uniform Cartesian grid.

    Half-weights boundary points per the trapezoidal rule.

    Args:
        u: Field values on the grid. Shape (Nx,), (Nx, Ny), or (Nx, Ny, Nz).
        h: Uniform spacing. Scalar if isotropic, tuple of (hx, hy, ...) otherwise.

    Returns:
        sqrt(integral of |u|^2 dx) over the grid.
    """
    raise NotImplementedError("Populated in Task 4.")
