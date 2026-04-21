"""MMS helpers for Task 4 Layer 2 norm-equivalence calibration.

Responsibility: compute true H1 error of a closed-form perturbation against
the MMS reference solution using analytical derivatives via central
differences on the perturbation callable and composite quadrature on the
grid. Supports both non-periodic [0,1]^2 (trapezoidal + one-sided boundary
diffs) and periodic [0, L]^2 (periodic quadrature + np.roll-based central
diffs) configurations.

Rev 1.7: the original scikit-fem P1 reference (`poisson_p1_reference_h1_error`)
was removed. The MMS setup has a known u_exact, so analytical H1 is both
simpler and more accurate than any FE approximation.

Rev 1.7.2 (Path A' for Task 4): extended to support periodic grids via a
`periodic` flag. Task 4 Layer 2 now characterizes both PH-RES-001 code
paths: periodic+spectral emits H^-1 (norm-equivalent to H^1), non-periodic
+FD emits L^2 (not norm-equivalent across frequencies by construction).
The same MMS helper is reused for both paths with the periodic flag
switching gradient + quadrature conventions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def mms_perturbation_h1_error(
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    *,
    perturbation: Callable[[np.ndarray, np.ndarray], np.ndarray],
    periodic: bool = False,
) -> float:
    """H1 error of u_pert = u_exact + perturbation against u_exact, analytically.

    H1 error squared = integral |u_pert - u_exact|^2 + integral |grad(u_pert - u_exact)|^2
                     = integral |perturbation|^2 + integral |grad perturbation|^2.

    Computes analytical gradients via central differences on the smooth
    closed-form perturbation callable, then integrates with either
    composite trapezoidal (non-periodic, endpoint-inclusive) or the
    periodic-grid rectangle rule (endpoint-exclusive on [0, L)).

    When `periodic=True`, gradients use `np.roll` (periodic continuation).
    When `periodic=False`, central diffs in [1:-1] with one-sided at
    boundaries; trapezoidal quadrature on endpoint-inclusive grid.
    """
    nx, ny = mesh_x.shape
    if nx != ny:
        raise ValueError(f"square grid required; got mesh_x.shape {mesh_x.shape}")
    hx = float(mesh_x[1, 0] - mesh_x[0, 0])
    hy = float(mesh_y[0, 1] - mesh_y[0, 0])

    p = perturbation(mesh_x, mesh_y)

    if periodic:
        dp_dx = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2.0 * hx)
        dp_dy = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2.0 * hy)
        # Periodic grid: rectangle rule (sum * cell area) is exact for
        # full-period samples; trapezoidal would double-count endpoints.
        l2_sq = float(np.sum(p**2) * hx * hy)
        grad_l2_sq = float(np.sum(dp_dx**2 + dp_dy**2) * hx * hy)
    else:
        dp_dx = np.zeros_like(p)
        dp_dy = np.zeros_like(p)
        dp_dx[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2.0 * hx)
        dp_dy[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * hy)
        dp_dx[0, :] = (p[1, :] - p[0, :]) / hx
        dp_dx[-1, :] = (p[-1, :] - p[-2, :]) / hx
        dp_dy[:, 0] = (p[:, 1] - p[:, 0]) / hy
        dp_dy[:, -1] = (p[:, -1] - p[:, -2]) / hy
        l2_sq = float(np.trapz(np.trapz(p**2, dx=hy, axis=1), dx=hx, axis=0))
        grad_l2_sq = float(np.trapz(np.trapz(dp_dx**2 + dp_dy**2, dx=hy, axis=1), dx=hx, axis=0))

    return float(np.sqrt(l2_sq + grad_l2_sq))


# Back-compat alias (retained for imports that land mid-migration).
mms_sin_sin_h1_error = mms_perturbation_h1_error
