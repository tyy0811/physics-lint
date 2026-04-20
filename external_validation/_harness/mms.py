"""MMS helpers for Task 4 Layer 2 norm-equivalence calibration.

Responsibility: compute true H1 error of a closed-form perturbation against
the MMS reference solution (`physics_lint.analytical.poisson.sin_sin_mms_square`)
using analytical derivatives via central differences on the perturbation
callable (spec section 2 Task 4 Layer 2 step 7 - no spectral differentiation
because the domain is non-periodic; no Simpson because a 64-point grid gives
63 subintervals per axis, which violates Simpson's parity precondition) and
composite trapezoidal quadrature on the 64x64 grid.

Rev 1.7: the original scikit-fem P1 reference (`poisson_p1_reference_h1_error`)
was removed. The MMS setup has a known u_exact, so analytical H1 is both
simpler and more accurate than any FE approximation. The FE path was never
used by the Layer-2 calibration and its smoke-check implementation had a
real correctness bug (scaled RHS computed but ignored); delete rather than
keep a bug-labeled-as-smoke.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def mms_sin_sin_h1_error(
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    *,
    perturbation: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> float:
    """H1 error of u_pert = u_exact + perturbation against u_exact, analytically.

    H1 error squared = integral |u_pert - u_exact|^2 + integral |grad(u_pert - u_exact)|^2
                     = integral |perturbation|^2 + integral |grad perturbation|^2.

    Computes analytical gradients via central differences on the smooth
    closed-form perturbation callable (equivalent to analytical differentiation
    to O(h^2)), then integrates via composite trapezoidal on the grid.
    """
    nx, ny = mesh_x.shape
    if nx != ny:
        raise ValueError(f"square grid required; got mesh_x.shape {mesh_x.shape}")
    hx = float(mesh_x[1, 0] - mesh_x[0, 0])
    hy = float(mesh_y[0, 1] - mesh_y[0, 0])

    p = perturbation(mesh_x, mesh_y)

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
