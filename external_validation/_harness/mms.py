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

Rev 1.9 (Task 12 — PH-NUM-002): adds Case A harness-layer observed-order
anchor. The rule ships FD4 + 2nd-order-boundary-band + spectral backends;
the harness layer uses a *pure-interior* 2nd-order central-difference
Laplacian so the measured slope is the textbook p_obs = 2 (no boundary
degradation, no 4th-order interior masking). This establishes the
observed-order methodology at a pedagogically clean rate independent of
the rule's FD4 stencil choices — mirrors Task 9's harness-authoritative
pattern for wave energy.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# numpy 2.0 removed np.trapz in favor of np.trapezoid. pyproject.toml supports
# numpy>=1.26 so both APIs must be reachable; prefer the numpy-2.x name when
# available, fall back to the legacy name on numpy 1.26.x. Must use hasattr
# + ternary rather than getattr(..., default=np.trapz) because the getattr
# default is evaluated eagerly and would itself raise AttributeError on
# numpy 2.x where np.trapz has been removed.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


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
        l2_sq = float(_trapz(_trapz(p**2, dx=hy, axis=1), dx=hx, axis=0))
        grad_l2_sq = float(_trapz(_trapz(dp_dx**2 + dp_dy**2, dx=hy, axis=1), dx=hx, axis=0))

    return float(np.sqrt(l2_sq + grad_l2_sq))


# Back-compat alias (retained for imports that land mid-migration).
mms_sin_sin_h1_error = mms_perturbation_h1_error


# ---------------------------------------------------------------------------
# Case A harness-layer 2nd-order-FD observed-order anchor (Task 12, PH-NUM-002)
# ---------------------------------------------------------------------------
#
# Purpose: anchor the observed-order methodology p_obs = log2(r_h / r_{h/2})
# at the textbook clean-second-order rate p = 2 independent of PH-NUM-002's
# production path (which uses FD4 with a 2nd-order boundary band, and
# spectral + periodic which saturates). This helper is authoritative for
# the F1 identity "2nd-order FD residual on a smooth harmonic u converges
# as O(h^2)" without entanglement with the rule's shipped FD4 stencil.
#
# Scope:
# - Smooth non-periodic harmonic u on [0, 1]^2 (e.g. exp(x) cos(y),
#   sin(pi x) sinh(pi y)).
# - Interior-only 2nd-order central-difference Laplacian; L^2 norm taken
#   over the interior grid (excluding the boundary ring) so no one-sided
#   or 2nd-order-boundary stencil enters the measurement.
# - Harmonic-polynomial fixtures (e.g. x^2 - y^2) are *excluded* because
#   2nd-order FD is exact on polynomials of degree <= 2 -> residuals
#   saturate at roundoff and p_obs becomes noise.


def laplacian_fd2_interior(u: np.ndarray, hx: float, hy: float) -> np.ndarray:
    """Pure interior 2nd-order central-difference Laplacian on a 2D grid.

    Returns a zero-padded array of the same shape as `u` with the Laplacian
    filled on `[1:-1, 1:-1]` only. No one-sided boundary stencil, so the
    measured residual reflects pure O(h^2) interior truncation error on a
    smooth harmonic u.
    """
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (hx * hx) + (
        u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]
    ) / (hy * hy)
    return lap


def interior_l2_norm(field: np.ndarray, hx: float, hy: float) -> float:
    """L^2 norm over the interior (exclude boundary ring). Rectangle rule."""
    interior = field[1:-1, 1:-1]
    return float(np.sqrt(np.sum(interior * interior) * hx * hy))


def mms_observed_order_fd2(
    fixture: Callable[[int], tuple[np.ndarray, float, float]],
    *,
    n_coarse: int,
    n_fine: int,
) -> tuple[float, float, float]:
    """Observed-order (Roy 2005 formula) on the 2nd-order interior-FD path.

    Returns ``(p_obs, r_coarse, r_fine)`` where `r` is the interior L^2 norm
    of the 2nd-order FD Laplacian applied to `fixture(N)`. The caller picks
    a smooth *harmonic* u so the true Laplacian is identically zero and
    the measured residual is pure truncation.

    `fixture(n)` must return `(u, hx, hy)`.
    """
    uc, hxc, hyc = fixture(n_coarse)
    uf, hxf, hyf = fixture(n_fine)
    rc = interior_l2_norm(laplacian_fd2_interior(uc, hxc, hyc), hxc, hyc)
    rf = interior_l2_norm(laplacian_fd2_interior(uf, hxf, hyf), hxf, hyf)
    if rc <= 0.0 or rf <= 0.0:
        return float("inf"), rc, rf
    return float(np.log2(rc / rf)), rc, rf
