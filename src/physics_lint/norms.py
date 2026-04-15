"""Norm computations for residuals and field differences.

- l2_grid: trapezoidal L^2 on a uniform Cartesian grid (half-weights at edges)
- trapezoidal_integral: weighted L^1 integral via the same trapezoidal rule
- h_minus_one_spectral: sqrt(sum_{k != 0} |u_hat|^2 / |k|^2) on periodic grids
- h_minus_one_fe: (Task 8, conditional on scikit-fem spike)
- bochner_l2_h_minus_one: (Week 2, heat/wave)
"""

from __future__ import annotations

import numbers

import numpy as np


def trapezoidal_integral(u: np.ndarray, h: tuple[float, ...]) -> float:
    """Multi-dimensional trapezoidal integral on a uniform grid.

    Half-weights all boundary points (1D: 0.5 at i=0 and i=N-1; 2D: quarter
    at the four corners; and so on). The grid is assumed endpoint-inclusive:
    u has shape (N_0, N_1, ...) with physical lengths L_i = (N_i - 1) * h_i.

    Args:
        u: Values on the grid.
        h: Spacing tuple, len == u.ndim.

    Returns:
        sum_i w_i u_i, where w_i are the trapezoidal weights scaled by prod(h).
    """
    if len(h) != u.ndim:
        raise ValueError(f"h length {len(h)} must match u.ndim {u.ndim}")
    weights = np.ones_like(u, dtype=float)
    for axis in range(u.ndim):
        slicer_front: list[slice | int] = [slice(None)] * u.ndim
        slicer_back: list[slice | int] = [slice(None)] * u.ndim
        slicer_front[axis] = 0
        slicer_back[axis] = -1
        weights[tuple(slicer_front)] *= 0.5
        weights[tuple(slicer_back)] *= 0.5
    cell_volume = float(np.prod(h))
    return float(np.sum(weights * u) * cell_volume)


def l2_grid(u: np.ndarray, h: float | tuple[float, ...]) -> float:
    """sqrt(integral |u|^2 dx) on a uniform Cartesian grid.

    Uses trapezoidal_integral with half-weights at boundaries. Assumes the
    grid is endpoint-inclusive (physical lengths L_i = (N_i - 1) * h_i).
    """
    # numbers.Real mirrors the GridField constructor dispatch: accepts Python
    # int/float plus numpy scalar types (np.float32, etc.) without tripping
    # the iterable branch (numpy scalars are not iterable and would raise a
    # confusing TypeError inside the generator expression).
    h_tuple = (float(h),) * u.ndim if isinstance(h, numbers.Real) else tuple(float(hi) for hi in h)
    return float(np.sqrt(trapezoidal_integral(u * u, h_tuple)))


def h_minus_one_spectral(r: np.ndarray, h: float | tuple[float, ...]) -> float:
    """sqrt(sum_{k != 0} |r_hat_k|^2 / |k|^2) on a periodic grid.

    Design doc §3.4 and §7.4. Only valid for periodic boundary conditions;
    the caller must ensure periodicity. The k=0 (DC) mode is excluded because
    the H^-1 norm is not defined on constants (Poincaré argument).

    Scaling note: numpy FFT is unnormalized, so |r_hat_k|^2 carries a factor
    of N^2 relative to the continuous Fourier coefficient. The division by
    N^2 converts back to the normalized spectrum, and the multiplication by
    the physical volume L^d = prod(L_i) = prod(N_i * h_i) converts the
    discrete sum to a Riemann approximation of the spectral integral.
    """
    h_tuple = (float(h),) * r.ndim if isinstance(h, numbers.Real) else tuple(float(hi) for hi in h)

    shape = r.shape
    r_hat = np.fft.fftn(r)
    # Build squared wavenumber grid
    k_sq_total = np.zeros(shape, dtype=float)
    for axis in range(r.ndim):
        k = np.fft.fftfreq(shape[axis], d=h_tuple[axis]) * (2.0 * np.pi)
        shape_broadcast = [1] * r.ndim
        shape_broadcast[axis] = shape[axis]
        k_sq_total = k_sq_total + (k.reshape(shape_broadcast)) ** 2
    # Exclude DC mode
    mask = k_sq_total > 0
    if not np.any(mask):
        return 0.0
    n_total = float(np.prod(shape))
    volume = float(np.prod([s * hi for s, hi in zip(shape, h_tuple, strict=True)]))
    # |r_hat|^2 / N^2 gives the normalized spectrum; dividing by |k|^2 gives
    # the H^-1 weight; summing and multiplying by the physical volume gives
    # the H^-1 squared norm as a Riemann approximation.
    h_minus_one_sq = float(
        np.sum(np.abs(r_hat[mask]) ** 2 / k_sq_total[mask]) / (n_total**2) * volume
    )
    return float(np.sqrt(h_minus_one_sq))
