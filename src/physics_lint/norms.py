"""Norm computations for residuals and field differences.

- l2_grid: trapezoidal L^2 on a uniform Cartesian grid (half-weights at edges)
- trapezoidal_integral: weighted L^1 integral via the same trapezoidal rule
- h_minus_one_spectral: sqrt(sum_{k != 0} |u_hat|^2 / |k|^2) on periodic grids
- h_minus_one_fe: (Task 8, conditional on scikit-fem spike)
- bochner_l2_h_minus_one: Bochner L^2(0,T; H^-1) norm for time-dependent
  residuals on periodic spatial grids; trapezoidal quadrature in t.
- bochner_l2_fallback: Bochner L^2(0,T; L^2) fallback for non-periodic
  grids where the spectral H^-1 is not available. NOT variationally
  correct — callers must mark the recommended_norm as "Bochner-L2"
  so downstream reports surface that the measurement is approximate.
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


def _bochner_trapz_weights(n_steps: int, dt: float) -> np.ndarray:
    """Trapezoidal quadrature weights for n_steps endpoint-inclusive samples.

    Returns a length-n_steps array where interior weights are dt and the
    two endpoint weights are dt/2, so the weights sum to (n_steps - 1) * dt,
    matching the total physical time span. This is the right rule for the
    endpoint-inclusive time sampling the loader produces via
    _compute_h_from_spec (dt = (t_hi - t_lo) / (n_t - 1)). Using a flat
    weight of dt per slice would overintegrate by a factor of
    n_steps / (n_steps - 1), the bug PH-RES-001 heat/wave residuals had
    before this fix (e.g. 8/7 ≈ 1.069 for the common nt=8 case).
    """
    if n_steps < 2:
        raise ValueError(f"Bochner quadrature requires n_steps >= 2; got {n_steps}")
    weights = np.full(n_steps, dt, dtype=float)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


def bochner_l2_h_minus_one(
    r_series: np.ndarray,
    *,
    spatial_h: tuple[float, ...],
    dt: float,
) -> float:
    """sqrt(integral_0^T ||r(., t)||_{H^-1}^2 dt) via trapezoidal rule.

    Per design doc §7.4. The caller provides a residual time series with
    time as the *last* axis and endpoint-inclusive sampling (dt is the
    uniform spacing between successive samples, so an n_steps-long series
    spans a physical interval of length (n_steps - 1) * dt).

    Uses a spatial H^-1 spectral norm per slice, so **only valid on
    periodic spatial grids** — non-periodic callers must use
    bochner_l2_fallback instead. h_minus_one_spectral drops the DC mode,
    which silently zeroes any purely constant-in-space residual; on a
    non-periodic grid that's a physics false-negative (e.g. u(x,y,t)=t
    on hD reports residual 0 despite u_t = 1 everywhere).

    Args:
        r_series: residual time series with time as the LAST axis. Shape
            (N_0, N_1, [N_2,] n_steps). At least one spatial axis is required.
        spatial_h: per-spatial-axis spacings. Length must equal
            r_series.ndim - 1 so that each slice r_series[..., k] has a
            well-formed spacing tuple for h_minus_one_spectral.
        dt: uniform time-axis spacing (between adjacent samples, not the
            total span).

    Returns:
        Bochner norm of r.
    """
    if r_series.ndim < 2:
        raise ValueError(
            f"bochner_l2_h_minus_one requires at least one spatial axis plus "
            f"a time axis; got r_series.ndim={r_series.ndim}"
        )
    if len(spatial_h) != r_series.ndim - 1:
        raise ValueError(
            f"spatial_h length {len(spatial_h)} must equal r_series.ndim - 1 ({r_series.ndim - 1})"
        )
    n_steps = r_series.shape[-1]
    weights = _bochner_trapz_weights(n_steps, dt)
    total_sq = 0.0
    for k in range(n_steps):
        slice_k = np.take(r_series, k, axis=-1)
        total_sq += float(weights[k]) * (h_minus_one_spectral(slice_k, spatial_h) ** 2)
    return float(np.sqrt(total_sq))


def bochner_l2_fallback(
    r_series: np.ndarray,
    *,
    spatial_h: tuple[float, ...],
    dt: float,
) -> float:
    """sqrt(integral_0^T ||r(., t)||_{L^2}^2 dt) via trapezoidal rule.

    Non-periodic fallback for heat/wave residuals. L^2 in space is not
    the variationally-correct norm for the parabolic setting (H^-1 is),
    but L^2 is a strict upper bound on the H^-1 norm and is defined on
    non-periodic grids, so using it on hD / hN / generic Dirichlet
    configurations gives a conservative pass/fail decision while we wait
    for the Week-3 scikit-fem H^-1 path. The DC-mode false negative in
    h_minus_one_spectral's periodic formula cannot happen here: l2_grid
    is a full-domain integral that picks up every spatial mode.

    Callers MUST label the recommended_norm as "Bochner-L2" so the
    report distinguishes this from the (tighter) Bochner-H-1 result.

    Same shape/quadrature convention as bochner_l2_h_minus_one.
    """
    if r_series.ndim < 2:
        raise ValueError(
            f"bochner_l2_fallback requires at least one spatial axis plus "
            f"a time axis; got r_series.ndim={r_series.ndim}"
        )
    if len(spatial_h) != r_series.ndim - 1:
        raise ValueError(
            f"spatial_h length {len(spatial_h)} must equal r_series.ndim - 1 ({r_series.ndim - 1})"
        )
    n_steps = r_series.shape[-1]
    weights = _bochner_trapz_weights(n_steps, dt)
    total_sq = 0.0
    for k in range(n_steps):
        slice_k = np.take(r_series, k, axis=-1)
        total_sq += float(weights[k]) * (l2_grid(slice_k, spatial_h) ** 2)
    return float(np.sqrt(total_sq))
