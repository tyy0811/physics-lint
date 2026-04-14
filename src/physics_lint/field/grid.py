"""GridField: regular Cartesian grid with FD or spectral derivative backends.

4th-order central FD per Fornberg 1988 (design doc §3.2). Spectral branch
selected automatically when periodic=True unless user forces backend="fd".
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from physics_lint.field._base import Field

# 4th-order central finite difference stencil for the second derivative.
# f''(x_i) ≈ (-f[i-2] + 16 f[i-1] - 30 f[i] + 16 f[i+1] - f[i+2]) / (12 h^2)
# One-sided 3rd-order variant for non-periodic boundaries: interior-only at
# depth 2; edges use numpy.gradient as a 2nd-order fallback for the outermost
# two layers (the rules that consume the Laplacian exclude the outer band via
# half-weight trapezoidal integration, so residual contributions there are
# suppressed but not zero).

_FD4_STENCIL = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0


class GridField(Field):
    """Field stored as a NumPy array on a uniform Cartesian grid."""

    def __init__(
        self,
        values: np.ndarray,
        h: float | tuple[float, ...],
        *,
        periodic: bool,
        backend: Literal["fd", "spectral", "auto"] = "auto",
    ) -> None:
        self._values = np.ascontiguousarray(values)
        ndim = self._values.ndim
        if isinstance(h, int | float):
            self.h: tuple[float, ...] = (float(h),) * ndim
        else:
            if len(h) != ndim:
                raise ValueError(f"h tuple length ({len(h)}) must match values.ndim ({ndim})")
            self.h = tuple(float(hi) for hi in h)
        self.periodic = bool(periodic)
        if backend == "auto":
            self.backend: Literal["fd", "spectral"] = "spectral" if self.periodic else "fd"
        else:
            self.backend = backend

    def values(self) -> np.ndarray:
        return self._values

    def at(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("GridField.at() lands in a later task.")

    def grad(self) -> GridField:
        raise NotImplementedError("Lands in Step 5 of this task.")

    def laplacian(self) -> GridField:
        if self.backend == "spectral":
            raise NotImplementedError("Spectral Laplacian lands in Task 4.")
        u = self._values
        out = np.zeros_like(u)
        for axis, h_ax in enumerate(self.h):
            out = out + _fd4_second_derivative(u, axis=axis, h=h_ax, periodic=self.periodic)
        return GridField(out, h=self.h, periodic=self.periodic, backend=self.backend)

    def integrate(self, weight: Field | None = None) -> float:
        raise NotImplementedError("Lands in Task 4 alongside l2_grid.")

    def values_on_boundary(self) -> np.ndarray:
        raise NotImplementedError("Lands in Task 6.")


def _fd4_second_derivative(u: np.ndarray, *, axis: int, h: float, periodic: bool) -> np.ndarray:
    """4th-order central FD second derivative along a single axis.

    Periodic: np.roll wraps the stencil around the boundary — exact on
    smooth periodic inputs to ~h^4.

    Non-periodic: interior uses the central stencil; outer 2 layers fall
    back to a 2nd-order one-sided form via numpy.gradient twice. This is
    a degradation — the 4th-order rate only holds in the interior.
    """
    n = u.shape[axis]
    if n < 5:
        raise ValueError(f"4th-order FD requires at least 5 points along axis {axis}; got {n}")

    if periodic:
        out = np.zeros_like(u)
        for offset, coef in zip((-2, -1, 0, 1, 2), _FD4_STENCIL, strict=True):
            out = out + coef * np.roll(u, -offset, axis=axis)
        return out / (h**2)

    # Non-periodic: central stencil in interior, 2nd-order fallback at edges.
    out = np.zeros_like(u)
    # Interior: slice [2:-2] along the target axis
    slicers_out = [slice(None)] * u.ndim
    slicers_out[axis] = slice(2, -2)
    for offset, coef in zip((-2, -1, 0, 1, 2), _FD4_STENCIL, strict=True):
        slicers_in = [slice(None)] * u.ndim
        slicers_in[axis] = slice(2 + offset, n - 2 + offset if n - 2 + offset != 0 else None)
        out[tuple(slicers_out)] = out[tuple(slicers_out)] + coef * u[tuple(slicers_in)]
    out[tuple(slicers_out)] = out[tuple(slicers_out)] / (h**2)
    # Edge fallback: numpy.gradient twice (2nd-order, still converges)
    first = np.gradient(u, h, axis=axis, edge_order=2)
    second = np.gradient(first, h, axis=axis, edge_order=2)
    edge_front = [slice(None)] * u.ndim
    edge_front[axis] = slice(0, 2)
    edge_back = [slice(None)] * u.ndim
    edge_back[axis] = slice(-2, None)
    out[tuple(edge_front)] = second[tuple(edge_front)]
    out[tuple(edge_back)] = second[tuple(edge_back)]
    return out
