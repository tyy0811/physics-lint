"""GridField: regular Cartesian grid with FD or spectral derivative backends.

4th-order central FD per Fornberg 1988 (design doc §3.2). Spectral branch
selected automatically when periodic=True unless user forces backend="fd".
"""

from __future__ import annotations

import numbers
from typing import Literal

import numpy as np

from physics_lint.field._base import Field

# 4th-order central finite difference stencil for the second derivative.
# f''(x_i) ≈ (-f[i-2] + 16 f[i-1] - 30 f[i] + 16 f[i+1] - f[i+2]) / (12 h^2)
#
# Non-periodic boundaries: the outer 2 layers use explicit one-sided /
# off-center 3- and 4-point second-derivative formulas with O(h^2)
# truncation error. The 4th-order rate only holds in the interior
# [2:-2] band. The rules that consume the Laplacian weight outer-band
# contributions via half-weight trapezoidal integration, but the
# pointwise values are still uniformly second-order accurate.

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
        # numbers.Real covers Python int/float AND all numpy scalar types
        # (np.float32, np.int32, etc.) — np.isscalar would also accept
        # strings and bytes, which we don't want. 0-d arrays
        # (np.array(0.125)) are rejected here — callers should .item() them
        # at the call site to make their intent explicit.
        if isinstance(h, numbers.Real):
            self.h: tuple[float, ...] = (float(h),) * ndim
        else:
            # Reject strings/bytes explicitly: they're iterable, so the
            # generic iterable branch below would step into them char-by-char
            # and produce a confusing error.
            if isinstance(h, str | bytes):
                raise TypeError(
                    f"h must be a scalar or an iterable of length {ndim}; got {type(h).__name__}"
                )
            try:
                h_tuple = tuple(float(hi) for hi in h)  # type: ignore[union-attr]
            except TypeError as exc:
                raise TypeError(
                    f"h must be a scalar or an iterable of length {ndim}; got {type(h).__name__}"
                ) from exc
            if len(h_tuple) != ndim:
                raise ValueError(f"h tuple length ({len(h_tuple)}) must match values.ndim ({ndim})")
            self.h = h_tuple
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

    Non-periodic: interior [2:-2] uses the central 5-point stencil
    (4th-order). The outer 2 layers along the axis use explicit
    one-sided / off-center formulas, all O(h^2):

        u''[0]  = ( 2 u[0] - 5 u[1] + 4 u[2] -   u[3]) / h^2
                  (4-point forward; leading error  -11/12 h^2 u'''')
        u''[1]  = (   u[0] - 2 u[1] +   u[2]         ) / h^2
                  (3-point central about index 1; leading error
                   -1/12 h^2 u'''')
        u''[-2] = (   u[-3] - 2 u[-2] +   u[-1]      ) / h^2
                  (3-point central about index -2)
        u''[-1] = ( 2 u[-1] - 5 u[-2] + 4 u[-3] -   u[-4]) / h^2
                  (4-point backward)

    The 4th-order rate only holds in the interior [2:-2] band; the
    outer band degrades to 2nd-order as documented above.
    """
    n = u.shape[axis]
    if n < 5:
        raise ValueError(f"4th-order FD requires at least 5 points along axis {axis}; got {n}")

    if periodic:
        out = np.zeros_like(u)
        for offset, coef in zip((-2, -1, 0, 1, 2), _FD4_STENCIL, strict=True):
            out = out + coef * np.roll(u, -offset, axis=axis)
        return out / (h**2)

    # Non-periodic: central stencil in interior, explicit one-sided at edges.
    out = np.zeros_like(u)
    # Interior: slice [2:-2] along the target axis.
    slicers_out = [slice(None)] * u.ndim
    slicers_out[axis] = slice(2, -2)
    for offset, coef in zip((-2, -1, 0, 1, 2), _FD4_STENCIL, strict=True):
        slicers_in = [slice(None)] * u.ndim
        slicers_in[axis] = slice(2 + offset, n - 2 + offset if n - 2 + offset != 0 else None)
        out[tuple(slicers_out)] = out[tuple(slicers_out)] + coef * u[tuple(slicers_in)]
    out[tuple(slicers_out)] = out[tuple(slicers_out)] / (h**2)

    # Helper: build a length-1 slicer along `axis` selecting index `i`,
    # with full slices on all other axes.
    def _at(i: int) -> tuple[slice | int, ...]:
        s: list[slice | int] = [slice(None)] * u.ndim
        s[axis] = i
        return tuple(s)

    h2 = h * h
    # Index 0: 4-point forward, O(h^2).
    out[_at(0)] = (2.0 * u[_at(0)] - 5.0 * u[_at(1)] + 4.0 * u[_at(2)] - u[_at(3)]) / h2
    # Index 1: 3-point central about index 1, O(h^2).
    out[_at(1)] = (u[_at(0)] - 2.0 * u[_at(1)] + u[_at(2)]) / h2
    # Index -2: 3-point central about index -2, O(h^2).
    out[_at(-2)] = (u[_at(-3)] - 2.0 * u[_at(-2)] + u[_at(-1)]) / h2
    # Index -1: 4-point backward, O(h^2).
    out[_at(-1)] = (2.0 * u[_at(-1)] - 5.0 * u[_at(-2)] + 4.0 * u[_at(-3)] - u[_at(-4)]) / h2
    return out
