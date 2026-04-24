"""Dirichlet trace-extraction helpers for PH-BC-001 F2 fixtures.

Populated for Task 4. Provides boundary-target construction for
analytic 2D functions, matching the discrete ordering emitted by
`physics_lint.field.GridField.values_on_boundary()`
(`[left | right | bottom | top]` with corners included only in left/right
per grid.py:150-158).

Mathematical context (Evans 2010 §5.5 Theorem 1 trace theorem,
section-level per `TEXTBOOK_AVAILABILITY.md` WARN): the trace operator
`gamma: H^1(Omega) -> H^{1/2}(partial Omega)` is bounded on Lipschitz-
boundary domains. The discrete trace extracted by
`GridField.values_on_boundary()` is this operator's evaluation at the
grid's boundary sample points.

Scope separation (per V1-stub CRITICAL-task pattern applied to
FLAG-level Task 4): this helper supports F2 **Dirichlet-trace**
correctness fixtures only. Neumann normal-derivative extraction would
require first-derivative FD on boundary facets; the production rule
PH-BC-001 checks Dirichlet-type value mismatch (`u - g` on boundary),
not Neumann flux. Neumann fixture handling is deferred to a future
release and is not covered by this helper.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def dirichlet_trace_on_unit_square_grid(
    u_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n: int,
) -> np.ndarray:
    """Build a Dirichlet boundary target from an analytic 2D function.

    Returns the flattened boundary trace of ``u_fn`` evaluated on an
    ``n x n`` uniform grid on `[0, 1]^2`, ordered to match
    `GridField(values_on_boundary)` for ``u.ndim == 2``:
    ``[left (x=0), right (x=n-1), bottom (y=0, interior x), top (y=n-1, interior x)]``.

    Parameters
    ----------
    u_fn : callable (np.ndarray, np.ndarray) -> np.ndarray
        The analytic 2D function `u: (x, y) -> R` evaluated pointwise
        on meshgrid arrays (broadcast-aware).
    n : int
        Uniform grid size per axis. Must be >= 2.

    Returns
    -------
    np.ndarray of shape (2 * n + 2 * n - 4,) containing the boundary
    trace values in GridField ordering.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2; got {n}")
    xs = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    u_grid = u_fn(mesh_x, mesh_y)
    left = u_grid[0, :]
    right = u_grid[-1, :]
    bottom = u_grid[1:-1, 0]
    top = u_grid[1:-1, -1]
    return np.concatenate([left, right, bottom, top])
