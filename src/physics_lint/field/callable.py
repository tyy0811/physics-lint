"""CallableField — Field wrapping a PyTorch callable via autograd.

The callable accepts a Tensor of shape ``(..., d)`` representing d-dimensional
points and returns a Tensor of shape ``(..., 1)`` (scalar field). ``values()``
and ``laplacian()`` materialize the field and its Laplacian on a user-provided
sampling grid, then return GridField-compatible numpy arrays.

Week 1 scope: values + laplacian + integrate + values_on_boundary. ``at()``
and ``grad()`` are deferred to later tasks. ``integrate()`` and
``values_on_boundary()`` delegate to an internally-materialized ``GridField``.

Limitations noted in the design doc §3.3:

- Requires C2 activations for second-order PDEs (PH-NUM-003 warns on
  best-effort submodule scan; does not detect ``F.relu`` in forward).
- For performance, the Week 1 implementation uses ``torch.func.hessian``
  composed with ``vmap`` for per-point Hessians. Future work can specialize
  for 2D/3D with a tailored reverse-mode graph.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable

import numpy as np
import torch
from torch.func import hessian, vmap

from physics_lint.field._base import Field
from physics_lint.field.grid import GridField


class CallableField(Field):
    """Field backed by a torch-callable mapping coords to scalar predictions."""

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        *,
        sampling_grid: torch.Tensor,
        h: float | tuple[float, ...],
        periodic: bool = False,
    ) -> None:
        if not callable(model):
            raise TypeError(f"model must be callable, got {type(model).__name__}")
        self._model = model
        # sampling_grid shape: (*spatial, d) where d = number of spatial dims.
        self._grid = sampling_grid
        self.ndim = int(sampling_grid.shape[-1])
        # numbers.Real covers Python int/float AND all numpy scalar types
        # (np.float32, np.int32, etc.) — np.isscalar would also accept
        # strings and bytes, which we don't want. 0-d arrays
        # (np.array(0.125)) are rejected here — callers should .item() them
        # at the call site to make their intent explicit. This matches the
        # dispatch in GridField.__init__.
        if isinstance(h, numbers.Real):
            self.h: tuple[float, ...] = (float(h),) * self.ndim
        else:
            # Reject strings/bytes explicitly: they're iterable, so the
            # generic iterable branch below would step into them char-by-char
            # and produce a confusing error.
            if isinstance(h, str | bytes):
                raise TypeError(
                    f"h must be a scalar or an iterable of length {self.ndim}; "
                    f"got {type(h).__name__}"
                )
            try:
                h_tuple = tuple(float(hi) for hi in h)  # type: ignore[union-attr]
            except TypeError as exc:
                raise TypeError(
                    f"h must be a scalar or an iterable of length {self.ndim}; "
                    f"got {type(h).__name__}"
                ) from exc
            if len(h_tuple) != self.ndim:
                raise ValueError(
                    f"h tuple length ({len(h_tuple)}) must match sampling_grid "
                    f"spatial dim ({self.ndim})"
                )
            self.h = h_tuple
        self.periodic = bool(periodic)
        self._cached_grid_field: GridField | None = None

    def _materialize(self) -> GridField:
        if self._cached_grid_field is None:
            with torch.no_grad():
                out = self._model(self._grid)
                if out.dim() == self._grid.dim():
                    out = out.squeeze(-1)
                vals = out.detach().cpu().numpy()
            self._cached_grid_field = GridField(vals, h=self.h, periodic=self.periodic)
        return self._cached_grid_field

    def values(self) -> np.ndarray:
        return self._materialize().values()

    def at(self, x: np.ndarray) -> np.ndarray:
        # Delegate to the materialized GridField (no interpolation yet; Week 2+).
        raise NotImplementedError("CallableField.at() lands in Week 2 if needed by rules.")

    def grad(self) -> list[CallableField]:
        raise NotImplementedError("CallableField.grad lands in Week 3 with PH-SYM-003.")

    def laplacian(self) -> GridField:
        """Compute the Laplacian via torch autograd at each sampling-grid point.

        For a 2D grid of shape ``(Nx, Ny, 2)``, flatten to ``(Nx*Ny, 2)``,
        compute the Hessian per point with ``torch.func.hessian`` composed
        with ``vmap``, and sum the diagonal. ``torch.func`` is guaranteed
        present: the package pins ``torch>=2.0`` in ``pyproject.toml``.
        """
        pts = self._grid.reshape(-1, self.ndim).clone().requires_grad_(True)

        def _scalar(p: torch.Tensor) -> torch.Tensor:
            y = self._model(p.unsqueeze(0))
            return y.reshape(())

        hess_fn = vmap(hessian(_scalar))
        hess_all = hess_fn(pts)  # shape (N, d, d)

        trace = hess_all.diagonal(dim1=-2, dim2=-1).sum(-1)  # (N,)
        lap = trace.reshape(self._grid.shape[:-1]).detach().cpu().numpy()
        return GridField(lap, h=self.h, periodic=self.periodic)

    def integrate(self, weight: Field | None = None) -> float:
        return self._materialize().integrate(weight)

    def values_on_boundary(self) -> np.ndarray:
        return self._materialize().values_on_boundary()
