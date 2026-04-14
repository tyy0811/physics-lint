"""Field abstract base class.

A `Field` represents a discretized scalar or vector field over a PDE domain,
with a uniform interface for values, evaluation, differentiation, integration,
and boundary trace extraction. Subclasses (GridField, CallableField, MeshField)
implement these methods against their specific storage backends.

Smoothness caveat: physics-lint cannot introspect arbitrary torch.nn.Modules for
non-C2 activations. PH-NUM-003's detection is a best-effort scan of named
submodules (nn.ReLU, nn.LeakyReLU, nn.ELU, etc.) and does NOT detect F.relu
functional calls inside a forward method. For second-order PDEs with callable
fields, treat PH-NUM-003 as a check against a common class of footguns, not
a proof of smoothness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Field(ABC):
    """Abstract field over a discretized domain.

    Shape contract
    --------------
    Let ``d`` be the spatial dimension of the domain and ``*spatial`` denote the
    native discretization shape (e.g. ``(Nx,)``, ``(Nx, Ny)``, or
    ``(Nx, Ny, Nz)``). Concrete subclasses (``GridField``, ``CallableField``,
    ``MeshField``) MUST honor the following conventions so rules consuming
    gradients, Laplacians, and boundary traces agree on shapes.

    - A scalar ``Field``'s ``values()`` has shape ``(*spatial,)``.
    - ``grad()`` returns a Python ``list`` of ``d`` per-axis partial-derivative
      ``Field`` instances. Each element's ``values()`` has shape ``(*spatial,)``
      and the list length equals the spatial dimension ``d``. List index ``i``
      is the partial derivative ``∂/∂x_i``. physics-lint never materializes
      vector ``Field`` instances: the FD-vs-AD cross-check (PH-RES-002) and
      boundary flux rules (PH-BC-002) consume partials component-wise, so
      returning a list rather than a stacked ``(d, *spatial)`` array avoids a
      gratuitous ``VectorField`` abstraction.
    - ``laplacian()`` returns a scalar ``Field`` with ``values()`` of shape
      ``(*spatial,)``.
    - ``at(x)`` takes ``x`` of shape ``(npts, d)`` and returns:
        - ``(npts,)`` if ``self`` is a scalar field, or
        - ``(d_out, npts)`` if ``self`` is a vector field (component axis
          first, matching ``grad()``).
    - ``values_on_boundary()`` returns a 1-D array of boundary-trace values in
      a deterministic ordering. The ABC promises only that the ordering is
      reproducible across calls on the same instance; the exact ordering is a
      concrete-subclass concern.

    Smoothness caveat for PH-NUM-003 is described in the module docstring.
    """

    @abstractmethod
    def values(self) -> np.ndarray:
        """Return the underlying stored values on the native discretization.

        See class docstring 'Shape contract' for the expected shape.
        """

    @abstractmethod
    def at(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the field at arbitrary coordinates ``x`` via interpolation or AD.

        See class docstring 'Shape contract' for ``x`` and return shapes.
        """

    @abstractmethod
    def grad(self) -> list[Field]:
        """Return per-axis partial derivatives as a list of scalar ``Field``s.

        See class docstring 'Shape contract': the returned list has length
        ``d`` (spatial dimension), and element ``i`` is the partial derivative
        ``∂/∂x_i`` as a scalar ``Field`` whose ``values()`` has shape
        ``(*spatial,)``.
        """

    @abstractmethod
    def laplacian(self) -> Field:
        """Return the Laplacian as a new Field (scalar-valued).

        See class docstring 'Shape contract': the returned field's ``values()``
        has shape ``(*spatial,)``.
        """

    @abstractmethod
    def integrate(self, weight: Field | None = None) -> float:
        """Integrate the field (optionally weighted by another Field) over the domain."""

    @abstractmethod
    def values_on_boundary(self) -> np.ndarray:
        """Return the field's trace on the domain boundary for BC checking.

        See class docstring 'Shape contract': a 1-D array in a reproducible
        ordering determined by the concrete subclass.
        """
