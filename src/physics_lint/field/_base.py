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
    """Abstract field over a discretized domain."""

    @abstractmethod
    def values(self) -> np.ndarray:
        """Return the underlying stored values on the native discretization."""

    @abstractmethod
    def at(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the field at arbitrary coordinates x via interpolation or AD."""

    @abstractmethod
    def grad(self) -> Field:
        """Return the gradient as a new Field (vector-valued)."""

    @abstractmethod
    def laplacian(self) -> Field:
        """Return the Laplacian as a new Field (scalar-valued)."""

    @abstractmethod
    def integrate(self, weight: Field | None = None) -> float:
        """Integrate the field (optionally weighted by another Field) over the domain."""

    @abstractmethod
    def values_on_boundary(self) -> np.ndarray:
        """Return the field's trace on the domain boundary for BC checking."""
