"""Shared analytical solutions for external-validation anchors.

Reuse rather than reimplement: where `physics_lint.analytical` already has
the PDE solution, import from there. Where this module adds a solution not
needed by the self-test battery (e.g., Task 1's three harmonic polynomials,
Task 4's MMS sin(pi x) sin(pi y)), define it here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class AnalyticalField:
    """Plain 2D analytical field on a rectangular domain.

    Used by per-rule anchors that evaluate a closed-form u(x, y) on a grid
    and feed it to a physics-lint rule.
    """

    name: str
    u: Callable[[np.ndarray, np.ndarray], np.ndarray]


def harmonic_xx_yy() -> AnalyticalField:
    """u(x, y) = x^2 - y^2 - harmonic, used by Task 1."""
    return AnalyticalField(name="x^2 - y^2", u=lambda x, y: x**2 - y**2)


def harmonic_xy() -> AnalyticalField:
    """u(x, y) = x y - harmonic, used by Task 1."""
    return AnalyticalField(name="x*y", u=lambda x, y: x * y)


def harmonic_cubic() -> AnalyticalField:
    """u(x, y) = x^3 - 3 x y^2 - Re[(x+iy)^3], harmonic, used by Task 1."""
    return AnalyticalField(name="x^3 - 3 x y^2", u=lambda x, y: x**3 - 3 * x * y**2)


def non_harmonic_x2_plus_y2() -> AnalyticalField:
    """u(x, y) = x^2 + y^2 - negative control for Task 1 (Laplacian = 4)."""
    return AnalyticalField(name="x^2 + y^2", u=lambda x, y: x**2 + y**2)


def unit_square_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) meshgrid of shape (n, n) covering [0, 1]^2 inclusive."""
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    return np.meshgrid(xs, ys, indexing="ij")
