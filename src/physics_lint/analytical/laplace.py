"""Analytical solutions for Laplace's equation (Delta u = 0)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class LaplaceSolution:
    name: str
    u: Callable[[np.ndarray, np.ndarray], np.ndarray]
    laplacian: Callable[[np.ndarray, np.ndarray], np.ndarray]


def harmonic_polynomial_square() -> LaplaceSolution:
    """u(x,y) = x^2 - y^2 on [0,1]^2 with Dirichlet trace. Harmonic."""
    return LaplaceSolution(
        name="harmonic_polynomial_square",
        u=lambda x, y: x**2 - y**2,
        laplacian=lambda x, y: np.zeros_like(x),
    )


def eigen_trace_square(n: int = 1) -> LaplaceSolution:
    """u = sin(n pi x) sinh(n pi y) / sinh(n pi) on [0,1]^2 with inhomogeneous
    Dirichlet trace (0 on three sides, sin(n pi x) on y=1). Still harmonic."""

    def u(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sin(n * np.pi * x) * np.sinh(n * np.pi * y) / np.sinh(n * np.pi)

    def lap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    return LaplaceSolution(name=f"eigen_trace_square_n{n}", u=u, laplacian=lap)
