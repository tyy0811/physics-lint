"""Analytical solutions for Poisson's equation (-Delta u = f)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class PoissonSolution:
    name: str
    u: Callable[[np.ndarray, np.ndarray], np.ndarray]
    laplacian: Callable[[np.ndarray, np.ndarray], np.ndarray]
    source: Callable[[np.ndarray, np.ndarray], np.ndarray]  # f = -Delta u


def sin_sin_mms_square() -> PoissonSolution:
    """u(x,y) = sin(pi x) sin(pi y) on [0,1]^2 with hD. -Delta u = 2 pi^2 u."""

    def u(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def lap(x, y):
        return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    def source(x, y):
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    return PoissonSolution(name="sin_sin_mms_square", u=u, laplacian=lap, source=source)


def periodic_sin_sin() -> PoissonSolution:
    """u = sin(x) sin(y) on [0, 2 pi]^2 periodic. -Delta u = 2 sin(x) sin(y)."""

    def u(x, y):
        return np.sin(x) * np.sin(y)

    def lap(x, y):
        return -2 * np.sin(x) * np.sin(y)

    def source(x, y):
        return 2 * np.sin(x) * np.sin(y)

    return PoissonSolution(name="periodic_sin_sin", u=u, laplacian=lap, source=source)
