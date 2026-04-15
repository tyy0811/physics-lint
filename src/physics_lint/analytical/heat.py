"""Analytical solutions for the heat equation u_t - kappa * Laplacian(u) = 0."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class HeatSolution:
    name: str
    kappa: float
    u: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    time_derivative: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    laplacian: Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def eigenfunction_decay_square(kappa: float) -> HeatSolution:
    """u = sin(pi x) sin(pi y) exp(-2 kappa pi^2 t) on [0,1]^2 with hD."""
    decay_rate = 2 * np.pi**2 * kappa

    def u(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-decay_rate * t)

    def u_t(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return -decay_rate * u(x, y, t)

    def lap(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return -2 * np.pi**2 * u(x, y, t)

    return HeatSolution(
        name="eigenfunction_decay_square",
        kappa=kappa,
        u=u,
        time_derivative=u_t,
        laplacian=lap,
    )


def periodic_cos_cos(kappa: float) -> HeatSolution:
    """u = cos(x) cos(y) exp(-2 kappa t) on [0, 2 pi]^2 periodic."""
    decay_rate = 2 * kappa

    def u(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return np.cos(x) * np.cos(y) * np.exp(-decay_rate * t)

    def u_t(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return -decay_rate * u(x, y, t)

    def lap(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return -2 * np.cos(x) * np.cos(y) * np.exp(-decay_rate * t)

    return HeatSolution(
        name="periodic_cos_cos",
        kappa=kappa,
        u=u,
        time_derivative=u_t,
        laplacian=lap,
    )
