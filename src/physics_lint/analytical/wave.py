"""Analytical solutions for the wave equation u_tt - c^2 * Laplacian(u) = 0."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class WaveSolution:
    name: str
    c: float
    u: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    time_derivative: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    second_time_derivative: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    laplacian: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    grad_x: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    grad_y: Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def standing_wave_square(c: float) -> WaveSolution:
    """u = sin(pi x) sin(pi y) cos(pi c sqrt(2) t) on [0,1]^2 with hD."""
    omega = np.pi * c * np.sqrt(2.0)

    def u(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(omega * t)

    def u_t(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return -omega * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(omega * t)

    def u_tt(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return -(omega**2) * np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(omega * t)

    def lap(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(omega * t)

    def gx(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(omega * t)

    def gy(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return np.pi * np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(omega * t)

    return WaveSolution(
        name="standing_wave_square",
        c=c,
        u=u,
        time_derivative=u_t,
        second_time_derivative=u_tt,
        laplacian=lap,
        grad_x=gx,
        grad_y=gy,
    )


def periodic_traveling(c: float, length: float = 2 * np.pi) -> WaveSolution:
    """u = cos(2 pi (x - c t) / L) on [0, L]^2 periodic. A plane wave along x."""
    k = 2 * np.pi / length
    omega = k * c

    def u(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return np.cos(k * x - omega * t) * np.ones_like(y)

    def u_t(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return omega * np.sin(k * x - omega * t) * np.ones_like(y)

    def u_tt(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return -(omega**2) * np.cos(k * x - omega * t) * np.ones_like(y)

    def lap(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return -(k**2) * np.cos(k * x - omega * t) * np.ones_like(y)

    def gx(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return -k * np.sin(k * x - omega * t) * np.ones_like(y)

    def gy(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return np.zeros_like(y)

    return WaveSolution(
        name="periodic_traveling",
        c=c,
        u=u,
        time_derivative=u_t,
        second_time_derivative=u_tt,
        laplacian=lap,
        grad_x=gx,
        grad_y=gy,
    )
