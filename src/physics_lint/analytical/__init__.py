"""Analytical solutions for the self-test battery.

Each solution module exposes factory functions returning a dataclass with
callable `u`, `laplacian`, and (variant-specific) time-derivative / source /
gradient attributes. Heat and wave variants carry a time argument; Laplace
and Poisson are time-independent.
"""

from physics_lint.analytical import heat, laplace, poisson, wave

__all__ = ["heat", "laplace", "poisson", "wave"]
