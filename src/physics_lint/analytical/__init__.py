"""Analytical solutions for the self-test battery.

Each solution module exposes factory functions returning an AnalyticalSolution
dataclass with callable `u(X, Y)`, `laplacian(X, Y)`, and optional `source(X, Y)`
(for Poisson). Heat/wave variants land in Week 2.
"""
