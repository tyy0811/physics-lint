"""MMS harness helpers - Task 4 Layer 2 dependency (closed-form H1 error)."""

from __future__ import annotations

import math

import numpy as np

from external_validation._harness.mms import mms_sin_sin_h1_error
from physics_lint.analytical.poisson import sin_sin_mms_square

N = 64


def test_mms_sin_sin_matches_closed_form():
    sol = sin_sin_mms_square()
    xs = np.linspace(0.0, 1.0, N)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    u = sol.u(mesh_x, mesh_y)
    expected = np.sin(math.pi * mesh_x) * np.sin(math.pi * mesh_y)
    assert np.allclose(u, expected, atol=1e-12)


def test_analytical_h1_error_of_self_is_zero():
    xs = np.linspace(0.0, 1.0, N)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    err = mms_sin_sin_h1_error(mesh_x, mesh_y, perturbation=lambda x, y: np.zeros_like(x))
    assert err < 1e-12, f"H1 error of self is {err:.3e}, expected ~0"


def test_analytical_h1_error_grows_with_perturbation_frequency():
    xs = np.linspace(0.0, 1.0, N)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")

    def low_freq(x, y):
        return 0.01 * np.sin(math.pi * x) * np.sin(math.pi * y)

    def high_freq(x, y):
        return 0.01 * np.sin(4 * math.pi * x) * np.sin(4 * math.pi * y)

    err_low = mms_sin_sin_h1_error(mesh_x, mesh_y, perturbation=low_freq)
    err_high = mms_sin_sin_h1_error(mesh_x, mesh_y, perturbation=high_freq)
    assert err_high > 3 * err_low, (
        f"high-freq H1 err={err_high:.3e} should be >> low-freq {err_low:.3e}"
    )
