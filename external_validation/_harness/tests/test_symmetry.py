"""Symmetry-harness unit tests - preconditions and primitives."""

from __future__ import annotations

import math

import pytest
import torch

from external_validation._harness.symmetry import (
    fft_laplace_inverse,
    non_equivariant_cnn,
    reflect_test,
    rotate_test,
)


def test_fft_laplace_inverse_is_exactly_c4_equivariant_on_smooth_input():
    """Laplace inverse on a smooth bandlimited field is C4-equivariant to float noise."""
    n = 64
    xs = torch.linspace(0, 1, n)
    mesh_x, mesh_y = torch.meshgrid(xs, xs, indexing="ij")
    f = torch.sin(2 * math.pi * mesh_x) * torch.sin(2 * math.pi * mesh_y)
    err = rotate_test(fft_laplace_inverse, f, k=1)
    assert err < 1e-5, f"rotate_test k=1 err={err:.2e}"


def test_fft_laplace_inverse_is_exactly_reflection_equivariant():
    n = 64
    xs = torch.linspace(0, 1, n)
    mesh_x, mesh_y = torch.meshgrid(xs, xs, indexing="ij")
    f = torch.sin(2 * math.pi * mesh_x) * torch.sin(2 * math.pi * mesh_y)
    err = reflect_test(fft_laplace_inverse, f, axis=-1)
    assert err < 1e-5, f"reflect_test axis=-1 err={err:.2e}"


def test_non_equivariant_cnn_is_not_equivariant():
    torch.manual_seed(0)
    cnn = non_equivariant_cnn()
    n = 64
    xs = torch.linspace(0, 1, n)
    mesh_x, mesh_y = torch.meshgrid(xs, xs, indexing="ij")
    f = torch.sin(2 * math.pi * mesh_x) * torch.sin(2 * math.pi * mesh_y)
    err = rotate_test(cnn, f, k=1)
    assert err > 0.1, f"non-equivariant CNN err={err:.2e} - should be > 0.1"


def test_rotate_test_precondition_square_grid():
    """rotate_test requires a square 2D grid; non-square should error."""
    f = torch.zeros((64, 128))
    with pytest.raises(ValueError, match="square"):
        rotate_test(fft_laplace_inverse, f, k=1)
