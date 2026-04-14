"""CallableField — wrap a PyTorch callable as a Field.

Week 1 scope: values() materializes on a user-provided sampling grid;
laplacian() via AD; grad() via AD. at() and integrate() delegate to
the materialized GridField.
"""

import numpy as np
import pytest
import torch

from physics_lint.field import CallableField, GridField


def _quadratic_model(x: torch.Tensor) -> torch.Tensor:
    # u(x, y) = x^2 + y^2 => Laplacian = 4
    return (x[..., 0] ** 2 + x[..., 1] ** 2).unsqueeze(-1)


def test_callable_field_values_on_grid():
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 16),
            torch.linspace(0.0, 1.0, 16),
            indexing="ij",
        ),
        dim=-1,
    )
    f = CallableField(_quadratic_model, sampling_grid=grid, h=(1.0 / 15, 1.0 / 15))
    vals = f.values()
    assert vals.shape == (16, 16)
    expected = grid[..., 0].numpy() ** 2 + grid[..., 1].numpy() ** 2
    assert np.allclose(vals, expected, atol=1e-6)


def test_callable_field_laplacian_quadratic():
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 8),
            torch.linspace(0.0, 1.0, 8),
            indexing="ij",
        ),
        dim=-1,
    )
    f = CallableField(_quadratic_model, sampling_grid=grid, h=(1.0 / 7, 1.0 / 7))
    lap = f.laplacian().values()
    # Exact: Laplacian of x^2 + y^2 is 4 everywhere. Analytically exact
    # for the autograd path — any deviation is machine roundoff.
    assert np.allclose(lap, 4.0, atol=1e-12)


def test_callable_field_laplacian_returns_gridfield():
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 4),
            torch.linspace(0.0, 1.0, 4),
            indexing="ij",
        ),
        dim=-1,
    )
    f = CallableField(_quadratic_model, sampling_grid=grid, h=(1.0 / 3, 1.0 / 3))
    lap = f.laplacian()
    assert isinstance(lap, GridField)


def test_callable_field_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        CallableField("not a callable", sampling_grid=torch.zeros(4, 4, 2), h=(0.25, 0.25))


def test_callable_field_integrate_delegates():
    # Integral of u = 1 over [0,1]^2 should be 1.
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 9),
            torch.linspace(0.0, 1.0, 9),
            indexing="ij",
        ),
        dim=-1,
    )
    f = CallableField(
        lambda p: torch.ones_like(p[..., :1]),
        sampling_grid=grid,
        h=(1.0 / 8, 1.0 / 8),
    )
    assert abs(f.integrate() - 1.0) < 1e-12


def test_callable_field_values_on_boundary_delegates():
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 4),
            torch.linspace(0.0, 1.0, 4),
            indexing="ij",
        ),
        dim=-1,
    )
    f = CallableField(_quadratic_model, sampling_grid=grid, h=(1.0 / 3, 1.0 / 3))
    boundary = f.values_on_boundary()
    # The boundary contains the four edges of a 4x4 grid (12 points total).
    assert boundary.shape == (12,)
    # Value at (0,0) = 0; at (1,1) = 2; at (1,0) = 1; at (0,1) = 1
    assert 0.0 in boundary
    assert abs(boundary.max() - 2.0) < 1e-6


def test_callable_field_scalar_h_expands_to_tuple():
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 4),
            torch.linspace(0.0, 1.0, 4),
            indexing="ij",
        ),
        dim=-1,
    )
    f = CallableField(_quadratic_model, sampling_grid=grid, h=1.0 / 3)
    assert f.h == (1.0 / 3, 1.0 / 3)


def test_callable_field_numpy_scalar_h_accepted():
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 4),
            torch.linspace(0.0, 1.0, 4),
            indexing="ij",
        ),
        dim=-1,
    )
    f = CallableField(_quadratic_model, sampling_grid=grid, h=np.float32(1.0 / 3))
    assert f.h == (pytest.approx(1.0 / 3), pytest.approx(1.0 / 3))


def test_callable_field_wrong_length_h_raises_valueerror():
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 4),
            torch.linspace(0.0, 1.0, 4),
            indexing="ij",
        ),
        dim=-1,
    )
    with pytest.raises(ValueError, match="must match"):
        CallableField(_quadratic_model, sampling_grid=grid, h=(0.25, 0.25, 0.25))


def test_callable_field_non_iterable_h_raises_typeerror_via_fallback():
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 4),
            torch.linspace(0.0, 1.0, 4),
            indexing="ij",
        ),
        dim=-1,
    )
    # A non-Real, non-iterable, non-str/bytes object should fall through to
    # the float() branch and raise TypeError wrapped with a helpful message.
    with pytest.raises(TypeError, match="scalar or"):
        CallableField(_quadratic_model, sampling_grid=grid, h=object())


def test_callable_field_bytes_h_raises_typeerror():
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 4),
            torch.linspace(0.0, 1.0, 4),
            indexing="ij",
        ),
        dim=-1,
    )
    with pytest.raises(TypeError, match="scalar or"):
        CallableField(_quadratic_model, sampling_grid=grid, h=b"not-valid")


def test_callable_field_model_returning_unsqueezed_tensor():
    # A model that already returns shape (*spatial,) without a trailing
    # singleton should still materialize correctly (no squeeze step taken).
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 4),
            torch.linspace(0.0, 1.0, 4),
            indexing="ij",
        ),
        dim=-1,
    )
    f = CallableField(
        lambda p: p[..., 0] + p[..., 1],  # returns (4, 4), not (4, 4, 1)
        sampling_grid=grid,
        h=(1.0 / 3, 1.0 / 3),
    )
    vals = f.values()
    assert vals.shape == (4, 4)
    expected = grid[..., 0].numpy() + grid[..., 1].numpy()
    assert np.allclose(vals, expected, atol=1e-6)


def test_callable_field_caches_materialized_grid():
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 4),
            torch.linspace(0.0, 1.0, 4),
            indexing="ij",
        ),
        dim=-1,
    )
    call_count = {"n": 0}

    def counting_model(x: torch.Tensor) -> torch.Tensor:
        call_count["n"] += 1
        return (x[..., 0] + x[..., 1]).unsqueeze(-1)

    f = CallableField(counting_model, sampling_grid=grid, h=(1.0 / 3, 1.0 / 3))
    _ = f.values()
    _ = f.values()
    _ = f.integrate()
    _ = f.values_on_boundary()
    # Materialization cache hits: 1 model call for all four subsequent reads.
    # (Laplacian path runs the model again under autograd — that's a separate
    # code path and not counted here.)
    assert call_count["n"] == 1
