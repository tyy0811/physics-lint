"""PH-SYM-003 — SO(2) Lie derivative equivariance; adapter-only."""

import numpy as np
import torch

from physics_lint import CallableField, DomainSpec, GridField
from physics_lint.rules import ph_sym_003


def _so2_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [16, 16],
            "domain": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "symmetries": {"declared": ["SO2"]},
            "field": {"type": "callable", "backend": "fd", "adapter_path": "x"},
        }
    )


def test_ph_sym_003_radial_pass():
    # A purely radial field is SO(2) invariant — Lie derivative is zero.
    def radial(pts: torch.Tensor) -> torch.Tensor:
        r2 = pts[..., 0] ** 2 + pts[..., 1] ** 2
        return r2.unsqueeze(-1)

    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(-0.5, 0.5, 16),
            torch.linspace(-0.5, 0.5, 16),
            indexing="ij",
        ),
        dim=-1,
    )
    field = CallableField(radial, sampling_grid=grid, h=(1 / 15, 1 / 15))
    result = ph_sym_003.check(field, _so2_spec())
    assert result.status == "PASS"
    assert result.raw_value is not None
    assert result.raw_value < 1e-5


def test_ph_sym_003_non_radial_warn_or_fail():
    def linear(pts: torch.Tensor) -> torch.Tensor:
        return (2.0 * pts[..., 0] + pts[..., 1]).unsqueeze(-1)

    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(-0.5, 0.5, 16),
            torch.linspace(-0.5, 0.5, 16),
            indexing="ij",
        ),
        dim=-1,
    )
    field = CallableField(linear, sampling_grid=grid, h=(1 / 15, 1 / 15))
    result = ph_sym_003.check(field, _so2_spec())
    assert result.status in {"WARN", "FAIL"}


def test_ph_sym_003_dump_mode_skipped():
    spec = _so2_spec()
    u = np.zeros((16, 16))
    field = GridField(u, h=(1 / 15, 1 / 15), periodic=False)
    result = ph_sym_003.check(field, spec)
    assert result.status == "SKIPPED"
    assert "callable" in (result.reason or "").lower()


def test_ph_sym_003_skipped_if_so2_not_declared():
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [16, 16],
            "domain": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "symmetries": {"declared": ["C4"]},
            "field": {"type": "callable", "backend": "fd", "adapter_path": "x"},
        }
    )

    def radial(pts):
        return (pts**2).sum(-1, keepdim=True)

    grid = torch.zeros(4, 4, 2)
    field = CallableField(radial, sampling_grid=grid, h=(0.25, 0.25))
    result = ph_sym_003.check(field, spec)
    assert result.status == "SKIPPED"
    assert "SO2" in (result.reason or "")


def test_ph_sym_003_non_centered_grid_skips():
    """A grid on [0, 1] x [0, 1] is not origin-centered; rule must SKIP."""

    def radial(pts):
        return (pts[..., 0] ** 2 + pts[..., 1] ** 2).unsqueeze(-1)

    n = 16
    axis = torch.linspace(0.0, 1.0, n)  # NOT centered at origin
    grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1)
    field = CallableField(radial, sampling_grid=grid, h=(1 / (n - 1), 1 / (n - 1)))
    result = ph_sym_003.check(field, _so2_spec())
    assert result.status == "SKIPPED"
    assert "origin-centered" in (result.reason or "").lower()


def test_ph_sym_003_loader_backed_grid_passes_on_radial(tmp_path):
    """Regression for loader bounds fix: adapter-mode path with a centered
    domain spec should produce a centered grid and pass on a radial model.

    This would have failed before the loader fix because _build_sampling_grid
    ignored the lower bound of domain.x / domain.y.
    """
    from physics_lint.loader import _build_sampling_grid

    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [16, 16],
            "domain": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "symmetries": {"declared": ["SO2"]},
            "field": {"type": "callable", "backend": "fd", "adapter_path": "x"},
        }
    )
    grid = _build_sampling_grid(spec)
    # Sanity: the loader produced an origin-centered grid.
    assert abs(float(grid[..., 0].mean())) < 1e-6
    assert abs(float(grid[..., 1].mean())) < 1e-6

    def radial(pts: torch.Tensor) -> torch.Tensor:
        return (pts[..., 0] ** 2 + pts[..., 1] ** 2).unsqueeze(-1)

    field = CallableField(radial, sampling_grid=grid, h=(1 / 15, 1 / 15))
    result = ph_sym_003.check(field, spec)
    assert result.status == "PASS"
    assert result.raw_value is not None
    assert result.raw_value < 1e-5
