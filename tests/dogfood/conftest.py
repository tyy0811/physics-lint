"""Shared fixtures for dogfood tests."""

from __future__ import annotations

import numpy as np
import pytest

from physics_lint.spec import (
    BCSpec,
    DomainSpec,
    FieldSourceSpec,
    GridDomain,
    SymmetrySpec,
)


@pytest.fixture
def a1_spec() -> DomainSpec:
    """DomainSpec matching the Week 2½ A1 configuration.

    64x64 grid on unit square, Laplace PDE, non-homogeneous Dirichlet BCs
    (so preserves_sign=False — PH-POS-001 will SKIP, PH-POS-002 will run).
    """
    return DomainSpec(
        pde="laplace",
        grid_shape=(64, 64),
        domain=GridDomain(x=(0.0, 1.0), y=(0.0, 1.0)),
        periodic=False,
        boundary_condition=BCSpec(kind="dirichlet"),
        symmetries=SymmetrySpec(declared=[]),
        field=FieldSourceSpec(type="grid", backend="fd", dump_path="unused"),
    )


@pytest.fixture
def linear_field() -> np.ndarray:
    """A known-harmonic test field: u(x, y) = x + y.

    On [0, 1]^2 with 64x64 endpoint-inclusive grid (h = 1/63), u is linear
    so Δu = 0 exactly. However, in **float32** the FD stencil produces a
    residual at the ~1e-3 level from roundoff propagated through the 1/h²
    stencil divisor (ε_float32 * h⁻² ≈ 1e-7 * 4e3 ≈ 4e-4 pointwise; l2_grid
    aggregation gives ~1.3e-3). Tolerances that assume analytical
    near-zero residual must be relaxed to >1e-3 to accommodate this.

    Empirically verified at plan-writing time: l2_grid(-lap) = 1.33e-3
    for x+y in float32 at 64x64. See plan self-review for the derivation.
    """
    n = 64
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return (xx + yy).astype(np.float32)
