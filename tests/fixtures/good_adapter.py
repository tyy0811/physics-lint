"""Test fixture: a valid adapter with load_model() and domain_spec()."""

import torch

from physics_lint import DomainSpec


def load_model():
    # Trivial "model": returns zero everywhere; quadratic field for Laplacian test
    def _model(x: torch.Tensor) -> torch.Tensor:
        return (x[..., 0] ** 2 - x[..., 1] ** 2).unsqueeze(-1)

    return _model


def domain_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [32, 32],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "field": {"type": "callable", "adapter_path": __file__},
        }
    )
