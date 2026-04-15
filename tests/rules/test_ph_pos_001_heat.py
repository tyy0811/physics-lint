"""PH-POS-001 on heat — nonneg IC stays nonneg under hD."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.analytical import heat as heat_sols
from physics_lint.rules import ph_pos_001


def _heat_hd_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [32, 32, 8],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "diffusivity": 0.01,
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def test_ph_pos_001_heat_eigenfunction_is_pass():
    # sin(pi x) sin(pi y) exp(-2 kappa pi^2 t) is nonneg on [0, 1]^2 at every
    # t. PH-POS-001 takes .min() across the whole 3D (spatial + time) tensor,
    # so the rule exercises the time axis transparently; a floor slightly
    # below zero absorbs fp noise around the hD boundary.
    spec = _heat_hd_spec()
    sol = heat_sols.eigenfunction_decay_square(kappa=0.01)
    n, nt = 32, 8
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 1.0, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    pred = np.stack([sol.u(X, Y, ti) for ti in t], axis=-1)
    field = GridField(pred, h=(1 / (n - 1), 1 / (n - 1), 1 / (nt - 1)), periodic=False)
    result = ph_pos_001.check(field, spec, floor=-1e-12)
    assert result.status == "PASS"


def test_ph_pos_001_heat_negative_prediction_is_fail():
    spec = _heat_hd_spec()
    n, nt = 32, 8
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 1.0, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    # Sign-flipped IC: negative at t=0, decays toward zero.
    pred = np.stack(
        [-np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-0.1 * ti) for ti in t], axis=-1
    )
    field = GridField(pred, h=(1 / (n - 1), 1 / (n - 1), 1 / (nt - 1)), periodic=False)
    result = ph_pos_001.check(field, spec)
    assert result.status == "FAIL"
