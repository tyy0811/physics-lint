"""PH-SYM-002 — reflection equivariance tests."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_sym_002


def _spec(declared: list[str]) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [32, 32],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "symmetries": {"declared": declared},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def test_ph_sym_002_pass_on_x_symmetric():
    n = 32
    x = np.linspace(0.0, 1.0, n) - 0.5
    X = np.tile(x[:, None], (1, n))  # noqa: N806
    Y = np.tile(x[None, :], (n, 1))  # noqa: N806
    u = X**2 + Y  # symmetric in X (axis 0 reflection), asymmetric in Y
    field = GridField(u, h=(1 / (n - 1), 1 / (n - 1)), periodic=False)
    spec = _spec(["reflection_x"])
    result = ph_sym_002.check(field, spec)
    assert result.status == "PASS"


def test_ph_sym_002_skipped_if_not_declared():
    field = GridField(np.zeros((32, 32)), h=(1 / 31, 1 / 31), periodic=False)
    spec = _spec(["C4"])  # no reflection declared
    result = ph_sym_002.check(field, spec)
    assert result.status == "SKIPPED"


def test_ph_sym_002_fail_on_asymmetric():
    n = 32
    x = np.linspace(0.0, 1.0, n) - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")  # noqa: N806
    u = X + 0.3 * Y  # asymmetric under x-reflection (sign of X flips but Y doesn't)
    field = GridField(u, h=(1 / (n - 1), 1 / (n - 1)), periodic=False)
    spec = _spec(["reflection_x"])
    result = ph_sym_002.check(field, spec)
    assert result.status in {"WARN", "FAIL"}
    assert result.raw_value is not None
    assert result.raw_value > 0.01
