"""PH-SYM-001 — C4 rotation equivariance."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_sym_001


def _square_laplace_spec(declared: list[str]) -> DomainSpec:
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


def _c4_symmetric_field() -> GridField:
    # x^2 + y^2 is C4-symmetric about the center of [0,1]^2
    n = 32
    x = np.linspace(0.0, 1.0, n) - 0.5
    y = np.linspace(0.0, 1.0, n) - 0.5
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    return GridField(X**2 + Y**2, h=(1 / (n - 1), 1 / (n - 1)), periodic=False)


def _c4_violating_field() -> GridField:
    n = 32
    x = np.linspace(0.0, 1.0, n) - 0.5
    y = np.linspace(0.0, 1.0, n) - 0.5
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    return GridField(X + 0.1 * Y, h=(1 / (n - 1), 1 / (n - 1)), periodic=False)


def test_ph_sym_001_pass_on_symmetric():
    spec = _square_laplace_spec(["C4"])
    field = _c4_symmetric_field()
    result = ph_sym_001.check(field, spec)
    assert result.status == "PASS"
    assert result.raw_value is not None
    assert result.raw_value < 1e-12


def test_ph_sym_001_warn_on_asymmetric():
    spec = _square_laplace_spec(["C4"])
    field = _c4_violating_field()
    result = ph_sym_001.check(field, spec)
    assert result.status in {"WARN", "FAIL"}
    assert result.raw_value is not None
    assert result.raw_value > 0.1


def test_ph_sym_001_skipped_if_not_declared():
    spec = _square_laplace_spec(["reflection_x"])  # C4 not declared
    field = _c4_symmetric_field()
    result = ph_sym_001.check(field, spec)
    assert result.status == "SKIPPED"


def test_ph_sym_001_d4_implies_c4():
    spec = _square_laplace_spec(["D4"])
    field = _c4_symmetric_field()
    result = ph_sym_001.check(field, spec)
    assert result.status == "PASS"
