"""PH-SYM-004 — periodic translation equivariance (periodic domains only)."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_sym_004


def _periodic_spec(declared: list[str]) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [32, 32],
            "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "symmetries": {"declared": declared},
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )


def _non_periodic_spec(declared: list[str]) -> DomainSpec:
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


def test_ph_sym_004_periodic_translation_passes_on_sin():
    n = 32
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    u = np.sin(X) + np.sin(Y)
    field = GridField(u, h=(2 * np.pi / n, 2 * np.pi / n), periodic=True)
    spec = _periodic_spec(["translation_x", "translation_y"])
    result = ph_sym_004.check(field, spec)
    assert result.status == "PASS"
    assert result.raw_value is not None


def test_ph_sym_004_skipped_on_nonperiodic():
    field = GridField(np.zeros((32, 32)), h=(1 / 31, 1 / 31), periodic=False)
    spec = _non_periodic_spec(["translation_x"])
    result = ph_sym_004.check(field, spec)
    assert result.status == "SKIPPED"
    assert "periodic" in (result.reason or "").lower()


def test_ph_sym_004_skipped_if_not_declared():
    n = 32
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    X, _Y = np.meshgrid(x, x, indexing="ij")  # noqa: N806
    field = GridField(np.sin(X), h=(2 * np.pi / n, 2 * np.pi / n), periodic=True)
    spec = _periodic_spec(["C4"])
    result = ph_sym_004.check(field, spec)
    assert result.status == "SKIPPED"
