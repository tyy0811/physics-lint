"""PH-POS-001 — Positivity violation."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_pos_001


def _heat_per_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [32, 32, 4],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.1]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "diffusivity": 0.01,
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )


def _poisson_hd_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [32, 32],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def test_ph_pos_001_nonneg_passes():
    spec = _poisson_hd_spec()
    u = np.ones((32, 32)) * 0.5
    field = GridField(u, h=1.0 / 31, periodic=False)
    result = ph_pos_001.check(field, spec)
    assert result.status == "PASS"


def test_ph_pos_001_negative_values_fail():
    spec = _poisson_hd_spec()
    u = np.ones((32, 32)) * 0.5
    u[10:20, 10:20] = -0.1
    field = GridField(u, h=1.0 / 31, periodic=False)
    result = ph_pos_001.check(field, spec)
    assert result.status == "FAIL"
    assert result.raw_value == -0.1


def test_ph_pos_001_skipped_on_non_sign_preserving_bc():
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [32, 32],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    u = np.ones((32, 32))
    field = GridField(u, h=1.0 / 31, periodic=False)
    result = ph_pos_001.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason and "sign" in result.reason.lower()


def test_ph_pos_001_metadata():
    assert ph_pos_001.__rule_id__ == "PH-POS-001"
    assert ph_pos_001.__default_severity__ == "error"
    assert "adapter" in ph_pos_001.__input_modes__
    assert "dump" in ph_pos_001.__input_modes__


def test_ph_pos_001_periodic_heat_passes():
    spec = _heat_per_spec()
    u = np.ones((32, 32)) * 0.3
    field = GridField(u, h=1.0 / 32, periodic=True)
    result = ph_pos_001.check(field, spec)
    assert result.status == "PASS"


def test_ph_pos_001_custom_floor():
    spec = _poisson_hd_spec()
    u = np.ones((16, 16)) * 0.3
    field = GridField(u, h=1.0 / 15, periodic=False)
    result = ph_pos_001.check(field, spec, floor=0.5)
    assert result.status == "FAIL"  # 0.3 < 0.5 everywhere
    assert result.violation_ratio == 1.0  # all cells violate
