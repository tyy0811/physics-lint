"""PH-SYM-001 external-validation anchor - Helwig 2023 Table 3 calibration.

Rule-level anchor: feeds a known C4-symmetric field and a known C4-breaking
field to ph_sym_001.check and verifies the rule classifies them correctly.
Operator-level equivariance (fft_laplace_inverse vs non_equivariant_cnn) is
validated separately in _harness/tests/test_symmetry.py.
"""

from __future__ import annotations

import math

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_sym_001

N = 64
H = 1.0 / (N - 1)


def _c4_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [N, N],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "symmetries": {"declared": ["C4"]},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def _cos_cos_field() -> np.ndarray:
    """u(x, y) = cos(2*pi*x) * cos(2*pi*y) - non-trivially C4-symmetric."""
    xs = np.linspace(0.0, 1.0, N)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    return np.cos(2 * math.pi * mesh_x) * np.cos(2 * math.pi * mesh_y)


def _sin_sin_field() -> np.ndarray:
    """u(x, y) = sin(2*pi*x) * sin(2*pi*y) - C4-breaking (rot90 maps to -u)."""
    xs = np.linspace(0.0, 1.0, N)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    return np.sin(2 * math.pi * mesh_x) * np.sin(2 * math.pi * mesh_y)


def test_fixture_cos_cos_c4_rotation_table_is_identity():
    """Sanity: rot90(cos*cos, k) = cos*cos for k in {1, 2, 3} within float noise."""
    u = _cos_cos_field()
    for k in (1, 2, 3):
        rotated = np.rot90(u, k=k)
        assert np.allclose(rotated, u, atol=1e-12), (
            f"rot90(cos*cos, k={k}) deviates from identity by {np.max(np.abs(rotated - u)):.3e}"
        )


def test_fixture_sin_sin_rot90_k1_maps_to_negative():
    """Sanity: rot90(sin*sin, k=1) = -sin*sin within float noise."""
    u = _sin_sin_field()
    rotated = np.rot90(u, k=1)
    assert np.allclose(rotated, -u, atol=1e-12), (
        f"rot90(sin*sin, k=1) deviates from -u by {np.max(np.abs(rotated + u)):.3e}"
    )


def test_ph_sym_001_pass_on_c4_symmetric_cos_cos():
    field = GridField(_cos_cos_field(), h=(H, H), periodic=False)
    result = ph_sym_001.check(field, _c4_spec())
    assert result.status == "PASS"
    assert result.raw_value is not None
    assert result.raw_value < 1e-12, (
        f"PASS fixture's max equivariance error {result.raw_value:.3e} is unexpectedly large"
    )


def test_ph_sym_001_warn_or_fail_on_c4_breaking_sin_sin():
    field = GridField(_sin_sin_field(), h=(H, H), periodic=False)
    result = ph_sym_001.check(field, _c4_spec())
    assert result.status in {"WARN", "FAIL"}, (
        f"FAIL fixture produced status={result.status!r}; "
        "expected WARN or FAIL because rot90(sin*sin, k=1) = -sin*sin."
    )
    assert result.raw_value is not None and result.raw_value > 1.0, (
        f"FAIL fixture's max equivariance error {result.raw_value!r} "
        "below 1.0; the sin*sin rot90 error should be exactly ~2.0."
    )
