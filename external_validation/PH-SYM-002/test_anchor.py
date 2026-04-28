"""PH-SYM-002 external-validation anchor - Helwig 2023 Table 1 calibration.

Rule-level anchor: feeds a known reflection-symmetric field and a known
flip-breaking field to ph_sym_002.check and verifies the rule classifies
them correctly. Both axis-0 (reflection_x) and axis-1 (reflection_y) are
exercised on the same fixtures.
"""

from __future__ import annotations

import math

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_sym_002

N = 64
H = 1.0 / (N - 1)


def _reflection_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [N, N],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "symmetries": {"declared": ["reflection_x", "reflection_y"]},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def _cos_cos_field() -> np.ndarray:
    """u(x, y) = cos(2*pi*x) * cos(2*pi*y) - flip-invariant on both axes."""
    xs = np.linspace(0.0, 1.0, N)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    return np.cos(2 * math.pi * mesh_x) * np.cos(2 * math.pi * mesh_y)


def _sin_sin_field() -> np.ndarray:
    """u(x, y) = sin(2*pi*x) * sin(2*pi*y) - flip maps to -u on either axis."""
    xs = np.linspace(0.0, 1.0, N)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    return np.sin(2 * math.pi * mesh_x) * np.sin(2 * math.pi * mesh_y)


def test_fixture_cos_cos_flip_table_is_identity():
    """Sanity: flip(cos*cos, axis) = cos*cos for axis in {0, 1} within float noise."""
    u = _cos_cos_field()
    for axis in (0, 1):
        flipped = np.flip(u, axis=axis)
        assert np.allclose(flipped, u, atol=1e-12), (
            f"flip(cos*cos, axis={axis}) deviates from identity by "
            f"{np.max(np.abs(flipped - u)):.3e}"
        )


def test_fixture_sin_sin_flip_maps_to_negative():
    """Sanity: flip(sin*sin, axis) = -sin*sin for axis in {0, 1} within float noise."""
    u = _sin_sin_field()
    for axis in (0, 1):
        flipped = np.flip(u, axis=axis)
        assert np.allclose(flipped, -u, atol=1e-12), (
            f"flip(sin*sin, axis={axis}) deviates from -u by {np.max(np.abs(flipped + u)):.3e}"
        )


def test_ph_sym_002_pass_on_reflection_symmetric_cos_cos():
    field = GridField(_cos_cos_field(), h=(H, H), periodic=False)
    result = ph_sym_002.check(field, _reflection_spec())
    assert result.status == "PASS"
    assert result.raw_value is not None
    assert result.raw_value < 1e-12, (
        f"PASS fixture's max equivariance error {result.raw_value:.3e} is unexpectedly large"
    )


def test_ph_sym_002_warn_or_fail_on_flip_breaking_sin_sin():
    field = GridField(_sin_sin_field(), h=(H, H), periodic=False)
    result = ph_sym_002.check(field, _reflection_spec())
    assert result.status in {"WARN", "FAIL"}, (
        f"FAIL fixture produced status={result.status!r}; "
        "expected WARN or FAIL because flip(sin*sin, axis) = -sin*sin."
    )
    assert result.raw_value is not None and result.raw_value > 1.0, (
        f"FAIL fixture's max equivariance error {result.raw_value!r} "
        "below 1.0; the sin*sin flip error should be exactly ~2.0 on each axis."
    )
