"""PH-RES-001 — Residual exceeds variationally-correct norm threshold.

Laplace/Poisson path: compute the strong-form residual of the Field against
the configured PDE, take its H^-1 norm via the spectral formula (for periodic
inputs) or a Riesz-lift surrogate (for non-periodic, Week-1 falls back to L2),
divide by the calibrated floor, emit tri-state.
"""

import numpy as np
import pytest

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_res_001


def _laplace_periodic_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [64, 64],
            "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )


def test_ph_res_001_exact_harmonic_is_pass():
    spec = _laplace_periodic_spec()
    # Harmonic: Laplacian is identically zero, so residual norm is ~0
    n = 64
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    xg, _yg = np.meshgrid(x, y, indexing="ij")
    u = np.zeros_like(xg)  # the trivial harmonic
    field = GridField(u, h=(2 * np.pi / n, 2 * np.pi / n), periodic=True)

    result = ph_res_001.check(field, spec)
    assert result.rule_id == "PH-RES-001"
    assert result.status == "PASS"
    assert result.raw_value is not None
    assert result.raw_value < 1e-12


def test_ph_res_001_nonzero_residual_is_warn_or_fail():
    spec = _laplace_periodic_spec()
    n = 64
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    # u = cos(x) cos(y) — Laplacian is -2 cos(x) cos(y), so this is NOT
    # a Laplace solution (residual is nonzero and large).
    u = np.cos(xg) * np.cos(yg)
    field = GridField(u, h=(2 * np.pi / n, 2 * np.pi / n), periodic=True)

    result = ph_res_001.check(field, spec)
    assert result.status in {"WARN", "FAIL"}
    assert result.violation_ratio is not None
    assert result.violation_ratio > 1.0


def test_ph_res_001_metadata():
    assert ph_res_001.__rule_id__ == "PH-RES-001"
    assert ph_res_001.__default_severity__ == "error"
    assert "adapter" in ph_res_001.__input_modes__
    assert "dump" in ph_res_001.__input_modes__


def test_ph_res_001_nonperiodic_fd_l2_fallback():
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [16, 16],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    u = np.zeros((16, 16))
    field = GridField(u, h=1.0 / 15, periodic=False, backend="fd")
    result = ph_res_001.check(field, spec)
    assert result.status == "PASS"
    assert result.recommended_norm == "L2"


def test_ph_res_001_poisson_is_skipped_until_week_2():
    # Poisson source-term plumbing lands in Week 2. For Week 1 the rule
    # should emit SKIPPED with a clear reason so a linter run over a
    # well-formed Poisson config doesn't crash mid-pipeline.
    spec = DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [16, 16],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    field = GridField(np.zeros((16, 16)), h=1.0 / 15, periodic=False, backend="fd")
    result = ph_res_001.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason is not None
    assert "Week 2" in result.reason


def test_ph_res_001_heat_is_skipped_until_week_2():
    spec = DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [16, 16, 4],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.1]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
            "diffusivity": 0.01,
        }
    )
    # Heat rules operate against the spatial slice at Week 1; the rule just
    # has to *not* raise when dispatched on a heat spec.
    field = GridField(np.zeros((16, 16)), h=1.0 / 15, periodic=False, backend="fd")
    result = ph_res_001.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason is not None
    assert "Week 2" in result.reason


def test_ph_res_001_rejects_non_gridfield():
    import torch

    from physics_lint import CallableField

    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 4),
            torch.linspace(0.0, 1.0, 4),
            indexing="ij",
        ),
        dim=-1,
    )
    field = CallableField(
        lambda x: x[..., 0].unsqueeze(-1),
        sampling_grid=grid,
        h=(1.0 / 3, 1.0 / 3),
    )
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [4, 4],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "field": {"type": "callable", "backend": "fd", "adapter_path": "x.py"},
        }
    )
    with pytest.raises(TypeError, match="requires a GridField"):
        ph_res_001.check(field, spec)
