"""PH-VAR-002 — hyperbolic norm-equivalence conjectural caveat, info-level."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_var_002


def _wave_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "wave",
            "grid_shape": [16, 16, 8],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "wave_speed": 1.0,
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def _heat_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [16, 16, 8],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "diffusivity": 0.01,
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def test_ph_var_002_fires_on_wave():
    # Info-level caveat: status is PASS (so overall_status stays PASS) but
    # severity is info (so the reporter can still surface it) and `reason`
    # carries the message. A WARN status here would poison every clean
    # wave report regardless of whether any real rule actually warned.
    spec = _wave_spec()
    field = GridField(np.zeros((16, 16, 8)), h=(1 / 15, 1 / 15, 1 / 7), periodic=False)
    result = ph_var_002.check(field, spec)
    assert result.rule_id == "PH-VAR-002"
    assert result.severity == "info"
    assert result.status == "PASS"
    assert "hyperbolic" in (result.reason or "").lower()


def test_ph_var_002_does_not_degrade_overall_status_on_wave():
    # Regression: the rule must not flip a clean wave report from overall
    # PASS to WARN just by being present.
    from physics_lint.report import PhysicsLintReport

    spec = _wave_spec()
    field = GridField(np.zeros((16, 16, 8)), h=(1 / 15, 1 / 15, 1 / 7), periodic=False)
    var_result = ph_var_002.check(field, spec)
    report = PhysicsLintReport(pde="wave", grid_shape=(16, 16, 8), rules=[var_result])
    assert report.overall_status == "PASS"


def test_ph_var_002_skipped_on_non_wave():
    spec = _heat_spec()
    field = GridField(np.zeros((16, 16, 8)), h=(1 / 15, 1 / 15, 1 / 7), periodic=False)
    result = ph_var_002.check(field, spec)
    assert result.status == "SKIPPED"
