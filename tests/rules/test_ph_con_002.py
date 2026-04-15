"""PH-CON-002 — wave energy conservation tests."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.analytical import wave as wave_sols
from physics_lint.rules import ph_con_002


def _wave_hd_spec(n: int, nt: int) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "wave",
            "grid_shape": [n, n, nt],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.5]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "wave_speed": 1.0,
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def test_ph_con_002_standing_wave_conserves_energy():
    n, nt = 64, 32
    spec = _wave_hd_spec(n, nt)
    sol = wave_sols.standing_wave_square(c=1.0)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 0.5, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    pred = np.stack([sol.u(X, Y, ti) for ti in t], axis=-1)
    field = GridField(pred, h=(1 / (n - 1), 1 / (n - 1), 0.5 / (nt - 1)), periodic=False)
    result = ph_con_002.check(field, spec)
    assert result.rule_id == "PH-CON-002"
    assert result.status == "PASS"


def _wave_periodic_spec(n: int, nt: int, length: float) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "wave",
            "grid_shape": [n, n, nt],
            "domain": {"x": [0.0, length], "y": [0.0, length], "t": [0.0, 0.5]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "wave_speed": 1.0,
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )


def test_ph_con_002_periodic_traveling_wave_conserves_energy():
    # Regression for Codex adversarial review Finding 3: the earlier
    # draft built the energy via np.gradient on each spatial slice and
    # trapezoidal quadrature, so a traveling wave that actually crossed
    # the periodic seam accumulated spurious drift at the boundary.
    # The IBP identity plus rectangle quadrature should collapse the
    # drift back to the time-derivative truncation floor.
    n, nt = 64, 32
    length = 2 * np.pi
    spec = _wave_periodic_spec(n, nt, length)
    sol = wave_sols.periodic_traveling(c=1.0, length=length)
    x = np.linspace(0.0, length, n, endpoint=False)
    y = np.linspace(0.0, length, n, endpoint=False)
    t = np.linspace(0.0, 0.5, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    pred = np.stack([sol.u(X, Y, ti) for ti in t], axis=-1)
    field = GridField(
        pred,
        h=(length / n, length / n, 0.5 / (nt - 1)),
        periodic=True,
        backend="spectral",
    )
    result = ph_con_002.check(field, spec)
    assert result.status == "PASS"
    assert result.raw_value is not None
    # Traveling wave over 0.5 s at c=1, wavelength 2 pi: the wave
    # traverses the seam multiple times. Expect small drift from the
    # 2nd-order central time derivative only.
    assert result.raw_value < 5e-3


def test_ph_con_002_growing_amplitude_fails():
    n, nt = 32, 16
    spec = _wave_hd_spec(n, nt)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 0.5, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    base = np.sin(np.pi * X) * np.sin(np.pi * Y)
    pred = np.stack(
        [base * (1.0 + 1.0 * ti) * np.cos(np.pi * np.sqrt(2.0) * ti) for ti in t],
        axis=-1,
    )
    field = GridField(pred, h=(1 / (n - 1), 1 / (n - 1), 0.5 / (nt - 1)), periodic=False)
    result = ph_con_002.check(field, spec)
    assert result.status in {"WARN", "FAIL"}
