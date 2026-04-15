"""Time-dependent loader tests — heat PDE, .npz dump with (Nx, Ny, Nt) prediction."""

from pathlib import Path

import numpy as np

from physics_lint import GridField
from physics_lint.loader import load_target


def test_dump_with_time_axis_heat(tmp_path: Path):
    N = 32  # noqa: N806
    Nt = 8  # noqa: N806
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    base = np.sin(np.pi * X) * np.sin(np.pi * Y)
    t = np.linspace(0.0, 1.0, Nt)
    pred = np.stack([base * np.exp(-2 * np.pi**2 * 0.01 * ti) for ti in t], axis=-1)

    metadata = {
        "pde": "heat",
        "grid_shape": [N, N, Nt],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "diffusivity": 0.01,
        "field": {"type": "grid", "backend": "fd"},
    }

    path = tmp_path / "pred.npz"
    np.savez(path, prediction=pred, metadata=metadata)

    loaded = load_target(path, cli_overrides={}, toml_path=None)
    assert loaded.spec.pde == "heat"
    assert loaded.spec.domain.is_time_dependent is True
    assert isinstance(loaded.field, GridField)
    assert loaded.field.values().shape == (N, N, Nt)
    # h for time axis should be (t_hi - t_lo) / (Nt - 1) = 1/7 for endpoint-inclusive
    assert abs(loaded.field.h[-1] - (1.0 / (Nt - 1))) < 1e-12
    # spatial h
    assert abs(loaded.field.h[0] - (1.0 / (N - 1))) < 1e-12


def test_dump_with_time_axis_wave(tmp_path: Path):
    N = 16  # noqa: N806
    Nt = 4  # noqa: N806
    pred = np.zeros((N, N, Nt))
    metadata = {
        "pde": "wave",
        "grid_shape": [N, N, Nt],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.5]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "wave_speed": 1.0,
        "field": {"type": "grid", "backend": "fd"},
    }
    path = tmp_path / "wave.npz"
    np.savez(path, prediction=pred, metadata=metadata)

    loaded = load_target(path, cli_overrides={}, toml_path=None)
    assert loaded.spec.pde == "wave"
    assert loaded.field.values().shape == (N, N, Nt)
    assert abs(loaded.field.h[-1] - (0.5 / (Nt - 1))) < 1e-12


def test_dump_with_time_axis_periodic_heat(tmp_path: Path):
    # Periodic spatial + time-dependent: spatial spacings use L/N (exclusive),
    # time still uses (T_hi - T_lo)/(Nt - 1) because time is sampled with
    # endpoints — the periodicity only applies to the spatial axes.
    N = 32  # noqa: N806
    Nt = 8  # noqa: N806
    pred = np.zeros((N, N, Nt))
    metadata = {
        "pde": "heat",
        "grid_shape": [N, N, Nt],
        "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi], "t": [0.0, 0.5]},
        "periodic": True,
        "boundary_condition": {"kind": "periodic"},
        "diffusivity": 0.01,
        "field": {"type": "grid", "backend": "spectral"},
    }
    path = tmp_path / "pred.npz"
    np.savez(path, prediction=pred, metadata=metadata)

    loaded = load_target(path, cli_overrides={}, toml_path=None)
    # Spatial axes: L / N = 2 pi / 32
    assert abs(loaded.field.h[0] - (2 * np.pi / N)) < 1e-12
    assert abs(loaded.field.h[1] - (2 * np.pi / N)) < 1e-12
    # Time axis: T / (Nt - 1) — NOT periodic
    assert abs(loaded.field.h[2] - (0.5 / (Nt - 1))) < 1e-12
