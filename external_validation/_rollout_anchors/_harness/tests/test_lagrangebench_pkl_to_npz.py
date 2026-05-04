"""Tests for ``lagrangebench_pkl_to_npz.convert_rollout_dir`` and helpers.

Synthetic-fixture tests only (no real LagrangeBench install, no JAX).
The synthetic pkls match the shape LagrangeBench's
``lagrangebench/evaluate/rollout.py:271-297`` writes; the synthetic
``metadata.json`` matches ``tests/3D_LJ_3_1214every1/metadata.json``
modulo dimensionality and bounds. Tests cover both the load-bearing
arithmetic (central differences, bounds transpose, dt computation)
and the validation surface (the five assertions enumerated in
DECISIONS.md D0-15 amendment 4).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from external_validation._rollout_anchors._harness.lagrangebench_pkl_to_npz import (
    RolloutMetadata,
    _central_diff_velocities,
    _hash_directory,
    convert_rollout_dir,
)
from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    load_rollout_npz,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _caller_metadata() -> RolloutMetadata:
    return RolloutMetadata(
        git_sha="abc123",
        lagrangebench_sha="def456",
        dataset="tgv2d",
        model="segnn",
        seed=0,
        framework="jax+haiku",
        framework_version="0.4.29",
    )


def _write_metadata_json(
    path: Path,
    *,
    dt: float = 0.005,
    write_every: int | None = 100,
    bounds: list[list[float]] | None = None,
) -> None:
    """Write a synthetic LagrangeBench-shaped metadata.json.

    bounds default is 2D unit square in (D, 2) per-axis-[min, max] layout.
    Pass ``write_every=None`` to omit the field (mimics the tutorial
    fixture ``tests/3D_LJ_3_1214every1/metadata.json``).
    """
    if bounds is None:
        bounds = [[0.0, 1.0], [0.0, 1.0]]
    payload: dict = {
        "dt": dt,
        "bounds": bounds,
    }
    if write_every is not None:
        payload["write_every"] = write_every
    with open(path, "w") as f:
        json.dump(payload, f)


def _write_lb_rollout_pkl(
    path: Path,
    *,
    n_timesteps: int = 5,
    n_particles: int = 4,
    d_dim: int = 2,
    velocity_per_step: tuple[float, ...] | None = None,
    initial_position_seed: int = 0,
) -> np.ndarray:
    """Write a LagrangeBench-shaped rollout pkl. Returns the positions.

    The rollout is constant-velocity (each particle drifts by
    ``velocity_per_step`` each frame), which makes the analytical
    central-difference velocity exactly ``velocity_per_step / dt``
    at every interior timestep — useful for the velocity-derivation
    tests. ``velocity_per_step`` defaults to all-zero in d_dim
    dimensions when omitted.
    """
    rng = np.random.default_rng(initial_position_seed)
    p0 = rng.uniform(0.1, 0.9, size=(n_particles, d_dim))
    if velocity_per_step is None:
        velocity_per_step = (0.0,) * d_dim
    drift_vec = np.asarray(velocity_per_step, dtype=np.float64)
    if drift_vec.shape != (d_dim,):
        raise ValueError(
            f"velocity_per_step shape {drift_vec.shape} must be ({d_dim},) for d_dim={d_dim}"
        )
    drift = np.broadcast_to(drift_vec, (n_particles, d_dim))
    predicted = np.stack([p0 + t * drift for t in range(n_timesteps)], axis=0)
    blob = {
        "predicted_rollout": predicted.astype(np.float32),
        "ground_truth_rollout": predicted[:1].astype(np.float32),  # len-1 input window
        "particle_type": np.zeros(n_particles, dtype=np.int32),
    }
    with open(path, "wb") as f:
        pickle.dump(blob, f)
    return predicted


def _make_ckpt_dir(root: Path, *, content: bytes = b"fake-checkpoint-payload") -> Path:
    ckpt = root / "best"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "params.pkl").write_bytes(content)
    (ckpt / "config.yaml").write_text("seed: 0\n")
    return ckpt


# ---------------------------------------------------------------------------
# 1. Central-difference velocity derivation
# ---------------------------------------------------------------------------


def test_central_diff_velocities_interior_constant_motion() -> None:
    """Constant per-step drift → central-diff velocity = drift/dt at interior."""
    n_timesteps, n_particles = 7, 3
    dt = 0.5
    drift = np.array([0.1, -0.2])
    positions = np.stack([np.tile(t * drift, (n_particles, 1)) for t in range(n_timesteps)], axis=0)
    velocities = _central_diff_velocities(positions, dt)
    expected = drift / dt
    for t in range(1, n_timesteps - 1):
        np.testing.assert_allclose(velocities[t], np.tile(expected, (n_particles, 1)))


def test_central_diff_velocities_endpoints_first_order() -> None:
    """t=0 is forward diff; t=T-1 is backward diff."""
    positions = np.array([[[0.0, 0.0]], [[1.0, 2.0]], [[3.0, 5.0]]], dtype=np.float64)
    dt = 0.1
    velocities = _central_diff_velocities(positions, dt)
    np.testing.assert_allclose(velocities[0], (positions[1] - positions[0]) / dt)
    np.testing.assert_allclose(velocities[-1], (positions[-1] - positions[-2]) / dt)
    np.testing.assert_allclose(velocities[1], (positions[2] - positions[0]) / (2.0 * dt))


def test_central_diff_velocities_requires_two_timesteps() -> None:
    with pytest.raises(ValueError, match="T >= 2"):
        _central_diff_velocities(np.zeros((1, 4, 2)), dt=0.1)


# ---------------------------------------------------------------------------
# 2. Directory hashing
# ---------------------------------------------------------------------------


def test_hash_directory_deterministic(tmp_path: Path) -> None:
    ckpt_a = _make_ckpt_dir(tmp_path / "a")
    ckpt_b = _make_ckpt_dir(tmp_path / "b")  # same contents, different parent
    assert _hash_directory(ckpt_a) == _hash_directory(ckpt_b)


def test_hash_directory_changes_with_content(tmp_path: Path) -> None:
    ckpt_a = _make_ckpt_dir(tmp_path / "a", content=b"payload-v1")
    ckpt_b = _make_ckpt_dir(tmp_path / "b", content=b"payload-v2")
    assert _hash_directory(ckpt_a) != _hash_directory(ckpt_b)


# ---------------------------------------------------------------------------
# 3. End-to-end conversion
# ---------------------------------------------------------------------------


def test_convert_produces_schema_conformant_npz(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_lb_rollout_pkl(rollout_dir / "rollout_0.pkl", n_timesteps=6)
    metadata_path = tmp_path / "metadata.json"
    _write_metadata_json(metadata_path)
    ckpt_dir = _make_ckpt_dir(tmp_path)

    written = convert_rollout_dir(rollout_dir, metadata_path, ckpt_dir, metadata=_caller_metadata())
    assert len(written) == 1
    assert written[0].name == "particle_rollout_traj00.npz"
    assert written[0].exists()


def test_round_trip_via_load_rollout_npz(tmp_path: Path) -> None:
    """convert_rollout_dir output is consumable by particle_rollout_adapter.load_rollout_npz."""
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_lb_rollout_pkl(rollout_dir / "rollout_0.pkl", n_timesteps=8, n_particles=5)
    metadata_path = tmp_path / "metadata.json"
    _write_metadata_json(metadata_path, dt=0.001, write_every=10)
    ckpt_dir = _make_ckpt_dir(tmp_path)

    written = convert_rollout_dir(rollout_dir, metadata_path, ckpt_dir, metadata=_caller_metadata())
    rollout = load_rollout_npz(written[0])

    assert rollout.positions.shape == (8, 5, 2)
    assert rollout.velocities.shape == (8, 5, 2)
    assert rollout.particle_type.shape == (5,)
    assert rollout.particle_mass.shape == (5,)
    assert rollout.dt == pytest.approx(0.001 * 10)
    assert rollout.domain_box.shape == (2, 2)
    assert rollout.metadata["dataset"] == "tgv2d"
    assert rollout.metadata["model"] == "segnn"
    assert rollout.metadata["seed"] == 0
    assert rollout.metadata["framework"] == "jax+haiku"
    assert rollout.metadata["framework_version"] == "0.4.29"
    assert rollout.metadata["git_sha"] == "abc123"
    assert rollout.metadata["lagrangebench_sha"] == "def456"
    assert rollout.metadata["ckpt_path"].endswith("best")
    assert len(rollout.metadata["ckpt_hash"]) == 64  # SHA-256 hex
    assert rollout.metadata["write_every"] == 10
    assert rollout.metadata["write_every_source"] == "dataset"


def test_multiple_trajectories_zero_padded_filenames(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    for j in range(3):
        _write_lb_rollout_pkl(rollout_dir / f"rollout_{j}.pkl")
    metadata_path = tmp_path / "metadata.json"
    _write_metadata_json(metadata_path)
    ckpt_dir = _make_ckpt_dir(tmp_path)

    written = convert_rollout_dir(rollout_dir, metadata_path, ckpt_dir, metadata=_caller_metadata())
    names = sorted(p.name for p in written)
    assert names == [
        "particle_rollout_traj00.npz",
        "particle_rollout_traj01.npz",
        "particle_rollout_traj02.npz",
    ]


def test_metrics_pkl_left_untouched(tmp_path: Path) -> None:
    """metrics{timestamp}.pkl in rollout_dir must not be consumed or overwritten."""
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_lb_rollout_pkl(rollout_dir / "rollout_0.pkl")
    metrics_path = rollout_dir / "metrics2026_05_04_12_00_00.pkl"
    metrics_path.write_bytes(b"opaque-metrics-blob")
    metadata_path = tmp_path / "metadata.json"
    _write_metadata_json(metadata_path)
    ckpt_dir = _make_ckpt_dir(tmp_path)

    convert_rollout_dir(rollout_dir, metadata_path, ckpt_dir, metadata=_caller_metadata())
    assert metrics_path.read_bytes() == b"opaque-metrics-blob"


# ---------------------------------------------------------------------------
# 4. Bounds transpose + dt computation
# ---------------------------------------------------------------------------


def test_bounds_transpose_d_2_to_2_d(tmp_path: Path) -> None:
    """LagrangeBench (D, 2) [[xmin,xmax],[ymin,ymax]] -> SCHEMA (2, D) [[xmin,ymin],[xmax,ymax]]."""
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_lb_rollout_pkl(rollout_dir / "rollout_0.pkl")
    metadata_path = tmp_path / "metadata.json"
    _write_metadata_json(
        metadata_path, bounds=[[-1.0, 2.0], [-3.0, 4.0]]
    )  # x in [-1, 2], y in [-3, 4]
    ckpt_dir = _make_ckpt_dir(tmp_path)

    written = convert_rollout_dir(rollout_dir, metadata_path, ckpt_dir, metadata=_caller_metadata())
    rollout = load_rollout_npz(written[0])
    np.testing.assert_array_equal(rollout.domain_box[0], [-1.0, -3.0])  # mins per axis
    np.testing.assert_array_equal(rollout.domain_box[1], [2.0, 4.0])  # maxes per axis


def test_write_every_source_dataset_when_present(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_lb_rollout_pkl(rollout_dir / "rollout_0.pkl")
    metadata_path = tmp_path / "metadata.json"
    _write_metadata_json(metadata_path, dt=0.01, write_every=50)
    ckpt_dir = _make_ckpt_dir(tmp_path)

    written = convert_rollout_dir(rollout_dir, metadata_path, ckpt_dir, metadata=_caller_metadata())
    rollout = load_rollout_npz(written[0])
    assert rollout.metadata["write_every"] == 50
    assert rollout.metadata["write_every_source"] == "dataset"
    assert rollout.dt == pytest.approx(0.01 * 50)


def test_write_every_source_default_when_missing(tmp_path: Path) -> None:
    """Mimics the tutorial fixture (tests/3D_LJ_3_1214every1/metadata.json)."""
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_lb_rollout_pkl(rollout_dir / "rollout_0.pkl")
    metadata_path = tmp_path / "metadata.json"
    _write_metadata_json(metadata_path, dt=0.005, write_every=None)
    ckpt_dir = _make_ckpt_dir(tmp_path)

    written = convert_rollout_dir(rollout_dir, metadata_path, ckpt_dir, metadata=_caller_metadata())
    rollout = load_rollout_npz(written[0])
    assert rollout.metadata["write_every"] == 1
    assert rollout.metadata["write_every_source"] == "default"
    assert rollout.dt == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# 5. Particle mass uniform unit + fp64
# ---------------------------------------------------------------------------


def test_particle_mass_uniform_unit_fp64(tmp_path: Path) -> None:
    """Conversion writes uniform unit mass per SCHEMA.md §1 v1.2 default."""
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_lb_rollout_pkl(rollout_dir / "rollout_0.pkl", n_particles=7)
    metadata_path = tmp_path / "metadata.json"
    _write_metadata_json(metadata_path)
    ckpt_dir = _make_ckpt_dir(tmp_path)

    written = convert_rollout_dir(rollout_dir, metadata_path, ckpt_dir, metadata=_caller_metadata())
    with np.load(written[0], allow_pickle=True) as data:
        pm = data["particle_mass"]
    assert pm.shape == (7,)
    assert pm.dtype == np.float64
    np.testing.assert_array_equal(pm, np.ones(7, dtype=np.float64))


# ---------------------------------------------------------------------------
# 6. Validation surface — assertions surface failures at conversion time
# ---------------------------------------------------------------------------


def test_rejects_bounds_min_above_max(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_lb_rollout_pkl(rollout_dir / "rollout_0.pkl")
    metadata_path = tmp_path / "metadata.json"
    _write_metadata_json(metadata_path, bounds=[[1.0, 0.0], [0.0, 1.0]])  # x has min=1 > max=0
    ckpt_dir = _make_ckpt_dir(tmp_path)

    with pytest.raises(ValueError, match="mins must be strictly less than maxes"):
        convert_rollout_dir(rollout_dir, metadata_path, ckpt_dir, metadata=_caller_metadata())


def test_rejects_bounds_wrong_shape(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_lb_rollout_pkl(rollout_dir / "rollout_0.pkl")
    metadata_path = tmp_path / "metadata.json"
    _write_metadata_json(metadata_path, bounds=[[0.0], [1.0]])  # not (D, 2)
    ckpt_dir = _make_ckpt_dir(tmp_path)

    with pytest.raises(ValueError, match="must have shape \\(D, 2\\)"):
        convert_rollout_dir(rollout_dir, metadata_path, ckpt_dir, metadata=_caller_metadata())


def test_rejects_missing_dt(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_lb_rollout_pkl(rollout_dir / "rollout_0.pkl")
    metadata_path = tmp_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({"bounds": [[0.0, 1.0], [0.0, 1.0]]}, f)
    ckpt_dir = _make_ckpt_dir(tmp_path)

    with pytest.raises(KeyError, match="missing required key 'dt'"):
        convert_rollout_dir(rollout_dir, metadata_path, ckpt_dir, metadata=_caller_metadata())


def test_rejects_missing_bounds(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_lb_rollout_pkl(rollout_dir / "rollout_0.pkl")
    metadata_path = tmp_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({"dt": 0.005}, f)
    ckpt_dir = _make_ckpt_dir(tmp_path)

    with pytest.raises(KeyError, match="missing required key 'bounds'"):
        convert_rollout_dir(rollout_dir, metadata_path, ckpt_dir, metadata=_caller_metadata())


def test_rejects_no_pkl_files(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    metadata_path = tmp_path / "metadata.json"
    _write_metadata_json(metadata_path)
    ckpt_dir = _make_ckpt_dir(tmp_path)

    with pytest.raises(FileNotFoundError, match="no rollout_\\*\\.pkl files found"):
        convert_rollout_dir(rollout_dir, metadata_path, ckpt_dir, metadata=_caller_metadata())


def test_rejects_missing_metadata_file(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_lb_rollout_pkl(rollout_dir / "rollout_0.pkl")
    ckpt_dir = _make_ckpt_dir(tmp_path)

    with pytest.raises(FileNotFoundError, match=r"dataset metadata\.json not found"):
        convert_rollout_dir(
            rollout_dir, tmp_path / "missing.json", ckpt_dir, metadata=_caller_metadata()
        )


def test_rejects_missing_ckpt_dir(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_lb_rollout_pkl(rollout_dir / "rollout_0.pkl")
    metadata_path = tmp_path / "metadata.json"
    _write_metadata_json(metadata_path)

    with pytest.raises(FileNotFoundError, match="checkpoint dir not found"):
        convert_rollout_dir(
            rollout_dir, metadata_path, tmp_path / "missing_ckpt", metadata=_caller_metadata()
        )


def test_rejects_pkl_with_d_mismatch(tmp_path: Path) -> None:
    """predicted_rollout D must match dataset bounds D."""
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_lb_rollout_pkl(rollout_dir / "rollout_0.pkl", d_dim=3)  # 3D positions
    metadata_path = tmp_path / "metadata.json"
    _write_metadata_json(metadata_path, bounds=[[0.0, 1.0], [0.0, 1.0]])  # 2D bounds
    ckpt_dir = _make_ckpt_dir(tmp_path)

    with pytest.raises(ValueError, match="does not match dataset metadata bounds D"):
        convert_rollout_dir(rollout_dir, metadata_path, ckpt_dir, metadata=_caller_metadata())


# ---------------------------------------------------------------------------
# 7. RolloutMetadata dataclass
# ---------------------------------------------------------------------------


def test_rollout_metadata_to_dict_round_trip() -> None:
    m = RolloutMetadata(
        git_sha="g",
        lagrangebench_sha="l",
        dataset="d",
        model="m",
        seed=42,
        framework="f",
        framework_version="v",
    )
    d = m.to_dict()
    assert d["git_sha"] == "g"
    assert d["seed"] == 42
    assert d["ckpt_hash"] == ""  # default for caller-supplied partial
    assert d["write_every"] == 1
    assert d["write_every_source"] == "default"


def test_rollout_metadata_replace_works_for_runtime_fields() -> None:
    m = _caller_metadata()
    m2 = replace(m, ckpt_hash="h" * 64, write_every=100, write_every_source="dataset")
    assert m2.ckpt_hash == "h" * 64
    assert m2.write_every == 100
    assert m2.write_every_source == "dataset"
    # Original unchanged (frozen dataclass)
    assert m.ckpt_hash == ""
