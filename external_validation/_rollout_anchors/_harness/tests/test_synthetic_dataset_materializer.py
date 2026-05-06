"""Unit tests for synthetic_dataset_materializer (rung 4b T7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest


def _identity_transform(positions: np.ndarray, _box_size: float) -> np.ndarray:
    """Identity transform for testing; preserves shape."""
    return positions.copy()


def test_apply_transform_to_window_appends_placeholder_frame():
    from external_validation._rollout_anchors._harness.synthetic_dataset_materializer import (
        apply_transform_to_window,
    )

    input_window = np.array(
        [
            [[0.10, 0.20], [0.30, 0.40]],
            [[0.11, 0.21], [0.31, 0.41]],
            [[0.12, 0.22], [0.32, 0.42]],
            [[0.13, 0.23], [0.33, 0.43]],
            [[0.14, 0.24], [0.34, 0.44]],
            [[0.15, 0.25], [0.35, 0.45]],
        ],
        dtype=np.float32,
    )
    out = apply_transform_to_window(
        input_window=input_window, transform_fn=_identity_transform, box_size=1.0, t_steps=7
    )
    assert out.shape == (7, 2, 2), f"expected (7, 2, 2), got {out.shape}"
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[:6], input_window)
    np.testing.assert_array_equal(out[6], input_window[5], err_msg="placeholder must equal frame 5")


def test_apply_transform_to_window_rejects_wrong_input_shape():
    from external_validation._rollout_anchors._harness.synthetic_dataset_materializer import (
        apply_transform_to_window,
    )

    bad = np.zeros((5, 2, 2), dtype=np.float32)  # only 5 frames; need 6
    with pytest.raises(ValueError, match=r"input_window must have 6 frames"):
        apply_transform_to_window(
            input_window=bad, transform_fn=_identity_transform, box_size=1.0, t_steps=7
        )


def test_apply_transform_to_window_t_steps_106_for_figure_subset():
    from external_validation._rollout_anchors._harness.synthetic_dataset_materializer import (
        apply_transform_to_window,
    )

    input_window = np.zeros((6, 4, 2), dtype=np.float32)
    out = apply_transform_to_window(
        input_window=input_window, transform_fn=_identity_transform, box_size=1.0, t_steps=106
    )
    assert out.shape == (106, 4, 2)
    # All placeholder frames (indices 6..105) equal frame 5 of the input window.
    for k in range(6, 106):
        np.testing.assert_array_equal(out[k], input_window[5])


def _make_input_windows(n_trajs: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic input windows: shape (n_trajs, 6, 4, 2). Particle types: int32 (4,)."""
    rng = np.random.default_rng(seed=42)
    windows = rng.uniform(0.0, 1.0, size=(n_trajs, 6, 4, 2)).astype(np.float32)
    particle_type = np.zeros(4, dtype=np.int32)  # all FLUID
    return windows, particle_type


def _published_metadata_stub() -> dict[str, Any]:
    """Minimal published-TGV2D-like metadata for tests."""
    return {
        "dim": 2,
        "dt": 0.01,
        "dx": 0.1,
        "bounds": [[0.0, 1.0], [0.0, 1.0]],
        "periodic_boundary_conditions": [True, True],
        "default_connectivity_radius": 0.145,
        "num_particles_max": 4,
        "vel_mean": [0.0, 0.0],
        "vel_std": [1.0, 1.0],
        "acc_mean": [0.0, 0.0],
        "acc_std": [1.0, 1.0],
        "solver": "SPH",
        "case": "TGV",
    }


def test_materialize_synthetic_dataset_writes_all_files(tmp_path):
    from external_validation._rollout_anchors._harness.synthetic_dataset_materializer import (
        materialize_synthetic_dataset,
    )

    windows, particle_type = _make_input_windows(n_trajs=2)
    transforms = [
        {
            "rule_id": "PH-SYM-001",
            "transform_kind": "rotation",
            "transform_param": "pi_2",
            "transform_fn": lambda pos, _bs: pos.copy(),
            "original_traj_index": 0,
        },
        {
            "rule_id": "PH-SYM-001",
            "transform_kind": "rotation",
            "transform_param": "pi_2",
            "transform_fn": lambda pos, _bs: pos.copy(),
            "original_traj_index": 1,
        },
    ]
    out_dir = tmp_path / "synthetic_segnn_main"
    materialize_synthetic_dataset(
        out_dir=out_dir,
        input_windows=windows,
        particle_type=particle_type,
        transforms=transforms,
        published_metadata=_published_metadata_stub(),
        t_steps=7,
        sweep_kind="main",
        stack="segnn",
        dataset="tgv2d",
        ckpt_hash="sha256:" + "a" * 64,
        physics_lint_sha_eps_computation="abcdef0123",
    )
    assert (out_dir / "test.h5").exists()
    assert (out_dir / "train.h5").exists()
    assert (out_dir / "valid.h5").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "manifest.json").exists()


def test_materialize_synthetic_dataset_test_h5_structure(tmp_path):
    from external_validation._rollout_anchors._harness.synthetic_dataset_materializer import (
        materialize_synthetic_dataset,
    )

    windows, particle_type = _make_input_windows(n_trajs=3)
    transforms = [
        {
            "rule_id": "PH-SYM-001",
            "transform_kind": "rotation",
            "transform_param": "pi_2",
            "transform_fn": lambda pos, _bs: pos.copy(),
            "original_traj_index": k,
        }
        for k in range(3)
    ]
    out_dir = tmp_path / "synthetic_main"
    materialize_synthetic_dataset(
        out_dir=out_dir,
        input_windows=windows,
        particle_type=particle_type,
        transforms=transforms,
        published_metadata=_published_metadata_stub(),
        t_steps=7,
        sweep_kind="main",
        stack="segnn",
        dataset="tgv2d",
        ckpt_hash="sha256:" + "a" * 64,
        physics_lint_sha_eps_computation="abcdef0123",
    )
    with h5py.File(out_dir / "test.h5", "r") as f:
        groups = sorted(f.keys())
        assert groups == ["00000", "00001", "00002"], f"got {groups}"
        for k in range(3):
            grp = f[f"{k:05d}"]
            assert "position" in grp
            assert "particle_type" in grp
            assert grp["position"].shape == (7, 4, 2)
            assert grp["position"].dtype == np.float32
            assert grp["particle_type"].shape == (4,)
            assert grp["particle_type"].dtype == np.int32


def test_materialize_synthetic_dataset_metadata_reuses_published_stats(tmp_path):
    from external_validation._rollout_anchors._harness.synthetic_dataset_materializer import (
        materialize_synthetic_dataset,
    )

    windows, particle_type = _make_input_windows(n_trajs=1)
    transforms = [
        {
            "rule_id": "PH-SYM-001",
            "transform_kind": "rotation",
            "transform_param": "pi_2",
            "transform_fn": lambda pos, _bs: pos.copy(),
            "original_traj_index": 0,
        }
    ]
    out_dir = tmp_path / "synthetic"
    pub = _published_metadata_stub()
    materialize_synthetic_dataset(
        out_dir=out_dir,
        input_windows=windows,
        particle_type=particle_type,
        transforms=transforms,
        published_metadata=pub,
        t_steps=7,
        sweep_kind="main",
        stack="segnn",
        dataset="tgv2d",
        ckpt_hash="sha256:" + "a" * 64,
        physics_lint_sha_eps_computation="abcdef0123",
    )
    written = json.loads((out_dir / "metadata.json").read_text())
    # Reuse-verbatim hazard fields:
    assert written["vel_mean"] == pub["vel_mean"]
    assert written["vel_std"] == pub["vel_std"]
    assert written["acc_mean"] == pub["acc_mean"]
    assert written["acc_std"] == pub["acc_std"]
    assert written["bounds"] == pub["bounds"]
    assert written["periodic_boundary_conditions"] == pub["periodic_boundary_conditions"]
    assert written["default_connectivity_radius"] == pub["default_connectivity_radius"]
    assert written["dt"] == pub["dt"]
    # Synthesized split sizes:
    assert written["num_trajs_test"] == 1
    assert written["num_trajs_train"] == 1
    assert written["sequence_length_test"] == 7


def test_materialize_synthetic_dataset_manifest_schema(tmp_path):
    from external_validation._rollout_anchors._harness.synthetic_dataset_materializer import (
        materialize_synthetic_dataset,
    )

    windows, particle_type = _make_input_windows(n_trajs=2)
    transforms = [
        {
            "rule_id": "PH-SYM-001",
            "transform_kind": "rotation",
            "transform_param": "pi_2",
            "transform_fn": lambda pos, _bs: pos.copy(),
            "original_traj_index": 0,
        },
        {
            "rule_id": "PH-SYM-002",
            "transform_kind": "reflection",
            "transform_param": "y_axis",
            "transform_fn": lambda pos, _bs: pos.copy(),
            "original_traj_index": 1,
        },
    ]
    out_dir = tmp_path / "synthetic"
    materialize_synthetic_dataset(
        out_dir=out_dir,
        input_windows=windows,
        particle_type=particle_type,
        transforms=transforms,
        published_metadata=_published_metadata_stub(),
        t_steps=7,
        sweep_kind="main",
        stack="segnn",
        dataset="tgv2d",
        ckpt_hash="sha256:" + "a" * 64,
        physics_lint_sha_eps_computation="abcdef0123",
    )
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "1.0"
    assert manifest["stack"] == "segnn"
    assert manifest["dataset"] == "tgv2d"
    assert manifest["sweep_kind"] == "main"
    assert manifest["physics_lint_sha_eps_computation"] == "abcdef0123"
    assert manifest["ckpt_hash"].startswith("sha256:")
    assert len(manifest["trajectories"]) == 2
    # Contiguity: synthetic_traj_index covers range(2)
    indices = [t["synthetic_traj_index"] for t in manifest["trajectories"]]
    assert indices == [0, 1]
    # Per-trajectory mapping is preserved verbatim from `transforms` arg
    assert manifest["trajectories"][0]["rule_id"] == "PH-SYM-001"
    assert manifest["trajectories"][0]["transform_kind"] == "rotation"
    assert manifest["trajectories"][0]["transform_param"] == "pi_2"
    assert manifest["trajectories"][0]["original_traj_index"] == 0
    assert manifest["trajectories"][1]["rule_id"] == "PH-SYM-002"


def test_materialize_synthetic_dataset_rejects_input_window_traj_count_mismatch(tmp_path):
    """If transforms reference original_traj_index outside input_windows.shape[0],
    fail loud rather than silently using bogus data."""
    from external_validation._rollout_anchors._harness.synthetic_dataset_materializer import (
        materialize_synthetic_dataset,
    )

    windows, particle_type = _make_input_windows(n_trajs=2)
    transforms = [
        {
            "rule_id": "PH-SYM-001",
            "transform_kind": "rotation",
            "transform_param": "pi_2",
            "transform_fn": lambda pos, _bs: pos.copy(),
            "original_traj_index": 5,  # out of range
        }
    ]
    with pytest.raises(IndexError, match=r"original_traj_index 5 out of range"):
        materialize_synthetic_dataset(
            out_dir=tmp_path / "synthetic",
            input_windows=windows,
            particle_type=particle_type,
            transforms=transforms,
            published_metadata=_published_metadata_stub(),
            t_steps=7,
            sweep_kind="main",
            stack="segnn",
            dataset="tgv2d",
            ckpt_hash="sha256:" + "a" * 64,
            physics_lint_sha_eps_computation="abcdef0123",
        )


def test_materialize_synthetic_dataset_rejects_unnamespaced_ckpt_hash(tmp_path):
    """ckpt_hash must be `sha256:<hex>` per design §3.4."""
    from external_validation._rollout_anchors._harness.synthetic_dataset_materializer import (
        materialize_synthetic_dataset,
    )

    windows, particle_type = _make_input_windows(n_trajs=1)
    transforms = [
        {
            "rule_id": "PH-SYM-001",
            "transform_kind": "rotation",
            "transform_param": "pi_2",
            "transform_fn": lambda pos, _bs: pos.copy(),
            "original_traj_index": 0,
        }
    ]
    with pytest.raises(ValueError, match=r"ckpt_hash must be namespaced"):
        materialize_synthetic_dataset(
            out_dir=tmp_path / "synthetic",
            input_windows=windows,
            particle_type=particle_type,
            transforms=transforms,
            published_metadata=_published_metadata_stub(),
            t_steps=7,
            sweep_kind="main",
            stack="segnn",
            dataset="tgv2d",
            ckpt_hash="a" * 64,  # missing sha256: prefix
            physics_lint_sha_eps_computation="abcdef0123",
        )


def _create_published_test_h5_fixture(tmp_path: Path, n_trajs: int = 3) -> Path:
    """Create a fixture H5 mimicking the published TGV2D test.h5 shape."""
    h5_path = tmp_path / "test.h5"
    rng = np.random.default_rng(seed=7)
    particle_type = np.zeros(4, dtype=np.int32)
    with h5py.File(h5_path, "w") as f:
        for k in range(n_trajs):
            grp = f.create_group(f"{k:05d}")
            # Published trajs have full T=125 frames (LB convention); we read only 0:6.
            grp.create_dataset(
                "position", data=rng.uniform(0.0, 1.0, size=(125, 4, 2)).astype(np.float32)
            )
            grp.create_dataset("particle_type", data=particle_type)
    return h5_path


def test_read_published_input_windows_extracts_first_6_frames(tmp_path):
    from external_validation._rollout_anchors._harness.synthetic_dataset_materializer import (
        read_published_input_windows,
    )

    h5_path = _create_published_test_h5_fixture(tmp_path, n_trajs=3)
    windows, particle_type = read_published_input_windows(h5_path=h5_path, n_trajs=3)
    assert windows.shape == (3, 6, 4, 2), f"got {windows.shape}"
    assert windows.dtype == np.float32
    assert particle_type.shape == (4,)
    assert particle_type.dtype == np.int32

    # Verify each window equals the first 6 frames of the corresponding traj
    with h5py.File(h5_path, "r") as f:
        for k in range(3):
            np.testing.assert_array_equal(windows[k], f[f"{k:05d}/position"][0:6])


def test_read_published_input_windows_rejects_too_few_trajs(tmp_path):
    from external_validation._rollout_anchors._harness.synthetic_dataset_materializer import (
        read_published_input_windows,
    )

    h5_path = _create_published_test_h5_fixture(tmp_path, n_trajs=2)
    with pytest.raises(ValueError, match=r"requested 5 trajs but only 2 available"):
        read_published_input_windows(h5_path=h5_path, n_trajs=5)
