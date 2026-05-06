"""Unit tests for eps_pkl_consumer (rung 4b T7)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest


def test_inverse_transform_per_step_rotation_pi_2_reverses_rotation():
    from external_validation._rollout_anchors._harness.eps_pkl_consumer import (
        inverse_transform_per_step,
    )
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        rotate_about_box_center,
    )

    # Original positions (T=2 steps, N=4 particles).
    original = np.array(
        [
            [[0.10, 0.20], [0.30, 0.40], [0.50, 0.60], [0.70, 0.80]],
            [[0.11, 0.21], [0.31, 0.41], [0.51, 0.61], [0.71, 0.81]],
        ],
        dtype=np.float32,
    )
    velocities_dummy = np.zeros_like(original[0])
    box_size = 1.0

    # Rotate each step by pi/2.
    rotated = np.stack(
        [
            rotate_about_box_center(
                positions=original[t],
                velocities=velocities_dummy,
                theta=np.pi / 2,
                box_size=box_size,
            )[0]
            for t in range(2)
        ],
        axis=0,
    )

    recovered = inverse_transform_per_step(
        positions=rotated,
        transform_kind="rotation",
        transform_param="pi_2",
        box_size=box_size,
    )
    np.testing.assert_allclose(recovered, original, atol=1e-5)


def test_inverse_transform_per_step_reflection_is_involution():
    from external_validation._rollout_anchors._harness.eps_pkl_consumer import (
        inverse_transform_per_step,
    )
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        reflect_y_axis,
    )

    original = np.array(
        [[[0.30, 0.50], [0.70, 0.50]]],
        dtype=np.float32,
    )
    velocities_dummy = np.zeros_like(original[0])
    box_size = 1.0
    reflected = reflect_y_axis(
        positions=original[0], velocities=velocities_dummy, box_size=box_size
    )[0][None, ...]
    recovered = inverse_transform_per_step(
        positions=reflected,
        transform_kind="reflection",
        transform_param="y_axis",
        box_size=box_size,
    )
    np.testing.assert_allclose(recovered, original, atol=1e-5)


def test_inverse_transform_per_step_translation_subtracts_t():
    from external_validation._rollout_anchors._harness.eps_pkl_consumer import (
        inverse_transform_per_step,
    )

    original = np.array(
        [[[0.10, 0.20], [0.30, 0.40]]],
        dtype=np.float32,
    )
    box_size = 1.0
    # Translate by (1/3, 1/7) and PBC wrap.
    t = np.array([1 / 3, 1 / 7], dtype=np.float32)
    translated = np.mod(original + t, box_size).astype(np.float32)

    recovered = inverse_transform_per_step(
        positions=translated,
        transform_kind="translation",
        transform_param="L_3_L_7",
        box_size=box_size,
    )
    np.testing.assert_allclose(recovered, original, atol=1e-5)


def test_inverse_transform_per_step_identity_returns_input():
    from external_validation._rollout_anchors._harness.eps_pkl_consumer import (
        inverse_transform_per_step,
    )

    original = np.zeros((1, 4, 2), dtype=np.float32)
    recovered = inverse_transform_per_step(
        positions=original,
        transform_kind="identity",
        transform_param="0",
        box_size=1.0,
    )
    np.testing.assert_array_equal(recovered, original)


def test_inverse_transform_per_step_rejects_unknown_kind():
    from external_validation._rollout_anchors._harness.eps_pkl_consumer import (
        inverse_transform_per_step,
    )

    with pytest.raises(ValueError, match=r"unknown transform_kind"):
        inverse_transform_per_step(
            positions=np.zeros((1, 4, 2), dtype=np.float32),
            transform_kind="bogus",
            transform_param="x",
            box_size=1.0,
        )


def _create_synthetic_pkl(
    pkl_path: Path,
    predicted_rollout: np.ndarray,
    particle_type: np.ndarray,
) -> None:
    """Create a synthetic LB-shaped pkl for testing."""
    blob = {
        "predicted_rollout": predicted_rollout,
        "ground_truth_rollout": predicted_rollout,
        "particle_type": particle_type,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(blob, f)


def _create_synthetic_reference_npz(
    npz_path: Path,
    positions: np.ndarray,
    particle_type: np.ndarray,
) -> None:
    """Create a synthetic rung-4a-shaped npz for testing."""
    n = positions.shape[1]
    np.savez(
        npz_path,
        positions=positions.astype(np.float32),
        velocities=np.zeros_like(positions, dtype=np.float32),
        particle_type=particle_type.astype(np.int32),
        particle_mass=np.ones(n, dtype=np.float64),
        dt=np.float64(0.01),
        domain_box=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        metadata=np.array({"dummy": "ok"}, dtype=object),
    )


def test_eps_t_from_pkl_and_reference_identity_yields_zero(tmp_path):
    """If the synthetic and reference pkls have identical predicted_rollout,
    eps_t should be ~0 across all steps."""
    from external_validation._rollout_anchors._harness.eps_pkl_consumer import (
        eps_t_from_pkl_and_reference,
    )

    rng = np.random.default_rng(seed=11)
    # Reference: shape (106, N, D) — input window + 100 predictions.
    ref_positions = rng.uniform(0.0, 1.0, size=(106, 4, 2)).astype(np.float32)
    particle_type = np.zeros(4, dtype=np.int32)

    pkl_path = tmp_path / "rollout_0.pkl"
    npz_path = tmp_path / "particle_rollout_traj00.npz"
    _create_synthetic_pkl(pkl_path, ref_positions, particle_type)
    _create_synthetic_reference_npz(npz_path, ref_positions, particle_type)

    eps_t = eps_t_from_pkl_and_reference(
        synthetic_pkl_path=pkl_path,
        reference_npz_path=npz_path,
        transform_kind="identity",
        transform_param="0",
        t_steps=1,
        box_size=1.0,
    )
    assert eps_t.shape == (1,)
    assert eps_t[0] < 1e-7, f"identity eps should be ~0, got {eps_t[0]}"


def test_eps_t_from_pkl_and_reference_rotation_pi_2_round_trips_to_zero(tmp_path):
    """If the synthetic predicted_rollout is the rotated version of the reference
    (perfect equivariance), R^-1 applied to synthetic recovers reference, eps ~0."""
    from external_validation._rollout_anchors._harness.eps_pkl_consumer import (
        eps_t_from_pkl_and_reference,
    )
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        rotate_about_box_center,
    )

    rng = np.random.default_rng(seed=13)
    ref_positions = rng.uniform(0.0, 1.0, size=(106, 4, 2)).astype(np.float32)
    particle_type = np.zeros(4, dtype=np.int32)
    velocities_dummy = np.zeros_like(ref_positions[0])

    # Build "synthetic predicted_rollout" = perfectly rotated reference.
    rotated = np.stack(
        [
            rotate_about_box_center(
                positions=ref_positions[t],
                velocities=velocities_dummy,
                theta=np.pi / 2,
                box_size=1.0,
            )[0]
            for t in range(106)
        ],
        axis=0,
    )

    pkl_path = tmp_path / "rollout_0.pkl"
    npz_path = tmp_path / "particle_rollout_traj00.npz"
    _create_synthetic_pkl(pkl_path, rotated, particle_type)
    _create_synthetic_reference_npz(npz_path, ref_positions, particle_type)

    eps_t = eps_t_from_pkl_and_reference(
        synthetic_pkl_path=pkl_path,
        reference_npz_path=npz_path,
        transform_kind="rotation",
        transform_param="pi_2",
        t_steps=1,
        box_size=1.0,
    )
    assert eps_t.shape == (1,)
    # Rotation arithmetic is in float32; expect float32 floor (~1e-7).
    assert eps_t[0] < 1e-5, f"perfect-equivariance eps should be near float32 floor, got {eps_t[0]}"


def test_eps_t_from_pkl_and_reference_t_steps_100_for_figure(tmp_path):
    from external_validation._rollout_anchors._harness.eps_pkl_consumer import (
        eps_t_from_pkl_and_reference,
    )

    rng = np.random.default_rng(seed=17)
    ref_positions = rng.uniform(0.0, 1.0, size=(106, 4, 2)).astype(np.float32)
    particle_type = np.zeros(4, dtype=np.int32)

    pkl_path = tmp_path / "rollout_0.pkl"
    npz_path = tmp_path / "particle_rollout_traj00.npz"
    _create_synthetic_pkl(pkl_path, ref_positions, particle_type)
    _create_synthetic_reference_npz(npz_path, ref_positions, particle_type)

    eps_t = eps_t_from_pkl_and_reference(
        synthetic_pkl_path=pkl_path,
        reference_npz_path=npz_path,
        transform_kind="identity",
        transform_param="0",
        t_steps=100,
        box_size=1.0,
    )
    assert eps_t.shape == (100,)


def test_eps_t_from_pkl_and_reference_rejects_short_pkl(tmp_path):
    """If predicted_rollout is shorter than 6 + t_steps, raise loudly."""
    from external_validation._rollout_anchors._harness.eps_pkl_consumer import (
        eps_t_from_pkl_and_reference,
    )

    short_positions = np.zeros((6, 4, 2), dtype=np.float32)  # only input window, no predictions
    particle_type = np.zeros(4, dtype=np.int32)

    pkl_path = tmp_path / "rollout_short.pkl"
    npz_path = tmp_path / "ref.npz"
    _create_synthetic_pkl(pkl_path, short_positions, particle_type)
    _create_synthetic_reference_npz(
        npz_path,
        np.zeros((106, 4, 2), dtype=np.float32),
        particle_type,
    )

    with pytest.raises(ValueError, match=r"predicted_rollout has 6 frames"):
        eps_t_from_pkl_and_reference(
            synthetic_pkl_path=pkl_path,
            reference_npz_path=npz_path,
            transform_kind="identity",
            transform_param="0",
            t_steps=1,
            box_size=1.0,
        )
