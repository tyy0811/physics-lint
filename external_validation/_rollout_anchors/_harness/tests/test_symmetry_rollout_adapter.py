"""Unit tests for symmetry_rollout_adapter primitives.

Pure NumPy fixtures; no JAX, no Modal. Each primitive's contract is
asserted via hand-crafted synthetic-but-realistic fixtures per the
"test fixtures hand-crafted, not copied from production" discipline.
"""

from __future__ import annotations

import numpy as np
import pytest


def _box_center_4particle_fixture(box_size: float = 1.0):
    """Four particles at corners of a unit square inscribed in [0, box_size]^2.

    Particles arranged at (box_size/2 +/- 0.25, box_size/2 +/- 0.25). C4
    rotation about box center maps corner_0 -> corner_1 -> corner_2 ->
    corner_3 -> corner_0.
    """
    half = box_size / 2
    d = 0.25
    positions = np.array(
        [
            [half - d, half - d],  # corner 0 (bottom-left)
            [half + d, half - d],  # corner 1 (bottom-right)
            [half + d, half + d],  # corner 2 (top-right)
            [half - d, half + d],  # corner 3 (top-left)
        ],
        dtype=np.float32,
    )
    velocities = np.array(
        [
            [+1.0, -1.0],  # corner 0: velocity pointing toward corner 1
            [+1.0, +1.0],  # corner 1: velocity pointing toward corner 2
            [-1.0, +1.0],  # corner 2: velocity pointing toward corner 3
            [-1.0, -1.0],  # corner 3: velocity pointing toward corner 0
        ],
        dtype=np.float32,
    )
    return positions, velocities, box_size


def test_rotate_about_box_center_pi_2_maps_corner0_to_corner1():
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        rotate_about_box_center,
    )

    positions, velocities, box_size = _box_center_4particle_fixture()
    rotated_pos, rotated_vel = rotate_about_box_center(
        positions=positions, velocities=velocities, theta=np.pi / 2, box_size=box_size
    )

    np.testing.assert_allclose(
        rotated_pos[0],
        positions[1],
        atol=1e-6,
        err_msg="C4 rotation by pi/2 should map corner_0 position to corner_1 position",
    )
    np.testing.assert_allclose(
        rotated_vel[0],
        velocities[1],
        atol=1e-6,
        err_msg="C4 rotation by pi/2 should map corner_0 velocity to corner_1 velocity",
    )


def test_rotate_about_box_center_2pi_is_identity_within_floor():
    """Rotation by 2*pi must return positions to within float32 round-off."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        rotate_about_box_center,
    )

    positions, velocities, box_size = _box_center_4particle_fixture()
    rot_pos, rot_vel = rotate_about_box_center(
        positions=positions, velocities=velocities, theta=2 * np.pi, box_size=box_size
    )
    np.testing.assert_allclose(rot_pos, positions, atol=1e-6)
    np.testing.assert_allclose(rot_vel, velocities, atol=1e-6)


def test_rotate_about_box_center_zero_is_exactly_identity():
    """Rotation by 0 must return positions exactly bit-equal."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        rotate_about_box_center,
    )

    positions, velocities, box_size = _box_center_4particle_fixture()
    rot_pos, rot_vel = rotate_about_box_center(
        positions=positions, velocities=velocities, theta=0.0, box_size=box_size
    )
    np.testing.assert_array_equal(rot_pos, positions)
    np.testing.assert_array_equal(rot_vel, velocities)


def _reflection_symmetric_fixture(box_size: float = 1.0):
    """Four particles arranged y-axis-reflection-symmetric about x = box_size/2."""
    half = box_size / 2
    d = 0.25
    positions = np.array(
        [
            [half - d, half - d],  # left-bottom
            [half + d, half - d],  # right-bottom (mirror of left-bottom)
            [half - d, half + d],  # left-top
            [half + d, half + d],  # right-top (mirror of left-top)
        ],
        dtype=np.float32,
    )
    velocities = np.array(
        [
            [+1.0, +0.5],
            [-1.0, +0.5],
            [+1.0, -0.5],
            [-1.0, -0.5],
        ],
        dtype=np.float32,
    )
    return positions, velocities, box_size


def test_reflect_y_axis_swaps_left_right_pairs():
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        reflect_y_axis,
    )

    positions, velocities, box_size = _reflection_symmetric_fixture()
    reflected_pos, reflected_vel = reflect_y_axis(
        positions=positions, velocities=velocities, box_size=box_size
    )
    np.testing.assert_allclose(reflected_pos[0], positions[1], atol=1e-6)
    np.testing.assert_allclose(reflected_pos[1], positions[0], atol=1e-6)
    np.testing.assert_allclose(reflected_pos[2], positions[3], atol=1e-6)
    np.testing.assert_allclose(reflected_pos[3], positions[2], atol=1e-6)
    np.testing.assert_allclose(reflected_vel[0], np.array([-1.0, 0.5]), atol=1e-6)
    np.testing.assert_allclose(reflected_vel[1], np.array([+1.0, 0.5]), atol=1e-6)


def test_translate_pbc_wraps_at_box_boundary():
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        translate_pbc,
    )

    box_size = 1.0
    positions = np.array(
        [
            [0.1, 0.2],
            [0.9, 0.5],  # near right boundary
            [0.5, 0.95],  # near top boundary
        ],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)
    t = (np.float32(1.0 / 3), np.float32(1.0 / 7))

    translated_pos, translated_vel = translate_pbc(
        positions=positions, velocities=velocities, t=t, box_size=box_size
    )

    np.testing.assert_allclose(translated_pos[0], np.array([0.1 + 1 / 3, 0.2 + 1 / 7]), atol=1e-6)
    np.testing.assert_allclose(
        translated_pos[1], np.array([(0.9 + 1 / 3) - 1.0, 0.5 + 1 / 7]), atol=1e-6
    )
    np.testing.assert_allclose(
        translated_pos[2], np.array([0.5 + 1 / 3, (0.95 + 1 / 7) - 1.0]), atol=1e-6
    )
    np.testing.assert_array_equal(translated_vel, velocities)


def test_eps_pos_rms_matches_hand_computed_4particle():
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        eps_pos_rms,
    )

    a = np.array(
        [
            [0.10, 0.20],
            [0.30, 0.40],
            [0.50, 0.60],
            [0.70, 0.80],
        ],
        dtype=np.float32,
    )
    b = a + np.array(
        [
            [0.01, 0.02],
            [0.00, 0.00],
            [0.05, 0.00],
            [0.00, 0.04],
        ],
        dtype=np.float32,
    )
    expected = np.sqrt((0.0005 + 0.0 + 0.0025 + 0.0016) / 4)
    actual = eps_pos_rms(a=a, b=b)
    assert isinstance(actual, float)
    assert abs(actual - expected) < 1e-7, f"got {actual}, expected {expected}"


def test_eps_pos_rms_zero_when_arrays_equal():
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        eps_pos_rms,
    )

    a = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    assert eps_pos_rms(a=a, b=a.copy()) == 0.0


def test_so2_substrate_skip_trigger_fires_on_periodic_square():
    """The SO(2) trigger should fire for any non-{0, pi/2, pi, 3pi/2} angle."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        so2_substrate_skip_trigger,
    )

    assert so2_substrate_skip_trigger(theta=np.pi / 4, has_periodic_boundaries=True) is True
    assert so2_substrate_skip_trigger(theta=0.5, has_periodic_boundaries=True) is True


def test_so2_substrate_skip_trigger_does_not_fire_on_c4_angles():
    """C4-angles {0, pi/2, pi, 3pi/2} preserve the periodic-square cell."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        so2_substrate_skip_trigger,
    )

    for theta in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        assert so2_substrate_skip_trigger(theta=theta, has_periodic_boundaries=True) is False, (
            f"trigger should not fire on C4 angle {theta}"
        )


def test_so2_substrate_skip_trigger_does_not_fire_on_non_periodic_substrate():
    """If the substrate has no periodic boundaries, SO(2) is structurally measurable."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        so2_substrate_skip_trigger,
    )

    assert so2_substrate_skip_trigger(theta=np.pi / 4, has_periodic_boundaries=False) is False


def test_compute_eps_t_from_pair_t_equal_1():
    """T_steps=1: input is two (1, N, 2) arrays, output is (1,) eps_t array."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        compute_eps_t_from_pair,
    )

    reference = np.array(
        [[[0.10, 0.20], [0.30, 0.40], [0.50, 0.60], [0.70, 0.80]]], dtype=np.float32
    )
    candidate = reference + np.array(
        [[[0.01, 0.02], [0.0, 0.0], [0.05, 0.0], [0.0, 0.04]]], dtype=np.float32
    )

    eps_t = compute_eps_t_from_pair(reference=reference, candidate=candidate)
    assert eps_t.shape == (1,), f"expected shape (1,), got {eps_t.shape}"
    expected = np.sqrt((0.0005 + 0.0 + 0.0025 + 0.0016) / 4)
    np.testing.assert_allclose(eps_t[0], expected, atol=1e-7)


def test_compute_eps_t_from_pair_t_equal_3():
    """T_steps=3: input is two (3, N, 2) arrays, output is (3,) eps_t array."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        compute_eps_t_from_pair,
    )

    reference = np.zeros((3, 4, 2), dtype=np.float32)
    candidate = reference.copy()
    candidate[1, 0] = np.array([0.1, 0.0])
    candidate[2] = np.array(
        [
            [0.1, 0.0],
            [0.0, 0.1],
            [0.0, 0.0],
            [0.1, 0.1],
        ],
        dtype=np.float32,
    )

    eps_t = compute_eps_t_from_pair(reference=reference, candidate=candidate)
    assert eps_t.shape == (3,)
    np.testing.assert_allclose(eps_t[0], 0.0, atol=1e-7)
    np.testing.assert_allclose(eps_t[1], 0.05, atol=1e-6)
    np.testing.assert_allclose(eps_t[2], 0.1, atol=1e-6)


def test_compute_eps_t_from_pair_shape_mismatch_raises():
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        compute_eps_t_from_pair,
    )

    reference = np.zeros((1, 4, 2), dtype=np.float32)
    candidate = np.zeros((1, 5, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_eps_t_from_pair(reference=reference, candidate=candidate)


def test_write_eps_t_npz_then_read_round_trips(tmp_path):
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        read_eps_t_npz,
        write_eps_t_npz,
    )

    eps_t = np.array([4.3e-7, 4.4e-7, 4.5e-7], dtype=np.float32)
    written = write_eps_t_npz(
        out_dir=tmp_path,
        eps_t=eps_t,
        rule_id="PH-SYM-001",
        transform_kind="rotation",
        transform_param="pi_2",
        traj_index=7,
        model_name="segnn",
        dataset_name="tgv2d",
        ckpt_hash="abc123",
        physics_lint_sha_pkl_inference="aaaaaaaaaa",
        physics_lint_sha_npz_conversion="bbbbbbbbbb",
        physics_lint_sha_eps_computation="cccccccccc",
        skip_reason=None,
    )
    assert written.name == "eps_PH-SYM-001_rotation_pi_2_traj07.npz"
    assert written.exists()

    record = read_eps_t_npz(written)
    np.testing.assert_array_equal(record["eps_t"], eps_t)
    assert record["rule_id"] == "PH-SYM-001"
    assert record["transform_kind"] == "rotation"
    assert record["transform_param"] == "pi_2"
    assert record["traj_index"] == 7
    assert record["physics_lint_sha_eps_computation"] == "cccccccccc"
    assert record["skip_reason"] is None


def test_write_eps_t_npz_skip_path_records_skip_reason(tmp_path):
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        read_eps_t_npz,
        write_eps_t_npz,
    )

    written = write_eps_t_npz(
        out_dir=tmp_path,
        eps_t=np.array([np.nan], dtype=np.float32),
        rule_id="PH-SYM-003",
        transform_kind="skip",
        transform_param="so2_continuous",
        traj_index=0,
        model_name="segnn",
        dataset_name="tgv2d",
        ckpt_hash="abc123",
        physics_lint_sha_pkl_inference="aaaaaaaaaa",
        physics_lint_sha_npz_conversion="bbbbbbbbbb",
        physics_lint_sha_eps_computation="cccccccccc",
        skip_reason="PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original",
    )
    record = read_eps_t_npz(written)
    assert record["transform_kind"] == "skip"
    assert record["skip_reason"].startswith("PBC-square breaks SO(2)")
    assert np.isnan(record["eps_t"][0])
