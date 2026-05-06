"""Rung 4b equivariance harness primitives.

Pure-NumPy transforms (rotation, reflection, translation) on
particle-rollout state, the substrate-skip trigger for PH-SYM-003
(SO(2) on PBC-square), the per-particle RMS aggregation primitive for
eps computation, and the eps(t) computation orchestrator.

JAX is *not* imported here — the network forward pass lives in the
Modal-side entrypoint (`01-lagrangebench/modal_app.py`), which feeds
this module the pre-computed pair (f1(x_0), f1(R x_0)). This separation
keeps the consumer-side dependency surface clean: rendering, linting,
and test infrastructure only need NumPy.

Per design §3.5: rotation/reflection pivot at box center
(box_size/2, box_size/2); velocities transform with the same R as
positions; PBC `mod box_size` wrap after every transform.

Naming note: the design and writeup use the physics convention `L` for
box side length. Module code uses `box_size` (snake_case) to satisfy
project-wide N803/N806 linting; the substantive contract is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


def rotate_about_box_center(
    *,
    positions: NDArray[np.float32],
    velocities: NDArray[np.float32],
    theta: float,
    box_size: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Rotate positions and velocities by angle theta about the box center.

    Pivot is (box_size/2, box_size/2). Velocities transform with the
    same rotation matrix as positions (canonical equivariance-test
    correctness — see design §3.5 item 2). PBC `mod box_size` wrap after
    rotation handles floating-point excursions outside [0, box_size]^2
    at cell boundaries.

    Returns
    -------
    rotated_positions : (N, 2) fp32, in [0, box_size]^2 after PBC wrap
    rotated_velocities : (N, 2) fp32, no PBC wrap (velocity is a tangent vector)
    """
    cos_t = np.cos(theta).astype(np.float32)
    sin_t = np.sin(theta).astype(np.float32)
    rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

    box_center = np.array([box_size / 2, box_size / 2], dtype=np.float32)

    rel_positions = positions - box_center
    rotated_rel = rel_positions @ rot_matrix.T
    rotated_positions = (rotated_rel + box_center).astype(np.float32)
    rotated_positions = np.mod(rotated_positions, np.float32(box_size))

    rotated_velocities = (velocities @ rot_matrix.T).astype(np.float32)

    return rotated_positions, rotated_velocities


def reflect_y_axis(
    *,
    positions: NDArray[np.float32],
    velocities: NDArray[np.float32],
    box_size: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Reflect positions and velocities across the line x = box_size / 2.

    Operation: pos'_x = box_size - pos_x; pos'_y = pos_y; vel'_x = -vel_x;
    vel'_y = vel_y. PBC `mod box_size` wrap on positions after operation.

    The "y-axis reflection" name follows the standard physics convention:
    reflection across an axis parallel to the y-direction (here, the
    line x = box_size / 2 through box center). The component
    perpendicular to the axis (x-component) flips; the component
    parallel to the axis (y-component) is preserved.
    """
    reflected_positions = positions.copy()
    reflected_positions[:, 0] = np.float32(box_size) - reflected_positions[:, 0]
    reflected_positions = np.mod(reflected_positions, np.float32(box_size))

    reflected_velocities = velocities.copy()
    reflected_velocities[:, 0] = -reflected_velocities[:, 0]

    return reflected_positions, reflected_velocities


def translate_pbc(
    *,
    positions: NDArray[np.float32],
    velocities: NDArray[np.float32],
    t: tuple[float, float],
    box_size: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Translate positions by t = (tx, ty), then PBC `mod box_size` wrap.

    Translation is identity on velocities — the translation matrix is
    the identity on tangent vectors (per design §3.5 item 2).

    For the rung 4b construction-trivial smoke test, t = (box_size/3,
    box_size/7). Per design §3.5, the choice of t doesn't affect
    *whether* PH-SYM-004 passes (translation + PBC commute exactly is
    a substrate property). (box_size/3, box_size/7) is non-grid-
    commensurate to avoid accidental commensurability with structure
    in x_0.
    """
    t_arr = np.asarray(t, dtype=np.float32)
    translated_positions = np.mod(positions + t_arr, np.float32(box_size)).astype(np.float32)
    return translated_positions, velocities.copy()


def eps_pos_rms(*, a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Per-particle RMS aggregation: eps = sqrt(mean_i ||a_i - b_i||^2).

    Per design §3.4 and SCHEMA.md §3.x. `a` and `b` are (N, D) arrays
    of per-particle positions; the output is a scalar eps. Order:
    per-particle squared L2 norm -> mean over N -> sqrt.

    This is the "RMS-across-particles" aggregation; alternatives
    (mean-across-particles, max-across-particles) would change what
    the threshold means and are excluded by the design.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a {a.shape} vs b {b.shape}")
    diff = a - b
    per_particle_sq_norm = np.sum(diff * diff, axis=-1)
    return float(np.sqrt(np.mean(per_particle_sq_norm)))


_C4_ANGLES = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi)


def so2_substrate_skip_trigger(
    *,
    theta: float,
    has_periodic_boundaries: bool,
    angle_tolerance: float = 1e-9,
) -> bool:
    """Return True if (rule, substrate) compatibility makes eps measurement
    structurally invalid for the given rotation angle.

    Per design §3.6 trigger-vs-emission separation: this function
    contains the rule-specific trigger logic. Emission (skip_reason
    population, raw_value=None, level="note") happens downstream in
    the consumer (lint_eps_dir.py).

    Trigger condition: substrate has periodic boundaries AND theta is
    not a non-trivial-symmetry angle of the substrate cell.

    For a periodic-square substrate, the cell-preserving rotations are
    C4 = {0, pi/2, pi, 3pi/2} (and 2pi = 0). Any other angle rotates
    the cell to one that doesn't tile with the original — the rotated
    state is not a valid input to f, and eps computed from it is
    substrate-confounded rather than architectural.
    """
    if not has_periodic_boundaries:
        return False
    theta_norm = float(theta) % (2 * np.pi)
    return all(abs(theta_norm - c4_angle) >= angle_tolerance for c4_angle in _C4_ANGLES)


def compute_eps_t_from_pair(
    *,
    reference: NDArray[np.float32],
    candidate: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute eps(t) = sqrt(mean_i ||reference_t,i - candidate_t,i||^2) for each t.

    Inputs are (T_steps, N_particles, D) arrays. Output is (T_steps,)
    fp32. The Modal-side entrypoint is responsible for producing the
    `reference` (= forward(x_0)) and `candidate` (= R^-1 forward(R x_0))
    arrays; this primitive handles the per-step RMS aggregation only.

    See SCHEMA.md §1.5 and design §3.4 for the artifact-tier shape.
    """
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape} vs candidate {candidate.shape}"
        )
    diff = reference - candidate
    per_particle_sq_norm = np.sum(diff * diff, axis=-1)
    eps_t = np.sqrt(np.mean(per_particle_sq_norm, axis=-1))
    return eps_t.astype(np.float32)


def write_eps_t_npz(
    *,
    out_dir: Path,
    eps_t: NDArray[np.float32],
    rule_id: str,
    transform_kind: Literal["rotation", "reflection", "translation", "identity", "skip"],
    transform_param: str,
    traj_index: int,
    model_name: str,
    dataset_name: str,
    ckpt_hash: str,
    physics_lint_sha_pkl_inference: str,
    physics_lint_sha_npz_conversion: str,
    physics_lint_sha_eps_computation: str,
    skip_reason: str | None,
) -> Path:
    """Persist one eps(t) npz per SCHEMA.md §1.5.

    Filename: eps_{rule_id}_{transform_kind}_{transform_param}_traj{NN}.npz
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"eps_{rule_id}_{transform_kind}_{transform_param}_traj{traj_index:02d}.npz"
    out_path = out_dir / filename

    np.savez(
        out_path,
        eps_t=eps_t,
        rule_id=np.array(rule_id),
        transform_kind=np.array(transform_kind),
        transform_param=np.array(transform_param),
        traj_index=np.array(traj_index, dtype=np.int32),
        model_name=np.array(model_name),
        dataset_name=np.array(dataset_name),
        ckpt_hash=np.array(ckpt_hash),
        physics_lint_sha_pkl_inference=np.array(physics_lint_sha_pkl_inference),
        physics_lint_sha_npz_conversion=np.array(physics_lint_sha_npz_conversion),
        physics_lint_sha_eps_computation=np.array(physics_lint_sha_eps_computation),
        skip_reason=np.array(skip_reason if skip_reason is not None else "", dtype=str),
        skip_reason_present=np.array(skip_reason is not None, dtype=bool),
    )
    return out_path


def read_eps_t_npz(path: Path) -> dict[str, Any]:
    """Read an eps(t) npz back into a dict matching SCHEMA.md §1.5."""
    with np.load(path, allow_pickle=False) as data:
        skip_reason_present = bool(data["skip_reason_present"])
        record: dict[str, Any] = {
            "eps_t": data["eps_t"].astype(np.float32),
            "rule_id": str(data["rule_id"]),
            "transform_kind": str(data["transform_kind"]),
            "transform_param": str(data["transform_param"]),
            "traj_index": int(data["traj_index"]),
            "model_name": str(data["model_name"]),
            "dataset_name": str(data["dataset_name"]),
            "ckpt_hash": str(data["ckpt_hash"]),
            "physics_lint_sha_pkl_inference": str(data["physics_lint_sha_pkl_inference"]),
            "physics_lint_sha_npz_conversion": str(data["physics_lint_sha_npz_conversion"]),
            "physics_lint_sha_eps_computation": str(data["physics_lint_sha_eps_computation"]),
            "skip_reason": str(data["skip_reason"]) if skip_reason_present else None,
        }
    return record
