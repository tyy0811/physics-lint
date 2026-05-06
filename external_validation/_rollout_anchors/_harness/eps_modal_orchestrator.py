"""Rung 4b T7 — Modal entrypoint orchestrator helpers.

Pure-code module (no Modal, no LB import). Builds the transform list
that the synthetic_dataset_materializer ingests, and provides the
sanity-probe verdict logic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    # Local / pytest path: external_validation package on sys.path.
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        reflect_y_axis,
        rotate_about_box_center,
        translate_pbc,
    )
except ImportError:
    # Modal-container path: harness modules co-shipped via add_local_file under
    # /opt/physics_lint_harness/ (bare-name import; mirrors rung-4a's
    # lagrangebench_pkl_to_npz.py shipping pattern).
    from symmetry_rollout_adapter import (  # type: ignore[no-redef]
        reflect_y_axis,
        rotate_about_box_center,
        translate_pbc,
    )

# Per-stack number of original trajectories from rung 4a (TGV2D test split).
N_ORIG_TRAJS = 20

# Figure subset traj indices (design §3.4: span 0..19 with gaps).
FIGURE_SUBSET_TRAJ_INDICES = (0, 7, 14)


def _rotation_fn(theta: float) -> Callable[[NDArray[np.float32], float], NDArray[np.float32]]:
    """Position-only rotation: drops velocities (not needed by materializer)."""

    def fn(positions: NDArray[np.float32], box_size: float) -> NDArray[np.float32]:
        velocities_dummy = np.zeros_like(positions)
        rotated_positions, _ = rotate_about_box_center(
            positions=positions,
            velocities=velocities_dummy,
            theta=theta,
            box_size=box_size,
        )
        return rotated_positions

    return fn


def _reflection_fn() -> Callable[[NDArray[np.float32], float], NDArray[np.float32]]:
    def fn(positions: NDArray[np.float32], box_size: float) -> NDArray[np.float32]:
        velocities_dummy = np.zeros_like(positions)
        reflected_positions, _ = reflect_y_axis(
            positions=positions,
            velocities=velocities_dummy,
            box_size=box_size,
        )
        return reflected_positions

    return fn


def _translation_fn() -> Callable[[NDArray[np.float32], float], NDArray[np.float32]]:
    def fn(positions: NDArray[np.float32], box_size: float) -> NDArray[np.float32]:
        velocities_dummy = np.zeros_like(positions)
        translated_positions, _ = translate_pbc(
            positions=positions,
            velocities=velocities_dummy,
            t=(box_size / 3.0, box_size / 7.0),
            box_size=box_size,
        )
        return translated_positions

    return fn


def _identity_fn() -> Callable[[NDArray[np.float32], float], NDArray[np.float32]]:
    def fn(positions: NDArray[np.float32], _box_size: float) -> NDArray[np.float32]:
        return positions.copy()

    return fn


def build_main_sweep_transforms(*, n_trajs: int = N_ORIG_TRAJS) -> list[dict[str, Any]]:
    """Build the 120-entry transform list for the main sweep.

    Per design §3.2: 4 PH-SYM-001 angles (incl. theta=0 identity smoke test)
    + 1 PH-SYM-002 reflection + 1 PH-SYM-004 translation, each x n_trajs.
    PH-SYM-003 SO(2) SKIP rows are NOT in this list — they are written
    independently without invoking LB.
    """
    transforms: list[dict[str, Any]] = []
    angle_specs = (
        ("0", "identity", _identity_fn()),
        ("pi_2", "rotation", _rotation_fn(np.pi / 2)),
        ("pi", "rotation", _rotation_fn(np.pi)),
        ("3pi_2", "rotation", _rotation_fn(3 * np.pi / 2)),
    )
    for traj in range(n_trajs):
        for transform_param, transform_kind, fn in angle_specs:
            transforms.append(
                {
                    "rule_id": "PH-SYM-001",
                    "transform_kind": transform_kind,
                    "transform_param": transform_param,
                    "transform_fn": fn,
                    "original_traj_index": traj,
                }
            )
    for traj in range(n_trajs):
        transforms.append(
            {
                "rule_id": "PH-SYM-002",
                "transform_kind": "reflection",
                "transform_param": "y_axis",
                "transform_fn": _reflection_fn(),
                "original_traj_index": traj,
            }
        )
    for traj in range(n_trajs):
        transforms.append(
            {
                "rule_id": "PH-SYM-004",
                "transform_kind": "translation",
                "transform_param": "L_3_L_7",
                "transform_fn": _translation_fn(),
                "original_traj_index": traj,
            }
        )
    return transforms


def build_figure_sweep_transforms() -> list[dict[str, Any]]:
    """Build the 3-entry figure-subset transform list.

    Per design §3.2: 1 angle (pi/2) x 3 trajs from FIGURE_SUBSET_TRAJ_INDICES.
    """
    transforms: list[dict[str, Any]] = []
    for traj in FIGURE_SUBSET_TRAJ_INDICES:
        transforms.append(
            {
                "rule_id": "PH-SYM-001",
                "transform_kind": "rotation",
                "transform_param": "pi_2",
                "transform_fn": _rotation_fn(np.pi / 2),
                "original_traj_index": traj,
            }
        )
    return transforms


SANITY_PROBE_PASS_THRESHOLD = 1e-5
SANITY_PROBE_CONCERNING_THRESHOLD = 1e-3


def interpret_sanity_probe_verdict(*, eps: float) -> dict[str, Any]:
    """Per design §6: classify the sanity-probe eps into one of three bands.

    Single gate: eps <= 1e-5 -> PASS.
    Diagnostic bands within the abort message:
      eps in (1e-5, 1e-3]   -> "concerning, possible borderline FP variation or partial-bug"
      eps > 1e-3            -> "clear bug, likely [coordinate-space / frame-index /
                                normalization / manifest-mapping]"
    """
    eps_str = f"{eps:.1e}"
    if eps <= SANITY_PROBE_PASS_THRESHOLD:
        return {
            "status": "PASS",
            "abort": False,
            "message": f"sanity probe PASS: eps={eps_str} <= 1.0e-05 (float32 floor band)",
            "eps": eps,
        }
    if eps <= SANITY_PROBE_CONCERNING_THRESHOLD:
        return {
            "status": "ABORT",
            "abort": True,
            "message": (
                f"sanity probe ABORT: eps={eps_str} in (1e-5, 1e-3] — concerning, "
                "possible borderline FP variation or partial-bug. Investigate before proceeding."
            ),
            "eps": eps,
        }
    return {
        "status": "ABORT",
        "abort": True,
        "message": (
            f"sanity probe ABORT: eps={eps_str} > 1e-3 — clear bug, likely one of "
            "[coordinate-space mismatch / off-by-one frame index / normalization stat "
            "divergence / manifest-mapping error]. Do NOT proceed to full sweep."
        ),
        "eps": eps,
    }
