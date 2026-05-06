"""Rung 4b T7 — LB rollout pkl -> eps(t) npz consumer.

Pure-code module. Reads each per-trajectory rollout pkl produced by
`mode=infer eval.infer.out_type=pkl`, applies R^-1 per-step to map
predictions back to the reference frame, computes eps(t) against the
rung-4a reference rollout, and writes one eps_t.npz per trajectory.

Per design §5 (off-by-one frame index resolution: predicted_rollout[6]
is f^1(x_0..x_5), NOT predicted_rollout[0]; rung-4a npzs have the same
shape).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

try:
    # Local / pytest path: external_validation package on sys.path.
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        compute_eps_t_from_pair,
        reflect_y_axis,
        rotate_about_box_center,
        translate_pbc,
    )
except ImportError:
    # Modal-container path: harness modules co-shipped via add_local_file under
    # /opt/physics_lint_harness/ (bare-name import; mirrors rung-4a's
    # lagrangebench_pkl_to_npz.py shipping pattern).
    from symmetry_rollout_adapter import (  # type: ignore[no-redef]
        compute_eps_t_from_pair,
        reflect_y_axis,
        rotate_about_box_center,
        translate_pbc,
    )

# LB default input_seq_length (verified at design time, confirmed against
# published TGV2D metadata at runtime).
INPUT_SEQ_LENGTH = 6


def _parse_rotation_angle(transform_param: str) -> float:
    """Parse path-safe angle strings: '0', 'pi_2', 'pi', '3pi_2'."""
    mapping = {
        "0": 0.0,
        "pi_2": np.pi / 2,
        "pi": np.pi,
        "3pi_2": 3 * np.pi / 2,
    }
    if transform_param not in mapping:
        raise ValueError(
            f"unknown rotation transform_param {transform_param!r}; expected one of {list(mapping)}"
        )
    return mapping[transform_param]


def _parse_translation_vector(transform_param: str, box_size: float) -> tuple[float, float]:
    """Parse 'L_3_L_7' -> (box_size/3, box_size/7)."""
    if transform_param == "L_3_L_7":
        return (box_size / 3.0, box_size / 7.0)
    raise ValueError(f"unknown translation transform_param {transform_param!r}; expected 'L_3_L_7'")


def inverse_transform_per_step(
    *,
    positions: NDArray[np.float32],
    transform_kind: str,
    transform_param: str,
    box_size: float,
) -> NDArray[np.float32]:
    """Apply R^-1 to each step of a (T, N, D) trajectory in the transformed frame.

    For each transform_kind:
      rotation     : rotate by -theta about box center, mod box_size
      reflection   : reflection is its own inverse (involution)
      translation  : subtract t, mod box_size
      identity     : no-op
      skip         : not supported (skip rows do not pass through this function)

    Velocities are not needed by compute_eps_t_from_pair; pass dummy zeros to
    the underlying primitives that require a velocity argument.
    """
    if positions.ndim != 3:
        raise ValueError(f"positions must be (T, N, D); got {positions.shape}")

    t_steps = positions.shape[0]
    velocities_dummy = np.zeros_like(positions[0])

    if transform_kind == "identity":
        return positions.astype(np.float32)

    if transform_kind == "rotation":
        theta = _parse_rotation_angle(transform_param)
        recovered = np.stack(
            [
                rotate_about_box_center(
                    positions=positions[t],
                    velocities=velocities_dummy,
                    theta=-theta,
                    box_size=box_size,
                )[0]
                for t in range(t_steps)
            ],
            axis=0,
        )
        return recovered.astype(np.float32)

    if transform_kind == "reflection":
        recovered = np.stack(
            [
                reflect_y_axis(
                    positions=positions[t],
                    velocities=velocities_dummy,
                    box_size=box_size,
                )[0]
                for t in range(t_steps)
            ],
            axis=0,
        )
        return recovered.astype(np.float32)

    if transform_kind == "translation":
        t_vec = _parse_translation_vector(transform_param, box_size)
        # Inverse translation: -t.
        inv_t = (-t_vec[0], -t_vec[1])
        recovered = np.stack(
            [
                translate_pbc(
                    positions=positions[t],
                    velocities=velocities_dummy,
                    t=inv_t,
                    box_size=box_size,
                )[0]
                for t in range(t_steps)
            ],
            axis=0,
        )
        return recovered.astype(np.float32)

    raise ValueError(f"unknown transform_kind {transform_kind!r}")


def eps_t_from_pkl_and_reference(
    *,
    synthetic_pkl_path: Path,
    reference_npz_path: Path,
    transform_kind: str,
    transform_param: str,
    t_steps: int,
    box_size: float,
) -> NDArray[np.float32]:
    """Compute eps(t) for a single (transform, traj) pair.

    Reads the synthetic-dataset rollout pkl, applies R^-1 to predicted_rollout
    sliced at [6:6+t_steps], reads the reference rung-4a npz positions sliced
    at the same range, and returns eps_t per compute_eps_t_from_pair.

    Per design §5 (off-by-one): both arrays are sliced [6:6+t_steps] before
    comparison; predicted_rollout[6] = f^1(x_0..x_5), NOT predicted_rollout[0].
    """
    synthetic_pkl_path = Path(synthetic_pkl_path)
    reference_npz_path = Path(reference_npz_path)

    with open(synthetic_pkl_path, "rb") as f:
        blob = pickle.load(f)
    if "predicted_rollout" not in blob:
        raise KeyError(
            f"{synthetic_pkl_path}: missing 'predicted_rollout' key; got {sorted(blob.keys())}"
        )
    predicted = np.asarray(blob["predicted_rollout"], dtype=np.float32)

    if predicted.shape[0] < INPUT_SEQ_LENGTH + t_steps:
        raise ValueError(
            f"{synthetic_pkl_path}: predicted_rollout has {predicted.shape[0]} frames; "
            f"need at least {INPUT_SEQ_LENGTH + t_steps} for t_steps={t_steps}"
        )

    candidate_transformed = predicted[INPUT_SEQ_LENGTH : INPUT_SEQ_LENGTH + t_steps]
    candidate = inverse_transform_per_step(
        positions=candidate_transformed,
        transform_kind=transform_kind,
        transform_param=transform_param,
        box_size=box_size,
    )

    with np.load(reference_npz_path, allow_pickle=True) as ref:
        ref_positions = np.asarray(ref["positions"], dtype=np.float32)

    if ref_positions.shape[0] < INPUT_SEQ_LENGTH + t_steps:
        raise ValueError(
            f"{reference_npz_path}: positions has {ref_positions.shape[0]} frames; "
            f"need at least {INPUT_SEQ_LENGTH + t_steps} for t_steps={t_steps}"
        )

    reference = ref_positions[INPUT_SEQ_LENGTH : INPUT_SEQ_LENGTH + t_steps]

    return compute_eps_t_from_pair(reference=reference, candidate=candidate)
