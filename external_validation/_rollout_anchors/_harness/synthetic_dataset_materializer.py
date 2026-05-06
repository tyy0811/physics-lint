"""Rung 4b T7 — synthetic LB-format dataset materializer.

Pure-code module. Reads input windows + transforms + metadata; writes
the (test.h5, train.h5, valid.h5, metadata.json, manifest.json) tuple
that LB's data loader ingests via `dataset.src=<dir>`.

Per design §3 (synthetic H5 spec) and §3.3 (metadata reuse policy).
JAX is NOT imported here — h5py + numpy + json only.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

INPUT_SEQ_LENGTH = 6  # LB default; verified at design time, asserted in materializer

# Pre-registration: SEGNN-TGV2D max pushforward unroll horizon, sourced from
# the config dump `pushforward.unrolls = [0, 1, 2, 3]` at LB sha
# b880a6c84a93792d2499d2a9b8ba3a077ddf44e2 and ckpt segnn_tgv2d/best
# (sha256:c0be98f9...441b). Pinned by D0-15 (rung-3 P0 invocation) inherited
# by D0-21 (rung 4b pre-registration). The LB sha is the captured-at-image-build
# value (rollout_image clones `--depth 1` of master, not a sha pin); a rebuild
# can shift this if upstream LB has moved. Re-derive when:
#   - the checkpoint changes (different model/dataset → different unrolls config)
#   - LB pushforward semantics change
LB_PUSHFORWARD_UNROLLS_LAST = 3

# LB's H5Dataset enforces `sequence_length >= subseq_length` at __init__
# (`lagrangebench/data/data.py:144`) for ALL splits at setup_data time
# (`lagrangebench/runner.py:163-188`). The required subseq_length is
# split-dependent:
#   - train:       input_seq_length + 1 + extra_seq_length        (data.py:131)
#                  where extra_seq_length = pushforward.unrolls[-1]
#                  → for SEGNN-TGV2D: 6 + 1 + 3 = 10 (constant; "+1" is the
#                    target frame, separate from the pushforward unroll count)
#   - valid/test:  input_seq_length + extra_seq_length            (data.py:138)
#                  where extra_seq_length = eval.n_rollout_steps
#                  → main/sanity sweeps (n_rollout_steps=1): 7
#                  → figure sweep (n_rollout_steps=100):     106
# LB_TRAIN_SUBSEQ_LENGTH is the floor for train.h5 across all sweeps. valid/test
# floors scale with n_rollout_steps and are the caller's responsibility to
# satisfy when sizing test.h5 (= the materializer's `t_steps` arg).
LB_TRAIN_SUBSEQ_LENGTH = INPUT_SEQ_LENGTH + 1 + LB_PUSHFORWARD_UNROLLS_LAST


def apply_transform_to_window(
    *,
    input_window: NDArray[np.float32],
    transform_fn: Callable[[NDArray[np.float32], float], NDArray[np.float32]],
    box_size: float,
    t_steps: int,
) -> NDArray[np.float32]:
    """Apply `transform_fn` to each of 6 input frames; append (t_steps - 6)
    placeholder frames (each = transformed frame 5) so the trajectory
    satisfies LB's `H5Dataset.__init__` assertion at
    `lagrangebench/data/data.py:144`. The minimum is split-dependent:
    train requires `input_seq_length + 1 + pushforward.unrolls[-1]`
    (= LB_TRAIN_SUBSEQ_LENGTH), valid/test require
    `input_seq_length + n_rollout_steps`. Materializer enforces the train
    floor at this layer; the caller sizes `t_steps` for the valid/test
    floor based on the sweep's `n_rollout_steps`.

    Per design §3.2 (placeholder = copy of frame 5; distribution-safe per
    features.py:68 — FD-velocity uses only frames[0:6]).

    Parameters
    ----------
    input_window : (6, N, D) fp32
        First 6 frames of an LB rollout's input window (from published test.h5).
    transform_fn : callable (positions, box_size) -> transformed_positions
        Per-frame symmetry transform; matches the signature of
        symmetry_rollout_adapter primitives' position-only flavor.
    box_size : float
        Periodic-square side length; passed to transform_fn.
    t_steps : int
        Total trajectory length. Must be >= LB_TRAIN_SUBSEQ_LENGTH (10) for
        the train-split assert, and >= INPUT_SEQ_LENGTH + n_rollout_steps for
        the valid/test-split assert. Main + sanity sweeps use
        LB_TRAIN_SUBSEQ_LENGTH (10, satisfies both since n_rollout_steps=1);
        figure sweep uses 106 (= INPUT_SEQ_LENGTH + 100 rollout steps).
        Caller is responsible for computing the right value for the sweep.

    Returns
    -------
    (t_steps, N, D) fp32 trajectory; frames [0:6] = transformed input window;
    frames [6:t_steps] = transformed frame 5 placeholder.
    """
    if input_window.shape[0] != INPUT_SEQ_LENGTH:
        raise ValueError(
            f"input_window must have 6 frames (LB input_seq_length); got {input_window.shape[0]}"
        )
    transformed_frames = np.stack(
        [transform_fn(input_window[k], box_size) for k in range(INPUT_SEQ_LENGTH)],
        axis=0,
    ).astype(np.float32)
    if t_steps < LB_TRAIN_SUBSEQ_LENGTH:
        raise ValueError(
            f"t_steps must be >= LB_TRAIN_SUBSEQ_LENGTH "
            f"({LB_TRAIN_SUBSEQ_LENGTH} = INPUT_SEQ_LENGTH {INPUT_SEQ_LENGTH} "
            f"+ 1 target frame + LB_PUSHFORWARD_UNROLLS_LAST "
            f"{LB_PUSHFORWARD_UNROLLS_LAST}); got t_steps={t_steps}. "
            f"LB's H5Dataset asserts sequence_length >= subseq_length at "
            f"config-load time for the train split."
        )
    n_placeholders = t_steps - INPUT_SEQ_LENGTH
    placeholder = transformed_frames[5:6]  # shape (1, N, D)
    placeholders = np.repeat(placeholder, n_placeholders, axis=0)
    return np.concatenate([transformed_frames, placeholders], axis=0).astype(np.float32)


def _write_h5_trajectory(
    h5_file: h5py.File,
    group_name: str,
    position: NDArray[np.float32],
    particle_type: NDArray[np.int32],
) -> None:
    """Write one trajectory group to an H5 file."""
    grp = h5_file.create_group(group_name)
    grp.create_dataset("position", data=position, compression="gzip")
    grp.create_dataset("particle_type", data=particle_type)


def materialize_synthetic_dataset(
    *,
    out_dir: Path,
    input_windows: NDArray[np.float32],
    particle_type: NDArray[np.int32],
    transforms: list[dict[str, Any]],
    published_metadata: dict[str, Any],
    t_steps: int,
    sweep_kind: str,
    stack: str,
    dataset: str,
    ckpt_hash: str,
    physics_lint_sha_eps_computation: str,
) -> Path:
    """Materialize a synthetic LB-format dataset directory.

    Per design §3 (synthetic H5 spec), §3.3 (metadata reuse policy),
    and §3.4 (manifest schema).

    Parameters
    ----------
    out_dir : Path
        Created if missing. Will contain test.h5, train.h5, valid.h5,
        metadata.json, manifest.json.
    input_windows : (n_orig_trajs, 6, N, D) fp32
        Read from published test.h5 (e.g., 20 trajs for TGV2D test split).
    particle_type : (N,) int32
        Copied verbatim from published test.h5 traj 0.
    transforms : list of dicts with keys
        rule_id, transform_kind, transform_param, transform_fn, original_traj_index.
    published_metadata : dict
        Decoded from published TGV2D metadata.json. Reused verbatim for
        normalization stats (silent-mismatch hazard if not reused).
    t_steps : int
        Total trajectory length. Must be >= LB_TRAIN_SUBSEQ_LENGTH (10) for
        the train-split assert, and >= INPUT_SEQ_LENGTH + n_rollout_steps for
        the valid/test-split assert. Main + sanity sweeps use
        LB_TRAIN_SUBSEQ_LENGTH (10, satisfies both since n_rollout_steps=1);
        figure sweep uses 106 (= INPUT_SEQ_LENGTH + 100 rollout steps).
        Caller is responsible for computing the right value for the sweep.
    sweep_kind : str
        "main" or "figure".
    stack, dataset : str
        For manifest provenance.
    ckpt_hash : str
        Must be namespaced "sha256:<hex>".
    physics_lint_sha_eps_computation : str
        10-char prefix; recorded in manifest.

    Returns
    -------
    out_dir (Path), populated.
    """
    if not ckpt_hash.startswith("sha256:"):
        raise ValueError(f"ckpt_hash must be namespaced (e.g., 'sha256:<hex>'); got {ckpt_hash!r}")

    n_orig_trajs = input_windows.shape[0]
    box_size = float(published_metadata["bounds"][0][1] - published_metadata["bounds"][0][0])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate transforms reference valid original_traj_index values.
    for entry in transforms:
        idx = entry["original_traj_index"]
        if not (0 <= idx < n_orig_trajs):
            raise IndexError(
                f"original_traj_index {idx} out of range [0, {n_orig_trajs}); "
                f"input_windows has shape {input_windows.shape}"
            )

    # ---- test.h5 ----
    test_h5_path = out_dir / "test.h5"
    with h5py.File(test_h5_path, "w") as f:
        for synthetic_traj_index, entry in enumerate(transforms):
            window = input_windows[entry["original_traj_index"]]
            traj = apply_transform_to_window(
                input_window=window,
                transform_fn=entry["transform_fn"],
                box_size=box_size,
                t_steps=t_steps,
            )
            _write_h5_trajectory(f, f"{synthetic_traj_index:05d}", traj, particle_type)

    # ---- train.h5 + valid.h5 (single dummy trajectory each; LB requires file exists) ----
    # LB's H5Dataset asserts sequence_length >= subseq_length for ALL splits
    # at setup_data time. Train's required subseq_length is constant
    # (LB_TRAIN_SUBSEQ_LENGTH=10), but valid's scales with eval.n_rollout_steps
    # — for the figure sweep (n_rollout_steps=100), valid.h5 needs >= 106
    # frames. Sizing both dummies at `t_steps` (= test.h5's length) covers
    # all cases uniformly: t_steps satisfies the train floor by virtue of
    # the assert above (`t_steps >= LB_TRAIN_SUBSEQ_LENGTH`), and matches
    # test.h5's length so valid's dynamic floor is satisfied iff test's is.
    dummy_traj = apply_transform_to_window(
        input_window=input_windows[0],
        transform_fn=lambda pos, _bs: pos.copy(),
        box_size=box_size,
        t_steps=t_steps,
    )
    for split_name in ("train.h5", "valid.h5"):
        with h5py.File(out_dir / split_name, "w") as f:
            _write_h5_trajectory(f, "00000", dummy_traj, particle_type)

    # ---- metadata.json ----
    metadata = copy.deepcopy(published_metadata)
    metadata["sequence_length_train"] = t_steps
    metadata["sequence_length_test"] = t_steps
    metadata["num_trajs_train"] = 1
    metadata["num_trajs_test"] = len(transforms)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # ---- manifest.json ----
    manifest = {
        "schema_version": "1.0",
        "stack": stack,
        "dataset": dataset,
        "sweep_kind": sweep_kind,
        "physics_lint_sha_eps_computation": physics_lint_sha_eps_computation,
        "ckpt_hash": ckpt_hash,
        "trajectories": [
            {
                "synthetic_traj_index": i,
                "rule_id": entry["rule_id"],
                "transform_kind": entry["transform_kind"],
                "transform_param": entry["transform_param"],
                "original_traj_index": entry["original_traj_index"],
            }
            for i, entry in enumerate(transforms)
        ],
    }
    # Contiguity assertion (defensive — duplicates the loop's natural order
    # but pins the contract for future readers):
    assert all(t["synthetic_traj_index"] == i for i, t in enumerate(manifest["trajectories"])), (
        "synthetic_traj_index must be contiguous from 0"
    )
    assert len(manifest["trajectories"]) == metadata["num_trajs_test"], (
        f"manifest.trajectories length {len(manifest['trajectories'])} != "
        f"metadata.num_trajs_test {metadata['num_trajs_test']}"
    )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return out_dir


def read_published_input_windows(
    *, h5_path: Path, n_trajs: int
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Read the first 6 frames of trajectories 0..n_trajs-1 from a published LB H5.

    Returns (windows of shape (n_trajs, 6, N, D), particle_type of shape (N,)).
    Particle_type is read from traj 0 (TGV2D has uniform particle_type across
    trajectories; design §3.2).
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        groups = sorted(f.keys())
        if len(groups) < n_trajs:
            raise ValueError(
                f"{h5_path}: requested {n_trajs} trajs but only {len(groups)} available"
            )
        windows = np.stack(
            [f[g]["position"][0:INPUT_SEQ_LENGTH] for g in groups[:n_trajs]],
            axis=0,
        ).astype(np.float32)
        particle_type = np.asarray(f[groups[0]]["particle_type"], dtype=np.int32)
    return windows, particle_type
