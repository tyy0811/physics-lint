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

# Pre-registration: TGV2D pushforward training horizon. Sourced from
# SEGNN-TGV2D's pushforward.unrolls config dump = [0, 1, 2, 3] (max unroll 3,
# +1 target frame = 4) at LB sha b880a6c84a93792d2499d2a9b8ba3a077ddf44e2 and
# ckpt segnn_tgv2d/best (sha256:c0be98f9...441b). Pinned by D0-15 (rung-3 P0
# invocation) inherited by D0-21 (rung 4b pre-registration). LB enforces
# `sequence_length >= input_seq_length + extra_seq_length` at H5Dataset
# init in `lagrangebench/data/data.py:144` for ALL splits (train/valid/test),
# regardless of `eval.n_rollout_steps`. Re-derive when:
#   - the checkpoint changes (different model/dataset → different unrolls config)
#   - LB pushforward semantics change (LB sha drift; the rollout_image clones
#     `--depth 1` of master, not a sha pin, so a rebuild can shift this too)
EXTRA_SEQ_LENGTH = 4
LB_SUBSEQ_LENGTH = INPUT_SEQ_LENGTH + EXTRA_SEQ_LENGTH


def apply_transform_to_window(
    *,
    input_window: NDArray[np.float32],
    transform_fn: Callable[[NDArray[np.float32], float], NDArray[np.float32]],
    box_size: float,
    t_steps: int,
) -> NDArray[np.float32]:
    """Apply `transform_fn` to each of 6 input frames; append (t_steps - 6)
    placeholder frames (each = transformed frame 5) so the trajectory
    satisfies LB's `T >= input_seq_length + extra_seq_length` requirement
    (asserted in `lagrangebench/data/data.py:144`, enforced for ALL splits
    at H5Dataset init regardless of `eval.n_rollout_steps`).

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
        Total trajectory length. Must be >= LB_SUBSEQ_LENGTH (10).
        Main sweep uses LB_SUBSEQ_LENGTH directly; figure sweep uses 106
        (= INPUT_SEQ_LENGTH + 100 figure rollout steps).

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
    if t_steps < LB_SUBSEQ_LENGTH:
        raise ValueError(
            f"t_steps must be >= LB_SUBSEQ_LENGTH ({LB_SUBSEQ_LENGTH} = "
            f"INPUT_SEQ_LENGTH {INPUT_SEQ_LENGTH} + EXTRA_SEQ_LENGTH {EXTRA_SEQ_LENGTH}); "
            f"got t_steps={t_steps}. LB's H5Dataset asserts "
            f"sequence_length >= subseq_length at config-load time."
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
        Total trajectory length. Must be >= LB_SUBSEQ_LENGTH (10).
        Main sweep uses LB_SUBSEQ_LENGTH directly; figure sweep uses 106
        (= INPUT_SEQ_LENGTH + 100 figure rollout steps).
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
    # at setup_data time (not just `test` even in mode=infer); use
    # LB_SUBSEQ_LENGTH so the assertion passes on the dummies too.
    dummy_traj = apply_transform_to_window(
        input_window=input_windows[0],
        transform_fn=lambda pos, _bs: pos.copy(),
        box_size=box_size,
        t_steps=LB_SUBSEQ_LENGTH,
    )
    for split_name in ("train.h5", "valid.h5"):
        with h5py.File(out_dir / split_name, "w") as f:
            _write_h5_trajectory(f, "00000", dummy_traj, particle_type)

    # ---- metadata.json ----
    metadata = copy.deepcopy(published_metadata)
    metadata["sequence_length_train"] = LB_SUBSEQ_LENGTH
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
