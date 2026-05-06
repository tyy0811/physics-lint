# Rung 4b T7 — Modal entrypoints (LB-integration shape c.1) implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the rung 4b T7 Modal entrypoints (`lagrangebench_eps_p0_segnn_tgv2d` + `_p1_gns_tgv2d`) that materialize a synthetic LB-format dataset of symmetry-transformed input windows, run LB inference via subprocess, post-process pkl outputs into eps_t npzs, and persist artifacts to Modal Volume — per the (c.1) shape pinned in the T7 design doc.

**Architecture:** Two thin Modal entrypoints (one per stack) wrap a materialization step (`_harness/synthetic_dataset_materializer.py`), an LB subprocess invocation (mirroring rung 4a's `lagrangebench_rollout_p0_segnn_tgv2d` shape), and a post-processing step (`_harness/eps_pkl_consumer.py`). The materializer and consumer are pure-code modules with paired pytest tests; the Modal entrypoint is a glue layer that orchestrates them. PH-SYM-003 SKIP rows are written upfront without invoking LB. A pre-execution sanity probe (1 traj × π/2 × SEGNN) gates the full sweep.

**Tech Stack:** Python 3.11+, NumPy, h5py (LB image already includes it), JSON, Modal CLI, pytest, ruff (lint), codespell. Pre-commit hooks via `.venv/bin/pre-commit`.

**Predecessor:** T7 design doc `methodology/docs/2026-05-06-rung-4b-t7-modal-entrypoints-design.md` (commit `8576ed1`). **Read it first** — it carries the load-bearing rationale for every decision in this plan: synthetic H5 schema (§3), reuse-vs-synthesize metadata policy (§3.3), manifest schema (§3.4), off-by-one frame index resolution (§5), sanity-probe gate (§6), wall-time framing (§7), and silent-mismatch hazards (§8).

**Branch:** All work on `feature/rung-4b-equivariance` (PR #7 is in flight; T7 commits stack on top).

**Working dir convention:** All paths absolute under `/Users/zenith/Desktop/physics-lint/`. Tests run via `cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest <args>`. Commits run with the same PATH override so pre-commit hooks resolve.

**Reuse from rung 4b consumer-side (frozen, do not modify):**
- `_harness/symmetry_rollout_adapter.py`: `rotate_about_box_center`, `reflect_y_axis`, `translate_pbc`, `compute_eps_t_from_pair`, `write_eps_t_npz`, `so2_substrate_skip_trigger` — all called by the new code.
- `_harness/SCHEMA.md` §1.5 (eps_t.npz contract) — synthetic-dataset materializer writes npzs that satisfy this contract.

---

## Task 0: gitignore for outputs/synthetic/

**Why first:** The Modal entrypoint will materialize synthetic datasets to a local mirror at `outputs/synthetic/...` during local development; future commits must not bring in the H5 files.

**Files:**
- Modify: `external_validation/_rollout_anchors/.gitignore`
- Create: `external_validation/_rollout_anchors/01-lagrangebench/outputs/synthetic/.gitkeep`

### T0.1: Append synthetic gitignore patterns

- [ ] **Step 1: Append three lines to `external_validation/_rollout_anchors/.gitignore`**

```
01-lagrangebench/outputs/synthetic/**/*.h5
01-lagrangebench/outputs/synthetic/**/*.json
!01-lagrangebench/outputs/synthetic/.gitkeep
```

- [ ] **Step 2: Create the .gitkeep**

```bash
mkdir -p /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/synthetic && touch /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/synthetic/.gitkeep
```

- [ ] **Step 3: Verify gitignore matches**

```bash
cd /Users/zenith/Desktop/physics-lint && touch external_validation/_rollout_anchors/01-lagrangebench/outputs/synthetic/test.h5 && touch external_validation/_rollout_anchors/01-lagrangebench/outputs/synthetic/test.json && git status --short external_validation/_rollout_anchors/01-lagrangebench/outputs/synthetic/ && rm external_validation/_rollout_anchors/01-lagrangebench/outputs/synthetic/test.h5 external_validation/_rollout_anchors/01-lagrangebench/outputs/synthetic/test.json
```

Expected: only `?? external_validation/_rollout_anchors/01-lagrangebench/outputs/synthetic/.gitkeep` (test.h5 + test.json hidden).

- [ ] **Step 4: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/.gitignore external_validation/_rollout_anchors/01-lagrangebench/outputs/synthetic/.gitkeep && git commit -m "01-lagrangebench/outputs/synthetic: gitignore Modal-Volume-mirrored synthetic H5 datasets (rung 4b T7)"
```

---

## Task 1: `synthetic_dataset_materializer.py` — write synthetic H5 + metadata + manifest

**Why this exists:** Materializes the (test.h5, train.h5, valid.h5, metadata.json, manifest.json) tuple per design §3. Pure-code module with paired pytest tests. Modal-side caller passes input windows + transforms + metadata; module returns the populated dataset directory ready for LB ingestion.

**Files:**
- Create: `external_validation/_rollout_anchors/_harness/synthetic_dataset_materializer.py`
- Create: `external_validation/_rollout_anchors/_harness/tests/test_synthetic_dataset_materializer.py`

### T1.1: Failing test for `apply_transform_to_window`

The materializer's first responsibility is "given a 6-frame input window and a transform spec, produce a transformed 7-frame trajectory (6 transformed input frames + 1 placeholder)." This primitive is testable in isolation.

- [ ] **Step 1: Create test file with first failing test**

Create `external_validation/_rollout_anchors/_harness/tests/test_synthetic_dataset_materializer.py`:

```python
"""Unit tests for synthetic_dataset_materializer (rung 4b T7)."""

from __future__ import annotations

from pathlib import Path

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
```

- [ ] **Step 2: Run test to confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_synthetic_dataset_materializer.py -v
```

Expected: 3 FAILED with `ModuleNotFoundError: No module named 'external_validation._rollout_anchors._harness.synthetic_dataset_materializer'`.

### T1.2: Implement `apply_transform_to_window`

- [ ] **Step 1: Create synthetic_dataset_materializer.py**

Create `external_validation/_rollout_anchors/_harness/synthetic_dataset_materializer.py`:

```python
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


def apply_transform_to_window(
    *,
    input_window: NDArray[np.float32],
    transform_fn: Callable[[NDArray[np.float32], float], NDArray[np.float32]],
    box_size: float,
    t_steps: int,
) -> NDArray[np.float32]:
    """Apply `transform_fn` to each of 6 input frames; append (t_steps - 6)
    placeholder frames (each = transformed frame 5) so the trajectory
    satisfies LB's `T >= input_seq_length + n_rollout_steps` requirement.

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
        Total trajectory length (7 for main sweep, 106 for figure sweep).

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
    n_placeholders = t_steps - INPUT_SEQ_LENGTH
    if n_placeholders < 1:
        raise ValueError(
            f"t_steps must be >= input_seq_length + 1; got t_steps={t_steps}"
        )
    placeholder = transformed_frames[5:6]  # shape (1, N, D)
    placeholders = np.repeat(placeholder, n_placeholders, axis=0)
    return np.concatenate([transformed_frames, placeholders], axis=0).astype(np.float32)
```

- [ ] **Step 2: Run tests, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_synthetic_dataset_materializer.py -v
```

Expected: 3 PASSED.

### T1.3: Failing test for `materialize_synthetic_dataset`

The orchestrator function: takes a list of (rule_id, transform_kind, transform_param, transform_fn, original_traj_index) entries plus input windows + metadata + ckpt_hash + provenance shas, writes the full dataset directory.

- [ ] **Step 1: Append test**

Append to `test_synthetic_dataset_materializer.py`:

```python


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
```

- [ ] **Step 2: Run tests, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_synthetic_dataset_materializer.py -v
```

Expected: 3 PASS (from T1.2), 5 FAIL (`materialize_synthetic_dataset` not implemented).

### T1.4: Implement `materialize_synthetic_dataset`

- [ ] **Step 1: Append `materialize_synthetic_dataset` to synthetic_dataset_materializer.py**

```python


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
        Total trajectory length (7 for main sweep, 106 for figure sweep).
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
        raise ValueError(
            f"ckpt_hash must be namespaced (e.g., 'sha256:<hex>'); got {ckpt_hash!r}"
        )

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
            _write_h5_trajectory(
                f, f"{synthetic_traj_index:05d}", traj, particle_type
            )

    # ---- train.h5 + valid.h5 (single dummy trajectory each; LB requires file exists) ----
    dummy_traj = apply_transform_to_window(
        input_window=input_windows[0],
        transform_fn=lambda pos, _bs: pos.copy(),
        box_size=box_size,
        t_steps=7,  # always 7 for dummies; train/valid never used in mode=infer
    )
    for split_name in ("train.h5", "valid.h5"):
        with h5py.File(out_dir / split_name, "w") as f:
            _write_h5_trajectory(f, "00000", dummy_traj, particle_type)

    # ---- metadata.json ----
    metadata = copy.deepcopy(published_metadata)
    metadata["sequence_length_train"] = 7
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
    assert all(
        t["synthetic_traj_index"] == i for i, t in enumerate(manifest["trajectories"])
    ), "synthetic_traj_index must be contiguous from 0"
    assert len(manifest["trajectories"]) == metadata["num_trajs_test"], (
        f"manifest.trajectories length {len(manifest['trajectories'])} != "
        f"metadata.num_trajs_test {metadata['num_trajs_test']}"
    )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return out_dir
```

- [ ] **Step 2: Run all tests, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_synthetic_dataset_materializer.py -v
```

Expected: 8 PASSED (3 from T1.2 + 5 from T1.3).

### T1.5: Failing test for `read_published_input_windows`

The materializer also needs to read input windows from the published test.h5 — a thin wrapper around h5py with shape validation.

- [ ] **Step 1: Append test**

```python


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
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_synthetic_dataset_materializer.py -k "read_published" -v
```

Expected: 2 FAILED.

### T1.6: Implement `read_published_input_windows`

- [ ] **Step 1: Append to synthetic_dataset_materializer.py**

```python


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
```

- [ ] **Step 2: Run all tests, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_synthetic_dataset_materializer.py -v
```

Expected: 10 PASSED.

### T1.7: Commit

- [ ] **Step 1: Commit T1**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/synthetic_dataset_materializer.py external_validation/_rollout_anchors/_harness/tests/test_synthetic_dataset_materializer.py && git commit -m "_harness/synthetic_dataset_materializer: T7 dataset writer (test.h5 + train.h5 + valid.h5 + metadata.json + manifest.json)"
```

---

## Task 2: `eps_pkl_consumer.py` — read LB rollout pkls and compute ε

**Why this exists:** Bridges LB's pkl output → `compute_eps_t_from_pair` → `write_eps_t_npz`. Lives in `_harness/` because it's pure code (no Modal); paired with pytest tests using synthetic pkl fixtures.

**Files:**
- Create: `external_validation/_rollout_anchors/_harness/eps_pkl_consumer.py`
- Create: `external_validation/_rollout_anchors/_harness/tests/test_eps_pkl_consumer.py`

### T2.1: Failing test for `inverse_transform_per_step`

The transformed pkl output is in the rotated/reflected/translated frame; we need to apply R⁻¹ per-step to get back to the reference frame for ε comparison.

- [ ] **Step 1: Create test file**

Create `external_validation/_rollout_anchors/_harness/tests/test_eps_pkl_consumer.py`:

```python
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
                positions=original[t], velocities=velocities_dummy, theta=np.pi / 2, box_size=box_size
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
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_eps_pkl_consumer.py -v
```

Expected: 5 FAILED with `ModuleNotFoundError`.

### T2.2: Implement `inverse_transform_per_step`

- [ ] **Step 1: Create eps_pkl_consumer.py**

Create `external_validation/_rollout_anchors/_harness/eps_pkl_consumer.py`:

```python
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
from typing import Any

import numpy as np
from numpy.typing import NDArray

from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
    INPUT_SEQ_LENGTH := None,  # placeholder; will be replaced below
)
```

Wait — the import needs reworking. Let me define `INPUT_SEQ_LENGTH` directly here (it's a constant matching LB's default input_seq_length=6, asserted at runtime against the dataset's metadata).

Replace the file contents with:

```python
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
from typing import Any

import numpy as np
from numpy.typing import NDArray

from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
    compute_eps_t_from_pair,
    reflect_y_axis,
    rotate_about_box_center,
    translate_pbc,
    write_eps_t_npz,
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
    raise ValueError(
        f"unknown translation transform_param {transform_param!r}; expected 'L_3_L_7'"
    )


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
```

- [ ] **Step 2: Run tests, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_eps_pkl_consumer.py -v
```

Expected: 5 PASSED.

### T2.3: Failing test for `eps_t_from_pkl_and_reference`

The end-to-end orchestrator: read a synthetic pkl + a reference npz, slice both at index `[6:6+T_steps]`, apply R⁻¹ to the synthetic, return `eps_t`.

- [ ] **Step 1: Append test**

Append to `test_eps_pkl_consumer.py`:

```python


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
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_eps_pkl_consumer.py -k "from_pkl_and_reference" -v
```

Expected: 4 FAILED.

### T2.4: Implement `eps_t_from_pkl_and_reference`

- [ ] **Step 1: Append to eps_pkl_consumer.py**

```python


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
```

- [ ] **Step 2: Run all tests, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_eps_pkl_consumer.py -v
```

Expected: 9 PASSED.

### T2.5: Commit

- [ ] **Step 1: Commit T2**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/eps_pkl_consumer.py external_validation/_rollout_anchors/_harness/tests/test_eps_pkl_consumer.py && git commit -m "_harness/eps_pkl_consumer: T7 LB rollout pkl -> eps(t) npz (off-by-one slice [6:6+t_steps])"
```

---

## Task 3: Modal entrypoint orchestrator helper

**Why this exists:** Both Modal entrypoints (SEGNN + GNS) share most of their orchestration logic. Factor the shared logic into a helper that can be unit-tested without Modal/LB.

**Files:**
- Create: `external_validation/_rollout_anchors/_harness/eps_modal_orchestrator.py`
- Create: `external_validation/_rollout_anchors/_harness/tests/test_eps_modal_orchestrator.py`

### T3.1: Failing test for `build_main_sweep_transforms`

The Modal entrypoint needs to build the list of (rule_id, transform_kind, transform_param, transform_fn, original_traj_index) entries. Pure-data construction; testable.

- [ ] **Step 1: Create test file**

Create `external_validation/_rollout_anchors/_harness/tests/test_eps_modal_orchestrator.py`:

```python
"""Unit tests for eps_modal_orchestrator (rung 4b T7)."""

from __future__ import annotations

import pytest


def test_build_main_sweep_transforms_yields_120_entries_for_n_trajs_20():
    """Main sweep: 4 PH-SYM-001 angles + 1 PH-SYM-002 reflection +
    1 PH-SYM-004 translation = 6 transforms x 20 trajs = 120 entries.
    PH-SYM-003 SKIP not in this list (handled separately)."""
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        build_main_sweep_transforms,
    )

    transforms = build_main_sweep_transforms(n_trajs=20)
    assert len(transforms) == 120, f"expected 120, got {len(transforms)}"

    rule_counts: dict[str, int] = {}
    for entry in transforms:
        rule_counts[entry["rule_id"]] = rule_counts.get(entry["rule_id"], 0) + 1
    assert rule_counts == {"PH-SYM-001": 80, "PH-SYM-002": 20, "PH-SYM-004": 20}

    # No PH-SYM-003 in main sweep (SKIP shortcut).
    assert "PH-SYM-003" not in rule_counts


def test_build_main_sweep_transforms_ph_sym_001_has_4_angles():
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        build_main_sweep_transforms,
    )

    transforms = build_main_sweep_transforms(n_trajs=20)
    angles = sorted(
        {e["transform_param"] for e in transforms if e["rule_id"] == "PH-SYM-001"}
    )
    assert angles == ["0", "3pi_2", "pi", "pi_2"]

    # Identity (theta=0) has transform_kind=identity; others have rotation.
    identity_kinds = {
        e["transform_kind"]
        for e in transforms
        if e["rule_id"] == "PH-SYM-001" and e["transform_param"] == "0"
    }
    assert identity_kinds == {"identity"}

    rotation_params = {
        e["transform_param"]
        for e in transforms
        if e["rule_id"] == "PH-SYM-001" and e["transform_kind"] == "rotation"
    }
    assert rotation_params == {"pi_2", "pi", "3pi_2"}


def test_build_main_sweep_transforms_each_rule_covers_all_traj_indices():
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        build_main_sweep_transforms,
    )

    transforms = build_main_sweep_transforms(n_trajs=20)
    # For each (rule, transform_param) pair, original_traj_index covers 0..19.
    by_pair: dict[tuple, list[int]] = {}
    for e in transforms:
        key = (e["rule_id"], e["transform_param"])
        by_pair.setdefault(key, []).append(e["original_traj_index"])
    for key, idxs in by_pair.items():
        assert sorted(idxs) == list(range(20)), f"{key} has {sorted(idxs)}"


def test_build_main_sweep_transforms_transform_fn_is_callable():
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        build_main_sweep_transforms,
    )

    transforms = build_main_sweep_transforms(n_trajs=20)
    for e in transforms:
        assert callable(e["transform_fn"]), f"{e['rule_id']}/{e['transform_param']}"


def test_build_figure_sweep_transforms_yields_3_entries():
    """Figure sweep: 1 angle (pi/2) x 3 trajs (0, 7, 14) = 3 entries."""
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        build_figure_sweep_transforms,
    )

    transforms = build_figure_sweep_transforms()
    assert len(transforms) == 3
    assert all(e["rule_id"] == "PH-SYM-001" for e in transforms)
    assert all(e["transform_kind"] == "rotation" for e in transforms)
    assert all(e["transform_param"] == "pi_2" for e in transforms)
    assert sorted(e["original_traj_index"] for e in transforms) == [0, 7, 14]
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_eps_modal_orchestrator.py -v
```

Expected: 5 FAILED.

### T3.2: Implement `build_main_sweep_transforms` and `build_figure_sweep_transforms`

- [ ] **Step 1: Create eps_modal_orchestrator.py**

Create `external_validation/_rollout_anchors/_harness/eps_modal_orchestrator.py`:

```python
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

from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
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
```

- [ ] **Step 2: Run tests, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_eps_modal_orchestrator.py -v
```

Expected: 5 PASSED.

### T3.3: Failing test for `interpret_sanity_probe_verdict`

Per design §6: ε ≤ 1e-5 → PASS; (1e-5, 1e-3] → "concerning"; > 1e-3 → "clear bug". Pure-function logic.

- [ ] **Step 1: Append test**

```python


def test_interpret_sanity_probe_verdict_pass_at_floor():
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        interpret_sanity_probe_verdict,
    )

    verdict = interpret_sanity_probe_verdict(eps=1e-7)
    assert verdict["status"] == "PASS"
    assert verdict["abort"] is False
    assert "1.0e-07" in verdict["message"]


def test_interpret_sanity_probe_verdict_pass_at_threshold_boundary():
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        interpret_sanity_probe_verdict,
    )

    # Exactly at 1e-5 should still pass (gate is <=).
    verdict = interpret_sanity_probe_verdict(eps=1e-5)
    assert verdict["status"] == "PASS"
    assert verdict["abort"] is False


def test_interpret_sanity_probe_verdict_concerning_band():
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        interpret_sanity_probe_verdict,
    )

    verdict = interpret_sanity_probe_verdict(eps=1e-4)
    assert verdict["status"] == "ABORT"
    assert verdict["abort"] is True
    assert "concerning" in verdict["message"].lower()
    # Diagnostic mentions borderline FP and partial-bug per design §6.
    assert "borderline" in verdict["message"].lower() or "partial" in verdict["message"].lower()


def test_interpret_sanity_probe_verdict_clear_bug_band():
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        interpret_sanity_probe_verdict,
    )

    verdict = interpret_sanity_probe_verdict(eps=1e-2)
    assert verdict["status"] == "ABORT"
    assert verdict["abort"] is True
    assert "clear bug" in verdict["message"].lower()
    # Diagnostic lists the four candidate causes per design §6.
    msg = verdict["message"].lower()
    assert "coordinate" in msg
    assert "frame" in msg
    assert "normalization" in msg
    assert "manifest" in msg
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_eps_modal_orchestrator.py -k "interpret_sanity" -v
```

Expected: 4 FAILED.

### T3.4: Implement `interpret_sanity_probe_verdict`

- [ ] **Step 1: Append to eps_modal_orchestrator.py**

```python


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
```

- [ ] **Step 2: Run all tests, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_eps_modal_orchestrator.py -v
```

Expected: 9 PASSED.

### T3.5: Commit

- [ ] **Step 1: Commit T3**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/eps_modal_orchestrator.py external_validation/_rollout_anchors/_harness/tests/test_eps_modal_orchestrator.py && git commit -m "_harness/eps_modal_orchestrator: T7 transform list builders + sanity-probe verdict logic"
```

---

## Task 4: Modal entrypoint — SEGNN

**Why this exists:** The Modal-side glue layer that ties together input-window reading, materialization, LB subprocess invocation, and post-processing. Lives in `01-lagrangebench/modal_app.py`. Not directly testable without Modal/LB; correctness gated by the embedded sanity probe at runtime.

**Files:**
- Modify: `external_validation/_rollout_anchors/01-lagrangebench/modal_app.py`

### T4.1: Find insertion point

- [ ] **Step 1: Locate the rung-4a entrypoint pattern**

```bash
grep -n "^@app.function\|^def lagrangebench_rollout_p\|^@app.local_entrypoint\|^def rollout_p" /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/modal_app.py | head -20
```

Expected output includes lines like:
```
624:@app.function(...
630:def lagrangebench_rollout_p0_segnn_tgv2d(git_sha: str, full_git_sha: str) -> dict:
... (later) :@app.local_entrypoint()
... :def rollout_p0_segnn_tgv2d() -> None:
```

Note the line numbers; the new `lagrangebench_eps_p0_segnn_tgv2d` function should land after the existing rung-4a entrypoints but before any `@app.local_entrypoint()` blocks.

### T4.2: Append the SEGNN ε entrypoint

- [ ] **Step 1: Append the new `@app.function` to modal_app.py**

Append this block to `01-lagrangebench/modal_app.py` AFTER the existing `lagrangebench_rollout_p1_gns_tgv2d` definition and BEFORE the `@app.local_entrypoint()` blocks. (Use the line numbers from T4.1's grep output to find the boundary.)

```python
@app.function(
    image=lagrangebench_image,
    gpu=ROLLOUT_GENERATION_GPU_CLASS,  # A10G — matched to rung 4a per D0-21 item 10
    timeout=60 * 30,
    volumes={"/rollouts": rollout_volume},
)
def lagrangebench_eps_p0_segnn_tgv2d(
    git_sha: str,
    full_git_sha: str,
    rung_4a_rollout_subdir: str,
) -> dict:
    """Rung 4b T7: SEGNN-TGV2D equivariance eps(t) computation.

    Pipeline (per `methodology/docs/2026-05-06-rung-4b-t7-modal-entrypoints-design.md`):
      1. Read 6-frame input windows for traj 0..19 from the published TGV2D test.h5 on Volume.
      2. Write 20 PH-SYM-003 SKIP eps_t npzs (no LB invocation).
      3. Run pre-execution sanity probe: 1 traj * pi/2 rotation. Abort if eps > 1e-5.
      4. Materialize main-sweep synthetic dataset (120 trajs * T_steps=7).
      5. Materialize figure-sweep synthetic dataset (3 trajs * T_steps=106).
      6. Run LB main sweep via subprocess (mode=infer eval.n_rollout_steps=1).
      7. Run LB figure sweep via subprocess (mode=infer eval.n_rollout_steps=100).
      8. Post-process each rollout pkl into eps_t npz; figure overwrites the 3 corresponding main npzs.

    Parameters
    ----------
    git_sha : str
        Short (10-char) sha of the physics-lint commit at execution time.
    full_git_sha : str
        Full 40-char sha (used in eps_computation provenance and synthetic-dataset
        directory names).
    rung_4a_rollout_subdir : str
        Path under /rollouts to rung-4a's reference rollouts; e.g., "segnn_tgv2d_post_d03df3e".
        Contains particle_rollout_traj{NN}.npz files.

    Returns
    -------
    Dict with keys: npz_count, sanity_probe_eps, elapsed_s, git_sha_eps,
    eps_out_dir, synthetic_dirs.
    """
    import json
    import subprocess
    import time
    from pathlib import Path

    import numpy as np

    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        build_figure_sweep_transforms,
        build_main_sweep_transforms,
        interpret_sanity_probe_verdict,
    )
    from external_validation._rollout_anchors._harness.eps_pkl_consumer import (
        eps_t_from_pkl_and_reference,
    )
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        write_eps_t_npz,
    )
    from external_validation._rollout_anchors._harness.synthetic_dataset_materializer import (
        materialize_synthetic_dataset,
        read_published_input_windows,
    )

    started = time.time()

    model_name = "segnn"
    dataset_name = "tgv2d"

    # --- Step 0: Locate published TGV2D dataset and SEGNN checkpoint on Volume ---
    published_dataset_dir = Path("/rollouts") / "datasets" / "lagrangebench" / "2D_TGV_2500_10kevery100"
    if not published_dataset_dir.exists():
        # Fallback: rung-4a stored datasets at /vol/datasets/...; symlink via /rollouts/datasets
        # may not be present. Try the alternate layout from rung-4a's modal_app convention.
        alt = Path("/rollouts").parent / "vol" / "datasets" / "lagrangebench" / "2D_TGV_2500_10kevery100"
        if alt.exists():
            published_dataset_dir = alt
        else:
            raise FileNotFoundError(
                f"Published TGV2D dataset not found at {published_dataset_dir} or alternate layout. "
                "Verify rung-4a populated the dataset on Volume."
            )

    published_test_h5 = published_dataset_dir / "test.h5"
    published_metadata_path = published_dataset_dir / "metadata.json"
    published_metadata = json.loads(published_metadata_path.read_text())

    rung_4a_dir = Path("/rollouts") / rung_4a_rollout_subdir
    if not rung_4a_dir.exists():
        raise FileNotFoundError(
            f"Rung-4a rollout dir not found: {rung_4a_dir}. "
            f"Got rung_4a_rollout_subdir={rung_4a_rollout_subdir!r}."
        )

    # Use the same ckpt_hash convention rung-4a wrote into its npz metadata.
    # Read it from rung-4a npz traj 0 to keep provenance consistent.
    sample_ref_npz = rung_4a_dir / "particle_rollout_traj00.npz"
    with np.load(sample_ref_npz, allow_pickle=True) as ref:
        ref_metadata = ref["metadata"].item()
    ckpt_hash_raw = str(ref_metadata.get("ckpt_hash", ""))
    if not ckpt_hash_raw:
        raise ValueError(f"{sample_ref_npz}: ckpt_hash missing from metadata")
    # Namespace per design §3.4: "sha256:<hex>".
    ckpt_hash = ckpt_hash_raw if ckpt_hash_raw.startswith("sha256:") else f"sha256:{ckpt_hash_raw}"

    # --- Step 1: Read 6-frame input windows from published test.h5 ---
    input_windows, particle_type = read_published_input_windows(
        h5_path=published_test_h5, n_trajs=20
    )

    # --- Step 2: Write 20 PH-SYM-003 SKIP eps_t npzs (no LB invocation) ---
    eps_out_dir = Path("/rollouts") / "trajectories" / f"{model_name}_{dataset_name}_{full_git_sha[:10]}"
    eps_out_dir.mkdir(parents=True, exist_ok=True)

    skip_reason = "PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original"
    common_provenance = dict(
        model_name=model_name,
        dataset_name=dataset_name,
        ckpt_hash=ckpt_hash,
        physics_lint_sha_pkl_inference=str(ref_metadata.get("physics_lint_sha_pkl_inference", "")),
        physics_lint_sha_npz_conversion=str(ref_metadata.get("physics_lint_sha_npz_conversion", "")),
        physics_lint_sha_eps_computation=full_git_sha,
    )
    for traj_index in range(20):
        write_eps_t_npz(
            out_dir=eps_out_dir,
            eps_t=np.array([np.nan], dtype=np.float32),
            rule_id="PH-SYM-003",
            transform_kind="skip",
            transform_param="so2_continuous",
            traj_index=traj_index,
            skip_reason=skip_reason,
            **common_provenance,
        )

    # --- Step 3: Pre-execution sanity probe ---
    box_size = float(published_metadata["bounds"][0][1] - published_metadata["bounds"][0][0])
    sanity_synth_dir = Path("/rollouts") / "synthetic" / f"{model_name}_{dataset_name}_sanity_{full_git_sha[:10]}"
    sanity_transforms = [
        {
            "rule_id": "PH-SYM-001",
            "transform_kind": "rotation",
            "transform_param": "pi_2",
            "transform_fn": build_main_sweep_transforms(n_trajs=1)[1]["transform_fn"],  # pi_2 fn
            "original_traj_index": 0,
        }
    ]
    materialize_synthetic_dataset(
        out_dir=sanity_synth_dir,
        input_windows=input_windows[:1],
        particle_type=particle_type,
        transforms=sanity_transforms,
        published_metadata=published_metadata,
        t_steps=7,
        sweep_kind="main",
        stack=model_name,
        dataset=dataset_name,
        ckpt_hash=ckpt_hash,
        physics_lint_sha_eps_computation=full_git_sha,
    )
    sanity_rollout_dir = sanity_synth_dir / "rollout_out"
    sanity_rollout_dir.mkdir(exist_ok=True)
    _run_lb_inference(
        dataset_dir=sanity_synth_dir,
        rollout_out_dir=sanity_rollout_dir,
        ckpt_path="/rollouts/checkpoints/lagrangebench/segnn_tgv2d/best",
        n_rollout_steps=1,
        n_trajs=1,
    )
    sanity_pkl = sanity_rollout_dir / "rollout_0.pkl"
    sanity_eps_t = eps_t_from_pkl_and_reference(
        synthetic_pkl_path=sanity_pkl,
        reference_npz_path=rung_4a_dir / "particle_rollout_traj00.npz",
        transform_kind="rotation",
        transform_param="pi_2",
        t_steps=1,
        box_size=box_size,
    )
    verdict = interpret_sanity_probe_verdict(eps=float(sanity_eps_t[0]))
    print(f"[sanity] {verdict['message']}")
    if verdict["abort"]:
        raise RuntimeError(verdict["message"])

    # --- Step 4: Materialize main-sweep synthetic dataset ---
    main_synth_dir = Path("/rollouts") / "synthetic" / f"{model_name}_{dataset_name}_main_{full_git_sha[:10]}"
    main_transforms = build_main_sweep_transforms(n_trajs=20)
    materialize_synthetic_dataset(
        out_dir=main_synth_dir,
        input_windows=input_windows,
        particle_type=particle_type,
        transforms=main_transforms,
        published_metadata=published_metadata,
        t_steps=7,
        sweep_kind="main",
        stack=model_name,
        dataset=dataset_name,
        ckpt_hash=ckpt_hash,
        physics_lint_sha_eps_computation=full_git_sha,
    )

    # --- Step 5: Materialize figure-sweep synthetic dataset ---
    figure_synth_dir = Path("/rollouts") / "synthetic" / f"{model_name}_{dataset_name}_figure_{full_git_sha[:10]}"
    figure_transforms = build_figure_sweep_transforms()
    materialize_synthetic_dataset(
        out_dir=figure_synth_dir,
        input_windows=input_windows,
        particle_type=particle_type,
        transforms=figure_transforms,
        published_metadata=published_metadata,
        t_steps=106,
        sweep_kind="figure",
        stack=model_name,
        dataset=dataset_name,
        ckpt_hash=ckpt_hash,
        physics_lint_sha_eps_computation=full_git_sha,
    )

    # --- Step 6: Run LB main sweep ---
    main_rollout_dir = main_synth_dir / "rollout_out"
    main_rollout_dir.mkdir(exist_ok=True)
    _run_lb_inference(
        dataset_dir=main_synth_dir,
        rollout_out_dir=main_rollout_dir,
        ckpt_path="/rollouts/checkpoints/lagrangebench/segnn_tgv2d/best",
        n_rollout_steps=1,
        n_trajs=120,
    )

    # --- Step 7: Run LB figure sweep ---
    figure_rollout_dir = figure_synth_dir / "rollout_out"
    figure_rollout_dir.mkdir(exist_ok=True)
    _run_lb_inference(
        dataset_dir=figure_synth_dir,
        rollout_out_dir=figure_rollout_dir,
        ckpt_path="/rollouts/checkpoints/lagrangebench/segnn_tgv2d/best",
        n_rollout_steps=100,
        n_trajs=3,
    )

    # --- Step 8: Post-process pkls -> eps_t npzs (main first, then figure overwrites) ---
    main_manifest = json.loads((main_synth_dir / "manifest.json").read_text())
    npz_count = 20  # the 20 SKIP rows already written
    for entry in main_manifest["trajectories"]:
        i = entry["synthetic_traj_index"]
        pkl_path = main_rollout_dir / f"rollout_{i}.pkl"
        ref_npz_path = rung_4a_dir / f"particle_rollout_traj{entry['original_traj_index']:02d}.npz"
        eps_t = eps_t_from_pkl_and_reference(
            synthetic_pkl_path=pkl_path,
            reference_npz_path=ref_npz_path,
            transform_kind=entry["transform_kind"],
            transform_param=entry["transform_param"],
            t_steps=1,
            box_size=box_size,
        )
        write_eps_t_npz(
            out_dir=eps_out_dir,
            eps_t=eps_t,
            rule_id=entry["rule_id"],
            transform_kind=entry["transform_kind"],
            transform_param=entry["transform_param"],
            traj_index=entry["original_traj_index"],
            skip_reason=None,
            **common_provenance,
        )
        npz_count += 1

    figure_manifest = json.loads((figure_synth_dir / "manifest.json").read_text())
    figure_overwrite_count = 0
    for entry in figure_manifest["trajectories"]:
        i = entry["synthetic_traj_index"]
        pkl_path = figure_rollout_dir / f"rollout_{i}.pkl"
        ref_npz_path = rung_4a_dir / f"particle_rollout_traj{entry['original_traj_index']:02d}.npz"
        eps_t = eps_t_from_pkl_and_reference(
            synthetic_pkl_path=pkl_path,
            reference_npz_path=ref_npz_path,
            transform_kind=entry["transform_kind"],
            transform_param=entry["transform_param"],
            t_steps=100,
            box_size=box_size,
        )
        write_eps_t_npz(
            out_dir=eps_out_dir,
            eps_t=eps_t,
            rule_id=entry["rule_id"],
            transform_kind=entry["transform_kind"],
            transform_param=entry["transform_param"],
            traj_index=entry["original_traj_index"],
            skip_reason=None,
            **common_provenance,
        )
        figure_overwrite_count += 1
    # The 3 figure overwrites do not increase npz_count (they overwrite main-sweep npzs).

    rollout_volume.commit()  # persist eps_out_dir + synthetic_dirs to Volume

    elapsed = time.time() - started
    return {
        "npz_count": npz_count,
        "sanity_probe_eps": float(sanity_eps_t[0]),
        "elapsed_s": elapsed,
        "git_sha_eps": full_git_sha,
        "eps_out_dir": str(eps_out_dir),
        "synthetic_dirs": [str(main_synth_dir), str(figure_synth_dir)],
        "figure_overwrite_count": figure_overwrite_count,
    }


def _run_lb_inference(
    *,
    dataset_dir,
    rollout_out_dir,
    ckpt_path: str,
    n_rollout_steps: int,
    n_trajs: int,
) -> None:
    """Invoke LB `mode=infer` via subprocess; mirrors rung 4a's CLI shape.

    Args mirror rung-4a's lagrangebench_rollout_p0_segnn_tgv2d invocation
    (eval.test=True, eval.infer.metrics=[mse,e_kin], eval.infer.out_type=pkl, seed=0).
    """
    import subprocess

    cmd = [
        "python",
        "main.py",
        "mode=infer",
        "eval.test=True",
        f"load_ckp={ckpt_path}",
        f"eval.n_rollout_steps={n_rollout_steps}",
        f"eval.infer.n_trajs={n_trajs}",
        f"dataset.src={dataset_dir}",
        "dataset.name=tgv2d",
        "eval.infer.metrics=[mse,e_kin]",
        "eval.infer.out_type=pkl",
        f"rollout_dir={rollout_out_dir}",
        "seed=0",
    ]
    print(f"[lb-infer] {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd="/lagrangebench",  # rung-4a convention
        capture_output=True,
        text=True,
        check=False,
        timeout=60 * 20,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"LB inference failed (returncode={proc.returncode})\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
    print(f"[lb-infer] success; stdout tail:\n{proc.stdout[-500:]}")
```

- [ ] **Step 2: Sanity-check Python parses cleanly**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" python -c "import ast; ast.parse(open('external_validation/_rollout_anchors/01-lagrangebench/modal_app.py').read()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Sanity-check the LB inference cwd path matches rung-4a**

```bash
grep -B1 -A1 "cwd=" /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/modal_app.py | head -20
```

Expected: at least one occurrence of `cwd="/lagrangebench"` (rung-4a's convention). If rung-4a uses a different cwd (e.g., a different path in the LB image), update `_run_lb_inference` to match. The actual path is whatever `subprocess.run` cwd was set to in `lagrangebench_rollout_p0_segnn_tgv2d`.

- [ ] **Step 4: Commit T4.2**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/01-lagrangebench/modal_app.py && git commit -m "01-lagrangebench/modal_app: T7 SEGNN-TGV2D eps Modal entrypoint (lagrangebench_eps_p0_segnn_tgv2d)"
```

---

## Task 5: Modal entrypoint — GNS

**Why this exists:** Mirror of T4 with `model_name="gns"`. Most of the body is byte-identical to SEGNN; the only differences are the rung_4a npz subdir, ckpt_path, and model_name string.

**Files:**
- Modify: `external_validation/_rollout_anchors/01-lagrangebench/modal_app.py`

### T5.1: Append the GNS ε entrypoint

- [ ] **Step 1: Append the new `@app.function` to modal_app.py**

Append immediately after `lagrangebench_eps_p0_segnn_tgv2d` (and its helper `_run_lb_inference`):

```python
@app.function(
    image=lagrangebench_image,
    gpu=ROLLOUT_GENERATION_GPU_CLASS,
    timeout=60 * 30,
    volumes={"/rollouts": rollout_volume},
)
def lagrangebench_eps_p1_gns_tgv2d(
    git_sha: str,
    full_git_sha: str,
    rung_4a_rollout_subdir: str,
) -> dict:
    """Rung 4b T7: GNS-TGV2D equivariance eps(t) computation.

    Mirror of lagrangebench_eps_p0_segnn_tgv2d with model_name="gns" and
    the GNS checkpoint path. Pipeline shape, sanity-probe gate, and
    post-processing logic are identical.
    """
    import json
    import time
    from pathlib import Path

    import numpy as np

    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        build_figure_sweep_transforms,
        build_main_sweep_transforms,
        interpret_sanity_probe_verdict,
    )
    from external_validation._rollout_anchors._harness.eps_pkl_consumer import (
        eps_t_from_pkl_and_reference,
    )
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        write_eps_t_npz,
    )
    from external_validation._rollout_anchors._harness.synthetic_dataset_materializer import (
        materialize_synthetic_dataset,
        read_published_input_windows,
    )

    started = time.time()

    model_name = "gns"
    dataset_name = "tgv2d"
    ckpt_path = "/rollouts/checkpoints/lagrangebench/gns_tgv2d/best"

    # --- Step 0: Locate published TGV2D dataset ---
    published_dataset_dir = Path("/rollouts") / "datasets" / "lagrangebench" / "2D_TGV_2500_10kevery100"
    if not published_dataset_dir.exists():
        alt = Path("/rollouts").parent / "vol" / "datasets" / "lagrangebench" / "2D_TGV_2500_10kevery100"
        if alt.exists():
            published_dataset_dir = alt
        else:
            raise FileNotFoundError(
                f"Published TGV2D dataset not found at {published_dataset_dir} or alternate layout."
            )
    published_test_h5 = published_dataset_dir / "test.h5"
    published_metadata = json.loads((published_dataset_dir / "metadata.json").read_text())

    rung_4a_dir = Path("/rollouts") / rung_4a_rollout_subdir
    if not rung_4a_dir.exists():
        raise FileNotFoundError(f"Rung-4a rollout dir not found: {rung_4a_dir}")

    sample_ref_npz = rung_4a_dir / "particle_rollout_traj00.npz"
    with np.load(sample_ref_npz, allow_pickle=True) as ref:
        ref_metadata = ref["metadata"].item()
    ckpt_hash_raw = str(ref_metadata.get("ckpt_hash", ""))
    if not ckpt_hash_raw:
        raise ValueError(f"{sample_ref_npz}: ckpt_hash missing from metadata")
    ckpt_hash = ckpt_hash_raw if ckpt_hash_raw.startswith("sha256:") else f"sha256:{ckpt_hash_raw}"

    # --- Step 1: Read input windows ---
    input_windows, particle_type = read_published_input_windows(
        h5_path=published_test_h5, n_trajs=20
    )

    # --- Step 2: Write 20 PH-SYM-003 SKIP rows ---
    eps_out_dir = Path("/rollouts") / "trajectories" / f"{model_name}_{dataset_name}_{full_git_sha[:10]}"
    eps_out_dir.mkdir(parents=True, exist_ok=True)
    skip_reason = "PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original"
    common_provenance = dict(
        model_name=model_name,
        dataset_name=dataset_name,
        ckpt_hash=ckpt_hash,
        physics_lint_sha_pkl_inference=str(ref_metadata.get("physics_lint_sha_pkl_inference", "")),
        physics_lint_sha_npz_conversion=str(ref_metadata.get("physics_lint_sha_npz_conversion", "")),
        physics_lint_sha_eps_computation=full_git_sha,
    )
    for traj_index in range(20):
        write_eps_t_npz(
            out_dir=eps_out_dir,
            eps_t=np.array([np.nan], dtype=np.float32),
            rule_id="PH-SYM-003",
            transform_kind="skip",
            transform_param="so2_continuous",
            traj_index=traj_index,
            skip_reason=skip_reason,
            **common_provenance,
        )

    # --- Step 3: Sanity probe ---
    box_size = float(published_metadata["bounds"][0][1] - published_metadata["bounds"][0][0])
    sanity_synth_dir = Path("/rollouts") / "synthetic" / f"{model_name}_{dataset_name}_sanity_{full_git_sha[:10]}"
    sanity_transforms = [
        {
            "rule_id": "PH-SYM-001",
            "transform_kind": "rotation",
            "transform_param": "pi_2",
            "transform_fn": build_main_sweep_transforms(n_trajs=1)[1]["transform_fn"],
            "original_traj_index": 0,
        }
    ]
    materialize_synthetic_dataset(
        out_dir=sanity_synth_dir,
        input_windows=input_windows[:1],
        particle_type=particle_type,
        transforms=sanity_transforms,
        published_metadata=published_metadata,
        t_steps=7,
        sweep_kind="main",
        stack=model_name,
        dataset=dataset_name,
        ckpt_hash=ckpt_hash,
        physics_lint_sha_eps_computation=full_git_sha,
    )
    sanity_rollout_dir = sanity_synth_dir / "rollout_out"
    sanity_rollout_dir.mkdir(exist_ok=True)
    _run_lb_inference(
        dataset_dir=sanity_synth_dir,
        rollout_out_dir=sanity_rollout_dir,
        ckpt_path=ckpt_path,
        n_rollout_steps=1,
        n_trajs=1,
    )
    sanity_eps_t = eps_t_from_pkl_and_reference(
        synthetic_pkl_path=sanity_rollout_dir / "rollout_0.pkl",
        reference_npz_path=rung_4a_dir / "particle_rollout_traj00.npz",
        transform_kind="rotation",
        transform_param="pi_2",
        t_steps=1,
        box_size=box_size,
    )
    verdict = interpret_sanity_probe_verdict(eps=float(sanity_eps_t[0]))
    print(f"[sanity] {verdict['message']}")
    # GNS path: gate is informational, never abort. GNS's expected eps is in the
    # (1e-5, 1e-2] APPROXIMATE band per design §3.3 + Helwig et al. ICML 2023's
    # architecture-level characterization — that's the headline finding to
    # report, not a gate condition to enforce. The verdict's diagnostic bands
    # still discriminate "concerning" (architectural approximation; expected)
    # from "clear bug" (> 1e-2), so a coord-space / frame-index / normalization /
    # manifest-mapping bug would still surface in the log even though no abort.
    print("[sanity] GNS gate is informational; proceeding regardless of band.")

    # --- Step 4: Main sweep ---
    main_synth_dir = Path("/rollouts") / "synthetic" / f"{model_name}_{dataset_name}_main_{full_git_sha[:10]}"
    main_transforms = build_main_sweep_transforms(n_trajs=20)
    materialize_synthetic_dataset(
        out_dir=main_synth_dir,
        input_windows=input_windows,
        particle_type=particle_type,
        transforms=main_transforms,
        published_metadata=published_metadata,
        t_steps=7,
        sweep_kind="main",
        stack=model_name,
        dataset=dataset_name,
        ckpt_hash=ckpt_hash,
        physics_lint_sha_eps_computation=full_git_sha,
    )
    main_rollout_dir = main_synth_dir / "rollout_out"
    main_rollout_dir.mkdir(exist_ok=True)
    _run_lb_inference(
        dataset_dir=main_synth_dir,
        rollout_out_dir=main_rollout_dir,
        ckpt_path=ckpt_path,
        n_rollout_steps=1,
        n_trajs=120,
    )

    # --- Step 5: Figure sweep ---
    figure_synth_dir = Path("/rollouts") / "synthetic" / f"{model_name}_{dataset_name}_figure_{full_git_sha[:10]}"
    figure_transforms = build_figure_sweep_transforms()
    materialize_synthetic_dataset(
        out_dir=figure_synth_dir,
        input_windows=input_windows,
        particle_type=particle_type,
        transforms=figure_transforms,
        published_metadata=published_metadata,
        t_steps=106,
        sweep_kind="figure",
        stack=model_name,
        dataset=dataset_name,
        ckpt_hash=ckpt_hash,
        physics_lint_sha_eps_computation=full_git_sha,
    )
    figure_rollout_dir = figure_synth_dir / "rollout_out"
    figure_rollout_dir.mkdir(exist_ok=True)
    _run_lb_inference(
        dataset_dir=figure_synth_dir,
        rollout_out_dir=figure_rollout_dir,
        ckpt_path=ckpt_path,
        n_rollout_steps=100,
        n_trajs=3,
    )

    # --- Step 6: Post-process pkls -> eps_t npzs ---
    main_manifest = json.loads((main_synth_dir / "manifest.json").read_text())
    npz_count = 20
    for entry in main_manifest["trajectories"]:
        i = entry["synthetic_traj_index"]
        pkl_path = main_rollout_dir / f"rollout_{i}.pkl"
        ref_npz_path = rung_4a_dir / f"particle_rollout_traj{entry['original_traj_index']:02d}.npz"
        eps_t = eps_t_from_pkl_and_reference(
            synthetic_pkl_path=pkl_path,
            reference_npz_path=ref_npz_path,
            transform_kind=entry["transform_kind"],
            transform_param=entry["transform_param"],
            t_steps=1,
            box_size=box_size,
        )
        write_eps_t_npz(
            out_dir=eps_out_dir,
            eps_t=eps_t,
            rule_id=entry["rule_id"],
            transform_kind=entry["transform_kind"],
            transform_param=entry["transform_param"],
            traj_index=entry["original_traj_index"],
            skip_reason=None,
            **common_provenance,
        )
        npz_count += 1

    figure_manifest = json.loads((figure_synth_dir / "manifest.json").read_text())
    figure_overwrite_count = 0
    for entry in figure_manifest["trajectories"]:
        i = entry["synthetic_traj_index"]
        pkl_path = figure_rollout_dir / f"rollout_{i}.pkl"
        ref_npz_path = rung_4a_dir / f"particle_rollout_traj{entry['original_traj_index']:02d}.npz"
        eps_t = eps_t_from_pkl_and_reference(
            synthetic_pkl_path=pkl_path,
            reference_npz_path=ref_npz_path,
            transform_kind=entry["transform_kind"],
            transform_param=entry["transform_param"],
            t_steps=100,
            box_size=box_size,
        )
        write_eps_t_npz(
            out_dir=eps_out_dir,
            eps_t=eps_t,
            rule_id=entry["rule_id"],
            transform_kind=entry["transform_kind"],
            transform_param=entry["transform_param"],
            traj_index=entry["original_traj_index"],
            skip_reason=None,
            **common_provenance,
        )
        figure_overwrite_count += 1

    rollout_volume.commit()

    elapsed = time.time() - started
    return {
        "npz_count": npz_count,
        "sanity_probe_eps": float(sanity_eps_t[0]),
        "elapsed_s": elapsed,
        "git_sha_eps": full_git_sha,
        "eps_out_dir": str(eps_out_dir),
        "synthetic_dirs": [str(main_synth_dir), str(figure_synth_dir)],
        "figure_overwrite_count": figure_overwrite_count,
    }
```

- [ ] **Step 2: Sanity-check Python parses cleanly**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" python -c "import ast; ast.parse(open('external_validation/_rollout_anchors/01-lagrangebench/modal_app.py').read()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit T5**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/01-lagrangebench/modal_app.py && git commit -m "01-lagrangebench/modal_app: T7 GNS-TGV2D eps Modal entrypoint (lagrangebench_eps_p1_gns_tgv2d)"
```

---

## Task 6: Local entrypoints

**Why this exists:** `@app.local_entrypoint()` wrappers that resolve the git sha from `git rev-parse HEAD` and invoke each Modal entrypoint with the right arguments. Mirrors rung-4a's `rollout_p0_segnn_tgv2d` pattern.

**Files:**
- Modify: `external_validation/_rollout_anchors/01-lagrangebench/modal_app.py`

### T6.1: Append local entrypoints

- [ ] **Step 1: Find the existing local-entrypoint section**

```bash
grep -n "^@app.local_entrypoint\|^def rollout_p" /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/modal_app.py
```

Expected: existing `rollout_p0_segnn_tgv2d` and `rollout_p1_gns_tgv2d` local entrypoints. Append after the GNS one.

- [ ] **Step 2: Append new local entrypoints**

Append at the end of the existing local-entrypoint section:

```python
@app.local_entrypoint()
def eps_p0_segnn_tgv2d() -> None:
    """Local entrypoint: invoke lagrangebench_eps_p0_segnn_tgv2d.remote()."""
    import subprocess

    full_git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_sha = full_git_sha[:10]
    rung_4a_subdir = "segnn_tgv2d_post_d03df3e"  # rung-4a's published rollout subdir
    res = lagrangebench_eps_p0_segnn_tgv2d.remote(
        git_sha=git_sha,
        full_git_sha=full_git_sha,
        rung_4a_rollout_subdir=rung_4a_subdir,
    )
    print("eps_p0_segnn_tgv2d:", res)


@app.local_entrypoint()
def eps_p1_gns_tgv2d() -> None:
    """Local entrypoint: invoke lagrangebench_eps_p1_gns_tgv2d.remote()."""
    import subprocess

    full_git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_sha = full_git_sha[:10]
    rung_4a_subdir = "gns_tgv2d_post_d03df3e"  # rung-4a's published rollout subdir
    res = lagrangebench_eps_p1_gns_tgv2d.remote(
        git_sha=git_sha,
        full_git_sha=full_git_sha,
        rung_4a_rollout_subdir=rung_4a_subdir,
    )
    print("eps_p1_gns_tgv2d:", res)
```

- [ ] **Step 3: Sanity-check Python parses cleanly**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" python -c "import ast; ast.parse(open('external_validation/_rollout_anchors/01-lagrangebench/modal_app.py').read()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Verify the rung-4a subdir names match what's actually on Volume**

```bash
cd /Users/zenith/Desktop/physics-lint && grep -n "post_d03df3e\|rollout_subdir" external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py | head -5
```

Expected: rung-4a's `emit_sarif.py` references the actual subdir names. If the substring `_post_d03df3e` does not appear (i.e., rung-4a uses different subdir names like `segnn_tgv2d_8c3d080397`), update the `rung_4a_subdir` literals in the local entrypoints to match exactly.

- [ ] **Step 5: Commit T6**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/01-lagrangebench/modal_app.py && git commit -m "01-lagrangebench/modal_app: T7 local entrypoints (eps_p0_segnn_tgv2d, eps_p1_gns_tgv2d)"
```

---

## Task 7: Drift-guard test for GPU class on T7 entrypoints

**Why this exists:** rung-4a maintains `tests/test_modal_app_gpu_class.py` to assert the GPU class doesn't drift from A10G. T7 entrypoints should be covered by the same drift-guard.

**Files:**
- Modify: `external_validation/_rollout_anchors/01-lagrangebench/tests/test_modal_app_gpu_class.py`

### T7.1: Extend the GPU drift-guard

- [ ] **Step 1: Read the existing drift-guard**

```bash
cat /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/tests/test_modal_app_gpu_class.py
```

Expected: a test that asserts `lagrangebench_rollout_p{0,1}_*` Modal functions have `gpu="A10G"`. Note the testing pattern (likely AST-based or attribute introspection).

- [ ] **Step 2: Append a test for the new T7 entrypoints**

Append to `test_modal_app_gpu_class.py`:

```python


def test_lagrangebench_eps_entrypoints_use_a10g():
    """T7 eps entrypoints must match rung-4a's A10G GPU class per D0-21 item 10."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "modal_app.py"
    text = src.read_text()
    # Look for the @app.function block immediately preceding each eps entrypoint.
    for fn_name in ("lagrangebench_eps_p0_segnn_tgv2d", "lagrangebench_eps_p1_gns_tgv2d"):
        pattern = rf"@app\.function\([^)]*?gpu=ROLLOUT_GENERATION_GPU_CLASS[^)]*?\)\s*\ndef {fn_name}"
        match = re.search(pattern, text, flags=re.DOTALL)
        assert match is not None, (
            f"{fn_name}: expected @app.function decorator with gpu=ROLLOUT_GENERATION_GPU_CLASS"
        )
    # Also assert ROLLOUT_GENERATION_GPU_CLASS is "A10G".
    rotation_class_match = re.search(
        r'ROLLOUT_GENERATION_GPU_CLASS\s*=\s*\(\s*"A10G"', text
    )
    assert rotation_class_match is not None, (
        "ROLLOUT_GENERATION_GPU_CLASS must be 'A10G' per D0-13 stage 2 / D0-21 item 10"
    )
```

- [ ] **Step 3: Run the test**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/01-lagrangebench/tests/test_modal_app_gpu_class.py -v
```

Expected: all PASSED, including the new test.

- [ ] **Step 4: Commit T7**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/01-lagrangebench/tests/test_modal_app_gpu_class.py && git commit -m "01-lagrangebench/tests: GPU drift-guard for T7 eps entrypoints"
```

---

## Task 8: Full external_validation regression

**Why this exists:** Final regression check after T0–T7 land. Confirms rung-4a tests + T0–T6+T8 consumer-side tests + new T7 module tests all pass with no cross-component breakage.

### T8.1: Run full external_validation test suite

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/ -q 2>&1 | tail -5
```

Expected: all tests pass (no regressions). New T7 contribution: ~22 additional tests across `test_synthetic_dataset_materializer.py`, `test_eps_pkl_consumer.py`, `test_eps_modal_orchestrator.py`. Total external_validation tests should now be ~444+ passed.

If any test fails, do NOT commit. Investigate the failure; the most likely cause is a name mismatch between a T7 module and a function it imports (e.g., `compute_eps_t_from_pair` was misspelled).

---

## Self-review (against design spec)

**Spec coverage:**

| Spec section | Task |
|---|---|
| §2 Pipeline shape (steps 1-8) | T1 (read+materialize), T2 (post-process), T3 (orchestrator), T4 (SEGNN entrypoint glue), T5 (GNS), T6 (local entrypoints) |
| §3 Synthetic H5 dataset spec (test/train/valid + metadata + manifest) | T1 (materialize_synthetic_dataset + read_published_input_windows) |
| §3.3 Metadata reuse policy | T1 (test_materialize_synthetic_dataset_metadata_reuses_published_stats) |
| §3.4 Manifest schema (incl. namespaced ckpt_hash) | T1 (test_materialize_synthetic_dataset_manifest_schema, test_..._rejects_unnamespaced_ckpt_hash) |
| §4 PH-SYM-003 SKIP shortcut | T4 + T5 (Step 2 of each entrypoint writes 20 SKIP rows before any LB invocation) |
| §5 Off-by-one frame index | T2 (eps_t_from_pkl_and_reference slices [6:6+t_steps]) |
| §6 Pre-execution sanity probe + diagnostic bands | T3 (interpret_sanity_probe_verdict + tests), T4 (SEGNN blocking gate at call site), T5 (GNS informational gate at call site, with explanatory comment) |
| §7 Compute scope (4 LB invocations) | T4 + T5 (each entrypoint runs 2 LB invocations: main + figure) |
| §8 Silent-mismatch hazards | T1 (metadata reuse, contiguity assertion, ckpt_hash namespace assertion), T2 (off-by-one slice + assertion), T3 (sanity-probe gate) |
| §9 Out of scope | Plan does not implement (a)/(b)/upstream patches/figure visualization (T10/T11 follow-up) |
| §10 Acceptance-criteria deltas | T4 + T5 sanity-probe gate; T0 gitignore; entrypoints record physics_lint_sha_eps_computation as single sha |
| §12 D-entry footprint (no new D-entries) | Plan does not modify DECISIONS.md |

No spec gaps.

**Placeholder scan:** No "TBD", "TODO", "fill in details", or "similar to Task N". Two sites where the plan asks the engineer to verify external state at runtime (T6.1 step 4: rung-4a subdir name; T4.2 step 3: LB inference cwd) — both are runtime-checks, not unfilled spec, and both have explicit fallback instructions.

**Type consistency:**
- `transform_kind` enum: same five values across T1, T2, T3, T4, T5 ("rotation" | "reflection" | "translation" | "identity" | "skip")
- `transform_param` is always a string in all read paths
- Manifest `ckpt_hash` is namespaced `"sha256:<hex>"` consistently across T1.4, T4, T5
- `t_steps`: 7 (main) and 106 (figure) consistently across T1.2, T1.4, T2.4, T4, T5
- `INPUT_SEQ_LENGTH = 6` defined once in eps_pkl_consumer.py and asserted at runtime

No inconsistencies found.

---

## Execution Handoff

**Plan complete and saved to `external_validation/_rollout_anchors/methodology/docs/2026-05-06-rung-4b-t7-modal-entrypoints-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks. Use `superpowers:subagent-driven-development`.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch with checkpoints for review.

**Which approach?**
