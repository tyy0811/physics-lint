"""Tests for emit_sarif_eps.py — rung 4b case-study driver."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Make the case-study directory importable as a top-level package.
# 01-lagrangebench is not a valid Python identifier, so we sys.path-insert
# the directory itself and import the file by its module name.
_DRIVER_DIR = Path(__file__).resolve().parents[1]
if str(_DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(_DRIVER_DIR))

from emit_sarif_eps import emit_sarif_eps  # type: ignore[import-not-found]  # noqa: E402


def _populate_eps_dir(eps_dir: Path) -> None:
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        write_eps_t_npz,
    )

    eps_dir.mkdir(parents=True, exist_ok=True)
    common_kwargs = dict(
        out_dir=eps_dir,
        model_name="segnn",
        dataset_name="tgv2d",
        ckpt_hash="abc123",
        physics_lint_sha_pkl_inference="8c3d080000",
        physics_lint_sha_npz_conversion="5857144000",
        physics_lint_sha_eps_computation="d9a8baa000",
    )
    # PH-SYM-001 active angles + identity smoke
    for ang_str in ("pi_2", "pi", "3pi_2", "0"):
        write_eps_t_npz(
            **common_kwargs,
            eps_t=np.array([4.3e-7], dtype=np.float32),
            rule_id="PH-SYM-001",
            transform_kind="rotation" if ang_str != "0" else "identity",
            transform_param=ang_str,
            traj_index=0,
            skip_reason=None,
        )
    # PH-SYM-002 active
    write_eps_t_npz(
        **common_kwargs,
        eps_t=np.array([4.4e-7], dtype=np.float32),
        rule_id="PH-SYM-002",
        transform_kind="reflection",
        transform_param="y_axis",
        traj_index=0,
        skip_reason=None,
    )
    # PH-SYM-004 active (translation)
    write_eps_t_npz(
        **common_kwargs,
        eps_t=np.array([1.5e-15], dtype=np.float32),
        rule_id="PH-SYM-004",
        transform_kind="translation",
        transform_param="L_3_L_7",
        traj_index=0,
        skip_reason=None,
    )
    # PH-SYM-003 SKIP
    write_eps_t_npz(
        **common_kwargs,
        eps_t=np.array([np.nan], dtype=np.float32),
        rule_id="PH-SYM-003",
        transform_kind="skip",
        transform_param="so2_continuous",
        traj_index=0,
        skip_reason="PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original",
    )


def test_emit_sarif_eps_produces_v1_1_sarif(tmp_path):
    eps_dir = tmp_path / "segnn_tgv2d_d9a8baa000"
    _populate_eps_dir(eps_dir)
    out_sarif = tmp_path / "segnn_tgv2d_eps.sarif"

    emit_sarif_eps(
        eps_dir=eps_dir,
        out_sarif_path=out_sarif,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model="segnn",
        ckpt_hash="abc123",
        ckpt_id="segnn_tgv2d/best",
        physics_lint_sha_pkl_inference="8c3d080000",
        physics_lint_sha_npz_conversion="5857144000",
        physics_lint_sha_eps_computation="d9a8baa000",
        physics_lint_sha_sarif_emission="d9a8baa000",
        lagrangebench_sha="ee0001eeee",
        rollout_subdir="rollouts/segnn_tgv2d_post_d03df3e",
    )

    sarif = json.loads(out_sarif.read_text())
    run_props = sarif["runs"][0]["properties"]
    assert run_props["harness_sarif_schema_version"] == "1.1"
    assert run_props["physics_lint_sha_eps_computation"] == "d9a8baa000"

    rule_ids = {r["ruleId"] for r in sarif["runs"][0]["results"]}
    assert rule_ids == {"PH-SYM-001", "PH-SYM-002", "PH-SYM-003", "PH-SYM-004"}

    skip_rows = [r for r in sarif["runs"][0]["results"] if r["ruleId"] == "PH-SYM-003"]
    assert len(skip_rows) == 1
    assert skip_rows[0]["properties"]["skip_reason"].startswith("PBC-square breaks SO(2)")
