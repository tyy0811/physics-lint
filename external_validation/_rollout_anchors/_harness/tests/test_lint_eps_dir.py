"""Tests for lint_eps_dir: eps(t) npz dir -> HarnessResult rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _make_eps_dir_with_one_active_row(tmp_path: Path) -> Path:
    """Create a fixture dir with one active PH-SYM-001 eps(t) npz."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        write_eps_t_npz,
    )

    eps_dir = tmp_path / "fixture_eps_dir"
    eps_dir.mkdir()
    write_eps_t_npz(
        out_dir=eps_dir,
        eps_t=np.array([4.3e-7], dtype=np.float32),
        rule_id="PH-SYM-001",
        transform_kind="rotation",
        transform_param="pi_2",
        traj_index=0,
        model_name="segnn",
        dataset_name="tgv2d",
        ckpt_hash="abc123",
        physics_lint_sha_pkl_inference="aaaaaaaaaa",
        physics_lint_sha_npz_conversion="bbbbbbbbbb",
        physics_lint_sha_eps_computation="cccccccccc",
        skip_reason=None,
    )
    return eps_dir


def test_lint_eps_dir_active_row_yields_one_harness_result(tmp_path):
    from external_validation._rollout_anchors._harness.lint_eps_dir import lint_eps_dir

    eps_dir = _make_eps_dir_with_one_active_row(tmp_path)
    results = lint_eps_dir(
        eps_dir=eps_dir,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model="segnn",
        ckpt_hash="abc123",
    )
    assert len(results) == 1
    r = results[0]
    assert r.rule_id == "PH-SYM-001"
    assert r.raw_value == pytest.approx(4.3e-7, abs=1e-12)
    assert r.level == "note"
    assert r.message.startswith("eps_pos_rms=")
    assert r.extra_properties["transform_kind"] == "rotation"
    assert r.extra_properties["transform_param"] == "pi_2"
    assert r.extra_properties["traj_index"] == 0
    assert r.extra_properties["eps_pos_rms"] == pytest.approx(4.3e-7, abs=1e-12)
    assert r.extra_properties["eps_t_npz_filename"] == "eps_PH-SYM-001_rotation_pi_2_traj00.npz"
    assert "skip_reason" not in r.extra_properties


def test_lint_eps_dir_empty_dir_raises():
    from external_validation._rollout_anchors._harness.lint_eps_dir import (
        EmptyEpsDirectoryError,
        lint_eps_dir,
    )

    with pytest.raises(EmptyEpsDirectoryError, match=r"No eps_.*\.npz files"):
        lint_eps_dir(
            eps_dir=Path("/tmp/nonexistent_eps_dir_for_test"),
            case_study="01-lagrangebench",
            dataset="tgv2d",
            model="segnn",
            ckpt_hash="abc123",
        )


def test_lint_eps_dir_skip_row_yields_skip_reason_in_extra_properties(tmp_path):
    from external_validation._rollout_anchors._harness.lint_eps_dir import lint_eps_dir
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        write_eps_t_npz,
    )

    eps_dir = tmp_path / "skip_fixture"
    eps_dir.mkdir()
    write_eps_t_npz(
        out_dir=eps_dir,
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

    results = lint_eps_dir(
        eps_dir=eps_dir,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model="segnn",
        ckpt_hash="abc123",
    )
    assert len(results) == 1
    r = results[0]
    assert r.rule_id == "PH-SYM-003"
    assert r.raw_value is None
    assert r.message.startswith("SKIP: ")
    assert "PBC-square breaks SO(2)" in r.message
    assert r.extra_properties["skip_reason"].startswith("PBC-square breaks SO(2)")
    assert r.extra_properties["eps_pos_rms"] is None
    assert r.extra_properties["transform_kind"] == "skip"


def test_lint_eps_dir_skip_reason_identical_across_rows_same_rule(tmp_path):
    """D0-19 §3.4 contract: skip_reason is guaranteed-identical across rows
    within (rule, stack). The lint_eps_dir consumer must preserve this —
    each PH-SYM-003 row carries the same skip_reason string."""
    from external_validation._rollout_anchors._harness.lint_eps_dir import lint_eps_dir
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        write_eps_t_npz,
    )

    eps_dir = tmp_path / "skip_multi"
    eps_dir.mkdir()
    skip_reason = "PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original"
    for traj in range(3):
        write_eps_t_npz(
            out_dir=eps_dir,
            eps_t=np.array([np.nan], dtype=np.float32),
            rule_id="PH-SYM-003",
            transform_kind="skip",
            transform_param="so2_continuous",
            traj_index=traj,
            model_name="segnn",
            dataset_name="tgv2d",
            ckpt_hash="abc123",
            physics_lint_sha_pkl_inference="aaaaaaaaaa",
            physics_lint_sha_npz_conversion="bbbbbbbbbb",
            physics_lint_sha_eps_computation="cccccccccc",
            skip_reason=skip_reason,
        )

    results = lint_eps_dir(
        eps_dir=eps_dir,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model="segnn",
        ckpt_hash="abc123",
    )
    assert len(results) == 3
    skip_reasons = {r.extra_properties["skip_reason"] for r in results}
    assert len(skip_reasons) == 1, (
        "skip_reason must be guaranteed-identical across rows (D0-19 §3.4)"
    )
