"""Tests for sarif_emitter at schema_version v1.1 (rung 4b).

The existing emitter (rung 4a) accepts any string-keyed dict for
`run_properties` and copies HarnessResult.extra_properties verbatim
into the result-level `properties`. v1.1 emission therefore needs no
emitter-side change; this test pins that contract so a future
restriction would fail loud.
"""

from __future__ import annotations

import json

from external_validation._rollout_anchors._harness.sarif_emitter import (
    HarnessResult,
    emit_sarif,
)


def _v1_1_run_properties() -> dict:
    return {
        "source": "rollout-anchor-harness",
        "harness_sarif_schema_version": "1.1",
        "physics_lint_sha_pkl_inference": "aaaaaaaaaa",
        "physics_lint_sha_npz_conversion": "bbbbbbbbbb",
        "physics_lint_sha_sarif_emission": "cccccccccc",
        "physics_lint_sha_eps_computation": "dddddddddd",
        "lagrangebench_sha": "eeeeeeeeee",
        "checkpoint_id": "segnn_tgv2d/best",
        "model_name": "segnn",
        "dataset_name": "tgv2d",
        "rollout_subdir": "rollouts/segnn_tgv2d_post_d03df3e",
    }


def test_emit_sarif_v1_1_writes_schema_version_and_extra_properties(tmp_path):
    out_path = tmp_path / "v1_1.sarif"
    results = [
        HarnessResult(
            rule_id="PH-SYM-001",
            level="note",
            message="eps_pos_rms=4.3e-07 (transform=rotation pi_2)",
            raw_value=4.3e-7,
            case_study="01-lagrangebench",
            dataset="tgv2d",
            model="segnn",
            ckpt_hash="abc123",
            extra_properties={
                "transform_kind": "rotation",
                "transform_param": "pi_2",
                "traj_index": 0,
                "eps_pos_rms": 4.3e-7,
                "eps_t_npz_filename": "eps_PH-SYM-001_rotation_pi_2_traj00.npz",
            },
        )
    ]
    emit_sarif(results, output_path=out_path, run_properties=_v1_1_run_properties())

    sarif = json.loads(out_path.read_text())
    run_props = sarif["runs"][0]["properties"]
    assert run_props["harness_sarif_schema_version"] == "1.1"
    assert run_props["physics_lint_sha_eps_computation"] == "dddddddddd"

    result_props = sarif["runs"][0]["results"][0]["properties"]
    assert result_props["transform_kind"] == "rotation"
    assert result_props["transform_param"] == "pi_2"
    assert result_props["eps_pos_rms"] == 4.3e-7
    assert result_props["eps_t_npz_filename"] == "eps_PH-SYM-001_rotation_pi_2_traj00.npz"


def test_emit_sarif_v1_0_path_still_works_unchanged(tmp_path):
    """Backward-compat: v1.0 emission with rung-4a-shaped properties unchanged."""
    out_path = tmp_path / "v1_0.sarif"
    results = [
        HarnessResult(
            rule_id="harness:mass_conservation_defect",
            level="note",
            message="raw_value=0.000e+00",
            raw_value=0.0,
            case_study="01-lagrangebench",
            dataset="tgv2d",
            model="segnn",
            ckpt_hash="abc123",
            extra_properties={
                "traj_index": 0,
                "npz_filename": "particle_rollout_traj00.npz",
            },
        )
    ]
    v1_0_run_props = {
        "source": "rollout-anchor-harness",
        "harness_sarif_schema_version": "1.0",
        "physics_lint_sha_pkl_inference": "aaaa",
        "physics_lint_sha_npz_conversion": "bbbb",
        "physics_lint_sha_sarif_emission": "cccc",
        "lagrangebench_sha": "dddd",
        "checkpoint_id": "segnn_tgv2d/best",
        "model_name": "segnn",
        "dataset_name": "tgv2d",
        "rollout_subdir": "rollouts/segnn_tgv2d_post_d03df3e",
    }
    emit_sarif(results, output_path=out_path, run_properties=v1_0_run_props)

    sarif = json.loads(out_path.read_text())
    run_props = sarif["runs"][0]["properties"]
    assert run_props["harness_sarif_schema_version"] == "1.0"
    assert "physics_lint_sha_eps_computation" not in run_props
