"""Tests for methodology/tools/render_eps_table.py — rung 4b sibling renderer.

Per design §5.1: renderer is v1.1-only. v1.0 SARIFs cannot be rendered;
schema_version mismatch raises SchemaVersionMismatchError. Tests use
hand-crafted v1.1 SARIF fixtures (per the project's
"test fixtures hand-crafted, not copied from production" discipline).
"""

from __future__ import annotations

import json

import pytest

from external_validation._rollout_anchors.methodology.tools.render_eps_table import (
    SchemaVersionMismatchError,
    render_eps_table,
)


def _v1_1_sarif_fixture(model: str = "segnn", with_skip: bool = True) -> dict:
    """Build a v1.1 SARIF dict in-memory matching the rung-4b emission shape."""
    results = []
    for ang_str, eps_val in (("pi_2", 4.3e-7), ("pi", 4.1e-7), ("3pi_2", 4.5e-7)):
        results.append(
            {
                "ruleId": "PH-SYM-001",
                "level": "note",
                "message": {"text": f"eps_pos_rms={eps_val:.3e} (transform=rotation {ang_str})"},
                "properties": {
                    "transform_kind": "rotation",
                    "transform_param": ang_str,
                    "traj_index": 0,
                    "eps_pos_rms": eps_val,
                    "eps_t_npz_filename": f"eps_PH-SYM-001_rotation_{ang_str}_traj00.npz",
                },
            }
        )
    results.append(
        {
            "ruleId": "PH-SYM-001",
            "level": "note",
            "message": {"text": "eps_pos_rms=0.000e+00 (transform=identity 0)"},
            "properties": {
                "transform_kind": "identity",
                "transform_param": "0",
                "traj_index": 0,
                "eps_pos_rms": 0.0,
                "eps_t_npz_filename": "eps_PH-SYM-001_identity_0_traj00.npz",
            },
        }
    )
    results.append(
        {
            "ruleId": "PH-SYM-002",
            "level": "note",
            "message": {"text": "eps_pos_rms=4.400e-07 (transform=reflection y_axis)"},
            "properties": {
                "transform_kind": "reflection",
                "transform_param": "y_axis",
                "traj_index": 0,
                "eps_pos_rms": 4.4e-7,
                "eps_t_npz_filename": "eps_PH-SYM-002_reflection_y_axis_traj00.npz",
            },
        }
    )
    results.append(
        {
            "ruleId": "PH-SYM-004",
            "level": "note",
            "message": {"text": "eps_pos_rms=1.500e-15 (transform=translation L_3_L_7)"},
            "properties": {
                "transform_kind": "translation",
                "transform_param": "L_3_L_7",
                "traj_index": 0,
                "eps_pos_rms": 1.5e-15,
                "eps_t_npz_filename": "eps_PH-SYM-004_translation_L_3_L_7_traj00.npz",
            },
        }
    )
    if with_skip:
        results.append(
            {
                "ruleId": "PH-SYM-003",
                "level": "note",
                "message": {
                    "text": (
                        "SKIP: PBC-square breaks SO(2) symmetry — "
                        "rotated cell doesn't tile with original"
                    )
                },
                "properties": {
                    "transform_kind": "skip",
                    "transform_param": "so2_continuous",
                    "traj_index": 0,
                    "eps_pos_rms": None,
                    "skip_reason": (
                        "PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original"
                    ),
                    "eps_t_npz_filename": "eps_PH-SYM-003_skip_so2_continuous_traj00.npz",
                },
            }
        )
    return {
        "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "physics-lint-rollout-anchor-harness",
                        "version": "0.1.0",
                    }
                },
                "results": results,
                "properties": {
                    "source": "rollout-anchor-harness",
                    "harness_sarif_schema_version": "1.1",
                    "physics_lint_sha_pkl_inference": "8c3d080000",
                    "physics_lint_sha_npz_conversion": "5857144000",
                    "physics_lint_sha_eps_computation": "d9a8baa000",
                    "physics_lint_sha_sarif_emission": "d9a8baa000",
                    "lagrangebench_sha": "ee0001eeee",
                    "checkpoint_id": f"{model}_tgv2d/best",
                    "model_name": model,
                    "dataset_name": "tgv2d",
                    "rollout_subdir": f"rollouts/{model}_tgv2d_post_d03df3e",
                },
            }
        ],
    }


def test_render_eps_table_reads_v1_1_sarif_pair(tmp_path):
    segnn_path = tmp_path / "segnn_tgv2d_eps.sarif"
    gns_path = tmp_path / "gns_tgv2d_eps.sarif"
    segnn_path.write_text(json.dumps(_v1_1_sarif_fixture(model="segnn")))
    gns_path.write_text(json.dumps(_v1_1_sarif_fixture(model="gns")))

    output = render_eps_table(segnn_sarif_path=segnn_path, gns_sarif_path=gns_path)

    assert "Architectural-evidence rows" in output
    assert "Construction-trivial rows" in output
    assert "Substrate-incompatible SKIP" in output
    assert "PH-SYM-001" in output
    assert "PH-SYM-002" in output
    assert "PH-SYM-003" in output
    assert "PH-SYM-004" in output


def test_render_eps_table_fails_loud_on_v1_0_input(tmp_path):
    v1_0_sarif = _v1_1_sarif_fixture(model="segnn")
    v1_0_sarif["runs"][0]["properties"]["harness_sarif_schema_version"] = "1.0"

    sarif_path = tmp_path / "v1_0.sarif"
    sarif_path.write_text(json.dumps(v1_0_sarif))

    with pytest.raises(SchemaVersionMismatchError, match=r"expected 1\.1, got 1\.0"):
        render_eps_table(segnn_sarif_path=sarif_path, gns_sarif_path=sarif_path)


def test_render_eps_table_classifies_rows_correctly(tmp_path):
    """Architectural rows include rotation+reflection; construction-trivial
    includes identity+translation; SKIP for PH-SYM-003."""
    segnn_path = tmp_path / "segnn_tgv2d_eps.sarif"
    gns_path = tmp_path / "gns_tgv2d_eps.sarif"
    segnn_path.write_text(json.dumps(_v1_1_sarif_fixture(model="segnn")))
    gns_path.write_text(json.dumps(_v1_1_sarif_fixture(model="gns")))

    output = render_eps_table(segnn_sarif_path=segnn_path, gns_sarif_path=gns_path)

    arch_section = output.split("### Architectural-evidence rows")[1].split("###")[0]
    assert "rotation" in arch_section.lower() or "pi_2" in arch_section
    assert "reflection" in arch_section.lower() or "y_axis" in arch_section
    assert "L_3_L_7" not in arch_section

    trivial_section = output.split("### Construction-trivial rows")[1].split("###")[0]
    assert "L_3_L_7" in trivial_section
    assert "identity" in trivial_section.lower() or "PH-SYM-001" in trivial_section
    assert "PH-SYM-002" not in trivial_section

    skip_section = output.split("### Substrate-incompatible SKIP")[1]
    assert "PH-SYM-003" in skip_section
    assert "SKIP" in skip_section


def test_render_eps_table_verdict_thresholds(tmp_path):
    """PASS / APPROXIMATE / FAIL labels per design §3.3 float32-floor band."""
    segnn = _v1_1_sarif_fixture(model="segnn")
    # Push GNS PH-SYM-001 pi_2 into APPROXIMATE band (1e-3); leave pi at floor.
    gns = _v1_1_sarif_fixture(model="gns")
    for r in gns["runs"][0]["results"]:
        if r["ruleId"] == "PH-SYM-001" and r["properties"]["transform_param"] == "pi_2":
            r["properties"]["eps_pos_rms"] = 1e-3
    segnn_path = tmp_path / "segnn.sarif"
    gns_path = tmp_path / "gns.sarif"
    segnn_path.write_text(json.dumps(segnn))
    gns_path.write_text(json.dumps(gns))

    output = render_eps_table(segnn_sarif_path=segnn_path, gns_sarif_path=gns_path)
    assert "APPROXIMATE" in output
    assert "PASS" in output
    assert "SKIP" in output
