"""Tests for dogfood/run_dogfood_real.py pure functions."""

import subprocess
import sys

import numpy as np
import pytest

from dogfood.run_dogfood_real import (
    aggregate_over_problems,
    apply_rules_to_prediction,
    build_a1_spec,
    check_binary_axis,
    check_ordinal_axis,
    compute_verdict,
    load_predictions,
)


class TestBuildA1Spec:
    def test_returns_laplace_spec_on_64x64_unit_square(self):
        spec = build_a1_spec()
        assert spec.pde == "laplace"
        assert spec.grid_shape == (64, 64)
        assert spec.domain.x == (0.0, 1.0)
        assert spec.domain.y == (0.0, 1.0)
        assert spec.periodic is False

    def test_bc_kind_is_dirichlet_not_homogeneous(self):
        """BCs span [-4.27, 4.28] on the dataset — not homogeneous."""
        spec = build_a1_spec()
        assert spec.boundary_condition.kind == "dirichlet"
        assert spec.boundary_condition.preserves_sign is False

    def test_symmetries_empty(self):
        spec = build_a1_spec()
        assert spec.symmetries.declared == []

    def test_field_source_uses_dummy_path(self):
        """FieldSourceSpec's exactly-one-source validator needs a non-None
        adapter_path or dump_path. Rules we use don't read this; the string
        'unused' is a deliberate sentinel (assertion pins it to catch
        accidental drift, not just any non-None value)."""
        spec = build_a1_spec()
        assert spec.field.type == "grid"
        assert spec.field.backend == "fd"
        assert spec.field.dump_path == "unused"


class TestApplyRulesToPrediction:
    def test_linear_field_res_raw_below_fp32_ceiling(self, a1_spec, linear_field):
        """u = x+y is analytically harmonic (Δu = 0). In float32, FD stencil
        roundoff puts the floor at ~1.3e-3 on 64x64. Tolerance is 1e-2 —
        10x above that floor, well below any real-model residual (DDPM=4.22
        is 3000x this). If raw_value blows past 1e-2, h or backend is wrong.
        """
        scores = apply_rules_to_prediction(
            prediction=linear_field,
            truth=linear_field,  # same field — zero BC error expected
            spec=a1_spec,
        )
        assert scores["PH-RES-001"] < 1e-2

    def test_identity_pred_truth_gives_zero_bc_err(self, a1_spec, linear_field):
        """Identical prediction and truth → zero boundary error."""
        scores = apply_rules_to_prediction(
            prediction=linear_field,
            truth=linear_field,
            spec=a1_spec,
        )
        # PH-BC-001 in relative mode returns err/g ~ 0/g = 0 exactly.
        assert scores["PH-BC-001"] == pytest.approx(0.0, abs=1e-10)

    def test_linear_field_satisfies_max_principle(self, a1_spec, linear_field):
        """u = x+y achieves its extrema on the boundary: min=0 at (0,0),
        max=2 at (1,1). Both are boundary points; bc_min = u.min() and
        bc_max = u.max(), so overshoot = 0 exactly.
        """
        scores = apply_rules_to_prediction(
            prediction=linear_field,
            truth=linear_field,
            spec=a1_spec,
        )
        assert scores["PH-POS-002"] < 1e-5

    def test_returns_three_rule_keys(self, a1_spec, linear_field):
        scores = apply_rules_to_prediction(
            prediction=linear_field,
            truth=linear_field,
            spec=a1_spec,
        )
        assert set(scores.keys()) == {"PH-RES-001", "PH-BC-001", "PH-POS-002"}


class TestCheckOrdinalAxis:
    def test_match_when_rankings_agree(self):
        result = check_ordinal_axis(
            axis_name="pde_residual",
            upstream_ranking=["ddpm", "unet_regressor", "fno"],
            physlint_scores={
                "unet_regressor": {"PH-RES-001": 20.5},
                "fno": {"PH-RES-001": 24.5},
                "ddpm": {"PH-RES-001": 4.2},
            },
            rule_id="PH-RES-001",
        )
        assert result["match"] is True
        assert result["mode"] == "ordinal"
        assert result["physlint"] == ["ddpm", "unet_regressor", "fno"]

    def test_mismatch_when_pair_inverts(self):
        """If physics-lint ranks UNet < FNO by smaller numerical margin and
        an L2/L1 norm difference flips them, the axis match fails."""
        result = check_ordinal_axis(
            axis_name="pde_residual",
            upstream_ranking=["ddpm", "unet_regressor", "fno"],
            physlint_scores={
                "unet_regressor": {"PH-RES-001": 25.0},  # inverted with fno
                "fno": {"PH-RES-001": 20.0},
                "ddpm": {"PH-RES-001": 4.2},
            },
            rule_id="PH-RES-001",
        )
        assert result["match"] is False
        assert result["physlint"] == ["ddpm", "fno", "unet_regressor"]

    def test_returns_metadata_for_report(self):
        result = check_ordinal_axis(
            axis_name="bc_err",
            upstream_ranking=["ddpm", "unet_regressor", "fno"],
            physlint_scores={
                "unet_regressor": {"PH-BC-001": 0.007},
                "fno": {"PH-BC-001": 0.2},
                "ddpm": {"PH-BC-001": 0.001},
            },
            rule_id="PH-BC-001",
        )
        assert result["axis"] == "bc_err"
        assert result["upstream"] == ["ddpm", "unet_regressor", "fno"]


class TestCheckBinaryAxis:
    def test_fno_only_violator_matches_expected(self):
        """Upstream has {FNO} violating; physics-lint also has {FNO} above
        threshold. Match is True."""
        result = check_binary_axis(
            axis_name="max_viol",
            expected_violators={"fno"},
            physlint_scores={
                "unet_regressor": {"PH-POS-002": 0.0},
                "fno": {"PH-POS-002": 0.05},
                "ddpm": {"PH-POS-002": 1e-12},
            },
            rule_id="PH-POS-002",
            threshold=1e-10,
        )
        assert result["match"] is True
        assert result["physlint_violators"] == {"fno"}
        assert result["mode"] == "binary"

    def test_universal_violation_case_surfaces_as_mismatch(self):
        """Threshold-mismatch edge case from §6.5: physics-lint 1e-10 is
        stricter than upstream 1e-6, so all three models might register
        as violators. This should report NO match."""
        result = check_binary_axis(
            axis_name="max_viol",
            expected_violators={"fno"},
            physlint_scores={
                "unet_regressor": {"PH-POS-002": 3e-10},
                "fno": {"PH-POS-002": 0.05},
                "ddpm": {"PH-POS-002": 2e-10},
            },
            rule_id="PH-POS-002",
            threshold=1e-10,
        )
        assert result["match"] is False
        assert result["physlint_violators"] == {"unet_regressor", "fno", "ddpm"}


class TestComputeVerdict:
    def test_pass_scoped_when_sanity_and_both_real_axes_match(self):
        result = compute_verdict(
            sanity_match=True,
            real_axis_matches=[True, True],
        )
        assert result == "PASS (scoped)"

    def test_pass_mixed_when_sanity_and_one_real_axis_matches(self):
        result = compute_verdict(
            sanity_match=True,
            real_axis_matches=[True, False],
        )
        assert result == "PASS (scoped, MIXED)"

    def test_fail_when_sanity_passes_but_both_real_axes_fail(self):
        result = compute_verdict(
            sanity_match=True,
            real_axis_matches=[False, False],
        )
        assert result == "FAIL"

    def test_bug_when_sanity_axis_fails(self):
        """L2-vs-L2 sanity disagreement is a bug, not a deferral."""
        result = compute_verdict(
            sanity_match=False,
            real_axis_matches=[True, True],
        )
        assert result == "BUG"

    def test_bug_verdict_even_if_real_axes_pass(self):
        """A sanity-axis failure always produces BUG regardless of real axes."""
        result = compute_verdict(
            sanity_match=False,
            real_axis_matches=[False, False],
        )
        assert result == "BUG"


class TestExtractPredictionsArgparse:
    def test_help_does_not_require_diffphys(self):
        """--help must work without importing diffphys, since this script
        is called from an env where diffphys may or may not be importable
        depending on venv state at plan-time vs runtime."""
        # Use the same python running the tests — not the diffphys venv —
        # because the argparse surface is separate from the import path.
        # If argparse requires diffphys at import time, this raises
        # ImportError here.
        result = subprocess.run(
            [sys.executable, "dogfood/_extract_predictions.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--model-name" in result.stdout
        assert "--checkpoint" in result.stdout
        assert "--max-samples" in result.stdout


class TestLoadPredictions:
    def test_shape_and_dtype(self, tiny_predictions_npz):
        predictions, truth = load_predictions(tiny_predictions_npz)
        assert predictions.shape == (4, 64, 64)
        assert truth.shape == (4, 64, 64)
        assert predictions.dtype == np.float32
        assert truth.dtype == np.float32


class TestAggregateOverProblems:
    def test_aggregates_mean_across_problems(self, a1_spec, tiny_predictions_npz):
        predictions, truth = load_predictions(tiny_predictions_npz)
        per_model_score = aggregate_over_problems(
            predictions=predictions,
            truth=truth,
            spec=a1_spec,
        )
        assert set(per_model_score.keys()) == {"PH-RES-001", "PH-BC-001", "PH-POS-002"}
        # All four copies of linear harmonic → mean residual sits at the
        # float32 FD stencil floor (~1.3e-3). Tolerance matches Task 3.
        assert per_model_score["PH-RES-001"] < 1e-2
        assert per_model_score["PH-BC-001"] == pytest.approx(0.0, abs=1e-10)
        assert per_model_score["PH-POS-002"] < 1e-5
