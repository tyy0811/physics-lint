"""Tests for dogfood/run_dogfood_real.py pure functions."""

import pytest

from dogfood.run_dogfood_real import apply_rules_to_prediction, build_a1_spec


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
        adapter_path or dump_path. Rules we use don't read this; dummy is fine."""
        spec = build_a1_spec()
        assert spec.field.type == "grid"
        assert spec.field.backend == "fd"
        assert spec.field.dump_path is not None


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
