"""Tests for dogfood/run_dogfood_real.py pure functions."""

from dogfood.run_dogfood_real import build_a1_spec


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
