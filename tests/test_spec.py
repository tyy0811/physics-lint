"""DomainSpec — pydantic v2 hierarchy tests."""

import warnings

import pytest
from pydantic import ValidationError

from physics_lint import DomainSpec
from physics_lint.spec import BCSpec, FieldSourceSpec, GridDomain, SARIFSpec, SymmetrySpec


def _valid_heat_dict() -> dict:
    return {
        "pde": "heat",
        "grid_shape": [64, 64, 32],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "symmetries": {"declared": ["D4"]},
        "field": {"type": "grid", "backend": "fd", "dump_path": "pred.npz"},
        "diffusivity": 0.01,
    }


def test_heat_valid_config_roundtrips():
    spec = DomainSpec.model_validate(_valid_heat_dict())
    assert spec.pde == "heat"
    assert spec.diffusivity == 0.01
    assert spec.domain.is_time_dependent is True


def test_heat_without_diffusivity_raises():
    cfg = _valid_heat_dict()
    cfg["diffusivity"] = None
    with pytest.raises(ValidationError, match="diffusivity"):
        DomainSpec.model_validate(cfg)


def test_heat_without_time_domain_raises():
    cfg = _valid_heat_dict()
    cfg["domain"] = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    with pytest.raises(ValidationError, match="time domain"):
        DomainSpec.model_validate(cfg)


def test_wave_requires_wave_speed():
    cfg = _valid_heat_dict()
    cfg["pde"] = "wave"
    cfg["diffusivity"] = None
    cfg["wave_speed"] = None
    with pytest.raises(ValidationError, match="wave_speed"):
        DomainSpec.model_validate(cfg)


def test_bcspec_computed_properties():
    assert BCSpec(kind="periodic").conserves_mass is True
    assert BCSpec(kind="periodic").conserves_energy is True
    assert BCSpec(kind="periodic").preserves_sign is True
    assert BCSpec(kind="dirichlet_homogeneous").conserves_mass is False
    assert BCSpec(kind="dirichlet_homogeneous").conserves_energy is True
    assert BCSpec(kind="dirichlet_homogeneous").preserves_sign is True
    assert BCSpec(kind="dirichlet").conserves_mass is False
    assert BCSpec(kind="dirichlet").preserves_sign is False
    assert BCSpec(kind="neumann_homogeneous").conserves_mass is True


def test_symmetry_spec_accepts_both_c4_and_d4():
    ss = SymmetrySpec(declared=["C4", "D4", "reflection_x"])
    assert "C4" in ss.declared
    assert "D4" in ss.declared


def test_field_source_exactly_one_source():
    with pytest.raises(ValidationError, match="Exactly one"):
        FieldSourceSpec(type="grid", adapter_path=None, dump_path=None)
    with pytest.raises(ValidationError, match="Exactly one"):
        FieldSourceSpec(type="grid", adapter_path="a.py", dump_path="b.npz")
    # These should succeed:
    FieldSourceSpec(type="grid", adapter_path="a.py")
    FieldSourceSpec(type="grid", dump_path="b.npz")


def test_d4_on_non_square_domain_warns():
    cfg = _valid_heat_dict()
    cfg["domain"]["y"] = [0.0, 2.0]  # non-square
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        DomainSpec.model_validate(cfg)
        assert any(
            "non-square" in str(warning.message).lower()
            or "not square" in str(warning.message).lower()
            for warning in w
        ), f"Expected a D4-on-non-square warning; got {[str(x.message) for x in w]}"


def test_bcspec_unknown_kind_rejected():
    with pytest.raises(ValidationError):
        BCSpec(kind="wibble")  # type: ignore[arg-type]


def test_gridded_domain_spatial_lengths_and_time_flag():
    d = GridDomain(x=(0.0, 1.0), y=(0.0, 2.5))
    assert d.spatial_lengths == (1.0, 2.5)
    assert d.is_time_dependent is False
    d2 = GridDomain(x=(0.0, 1.0), y=(0.0, 1.0), t=(0.0, 0.5))
    assert d2.is_time_dependent is True


def test_bcspec_neumann_kind_properties():
    b = BCSpec(kind="neumann")
    assert b.conserves_mass is False  # only neumann_homogeneous conserves mass
    assert b.preserves_sign is False
    assert b.conserves_energy is False


def test_bcspec_neumann_homogeneous_conserves_energy():
    assert BCSpec(kind="neumann_homogeneous").conserves_energy is True
    assert BCSpec(kind="neumann_homogeneous").preserves_sign is False


def test_sarif_spec_accepts_all_none():
    # All fields optional; default instance should validate.
    s = SARIFSpec()
    assert s.source_file is None
    assert s.pde_line is None


def test_sarif_spec_with_fields():
    s = SARIFSpec(source_file="pyproject.toml", pde_line=10, bc_line=12, symmetry_line=14)
    assert s.source_file == "pyproject.toml"
    assert s.pde_line == 10


def test_domain_spec_with_sarif():
    cfg = _valid_heat_dict()
    cfg["sarif"] = {"source_file": "pyproject.toml", "pde_line": 10}
    spec = DomainSpec.model_validate(cfg)
    assert spec.sarif is not None
    assert spec.sarif.source_file == "pyproject.toml"


def test_symmetries_not_d4_or_c4_skips_square_check():
    # Exercise the early-return path of symmetries_compatible_with_domain
    # when no D4/C4 is declared.
    cfg = _valid_heat_dict()
    cfg["symmetries"] = {"declared": ["reflection_x"]}
    cfg["domain"]["y"] = [0.0, 2.0]  # non-square, should NOT warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        DomainSpec.model_validate(cfg)
        non_square_warnings = [
            x
            for x in w
            if "non-square" in str(x.message).lower() or "not square" in str(x.message).lower()
        ]
        assert non_square_warnings == []


def test_grid_shape_length_validation():
    cfg = _valid_heat_dict()
    cfg["grid_shape"] = [64]  # too short
    with pytest.raises(ValidationError):
        DomainSpec.model_validate(cfg)
    cfg2 = _valid_heat_dict()
    cfg2["grid_shape"] = [64, 64, 32, 16]  # too long
    with pytest.raises(ValidationError):
        DomainSpec.model_validate(cfg2)


def test_field_source_backend_defaults_auto():
    # Spec says default backend is "auto"
    f = FieldSourceSpec(type="grid", dump_path="x.npz")
    assert f.backend == "auto"
