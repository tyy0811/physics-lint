"""Config merge path — TOML + adapter + CLI flags -> validated DomainSpec."""

from pathlib import Path

import pytest

from physics_lint import DomainSpec
from physics_lint.config import load_spec_from_toml, merge_into_spec


def _write_toml(tmp_path: Path, contents: str) -> Path:
    p = tmp_path / "pyproject.toml"
    p.write_text(contents)
    return p


_MINIMAL_LAPLACE_TOML = """
[tool.physics-lint]
pde = "laplace"
grid_shape = [64, 64]
domain = { x = [0.0, 1.0], y = [0.0, 1.0] }
periodic = false
boundary_condition = "dirichlet_homogeneous"

[tool.physics-lint.field]
type = "grid"
backend = "fd"
dump_path = "pred.npz"
"""


def test_load_toml_minimal_laplace(tmp_path: Path):
    path = _write_toml(tmp_path, _MINIMAL_LAPLACE_TOML)
    raw = load_spec_from_toml(path)
    assert raw["pde"] == "laplace"
    assert raw["boundary_condition"]["kind"] == "dirichlet_homogeneous"


def test_merge_cli_overrides(tmp_path: Path):
    path = _write_toml(tmp_path, _MINIMAL_LAPLACE_TOML)
    raw = load_spec_from_toml(path)
    merged = merge_into_spec(raw, adapter_spec=None, cli_overrides={"periodic": True})
    spec = DomainSpec.model_validate(merged)
    assert spec.periodic is True


def test_merge_adapter_overrides_toml(tmp_path: Path):
    path = _write_toml(tmp_path, _MINIMAL_LAPLACE_TOML)
    raw = load_spec_from_toml(path)
    adapter_spec = {
        "pde": "laplace",
        "grid_shape": [32, 32],  # override
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "field": {"type": "grid", "backend": "fd", "dump_path": "pred.npz"},
    }
    merged = merge_into_spec(raw, adapter_spec=adapter_spec, cli_overrides={})
    spec = DomainSpec.model_validate(merged)
    assert spec.grid_shape == (32, 32)


def test_missing_toml_falls_back_to_physics_lint_toml(tmp_path: Path):
    standalone = tmp_path / "physics-lint.toml"
    standalone.write_text(
        _MINIMAL_LAPLACE_TOML.replace("[tool.physics-lint]", "").replace(
            "[tool.physics-lint.field]", "[field]"
        )
    )
    raw = load_spec_from_toml(standalone)
    assert raw["pde"] == "laplace"


def test_invalid_toml_raises_config_error(tmp_path: Path):
    path = _write_toml(
        tmp_path,
        """
        [tool.physics-lint]
        pde = "heat"
        grid_shape = [64, 64]
        domain = { x = [0.0, 1.0], y = [0.0, 1.0] }
        boundary_condition = "dirichlet_homogeneous"
        [tool.physics-lint.field]
        type = "grid"
        dump_path = "p.npz"
        # heat requires diffusivity and a time domain -- missing both
        """,
    )
    raw = load_spec_from_toml(path)
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DomainSpec.model_validate(merge_into_spec(raw, adapter_spec=None, cli_overrides={}))


def test_deep_merge_wholesale_override_on_scalar(tmp_path: Path):
    # Non-dict override replaces wholesale (not merged)
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    out = merge_into_spec(base, adapter_spec={"a": 99}, cli_overrides={})
    assert out == {"a": 99, "b": {"c": 2, "d": 3}}


def test_normalize_symmetries_list_form(tmp_path: Path):
    path = _write_toml(
        tmp_path,
        "[tool.physics-lint]\n"
        'pde = "laplace"\n'
        "grid_shape = [64, 64]\n"
        "domain = { x = [0.0, 1.0], y = [0.0, 1.0] }\n"
        "periodic = false\n"
        'boundary_condition = "dirichlet_homogeneous"\n'
        'symmetries = ["D4", "reflection_x"]\n\n'
        "[tool.physics-lint.field]\n"
        'type = "grid"\n'
        'dump_path = "p.npz"\n',
    )
    raw = load_spec_from_toml(path)
    assert raw["symmetries"] == {"declared": ["D4", "reflection_x"]}


def test_merge_into_spec_empty_overrides_preserves_toml():
    raw = {"pde": "laplace", "grid_shape": [32, 32]}
    out = merge_into_spec(raw, adapter_spec=None, cli_overrides={})
    assert out == raw
