"""Hybrid adapter+dump loader tests.

Extension dispatch per design doc §5.1:
    .py         -> adapter path (exec + load_model + domain_spec)
    .npz / .npy -> dump path (np.load + metadata dict -> DomainSpec)
    .pt / .pth  -> error
"""

from pathlib import Path

import pytest

from physics_lint import DomainSpec, GridField
from physics_lint.loader import LoaderError, load_target


def test_load_adapter(tmp_path: Path, monkeypatch):
    import shutil

    fixture_src = Path(__file__).parent / "fixtures" / "good_adapter.py"
    adapter_copy = tmp_path / "physics_lint_adapter.py"
    shutil.copy(fixture_src, adapter_copy)

    loaded = load_target(adapter_copy, cli_overrides={}, toml_path=None)
    assert isinstance(loaded.spec, DomainSpec)
    assert loaded.spec.pde == "laplace"
    assert loaded.model is not None  # adapter's load_model() returned a callable
    assert callable(loaded.model)


def test_load_dump(tmp_path: Path):
    from tests.fixtures.good_dump import write_good_dump

    dump_path = write_good_dump(tmp_path / "pred.npz")

    loaded = load_target(dump_path, cli_overrides={}, toml_path=None)
    assert isinstance(loaded.spec, DomainSpec)
    assert loaded.spec.pde == "laplace"
    assert loaded.model is None  # dump mode: no callable
    assert isinstance(loaded.field, GridField)


def test_load_pt_file_errors(tmp_path: Path):
    p = tmp_path / "model.pt"
    p.write_bytes(b"\x80\x04")  # fake torch pickle header
    with pytest.raises(LoaderError, match=r"adapter or convert to \.npz"):
        load_target(p, cli_overrides={}, toml_path=None)


def test_load_unknown_extension_errors(tmp_path: Path):
    p = tmp_path / "model.bin"
    p.write_bytes(b"")
    with pytest.raises(LoaderError, match="unsupported"):
        load_target(p, cli_overrides={}, toml_path=None)


def test_load_adapter_missing_file_errors(tmp_path: Path):
    with pytest.raises(LoaderError, match="adapter file not found"):
        load_target(tmp_path / "nonexistent.py", cli_overrides={}, toml_path=None)


def test_load_adapter_missing_load_model_errors(tmp_path: Path):
    bad = tmp_path / "bad_adapter.py"
    bad.write_text("def domain_spec(): pass\n")
    with pytest.raises(LoaderError, match="load_model"):
        load_target(bad, cli_overrides={}, toml_path=None)


def test_load_adapter_missing_domain_spec_errors(tmp_path: Path):
    bad = tmp_path / "bad_adapter.py"
    bad.write_text("def load_model(): pass\n")
    with pytest.raises(LoaderError, match="domain_spec"):
        load_target(bad, cli_overrides={}, toml_path=None)


def test_load_adapter_exec_failure_errors(tmp_path: Path):
    bad = tmp_path / "raises_at_import.py"
    bad.write_text("raise RuntimeError('boom')\n")
    with pytest.raises(LoaderError, match="raised during import"):
        load_target(bad, cli_overrides={}, toml_path=None)


def test_load_adapter_load_model_raises(tmp_path: Path):
    bad = tmp_path / "raising_load_model.py"
    bad.write_text("def load_model(): raise RuntimeError('nope')\ndef domain_spec(): pass\n")
    with pytest.raises(LoaderError, match="load_model"):
        load_target(bad, cli_overrides={}, toml_path=None)


def test_load_adapter_domain_spec_raises(tmp_path: Path):
    bad = tmp_path / "raising_domain_spec.py"
    bad.write_text(
        "def load_model(): return lambda x: x\ndef domain_spec(): raise RuntimeError('nope')\n"
    )
    with pytest.raises(LoaderError, match="domain_spec"):
        load_target(bad, cli_overrides={}, toml_path=None)


def test_load_adapter_domain_spec_wrong_type_errors(tmp_path: Path):
    bad = tmp_path / "bad_return_type.py"
    bad.write_text("def load_model(): return lambda x: x\ndef domain_spec(): return 42\n")
    with pytest.raises(LoaderError, match="must return DomainSpec or dict"):
        load_target(bad, cli_overrides={}, toml_path=None)


def test_load_dump_missing_file_errors(tmp_path: Path):
    with pytest.raises(LoaderError, match="dump file not found"):
        load_target(tmp_path / "missing.npz", cli_overrides={}, toml_path=None)


def test_load_dump_missing_prediction_errors(tmp_path: Path):
    import numpy as np

    path = tmp_path / "bad.npz"
    np.savez(path, not_prediction=np.zeros(4))
    with pytest.raises(LoaderError, match="prediction"):
        load_target(path, cli_overrides={}, toml_path=None)


def test_load_target_with_toml_path(tmp_path: Path):
    # Exercise the toml_path parameter path (covers lines where toml_spec is populated).
    from tests.fixtures.good_dump import write_good_dump

    dump_path = write_good_dump(tmp_path / "pred.npz")
    toml_path = tmp_path / "pyproject.toml"
    toml_path.write_text(
        "[tool.physics-lint]\n"
        'pde = "laplace"\n'
        "grid_shape = [32, 32]\n"
        "domain = { x = [0.0, 1.0], y = [0.0, 1.0] }\n"
        "periodic = false\n"
        'boundary_condition = "dirichlet"\n\n'
        "[tool.physics-lint.field]\n"
        'type = "grid"\n'
        'backend = "fd"\n'
        'dump_path = "pred.npz"\n'
    )
    loaded = load_target(dump_path, cli_overrides={}, toml_path=toml_path)
    assert loaded.spec.pde == "laplace"


def test_build_sampling_grid_periodic_path(tmp_path: Path):
    # Build an adapter spec with periodic=True to exercise the periodic
    # branch of _compute_h_from_spec and _build_sampling_grid.
    bad = tmp_path / "periodic_adapter.py"
    bad.write_text(
        "import torch\n"
        "def load_model():\n"
        "    return lambda x: (x[..., 0] + x[..., 1]).unsqueeze(-1)\n"
        "def domain_spec():\n"
        "    return {\n"
        '        "pde": "laplace",\n'
        '        "grid_shape": [16, 16],\n'
        '        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},\n'
        '        "periodic": True,\n'
        '        "boundary_condition": {"kind": "periodic"},\n'
        '        "field": {"type": "callable", "adapter_path": "' + str(bad) + '"},\n'
        "    }\n"
    )
    loaded = load_target(bad, cli_overrides={}, toml_path=None)
    assert loaded.spec.periodic is True
    assert loaded.field is not None
