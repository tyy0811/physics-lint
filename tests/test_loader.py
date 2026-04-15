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


def test_load_dump_heat_rejected_until_week_2(tmp_path: Path):
    # Week 1 scope is Laplace/Poisson only (docs/plans/2026-04-14-
    # physics-lint-v1-week-1.md line 13: "no time-dependent PDEs"). A heat
    # dump is a *valid* DomainSpec but the loader can't plumb a time axis
    # through GridField yet, so the loader must reject it with a clear
    # LoaderError instead of crashing inside GridField on a shape mismatch.
    import numpy as np

    dump_path = tmp_path / "heat.npz"
    metadata = {
        "pde": "heat",
        "grid_shape": [8, 8, 4],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.1]},
        "periodic": False,
        "boundary_condition": "dirichlet_homogeneous",
        "field": {"type": "grid", "backend": "fd"},
        "diffusivity": 0.01,
    }
    np.savez(dump_path, prediction=np.zeros((8, 8, 4)), metadata=metadata)
    with pytest.raises(LoaderError, match="Week 2"):
        load_target(dump_path, cli_overrides={}, toml_path=None)


def test_load_adapter_heat_rejected_until_week_2(tmp_path: Path):
    adapter = tmp_path / "heat_adapter.py"
    adapter.write_text(
        "def load_model():\n"
        "    return lambda x: x[..., :1]\n"
        "def domain_spec():\n"
        "    return {\n"
        "        'pde': 'heat',\n"
        "        'grid_shape': [8, 8, 4],\n"
        "        'domain': {'x':[0.0,1.0],'y':[0.0,1.0],'t':[0.0,0.1]},\n"
        "        'periodic': False,\n"
        "        'boundary_condition': {'kind':'dirichlet_homogeneous'},\n"
        "        'field': {'type':'callable','backend':'auto'},\n"
        "        'diffusivity': 0.01,\n"
        "    }\n"
    )
    with pytest.raises(LoaderError, match="Week 2"):
        load_target(adapter, cli_overrides={}, toml_path=None)


def test_load_npy_with_toml_spec(tmp_path: Path):
    # .npy is a bare array — no metadata in the file — so the spec must come
    # from the TOML config. The loader should read np.load() as an ndarray
    # (not an NpzFile) and hand it to GridField with the toml-derived spec.
    import numpy as np

    pred = np.zeros((16, 16))
    npy_path = tmp_path / "pred.npy"
    np.save(npy_path, pred)

    toml_path = tmp_path / "pyproject.toml"
    toml_path.write_text(
        "[tool.physics-lint]\n"
        'pde = "laplace"\n'
        "grid_shape = [16, 16]\n"
        "domain = { x = [0.0, 1.0], y = [0.0, 1.0] }\n"
        "periodic = false\n"
        'boundary_condition = "dirichlet"\n\n'
        "[tool.physics-lint.field]\n"
        'type = "grid"\n'
        'backend = "fd"\n'
        'dump_path = "pred.npy"\n'
    )
    loaded = load_target(npy_path, cli_overrides={}, toml_path=toml_path)
    assert isinstance(loaded.field, GridField)
    assert loaded.spec.pde == "laplace"
    assert loaded.model is None


def test_load_npy_without_spec_errors(tmp_path: Path):
    # .npy carries no metadata; without a TOML or CLI spec the loader must
    # raise a clear LoaderError rather than silently crashing inside pydantic
    # or GridField construction.
    import numpy as np

    npy_path = tmp_path / "pred.npy"
    np.save(npy_path, np.zeros((8, 8)))
    with pytest.raises(LoaderError, match=r"\.npy"):
        load_target(npy_path, cli_overrides={}, toml_path=None)


def test_load_dump_shape_mismatch_errors(tmp_path: Path):
    # A mismatched (prediction.shape vs spec.grid_shape) dump silently picks
    # the wrong h and therefore the wrong calibrated floor, producing
    # numerically meaningless rule results. The loader must reject the
    # mismatch with a clear LoaderError.
    import numpy as np

    dump_path = tmp_path / "bad_shape.npz"
    metadata = {
        "pde": "laplace",
        "grid_shape": [32, 32],  # spec says 32x32
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": "dirichlet_homogeneous",
        "field": {"type": "grid", "backend": "fd"},
    }
    # ...but the prediction is 16x16
    np.savez(dump_path, prediction=np.zeros((16, 16)), metadata=metadata)
    with pytest.raises(LoaderError, match="grid_shape"):
        load_target(dump_path, cli_overrides={}, toml_path=None)


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
