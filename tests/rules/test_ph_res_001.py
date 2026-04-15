"""PH-RES-001 — Residual exceeds variationally-correct norm threshold.

Laplace/Poisson path: compute the strong-form residual of the Field against
the configured PDE, take its H^-1 norm via the spectral formula (for periodic
inputs) or a Riesz-lift surrogate (for non-periodic, Week-1 falls back to L2),
divide by the calibrated floor, emit tri-state.
"""

import numpy as np
import pytest

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_res_001


def _laplace_periodic_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [64, 64],
            "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )


def test_ph_res_001_exact_harmonic_is_pass():
    spec = _laplace_periodic_spec()
    # Harmonic: Laplacian is identically zero, so residual norm is ~0
    n = 64
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    xg, _yg = np.meshgrid(x, y, indexing="ij")
    u = np.zeros_like(xg)  # the trivial harmonic
    field = GridField(u, h=(2 * np.pi / n, 2 * np.pi / n), periodic=True)

    result = ph_res_001.check(field, spec)
    assert result.rule_id == "PH-RES-001"
    assert result.status == "PASS"
    assert result.raw_value is not None
    assert result.raw_value < 1e-12


def test_ph_res_001_nonzero_residual_is_warn_or_fail():
    spec = _laplace_periodic_spec()
    n = 64
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    # u = cos(x) cos(y) — Laplacian is -2 cos(x) cos(y), so this is NOT
    # a Laplace solution (residual is nonzero and large).
    u = np.cos(xg) * np.cos(yg)
    field = GridField(u, h=(2 * np.pi / n, 2 * np.pi / n), periodic=True)

    result = ph_res_001.check(field, spec)
    assert result.status in {"WARN", "FAIL"}
    assert result.violation_ratio is not None
    assert result.violation_ratio > 1.0


def test_ph_res_001_metadata():
    assert ph_res_001.__rule_id__ == "PH-RES-001"
    assert ph_res_001.__default_severity__ == "error"
    assert "adapter" in ph_res_001.__input_modes__
    assert "dump" in ph_res_001.__input_modes__


def test_ph_res_001_nonperiodic_fd_l2_fallback():
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [16, 16],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    u = np.zeros((16, 16))
    field = GridField(u, h=1.0 / 15, periodic=False, backend="fd")
    result = ph_res_001.check(field, spec)
    assert result.status == "PASS"
    assert result.recommended_norm == "L2"


def test_ph_res_001_poisson_without_source_is_skipped():
    # Week 2 plumbs Poisson sources through .npz dump metadata, but a spec
    # that's hand-constructed without a source array must still not crash —
    # emit SKIPPED with a clear reason so the rest of the rule suite runs.
    spec = DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [16, 16],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    field = GridField(np.zeros((16, 16)), h=1.0 / 15, periodic=False, backend="fd")
    result = ph_res_001.check(field, spec)
    assert result.status == "SKIPPED"
    assert "source" in (result.reason or "").lower()


def test_ph_res_001_poisson_with_source_plumbed_through_dump(tmp_path):
    """PH-RES-001 Poisson path with source plumbed through .npz dump metadata."""
    from physics_lint.analytical import poisson as poisson_sols
    from physics_lint.loader import load_target

    sol = poisson_sols.periodic_sin_sin()
    n = 64
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    u = sol.u(X, Y)
    f = sol.source(X, Y)

    metadata = {
        "pde": "poisson",
        "grid_shape": [n, n],
        "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi]},
        "periodic": True,
        "boundary_condition": {"kind": "periodic"},
        "field": {"type": "grid", "backend": "spectral"},
    }
    path = tmp_path / "pred.npz"
    np.savez(path, prediction=u, metadata=metadata, source=f)

    loaded = load_target(path, cli_overrides={}, toml_path=None)
    result = ph_res_001.check(loaded.field, loaded.spec)
    assert result.status == "PASS"
    assert result.recommended_norm == "H-1"


def test_ph_res_001_poisson_source_term_via_adapter(tmp_path):
    """Finding 2 regression: source_term on a validated spec is plumbed from disk.

    Previously only .npz dumps with an embedded 'source' key could feed
    the Poisson residual — an adapter module setting
    domain_spec()['source_term'] = 'source.npy' would validate but then
    skip PH-RES-001 because the loader never read the file. Now the
    loader resolves source_term relative to the adapter's directory.
    """
    from physics_lint.analytical import poisson as poisson_sols
    from physics_lint.loader import load_target

    sol = poisson_sols.periodic_sin_sin()
    n = 64
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    u = sol.u(X, Y)
    f = sol.source(X, Y)

    # Persist the source as a plain .npy next to the adapter.
    np.save(tmp_path / "source.npy", f)
    # Persist the prediction as a .npz next to it — adapter returns the
    # *values* directly via a closure over this file. (A hand-written
    # adapter for a real model would call the model instead.)
    np.save(tmp_path / "pred.npy", u)

    adapter_path = tmp_path / "poisson_adapter.py"
    adapter_path.write_text(
        "import numpy as np\n"
        "from pathlib import Path\n"
        "import torch\n"
        "_HERE = Path(__file__).parent\n"
        "_U = np.load(_HERE / 'pred.npy')\n"
        "def load_model():\n"
        "    def fn(coords):\n"
        "        return torch.from_numpy(_U).unsqueeze(-1)\n"
        "    return fn\n"
        "def domain_spec():\n"
        "    return {\n"
        "        'pde': 'poisson',\n"
        "        'grid_shape': [64, 64],\n"
        "        'domain': {'x': [0.0, 6.283185307179586], 'y': [0.0, 6.283185307179586]},\n"
        "        'periodic': True,\n"
        "        'boundary_condition': {'kind': 'periodic'},\n"
        "        'field': {'type': 'callable', 'backend': 'spectral'},\n"
        "        'source_term': 'source.npy',\n"
        "    }\n"
    )

    loaded = load_target(adapter_path, cli_overrides={}, toml_path=None)
    # Source array should have been plumbed via source_term.
    injected = getattr(loaded.spec, "_source_array", None)
    assert injected is not None
    assert injected.shape == (64, 64)

    # Materialize the callable onto a GridField (the rule requires GridField)
    # and run PH-RES-001. Periodic spectral path gives an exact H^-1 zero.
    from physics_lint.field import GridField as _GridField

    grid_field = _GridField(
        u, h=(2 * np.pi / 64, 2 * np.pi / 64), periodic=True, backend="spectral"
    )
    result = ph_res_001.check(grid_field, loaded.spec)
    assert result.status == "PASS"
    assert result.recommended_norm == "H-1"


def test_ph_res_001_poisson_source_term_missing_file_errors(tmp_path):
    from physics_lint.loader import LoaderError, load_target

    # Bare .npy prediction + a TOML that points at a source file that
    # doesn't exist. The loader should raise LoaderError with the resolved
    # path in the message, not pass a dangling pointer to the rule.
    np.save(tmp_path / "pred.npy", np.zeros((16, 16)))
    toml_path = tmp_path / "pyproject.toml"
    toml_path.write_text(
        "[tool.physics-lint]\n"
        'pde = "poisson"\n'
        "grid_shape = [16, 16]\n"
        "domain = { x = [0.0, 1.0], y = [0.0, 1.0] }\n"
        "periodic = false\n"
        'boundary_condition = { kind = "dirichlet_homogeneous" }\n'
        'source_term = "does_not_exist.npy"\n'
        "\n[tool.physics-lint.field]\n"
        'type = "grid"\n'
        'backend = "fd"\n'
        'dump_path = "pred.npy"\n'
    )
    with pytest.raises(LoaderError, match="source_term"):
        load_target(tmp_path / "pred.npy", cli_overrides={}, toml_path=toml_path)


def test_ph_res_001_poisson_source_shape_mismatch_skipped(tmp_path):
    """Source array with wrong shape -> SKIPPED, not crash."""
    from physics_lint.loader import load_target

    n = 32
    u = np.zeros((n, n))
    wrong_source = np.zeros((n + 2, n))  # off-by-a-halo shape mismatch

    metadata = {
        "pde": "poisson",
        "grid_shape": [n, n],
        "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi]},
        "periodic": True,
        "boundary_condition": {"kind": "periodic"},
        "field": {"type": "grid", "backend": "spectral"},
    }
    path = tmp_path / "pred.npz"
    np.savez(path, prediction=u, metadata=metadata, source=wrong_source)

    loaded = load_target(path, cli_overrides={}, toml_path=None)
    result = ph_res_001.check(loaded.field, loaded.spec)
    assert result.status == "SKIPPED"
    assert "shape" in (result.reason or "").lower()


def test_ph_res_001_rejects_non_gridfield():
    import torch

    from physics_lint import CallableField

    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, 4),
            torch.linspace(0.0, 1.0, 4),
            indexing="ij",
        ),
        dim=-1,
    )
    field = CallableField(
        lambda x: x[..., 0].unsqueeze(-1),
        sampling_grid=grid,
        h=(1.0 / 3, 1.0 / 3),
    )
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [4, 4],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "field": {"type": "callable", "backend": "fd", "adapter_path": "x.py"},
        }
    )
    with pytest.raises(TypeError, match="requires a GridField"):
        ph_res_001.check(field, spec)
