import numpy as np
import pytest

pytest.importorskip("skfem")
from physics_lint.field import MeshVectorField
from physics_lint.loader import load_target


def _write_mesh_vector_npz(path):
    np.savez(
        path,
        node_positions=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        cells=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        velocity=np.zeros((2, 4, 2), dtype=np.float32),
        metadata=np.array(
            {"pde": "incompressible_ns", "field": {"type": "mesh_vector"}}, dtype=object
        ),
    )


def test_loads_mesh_vector_dump(tmp_path):
    npz = tmp_path / "rollout.npz"
    _write_mesh_vector_npz(npz)
    loaded = load_target(npz, cli_overrides={}, toml_path=None)
    assert isinstance(loaded.field, MeshVectorField)
    assert loaded.spec.field.type == "mesh_vector"
    assert loaded.field.values().shape == (2, 4, 2)


def test_mesh_vector_dump_without_dt_loads(tmp_path):
    """dt is schema-optional; PH-CON-005 ignores it."""
    npz = tmp_path / "rollout.npz"
    _write_mesh_vector_npz(npz)
    loaded = load_target(npz, cli_overrides={}, toml_path=None)
    assert isinstance(loaded.field, MeshVectorField)


def test_grid_dump_with_mesh_keys_is_not_misclassified(tmp_path):
    """A grid dump (has 'prediction') that also carries node_positions/cells/
    velocity must load as a GridField, not a MeshVectorField. Detection requires
    'prediction' absent so a grid dump cannot misfire into the mesh branch."""
    from physics_lint.field import GridField

    npz = tmp_path / "g.npz"
    np.savez(
        npz,
        prediction=np.zeros((8, 8), dtype=np.float32),
        node_positions=np.zeros((4, 2), dtype=np.float32),
        cells=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        velocity=np.zeros((1, 4, 2), dtype=np.float32),
        metadata=np.array(
            {
                "pde": "laplace",
                "grid_shape": [8, 8],
                "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
                "boundary_condition": "dirichlet",
                "field": {"type": "grid", "backend": "fd"},
            },
            dtype=object,
        ),
    )
    loaded = load_target(npz, cli_overrides={}, toml_path=None)
    assert isinstance(loaded.field, GridField)
    assert loaded.spec.field.type == "grid"


def test_mesh_vector_adapter_rejected_with_clear_error(tmp_path):
    """Adapter mode does not support mesh_vector in v1.2.0 (the committed surface
    is the dump). A mesh_vector adapter must raise a clear LoaderError, not crash
    on a None grid domain in the CallableField sampling path."""
    from physics_lint.loader import LoaderError

    adapter = tmp_path / "mv_adapter.py"
    adapter.write_text(
        "def load_model():\n"
        "    return lambda x: x\n"
        "def domain_spec():\n"
        "    return {'pde': 'incompressible_ns', 'field': {'type': 'mesh_vector'}}\n"
    )
    with pytest.raises(LoaderError, match="mesh_vector"):
        load_target(adapter, cli_overrides={}, toml_path=None)
