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
