import numpy as np
import pytest

skfem = pytest.importorskip("skfem")
from physics_lint.field import MeshVectorField  # noqa: E402


def _unit_square():
    """4 corners of [0,1]^2 + 2 triangles."""
    node_positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return node_positions, cells


def test_constructs_basis_and_holds_velocity():
    nodes, cells = _unit_square()
    velocity = np.zeros((3, 4, 2), dtype=np.float32)  # T=3, N=4, D=2
    f = MeshVectorField(node_positions=nodes, cells=cells, velocity=velocity)
    assert f.values().shape == (3, 4, 2)
    assert f.basis.N == 4  # one P1 DOF per node


def test_rejects_3d_velocity():
    nodes, cells = _unit_square()
    with pytest.raises(ValueError, match="2D"):
        MeshVectorField(node_positions=nodes, cells=cells, velocity=np.zeros((1, 4, 3)))


def test_rejects_quad_cells():
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    cells = np.array([[0, 1, 2, 3]], dtype=np.int64)  # (M, 4) — not triangles
    with pytest.raises(ValueError, match="triangl"):
        MeshVectorField(node_positions=nodes, cells=cells, velocity=np.zeros((1, 4, 2)))


def test_scalar_abc_methods_stub():
    nodes, cells = _unit_square()
    f = MeshVectorField(node_positions=nodes, cells=cells, velocity=np.zeros((1, 4, 2)))
    for method in ("at", "grad", "laplacian", "integrate", "values_on_boundary"):
        with pytest.raises(NotImplementedError):
            getattr(f, method)() if method != "at" else f.at(np.zeros((1, 2)))
