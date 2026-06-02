import numpy as np
import pytest

pytest.importorskip("skfem")
from physics_lint.field import GridField, MeshVectorField
from physics_lint.rules import ph_con_005
from physics_lint.spec import DomainSpec


def _mesh(velocity_fn):
    # Refined unit square so a linear field has interior elements.
    import skfem

    m = skfem.MeshTri().refined(3)
    nodes = m.p.T.astype(np.float64)  # (N, 2)
    cells = m.t.T.astype(np.int64)  # (M, 3)
    vel = np.stack([velocity_fn(nodes)], axis=0).astype(np.float32)  # (1, N, 2)
    return MeshVectorField(node_positions=nodes, cells=cells, velocity=vel)


def _ns_spec():
    return DomainSpec.model_validate(
        {"pde": "incompressible_ns", "field": {"type": "mesh_vector", "dump_path": "x.npz"}}
    )


def test_divergence_free_field_near_zero():
    f = _mesh(lambda p: np.stack([p[:, 0], -p[:, 1]], axis=1))  # v=(x, -y)
    res = ph_con_005.check(f, _ns_spec())
    assert res.status == "PASS"
    assert res.severity == "info"
    assert res.raw_value < 1e-6


def test_known_divergent_field_defect_one():
    f = _mesh(lambda p: np.stack([p[:, 0], np.zeros(len(p))], axis=1))  # v=(x, 0)
    res = ph_con_005.check(f, _ns_spec())
    assert res.status == "PASS" and res.severity == "info"
    assert abs(res.raw_value - 1.0) < 1e-6


def test_skips_on_grid_field():
    g = GridField(np.zeros((8, 8)), h=(0.1, 0.1), periodic=False, backend="fd")
    res = ph_con_005.check(g, _ns_spec())
    assert res.status == "SKIPPED"
    assert "MeshVectorField" in res.reason
