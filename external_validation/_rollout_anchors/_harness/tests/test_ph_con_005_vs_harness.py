"""Gate-B harness fidelity: the public PH-CON-005 rule reproduces the private
harness `_fe_divergence_defect_max` within the pre-registered tolerance
ε ≤ 1e-4 (SCHEMA.md §4.1). Both paths build the same MeshTri + ElementTriP1
basis and run the same FE divergence kernel, so ε is expected to be ~0 by
design (a port-fidelity guarantee, not an independent cross-method check)."""

import numpy as np
import pytest

pytest.importorskip("skfem")
from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
    MeshRollout,
    _build_graph_mesh_basis,
    _fe_divergence_defect_max,
)
from physics_lint.field import MeshVectorField
from physics_lint.rules import ph_con_005
from physics_lint.spec import DomainSpec


def test_public_rule_reproduces_harness_within_gate_b():
    import skfem

    m = skfem.MeshTri().refined(4)
    nodes, cells = m.p.T.astype(np.float64), m.t.T.astype(np.int64)
    vel = np.stack(
        [np.stack([nodes[:, 0] ** 2, -nodes[:, 0] * nodes[:, 1]], axis=1)], axis=0
    ).astype(np.float32)  # a non-trivial, non-divergence-free field

    # Harness reference.
    rollout = MeshRollout(
        node_positions=nodes,
        node_type=np.zeros(len(nodes), dtype=np.int32),
        node_values={"velocity": vel},
        dt=1.0,
        metadata={"cells_2d": cells, "framework": "pytorch+dgl", "model": "x", "dataset": "y"},
    )
    _, basis = _build_graph_mesh_basis(rollout)
    harness_val = _fe_divergence_defect_max(basis, vel)

    # Public rule.
    f = MeshVectorField(node_positions=nodes, cells=cells, velocity=vel)
    spec = DomainSpec.model_validate(
        {"pde": "incompressible_ns", "field": {"type": "mesh_vector", "dump_path": "x.npz"}}
    )
    public_val = ph_con_005.check(f, spec).raw_value

    assert abs(public_val - harness_val) <= 1e-4
