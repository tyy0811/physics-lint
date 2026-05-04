"""Mesh-side read-only-path tests — Day 0.5 follow-up.

Per the post-Gate-B review hand-off: exercises the rollout-level
surface of `mesh_rollout_adapter.py` (``MeshRollout``,
``load_mesh_rollout_npz``, ``save_mesh_rollout_npz``,
``mass_conservation_defect_on_mesh``, ``energy_drift_on_mesh``,
``dissipation_sign_violation_on_mesh``) against the synthetic NS
channel-flow builders in ``synthetic_rollouts.py``.

Mirrors the particle-side ``test_read_only_path.py`` shape:

- Round-trip save / load of `mesh_rollout.npz`.
- Defect-on-conservative-fixture = 0.
- Defect-on-violation-fixture > 0.
- Skip-with-reason on graph-mesh rollouts (Day 2 audit gate).
- Skip-with-reason on missing 'velocity' field.

Framing per Day 0.5 review: this is a Gate-B-style symmetry pass for
the mesh side — not a pre-registered gate (the v3 plan does not list
one for the mesh side), but a free confidence boost that makes the
mesh-harness behaviour unambiguous before Day 2 / Modal session
begins.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
    MeshRollout,
    dissipation_sign_violation_on_mesh,
    energy_drift_on_mesh,
    kinetic_energy_series_on_mesh,
    load_mesh_rollout_npz,
    mass_conservation_defect_on_mesh,
    save_mesh_rollout_npz,
)
from external_validation._rollout_anchors._harness.tests.synthetic_rollouts import (
    build_divergence_violation_channel,
    build_graph_mesh_skip_case,
    build_uniform_channel_flow,
)

# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_save_load_mesh_rollout_npz_round_trip(tmp_path):
    """save_mesh_rollout_npz then load_mesh_rollout_npz reconstruct the dataclass."""
    case = build_uniform_channel_flow(nx=8, ny=4, n_timesteps=5)
    saved = save_mesh_rollout_npz(case.rollout, tmp_path / "mesh.npz")
    assert saved.exists()
    reloaded = load_mesh_rollout_npz(saved)
    np.testing.assert_allclose(reloaded.node_positions, case.rollout.node_positions, atol=1e-5)
    np.testing.assert_array_equal(reloaded.node_type, case.rollout.node_type)
    assert reloaded.node_values.keys() == case.rollout.node_values.keys()
    np.testing.assert_allclose(
        reloaded.node_values["velocity"],
        case.rollout.node_values["velocity"],
        atol=1e-5,
    )
    assert reloaded.dt == case.rollout.dt
    assert reloaded.metadata.get("model") == "synthetic-uniform-channel"


def test_load_mesh_rollout_npz_missing_field_raises(tmp_path):
    np.savez(tmp_path / "incomplete.npz", node_positions=np.zeros((1, 2)))
    with pytest.raises(KeyError, match="missing required fields"):
        load_mesh_rollout_npz(tmp_path / "incomplete.npz")


# ---------------------------------------------------------------------------
# Uniform channel flow (mass-conservation by construction)
# ---------------------------------------------------------------------------


def test_uniform_channel_mass_conservation_zero():
    """∂u/∂x = 0, ∂v/∂y = 0 ⇒ ∇·v = 0 ⇒ defect ~ machine epsilon."""
    case = build_uniform_channel_flow()
    result = mass_conservation_defect_on_mesh(case.rollout)
    assert result.skip_reason is None, f"unexpected skip: {result.skip_reason}"
    assert result.value is not None
    # FD gradient on a constant field gives zero modulo numpy edge handling;
    # allow generous floating-point bound.
    assert result.value < 1e-10, (
        f"uniform-channel mass_conservation_defect_on_mesh={result.value:.6e} "
        f"should be ~ machine epsilon"
    )


def test_uniform_channel_energy_drift_zero():
    case = build_uniform_channel_flow()
    result = energy_drift_on_mesh(case.rollout)
    assert result.skip_reason is None
    assert result.value is not None
    assert result.value < 1e-12, (
        f"uniform-channel energy_drift_on_mesh={result.value:.6e} should be ~ machine epsilon"
    )


def test_uniform_channel_dissipation_zero():
    case = build_uniform_channel_flow()
    result = dissipation_sign_violation_on_mesh(case.rollout)
    assert result.skip_reason is None
    assert result.value is not None
    assert math.isclose(result.value, 0.0, abs_tol=1e-12), (
        f"uniform-channel dissipation_sign_violation_on_mesh={result.value:.6e} should be 0"
    )


# ---------------------------------------------------------------------------
# Divergence-violation channel (∂u/∂x = alpha)
# ---------------------------------------------------------------------------


def test_divergence_violation_mass_conservation_nonzero():
    """∂u/∂x = alpha != 0 ⇒ harness emits a positive defect."""
    case = build_divergence_violation_channel(alpha=0.1)
    result = mass_conservation_defect_on_mesh(case.rollout)
    assert result.skip_reason is None
    assert result.value is not None
    # FD-on-(1 + alpha x) should give exactly alpha to ~ machine eps in the
    # interior. The relative L^2 of (alpha) over (1 + alpha x) is bounded
    # below by alpha / max(1 + alpha x) and above by alpha / min(1 + alpha x).
    # For alpha=0.1, x in (0, 2): the relative defect lands in (0.05, 0.1).
    # Assert a generous range — the test is "non-trivial defect", not
    # "exact value".
    assert 0.01 < result.value < 0.5, (
        f"divergence-violation channel defect={result.value:.6e} outside "
        f"expected (0.01, 0.5) range for alpha=0.1"
    )


def test_divergence_violation_energy_drift_zero():
    """KE is t-independent in this fixture ⇒ drift = 0."""
    case = build_divergence_violation_channel(alpha=0.1)
    result = energy_drift_on_mesh(case.rollout)
    assert result.skip_reason is None
    assert result.value is not None
    assert result.value < 1e-12


def test_divergence_violation_scales_linearly_with_alpha():
    """Doubling alpha should approximately double the harness's emitted defect.

    Sanity check on the FD divergence operator's linearity. Catches
    regressions to a saturating or quadratic response.
    """
    case_a = build_divergence_violation_channel(alpha=0.05)
    case_2a = build_divergence_violation_channel(alpha=0.10)
    eps_a = mass_conservation_defect_on_mesh(case_a.rollout).value
    eps_2a = mass_conservation_defect_on_mesh(case_2a.rollout).value
    assert eps_a is not None and eps_2a is not None
    ratio = eps_2a / eps_a
    # Allow 1.7-2.4x to absorb sub-linear / super-linear curvature.
    assert 1.7 < ratio < 2.4, (
        f"doubling alpha should give ratio ~2x; got {ratio:.3f} "
        f"(eps_a={eps_a:.3e}, eps_2a={eps_2a:.3e})"
    )


# ---------------------------------------------------------------------------
# Graph-mesh skip-with-reason (Day 2 audit gate)
# ---------------------------------------------------------------------------


def test_graph_mesh_mass_conservation_skips_with_audit_reason():
    case = build_graph_mesh_skip_case()
    result = mass_conservation_defect_on_mesh(case.rollout)
    assert result.value is None
    assert result.skip_reason is not None
    assert "graph-topology" in result.skip_reason
    assert "Day 2 hour 1 NGC audit" in result.skip_reason or "D0-03" in result.skip_reason


def test_graph_mesh_energy_drift_skips_with_audit_reason():
    case = build_graph_mesh_skip_case()
    result = energy_drift_on_mesh(case.rollout)
    assert result.value is None
    assert result.skip_reason is not None


def test_graph_mesh_dissipation_skips_with_audit_reason():
    case = build_graph_mesh_skip_case()
    result = dissipation_sign_violation_on_mesh(case.rollout)
    assert result.value is None
    assert result.skip_reason is not None


# ---------------------------------------------------------------------------
# Missing-velocity skip-with-reason
# ---------------------------------------------------------------------------


def test_missing_velocity_skips_with_reason():
    """node_values without 'velocity' triggers SKIP on all three functions."""
    case = build_uniform_channel_flow(nx=4, ny=4, n_timesteps=3)
    # Construct a mesh rollout with only a 'pressure' field, no 'velocity'.
    rollout_no_vel = MeshRollout(
        node_positions=case.rollout.node_positions,
        node_type=case.rollout.node_type,
        node_values={"pressure": np.zeros((3, 16))},
        dt=case.rollout.dt,
        metadata={
            "ckpt_hash": "synthetic",
            "ngc_version": "synthetic",
            "git_sha": "synthetic",
            "dataset": "synthetic",
            "model": "synthetic",
            "framework": "synthetic",
            "framework_version": "0.0.0",
            "resampling_applied": False,
            "regular_grid": True,
            "grid_shape": (4, 4),
        },
    )
    for fn in (
        mass_conservation_defect_on_mesh,
        energy_drift_on_mesh,
        dissipation_sign_violation_on_mesh,
    ):
        result = fn(rollout_no_vel)
        assert result.value is None, (
            f"{fn.__name__}: expected SKIP on missing velocity; got value={result.value!r}"
        )
        assert result.skip_reason is not None
        assert "velocity" in result.skip_reason


# ---------------------------------------------------------------------------
# MeshRollout invariants
# ---------------------------------------------------------------------------


def test_mesh_rollout_post_init_rejects_node_type_mismatch():
    with pytest.raises(ValueError, match="node_type shape"):
        MeshRollout(
            node_positions=np.zeros((10, 2)),
            node_type=np.zeros(5, dtype=np.int32),  # wrong size
            node_values={"velocity": np.zeros((3, 10, 2))},
            dt=0.01,
            metadata={},
        )


def test_mesh_rollout_post_init_rejects_node_values_shape_mismatch():
    with pytest.raises(ValueError, match="second axis"):
        MeshRollout(
            node_positions=np.zeros((10, 2)),
            node_type=np.zeros(10, dtype=np.int32),
            node_values={"velocity": np.zeros((3, 7, 2))},  # wrong N
            dt=0.01,
            metadata={},
        )


def test_is_regular_grid_detection_modes():
    """Regular-grid detection accepts 3 modes per metadata."""
    common = dict(
        node_positions=np.zeros((4, 2)),
        node_type=np.zeros(4, dtype=np.int32),
        node_values={"velocity": np.zeros((3, 4, 2))},
        dt=0.01,
    )
    # framework match
    r1 = MeshRollout(**common, metadata={"framework": "pytorch+neuraloperator"})
    assert r1.is_regular_grid
    # resampling_applied
    r2 = MeshRollout(**common, metadata={"resampling_applied": True})
    assert r2.is_regular_grid
    # explicit regular_grid flag
    r3 = MeshRollout(**common, metadata={"regular_grid": True})
    assert r3.is_regular_grid
    # graph default
    r4 = MeshRollout(**common, metadata={"framework": "pytorch+dgl"})
    assert not r4.is_regular_grid


def test_kinetic_energy_series_shape():
    case = build_uniform_channel_flow(nx=4, ny=4, n_timesteps=11)
    e_series = kinetic_energy_series_on_mesh(case.rollout)
    assert e_series.shape == (11,)
