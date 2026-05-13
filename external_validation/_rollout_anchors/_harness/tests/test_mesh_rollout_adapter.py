"""Tests for the mesh rollout adapter's loader-contract helpers.

Lands the Phase-1D code-absorption coverage for case study 02:

- Task 10 (D0-23 v8): ``_expect_velocity`` resolves the NGC vortex-
  shedding velocity-field key. Pattern-B P0 single-instance enumeration:
  the audit (preflight/loader_contract_audit.json V3) confirms the NGC
  key is ``"velocity"`` — same as the legacy LB / synthetic path — so
  no pre-generalization (no helper key list, no metadata pivot). A
  second NGC naming (amendment 1's Ahmed Body) triggers a refactor;
  one instance does not.

- Task 11 (D0-23 v9): ``MGN_DATASET_SYSTEM_CLASS`` and the substrate-
  class dispatch on ``energy_drift_on_mesh`` /
  ``dissipation_sign_violation_on_mesh``. Mirrors the particle-side
  D0-22 dispatch (parallel route, duplicate-logic-drift risk *named*
  in round-codex-4 — not eliminated; stack-agnostic refactor gated
  on case study 03 evidence).

- Task 12 (D0-23 v10): ``_assert_loader_contract_mgn`` defensive
  validation per preflight V1-V18 + the 5 secondary known-unknowns.
  Fires on incoming MGN rollouts before rule kernels consume them.
"""

from __future__ import annotations

import numpy as np
import pytest

from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
    MGN_DATASET_SYSTEM_CLASS,
    MGN_ROLLOUT_CONTRACT_P0,
    MeshRollout,
    _assert_loader_contract_mgn,
    _expect_velocity,
    dissipation_sign_violation_on_mesh,
    energy_drift_on_mesh,
    load_mesh_rollout_npz,
    mass_conservation_defect_on_mesh,
    save_mesh_rollout_npz,
)

# ---------------------------------------------------------------------------
# Task 10 — _expect_velocity NGC key resolution (D0-23 verdict 8)
# ---------------------------------------------------------------------------


def test_expect_velocity_resolves_ngc_velocity_key_for_vortex_shedding() -> None:
    """D0-23 verdict 8: the NGC cylinder_flow dataset emits node-resolved
    velocity under the literal key ``"velocity"`` (preflight
    loader_contract_audit.json V3_field_names: cells, mesh_pos, node_type,
    velocity, pressure). Pattern-B P0 single-instance enumeration —
    confirms the legacy default is the correct NGC key; no helper
    branching required.
    """
    rollout = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",
            "model": "modulus_ns_meshgraphnet",
            "dataset": "vortex_shedding_2d",
        },
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )

    velocity = _expect_velocity(rollout)
    assert isinstance(velocity, np.ndarray), (
        f"_expect_velocity must resolve the NGC velocity key 'velocity' "
        f"(per D0-23 v8); got {type(velocity)}"
    )
    assert velocity.shape == (5, 10, 2)


def test_expect_velocity_still_accepts_legacy_synthetic_path() -> None:
    """Pattern-B P0 discipline: the helper accepts the literal
    ``"velocity"`` key regardless of framework metadata. Single-instance
    enumeration — amendment 1's Ahmed Body is the multi-instance trigger
    that would force a key-list / metadata-pivot refactor.
    """
    rollout = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={"framework": "synthetic"},
    )
    velocity = _expect_velocity(rollout)
    assert isinstance(velocity, np.ndarray)
    assert velocity.shape == (5, 10, 2)


# ---------------------------------------------------------------------------
# Task 11 — MGN_DATASET_SYSTEM_CLASS + substrate-class dispatch (D0-23 v9)
# ---------------------------------------------------------------------------


def test_mgn_dataset_system_class_pins_vortex_shedding_2d() -> None:
    """D0-23 verdict 6: vortex_shedding_2d is pinned as open-driven-
    dissipative (boundary-driven sub-class) per the substrate-class
    smoke (preflight/substrate_class_smoke.json). Empirical-only
    classification per the design §2.2 "classify when you exercise"
    rule.
    """
    assert "vortex_shedding_2d" in MGN_DATASET_SYSTEM_CLASS
    assert MGN_DATASET_SYSTEM_CLASS["vortex_shedding_2d"] == "open-driven-dissipative"


def test_energy_drift_on_mesh_skips_when_open_driven_dissipative() -> None:
    """D0-23 verdict 9 (parallel to D0-22 amendment 1):
    energy_drift_on_mesh SKIPs with reason on open-driven-dissipative
    substrates. The strictly-dissipative-or-conservative assumption
    underpinning energy_drift fails on boundary-driven flows.
    """
    rollout = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int64),
        # KE clears KE_REST_THRESHOLD so D0-08 does NOT fire; the
        # D0-23 substrate-class dispatch must fire instead.
        node_values={"velocity": 10 * np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",
            "model": "modulus_ns_meshgraphnet",
            "dataset": "vortex_shedding_2d",
            # Force is_regular_grid=True so the graph-mesh SKIP doesn't
            # short-circuit before the substrate-class dispatch (which
            # is gated on Phase-2's DGL→MeshField materialization in
            # production code paths).
            "regular_grid": True,
        },
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )

    result = energy_drift_on_mesh(rollout)
    assert result.value is None, (
        "energy_drift_on_mesh must SKIP on open-driven-dissipative; "
        f"got value={result.value}, skip_reason={result.skip_reason}"
    )
    skip_reason = result.skip_reason or ""
    assert "open-driven-dissipative" in skip_reason, (
        f"SKIP reason must cite the substrate class; got: {skip_reason!r}"
    )
    assert "D0-22" in skip_reason or "D0-23" in skip_reason


def test_dissipation_sign_violation_on_mesh_skips_when_open_driven_dissipative() -> None:
    """D0-23 verdict 9 (parallel to D0-22 base gate):
    dissipation_sign_violation_on_mesh SKIPs with reason on open-driven-
    dissipative substrates. dE/dt > 0 over a stretch is physics, not a
    model violation.
    """
    rollout = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": 10 * np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",
            "model": "modulus_ns_meshgraphnet",
            "dataset": "vortex_shedding_2d",
            "regular_grid": True,
        },
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )

    result = dissipation_sign_violation_on_mesh(rollout)
    assert result.value is None, (
        "dissipation_sign_violation_on_mesh must SKIP on open-driven-dissipative; "
        f"got value={result.value}, skip_reason={result.skip_reason}"
    )
    skip_reason = result.skip_reason or ""
    assert "open-driven-dissipative" in skip_reason
    assert "D0-22" in skip_reason or "D0-23" in skip_reason


def test_energy_drift_on_mesh_does_not_skip_when_no_substrate_class_match() -> None:
    """Regression guard: dispatch is dataset-name-conditioned. An unknown
    dataset name (or absence) must NOT trigger the substrate-class SKIP —
    the function proceeds to the KE-rest / compute path as before.
    """
    # Use a regular-grid synthetic fixture so we reach compute. Positions
    # must form a valid 2x5 grid so grid_shape inference succeeds.
    xs, ys = np.meshgrid(np.arange(2), np.arange(5), indexing="ij")
    positions = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    # Constant velocity → KE constant → drift = 0; passes through dispatch
    # path without firing any SKIP.
    velocity_t = np.ones((3, 10, 2), dtype=np.float32)

    rollout = MeshRollout(
        node_positions=positions,
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": velocity_t},
        dt=0.01,
        metadata={
            "framework": "synthetic",
            "regular_grid": True,
            # Note: NO "dataset" key → MGN_DATASET_SYSTEM_CLASS.get("") is None
            # → dispatch does NOT fire.
        },
    )
    result = energy_drift_on_mesh(rollout)
    assert result.value is not None, (
        f"unknown dataset must not trigger substrate-class SKIP; "
        f"got skip_reason={result.skip_reason}"
    )
    # Constant velocity → drift = 0.
    assert result.value == 0.0


# ---------------------------------------------------------------------------
# Task 12 — pre-flight loader-contract assertions (D0-23 v10)
# ---------------------------------------------------------------------------


def _well_formed_mgn_rollout(**overrides) -> MeshRollout:
    """Builder for an MGN-shaped rollout that satisfies all loader-contract
    assertions out of the box. Tests override one field at a time to
    exercise a single assertion in isolation.
    """
    kwargs = dict(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int64),  # all NORMAL → in {0,3,4,5,6}
        node_values={"velocity": np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",
            "model": "modulus_ns_meshgraphnet",
            "dataset": "vortex_shedding_2d",
        },
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    if "node_values" in overrides:
        kwargs["node_values"] = overrides.pop("node_values")
    if "metadata" in overrides:
        kwargs["metadata"] = overrides.pop("metadata")
    kwargs.update(overrides)
    return MeshRollout(**kwargs)


def test_assert_loader_contract_mgn_passes_on_well_formed_rollout() -> None:
    """Sanity: a well-formed NGC-shaped rollout passes all assertions."""
    rollout = _well_formed_mgn_rollout()
    _assert_loader_contract_mgn(rollout)  # no AssertionError


def test_assert_loader_contract_mgn_rejects_fp64_velocity() -> None:
    """Per preflight known-unknown §5.6: velocity must be float32. fp64
    surfaces as a loader-contract violation. Citation:
    vortex_shedding_dataset.py:373 @ 1ca85d65.
    """
    rollout = _well_formed_mgn_rollout(
        node_values={"velocity": np.ones((5, 10, 2), dtype=np.float64)},
    )
    with pytest.raises(AssertionError, match="float32"):
        _assert_loader_contract_mgn(rollout)


def test_assert_loader_contract_mgn_rejects_2d_velocity_shape() -> None:
    """Per preflight V12 + V18: velocity must be 3D (T, N_nodes, D).
    A degenerate 2D shape (T, N) would slip past _expect_velocity's
    lifting branch into the rule kernels with the wrong contract.

    Note: MeshRollout.__post_init__ accepts ndim>=2, so a (T,N) shape
    constructs successfully; the assertion fires on the explicit ndim
    check inside _assert_loader_contract_mgn.
    """
    rollout = _well_formed_mgn_rollout(
        node_values={"velocity": np.ones((5, 10), dtype=np.float32)},
    )
    with pytest.raises(AssertionError, match="3D"):
        _assert_loader_contract_mgn(rollout)


def test_assert_loader_contract_mgn_rejects_node_type_out_of_set() -> None:
    """Per preflight known-unknown §5.7 / V16: node_type ∈ {0,3,4,5,6}.
    The loader's one-hot encoder triggers a downstream RuntimeError on
    out-of-range values; pre-flight surfaces this as a diagnostic
    AssertionError instead.
    """
    rollout = _well_formed_mgn_rollout(
        node_type=np.array([0, 3, 4, 5, 6, 7, 0, 0, 0, 0], dtype=np.int64),  # 7 invalid
    )
    with pytest.raises(AssertionError, match="node_type"):
        _assert_loader_contract_mgn(rollout)


def test_assert_loader_contract_mgn_requires_framework_and_model_and_dataset_metadata() -> None:
    """Metadata must include framework + model + dataset — anchors for
    the v9 substrate-class dispatch and framework-conditioned SKIP paths.
    Phase-1 cross-review Finding 1: `dataset` was previously missing from
    the required set, allowing the dispatch to silently no-op on
    well-shaped-but-incomplete MGN rollouts.
    """
    rollout = _well_formed_mgn_rollout(
        metadata={"framework": "pytorch+dgl", "dataset": "vortex_shedding_2d"},
    )
    with pytest.raises(AssertionError, match="model"):
        _assert_loader_contract_mgn(rollout)


def test_assert_loader_contract_mgn_requires_dataset_metadata() -> None:
    """Phase-1 cross-review Finding 1: `dataset` is load-bearing for the
    v9 substrate-class dispatch (MGN_DATASET_SYSTEM_CLASS). The helper
    must require it so a materializer that forgets to set it cannot fail
    open into rule kernels that emit raw values on an open-driven-
    dissipative substrate.
    """
    rollout = _well_formed_mgn_rollout(
        metadata={"framework": "pytorch+dgl", "model": "modulus_ns_meshgraphnet"},
    )
    with pytest.raises(AssertionError, match="dataset"):
        _assert_loader_contract_mgn(rollout)


def test_assert_loader_contract_mgn_raises_when_velocity_key_absent() -> None:
    """Phase-1 cross-review Finding 2: absence of the "velocity" key in
    an MGN-scoped rollout is a contract failure (D0-23 verdict 8 pins
    the NGC key to literal "velocity"), not an optional rule SKIP.
    Previously this path no-op'd — creating a layered-fail-open where
    _assert_loader_contract_mgn passed silently while _expect_velocity
    skipped downstream, hiding the underlying loader-contract violation
    behind a legitimate-looking SKIP report.
    """
    rollout = _well_formed_mgn_rollout(
        node_values={"pressure": np.ones((5, 10), dtype=np.float32)},
    )
    with pytest.raises(AssertionError, match="velocity"):
        _assert_loader_contract_mgn(rollout)


# ---------------------------------------------------------------------------
# Phase 2 Task 3 — Finding 3 absorption: ``_assert_loader_contract_mgn`` is
# wired into ``load_mesh_rollout_npz`` at the first trusted MGN boundary.
# Tests exercise the helper through the wired path (NPZ round-trip), not
# direct invocation. Scope detection: ``metadata["model"].startswith
# ("modulus_")`` ⇒ MGN contract; everything else (``framework="synthetic"``,
# FNO-on-Darcy, future stacks) bypasses.
# ---------------------------------------------------------------------------


def test_load_mesh_rollout_npz_rejects_mgn_rollout_with_fp64_velocity(tmp_path) -> None:
    """Phase-1 cross-review Finding 3 absorption: an on-disk NPZ that
    identifies as MGN (``metadata["model"]`` starts with ``"modulus_"``)
    but carries fp64 velocity must fail-loud at load. Because
    ``save_mesh_rollout_npz`` unconditionally down-casts to fp32 (the
    intended sanitization path), the test bypasses it and writes the
    NPZ directly with ``np.savez`` — modelling a materializer bug or
    a third-party stack that bypasses the canonical save helper.
    """
    bad_path = tmp_path / "bad_fp64.npz"
    np.savez(
        bad_path,
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int32),
        node_values=np.array(
            {"velocity": np.ones((5, 10, 2), dtype=np.float64)},  # fp64 — contract violation
            dtype=object,
        ),
        dt=np.float64(0.01),
        metadata=np.array(
            {
                "framework": "pytorch+dgl",
                "model": "modulus_ns_meshgraphnet",  # MGN-scoped
                "dataset": "vortex_shedding_2d",
            },
            dtype=object,
        ),
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    with pytest.raises(AssertionError, match="float32"):
        load_mesh_rollout_npz(bad_path)


def test_load_mesh_rollout_npz_rejects_mgn_rollout_with_wrong_velocity_key(tmp_path) -> None:
    """Finding 3 + Finding 2 absorption: an MGN-scoped NPZ with
    ``node_values`` under the wrong key (e.g. ``"u"`` instead of
    ``"velocity"``) fails-loud at load. The helper's velocity-key
    enforcement was deepened in Finding 2; this verifies the wiring
    propagates it through the NPZ round-trip path.
    """
    bad = _well_formed_mgn_rollout(
        node_values={"u": np.ones((5, 10, 2), dtype=np.float32)},
    )
    save_mesh_rollout_npz(bad, tmp_path / "bad_key.npz")
    with pytest.raises(AssertionError, match="velocity"):
        load_mesh_rollout_npz(tmp_path / "bad_key.npz")


def test_load_mesh_rollout_npz_rejects_mgn_rollout_missing_dataset_metadata(tmp_path) -> None:
    """Finding 3 + Finding 1 absorption: an MGN-scoped NPZ missing
    ``metadata["dataset"]`` fails-loud at load (else the v9 substrate-
    class dispatch silently no-ops and the rule emits a misleading raw
    value on an open-driven-dissipative substrate).
    """
    bad = _well_formed_mgn_rollout(
        metadata={"framework": "pytorch+dgl", "model": "modulus_ns_meshgraphnet"},
    )
    save_mesh_rollout_npz(bad, tmp_path / "bad_meta.npz")
    with pytest.raises(AssertionError, match="dataset"):
        load_mesh_rollout_npz(tmp_path / "bad_meta.npz")


def test_load_mesh_rollout_npz_rejects_mgn_rollout_with_invalid_node_type(tmp_path) -> None:
    """Finding 3: an MGN-scoped NPZ with ``node_type`` values outside
    ``{0, 3, 4, 5, 6}`` (the loader's one-hot domain after the value-3
    shift; vortex_shedding_dataset.py:363-368 @ 1ca85d65) fails-loud at
    load — the alternative is a downstream RuntimeError during the
    one-hot encode rather than a diagnostic AssertionError here.
    """
    bad = _well_formed_mgn_rollout(
        node_type=np.array([0, 3, 4, 5, 6, 7, 0, 0, 0, 0], dtype=np.int64),
    )
    save_mesh_rollout_npz(bad, tmp_path / "bad_ntype.npz")
    with pytest.raises(AssertionError, match="node_type"):
        load_mesh_rollout_npz(tmp_path / "bad_ntype.npz")


def test_load_mesh_rollout_npz_passes_well_formed_mgn_rollout_round_trip(tmp_path) -> None:
    """Sanity: a well-formed MGN rollout round-trips through save+load
    without raising. Distinguishes Finding-3 wiring's fail-loud paths
    from a wholesale block on the MGN scope.
    """
    good = _well_formed_mgn_rollout()
    save_mesh_rollout_npz(good, tmp_path / "good.npz")
    recovered = load_mesh_rollout_npz(tmp_path / "good.npz")
    assert recovered.metadata["model"] == "modulus_ns_meshgraphnet"
    assert recovered.node_values["velocity"].dtype == np.float32


def test_load_mesh_rollout_npz_does_not_apply_mgn_contract_to_synthetic_rollouts(tmp_path) -> None:
    """Finding 3 scoping: generic mesh rollouts (``framework="synthetic"``,
    ``model`` NOT starting with ``"modulus_"``) bypass the MGN contract —
    they don't claim to satisfy it, and the contract's narrow scope is
    P0 vortex_shedding_2d per design §2.1 and D0-23 v10 scope note. A
    synthetic rollout authored with a node_type value outside the MGN
    one-hot domain (which would violate the MGN contract) must
    round-trip without raising.
    """
    synthetic = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.array([0, 1, 2, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),  # 1,2 invalid for MGN
        node_values={"velocity": np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={
            "framework": "synthetic",
            "model": "synthetic_unit_square",  # NOT "modulus_*"
            "dataset": "synthetic_unit_square",
        },
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    save_mesh_rollout_npz(synthetic, tmp_path / "synthetic.npz")
    recovered = load_mesh_rollout_npz(tmp_path / "synthetic.npz")  # must NOT raise
    assert recovered.metadata["framework"] == "synthetic"
    assert recovered.metadata["model"] == "synthetic_unit_square"


# ---------------------------------------------------------------------------
# Phase 2 round-codex-Phase2 Finding 4 absorption — rollout_contract
# metadata key triggers the F3 loader contract independently of the
# legacy model-name prefix. Forward-compatible with vendor / artifact-name
# rebrand (e.g., nvidia_physicsnemo_*).
# ---------------------------------------------------------------------------


def test_load_mesh_rollout_npz_fires_f3_contract_on_explicit_rollout_contract_key(tmp_path) -> None:
    """round-codex-Phase2 Finding 4: an NPZ that names a known MGN
    rollout_contract triggers the F3 helper even if the model name does
    NOT start with the legacy ``"modulus_"`` prefix. This is the
    forward-compatible path for future PhysicsNeMo / NVIDIA rebrands.
    """
    bad_path = tmp_path / "future_rebrand_bad.npz"
    np.savez(
        bad_path,
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int32),
        node_values=np.array(
            {"velocity": np.ones((5, 10, 2), dtype=np.float64)},  # fp64 — contract violation
            dtype=object,
        ),
        dt=np.float64(0.01),
        metadata=np.array(
            {
                "framework": "pytorch+dgl",
                # NOT "modulus_*" — simulates a future PhysicsNeMo rebrand.
                "model": "nvidia_physicsnemo_ns_meshgraphnet",
                "dataset": "vortex_shedding_2d",
                # Explicit contract key — triggers F3 by design.
                "rollout_contract": MGN_ROLLOUT_CONTRACT_P0,
            },
            dtype=object,
        ),
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    with pytest.raises(AssertionError, match="float32"):
        load_mesh_rollout_npz(bad_path)


def test_load_mesh_rollout_npz_fires_f3_contract_on_legacy_modulus_prefix_fallback(
    tmp_path,
) -> None:
    """round-codex-Phase2 Finding 4: backward-compatibility — rollouts
    produced before the absorption (those in the repo at f22319d..3380742)
    have only ``model.startswith("modulus_")`` and no rollout_contract
    key. The legacy fallback still triggers F3 so the in-repo artifacts
    don't silently regress on a re-load.
    """
    bad_path = tmp_path / "legacy_no_contract_bad.npz"
    np.savez(
        bad_path,
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int32),
        node_values=np.array(
            {"velocity": np.ones((5, 10, 2), dtype=np.float64)},
            dtype=object,
        ),
        dt=np.float64(0.01),
        metadata=np.array(
            {
                "framework": "pytorch+dgl",
                "model": "modulus_ns_meshgraphnet",  # legacy prefix only
                "dataset": "vortex_shedding_2d",
                # No rollout_contract key — pre-absorption shape.
            },
            dtype=object,
        ),
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    with pytest.raises(AssertionError, match="float32"):
        load_mesh_rollout_npz(bad_path)


def test_load_mesh_rollout_npz_bypasses_f3_when_rollout_contract_unknown(tmp_path) -> None:
    """round-codex-Phase2 Finding 4: a rollout that names an UNKNOWN
    rollout_contract value and has a non-modulus model bypasses F3.
    Future case-study contracts must be explicitly registered in
    KNOWN_MGN_ROLLOUT_CONTRACTS to opt in — typo / unregistered values
    fail open (which is correct: the rollout doesn't claim the MGN
    contract by claiming an unrecognized contract).
    """
    rollout = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={
            "framework": "fno",
            "model": "fno_darcy_baseline",  # NOT "modulus_*"
            "dataset": "darcy_2d",
            "rollout_contract": "fno_on_darcy_p0",  # unknown contract — bypasses
        },
        edge_index=None,
    )
    save_mesh_rollout_npz(rollout, tmp_path / "fno_bypass.npz")
    recovered = load_mesh_rollout_npz(tmp_path / "fno_bypass.npz")  # must NOT raise
    assert recovered.metadata["rollout_contract"] == "fno_on_darcy_p0"


# ---------------------------------------------------------------------------
# Phase 2 Task 4 — Gate A PASS branch lift: the *_on_mesh rule mirrors run
# on graph-mesh inputs via scikit-fem P1 finite elements rather than SKIP.
# Activates the D0-23 v9 substrate-class dispatch on real-shape (graph)
# inputs — previously dead code, preempted by the graph-mesh blanket SKIP.
# ---------------------------------------------------------------------------


def _build_unit_square_triangulation(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(positions, cells)`` for an ``nx*ny`` vertex grid on
    ``[0, 1]^2``. Each rectangular cell is split into two triangles
    (lower-right + upper-left), giving ``2 * (nx-1) * (ny-1)`` triangles.
    Node index = ``i * ny + j`` so the mesh is ``np.meshgrid(..., indexing='ij')``-
    compatible. Returns fp32 positions, int64 cells.
    """
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny), indexing="ij")
    positions = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    cells: list[list[int]] = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            n00 = i * ny + j
            n01 = i * ny + (j + 1)
            n10 = (i + 1) * ny + j
            n11 = (i + 1) * ny + (j + 1)
            cells.append([n00, n10, n11])  # lower-right triangle
            cells.append([n00, n11, n01])  # upper-left triangle
    return positions, np.array(cells, dtype=np.int64)


def _cells_to_edge_index(cells: np.ndarray) -> np.ndarray:
    """Convert ``(M, 3)`` triangle cells into an undirected ``(2, n_edges)``
    edge_index (each unordered edge once). Used to satisfy the
    :class:`MeshRollout` invariant that graph-mesh rollouts carry an
    edge_index; the FE path itself reads cells directly.
    """
    edge_set: set[tuple[int, int]] = set()
    for tri in cells:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            edge_set.add((min(u, v), max(u, v)))
    return np.array(sorted(edge_set), dtype=np.int64).T  # (2, n_edges)


def _build_graph_mesh_rollout(
    *,
    velocity_field: str = "divfree",  # "divfree" => v=(y,-x); "constant" => v=(1,0)
    dataset: str = "synthetic_unit_square",
    n_timesteps: int = 3,
) -> MeshRollout:
    """Build a graph-mesh MeshRollout on a 5x5 unit-square triangulation.

    Two velocity-field choices:

    - ``"divfree"``: ``v = (y, -x)`` — analytically divergence-free; FE
      integration of ``|∇·v|`` returns the discretization floor
      (machine-epsilon-ish on a P1 basis since v is itself P1).
    - ``"constant"``: ``v = (1, 0)`` — KE constant over t; drift = 0;
      dissipation_sign_violation = 0.
    """
    nx = ny = 5
    positions, cells = _build_unit_square_triangulation(nx, ny)
    n_nodes = positions.shape[0]
    if velocity_field == "divfree":
        v0 = np.stack([positions[:, 1], -positions[:, 0]], axis=1).astype(np.float32)
    elif velocity_field == "constant":
        v0 = np.stack(
            [np.ones(n_nodes, dtype=np.float32), np.zeros(n_nodes, dtype=np.float32)], axis=1
        )
    else:
        raise ValueError(f"unknown velocity_field {velocity_field!r}")
    velocity = np.tile(v0[None, ...], (n_timesteps, 1, 1))
    return MeshRollout(
        node_positions=positions,
        node_type=np.zeros(n_nodes, dtype=np.int64),
        node_values={"velocity": velocity},
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",  # graph mesh (is_regular_grid == False)
            "model": "synthetic-test",  # NOT "modulus_*" — bypasses MGN loader contract
            "dataset": dataset,
            "cells_2d": cells,  # supplied so the FE path runs
        },
        edge_index=_cells_to_edge_index(cells),
    )


def test_mass_conservation_defect_on_mesh_runs_on_graph_mesh_via_meshfield() -> None:
    """Phase 2 Task 4: Gate A PASS branch wires scikit-fem P1 into the
    rule mirror. A graph-mesh rollout no longer SKIPs; instead, the rule
    computes ``∫|∇·v|/∫‖∇v‖_F`` via FE.

    Fixture: 5x5 unit-square triangulation + analytically-divergence-free
    velocity ``v = (y, -x)``. Expected: defect at the FE floor
    (machine-epsilon-ish — ``v`` is exactly P1-representable so its
    divergence is exactly 0 in the P1 basis up to float arithmetic).
    """
    rollout = _build_graph_mesh_rollout(velocity_field="divfree")
    result = mass_conservation_defect_on_mesh(rollout)
    assert result.value is not None, (
        f"graph-mesh path must lift the SKIP; got skip_reason={result.skip_reason}"
    )
    assert result.value < 1e-12, (
        f"divergence-free v=(y,-x) on P1 must give FE-floor defect; got {result.value:.3e}"
    )


def test_mass_conservation_defect_on_mesh_skips_when_cells_2d_absent() -> None:
    """Phase 2 Task 4 precondition gate: a graph-mesh rollout WITHOUT
    metadata['cells_2d'] SKIPs with an informative reason rather than
    silently failing inside scikit-fem. Mirrors the F3 contract's
    fail-loud discipline on missing metadata.
    """
    rollout = _build_graph_mesh_rollout(velocity_field="divfree")
    # Strip cells_2d
    md = dict(rollout.metadata)
    md.pop("cells_2d")
    rollout_no_cells = MeshRollout(
        node_positions=rollout.node_positions,
        node_type=rollout.node_type,
        node_values=rollout.node_values,
        dt=rollout.dt,
        metadata=md,
        edge_index=rollout.edge_index,
    )
    result = mass_conservation_defect_on_mesh(rollout_no_cells)
    assert result.value is None
    assert result.skip_reason is not None
    assert "cells_2d" in result.skip_reason


def test_energy_drift_on_graph_mesh_fires_substrate_class_dispatch() -> None:
    """Phase 2 Task 4 + D0-23 v9 cross-check: with the graph-mesh blanket
    SKIP lifted, the v9 substrate-class dispatch now fires on real-shape
    (graph) MGN inputs. ``metadata['dataset']='vortex_shedding_2d'`` ⇒
    SKIP-with-reason citing the substrate class. The substrate-class
    SKIP outranks both the topology SKIP and any KE-rest gating —
    physics-grounded reasons are more informative than implementation-
    grounded ones.
    """
    rollout = _build_graph_mesh_rollout(velocity_field="constant", dataset="vortex_shedding_2d")
    result = energy_drift_on_mesh(rollout)
    assert result.value is None
    assert result.skip_reason is not None
    assert "open-driven-dissipative" in result.skip_reason
    assert "D0-22" in result.skip_reason or "D0-23" in result.skip_reason


def test_dissipation_sign_violation_on_graph_mesh_fires_substrate_class_dispatch() -> None:
    """Phase 2 Task 4 + D0-23 v9: same as the energy_drift test, for
    dissipation_sign_violation. The mesh-side v9 dispatch fires uniformly
    across both KE-anchored rules on the open-driven-dissipative
    substrate class.
    """
    rollout = _build_graph_mesh_rollout(velocity_field="constant", dataset="vortex_shedding_2d")
    result = dissipation_sign_violation_on_mesh(rollout)
    assert result.value is None
    assert result.skip_reason is not None
    assert "open-driven-dissipative" in result.skip_reason


def test_energy_drift_on_mesh_runs_on_graph_mesh_with_non_substrate_dataset() -> None:
    """Phase 2 Task 4: on a graph mesh with a dataset NOT in
    MGN_DATASET_SYSTEM_CLASS (so the substrate-class dispatch does
    not fire), the FE KE-integration path runs and computes drift =
    0 for a constant-velocity rollout (KE is constant in time, so
    max|KE(t) - KE(0)| / |KE(0)| = 0).
    """
    rollout = _build_graph_mesh_rollout(velocity_field="constant", dataset="synthetic_unit_square")
    result = energy_drift_on_mesh(rollout)
    assert result.value is not None, (
        f"non-substrate-class graph-mesh path must run; got skip_reason={result.skip_reason}"
    )
    # Constant velocity → KE constant → drift = 0.
    assert result.value == 0.0, f"constant-velocity graph-mesh drift must be 0; got {result.value}"


def test_dissipation_sign_violation_on_mesh_runs_on_graph_mesh_with_non_substrate_dataset() -> None:
    """Phase 2 Task 4 mirror of the energy_drift test for
    dissipation_sign_violation. Constant velocity ⇒ KE constant ⇒
    dKE/dt = 0 ⇒ max(0, max(dKE/dt)) = 0 ⇒ violation = 0.
    """
    rollout = _build_graph_mesh_rollout(velocity_field="constant", dataset="synthetic_unit_square")
    result = dissipation_sign_violation_on_mesh(rollout)
    assert result.value is not None, (
        f"non-substrate-class graph-mesh path must run; got skip_reason={result.skip_reason}"
    )
    assert result.value == 0.0, (
        f"constant-velocity graph-mesh dissipation violation must be 0; got {result.value}"
    )
