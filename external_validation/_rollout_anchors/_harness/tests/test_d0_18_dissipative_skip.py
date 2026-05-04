"""Tests for the D0-18 dissipative-system skip-with-reason gate on energy_drift.

Per DECISIONS.md D0-18: PH-CON-002's relative-drift form is a misfire on
dissipative-by-design systems (TGV2D, RPF2D, etc.) where the dissipation
magnitude IS the physics. The harness's ``energy_drift`` gates a
skip-with-reason path on positive evidence on BOTH axes:

1. ``rollout.metadata["dataset"]`` resolves to "dissipative" via
   ``LAGRANGEBENCH_DATASET_SYSTEM_CLASS``
2. ``KE(t)`` is monotone-non-increasing across the rollout

Either alone is insufficient: monotone-decreasing without a system_class
hint could be a buggy supposed-conservative surrogate; a system_class
hint without monotonicity could be a buggy "dissipative" model that
spuriously gains energy. Both required → SKIP; otherwise fire raw value.

Tests below cover all four corners of the (system_class, monotonicity)
truth table, plus the existing-behavior regression guard (synthetic
rollouts with non-LB dataset names continue to fire raw value as in
v1 pre-D0-18).
"""

from __future__ import annotations

import numpy as np

from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    LAGRANGEBENCH_DATASET_SYSTEM_CLASS,
    ParticleRollout,
    energy_drift,
)


def _build_dissipative_rollout(
    *,
    dataset_name: str,
    n_timesteps: int = 8,
    n_particles: int = 4,
    decay_rate: float = 0.5,
) -> ParticleRollout:
    """Decaying-KE rollout with arbitrary metadata.dataset.

    Velocities decay exponentially: ``v(t) = v(0) * exp(-decay_rate * t * dt)``.
    KE = 0.5 * sum(m_i * |v_i|^2) decays as ``KE(0) * exp(-2 * decay_rate * t * dt)``.
    Strictly monotone-decreasing when decay_rate > 0.
    """
    rng = np.random.default_rng(20260504)
    dt = 0.01
    positions = np.zeros((n_timesteps, n_particles, 2), dtype=float)
    velocities = np.zeros((n_timesteps, n_particles, 2), dtype=float)
    v0 = rng.normal(scale=1.0, size=(n_particles, 2))
    for t in range(n_timesteps):
        decay_factor = np.exp(-decay_rate * t * dt)
        velocities[t] = v0 * decay_factor
        # Positions are not load-bearing for energy_drift (uses velocities directly)
        positions[t] = 0.5 + 0.0 * t  # trivial constant
    return ParticleRollout(
        positions=positions,
        velocities=velocities,
        particle_type=np.zeros(n_particles, dtype=np.int32),
        particle_mass=np.ones(n_particles, dtype=np.float64),
        dt=dt,
        domain_box=np.array([[0.0, 0.0], [1.0, 1.0]]),
        metadata={"dataset": dataset_name},
    )


def _build_growing_rollout(
    *,
    dataset_name: str,
    n_timesteps: int = 8,
    n_particles: int = 4,
    growth_rate: float = 0.5,
) -> ParticleRollout:
    """Velocities grow linearly per timestep — non-monotone-decreasing KE.

    Tests the monotonicity gate: even with a "dissipative" system_class
    hint, this rollout must NOT skip (KE is increasing, which is a
    methodologically interesting signal — possibly a buggy model).
    """
    rng = np.random.default_rng(20260504)
    dt = 0.01
    positions = np.zeros((n_timesteps, n_particles, 2), dtype=float)
    velocities = np.zeros((n_timesteps, n_particles, 2), dtype=float)
    v0 = rng.normal(scale=1.0, size=(n_particles, 2))
    for t in range(n_timesteps):
        velocities[t] = v0 * (1.0 + growth_rate * t * dt)
        positions[t] = 0.5
    return ParticleRollout(
        positions=positions,
        velocities=velocities,
        particle_type=np.zeros(n_particles, dtype=np.int32),
        particle_mass=np.ones(n_particles, dtype=np.float64),
        dt=dt,
        domain_box=np.array([[0.0, 0.0], [1.0, 1.0]]),
        metadata={"dataset": dataset_name},
    )


# ---------------------------------------------------------------------------
# 1. The system_class mapping itself
# ---------------------------------------------------------------------------


def test_lagrangebench_dataset_system_class_includes_all_2d_sph() -> None:
    """All five 2D LB SPH datasets must be in the dissipative mapping (D0-18)."""
    for dataset in ("tgv2d", "rpf2d", "ldc2d", "dam2d"):
        assert LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get(dataset) == "dissipative", (
            f"D0-18: {dataset} must be classified 'dissipative' (SPH viscous)"
        )


def test_lagrangebench_dataset_system_class_does_not_include_unknown() -> None:
    """Unknown dataset names return None (default = fire raw value, not skip)."""
    assert LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get("synthetic-tgv-decay") is None
    assert LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get("hamiltonian-flow") is None
    assert LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get("") is None


# ---------------------------------------------------------------------------
# 2. Two-half gate truth table
# ---------------------------------------------------------------------------


def test_skip_when_dissipative_and_monotone_decreasing() -> None:
    """Both gate halves satisfied → SKIP with reason naming D0-18."""
    rollout = _build_dissipative_rollout(dataset_name="tgv2d")
    result = energy_drift(rollout)
    assert result.value is None
    assert result.skip_reason is not None
    assert "system_class='dissipative'" in result.skip_reason
    assert "monotone-non-increasing" in result.skip_reason
    assert "D0-18" in result.skip_reason
    assert "dissipation_sign_violation" in result.skip_reason  # signposts the alternative


def test_fire_raw_when_dissipative_but_increasing() -> None:
    """Half a: system_class=dissipative; Half b: NOT monotone-decreasing.

    Skip would be wrong here — a "dissipative" model that's spuriously
    gaining energy is methodologically interesting and should fire the
    raw drift, not skip silently. This is the buggy-supposed-dissipative
    case the gate's monotonicity half exists to catch.
    """
    rollout = _build_growing_rollout(dataset_name="tgv2d")
    result = energy_drift(rollout)
    assert result.value is not None
    assert result.skip_reason is None
    assert result.value > 0


def test_fire_raw_when_monotone_decreasing_but_unknown_system_class() -> None:
    """Half a: system_class missing; Half b: monotone-decreasing.

    Skip would be wrong here — a supposed-conservative surrogate
    (e.g., a Hamiltonian flow model) that's leaking energy looks
    monotone-decreasing but PH-CON-002 should catch it. This is the
    buggy-supposed-conservative case the gate's system_class half exists
    to catch.
    """
    rollout = _build_dissipative_rollout(dataset_name="hamiltonian-flow-surrogate")
    result = energy_drift(rollout)
    assert result.value is not None
    assert result.skip_reason is None
    assert result.value > 0


def test_fire_raw_when_neither_dissipative_nor_monotone_decreasing() -> None:
    """Default behavior absent positive evidence on either axis."""
    rollout = _build_growing_rollout(dataset_name="hamiltonian-flow-surrogate")
    result = energy_drift(rollout)
    assert result.value is not None
    assert result.skip_reason is None


# ---------------------------------------------------------------------------
# 3. Regression guards
# ---------------------------------------------------------------------------


def test_synthetic_dataset_names_do_not_trigger_skip() -> None:
    """Pre-D0-18 behavior on synthetic rollouts (test_read_only_path.py fixtures)
    must be unchanged. The synthetic builders use dataset names like
    'synthetic-tgv-decay' which are NOT in the LB mapping.
    """
    for synthetic_name in (
        "synthetic-tgv-trivial",
        "synthetic-tgv-decay",
        "synthetic-violation-rollout",
    ):
        rollout = _build_dissipative_rollout(dataset_name=synthetic_name)
        result = energy_drift(rollout)
        assert result.value is not None, (
            f"D0-18 must not skip on synthetic dataset '{synthetic_name}' "
            "(would be a regression on existing read-only-path tests)"
        )


def test_ke_rest_skip_takes_precedence_over_d0_18() -> None:
    """If KE(0) < KE_REST_THRESHOLD, the D0-08 KE-rest skip fires before
    the D0-18 dissipative-system skip is even evaluated.

    Order matters: a rollout that starts at rest AND has dissipative
    system_class should report the KE-rest skip reason (more specific
    physical condition) rather than the D0-18 reason.
    """
    rollout = _build_dissipative_rollout(dataset_name="tgv2d")
    # Force KE(0) ~ 0 by zeroing velocities at t=0
    velocities = rollout.velocities.copy()
    velocities[0] = 0.0
    rollout = ParticleRollout(
        positions=rollout.positions,
        velocities=velocities,
        particle_type=rollout.particle_type,
        particle_mass=rollout.particle_mass,
        dt=rollout.dt,
        domain_box=rollout.domain_box,
        metadata=rollout.metadata,
    )
    result = energy_drift(rollout)
    assert result.value is None
    assert result.skip_reason is not None
    assert "D0-08" in result.skip_reason  # KE-rest reason, not D0-18
    assert "rest" in result.skip_reason.lower()


def test_metadata_dataset_missing_falls_back_to_fire_raw() -> None:
    """A rollout with no 'dataset' key in metadata defaults to fire raw value
    (positive-evidence-required gate; absent evidence = absent skip).
    """
    rollout = _build_dissipative_rollout(dataset_name="tgv2d")
    rollout = ParticleRollout(
        positions=rollout.positions,
        velocities=rollout.velocities,
        particle_type=rollout.particle_type,
        particle_mass=rollout.particle_mass,
        dt=rollout.dt,
        domain_box=rollout.domain_box,
        metadata={"git_sha": "x"},  # no 'dataset' key
    )
    result = energy_drift(rollout)
    assert result.value is not None
    assert result.skip_reason is None


# ---------------------------------------------------------------------------
# 4. Reason-string contract
# ---------------------------------------------------------------------------


def test_skip_reason_names_dataset_for_audit_trail() -> None:
    """The skip_reason must include the dataset name so a future SARIF
    consumer can attribute the skip without re-reading the rollout
    metadata.
    """
    rollout = _build_dissipative_rollout(dataset_name="tgv2d")
    result = energy_drift(rollout)
    assert result.skip_reason is not None
    assert "'tgv2d'" in result.skip_reason


def test_skip_reason_includes_ke_endpoints() -> None:
    """The skip_reason must include KE(0) and KE(end) so a future reader
    can quickly assess whether the dissipation magnitude is physical
    (e.g., partial decay vs ~total decay).
    """
    rollout = _build_dissipative_rollout(dataset_name="tgv2d")
    result = energy_drift(rollout)
    assert result.skip_reason is not None
    assert "KE(0)" in result.skip_reason
    assert "KE(end)" in result.skip_reason
