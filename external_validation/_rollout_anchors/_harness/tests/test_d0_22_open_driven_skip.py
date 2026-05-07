"""Tests for the D0-22 open-driven-dissipative skip-with-reason gate on dissipation_sign_violation.

Per DECISIONS.md D0-22: `dissipation_sign_violation` is "zero for strictly dissipative
or strictly conservative rollouts" by docstring; on open-driven systems (gravity-loaded
or forced-flow) where dE/dt > 0 over a stretch by physics, the strictly-dissipative-or-
conservative assumption fails and the rule should SKIP-with-reason rather than emit a
spurious raw value.

The gate dispatches purely on `system_class == "open-driven-dissipative"`; KE shape is
NOT a co-condition (open-driven systems with monotone KE due to model bugs should still
SKIP — the surfacing of such bugs is the job of a different rule). Tests cover the gate's
truth table:

1. Positive path: open-driven dataset + rise-then-fall KE → SKIP D0-22.
2. Negative path A: non-open-driven dataset + rise-then-fall KE → fire raw value
   (gate is system_class-conditioned, not KE-shape-conditioned).
3. Negative path B: open-driven dataset + monotone-decreasing KE → SKIP D0-22 anyway
   (gate ignores KE shape).
4. Fail-loud path: typo'd dataset name (not in mapping) → falls through to raw,
   no silent default-to-skip behavior.

All fixtures are hand-crafted, not copied from production rollouts (per
feedback_test_fixtures_hand_crafted_not_copied).
"""

from __future__ import annotations

import numpy as np

from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    LAGRANGEBENCH_DATASET_SYSTEM_CLASS,
    ParticleRollout,
    dissipation_sign_violation,
)

# ---------------------------------------------------------------------------
# Hand-crafted fixtures
# ---------------------------------------------------------------------------


def _build_rise_then_fall_rollout(
    *,
    dataset_name: str,
    n_timesteps: int = 10,
    n_particles: int = 4,
    peak_t: int = 4,
    rise_rate: float = 0.5,
    fall_rate: float = 0.3,
) -> ParticleRollout:
    """Synthetic rollout with KE rising to t=peak_t then falling.

    Mimics dam-break-2D's gravity-loaded PE->KE->dissipation profile in synthetic
    form. Velocities scale linearly during the rise phase and decay exponentially
    during the fall phase. KE = 0.5 * sum(m_i * |v_i|^2) tracks |v|^2 trajectory.

    KE(0) is set above KE_REST_THRESHOLD so the existing D0-08 KE-rest gate does
    not fire - isolates D0-22 specifically.
    """
    rng = np.random.default_rng(20260507)
    dt = 0.01
    positions = np.zeros((n_timesteps, n_particles, 2), dtype=float)
    velocities = np.zeros((n_timesteps, n_particles, 2), dtype=float)
    # v0 above rest threshold so D0-08 KE-rest skip doesn't preempt D0-22
    v0 = rng.normal(scale=1.0, size=(n_particles, 2))
    for t in range(n_timesteps):
        if t <= peak_t:
            scale = 1.0 + rise_rate * t * dt
        else:
            scale = (1.0 + rise_rate * peak_t * dt) * np.exp(-fall_rate * (t - peak_t) * dt)
        velocities[t] = v0 * scale
        positions[t] = 0.5  # not load-bearing
    return ParticleRollout(
        positions=positions,
        velocities=velocities,
        particle_type=np.zeros(n_particles, dtype=np.int32),
        particle_mass=np.ones(n_particles, dtype=np.float64),
        dt=dt,
        domain_box=np.array([[0.0, 0.0], [1.0, 1.0]]),
        metadata={"dataset": dataset_name},
    )


def _build_monotone_decreasing_rollout(
    *,
    dataset_name: str,
    n_timesteps: int = 10,
    n_particles: int = 4,
    decay_rate: float = 0.5,
) -> ParticleRollout:
    """Strictly monotone-decreasing KE rollout. Mirrors test_d0_18's
    `_build_dissipative_rollout` shape but parametrized for D0-22's negative-path-B
    fixture (open-driven dataset name with closed-dissipative KE shape - the gate
    must SKIP regardless of the shape mismatch).
    """
    rng = np.random.default_rng(20260507)
    dt = 0.01
    positions = np.zeros((n_timesteps, n_particles, 2), dtype=float)
    velocities = np.zeros((n_timesteps, n_particles, 2), dtype=float)
    v0 = rng.normal(scale=1.0, size=(n_particles, 2))
    for t in range(n_timesteps):
        velocities[t] = v0 * np.exp(-decay_rate * t * dt)
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
# 1. Taxonomy: dam2d post-D0-22 maps to open-driven-dissipative
# ---------------------------------------------------------------------------


def test_dam2d_classified_as_open_driven_dissipative() -> None:
    """D0-22 reclassifies dam2d empirically (post-rollout KE shape verified
    rise-then-fall). The taxonomy map is the source-of-truth for substrate
    class labels.
    """
    assert LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get("dam2d") == "open-driven-dissipative", (
        "D0-22: dam2d must be classified 'open-driven-dissipative' (gravity-loaded SPH)"
    )


# ---------------------------------------------------------------------------
# 2. Gate truth table
# ---------------------------------------------------------------------------


def test_skip_when_open_driven_with_rise_then_fall_ke() -> None:
    """Positive path: open-driven dataset + rise-then-fall KE -> SKIP D0-22.

    The headline fixture for D0-22. KE rises during the synthetic gravity-load
    phase (t < peak_t), then decays during the synthetic dissipation phase
    (t > peak_t). Without D0-22, dissipation_sign_violation would fire a non-zero
    raw value (max(0, max(dE/dt)) > 0 during the rise phase) and falsely flag
    the trajectory as a violation. D0-22 SKIPs instead.
    """
    rollout = _build_rise_then_fall_rollout(dataset_name="dam2d")
    result = dissipation_sign_violation(rollout)
    assert result.value is None, "open-driven dataset + rise-then-fall KE must SKIP, not fire raw"
    assert result.skip_reason is not None
    assert "system_class='open-driven-dissipative'" in result.skip_reason
    assert "dataset='dam2d'" in result.skip_reason
    assert "gravitational PE" in result.skip_reason
    assert "D0-22" in result.skip_reason


def test_fire_raw_when_non_open_driven_with_rise_then_fall_ke() -> None:
    """Negative path A: non-open-driven dataset + rise-then-fall KE -> fire raw value.

    Proves the gate dispatches on system_class, not KE shape. A "dissipative"-class
    rollout with non-monotone KE is methodologically interesting (possibly a buggy
    supposed-conservative model gaining energy) and should fire raw, not SKIP.
    """
    rollout = _build_rise_then_fall_rollout(dataset_name="tgv2d")
    result = dissipation_sign_violation(rollout)
    assert result.value is not None, (
        "tgv2d (system_class='dissipative') + rise-then-fall KE must fire raw, not SKIP - "
        "the gate must NOT co-condition on KE shape"
    )
    assert result.skip_reason is None
    assert result.value > 0  # non-monotone KE has positive dE/dt somewhere


def test_skip_when_open_driven_regardless_of_ke_shape() -> None:
    """Negative path B: open-driven dataset + monotone-decreasing KE -> SKIP D0-22.

    Proves the gate is purely system_class-conditioned. If `dam2d` happens to
    show monotone-decreasing KE on a particular trajectory (unusual but possible -
    e.g., an IC where the fluid column has already fallen), D0-22 still SKIPs
    because the substrate is open-driven by physics and the rule's strictly-
    dissipative-or-conservative assumption still doesn't hold (other dam-break
    trajectories will show non-monotone KE).
    """
    rollout = _build_monotone_decreasing_rollout(dataset_name="dam2d")
    result = dissipation_sign_violation(rollout)
    assert result.value is None, (
        "open-driven dataset must SKIP regardless of KE shape - gate is system_class-"
        "conditioned, not KE-shape-conditioned"
    )
    assert result.skip_reason is not None
    assert "system_class='open-driven-dissipative'" in result.skip_reason


def test_unknown_dataset_falls_through_to_raw() -> None:
    """Fail-loud path: typo'd dataset name not in mapping -> falls through to raw.

    Guards against future taxonomy entries with typos creating silent fall-through
    behavior. A dataset name that is neither in `LAGRANGEBENCH_DATASET_SYSTEM_CLASS`
    nor an empty string must NOT default to any SKIP path; it must fire raw value.
    Catches the mistake of `default_skip_class = "open-driven-dissipative"` etc.
    """
    rollout = _build_rise_then_fall_rollout(dataset_name="dam2d_typo_not_in_mapping")
    result = dissipation_sign_violation(rollout)
    assert LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get("dam2d_typo_not_in_mapping") is None, (
        "test precondition: typo'd name must not be in mapping"
    )
    assert result.value is not None, "unknown dataset must fall through to raw, never silent SKIP"
    assert result.skip_reason is None


# ---------------------------------------------------------------------------
# 3. Reason-template invariants (D0-19 §3.4)
# ---------------------------------------------------------------------------


def test_skip_reason_template_constant_across_invocations() -> None:
    """D0-19 §3.4 requires the skip_reason string to be template-constant
    across rows of the same (rule, stack). Two invocations with the same
    `dataset` value must return the same skip_reason string verbatim.
    """
    r1 = _build_rise_then_fall_rollout(dataset_name="dam2d")
    r2 = _build_rise_then_fall_rollout(dataset_name="dam2d", n_particles=8)  # different N
    s1 = dissipation_sign_violation(r1).skip_reason
    s2 = dissipation_sign_violation(r2).skip_reason
    assert s1 == s2, (
        "D0-19 §3.4: skip_reason must be template-constant within (rule, stack); "
        "got divergent strings on the same dataset name"
    )


def test_skip_reason_signposts_d0_22() -> None:
    """The skip_reason must cite D0-22 by name for audit-trail traceability -
    same discipline as D0-18's reason which cites 'D0-18' verbatim.
    """
    rollout = _build_rise_then_fall_rollout(dataset_name="dam2d")
    result = dissipation_sign_violation(rollout)
    assert result.skip_reason is not None
    assert "D0-22" in result.skip_reason
