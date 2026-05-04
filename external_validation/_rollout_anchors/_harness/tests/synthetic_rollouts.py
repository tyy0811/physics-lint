"""Synthetic-rollout builders for the Day-0.5 read-only-path tests.

Per `physics-lint-validation/DECISIONS.md` D0-04 + the post-Gate-B
review handoff: the read-only path of `particle_rollout_adapter.py`
(``ParticleRollout``, ``mass_conservation_defect``, ``energy_drift``,
``dissipation_sign_violation``) is exercised against synthetic
rollouts here. **Framing: rule-plumbing regression tests, not
real-rollout claims.** The synthetic configurations have known
analytical conservation properties; deviations from those properties
in the harness's emitted scalar are bugs in the harness, not in the
synthetic data.

Day 1 will add a path that consumes pre-recorded LagrangeBench
``.npz`` (if available without a JAX install) or a real rollout
generated on Modal A100 (with JAX). Until then, the synthetic
fixtures here are the only thing the read-only path runs against.

Three configurations:

- ``build_constant_velocity_rollout`` — closed-system, constant per-
  particle velocity. Mass exactly conserved, KE exactly constant,
  dE/dt = 0 ⇒ all three harness defects emit zero. Tests rule
  plumbing on the trivial-conservation case.

- ``build_damped_decay_rollout`` — exponentially-damped velocities
  with closed-form decay. Mass conserved, KE strictly decreasing
  with KE(t) = KE(0) · exp(-2 gamma t), dE/dt < 0 always ⇒
  ``mass_conservation_defect = 0``,
  ``energy_drift = 1 - exp(-2 gamma T)`` (analytic),
  ``dissipation_sign_violation = 0``. Tests dissipative-case
  semantics.

- ``build_energy_growth_rollout`` — deliberate violation. Particle 0
  has linearly-growing speed (unphysical) ⇒ dE/dt > 0 at every
  timestep, ``dissipation_sign_violation`` non-zero. Tests violation
  detection.

Builders return both a ``ParticleRollout`` and an ``expected``
dict carrying the analytical defect values for the test assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    ParticleRollout,
)


@dataclass(frozen=True)
class SyntheticRolloutCase:
    """A synthetic rollout plus its analytically-known defect values."""

    rollout: ParticleRollout
    expected_mass_conservation_defect: float
    expected_energy_drift: float | None  # None when not analytically tractable
    expected_dissipation_sign_violation: float
    description: str


def _base_metadata(model: str, dataset: str) -> dict[str, Any]:
    """Boilerplate metadata fields per `SCHEMA.md` §1; synthetic placeholders."""
    return {
        "ckpt_hash": "synthetic-no-checkpoint",
        "ckpt_path": "synthetic://no-checkpoint",
        "git_sha": "synthetic-no-git-sha",
        "lagrangebench_sha": "synthetic-no-lagrangebench-sha",
        "dataset": dataset,
        "model": model,
        "seed": 0,
        "framework": "synthetic",
        "framework_version": "0.0.0",
    }


def build_constant_velocity_rollout(
    *,
    n_particles: int = 16,
    n_timesteps: int = 50,
    dt: float = 0.01,
    domain_size: float = 1.0,
    speed: float = 0.05,
    seed: int = 20260504,
) -> SyntheticRolloutCase:
    """Closed-system, constant-velocity rollout. All three defects = 0.

    Particles arranged on a square sub-grid inside ``[0, domain_size]^2``.
    Each particle moves with a small randomly-oriented constant
    velocity; trajectories may exit the box (no wrap-around applied)
    but mass / KE / dE/dt only depend on velocities, which are constant.
    """
    rng = np.random.default_rng(seed=seed)
    side = int(np.ceil(np.sqrt(n_particles)))
    n_particles = side * side
    xs = np.linspace(0.2 * domain_size, 0.8 * domain_size, side)
    ys = np.linspace(0.2 * domain_size, 0.8 * domain_size, side)
    mesh_x, mesh_y = np.meshgrid(xs, ys, indexing="ij")
    initial_positions = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)

    angles = rng.uniform(0.0, 2 * np.pi, size=n_particles)
    velocity_per_particle = speed * np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Linear free flight: x(t) = x(0) + v t.
    times = np.arange(n_timesteps) * dt
    positions = initial_positions[None, :, :] + (
        times[:, None, None] * velocity_per_particle[None, :, :]
    )
    velocities = np.broadcast_to(velocity_per_particle[None, :, :], positions.shape).copy()

    rollout = ParticleRollout(
        positions=positions,
        velocities=velocities,
        particle_type=np.zeros(n_particles, dtype=np.int32),
        particle_mass=np.ones(n_particles),
        dt=dt,
        domain_box=np.array([[0.0, 0.0], [domain_size, domain_size]]),
        metadata=_base_metadata("synthetic-constant-velocity", "synthetic-tgv-trivial"),
    )
    return SyntheticRolloutCase(
        rollout=rollout,
        expected_mass_conservation_defect=0.0,
        expected_energy_drift=0.0,
        expected_dissipation_sign_violation=0.0,
        description="closed-system constant-velocity flow; trivial conservation",
    )


def build_damped_decay_rollout(
    *,
    n_particles: int = 16,
    n_timesteps: int = 100,
    dt: float = 0.01,
    domain_size: float = 1.0,
    initial_speed: float = 0.5,
    decay_rate: float = 0.5,
    seed: int = 20260504,
) -> SyntheticRolloutCase:
    """Exponentially-damped velocities; KE(t) = KE(0) exp(-2 gamma t).

    Mass is conserved exactly. Total KE has known closed-form decay.
    ``dE/dt = -2 gamma KE(t) < 0`` for all t > 0, so the
    dissipation-sign-violation defect is exactly zero.

    The expected ``energy_drift`` value is
    ``max |E(t) - E(0)| / |E(0)| = 1 - exp(-2 gamma T_final)``, where
    T_final = (n_timesteps - 1) · dt.
    """
    rng = np.random.default_rng(seed=seed)
    side = int(np.ceil(np.sqrt(n_particles)))
    n_particles = side * side
    xs = np.linspace(0.2 * domain_size, 0.8 * domain_size, side)
    ys = np.linspace(0.2 * domain_size, 0.8 * domain_size, side)
    mesh_x, mesh_y = np.meshgrid(xs, ys, indexing="ij")
    initial_positions = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)

    angles = rng.uniform(0.0, 2 * np.pi, size=n_particles)
    initial_velocity = initial_speed * np.stack([np.cos(angles), np.sin(angles)], axis=1)

    times = np.arange(n_timesteps) * dt
    decay = np.exp(-decay_rate * times)  # (T,)
    velocities = initial_velocity[None, :, :] * decay[:, None, None]
    # Position is the integral of velocity: x(t) = x(0) + v0 (1 - exp(-gamma t)) / gamma
    integrated_decay = (1.0 - decay) / decay_rate  # (T,)
    positions = initial_positions[None, :, :] + (
        integrated_decay[:, None, None] * initial_velocity[None, :, :]
    )

    t_final = (n_timesteps - 1) * dt
    expected_drift = 1.0 - float(np.exp(-2.0 * decay_rate * t_final))

    rollout = ParticleRollout(
        positions=positions,
        velocities=velocities,
        particle_type=np.zeros(n_particles, dtype=np.int32),
        particle_mass=np.ones(n_particles),
        dt=dt,
        domain_box=np.array([[0.0, 0.0], [domain_size, domain_size]]),
        metadata=_base_metadata("synthetic-damped-decay", "synthetic-tgv-decay"),
    )
    return SyntheticRolloutCase(
        rollout=rollout,
        expected_mass_conservation_defect=0.0,
        expected_energy_drift=expected_drift,
        expected_dissipation_sign_violation=0.0,
        description=f"damped decay KE(t) = KE(0) exp(-2 * {decay_rate} t); strictly dissipative",
    )


def build_energy_growth_rollout(
    *,
    n_particles: int = 16,
    n_timesteps: int = 50,
    dt: float = 0.01,
    domain_size: float = 1.0,
    growth_rate: float = 1.0,
    seed: int = 20260504,
) -> SyntheticRolloutCase:
    """Deliberate energy-growth violation. Particle 0 has linearly-growing speed.

    Initial velocity: all particles at rest (KE(0) = 0). Particle 0
    accelerates linearly: ``|v_0(t)| = growth_rate * t``. KE(t) grows
    as ``0.5 * (growth_rate * t)^2``, so ``dE/dt > 0`` at every t > 0
    ⇒ ``dissipation_sign_violation`` is non-zero.

    The harness should detect this and emit a non-trivial violation
    (raw value > 0). The exact value depends on the rollout length and
    the growth rate; the test asserts non-zero rather than a specific
    number to avoid brittleness against floating-point edge effects.
    """
    rng = np.random.default_rng(seed=seed)
    side = int(np.ceil(np.sqrt(n_particles)))
    n_particles = side * side
    xs = np.linspace(0.2 * domain_size, 0.8 * domain_size, side)
    ys = np.linspace(0.2 * domain_size, 0.8 * domain_size, side)
    mesh_x, mesh_y = np.meshgrid(xs, ys, indexing="ij")
    initial_positions = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)

    times = np.arange(n_timesteps) * dt
    velocities = np.zeros((n_timesteps, n_particles, 2))
    # Particle 0 accelerates in a fixed (random but seeded) direction.
    angle_0 = rng.uniform(0.0, 2 * np.pi)
    direction_0 = np.array([np.cos(angle_0), np.sin(angle_0)])
    velocities[:, 0, :] = (growth_rate * times)[:, None] * direction_0[None, :]

    # Positions: particle 0 has integrated displacement; others static.
    positions = np.broadcast_to(initial_positions[None, :, :], (n_timesteps, n_particles, 2)).copy()
    # x_0(t) = x_0(0) + ∫ v_0 dt = x_0(0) + 0.5 growth_rate t^2 direction_0.
    positions[:, 0, :] = initial_positions[0, :] + (
        0.5 * growth_rate * times[:, None] ** 2 * direction_0[None, :]
    )

    rollout = ParticleRollout(
        positions=positions,
        velocities=velocities,
        particle_type=np.zeros(n_particles, dtype=np.int32),
        particle_mass=np.ones(n_particles),
        dt=dt,
        domain_box=np.array([[0.0, 0.0], [domain_size, domain_size]]),
        metadata=_base_metadata("synthetic-energy-growth", "synthetic-violation-rollout"),
    )
    return SyntheticRolloutCase(
        rollout=rollout,
        expected_mass_conservation_defect=0.0,
        expected_energy_drift=None,  # nonzero but not analytically constrained
        expected_dissipation_sign_violation=float("inf"),  # > 0; exact value not asserted
        description=(
            f"deliberate violation: particle 0 accelerates at rate {growth_rate}; "
            f"dissipation_sign_violation > 0 expected"
        ),
    )
