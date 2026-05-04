"""Read-only path tests — Day 0.5 rule-plumbing regression.

Per `physics-lint-validation/DECISIONS.md` D0-04 + post-Gate-B handoff:
exercises the rollout-level surface of `particle_rollout_adapter.py`
(``ParticleRollout``, ``load_rollout_npz``, ``save_rollout_npz``,
``mass_conservation_defect``, ``energy_drift``, ``dissipation_sign_violation``)
against the synthetic builders in ``synthetic_rollouts.py``.

**Framing.** These are rule-plumbing regression tests: they verify that
the harness's emitted scalars match analytically-known conservation
properties of the synthetic rollouts. They are **not** real-rollout
validation claims; LagrangeBench-side validation lives on Day 1 work
(JAX, Modal A100, real checkpoints). Pre-recorded LagrangeBench `.npz`
samples without a JAX install were not pursued — at the time of
writing, LagrangeBench's data ships as a `bash download_data.sh`
multi-GB download via HuggingFace, with no obvious small published
sample we can pull cheaply. If a small fixture sample becomes
available, an additional test under ``test_real_rollout_pickup`` can
land without restructuring this file.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    ParticleRollout,
    dissipation_sign_violation,
    energy_drift,
    kinetic_energy_series,
    load_rollout_npz,
    mass_conservation_defect,
    save_rollout_npz,
    total_mass_series,
)
from external_validation._rollout_anchors._harness.tests.synthetic_rollouts import (
    build_constant_velocity_rollout,
    build_damped_decay_rollout,
    build_energy_growth_rollout,
)

# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_save_load_npz_round_trip(tmp_path):
    """save_rollout_npz then load_rollout_npz must reconstruct field-by-field."""
    case = build_constant_velocity_rollout(n_particles=9, n_timesteps=10)
    saved = save_rollout_npz(case.rollout, tmp_path / "rollout.npz")
    assert saved.exists()
    reloaded = load_rollout_npz(saved)
    # Allow the float32 cast on save to introduce small numeric error.
    np.testing.assert_allclose(reloaded.positions, case.rollout.positions, rtol=0, atol=1e-5)
    np.testing.assert_allclose(reloaded.velocities, case.rollout.velocities, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(reloaded.particle_type, case.rollout.particle_type)
    np.testing.assert_allclose(
        reloaded.particle_mass, case.rollout.particle_mass, rtol=0, atol=1e-12
    )
    assert reloaded.dt == case.rollout.dt
    np.testing.assert_array_equal(reloaded.domain_box, case.rollout.domain_box)
    # Metadata round-trips through pickle; check a representative key.
    assert reloaded.metadata.get("model") == "synthetic-constant-velocity"


def test_load_rollout_npz_missing_field_raises(tmp_path):
    """Schema-incomplete .npz files must surface a clear KeyError."""
    np.savez(tmp_path / "incomplete.npz", positions=np.zeros((1, 1, 2)))
    with pytest.raises(KeyError, match="missing required fields"):
        load_rollout_npz(tmp_path / "incomplete.npz")


# ---------------------------------------------------------------------------
# Constant-velocity case (trivial conservation)
# ---------------------------------------------------------------------------


def test_constant_velocity_mass_conservation_zero():
    case = build_constant_velocity_rollout()
    assert mass_conservation_defect(case.rollout) == case.expected_mass_conservation_defect == 0.0


def test_constant_velocity_energy_drift_zero():
    case = build_constant_velocity_rollout()
    drift = energy_drift(case.rollout)
    # Floating-point sum may pick up ~ machine epsilon; the analytical
    # value is exactly 0, so allow a tight bound rather than ==.
    assert drift < 1e-12, f"constant-velocity energy_drift={drift:.6e} should be ~ machine epsilon"


def test_constant_velocity_dissipation_zero():
    case = build_constant_velocity_rollout()
    assert dissipation_sign_violation(case.rollout) == 0.0


# ---------------------------------------------------------------------------
# Damped-decay case (strictly dissipative; analytic decay)
# ---------------------------------------------------------------------------


def test_damped_decay_mass_conservation_zero():
    case = build_damped_decay_rollout()
    assert mass_conservation_defect(case.rollout) == 0.0


def test_damped_decay_energy_drift_matches_analytic():
    """KE(t) = KE(0) exp(-2 gamma t) ⇒ drift = 1 - exp(-2 gamma T_final).

    The synthetic rollout's velocities are constructed with this exact
    closed form, so the harness's emitted drift should match the
    analytic value to a tight floating-point bound.
    """
    case = build_damped_decay_rollout()
    drift = energy_drift(case.rollout)
    assert case.expected_energy_drift is not None
    assert math.isclose(drift, case.expected_energy_drift, rel_tol=1e-6, abs_tol=1e-9), (
        f"damped-decay energy_drift={drift:.6e} != analytic {case.expected_energy_drift:.6e}"
    )


def test_damped_decay_dissipation_zero():
    """Strictly dissipative rollout: dE/dt < 0 always ⇒ violation = 0."""
    case = build_damped_decay_rollout()
    violation = dissipation_sign_violation(case.rollout)
    assert violation == 0.0, (
        f"damped-decay dissipation_sign_violation={violation:.6e} should be 0 "
        f"(dE/dt < 0 at every step)"
    )


def test_damped_decay_kinetic_energy_series_monotone():
    """KE(t) is strictly decreasing — direct sanity check on the series getter."""
    case = build_damped_decay_rollout()
    energies = kinetic_energy_series(case.rollout)
    de = np.diff(energies)
    assert np.all(de < 0), (
        f"damped-decay KE(t) should be strictly decreasing; saw {np.sum(de >= 0)} "
        f"non-decreasing steps"
    )


# ---------------------------------------------------------------------------
# Deliberate violation case
# ---------------------------------------------------------------------------


def test_energy_growth_dissipation_violation_nonzero():
    """Particle 0 accelerates linearly ⇒ violation > 0."""
    case = build_energy_growth_rollout()
    violation = dissipation_sign_violation(case.rollout)
    assert violation > 0.0, (
        f"energy-growth rollout dissipation_sign_violation={violation:.6e} "
        f"should be strictly positive (dE/dt > 0 at every step)"
    )


def test_energy_growth_energy_drift_nonzero():
    """Linearly-growing speed ⇒ KE grows ⇒ drift = max|E - E(0)|/|E(0)| > 0.

    For the energy-growth synthetic, E(0) = 0 (all particles at rest).
    The denominator-stabilisation in energy_drift falls back to eps =
    1e-12, so the emitted drift is |E_max| / 1e-12 = a large finite
    number. The test asserts non-zero but not a specific magnitude
    because the eps-fallback floor is implementation detail, not
    methodologically meaningful.
    """
    case = build_energy_growth_rollout()
    drift = energy_drift(case.rollout)
    assert drift > 0.0


def test_energy_growth_mass_conservation_zero():
    """Mass is still conserved even when KE is not."""
    case = build_energy_growth_rollout()
    assert mass_conservation_defect(case.rollout) == 0.0


# ---------------------------------------------------------------------------
# Series getter cross-checks
# ---------------------------------------------------------------------------


def test_total_mass_series_constant_in_v1():
    """V1 mass model: per-particle masses are time-invariant ⇒ total constant."""
    case = build_damped_decay_rollout()
    m_series = total_mass_series(case.rollout)
    assert np.all(m_series == m_series[0])
    assert m_series[0] == case.rollout.n_particles  # unit mass per particle


def test_kinetic_energy_series_shape_matches_timesteps():
    case = build_constant_velocity_rollout(n_timesteps=37)
    e_series = kinetic_energy_series(case.rollout)
    assert e_series.shape == (37,)


# ---------------------------------------------------------------------------
# ParticleRollout invariants
# ---------------------------------------------------------------------------


def test_rollout_post_init_rejects_mismatched_shapes():
    """ParticleRollout.__post_init__ should reject shape mismatches."""
    pos = np.zeros((10, 4, 2))
    vel = np.zeros((10, 4, 3))  # mismatched D
    with pytest.raises(ValueError, match=r"positions .* != velocities"):
        ParticleRollout(
            positions=pos,
            velocities=vel,
            particle_type=np.zeros(4, dtype=np.int32),
            particle_mass=np.ones(4),
            dt=0.01,
            domain_box=np.array([[0.0, 0.0], [1.0, 1.0]]),
            metadata={},
        )


def test_rollout_post_init_rejects_2d_positions():
    pos = np.zeros((4, 2))  # missing time axis
    vel = np.zeros((4, 2))
    with pytest.raises(ValueError, match="positions must be"):
        ParticleRollout(
            positions=pos,
            velocities=vel,
            particle_type=np.zeros(4, dtype=np.int32),
            particle_mass=np.ones(4),
            dt=0.01,
            domain_box=np.array([[0.0, 0.0], [1.0, 1.0]]),
            metadata={},
        )


def test_snapshot_at_returns_consistent_view():
    """snapshot_at(t) must align with rollout.positions[t] / .velocities[t]."""
    case = build_constant_velocity_rollout()
    snap = case.rollout.snapshot_at(5)
    np.testing.assert_array_equal(snap.positions, case.rollout.positions[5])
    np.testing.assert_array_equal(snap.velocities, case.rollout.velocities[5])
    np.testing.assert_array_equal(snap.particle_type, case.rollout.particle_type)
    np.testing.assert_array_equal(snap.particle_mass, case.rollout.particle_mass)
