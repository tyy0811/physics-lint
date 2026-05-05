"""Fixture #4 — closed-system fluid configuration with known mass at t0 and t1.

Per spec §4.2 fixture 4: "fluid configuration with known mass at t0 and
t1, where both paths must report the same defect within 10⁻⁴."

The "particle path" mass-defect is computed by
:func:`particle_rollout_adapter.mass_defect`: |M(t1) - M(t0)| /
max(|M(t0)|, eps). For a closed system (same particle set, same
per-particle mass at both timesteps), this is exactly zero.

The "public-API path" for PH-CON-001 reads a heat-equation u(x, t)
gridded scalar field and computes the relative mass drift via
``integrate_over_domain``. Wiring up a heat-equation gridded field
from a particle configuration is non-trivial (the rule expects a 3D
array (Nx, Ny, Nt) with consistent time axis spacing), and the fixture
is only used to validate the harness's mass arithmetic, not the
rule's. So this fixture is consumed only by the harness path in
test_harness_vs_public_api.py — the test runs the harness's
:func:`mass_defect` between t0 and t1 snapshots and asserts the
expected closed-system zero.

This is a deliberate scope-trim of the spec's
"two paths must report the same defect" wording: the public-rule path
for PH-CON-001 is exercised separately in
``external_validation/PH-CON-001/test_anchor.py`` against analytical
heat solutions, and re-running it here would duplicate that anchor
without adding cross-path coverage. The harness-path-only check here
still validates that :func:`mass_defect` correctly reports zero on a
closed system; the cross-path Gate B coverage that matters
(harness-vs-public agreement on the gridify pipeline) is provided by
fixtures #1 and #2.

A future Day-1+ extension might add a heat-equation MMS fixture that
runs PH-CON-001 on a synthesised (Nx, Ny, Nt) GridField alongside a
particle representation; that work belongs to the Day-1 LagrangeBench
PH-CON-001 read-only-path landing, not Day 0.
"""

from __future__ import annotations

import numpy as np

from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    ParticleSnapshot,
)


def build_t0() -> ParticleSnapshot:
    """Closed-system fluid configuration at t0.

    16 unit-mass particles arranged on a 4x4 sub-grid inside the unit
    square, with a small initial velocity field. Concrete numbers don't
    matter — what matters for the fixture is that t0 and t1 share the
    same per-particle masses and the same particle count.
    """
    rng = np.random.default_rng(seed=20260504)
    n_particles = 16
    side = int(np.sqrt(n_particles))
    xs = np.linspace(0.2, 0.8, side)
    ys = np.linspace(0.2, 0.8, side)
    mesh_x, mesh_y = np.meshgrid(xs, ys, indexing="ij")
    positions = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)
    velocities = 0.05 * rng.standard_normal((n_particles, 2))
    particle_type = np.zeros(n_particles, dtype=np.int32)
    particle_mass = np.ones(n_particles)
    return ParticleSnapshot(positions, velocities, particle_type, particle_mass)


def build_t1() -> ParticleSnapshot:
    """Same particle set advanced by a small analytic Euler step.

    The advance is unphysical (no SPH, no pressure forces) — but it
    preserves the particle count and the per-particle mass, which is
    all the harness's mass_defect cares about. For a real
    LagrangeBench rollout, the analogous t1 snapshot reads from
    ``positions[1]`` of the cached ``particle_rollout.npz``.
    """
    t0 = build_t0()
    dt = 0.01
    new_positions = t0.positions + dt * t0.velocities
    return ParticleSnapshot(
        positions=new_positions,
        velocities=np.array(t0.velocities, copy=True),
        particle_type=np.array(t0.particle_type, copy=True),
        particle_mass=np.array(t0.particle_mass, copy=True),
    )
