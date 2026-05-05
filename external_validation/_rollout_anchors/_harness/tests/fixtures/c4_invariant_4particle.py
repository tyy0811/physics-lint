"""Fixture #1 — four particles at an exact C4 orbit; expected ε ~ machine epsilon.

Per spec §4.2 fixture 1: "four particles at the vertices of a square
centered on the origin, with C4-invariant velocity assignment.
**Expected:** both paths emit ε_C4 ≤ 10⁻⁶ (machine precision; the
configuration is exactly C4-invariant by construction)."

Implementation note. ``np.rot90`` is a *discrete* rotation around the
array center, which sits at coordinate ((N-1)/(2N), (N-1)/(2N)) on the
``endpoint=False`` periodic grid — not at (0.5, 0.5). To make the
gridified density exactly invariant under ``np.rot90``, the four
particles must form an exact C4 orbit *under the discrete rotation*,
not under the continuous rotation around (0.5, 0.5). Concretely: place
particles at four cells (i, j), (j, N-1-i), (N-1-i, N-1-j), (N-1-j, i)
which the discrete rotation cycles through. The continuous coordinates
of these cells are ``(i/N, j/N)`` and friends, recentered on the array
center rather than on (0.5, 0.5).

Velocities are zero; the gridify function builds the density from
positions only, so velocities don't enter the C4 defect computation
(the velocity-equivariance test belongs to the Day-1 model-loading
path, see particle_rollout_adapter.py docstring).
"""

from __future__ import annotations

import numpy as np

from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    ParticleSnapshot,
)
from external_validation._rollout_anchors._harness.tests.fixtures.c4_grid_equivalent import (
    GRID_SIZE,
)


def build_snapshot() -> ParticleSnapshot:
    """Four particles at an exact discrete-C4 orbit on the GRID_SIZE grid."""
    n = GRID_SIZE
    # Cell indices forming an exact orbit under np.rot90 (k=1):
    #   (i, j) -> (j, N-1-i) -> (N-1-i, N-1-j) -> (N-1-j, i) -> (i, j).
    # Pick (i, j) = (n // 4, n // 4), giving an orbit at quarter-grid spacing.
    i, j = n // 4, n // 4
    cell_indices = np.array(
        [
            [i, j],
            [j, n - 1 - i],
            [n - 1 - i, n - 1 - j],
            [n - 1 - j, i],
        ],
        dtype=int,
    )
    positions = cell_indices.astype(float) / n  # continuous coordinates on (0, 1)^2
    velocities = np.zeros_like(positions)
    particle_type = np.zeros(positions.shape[0], dtype=np.int32)
    particle_mass = np.ones(positions.shape[0])
    return ParticleSnapshot(
        positions=positions,
        velocities=velocities,
        particle_type=particle_type,
        particle_mass=particle_mass,
    )
