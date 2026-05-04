"""Fixture #2 — same C4 orbit with one particle displaced by delta.

Per spec §4.2 fixture 2: "same configuration with one particle displaced
by a known δ. **Expected:** both paths emit ε_C4 = O(δ); the two
paths' values agree to within 10⁻⁴."

The "agreement to 10⁻⁴" is the load-bearing claim — both paths route
through the same gridify + rot90 pipeline, so the expected agreement
is in fact bit-identical floating-point output. The test in
test_harness_vs_public_api.py asserts the looser 10⁻⁴ envelope to
absorb future divergence headroom (e.g., if the public rule path
later adds an internal normalisation step).
"""

from __future__ import annotations

import numpy as np

from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    ParticleSnapshot,
)
from external_validation._rollout_anchors._harness.tests.fixtures.c4_invariant_4particle import (
    build_snapshot as build_invariant_snapshot,
)

# Pre-registered perturbation magnitude. 0.01 is chosen so:
#   (a) it is well above floating-point precision (so ε is comfortably > eps);
#   (b) it is well below the bandwidth (0.04), so the resulting density
#       perturbation is in the linear regime and ε is genuinely O(delta).
DELTA: float = 0.01


def build_snapshot(*, delta: float = DELTA) -> ParticleSnapshot:
    """Invariant 4-particle snapshot with particle 0 displaced by (+delta, 0)."""
    base = build_invariant_snapshot()
    positions = np.array(base.positions, copy=True)
    positions[0, 0] += delta
    return ParticleSnapshot(
        positions=positions,
        velocities=np.array(base.velocities, copy=True),
        particle_type=np.array(base.particle_type, copy=True),
        particle_mass=np.array(base.particle_mass, copy=True),
    )
