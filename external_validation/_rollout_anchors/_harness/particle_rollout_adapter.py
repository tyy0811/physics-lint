"""Particle-side rollout adapter for `_rollout_anchors/_harness`.

Two halves per spec §3.1:

- **Read-only path** (PH-CON-001/002/003 + Day-0 fixture defect
  computation): reads cached `particle_rollout.npz` (or accepts a
  `ParticleSnapshot` directly), computes mass / KE / static C₄ and
  reflection defects via the gridded-density representation, and
  emits SARIF. Does not require the JAX model object.

- **Model-loading path** (PH-SYM-001/002 on a trained checkpoint's
  rollout): requires `jax`/`haiku` from the `[validation-rollout]`
  extra. **Not implemented in this Day 0 scaffold.** Lives behind
  `pip install 'physics-lint[validation-rollout]'` and is gated on
  actual JAX checkpoint availability (Day 1 work, GPU-bound).

This module is private to `_rollout_anchors/`. It does not expose a
public surface in `physics_lint.field.*`. See spec §1.1 / §2.2.

## Why the static defect uses a gridded density representation

The Day-0 controlled-fixture validation needs a particle-side defect
ε_part that is comparable to what the public PH-SYM-001 rule emits on
the gridded equivalent (Gate B, ε_harness_vs_public ≤ 10⁻⁴). A naïve
``||R x - x||/||x||`` comparison on raw positions gives ε = O(1) on
any honest C₄ orbit, because the rotation *permutes* particle indices
and the unmatched per-index L² picks up the relabelling. To produce a
permutation-invariant defect that agrees numerically with the public
rule, the harness gridifies the particle configuration into a smooth
scalar density field (one Gaussian bump per particle), rotates the
gridded field via ``np.rot90``, and reports the relative L^2 — exactly
the operation PH-SYM-001 performs on its input. Both paths therefore
consume the same gridded representation; Gate B's role is to validate
that the harness's gridify-then-rotate pipeline reproduces the public
rule's emission to within 10⁻⁴ on the controlled fixtures.

The Day-1 model-loading path uses a *different* defect computation —
per-index relative L^2 between ``f(x_0)`` and ``R^{-1} f(R x_0)`` —
because trained-model rollouts preserve particle identity across the
identity-vs-rotated rollout pair. That code does not land here on
Day 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# In-memory snapshot type (not the .npz schema; that lives in SCHEMA.md §1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParticleSnapshot:
    """One timestep of a particle configuration, in harness-internal form.

    Decoupled from the on-disk `.npz` schema so fixtures can construct
    snapshots in-memory without round-tripping through `np.savez`.
    """

    positions: np.ndarray  # (N_particles, D)  fp64 internal
    velocities: np.ndarray  # (N_particles, D)  fp64 internal
    particle_type: np.ndarray  # (N_particles,)
    particle_mass: np.ndarray  # (N_particles,)  uniform unit mass by default

    def __post_init__(self) -> None:
        if self.positions.shape != self.velocities.shape:
            raise ValueError(
                f"positions {self.positions.shape} != velocities {self.velocities.shape}"
            )
        if self.particle_type.shape != (self.positions.shape[0],):
            raise ValueError(
                f"particle_type shape {self.particle_type.shape} must be ({self.positions.shape[0]},)"
            )
        if self.particle_mass.shape != (self.positions.shape[0],):
            raise ValueError(
                f"particle_mass shape {self.particle_mass.shape} must be ({self.positions.shape[0]},)"
            )


# ---------------------------------------------------------------------------
# Gridded-density representation
# ---------------------------------------------------------------------------
#
# The harness's static defects route through this representation because it
# is the natural permutation-invariant view of a particle configuration and
# because it lets us reuse the public PH-SYM-001 rule's `np.rot90` machinery
# directly. Domain default is the unit square (0, 1)^2 with periodic
# wrap-around; this matches the convention of PH-SYM-001's existing
# external-validation anchor (`PH-SYM-001/test_anchor.py`).


def gridify(
    snapshot: ParticleSnapshot,
    *,
    grid_size: int = 64,
    bandwidth: float = 0.04,
    domain: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0)),
) -> np.ndarray:
    """Smooth Gaussian-bump kernel-density estimator on a uniform grid.

    The particle configuration is mapped to a scalar density field:

        u(x, y) = sum_i exp(- ((x - p_i.x)^2 + (y - p_i.y)^2) / (2 sigma^2))

    on the periodic ``[domain.x.min, domain.x.max] x [domain.y.min,
    domain.y.max]`` grid with ``grid_size x grid_size`` points and
    bandwidth ``sigma = bandwidth``.

    The kernel is normalised so that each particle contributes the same
    integrated mass; the absolute scale is irrelevant because every
    downstream defect is a *relative* L^2 ratio.

    D=2 only in V1 — fixtures #1 and #2 are 2D. D=3 lands when the
    corresponding fixture lands.
    """
    pos = snapshot.positions
    if pos.shape[1] != 2:
        raise ValueError(f"gridify requires D=2; got D={pos.shape[1]}")

    (xmin, xmax), (ymin, ymax) = domain
    # endpoint=False matches the periodic convention used in the public
    # PH-SYM-001 anchor (and in physics-lint.field.GridField when
    # periodic=True).
    xs = np.linspace(xmin, xmax, grid_size, endpoint=False)
    ys = np.linspace(ymin, ymax, grid_size, endpoint=False)
    mesh_x, mesh_y = np.meshgrid(xs, ys, indexing="ij")

    lx = xmax - xmin
    ly = ymax - ymin

    u = np.zeros_like(mesh_x)
    inv_two_sigma_sq = 1.0 / (2.0 * bandwidth * bandwidth)

    for px, py in pos:
        # Periodic minimum-image distance — wrap each particle's
        # contribution around the periodic boundary so the gridded
        # field is genuinely C4-equivariant under np.rot90 when the
        # particle set is C4-equivariant.
        dx = mesh_x - px
        dx -= lx * np.round(dx / lx)
        dy = mesh_y - py
        dy -= ly * np.round(dy / ly)
        u += np.exp(-(dx * dx + dy * dy) * inv_two_sigma_sq)

    return u


# ---------------------------------------------------------------------------
# Conservation defects (PH-CON-001/002 read-only path)
# ---------------------------------------------------------------------------


def total_mass(snapshot: ParticleSnapshot) -> float:
    """Σ m_i. Trivially exact under particle conservation."""
    return float(np.sum(snapshot.particle_mass))


def kinetic_energy(snapshot: ParticleSnapshot) -> float:
    """Σ (1/2) m_i ||v_i||²."""
    speeds_sq = np.sum(snapshot.velocities**2, axis=1)
    return float(0.5 * np.sum(snapshot.particle_mass * speeds_sq))


def mass_defect(
    snapshot_t0: ParticleSnapshot,
    snapshot_t1: ParticleSnapshot,
) -> float:
    """Relative absolute drift in total mass: |M(t1) - M(t0)| / max(|M(t0)|, eps).

    For closed-system fluid configurations, both snapshots share the same
    particle set (no creation / destruction), so this is exactly zero.
    """
    m0 = total_mass(snapshot_t0)
    m1 = total_mass(snapshot_t1)
    eps = 1e-12
    return float(abs(m1 - m0) / max(abs(m0), eps))


# ---------------------------------------------------------------------------
# Symmetry defects (Day-0 static, gridded representation)
# ---------------------------------------------------------------------------


def _relative_l2(diff: np.ndarray, ref: np.ndarray, *, eps: float = 1e-12) -> float:
    """``||diff|| / max(||ref||, eps)`` — same convention as the public rule.

    Mirrors `physics_lint.rules._symmetry_helpers.equivariance_error_np`
    so the harness emits a comparable scalar to what PH-SYM-001 / 002
    report on the gridded equivalent.
    """
    denom = float(np.linalg.norm(ref))
    return float(np.linalg.norm(diff) / max(denom, eps))


def c4_static_defect(
    snapshot: ParticleSnapshot,
    *,
    grid_size: int = 64,
    bandwidth: float = 0.04,
    domain: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0)),
) -> float:
    """Max relative L^2 of (u - rot90(u, k)) for k in {1, 2, 3}.

    Mirrors PH-SYM-001's emission verbatim: the public rule takes the
    max over the three non-identity C₄ generators (k=1, 2, 3) so the
    harness does the same. For an exactly C₄-invariant gridified
    configuration, all three are zero to floating-point precision.

    Uses the periodic gridded-density representation produced by
    :func:`gridify`. The "particle path" is therefore literally the
    same sequence of operations the public rule applies to a
    pre-gridified GridField — that is the whole point of Gate B.
    """
    u = gridify(
        snapshot,
        grid_size=grid_size,
        bandwidth=bandwidth,
        domain=domain,
    )
    errors = [_relative_l2(u - np.rot90(u, k=k), u) for k in (1, 2, 3)]
    return max(errors)


def reflection_static_defect(
    snapshot: ParticleSnapshot,
    *,
    axis: int = 0,
    grid_size: int = 64,
    bandwidth: float = 0.04,
    domain: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0)),
) -> float:
    """Relative L^2 of (u - flip(u, axis)) on the gridded density.

    For an axis-symmetric configuration this is zero to floating-point
    precision. ``axis`` matches numpy's axis convention (0 ≡ x-axis
    flip on a 2D ij-indexed array).

    Note: the gridify function builds the density from positions only;
    velocities do not enter the defect. PH-SYM-002 on the public-API
    path (gridded scalar field) likewise sees only the scalar field,
    not the underlying particle velocities. Reflection-equivariance of
    the velocity field is an orthogonal test that lives on the Day-1
    model-loading path (rotated rollouts compared per-index).
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1 for D=2; got {axis}")
    u = gridify(
        snapshot,
        grid_size=grid_size,
        bandwidth=bandwidth,
        domain=domain,
    )
    return _relative_l2(u - np.flip(u, axis=axis), u)


# ---------------------------------------------------------------------------
# Model-loading path (PH-SYM-001/002 on trained checkpoints, Day 1 work)
# ---------------------------------------------------------------------------
#
# Deliberately not implemented in the Day 0 scaffold. The model-loading
# half requires `jax`/`haiku` from the `[validation-rollout]` extra and a
# real LagrangeBench checkpoint to load — both unavailable in the Day 0
# CPU-only environment. Per the executing agent's "no speculative stubs"
# rule, this code lands in a separate commit on Day 1 once a checkpoint
# is verifiably available and the JAX micro-gate (plan §7) has passed.
#
# Until then, callers requesting the model-loading path will receive an
# explicit `ModuleNotFoundError` from `import jax` upstream — that is the
# intended behaviour.
