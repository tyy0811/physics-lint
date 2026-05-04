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
from pathlib import Path
from typing import Any

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
# Rollout-level read-only path (Day 0.5)
# ---------------------------------------------------------------------------
#
# Time-resolved analogues of PH-CON-001/002/003 on cached `.npz` rollouts,
# computed directly from particle positions, velocities, and per-particle
# mass — no JAX dependency. Emission forms mirror the public rules so the
# harness output is comparable in scalar shape:
#
#   PH-CON-001 (mass)        → max |M(t) - M(0)| / max(|M(0)|, eps)
#   PH-CON-002 (energy)      → max |E(t) - E(0)| / max(|E(0)|, eps)
#   PH-CON-003 (dE/dt sign)  → max(0, max(dE/dt)) / max(|E_max|, eps)
#
# All three functions return a HarnessDefect dataclass that is either
# (value=numeric, skip_reason=None) for a computed defect or
# (value=None, skip_reason=str) for an input-domain mismatch — see
# DECISIONS.md D0-08 for the KE-rest skip threshold pre-registration.
# This polymorphic shape mirrors physics_lint.report.RuleResult's
# (status="PASS"/"WARN"/"FAIL"/"SKIPPED", raw_value=...) idiom.
#
# Caveat per `physics-lint-validation/DECISIONS.md` D0-03: the public
# PH-CON-001/002/003 rules are heat-or-wave-only in V1 (return SKIPPED on
# `pde != "heat"`/"wave"). The harness's rollout-level functions below
# *reapply the structural-conservation identities* on particle data —
# this is structural-identity reapplication, not a public-API rule
# invocation. The cross-stack table in `_rollout_anchors/README.md`
# captures the routing.

# Pre-registered KE-rest skip-with-reason threshold per
# DECISIONS.md D0-08. Absolute, in the dataset's natural KE units; v1
# scope. v1.1 may switch to a relative-within-rollout form once
# cross-dataset comparison becomes load-bearing. If a real dataset
# surfaces KE(0) within an order of magnitude of this threshold, log a
# new D0-09+ entry and amend; do not silently shift.
KE_REST_THRESHOLD: float = 1e-10


# DECISIONS.md D0-18: PH-CON-002 skip-with-reason gate for
# dissipative-by-design systems. The harness's `energy_drift` measures
# max|KE(t) - KE(0)| / |KE(0)|, which is zero for conservative rollouts
# and grows toward 1 for dissipative ones (TGV2D under viscous decay
# dissipates ~99.99% of initial KE over one viscous time scale; the
# spot-check on SEGNN-TGV2D traj00 surfaced this at 0.9999). The numeric
# value is a correct measurement of dissipation magnitude; the
# rule-semantics interpretation as "drift = FAIL" is wrong for
# dissipative systems. Skip-with-reason restores the right semantics by
# emitting SKIPPED + a reason string instead of a numeric defect that
# downstream PASS/FAIL classification will misread.
#
# v1 implementation: hardcoded mapping of LagrangeBench dataset names
# to system_class. v1.1 promotes to a metadata field on the npz so
# non-LB datasets can declare their class without requiring an update
# here. All five SPH datasets in the LB corpus are dissipative:
# TGV2D / RPF2D viscous Navier-Stokes; LDC2D wall-bounded forced
# convection but still viscously dissipative without sustained
# forcing; DAM2D free-surface dam break (gravity-driven, viscous
# dissipation in the SPH solver). DOM3D / TGV3D / RPF3D / LDC3D
# inherit from their 2D analogs (P3 stretch only per plan §3.1).
#
# Two-half positive-evidence gate (D0-18): skip ONLY when both
# (a) system_class hint says "dissipative" AND
# (b) KE(t) is monotone-non-increasing across the rollout.
# Either alone is insufficient: monotone-decreasing-without-hint
# could be a buggy supposed-conservative surrogate; hint-without-
# monotonicity could be a buggy "dissipative" model that's actually
# gaining energy somewhere. Both required → defaults to existing
# fire-raw-value behavior absent positive evidence on either axis.
LAGRANGEBENCH_DATASET_SYSTEM_CLASS: dict[str, str] = {
    "tgv2d": "dissipative",
    "rpf2d": "dissipative",
    "ldc2d": "dissipative",
    "dam2d": "dissipative",
    "tgv3d": "dissipative",
    "rpf3d": "dissipative",
    "ldc3d": "dissipative",
}


@dataclass(frozen=True)
class HarnessDefect:
    """Result of a rollout-level defect computation.

    Mirrors the polymorphic shape of ``physics_lint.report.RuleResult``:
    either ``value`` is set (computed numeric defect, ready for the
    SARIF properties.raw_value field) or ``value is None`` and
    ``skip_reason`` is set (input-domain mismatch — analogous to
    ``RuleResult.status = "SKIPPED"``).

    Exactly one of ``value`` / ``skip_reason`` must be populated; the
    constructor enforces this so downstream consumers can branch on
    ``defect.value is None`` without ambiguity.
    """

    value: float | None = None
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.skip_reason is None):
            raise ValueError(
                "HarnessDefect must have exactly one of value, skip_reason set; "
                f"got value={self.value!r}, skip_reason={self.skip_reason!r}"
            )


@dataclass(frozen=True)
class ParticleRollout:
    """A trajectory of T snapshots, in harness-internal form.

    Decoupled from the on-disk `.npz` schema (`SCHEMA.md` §1) so synthetic
    fixtures can construct rollouts in-memory and consumers can construct
    them from cached files via :func:`load_rollout_npz`.

    Particle count and per-particle mass / type are constant across the
    trajectory; particle creation / destruction is not supported in V1.
    """

    positions: np.ndarray  # (T, N_particles, D)  fp64 internal
    velocities: np.ndarray  # (T, N_particles, D)  fp64 internal
    particle_type: np.ndarray  # (N_particles,)
    particle_mass: np.ndarray  # (N_particles,)
    dt: float
    domain_box: np.ndarray  # (2, D)  [[xmin,...], [xmax,...]]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.positions.shape != self.velocities.shape:
            raise ValueError(
                f"positions {self.positions.shape} != velocities {self.velocities.shape}"
            )
        if self.positions.ndim != 3:
            raise ValueError(f"positions must be (T, N, D); got shape {self.positions.shape}")
        n_particles = self.positions.shape[1]
        if self.particle_type.shape != (n_particles,):
            raise ValueError(
                f"particle_type shape {self.particle_type.shape} must be ({n_particles},)"
            )
        if self.particle_mass.shape != (n_particles,):
            raise ValueError(
                f"particle_mass shape {self.particle_mass.shape} must be ({n_particles},)"
            )
        if self.domain_box.shape != (2, self.positions.shape[2]):
            raise ValueError(
                f"domain_box shape {self.domain_box.shape} must be (2, D={self.positions.shape[2]})"
            )

    @property
    def n_timesteps(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[1])

    def snapshot_at(self, t_idx: int) -> ParticleSnapshot:
        """Return the t_idx-th timestep as a snapshot-shaped view."""
        return ParticleSnapshot(
            positions=self.positions[t_idx],
            velocities=self.velocities[t_idx],
            particle_type=self.particle_type,
            particle_mass=self.particle_mass,
        )


def load_rollout_npz(path: Path | str) -> ParticleRollout:
    """Read a `particle_rollout.npz` file per `SCHEMA.md` §1.

    Uses ``np.load(allow_pickle=True)`` to round-trip the metadata dict;
    metadata is validated only loosely (presence of required keys per
    `SCHEMA.md`, types not deeply checked). The caller is expected to
    have already validated checkpoint hashes / git SHAs / framework
    versions out-of-band.

    Returns
    -------
    ParticleRollout

    Raises
    ------
    KeyError
        If required schema fields are missing from the .npz.
    ValueError
        If shapes / dtypes are inconsistent with the schema.
    """
    p = Path(path)
    with np.load(p, allow_pickle=True) as data:
        required = {
            "positions",
            "velocities",
            "particle_type",
            "particle_mass",
            "dt",
            "domain_box",
            "metadata",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"particle_rollout.npz {p} missing required fields: {sorted(missing)}")
        positions = np.asarray(data["positions"], dtype=float)
        velocities = np.asarray(data["velocities"], dtype=float)
        particle_type = np.asarray(data["particle_type"])
        particle_mass = np.asarray(data["particle_mass"], dtype=float)
        dt_arr = data["dt"]
        dt = float(dt_arr.item() if hasattr(dt_arr, "item") else dt_arr)
        domain_box = np.asarray(data["domain_box"], dtype=float)
        meta_obj = data["metadata"]
        metadata: dict[str, Any] = meta_obj.item() if hasattr(meta_obj, "item") else dict(meta_obj)
    return ParticleRollout(
        positions=positions,
        velocities=velocities,
        particle_type=particle_type,
        particle_mass=particle_mass,
        dt=dt,
        domain_box=domain_box,
        metadata=metadata,
    )


def save_rollout_npz(rollout: ParticleRollout, path: Path | str) -> Path:
    """Write a `ParticleRollout` to disk per `SCHEMA.md` §1.

    Round-trippable with :func:`load_rollout_npz`. Used by the synthetic
    fixture builders under `_harness/tests/synthetic_rollouts.py` to
    produce real on-disk `.npz` files for round-trip tests.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        positions=rollout.positions.astype(np.float32),
        velocities=rollout.velocities.astype(np.float32),
        particle_type=rollout.particle_type.astype(np.int32),
        particle_mass=rollout.particle_mass.astype(np.float64),
        dt=np.float64(rollout.dt),
        domain_box=rollout.domain_box.astype(np.float64),
        metadata=np.array(rollout.metadata, dtype=object),
    )
    return out


def total_mass_series(rollout: ParticleRollout) -> np.ndarray:
    """(T,) array of M(t) = Σ m_i across all particles at each timestep.

    Particle masses are time-invariant in V1, so this is constant. The
    function exists for symmetry with :func:`kinetic_energy_series` and
    so future versions that admit particle creation / destruction can
    populate a non-trivial series without changing the consumer surface.
    """
    return np.broadcast_to(np.sum(rollout.particle_mass), (rollout.n_timesteps,)).astype(float)


def kinetic_energy_series(rollout: ParticleRollout) -> np.ndarray:
    """(T,) array of KE(t) = Σ (1/2) m_i ||v_i(t)||²."""
    # rollout.velocities is (T, N, D); per-particle KE_i(t) = 0.5 m_i |v_i(t)|^2.
    speeds_sq = np.sum(rollout.velocities**2, axis=2)  # (T, N)
    weighted = 0.5 * speeds_sq * rollout.particle_mass[None, :]  # (T, N)
    return np.sum(weighted, axis=1).astype(float)  # (T,)


def mass_conservation_defect(rollout: ParticleRollout) -> HarnessDefect:
    """max |M(t) - M(0)| / max(|M(0)|, eps).

    Mirrors PH-CON-001's emitted `relative_drift` form. Zero for
    closed-system rollouts (LagrangeBench TGV / dam break with fixed
    particle count and time-invariant per-particle mass).

    Always returns a numeric value (never skips): M(0) = sum(particle_mass)
    is strictly positive in any physical configuration. The HarnessDefect
    return type is preserved for downstream SARIF emission symmetry with
    energy_drift / dissipation_sign_violation per DECISIONS.md D0-08.
    """
    m_series = total_mass_series(rollout)
    m0 = float(m_series[0])
    drift = float(np.max(np.abs(m_series - m0)))
    eps = 1e-12
    return HarnessDefect(value=drift / max(abs(m0), eps))


def energy_drift(rollout: ParticleRollout) -> HarnessDefect:
    """max |E(t) - E(0)| / max(|E(0)|, eps), or SKIP if input-domain mismatch.

    Mirrors PH-CON-002's emitted `drift` form on E = kinetic energy
    (potential energy lives in inter-particle interactions and is not
    accessible from positions+velocities alone — this is the "what
    physics-lint did NOT catch" caveat for KE-only rollouts; see
    `_rollout_anchors/README.md`).

    Zero for conservative rollouts; non-zero and growing for dissipative
    rollouts. Two skip-with-reason paths:

    1. **KE-rest** (DECISIONS.md D0-08): SKIPS when KE(0) <
       ``KE_REST_THRESHOLD``. A near-zero KE(0) makes the relative
       drift undefined, and the eps-floored denominator otherwise
       inflates the emitted value to a meaningless large number.

    2. **Dissipative-by-design** (DECISIONS.md D0-18): SKIPS when both
       (a) ``rollout.metadata["dataset"]`` resolves to a known
       dissipative system via ``LAGRANGEBENCH_DATASET_SYSTEM_CLASS``,
       AND (b) ``KE(t)`` is monotone-non-increasing across the rollout.
       The dissipation magnitude IS the physics for these systems
       (TGV2D under viscous decay dissipates ~99.99% of initial KE);
       the relative-drift form is a meaningful conservation test for
       conservative PDEs (wave, Schrödinger) but a misfire for
       dissipative ones. Both halves of the gate required to avoid
       masking buggy supposed-conservative surrogates.

    The harness emits the raw drift (or a SKIP); downstream
    interpretation against PH-CON-002's tristate floor classification
    is left to the test harness or to Day 1+'s SARIF emitter.
    """
    e_series = kinetic_energy_series(rollout)
    e0 = float(e_series[0])
    if abs(e0) < KE_REST_THRESHOLD:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"KE(0)={e0:.3e} < {KE_REST_THRESHOLD:.0e} (rollout starts at "
                f"rest; relative drift undefined; see DECISIONS.md D0-08)"
            ),
        )
    # D0-18 skip-with-reason gate: positive evidence on both axes.
    dataset_name = rollout.metadata.get("dataset", "") if rollout.metadata else ""
    system_class = LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get(dataset_name)
    is_monotone_decreasing = bool(np.all(np.diff(e_series) <= 0))
    if system_class == "dissipative" and is_monotone_decreasing:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"system_class='dissipative' (dataset={dataset_name!r}) and "
                f"KE(t) monotone-non-increasing across the rollout (KE(0)={e0:.3e}, "
                f"KE(end)={float(e_series[-1]):.3e}); relative drift is a "
                f"misfire for dissipative-by-design systems where the "
                f"dissipation magnitude IS the physics. See "
                f"DECISIONS.md D0-18; consult dissipation_sign_violation "
                f"for the load-bearing test on this system class."
            ),
        )
    drift = float(np.max(np.abs(e_series - e0)))
    return HarnessDefect(value=drift / abs(e0))


def dissipation_sign_violation(rollout: ParticleRollout) -> HarnessDefect:
    """max(0, max(dE/dt)) / max(|E_max|, eps), or SKIP if max(KE) below threshold.

    Mirrors PH-CON-003's emitted `violation` form. Zero for strictly
    dissipative or strictly conservative rollouts (dE/dt ≤ 0 or = 0
    everywhere); non-zero for rollouts where the model spuriously gains
    energy at any timestep. SKIPS with reason when ``max(KE) <
    KE_REST_THRESHOLD`` (the trajectory has effectively no kinetic
    energy at any timestep, so the dissipation question is meaningless;
    pre-registered in DECISIONS.md D0-08).

    Uses forward differences ``np.diff(E) / dt`` to match PH-CON-003's
    Week-2 endpoint-pathology fix verbatim — second-order ``np.gradient``
    edge-extrapolation produces spurious positive endpoint slopes on
    fast-decaying signals; forward differences sample at nt - 1 step
    boundaries and have no such pathology.
    """
    if rollout.n_timesteps < 2:
        raise ValueError(
            f"dissipation_sign_violation needs at least 2 timesteps; got {rollout.n_timesteps}"
        )
    e_series = kinetic_energy_series(rollout)
    e_max = float(np.max(e_series))
    if e_max < KE_REST_THRESHOLD:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"max(KE)={e_max:.3e} < {KE_REST_THRESHOLD:.0e} (trajectory "
                f"has no kinetic energy; dissipation question undefined; "
                f"see DECISIONS.md D0-08)"
            ),
        )
    de_dt = np.diff(e_series) / rollout.dt
    max_growth = float(np.max(de_dt))
    return HarnessDefect(value=max(0.0, max_growth) / e_max)


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
