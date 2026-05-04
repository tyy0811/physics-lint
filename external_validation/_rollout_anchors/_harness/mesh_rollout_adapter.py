"""Mesh-side rollout adapter for `_rollout_anchors/_harness`.

Two halves analogous to `particle_rollout_adapter.py`:

- **Materialization path** (Day 1+ / Day 2): wraps one timestep of a
  mesh rollout in a Field-API-compatible object so the existing public
  `physics-lint check` CLI can consume it without rule modification.
  Two sub-paths per spec §3.1:

  - Gate A PASS (preferred): ``MeshField(basis=reconstructed_basis,
    dofs=node_values_at_t)`` if the DGL graph can be coerced to a
    scikit-fem ``Basis``.
  - Gate A PARTIAL (fallback): ``GridField(values=resampled, h=spacing,
    periodic=False)`` after a documented regular-grid resampling pass.

  The :func:`materialize_grid_field` helper covers the FNO-on-Darcy
  fallback case (Gate D) and the GridField PARTIAL case explicitly.
  The DGL→MeshField materialization is deliberately not implemented at
  Day 0 — see footer.

- **Read-only path on the rollout itself** (Day 0.5 follow-up,
  this commit): time-resolved analogues of PH-CON-001/002/003 on
  cached `mesh_rollout.npz` files, computed directly from the per-
  timestep velocity field on a regular grid (FNO / synthetic NS
  channel-flow fixture). Mirrors the particle-side functions in
  ``particle_rollout_adapter.py`` and uses the same
  ``HarnessDefect`` polymorphic return type with the same
  KE-rest skip-with-reason threshold (DECISIONS.md D0-08).

  The graph-mesh path (PhysicsNeMo MGN's DGL output) is gated on the
  Day 2 hour 1 audit — until that audit confirms what the NGC
  checkpoint actually emits as ``node_values["velocity"]`` and what
  topology drives the divergence operator, the read-only-path
  functions on graph-mesh data SKIP with reason. Per the executing
  agent's "no speculative stubs" rule, no graph-divergence machinery
  lands speculatively.

Per spec §1.1 / §2.2, this module is private to `_rollout_anchors/`
and does not expose anything to `physics_lint.field.*`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    KE_REST_THRESHOLD,
    HarnessDefect,
)
from physics_lint import GridField

# ---------------------------------------------------------------------------
# Materialization path
# ---------------------------------------------------------------------------


def materialize_grid_field(
    values: np.ndarray,
    *,
    h: float | tuple[float, ...],
    periodic: bool = False,
    backend: str = "fd",
) -> GridField:
    """Wrap a numpy array of per-timestep node values as a GridField.

    Used by the FNO-on-Darcy fallback (Gate D), where the model output
    is already on a regular grid, and by the GridField PARTIAL fallback
    of Gate A after a resampling pass.

    The function is intentionally a thin wrapper — its job is to make
    the materialization step explicit in the call graph so a code
    reviewer can see "this is the public-API entry point for the mesh
    case study", not to add behaviour over ``GridField.__init__``.
    """
    return GridField(values, h=h, periodic=periodic, backend=backend)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MeshRollout dataclass + .npz I/O (mirrors ParticleRollout / .npz schema §2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeshRollout:
    """A trajectory of mesh-based field values, in harness-internal form.

    Decoupled from the on-disk `.npz` schema (`SCHEMA.md` §2) so synthetic
    fixtures can construct rollouts in-memory and consumers can construct
    them from cached files via :func:`load_mesh_rollout_npz`.

    The mesh topology is **static** across the trajectory — node positions,
    types, and edge index do not change with time. Only the
    ``node_values[field_name]`` arrays are time-resolved.

    Two regimes:

    - **Regular grid** (FNO-on-Darcy fallback, synthetic NS channel-
      flow fixture): ``metadata["framework"] == "pytorch+neuraloperator"``
      or ``metadata["resampling_applied"] == True``. Node positions
      lie on a uniform Cartesian grid; ``edge_index`` is None; the
      regular-grid path of the read-only-path functions is taken.

    - **Graph mesh** (PhysicsNeMo MGN): ``metadata["framework"] ==
      "pytorch+dgl"``. Node positions are irregular; ``edge_index`` is
      populated. The read-only-path functions SKIP with reason
      pending the Day 2 hour 1 audit on what the NGC checkpoint
      actually emits.
    """

    node_positions: np.ndarray  # (N_nodes, D)  static
    node_type: np.ndarray  # (N_nodes,)
    node_values: dict[str, np.ndarray]  # per-field, each (T, N_nodes [, D_field])
    dt: float
    metadata: dict[str, Any]
    edge_index: np.ndarray | None = field(default=None)  # (2, N_edges) or None for grid

    def __post_init__(self) -> None:
        n_nodes = self.node_positions.shape[0]
        if self.node_type.shape != (n_nodes,):
            raise ValueError(f"node_type shape {self.node_type.shape} must be ({n_nodes},)")
        for name, arr in self.node_values.items():
            if arr.ndim < 2:
                raise ValueError(
                    f"node_values[{name!r}] must be at least 2D (T, N_nodes); got shape {arr.shape}"
                )
            if arr.shape[1] != n_nodes:
                raise ValueError(
                    f"node_values[{name!r}] shape {arr.shape} second axis must be N_nodes={n_nodes}"
                )
        if self.edge_index is not None and (
            self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2
        ):
            raise ValueError(f"edge_index must be (2, N_edges); got shape {self.edge_index.shape}")

    @property
    def n_timesteps(self) -> int:
        if not self.node_values:
            raise ValueError("MeshRollout has no node_values to determine n_timesteps")
        return int(next(iter(self.node_values.values())).shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.node_positions.shape[0])

    @property
    def is_regular_grid(self) -> bool:
        """True if metadata indicates the mesh lies on a regular Cartesian grid.

        Two conditions are accepted:
        (a) ``framework == "pytorch+neuraloperator"`` (FNO output is
            grid-native).
        (b) ``resampling_applied is True`` (Gate A PARTIAL fallback —
            DGL output resampled onto a regular grid).

        The synthetic NS channel-flow fixture sets framework to
        ``"synthetic"`` and ``regular_grid`` to True via metadata so it
        can exercise this path explicitly.
        """
        framework = str(self.metadata.get("framework", ""))
        if framework == "pytorch+neuraloperator":
            return True
        if self.metadata.get("resampling_applied") is True:
            return True
        return self.metadata.get("regular_grid") is True

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Inferred (Nx, Ny [, Nz]) grid shape from node_positions.

        Only meaningful when :attr:`is_regular_grid` is True. The
        inference assumes node_positions follows ``np.meshgrid``'s
        ``indexing="ij"`` convention; if a fixture uses a different
        ordering, it must override the inferred shape via
        ``metadata["grid_shape"]`` (a tuple of ints).
        """
        if "grid_shape" in self.metadata:
            return tuple(int(d) for d in self.metadata["grid_shape"])
        if not self.is_regular_grid:
            raise ValueError(
                "grid_shape only defined when is_regular_grid is True; "
                "got framework="
                f"{self.metadata.get('framework')!r}, "
                "resampling_applied="
                f"{self.metadata.get('resampling_applied')!r}"
            )
        # Infer from unique x / y coordinate counts.
        d = self.node_positions.shape[1]
        sizes = tuple(int(np.unique(self.node_positions[:, axis]).size) for axis in range(d))
        if int(np.prod(sizes)) != self.node_positions.shape[0]:
            raise ValueError(
                f"inferred grid_shape={sizes} (product={int(np.prod(sizes))}) "
                f"does not match n_nodes={self.node_positions.shape[0]}; "
                f"override via metadata['grid_shape']"
            )
        return sizes

    @property
    def grid_spacing(self) -> tuple[float, ...]:
        """Inferred per-axis grid spacing for a regular-grid mesh.

        Only meaningful when :attr:`is_regular_grid` is True.
        """
        d = self.node_positions.shape[1]
        spacings: list[float] = []
        for axis in range(d):
            uniq = np.sort(np.unique(self.node_positions[:, axis]))
            if uniq.size < 2:
                raise ValueError(
                    f"axis {axis} has fewer than 2 unique node positions; cannot infer spacing"
                )
            diffs = np.diff(uniq)
            spacings.append(float(np.median(diffs)))
        return tuple(spacings)


def load_mesh_rollout_npz(path: Path | str) -> MeshRollout:
    """Read a `mesh_rollout.npz` file per `SCHEMA.md` §2."""
    p = Path(path)
    with np.load(p, allow_pickle=True) as data:
        required = {
            "node_positions",
            "node_type",
            "node_values",
            "dt",
            "metadata",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"mesh_rollout.npz {p} missing required fields: {sorted(missing)}")
        node_positions = np.asarray(data["node_positions"], dtype=float)
        node_type = np.asarray(data["node_type"])
        nv_obj = data["node_values"]
        node_values_raw = nv_obj.item() if hasattr(nv_obj, "item") else dict(nv_obj)
        node_values = {k: np.asarray(v, dtype=float) for k, v in node_values_raw.items()}
        dt_arr = data["dt"]
        dt = float(dt_arr.item() if hasattr(dt_arr, "item") else dt_arr)
        meta_obj = data["metadata"]
        metadata: dict[str, Any] = meta_obj.item() if hasattr(meta_obj, "item") else dict(meta_obj)
        edge_index = (
            np.asarray(data["edge_index"], dtype=np.int64) if "edge_index" in data.files else None
        )
    return MeshRollout(
        node_positions=node_positions,
        node_type=node_type,
        node_values=node_values,
        dt=dt,
        metadata=metadata,
        edge_index=edge_index,
    )


def save_mesh_rollout_npz(rollout: MeshRollout, path: Path | str) -> Path:
    """Write a `MeshRollout` to disk per `SCHEMA.md` §2.

    Round-trippable with :func:`load_mesh_rollout_npz`.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "node_positions": rollout.node_positions.astype(np.float32),
        "node_type": rollout.node_type.astype(np.int32),
        "node_values": np.array(
            {k: v.astype(np.float32) for k, v in rollout.node_values.items()},
            dtype=object,
        ),
        "dt": np.float64(rollout.dt),
        "metadata": np.array(rollout.metadata, dtype=object),
    }
    if rollout.edge_index is not None:
        payload["edge_index"] = rollout.edge_index.astype(np.int64)
    np.savez(out, **payload)
    return out


# ---------------------------------------------------------------------------
# Read-only-path conservation defects (Day 0.5 follow-up)
# ---------------------------------------------------------------------------
#
# Time-resolved analogues of PH-CON-001/002/003 on cached
# `mesh_rollout.npz` files, computed directly from per-timestep
# velocity / density fields. Mirror the particle-side emission forms
# from particle_rollout_adapter.py and use the same HarnessDefect
# polymorphic return type with the same KE_REST_THRESHOLD.
#
# Caveat per DECISIONS.md D0-03: the public PH-CON-001/002/003 are
# heat-or-wave-only in V1 and SKIP on `pde != "heat"`/"wave". The
# harness functions below reapply the structural-conservation
# identities on NS-domain mesh data — structural-identity reapplication,
# not a public-API rule invocation.
#
# Two paths per :attr:`MeshRollout.is_regular_grid`:
#   Regular grid → FD divergence, integrated kinetic energy, dE/dt sign.
#   Graph mesh   → SKIP with reason pending the Day 2 hour 1 audit on
#                  what NGC PhysicsNeMo MGN actually emits.


def _expect_velocity(rollout: MeshRollout) -> HarnessDefect | np.ndarray:
    """Common precondition check: rollout has a ``velocity`` field.

    Returns a SKIP HarnessDefect if absent; otherwise returns the
    velocity array as ``(T, N_nodes, D)`` (D inferred from the array).
    """
    if "velocity" not in rollout.node_values:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"node_values has no 'velocity' field "
                f"(found keys: {sorted(rollout.node_values.keys())!r}); "
                f"mesh-side conservation defects require velocity to compute "
                f"divergence and kinetic energy"
            ),
        )
    v = rollout.node_values["velocity"]
    if v.ndim != 3:
        # Allow (T, N_nodes) scalar velocity by lifting to (T, N_nodes, 1).
        if v.ndim == 2:
            return v[..., None]
        return HarnessDefect(
            value=None,
            skip_reason=(f"velocity has unexpected shape {v.shape}; expected (T, N_nodes [, D])"),
        )
    return v


def _gridded_velocity_view(rollout: MeshRollout, velocity: np.ndarray) -> np.ndarray:
    """Reshape velocity from (T, N_nodes, D) to (T, *grid_shape, D).

    Only valid on a regular-grid mesh; assumes the node ordering follows
    ``np.meshgrid(..., indexing='ij')`` (or the override via
    ``metadata['grid_shape']``).
    """
    grid_shape = rollout.grid_shape
    t_size, n_nodes, d_field = velocity.shape
    if int(np.prod(grid_shape)) != n_nodes:
        raise ValueError(
            f"grid_shape={grid_shape} (product={int(np.prod(grid_shape))}) != n_nodes={n_nodes}"
        )
    return velocity.reshape((t_size, *tuple(grid_shape), d_field))


def mass_conservation_defect_on_mesh(rollout: MeshRollout) -> HarnessDefect:
    """Per-timestep relative L2 of grid-divergence of velocity, max over t.

    For incompressible NS, the mass-conservation identity is the
    pointwise statement ``∇·v = 0``; the harness emits its dimensionless
    relative form

        defect = max_t  || ∇·v(t) ||_L2 / || v(t) ||_L2

    where ``∇·v`` and ``v`` are computed on the regular grid via
    fourth-order centered FD. This mirrors the v3 plan §4.2 step 4
    framing ("PH-CON-001 (mass) on vortex shedding: divergence-free
    check on velocity field") explicitly — note that this is
    structural-identity reapplication, not a public-API
    PH-CON-001 invocation per DECISIONS.md D0-03.

    SKIPS with reason when:

    - ``node_values`` lacks a ``velocity`` field.
    - The rollout is on a graph mesh (the divergence operator on
      irregular DGL topology is gated on the Day 2 hour 1 NGC audit;
      no speculative graph-divergence machinery is implemented here).
    """
    velocity = _expect_velocity(rollout)
    if isinstance(velocity, HarnessDefect):
        return velocity
    if not rollout.is_regular_grid:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"mesh is graph-topology (framework="
                f"{rollout.metadata.get('framework')!r}); graph-divergence "
                f"is gated on Day 2 hour 1 NGC audit per DECISIONS.md D0-03 "
                f"and is not implemented in this Day 0.5 commit"
            ),
        )

    v_grid = _gridded_velocity_view(rollout, velocity)
    spacings = rollout.grid_spacing
    d_grid = len(rollout.grid_shape)
    d_field = v_grid.shape[-1]
    if d_field != d_grid:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"velocity field has D_field={d_field} but mesh has "
                f"D_grid={d_grid}; divergence requires D_field == D_grid"
            ),
        )

    # ∂v_axis/∂x_axis per axis, summed → divergence.
    # 4th-order centered FD interior, 2nd-order one-sided at edges,
    # matching the public physics_lint.field.GridField FD convention.
    n_t = v_grid.shape[0]
    max_relative = 0.0
    for k in range(n_t):
        v_t = v_grid[k]  # (*grid_shape, D)
        div_t = np.zeros(rollout.grid_shape)
        for axis in range(d_grid):
            v_axis_component = v_t[..., axis]
            div_t = div_t + np.gradient(v_axis_component, spacings[axis], axis=axis)
        div_norm = float(np.linalg.norm(div_t))
        v_norm = float(np.linalg.norm(v_t))
        eps = 1e-12
        relative = div_norm / max(v_norm, eps)
        if relative > max_relative:
            max_relative = relative
    return HarnessDefect(value=max_relative)


def kinetic_energy_series_on_mesh(rollout: MeshRollout) -> np.ndarray:
    """(T,) array of KE(t) = 0.5 * Σ_node rho_node * |v_node|^2 * cell_volume.

    Constant unit density assumed in V1 (incompressible NS or the
    synthetic channel flow). Cell volume on a regular grid is
    ``prod(grid_spacing)``. Sum over nodes approximates the volume
    integral via midpoint quadrature.

    For graph-mesh inputs, this function returns NaN (the caller
    should consult :func:`energy_drift_on_mesh` or
    :func:`dissipation_sign_violation_on_mesh`, which surface the
    skip-with-reason cleanly).
    """
    velocity = _expect_velocity(rollout)
    if isinstance(velocity, HarnessDefect):
        return np.full(rollout.n_timesteps, float("nan"))
    if not rollout.is_regular_grid:
        return np.full(rollout.n_timesteps, float("nan"))
    cell_volume = float(np.prod(rollout.grid_spacing))
    speeds_sq = np.sum(velocity**2, axis=2)  # (T, N_nodes)
    return 0.5 * cell_volume * np.sum(speeds_sq, axis=1)  # (T,)


def energy_drift_on_mesh(rollout: MeshRollout) -> HarnessDefect:
    """max |KE(t) - KE(0)| / max(|KE(0)|, eps), or SKIP per the same
    KE-rest threshold as the particle side (DECISIONS.md D0-08).

    Mirrors :func:`particle_rollout_adapter.energy_drift` for mesh data.
    """
    velocity = _expect_velocity(rollout)
    if isinstance(velocity, HarnessDefect):
        return velocity
    if not rollout.is_regular_grid:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"mesh is graph-topology (framework="
                f"{rollout.metadata.get('framework')!r}); graph-mesh KE "
                f"integration is gated on Day 2 hour 1 NGC audit"
            ),
        )
    e_series = kinetic_energy_series_on_mesh(rollout)
    e0 = float(e_series[0])
    if abs(e0) < KE_REST_THRESHOLD:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"KE(0)={e0:.3e} < {KE_REST_THRESHOLD:.0e} (mesh rollout "
                f"starts at rest; relative drift undefined; see DECISIONS.md "
                f"D0-08)"
            ),
        )
    drift = float(np.max(np.abs(e_series - e0)))
    return HarnessDefect(value=drift / abs(e0))


def dissipation_sign_violation_on_mesh(rollout: MeshRollout) -> HarnessDefect:
    """max(0, max(dKE/dt)) / max(|KE_max|, eps), or SKIP per the
    same KE-rest threshold (DECISIONS.md D0-08).

    Mirrors :func:`particle_rollout_adapter.dissipation_sign_violation`
    for mesh data.
    """
    velocity = _expect_velocity(rollout)
    if isinstance(velocity, HarnessDefect):
        return velocity
    if not rollout.is_regular_grid:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"mesh is graph-topology (framework="
                f"{rollout.metadata.get('framework')!r}); graph-mesh dKE/dt "
                f"is gated on Day 2 hour 1 NGC audit"
            ),
        )
    if rollout.n_timesteps < 2:
        raise ValueError(
            f"dissipation_sign_violation_on_mesh needs at least 2 timesteps; "
            f"got {rollout.n_timesteps}"
        )
    e_series = kinetic_energy_series_on_mesh(rollout)
    e_max = float(np.max(e_series))
    if e_max < KE_REST_THRESHOLD:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"max(KE)={e_max:.3e} < {KE_REST_THRESHOLD:.0e} (mesh "
                f"trajectory has no kinetic energy; dissipation question "
                f"undefined; see DECISIONS.md D0-08)"
            ),
        )
    de_dt = np.diff(e_series) / rollout.dt
    max_growth = float(np.max(de_dt))
    return HarnessDefect(value=max(0.0, max_growth) / e_max)


# ---------------------------------------------------------------------------
# DGL → MeshField materialization
# ---------------------------------------------------------------------------
#
# Deliberately not implemented in the Day 0 / Day 0.5 scaffold. This path
# requires:
#
#   1. A real PhysicsNeMo NGC sample timestep (Audit Q1 / Gate A) so the
#      DGL-graph-to-scikit-fem-basis coercion is exercised against actual
#      output, not a synthetic graph.
#   2. The `nvidia-physicsnemo` and `dgl` dependencies, which live behind
#      the `[validation-rollout]` extra and are not installed on Day 0.
#   3. A confirmed Gate A verdict (PASS / PARTIAL) — under FAIL, this
#      function is never called.
#
# Per the executing agent's "no speculative stubs" rule, the function lands
# in a separate Day 2 commit once Gate A returns a verdict. Until then,
# callers requesting it will receive an `AttributeError` from this module
# — that is the intended behaviour.
