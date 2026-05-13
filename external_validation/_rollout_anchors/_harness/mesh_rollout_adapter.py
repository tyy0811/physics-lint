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

# Pre-registered FD-noise upper bound on `mass_conservation_defect_on_mesh`
# for divergence-free input fields. See `physics-lint-validation/DECISIONS.md`
# D0-09 for rationale: numpy `np.gradient`'s second-order edge stencil
# produces ~1e-15 noise on a constant input, so 1e-10 is a five-orders-of-
# magnitude headroom bound that absorbs implementation-level FD variation
# without masking real divergence violations (the deliberate-violation
# fixture at alpha=0.1 produces 0.01-0.5).
#
# This constant is consumed by `test_uniform_channel_mass_conservation_zero`
# in `tests/test_mesh_read_only_path.py` and guarded against silent drift
# by `test_mesh_fd_noise_tolerance_matches_pre_registration`. If a future
# numpy release changes `np.gradient`'s edge-stencil behaviour and the
# noise floor moves, log a DECISIONS.md D0-12+ entry citing the
# discrepancy and amend; do not silently shift in code.
MESH_FD_NOISE_TOLERANCE: float = 1e-10


# Mesh-side substrate-class taxonomy. Parallel to
# `particle_rollout_adapter.py::LAGRANGEBENCH_DATASET_SYSTEM_CLASS`.
# Per case study 02 design §2.2 P0-resolvable Pattern-B response
# (duplicated route, NOT a stack-agnostic refactor): the duplicate-logic-
# drift risk is *named* per round-codex-4 catalogue, not eliminated.
# A stack-agnostic refactor triggers only on amendment 1 / case study 03
# evidence (a second mesh-side substrate that disagrees with its
# particle-side analog on substrate class).
#
# Empirical classification per the "classify when you exercise" rule:
# entries land only after Phase 1's empirical probe confirms the
# substrate's behavior. The vortex_shedding_2d entry below is anchored to
# D0-23 verdict 6 (substrate-class smoke on cylinder_flow GT + 399-step
# MGN rollout: ∫|∇·v|/∫‖∇v‖_F ≈ 5%, KE oscillates around steady mean
# with 42 sign-changes, Strouhal St ≈ 0.16 in design band [0.16, 0.21]
# — boundary-driven sub-class). Findings:
# 02-physicsnemo-mgn/preflight/substrate_class_smoke.json.
MGN_DATASET_SYSTEM_CLASS: dict[str, str] = {
    "vortex_shedding_2d": "open-driven-dissipative",  # D0-23 verdict 6
}


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


def _assert_loader_contract_mgn(rollout: MeshRollout) -> None:
    """MGN materializer loader-contract assertions per D0-23 verdict 10.

    Each assertion is grounded in a preflight V-entry or known-unknown
    (see ``02-physicsnemo-mgn/preflight/mgn_loader_contract.md`` §3.1
    and §5). Fires defensively on incoming MGN rollouts BEFORE the rule
    kernels consume them; informative AssertionError if any contract is
    violated.

    Per case study 02 design §2.1 (source-method-implementing-pattern-A-
    discipline): written from source review, catches Pattern-A divergence
    at runtime before P0 inference data flows into the rules.

    Scope: P0 vortex_shedding_2d only — assertions hold for the NGC
    cylinder_flow record schema (preflight loader_contract_audit.json
    V3 / V4 / V5). Amendment 1 (Ahmed Body, D0-23 forward-flag) is the
    multi-instance trigger that would force generalization (e.g., a
    dataset-keyed assertion dispatch table).

    Phase-1 cross-review absorption (Findings 1 + 2): the helper is
    MGN-contract-fail-loud, not lax. Absence of the "velocity" key or
    the "dataset" / "framework" / "model" metadata keys raises
    AssertionError rather than no-op'ing, closing the layered-fail-open
    path where _assert_loader_contract_mgn passes silently while the
    downstream rule SKIPs on a contract violation.
    """
    # V8 / V12: the "velocity" key must be present. Absorbed in-rung
    # per Phase-1 cross-review Finding 2: previously this helper
    # no-op'd on velocity-absent (delegating SKIP wording to
    # _expect_velocity), creating a layered-fail-open path —
    # _assert_loader_contract_mgn passes, then _expect_velocity SKIPs,
    # then the rule reports a legitimate-looking skip while the
    # underlying loader-contract violation is invisible. The helper is
    # MGN-scoped (caller assertion implied by the function name and
    # docstring); absence of "velocity" is therefore a contract
    # failure, not an optional rule SKIP. D0-23 verdict 8 pins the key
    # to literal "velocity".
    velocity = rollout.node_values.get("velocity")
    assert velocity is not None, (
        f"MGN rollout must include node_values['velocity'] per preflight V8 "
        f"+ D0-23 verdict 8 (NGC cylinder_flow record schema; "
        f"vortex_shedding_dataset.py:86-124 @ 1ca85d65). Got keys: "
        f"{sorted(rollout.node_values.keys())}. See "
        f"docs/2026-05-13-case-study-02-phase-1-cross-review.md Finding 2."
    )

    velocity_arr = np.asarray(velocity)

    # Known-unknown §5.6: fp32 vs fp64 precision contract. The NGC
    # checkpoint was trained at default-torch-dtype=float32; if the
    # materializer image runs at float64, `torch.tensor(..., dtype=
    # torch.float)` (vortex_shedding_dataset.py:373) promotes silently
    # and changes numerical behavior. Materializer must
    # torch.set_default_dtype(torch.float32) before dataset construction.
    assert velocity_arr.dtype == np.float32, (
        f"MGN velocity dtype must be float32 per preflight known-unknown §5.6 "
        f"(materializer must torch.set_default_dtype(torch.float32) before "
        f"dataset construction; vortex_shedding_dataset.py:373 @ 1ca85d65). "
        f"Got: {velocity_arr.dtype}. See DECISIONS.md D0-23 verdict 10."
    )

    # V12 / V18: time-axis-first velocity shape (T, N_nodes, D).
    assert velocity_arr.ndim == 3, (
        f"MGN velocity must be 3D (T, N_nodes, D); got shape "
        f"{velocity_arr.shape}. See preflight V12 + V18."
    )
    # V17 + V18: D ∈ {2, 3} (2D vortex shedding uses D=2; 3D Ahmed body
    # would use D=3). A degenerate D=1 lift would slip past _expect_velocity
    # (which lifts 2D arrays to (T,N,1)), so explicit-check here.
    assert velocity_arr.shape[2] in (2, 3), (
        f"MGN velocity last-dim must be 2 (2D) or 3 (3D); got "
        f"{velocity_arr.shape[2]}. See preflight V17 + V18."
    )

    # Known-unknown §5.7 / V16: node_type values must lie in
    # {0, 3, 4, 5, 6}. The loader's one-hot encoder
    # (vortex_shedding_dataset.py:363-368 @ 1ca85d65) maps 0→0 and
    # non-zero→value-3, then F.one_hot(num_classes=4) — any out-of-range
    # value triggers RuntimeError in production rather than a
    # diagnostic message at validation time. Pre-flight makes the
    # implicit contract explicit.
    node_type = np.asarray(rollout.node_type)
    valid_node_types = {0, 3, 4, 5, 6}
    actual_types = set(int(v) for v in np.unique(node_type).tolist())
    invalid = actual_types - valid_node_types
    assert not invalid, (
        f"MGN node_type values must be in {sorted(valid_node_types)} per preflight "
        f"known-unknown §5.7 / V16 (one_hot num_classes=4 bound after value-3 shift; "
        f"vortex_shedding_dataset.py:363-368 @ 1ca85d65). Invalid values: "
        f"{sorted(invalid)}. See DECISIONS.md D0-23 verdict 10."
    )

    # V-entries on metadata schema: framework + model + dataset must
    # all be present. Absorbed in-rung per Phase-1 cross-review
    # Finding 1: `dataset` was missing from the required set even
    # though MGN_DATASET_SYSTEM_CLASS (D0-23 v9) keys the substrate-
    # class dispatch off it. Without the dataset key, the dispatch
    # silently no-ops and the rule emits a misleading raw value on
    # an open-driven-dissipative substrate. Making dataset required
    # closes that fail-open path at the contract boundary.
    for required_meta_key in ("framework", "model", "dataset"):
        assert required_meta_key in rollout.metadata, (
            f"MGN rollout metadata must include {required_meta_key!r}; "
            f"got keys: {sorted(rollout.metadata.keys())}. See preflight "
            f"V-entries on rollout schema + DECISIONS.md D0-23 verdict 10. "
            f"`dataset` is load-bearing for the v9 substrate-class dispatch."
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
    """Returns the velocity array if present, or a HarnessDefect describing the skip if absent.

    Callers must isinstance-check the return value before using it as
    an array — see the union-return contract documented in
    `physics-lint-validation/DECISIONS.md` D0-04 / Day-0.5 review hand-back.
    Convention: ``if isinstance(result, HarnessDefect): return result``
    at the call site, then proceed with the ndarray.

    Returns the velocity array shaped ``(T, N_nodes, D)`` (D inferred
    from the field; scalar (T, N_nodes) velocity is lifted to (T, N_nodes, 1)).

    NGC key pinned per D0-23 verdict 8: the NGC cylinder_flow dataset
    (vortex_shedding_2d, modulus_ns_meshgraphnet v0.1 checkpoint) emits
    node-resolved velocity under the literal key ``"velocity"`` (preflight
    loader_contract_audit.json V3_field_names). Pattern-B P0 single-
    instance enumeration: legacy LB / synthetic and NGC vortex-shedding
    both use the same key, so no helper key list / metadata pivot lands
    here. Amendment 1's Ahmed Body (a second NGC dataset) is the multi-
    instance trigger that would force a generalization refactor.
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

    Substrate-class dispatch added at D0-23 verdict 9 (case study 02
    Phase 1): mirrors D0-22 amendment 1's particle-side gate. Open-
    driven-dissipative substrates SKIP with reason — the strictly-
    dissipative-or-conservative assumption underpinning energy_drift
    does not apply when boundary-driven inflow continuously supplies KE.
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

    # D0-23 verdict 9 substrate-class dispatch (parallel to D0-22
    # amendment 1 on particle side). Fires BEFORE the KE-rest gate
    # because the substrate class is the load-bearing assumption that
    # energy_drift's contract depends on; if the assumption is violated,
    # KE-rest gating is moot.
    dataset_name = rollout.metadata.get("dataset", "") if rollout.metadata else ""
    system_class = MGN_DATASET_SYSTEM_CLASS.get(dataset_name)
    if system_class == "open-driven-dissipative":
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"system_class='open-driven-dissipative' (dataset={dataset_name!r}); "
                "boundary-driven inflow continuously supplies KE; the strictly-"
                "dissipative-or-conservative assumption underpinning "
                "energy_drift does not apply. See DECISIONS.md D0-22 "
                "(amendment 1) for the particle-side precedent and D0-23 "
                "(verdict 9) for the mesh-side extension."
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

    Substrate-class dispatch added at D0-23 verdict 9 (case study 02
    Phase 1): mirrors D0-22 base gate's particle-side dispatch. Open-
    driven-dissipative substrates SKIP with reason — dE/dt > 0 over a
    stretch by physics (boundary-driven inflow supplies KE); the
    strictly-dissipative-or-conservative assumption that
    dissipation_sign_violation encodes does not apply.
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

    # D0-23 verdict 9 substrate-class dispatch (parallel to D0-22 base
    # gate on particle side). Fires BEFORE the timestep / KE-rest gates
    # because the substrate class is the load-bearing assumption.
    dataset_name = rollout.metadata.get("dataset", "") if rollout.metadata else ""
    system_class = MGN_DATASET_SYSTEM_CLASS.get(dataset_name)
    if system_class == "open-driven-dissipative":
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"system_class='open-driven-dissipative' (dataset={dataset_name!r}); "
                "dE/dt > 0 over a stretch by physics (boundary-driven inflow "
                "supplies KE); the strictly-dissipative-or-conservative "
                "assumption underpinning dissipation_sign_violation does not "
                "apply. See DECISIONS.md D0-22 for the particle-side precedent "
                "and D0-23 (verdict 9) for the mesh-side extension."
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
