"""LagrangeBench pkl → SCHEMA.md §1 particle_rollout.npz conversion.

Pure-Python (numpy + stdlib only). Shipped into the rung-3.5 Modal
container via ``Image.add_local_file`` so the same module is used by
both the Modal-side post-inference conversion step and any local-side
read of converted rollouts; reused for P1 (GNS-TGV2D) and likely
re-shaped for the PhysicsNeMo equivalent on Day 2 per the conversion-
factoring decision in DECISIONS.md D0-15 amendment 4.

## What LagrangeBench writes vs what SCHEMA.md §1 expects

LagrangeBench's ``eval_rollout`` (``lagrangebench/evaluate/rollout.py``)
writes one pickle per trajectory under ``eval.rollout_dir`` when
``eval.infer.out_type=pkl``::

    {
        "predicted_rollout":    ndarray,  # (T, N_particles, D)  positions
        "ground_truth_rollout": ndarray,  # (T_gt, N_particles, D)
        "particle_type":        ndarray,  # (N_particles,)
    }

Plus a ``metrics{timestamp}.pkl`` with the eval metrics dict, which is
not consumed by this conversion but left in place for cross-validation
against LagrangeBench's own published numbers (rung 4 / plan §10 item 2).

SCHEMA.md §1 expects per-trajectory ``particle_rollout.npz`` with
positions, velocities, particle_type, particle_mass, dt, domain_box,
metadata. This module performs the four reconciliations:

1. **velocities derived** from ``predicted_rollout`` via central
   differences over ``dt = metadata["dt"] * metadata["write_every"]``
   (matching LagrangeBench's own factor at ``runner.py:260`` and
   ``metrics.py:99``); endpoints use first-order forward/backward
   differences. **Periodic-aware**: on datasets with
   ``periodic_boundary_conditions`` set in dataset metadata.json, the
   minimum-image convention is applied along each periodic axis to
   prevent boundary-wraparound from producing spurious O(L/dt)
   velocities (DECISIONS.md D0-17, motivated by the rung-3.5 spot-check
   on f75e22d8dd which surfaced 5+-order-of-magnitude KE inflation on
   SEGNN-TGV2D). Methodology framing in DECISIONS.md D0-15 amendment 4:
   PH-CON-002 tests dynamical consistency of the positional rollout
   under conservation, not direct velocity-output conservation.

2. **particle_mass populated** as uniform unit mass per particle. SPH
   datasets (LagrangeBench's entire 2D / 3D corpus) fold the per-
   particle mass into the smoothing-length normalization and don't
   carry an explicit mass field; SCHEMA.md §1 v1.2 documents the
   uniform-unit-mass default and its methodological equivalence to
   the dataset's implicit normalization for the conservation /
   dissipation rules.

3. **domain_box transposed** from LagrangeBench's ``(D, 2)``
   per-axis-[min,max] convention to SCHEMA.md's ``(2, D)`` first-row-
   mins-second-row-maxes convention.

4. **dt computed** as ``metadata["dt"] * metadata.get("write_every", 1)``
   defensively (write_every is conditionally present in
   LagrangeBench dataset metadata.json; absent in the
   ``tests/3D_LJ_3_1214every1`` tutorial fixture; expected present
   for production datasets like ``2D_TGV_2500_10kevery100``). The
   ``write_every_source`` field on ``RolloutMetadata`` records
   whether the value was read from the dataset or defaulted to 1, so
   future audit-trail reconstruction can distinguish the two.

## Validation surface

Every load-bearing field has a shape / dtype / value-range check at
conversion time. Eight assertions total (five from DECISIONS.md D0-15
amendment 4 + one from D0-17 + two from D0-17 amendment 1):

- ``particle_mass.shape == (N_particles,)``
- ``particle_mass.dtype == np.float64``
- ``domain_box.shape == (2, D)``
- ``domain_box[0] < domain_box[1]`` elementwise (mins below maxes)
- ``RolloutMetadata.write_every_source`` set to ``"dataset"`` or
  ``"default"`` (no other values; populated by this module from the
  conversion's read of dataset metadata.json)
- ``periodic_boundary_conditions`` length equals D after truncation
  (D0-17; rejected if upstream length < D — no silent zero-padding)
- ``periodic_boundary_conditions`` truncated trailing entries all True
  (D0-17 amendment 1; matches LB's stable upstream convention of
  vestigial-axes-always-periodic; trailing False fires hard error)
- ``RolloutMetadata.periodic_boundary_conditions_source`` set to
  ``"dataset"``, ``"truncated_from_oversize"``, or ``"default"``

The assertions raise ``ValueError`` with the rollout filename in the
message, so a failure surfaces both the rule violated and the file
that triggered it. Cheaper to fail at conversion time than to
reconstruct post-hoc when the harness ``load_rollout_npz`` errors on
a malformed npz.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Metadata dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutMetadata:
    """Metadata fields populated into ``particle_rollout.npz`` per SCHEMA.md §1.

    Caller-supplied: ``git_sha``, ``lagrangebench_sha``, ``dataset``, ``model``,
    ``seed``, ``framework``, ``framework_version``. The remaining fields
    (``ckpt_hash``, ``ckpt_path``, ``write_every``, ``write_every_source``)
    are populated by :func:`convert_rollout_dir` from the conversion's
    runtime context — callers leave them at their defaults.

    Frozen for accidental-mutation safety; use :func:`dataclasses.replace`
    to derive an updated instance.

    Adding a field here is also a SCHEMA.md §1 addition (since this class
    is the single source of truth for what gets written into the npz's
    metadata dict). Bump SCHEMA.md's version line and document the new
    field in §1 alongside any addition here.
    """

    git_sha: str
    lagrangebench_sha: str
    dataset: str
    model: str
    seed: int
    framework: str
    framework_version: str
    # Filled in by convert_rollout_dir from runtime context:
    ckpt_hash: str = ""
    ckpt_path: str = ""
    write_every: int = 1
    write_every_source: str = "default"
    # D0-17 (rung-3.5 spot-check finding): periodic_boundary_conditions
    # is read from LB dataset metadata.json and threaded through into the
    # npz so consumers know whether positions are wrap-around-aware.
    # Stored as tuple[bool, ...] for hashability; len matches positions D.
    # Empty tuple is the dataclass-default sentinel only — convert_rollout_dir
    # always populates it with len-D bools (defaulting to all-False per axis
    # if the dataset metadata omits the key).
    periodic_boundary_conditions: tuple[bool, ...] = ()
    # D0-17 amendment 1 (post-D0-17 regen FAIL on real TGV2D metadata):
    # LagrangeBench's stable upstream convention is "PBC field is always
    # length 3 regardless of dim" (verified: 2D TGV and 3D LJ tutorial
    # fixture both have [True, True, True] PBC; explicit ``dim`` field
    # disambiguates intended dimensionality). Conversion truncates PBC
    # to D entries when len > D; audit fields below capture the original
    # upstream vector + the source classification so future readers can
    # verify the truncation was sensible without re-reading LB's
    # metadata.json. Same shape as ``write_every_source`` from D0-15
    # amendment 4.
    periodic_boundary_conditions_upstream: tuple[bool, ...] = ()
    periodic_boundary_conditions_source: str = (
        "default"  # "dataset" | "truncated_from_oversize" | "default"
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict converts tuple to list — keep as list for JSON / npz round-trip.
        d["periodic_boundary_conditions"] = list(self.periodic_boundary_conditions)
        d["periodic_boundary_conditions_upstream"] = list(
            self.periodic_boundary_conditions_upstream
        )
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ROLLOUT_PKL_PATTERN = re.compile(r"^rollout_(\d+)\.pkl$")


def _hash_directory(root: Path) -> str:
    """Deterministic SHA-256 of all files under ``root``.

    Walk in sorted-relpath order; for each file, hash its contents,
    then hash the concatenation of ``f"{relpath}\\t{file_hash}\\n"``
    lines. Result is invariant to walk order and to absolute path —
    only the file tree's relative structure and file contents
    contribute. Symlinks are followed; missing files raise
    ``FileNotFoundError``.
    """
    root = Path(root)
    files: list[str] = []
    for dp, _, fns in root.walk() if hasattr(root, "walk") else _walk_compat(root):
        for fn in fns:
            files.append(str(Path(dp).relative_to(root) / fn))
    files.sort()
    h = hashlib.sha256()
    for rel in files:
        with open(root / rel, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        h.update(f"{rel}\t{file_hash}\n".encode())
    return h.hexdigest()


def _walk_compat(root: Path):
    """``Path.walk`` shim for Python < 3.12 (rung-3 image is on 3.10)."""
    import os

    for dp, dns, fns in os.walk(root):
        yield Path(dp), dns, fns


def _central_diff_velocities(
    positions: np.ndarray,
    dt: float,
    *,
    domain_extent: np.ndarray | None = None,
    periodic_axes: np.ndarray | None = None,
) -> np.ndarray:
    """Velocities from positions via second-order central differences.

    Interior: ``v[t] = (pos[t+1] - pos[t-1]) / (2*dt)``.
    Endpoints: forward at ``t=0`` (``v[0] = (pos[1] - pos[0]) / dt``)
    and backward at ``t=T-1`` (``v[T-1] = (pos[T-1] - pos[T-2]) / dt``).
    Both endpoint forms are first-order accurate.

    Shape is preserved: ``(T, N, D) -> (T, N, D)``.

    The half-timestep offset of forward-only differences would silently
    bias kinetic energy; central differences keep velocity aligned with
    position. See DECISIONS.md D0-15 amendment 4 for the original
    methodology framing on derived-vs-model-output velocities and
    DECISIONS.md D0-17 for the periodic-distance correction added after
    the rung-3.5 spot-check on f75e22d8dd surfaced wraparound-induced
    spurious velocities on TGV2D.

    Periodic-distance correction
    ----------------------------
    On periodic-boundary datasets, a particle crossing a boundary
    (e.g. position 0.999 → 0.001) has a true displacement of +0.002,
    not -0.998. Without correction, the central difference reports a
    spurious O(L/dt) velocity at every wraparound frame — verified on
    SEGNN-TGV2D traj00 to inflate KE(0) by 5+ orders of magnitude.

    When ``domain_extent`` and ``periodic_axes`` are both supplied, the
    correction applies the minimum-image convention along each periodic
    axis::

        delta = pos[t+1] - pos[t-1]
        delta -= L * round(delta / L)          # only for periodic axes
        v[t]   = delta / (2 * dt)

    On non-periodic axes the raw delta is preserved. Calling without
    ``domain_extent`` / ``periodic_axes`` (or with all-False
    ``periodic_axes``) reproduces the pre-D0-17 behavior exactly — this
    is the regression-guard contract enforced by
    ``test_central_diff_no_op_when_non_periodic``.

    Parameters
    ----------
    positions
        ``(T, N, D)`` position array.
    dt
        Rollout timestep.
    domain_extent
        ``(D,)`` per-axis domain extent (``domain_box[1] - domain_box[0]``).
        Required if any axis is periodic; ignored if ``periodic_axes`` is
        ``None`` or all-False.
    periodic_axes
        ``(D,)`` boolean array; ``True`` indicates the axis has periodic
        boundary conditions and the minimum-image correction applies.
    """
    if positions.shape[0] < 2:
        raise ValueError(
            f"central_diff_velocities requires T >= 2 timesteps; got T={positions.shape[0]}"
        )
    apply_periodic = (
        periodic_axes is not None and domain_extent is not None and np.any(periodic_axes)
    )
    if apply_periodic:
        if periodic_axes.shape != (positions.shape[2],):
            raise ValueError(
                f"periodic_axes shape {periodic_axes.shape} must be ({positions.shape[2]},) "
                f"to match positions D"
            )
        if domain_extent.shape != (positions.shape[2],):
            raise ValueError(
                f"domain_extent shape {domain_extent.shape} must be ({positions.shape[2]},) "
                f"to match positions D"
            )

    velocities = np.empty_like(positions, dtype=np.float64)

    def _diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        delta = a - b
        if apply_periodic:
            wrap = domain_extent * np.round(delta / domain_extent)
            delta = np.where(periodic_axes[None, None, :], delta - wrap, delta)
        return delta

    velocities[1:-1] = _diff(positions[2:], positions[:-2]) / (2.0 * dt)
    velocities[0] = _diff(positions[1:2], positions[0:1])[0] / dt
    velocities[-1] = _diff(positions[-1:], positions[-2:-1])[0] / dt
    return velocities


def _read_lagrangebench_metadata(metadata_path: Path) -> dict[str, Any]:
    """Read LagrangeBench dataset ``metadata.json``.

    Returns the dict as-is. Required keys (``dt``, ``bounds``) are
    validated downstream when the conversion accesses them — this
    function is just I/O.
    """
    with open(metadata_path) as f:
        return json.loads(f.read())


# ---------------------------------------------------------------------------
# Conversion entrypoint
# ---------------------------------------------------------------------------


def convert_rollout_dir(
    rollout_dir: Path | str,
    dataset_metadata_path: Path | str,
    ckpt_dir: Path | str,
    *,
    metadata: RolloutMetadata,
    output_dir: Path | str | None = None,
    filename_pattern: str = "particle_rollout_traj{j:02d}.npz",
) -> list[Path]:
    """Convert all ``rollout_*.pkl`` in ``rollout_dir`` to per-trajectory npzs.

    Parameters
    ----------
    rollout_dir
        Directory containing LagrangeBench's ``rollout_{j}.pkl`` files
        (one per trajectory, indexed from 0). Also typically contains
        ``metrics{timestamp}.pkl``, which is left untouched.
    dataset_metadata_path
        Path to the LagrangeBench dataset's ``metadata.json``. Provides
        ``dt``, ``write_every`` (optional), and ``bounds``.
    ckpt_dir
        Path to the checkpoint directory (e.g., ``checkpoints/segnn_tgv2d/best/``).
        Hashed to populate ``RolloutMetadata.ckpt_hash``; stringified
        to populate ``RolloutMetadata.ckpt_path``.
    metadata
        Caller-supplied ``RolloutMetadata`` (the seven non-runtime fields
        populated; the four runtime fields left at defaults). The conversion
        derives ``ckpt_hash``, ``ckpt_path``, ``write_every``,
        ``write_every_source`` from runtime context and merges them in.
    output_dir
        Target directory for the converted npzs. Defaults to ``rollout_dir``
        so the native pkls and converted npzs sit side-by-side per the
        D0-14 / D0-15 amendment 4 layout.
    filename_pattern
        Format string for output filenames. Default zero-pads the
        trajectory index to two digits so directory listings sort
        lexicographically; bump the width if rolling out >100 trajectories.

    Returns
    -------
    list of Path
        Paths to the written ``particle_rollout_traj{j:02d}.npz`` files,
        in trajectory-index order.

    Raises
    ------
    FileNotFoundError
        If ``rollout_dir`` contains no ``rollout_*.pkl`` files, or if
        ``dataset_metadata_path`` / ``ckpt_dir`` doesn't exist.
    ValueError
        If any per-trajectory validation assertion fails (shape, dtype,
        domain-box ordering). Message names the rollout filename so the
        failing trajectory is identifiable without re-running.
    KeyError
        If ``dataset_metadata_path`` lacks required keys (``dt``,
        ``bounds``).
    """
    rollout_dir = Path(rollout_dir)
    dataset_metadata_path = Path(dataset_metadata_path)
    ckpt_dir = Path(ckpt_dir)
    output_dir = Path(output_dir) if output_dir is not None else rollout_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_metadata_path.is_file():
        raise FileNotFoundError(f"dataset metadata.json not found: {dataset_metadata_path}")
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")

    pkl_files: list[tuple[int, Path]] = []
    for entry in sorted(rollout_dir.iterdir()):
        m = _ROLLOUT_PKL_PATTERN.match(entry.name)
        if m is not None:
            pkl_files.append((int(m.group(1)), entry))
    pkl_files.sort(key=lambda x: x[0])
    if not pkl_files:
        raise FileNotFoundError(
            f"no rollout_*.pkl files found in {rollout_dir} (expected at least one)"
        )

    ds_meta = _read_lagrangebench_metadata(dataset_metadata_path)
    if "dt" not in ds_meta:
        raise KeyError(f"{dataset_metadata_path} missing required key 'dt'")
    if "bounds" not in ds_meta:
        raise KeyError(f"{dataset_metadata_path} missing required key 'bounds'")

    write_every_present = "write_every" in ds_meta
    write_every = int(ds_meta["write_every"]) if write_every_present else 1
    write_every_source = "dataset" if write_every_present else "default"
    dt_rollout = float(ds_meta["dt"]) * float(write_every)

    # bounds: LagrangeBench (D, 2) per-axis-[min, max] -> SCHEMA (2, D) [mins; maxes]
    bounds_lb = np.asarray(ds_meta["bounds"], dtype=np.float64)
    if bounds_lb.ndim != 2 or bounds_lb.shape[1] != 2:
        raise ValueError(
            f"{dataset_metadata_path} 'bounds' must have shape (D, 2); got shape {bounds_lb.shape}"
        )
    domain_box = bounds_lb.T  # (2, D)
    d_dim = int(domain_box.shape[1])
    if domain_box.shape != (2, d_dim):
        raise ValueError(f"domain_box post-transpose shape {domain_box.shape} != (2, D={d_dim})")
    if not np.all(domain_box[0] < domain_box[1]):
        raise ValueError(
            f"domain_box mins must be strictly less than maxes elementwise; "
            f"got mins={domain_box[0].tolist()}, maxes={domain_box[1].tolist()}"
        )
    domain_extent = domain_box[1] - domain_box[0]  # (D,) per-axis extent for periodic wrap

    # D0-17 + amendment 1: periodic_boundary_conditions threaded through
    # from LB dataset metadata.json. Defaults to all-False per axis if
    # the dataset metadata omits the key (non-periodic fallback). The
    # periodic-distance correction in _central_diff_velocities is a
    # no-op when periodic_axes is all-False (regression-guarded by
    # test_central_diff_no_op_when_non_periodic).
    #
    # D0-17 amendment 1: LagrangeBench's stable upstream convention is
    # "PBC field is always length 3 regardless of dim" (verified across
    # 2D TGV2D production dataset and 3D LJ tutorial fixture). When
    # len(pbc_raw) > D, truncate to first D entries; trailing entries
    # are vestigial-axes-which-are-always-True per the upstream
    # convention, so a trailing False is a methodologically suspicious
    # signal (either upstream changed convention, or the dataset
    # metadata is corrupted) and the conversion fires a hard error.
    # Length-D-exact and length-< D paths preserved as before
    # (length-< D rejected; no silent zero-padding).
    pbc_key_present = "periodic_boundary_conditions" in ds_meta
    pbc_raw = ds_meta.get("periodic_boundary_conditions", [False] * d_dim)
    pbc_raw_arr = np.asarray(pbc_raw, dtype=bool)
    if pbc_raw_arr.ndim != 1:
        raise ValueError(
            f"{dataset_metadata_path} 'periodic_boundary_conditions' must be a 1-D "
            f"array; got shape {pbc_raw_arr.shape}"
        )
    pbc_upstream_len = int(pbc_raw_arr.shape[0])
    if pbc_upstream_len < d_dim:
        # Genuine bug: length < D cannot be silently zero-padded since the
        # missing entries are unspecified, not implicitly any value.
        raise ValueError(
            f"{dataset_metadata_path} 'periodic_boundary_conditions' length "
            f"{pbc_upstream_len} is less than dataset dimension D={d_dim}; "
            f"cannot silently zero-pad. Got {pbc_raw!r}"
        )
    if pbc_upstream_len > d_dim:
        # D0-17 amendment 1: truncate to D, sanity-check the trailing
        # entries are all True (matches upstream's vestigial-axes
        # convention).
        truncated_tail = pbc_raw_arr[d_dim:]
        if not np.all(truncated_tail):
            raise ValueError(
                f"{dataset_metadata_path} 'periodic_boundary_conditions' length "
                f"{pbc_upstream_len} > D={d_dim} is expected (upstream LB "
                f"convention is length 3 regardless of dim), but the truncated "
                f"trailing entries {truncated_tail.tolist()} are not all True. "
                f"Stable upstream convention is 'vestigial axes are always "
                f"periodic'; a trailing False here means either the convention "
                f"changed or the dataset metadata is corrupted. Got {pbc_raw!r}"
            )
        periodic_axes = pbc_raw_arr[:d_dim]
        pbc_source = "truncated_from_oversize"
    else:  # pbc_upstream_len == d_dim
        periodic_axes = pbc_raw_arr
        pbc_source = "dataset" if pbc_key_present else "default"

    ckpt_hash = _hash_directory(ckpt_dir)
    runtime_metadata = replace(
        metadata,
        ckpt_hash=ckpt_hash,
        ckpt_path=str(ckpt_dir),
        write_every=write_every,
        write_every_source=write_every_source,
        periodic_boundary_conditions=tuple(bool(x) for x in periodic_axes.tolist()),
        periodic_boundary_conditions_upstream=tuple(bool(x) for x in pbc_raw_arr.tolist()),
        periodic_boundary_conditions_source=pbc_source,
    )
    if runtime_metadata.write_every_source not in {"dataset", "default"}:
        raise ValueError(
            f"write_every_source must be 'dataset' or 'default'; got "
            f"{runtime_metadata.write_every_source!r}"
        )
    if len(runtime_metadata.periodic_boundary_conditions) != d_dim:
        raise ValueError(
            f"periodic_boundary_conditions length "
            f"{len(runtime_metadata.periodic_boundary_conditions)} must equal D={d_dim}"
        )
    if runtime_metadata.periodic_boundary_conditions_source not in {
        "dataset",
        "truncated_from_oversize",
        "default",
    }:
        raise ValueError(
            f"periodic_boundary_conditions_source must be 'dataset', "
            f"'truncated_from_oversize', or 'default'; got "
            f"{runtime_metadata.periodic_boundary_conditions_source!r}"
        )

    written: list[Path] = []
    for j, pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            blob = pickle.load(f)
        if "predicted_rollout" not in blob or "particle_type" not in blob:
            raise KeyError(
                f"{pkl_path} missing required pkl keys (expected at least "
                f"'predicted_rollout' and 'particle_type'); got {sorted(blob.keys())}"
            )
        positions = np.asarray(blob["predicted_rollout"], dtype=np.float64)
        particle_type = np.asarray(blob["particle_type"])
        if positions.ndim != 3:
            raise ValueError(
                f"{pkl_path} predicted_rollout must have shape (T, N, D); "
                f"got shape {positions.shape}"
            )
        n_particles = int(positions.shape[1])
        d_pos = int(positions.shape[2])
        if d_pos != d_dim:
            raise ValueError(
                f"{pkl_path} predicted_rollout D={d_pos} does not match "
                f"dataset metadata bounds D={d_dim}"
            )
        if particle_type.shape != (n_particles,):
            raise ValueError(
                f"{pkl_path} particle_type shape {particle_type.shape} "
                f"must be ({n_particles},) to match predicted_rollout"
            )
        velocities = _central_diff_velocities(
            positions,
            dt_rollout,
            domain_extent=domain_extent,
            periodic_axes=periodic_axes,
        )
        particle_mass = np.ones(n_particles, dtype=np.float64)
        if particle_mass.shape != (n_particles,):
            raise ValueError(
                f"{pkl_path} particle_mass shape {particle_mass.shape} must be ({n_particles},)"
            )
        if particle_mass.dtype != np.float64:
            raise ValueError(
                f"{pkl_path} particle_mass dtype {particle_mass.dtype} must be float64"
            )

        out_path = output_dir / filename_pattern.format(j=j)
        np.savez(
            out_path,
            positions=positions.astype(np.float32),
            velocities=velocities.astype(np.float32),
            particle_type=particle_type.astype(np.int32),
            particle_mass=particle_mass,
            dt=np.float64(dt_rollout),
            domain_box=domain_box,
            metadata=np.array(runtime_metadata.to_dict(), dtype=object),
        )
        written.append(out_path)
    return written
