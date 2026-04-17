"""Hybrid adapter+dump loader with extension-based dispatch.

Design doc §5. Entry point: load_target(path, cli_overrides, toml_path)
returns a LoadedTarget(spec, field, model) where:

- spec: validated DomainSpec
- field: a Field instance materialized from the target (GridField for dumps,
  CallableField for adapter with model, or None if the rule will create it later)
- model: the adapter's load_model() return (None in dump mode)

The loader is the single place where user-supplied Python (`exec`) or user-
supplied data files (`np.load`) enter physics-lint. All subsequent code
assumes inputs have been validated.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from physics_lint import DomainSpec
from physics_lint.config import load_spec_from_toml, merge_into_spec
from physics_lint.field import CallableField, Field, GridField

if TYPE_CHECKING:
    import torch


class LoaderError(RuntimeError):
    """Raised when physics-lint cannot load the user's target."""


@dataclass
class LoadedTarget:
    spec: DomainSpec
    field: Field
    model: Callable[..., Any] | None
    # Optional per-target auxiliary data. `boundary_target` carries the
    # true Dirichlet BC values for the problem when the caller has them
    # (e.g. a dogfood dump that extracted predictions alongside the
    # ground-truth BC). CLI auto-extraction uses this for PH-BC-001 and
    # PH-POS-002 without requiring a full V1.1 per-rule kwarg injection
    # plumb. None for adapter-mode and for dumps that don't include it.
    boundary_target: Any | None = None


def load_target(
    path: Path,
    *,
    cli_overrides: dict[str, Any],
    toml_path: Path | None,
) -> LoadedTarget:
    """Load a target file, merge config, and return a LoadedTarget.

    After dispatch, _resolve_source_term runs for both adapter and dump
    paths so a Poisson config can carry a ``source_term="source.npy"``
    pointer and have PH-RES-001 see the source array without the user
    having to use the .npz-with-embedded-source form.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    toml_spec: dict[str, Any] = {}
    if toml_path is not None:
        toml_spec = load_spec_from_toml(toml_path)

    if suffix == ".py":
        target = _load_adapter(path, toml_spec=toml_spec, cli_overrides=cli_overrides)
    elif suffix in (".npz", ".npy"):
        target = _load_dump(path, toml_spec=toml_spec, cli_overrides=cli_overrides)
    elif suffix in (".pt", ".pth"):
        raise LoaderError(
            f"{path.name}: .pt/.pth files are not supported directly. "
            "Please use an adapter or convert to .npz; see docs/loading.html"
        )
    else:
        raise LoaderError(f"{path.name}: unsupported file extension {suffix}")

    _resolve_source_term(target, base_path=path)
    return target


def _load_adapter(
    path: Path,
    *,
    toml_spec: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> LoadedTarget:
    """Load a user adapter module: exec + call load_model() and domain_spec()."""
    if not path.is_file():
        raise LoaderError(f"adapter file not found: {path}")

    module_name = f"_physics_lint_adapter_{path.stem}_{id(path)}"
    spec_obj = importlib.util.spec_from_file_location(module_name, str(path))
    if spec_obj is None or spec_obj.loader is None:
        raise LoaderError(f"cannot import adapter from {path}")
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    try:
        spec_obj.loader.exec_module(module)
    except Exception as e:
        raise LoaderError(f"adapter {path} raised during import: {e}") from e

    if not hasattr(module, "load_model"):
        raise LoaderError(f"adapter {path} missing required load_model() function")
    if not hasattr(module, "domain_spec"):
        raise LoaderError(f"adapter {path} missing required domain_spec() function")

    try:
        model = module.load_model()
    except Exception as e:
        raise LoaderError(f"adapter {path}.load_model() raised: {e}") from e
    try:
        adapter_spec_obj = module.domain_spec()
    except Exception as e:
        raise LoaderError(f"adapter {path}.domain_spec() raised: {e}") from e

    adapter_spec_dict: dict[str, Any]
    if isinstance(adapter_spec_obj, DomainSpec):
        adapter_spec_dict = adapter_spec_obj.model_dump()
    elif isinstance(adapter_spec_obj, dict):
        adapter_spec_dict = adapter_spec_obj
    else:
        raise LoaderError(
            f"adapter {path}.domain_spec() must return DomainSpec or dict; "
            f"got {type(adapter_spec_obj).__name__}"
        )
    # Make sure the adapter source path is recorded
    adapter_spec_dict.setdefault("field", {})
    adapter_spec_dict["field"]["adapter_path"] = str(path)
    adapter_spec_dict["field"].pop("dump_path", None)

    merged = merge_into_spec(toml_spec, adapter_spec=adapter_spec_dict, cli_overrides=cli_overrides)
    spec = DomainSpec.model_validate(merged)

    if spec.field.type == "mesh":
        raise LoaderError(
            "field.type = 'mesh' is not supported by the loader in V1. "
            "MeshField requires a scikit-fem Basis + DOF vector that the "
            "loader cannot construct from an adapter module or dump file. "
            "Construct MeshField directly via "
            "physics_lint.field.MeshField(basis=..., dofs=...) and pass it "
            "to rule check() functions. See docs/backlog/v1.1.md for the "
            "planned loader integration."
        )

    # For Week 1 we materialize the callable onto a GridField via CallableField.
    # Build a sampling grid from the spec. `_build_sampling_grid` imports
    # torch inline so dump-only use never pays the module-level import cost.
    grid_tensor = _build_sampling_grid(spec)
    field = CallableField(
        model=model,
        sampling_grid=grid_tensor,
        h=_compute_h_from_spec(spec),
        periodic=spec.periodic,
    )
    return LoadedTarget(spec=spec, field=field, model=model)


def _load_dump(
    path: Path,
    *,
    toml_spec: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> LoadedTarget:
    """Load a .npz/.npy dump: read prediction + metadata and wrap in GridField.

    .npz carries a ``prediction`` array and an optional ``metadata`` 0-d object
    array that holds the adapter spec dict. .npy is a bare ndarray with no
    embedded metadata, so its spec must come from the TOML config or CLI
    overrides; if both are empty, we raise LoaderError rather than letting
    pydantic fail with an obscure "missing pde" error.
    """
    if not path.is_file():
        raise LoaderError(f"dump file not found: {path}")

    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray):
        # .npy: bare prediction array, no embedded metadata. Spec must come
        # from toml_spec / cli_overrides; if neither supplies the required
        # fields, surface a clear error pointing at the .npz alternative.
        prediction = loaded
        adapter_spec_dict: dict[str, Any] = {}
        if not toml_spec and not cli_overrides:
            raise LoaderError(
                f"{path}: .npy dumps carry no metadata; supply a "
                "[tool.physics-lint] TOML config (or CLI overrides) with "
                "pde/grid_shape/domain/boundary_condition, or convert to "
                ".npz with a metadata dict."
            )
    else:
        if "prediction" not in loaded.files:
            raise LoaderError(f"{path}: .npz must contain a 'prediction' array")
        prediction = loaded["prediction"]
        metadata_raw = loaded.get("metadata") if "metadata" in loaded.files else None
        if metadata_raw is None:
            adapter_spec_dict = {}
        else:
            # np.savez wraps dicts in 0-dim object arrays
            adapter_spec_dict = (
                metadata_raw.item() if metadata_raw.shape == () else dict(metadata_raw)
            )
        if not isinstance(adapter_spec_dict, dict):
            raise LoaderError(f"{path}: metadata must be a dict")

    adapter_spec_dict.setdefault("field", {})
    adapter_spec_dict["field"]["dump_path"] = str(path)
    adapter_spec_dict["field"].pop("adapter_path", None)

    merged = merge_into_spec(toml_spec, adapter_spec=adapter_spec_dict, cli_overrides=cli_overrides)
    spec = DomainSpec.model_validate(merged)

    if spec.field.type == "mesh":
        raise LoaderError(
            "field.type = 'mesh' is not supported by the loader in V1. "
            "MeshField requires a scikit-fem Basis + DOF vector that the "
            "loader cannot construct from an adapter module or dump file. "
            "Construct MeshField directly via "
            "physics_lint.field.MeshField(basis=..., dofs=...) and pass it "
            "to rule check() functions. See docs/backlog/v1.1.md for the "
            "planned loader integration."
        )

    # Plumb optional runtime-injected attributes: Poisson source term and
    # heat/wave initial condition. Rules read these via
    # getattr(spec, '_source_array', None) etc. Using object.__setattr__
    # bypasses pydantic's frozen-model guard while keeping the validated
    # fields of spec immutable.
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "source" in loaded.files:
            object.__setattr__(spec, "_source_array", np.asarray(loaded["source"]))
        if "initial_condition" in loaded.files:
            object.__setattr__(spec, "_initial_condition", np.asarray(loaded["initial_condition"]))

    # Catch mismatched dumps *before* computing h: a wrong grid_shape picks
    # the wrong spacing and the wrong calibrated floor, producing rule
    # results that look numerical but mean nothing. The spatial + time axes
    # must all line up, so compare the full shape tuple.
    if tuple(prediction.shape) != tuple(spec.grid_shape):
        raise LoaderError(
            f"{path}: prediction shape {tuple(prediction.shape)} does not match "
            f"spec grid_shape {tuple(spec.grid_shape)}; fix the dump or the "
            "[tool.physics-lint] config so they agree."
        )

    h = _compute_h_from_spec(spec)
    field = GridField(
        prediction,
        h=h,
        periodic=spec.periodic,
        backend=spec.field.backend
        if spec.field.backend != "auto"
        else ("spectral" if spec.periodic else "fd"),
    )
    # Optional boundary_target: when a dump ships the true Dirichlet BC
    # alongside the prediction (dogfood dumps from laplace-uq-bench do
    # this via make_ci_dumps.py), carry it through so PH-BC-001 and
    # PH-POS-002 can run from the CLI against the real BC rather than
    # skipping.
    boundary_target = None
    if isinstance(loaded, np.lib.npyio.NpzFile) and "boundary_target" in loaded.files:
        boundary_target = np.asarray(loaded["boundary_target"])
    return LoadedTarget(spec=spec, field=field, model=None, boundary_target=boundary_target)


def _compute_h_from_spec(spec: DomainSpec) -> tuple[float, ...]:
    """Derive uniform grid spacings for spatial + (optional) time axes.

    Convention: spatial axes first, then — if spec.domain.is_time_dependent
    — a time axis as the LAST entry. This matches the Week-2 time-last
    convention throughout the library (Bochner norm, heat/wave rule
    branches, dump prediction tensors). Non-periodic uses endpoint-inclusive
    spacing L/(N-1); periodic uses endpoint-exclusive L/N. Time is ALWAYS
    endpoint-inclusive — spec.periodic refers only to spatial periodicity.
    """
    lengths = spec.domain.spatial_lengths
    shape = spec.grid_shape
    ndim_spatial = len(lengths)
    h_spatial: tuple[float, ...]
    if spec.periodic:
        h_spatial = tuple(
            length / n for length, n in zip(lengths, shape[:ndim_spatial], strict=False)
        )
    else:
        h_spatial = tuple(
            length / (n - 1) for length, n in zip(lengths, shape[:ndim_spatial], strict=False)
        )
    if not spec.domain.is_time_dependent:
        return h_spatial
    if len(shape) < ndim_spatial + 1:
        raise LoaderError(
            f"time-dependent spec requires grid_shape with a time entry after "
            f"the {ndim_spatial} spatial entries; got grid_shape={tuple(shape)}"
        )
    t_lo, t_hi = spec.domain.t  # type: ignore[misc]
    n_t = shape[ndim_spatial]
    h_t = (t_hi - t_lo) / (n_t - 1)
    return (*h_spatial, h_t)


def _build_sampling_grid(spec: DomainSpec) -> torch.Tensor:
    import torch

    bounds = spec.domain.spatial_bounds
    shape = spec.grid_shape
    ndim_spatial = len(bounds)
    axes: list[torch.Tensor] = []
    for (lo, hi), n in zip(bounds, shape[:ndim_spatial], strict=False):
        if spec.periodic:
            axes.append(torch.linspace(lo, hi, n + 1)[:-1])
        else:
            axes.append(torch.linspace(lo, hi, n))
    if spec.domain.is_time_dependent:
        t_lo, t_hi = spec.domain.t  # type: ignore[misc]
        n_t = shape[ndim_spatial]
        axes.append(torch.linspace(t_lo, t_hi, n_t))
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def _resolve_source_term(target: LoadedTarget, *, base_path: Path) -> None:
    """Load spec.source_term (if set) from disk and inject into the spec.

    Poisson residuals need a source array, and the user-facing API offers
    two ways to provide one:

    1. .npz dump with an embedded ``source`` key — handled inline in
       _load_dump by stashing it under spec._source_array.
    2. A ``source_term`` string on the validated DomainSpec pointing at a
       separate .npy/.npz file — handled here so adapter mode and dump
       mode share one code path. Relative paths resolve against the
       adapter/dump file's directory so user projects can ship a
       ``model_adapter.py`` + ``source.npy`` pair without hardcoding
       absolute paths.

    If both forms are used and disagree, source_term wins (it's the more
    explicit of the two). If neither is used, PH-RES-001's Poisson branch
    emits SKIPPED with a clear reason.
    """
    source_term = target.spec.source_term
    if source_term is None:
        return

    source_path = Path(source_term)
    if not source_path.is_absolute():
        source_path = base_path.parent / source_path
    if not source_path.is_file():
        raise LoaderError(
            f"spec.source_term='{source_term}' resolved to {source_path}: "
            "file not found. Use an absolute path, or place the source "
            "file next to the adapter/dump."
        )

    loaded = np.load(source_path, allow_pickle=True)
    source: np.ndarray
    if isinstance(loaded, np.ndarray):
        source = loaded
    elif "source" in loaded.files:
        source = np.asarray(loaded["source"])
    else:
        raise LoaderError(
            f"{source_path}: expected a bare .npy array or an .npz with a "
            "'source' key; got .npz with keys "
            f"{sorted(loaded.files)!r}."
        )
    object.__setattr__(target.spec, "_source_array", source)
