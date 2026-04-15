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


# Week 1 scope is Laplace/Poisson only
# (docs/plans/2026-04-14-physics-lint-v1-week-1.md line 13:
# "no time-dependent PDEs"). The DomainSpec pydantic model accepts heat/wave
# — Week 2 will wire them through GridField's time axis and the Bochner
# norms — but the Week 1 loader has no code path for a (Nx, Ny, Nt) tensor.
# Gate here with a clear "Week 2" error instead of crashing in GridField's
# h-tuple length check.
_WEEK_2_PDES: frozenset[str] = frozenset({"heat", "wave"})


def _assert_week_1_scope(spec: DomainSpec, path: Path) -> None:
    if spec.pde in _WEEK_2_PDES:
        raise LoaderError(
            f"{path}: PDE '{spec.pde}' is time-dependent and lands in Week 2; "
            "Week 1 supports laplace/poisson on spatial-only grids. "
            "Remove the time domain and use pde='laplace' or 'poisson' to run today."
        )


@dataclass
class LoadedTarget:
    spec: DomainSpec
    field: Field
    model: Callable[..., Any] | None


def load_target(
    path: Path,
    *,
    cli_overrides: dict[str, Any],
    toml_path: Path | None,
) -> LoadedTarget:
    """Load a target file, merge config, and return a LoadedTarget."""
    path = Path(path)
    suffix = path.suffix.lower()

    toml_spec: dict[str, Any] = {}
    if toml_path is not None:
        toml_spec = load_spec_from_toml(toml_path)

    if suffix == ".py":
        return _load_adapter(path, toml_spec=toml_spec, cli_overrides=cli_overrides)
    if suffix in (".npz", ".npy"):
        return _load_dump(path, toml_spec=toml_spec, cli_overrides=cli_overrides)
    if suffix in (".pt", ".pth"):
        raise LoaderError(
            f"{path.name}: .pt/.pth files are not supported directly. "
            "Please use an adapter or convert to .npz; see docs/loading.html"
        )
    raise LoaderError(f"{path.name}: unsupported file extension {suffix}")


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
    _assert_week_1_scope(spec, path)

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
    _assert_week_1_scope(spec, path)

    # Catch mismatched dumps *before* computing h: a wrong grid_shape picks
    # the wrong spacing and the wrong calibrated floor, producing rule
    # results that look numerical but mean nothing. Week 1 is spatial-only
    # (heat/wave already gated above), so prediction.shape must match
    # spec.grid_shape exactly.
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
    return LoadedTarget(spec=spec, field=field, model=None)


def _compute_h_from_spec(spec: DomainSpec) -> tuple[float, ...]:
    """Derive uniform grid spacings from domain extents and grid_shape."""
    lengths = spec.domain.spatial_lengths
    # endpoint-inclusive for non-periodic, endpoint-exclusive for periodic
    shape = spec.grid_shape
    if spec.periodic:
        return tuple(length / n for length, n in zip(lengths, shape[: len(lengths)], strict=False))
    return tuple(
        length / (n - 1) for length, n in zip(lengths, shape[: len(lengths)], strict=False)
    )


def _build_sampling_grid(spec: DomainSpec) -> torch.Tensor:
    import torch

    lengths = spec.domain.spatial_lengths
    shape = spec.grid_shape[: len(lengths)]
    axes = []
    for length, n in zip(lengths, shape, strict=False):
        if spec.periodic:
            axes.append(torch.linspace(0.0, length, n + 1)[:-1])
        else:
            axes.append(torch.linspace(0.0, length, n))
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
