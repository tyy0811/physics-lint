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
    """Load a .npz dump: read prediction + metadata and wrap in GridField."""
    if not path.is_file():
        raise LoaderError(f"dump file not found: {path}")

    loaded = np.load(path, allow_pickle=True)
    if "prediction" not in loaded.files:
        raise LoaderError(f"{path}: .npz must contain a 'prediction' array")
    prediction = loaded["prediction"]
    metadata_raw = loaded.get("metadata") if "metadata" in loaded.files else None
    if metadata_raw is None:
        adapter_spec_dict = {}
    else:
        # np.savez wraps dicts in 0-dim object arrays
        adapter_spec_dict = metadata_raw.item() if metadata_raw.shape == () else dict(metadata_raw)
    if not isinstance(adapter_spec_dict, dict):
        raise LoaderError(f"{path}: metadata must be a dict")

    adapter_spec_dict.setdefault("field", {})
    adapter_spec_dict["field"]["dump_path"] = str(path)
    adapter_spec_dict["field"].pop("adapter_path", None)

    merged = merge_into_spec(toml_spec, adapter_spec=adapter_spec_dict, cli_overrides=cli_overrides)
    spec = DomainSpec.model_validate(merged)

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
