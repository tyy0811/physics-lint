"""Config loading and merge path: TOML + adapter + CLI -> dict (validated by DomainSpec).

Design doc §12.4. Load order:

    1. Shipped defaults (live in DomainSpec field defaults)
    2. pyproject.toml [tool.physics-lint]  (or standalone physics-lint.toml fallback)
    3. Adapter domain_spec() return value
    4. CLI flag overrides

The merge returns a plain dict; DomainSpec.model_validate() is called at the
end of load_spec() in loader.py as the single validation point.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def load_spec_from_toml(path: Path) -> dict[str, Any]:
    """Read [tool.physics-lint] from pyproject.toml OR the top table of physics-lint.toml.

    Raises FileNotFoundError if the file doesn't exist.
    Returns the raw dict with nested `boundary_condition` wrapped as a dict
    so it matches BCSpec's shape (users write `boundary_condition = "periodic"`;
    we expand to `{"kind": "periodic"}`).
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    if "tool" in data and "physics-lint" in data["tool"]:
        raw = data["tool"]["physics-lint"]
    else:
        raw = data  # physics-lint.toml fallback: whole file is the spec

    return _normalize_config_shape(raw)


def _normalize_config_shape(raw: dict) -> dict:
    """Expand user-friendly config shapes into the shapes DomainSpec expects."""
    raw = dict(raw)  # shallow copy

    bc = raw.get("boundary_condition")
    if isinstance(bc, str):
        raw["boundary_condition"] = {"kind": bc}

    sym = raw.get("symmetries")
    if isinstance(sym, list):
        raw["symmetries"] = {"declared": sym}

    return raw


def merge_into_spec(
    toml_spec: dict,
    *,
    adapter_spec: dict | None,
    cli_overrides: dict,
) -> dict:
    """Merge four sources into a single dict ready for DomainSpec.model_validate().

    Precedence (later overrides earlier):
        1. toml_spec (from load_spec_from_toml)
        2. adapter_spec (from adapter.domain_spec().model_dump())
        3. cli_overrides (from CLI flags)

    Top-level keys override wholesale; nested dicts are merged one level deep
    so that `[tool.physics-lint.field]` can be partially overridden by an
    adapter's `field` return without clobbering the other keys.
    """
    merged: dict[str, Any] = dict(toml_spec)

    if adapter_spec is not None:
        merged = _deep_merge_one_level(merged, _normalize_config_shape(adapter_spec))

    if cli_overrides:
        merged = _deep_merge_one_level(merged, _normalize_config_shape(cli_overrides))

    return merged


def _deep_merge_one_level(base: dict, override: dict) -> dict:
    """Merge `override` into `base`. For dict-valued keys, merge one level deep."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out
