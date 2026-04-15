"""Shared rule helpers: tristate thresholds, floor loading, safe ratios,
and CallableField -> GridField materialization for adapter-mode input.

Floors are loaded from physics_lint/data/floors.toml; until Task 14
populates that file, a conservative shipped default is returned so
rules can still compute a violation_ratio. The default is intentionally
pessimistic (large) so that real floor calibration later produces
violation_ratios < the shipped values — avoids spurious PASS on uncalibrated
installs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

if TYPE_CHECKING:
    from physics_lint.field import Field, GridField
    from physics_lint.spec import DomainSpec

Status = Literal["PASS", "WARN", "FAIL", "SKIPPED"]

# Module-level constant so tests can monkey-patch the floors file location
# without racing the real file at physics_lint/data/floors.toml. _load_floor
# reads from this path at call time, so monkey-patching via
# `monkeypatch.setattr(_helpers, "_FLOORS_PATH", tmp_path / "floors.toml")`
# redirects cleanly.
_FLOORS_PATH: Path = Path(__file__).parent.parent / "data" / "floors.toml"


@dataclass
class Floor:
    value: float
    tolerance: float
    source: str  # "shipped" | "calibrated"


_SHIPPED_DEFAULTS: dict[tuple[str, str, str, str], float] = {
    # Conservative defaults; refined by Task 14 floors.toml
    ("PH-RES-001", "laplace", "fd4", "H-1"): 1e-5,
    ("PH-RES-001", "laplace", "spectral", "H-1"): 1e-13,
    ("PH-RES-001", "poisson", "fd4", "H-1"): 1e-5,
    ("PH-RES-001", "poisson", "spectral", "H-1"): 1e-13,
    ("PH-RES-001", "laplace", "fd4", "L2"): 1e-5,
    ("PH-RES-001", "laplace", "spectral", "L2"): 1e-13,
    ("PH-BC-001", "laplace", "fd4", "L2-rel"): 1e-11,
    ("PH-BC-001", "poisson", "fd4", "L2-rel"): 1e-11,
}

_TOLERANCE_DEFAULTS: dict[str, float] = {
    "spectral": 3.0,
    "fd4": 2.0,
}


def _tristate(ratio: float, pass_: float, fail_: float) -> Status:
    """Tri-state classification against the calibrated floor.

    ratio <= pass_: PASS
    pass_ < ratio <= fail_: WARN
    ratio > fail_: FAIL
    """
    if ratio <= pass_:
        return "PASS"
    if ratio <= fail_:
        return "WARN"
    return "FAIL"


def _load_floor(
    *,
    rule: str,
    pde: str,
    grid_shape: tuple[int, ...],
    method: str,
    norm: str,
) -> Floor:
    """Load a floor entry from floors.toml, falling back to shipped defaults.

    Task 14 populates physics_lint/data/floors.toml with calibrated values;
    the shipped-default fallback is exercised in CI and in the Week 1 test
    suite so rules never raise KeyError on first install.
    """
    if _FLOORS_PATH.is_file():
        with open(_FLOORS_PATH, "rb") as f:
            data = tomllib.load(f)
        for entry in data.get("floor", []):
            if (
                entry.get("rule") == rule
                and entry.get("pde") == pde
                and tuple(entry.get("grid_shape", ())) == tuple(grid_shape)
                and entry.get("method") == method
                and entry.get("norm") == norm
            ):
                return Floor(
                    value=float(entry["value"]),
                    tolerance=float(entry.get("tolerance", _TOLERANCE_DEFAULTS.get(method, 2.0))),
                    source="calibrated",
                )

    default = _SHIPPED_DEFAULTS.get((rule, pde, method, norm))
    if default is None:
        default = 1e-5
    return Floor(
        value=default,
        tolerance=_TOLERANCE_DEFAULTS.get(method, 2.0),
        source="shipped",
    )


def ensure_grid_field(field: Field, spec: DomainSpec) -> GridField:
    """Return a GridField, materializing a CallableField if necessary.

    Rules that compute FD/spectral-based quantities (Laplacian, boundary
    traces, per-slice sub-fields) need a GridField with a concrete
    backend. CallableField is the loader's representation of adapter-mode
    inputs: torch-callable plus a sampling grid. This helper materializes
    the callable onto the grid, picks a backend from spec.field.backend
    (resolving "auto" via spec.periodic the same way the dump loader
    does), and rebuilds a GridField. The materialization is cheap —
    CallableField caches it internally via _materialize — so multiple
    rules in a run share one materialization.

    Non-CallableField non-GridField inputs raise TypeError with a
    diagnostic so malformed loader returns surface immediately.
    """
    # Local imports to avoid circulars: field and spec pull in physics_lint
    # top-level packages that import this module at init time.
    from physics_lint.field import CallableField
    from physics_lint.field import GridField as _GridField

    if isinstance(field, _GridField):
        return field
    if isinstance(field, CallableField):
        resolved_backend: str
        if spec.field.backend == "auto":
            resolved_backend = "spectral" if spec.periodic else "fd"
        else:
            resolved_backend = spec.field.backend
        return _GridField(
            field.values(),
            h=field.h,
            periodic=field.periodic,
            backend=resolved_backend,
        )
    raise TypeError(f"rule requires GridField or CallableField; got {type(field).__name__}")
