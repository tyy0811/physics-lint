"""Shared rule helpers: tristate thresholds, floor loading, safe ratios.

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
from typing import Literal

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

Status = Literal["PASS", "WARN", "FAIL", "SKIPPED"]


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
    floors_path = Path(__file__).parent.parent / "data" / "floors.toml"
    if floors_path.is_file():
        with open(floors_path, "rb") as f:
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
