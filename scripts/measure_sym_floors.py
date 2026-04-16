"""Floor calibration script for SYM rules (PH-SYM-001/002/003).

Measures the analytical equivariance error for each SYM rule on the
identity-map input u = x^2 + y^2 (centered grid) at resolutions
32/64/128.  Output is a JSON summary suitable for cross-environment
comparison: run on local macOS, GHA macOS arm64, and GHA Ubuntu x86_64,
then take the MAXIMUM across environments as the floors.toml ``value``.

Week 3 three-environment calibration found bitwise-identical results
across all platforms for SYM-001/002 (index permutations are
deterministic) and SYM-003 (algebraic cancellation in forward-mode AD).
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch

from physics_lint.rules._symmetry_helpers import equivariance_error_np


@dataclass
class FloorEntry:
    rule: str
    pde: str
    grid_shape: tuple[int, ...]
    method: str
    norm: str
    measured: float
    analytical_solution: str


def _measure_sym001_rot90(n: int) -> float:
    """PH-SYM-001: max relative L2 over k in {1, 2, 3} for np.rot90.

    u = x^2 + y^2 on a centered grid is C4-symmetric; rot90 is an exact
    index permutation, so the error is pure ULP noise.
    """
    half = (n - 1) / 2.0
    x = np.linspace(-half, half, n) / half
    y = np.linspace(-half, half, n) / half
    xg, yg = np.meshgrid(x, y, indexing="ij")
    u = xg**2 + yg**2
    errs = [equivariance_error_np(np.rot90(u, k=k), u) for k in (1, 2, 3)]
    return max(errs)


def _measure_sym002_flip(n: int) -> float:
    """PH-SYM-002: max relative L2 over reflection axes for np.flip.

    u = x^2 + y^2 on a centered grid is axis-symmetric; np.flip is an
    exact index permutation.
    """
    half = (n - 1) / 2.0
    x = np.linspace(-half, half, n) / half
    y = np.linspace(-half, half, n) / half
    xg, yg = np.meshgrid(x, y, indexing="ij")
    u = xg**2 + yg**2
    err_x = equivariance_error_np(np.flip(u, axis=0), u)
    err_y = equivariance_error_np(np.flip(u, axis=1), u)
    return max(err_x, err_y)


def _measure_sym003_jvp(n: int) -> float:
    """PH-SYM-003: per-point L2 of the SO(2) Lie derivative via JVP.

    u = x^2 + y^2 is radially symmetric; the Lie derivative d/dtheta of
    u(R_theta x) at theta=0 is 2x(-y) + 2y(x) = 0 analytically.
    Forward-mode AD preserves this exactly.
    """
    axis = torch.linspace(-0.5, 0.5, n)
    grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1)

    def radial(pts: torch.Tensor) -> torch.Tensor:
        return (pts[..., 0] ** 2 + pts[..., 1] ** 2).unsqueeze(-1)

    def rotated_model(theta_param: torch.Tensor) -> torch.Tensor:
        c = torch.cos(theta_param)
        s = torch.sin(theta_param)
        x = grid[..., 0]
        y = grid[..., 1]
        x_rot = c * x - s * y
        y_rot = s * x + c * y
        rotated_grid = torch.stack([x_rot, y_rot], dim=-1)
        return radial(rotated_grid).squeeze(-1)

    theta0 = torch.zeros(1)
    tangent = torch.ones_like(theta0)

    from torch.autograd.functional import jvp

    _, lie_deriv = jvp(rotated_model, (theta0,), v=(tangent,))
    return float(torch.norm(lie_deriv).item() / max(float(lie_deriv.numel()), 1.0) ** 0.5)


def main() -> int:
    resolutions = [32, 64, 128]
    entries: list[FloorEntry] = []

    for n in resolutions:
        val_001 = _measure_sym001_rot90(n)
        entries.append(
            FloorEntry(
                rule="PH-SYM-001",
                pde="laplace",
                grid_shape=(n, n),
                method="rot90",
                norm="max-rel-L2",
                measured=val_001,
                analytical_solution="radial_x2_y2_centered",
            )
        )

        val_002 = _measure_sym002_flip(n)
        entries.append(
            FloorEntry(
                rule="PH-SYM-002",
                pde="laplace",
                grid_shape=(n, n),
                method="flip",
                norm="max-rel-L2",
                measured=val_002,
                analytical_solution="radial_x2_y2_centered",
            )
        )

        val_003 = _measure_sym003_jvp(n)
        entries.append(
            FloorEntry(
                rule="PH-SYM-003",
                pde="laplace",
                grid_shape=(n, n),
                method="autograd_jvp",
                norm="per-point-L2",
                measured=val_003,
                analytical_solution="radial_x2_y2_centered",
            )
        )

    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    payload = {"environment": env, "entries": [asdict(e) for e in entries]}
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
