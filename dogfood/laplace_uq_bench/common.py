"""Shared helpers for the dogfood dump generators.

One place to put the laplace-uq-bench import bootstrap (it is not a
physics-lint dependency and is not installed; we just add its src/ to
sys.path), the canonical test-case definition, and the .npz writer so
every surrogate dump has identical metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
DOGFOOD = HERE.parent
REPO = DOGFOOD / "laplace-uq-bench"

NX = 64
SEED = 20260415


def _bootstrap_repo_imports() -> None:
    """Put the cloned laplace-uq-bench src/ directory on sys.path."""
    repo_src = REPO / "src"
    if not repo_src.is_dir():
        raise RuntimeError(
            f"{repo_src} does not exist. Run dogfood/clone_laplace_uq_bench.sh first."
        )
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))


def canonical_test_case() -> tuple[np.ndarray, np.ndarray]:
    """Build the reference oracle solution on a 64x64 grid.

    Uses the repo's own LaplaceSolver and boundary sampler so the dogfood
    truly exercises laplace-uq-bench code. Returns (oracle_field, bc_bundle)
    where bc_bundle is a stacked (4, NX) array of [top, bottom, left, right]
    that the defect generators can reuse to keep BCs consistent.
    """
    _bootstrap_repo_imports()
    from diffphys.pde.boundary import IN_DIST_TYPES, sample_four_edges
    from diffphys.pde.laplace import LaplaceSolver

    rng = np.random.default_rng(SEED)
    bc_top, bc_bottom, bc_left, bc_right = sample_four_edges(
        rng, allowed_types=IN_DIST_TYPES, nx=NX
    )
    solver = LaplaceSolver(nx=NX)
    field = solver.solve(bc_top, bc_bottom, bc_left, bc_right).astype(np.float64)
    bc_bundle = np.stack([bc_top, bc_bottom, bc_left, bc_right]).astype(np.float64)
    return field, bc_bundle


def dump_metadata() -> dict[str, Any]:
    """DomainSpec-shaped metadata dict that matches every surrogate dump."""
    return {
        "pde": "laplace",
        "grid_shape": [NX, NX],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet"},
        "symmetries": {"declared": []},
        "field": {"type": "grid", "backend": "fd"},
    }


def write_dump(path: Path, prediction: np.ndarray) -> None:
    """Save an .npz dump at ``path`` with a matching metadata dict."""
    if prediction.shape != (NX, NX):
        raise ValueError(f"prediction must be ({NX}, {NX}); got {prediction.shape}")
    np.savez(path, prediction=prediction.astype(np.float64), metadata=dump_metadata())
