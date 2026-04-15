"""Test fixture: generate a valid .npz dump file on demand.

Called by test_loader.py via `write_good_dump(tmp_path)`. Not committed as
a .npz artifact because the dump file should be regenerated fresh for each
test run to catch silent drift.
"""

from pathlib import Path

import numpy as np


def write_good_dump(path: Path) -> Path:
    n = 32
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    u = xx**2 - yy**2  # harmonic on [0,1]^2
    metadata = {
        "pde": "laplace",
        "grid_shape": [n, n],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": "dirichlet",
        "field": {"type": "grid", "backend": "fd"},
    }
    np.savez(path, prediction=u, metadata=metadata)
    return path
