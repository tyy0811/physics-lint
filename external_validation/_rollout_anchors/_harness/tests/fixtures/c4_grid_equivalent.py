"""Shared discretisation utility for the controlled-fixture validation.

Per spec §4.2 (fixture #3, `c4_grid_equivalent.py`): used by fixtures #1
and #2 to materialize the gridded equivalent of a particle configuration
on which the public-API path is run.

This module's job is to keep the gridify parameters (`grid_size`,
`bandwidth`, `domain`, periodicity convention) consistent across fixtures
so the harness-vs-public-API comparison in
`test_harness_vs_public_api.py` is apples-to-apples.

Both halves of Gate B's comparison consume the same gridded
representation:

- Harness path: ``c4_static_defect(snapshot, grid_size=GRID_SIZE,
  bandwidth=BANDWIDTH, domain=DOMAIN)`` — internally calls ``gridify``
  with these parameters.

- Public-API path: ``gridify(snapshot, grid_size=GRID_SIZE,
  bandwidth=BANDWIDTH, domain=DOMAIN)`` produces the same array, which
  is then wrapped in a ``GridField`` and handed to ``ph_sym_001.check``.

Centralising the parameter triple here means a future bug in either
path can only show up as a real divergence — not as a parameter
mismatch artefact.
"""

from __future__ import annotations

import numpy as np

from physics_lint import DomainSpec, GridField

# Grid parameters shared by all fixtures and by the Gate B test.
GRID_SIZE: int = 64
BANDWIDTH: float = 0.04
DOMAIN: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0))
GRID_SPACING: float = (DOMAIN[0][1] - DOMAIN[0][0]) / GRID_SIZE


def make_grid_field(values: np.ndarray) -> GridField:
    """Wrap a gridded scalar field in physics-lint's public ``GridField``.

    Constructed with ``periodic=True`` and ``backend='spectral'``, which
    matches the periodic convention used in :func:`gridify` (minimum-image
    distance, ``endpoint=False`` linspace) and lets the public PH-SYM-001
    rule consume the field without reflection-axis padding artefacts.
    """
    return GridField(values, h=GRID_SPACING, periodic=True, backend="spectral")


def make_c4_spec() -> DomainSpec:
    """DomainSpec for fixtures #1 and #2 (PH-SYM-001 invocation).

    Matches the parameters declared in :func:`make_grid_field`. C4,
    reflection_x, and reflection_y are all declared so PH-SYM-001 and
    PH-SYM-002 both fire on the same field.
    """
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [GRID_SIZE, GRID_SIZE],
            "domain": {"x": list(DOMAIN[0]), "y": list(DOMAIN[1])},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "symmetries": {"declared": ["C4", "reflection_x", "reflection_y"]},
            "field": {
                "type": "grid",
                "backend": "spectral",
                "dump_path": "fixture.npz",
            },
        }
    )
