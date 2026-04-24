"""Conservation-law energy helpers (Task 8 mass; Task 9 wave-energy TBD).

Populated by Task 8 (PH-CON-001 heat mass conservation). Task 9
(PH-CON-002 wave energy) extends this module with `wave_energy` and
`energy_conservation_ratio` helpers at its execution time.

Scope-separation discipline (Task 8 analytic-snapshot mode, per
2026-04-24 user-approved revised Task 8 contract):

PH-CON-001 validates the production rule's ability to measure
integral conservation drift on analytically controlled source-free
snapshots. It does not certify the accuracy of a heat-equation time
integrator. The helper in this module constructs 2D periodic
analytical snapshots with known zero spatial mean at all t; the rule
consumes those snapshots and emits a conservation-drift metric that
should land at numerical roundoff.

Numerically-evolved heat solutions (FD time-stepper on a discretized
PDE) are out of scope for the Task 8 F2 layer. A solver-accuracy
fixture would be a separate anchor (not in V1 scope). Mixing
tolerances between analytical-snapshot and numerically-evolved modes
is explicitly disallowed per the revised contract.

References (F1 mathematical-legitimacy):
- Evans 2010 §2.3 (heat-equation fundamental solution preserves
  integral), section-level per `TEXTBOOK_AVAILABILITY.md` ⚠.
- Dafermos 2016 Chapter I (balance-law framing), section-level.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

HeatSnapshotFixture = Literal["cos_2pi_2d"]


def analytical_heat_snapshot_2d(
    *,
    fixture: HeatSnapshotFixture = "cos_2pi_2d",
    nx: int,
    nt: int,
    kappa: float = 1.0,
    t_end: float = 0.1,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Analytical-snapshot heat solution on 2D periodic unit square.

    Fixture ``"cos_2pi_2d"``:
        ``u(x, y, t) = cos(2 pi x) cos(2 pi y) * exp(-8 pi^2 kappa t)``

    This is an exact solution to the 2D heat equation with periodic
    boundary conditions on ``[0, 1]^2`` with diffusivity ``kappa``.
    Its spatial integral ``int_Omega u(x, y, t) dV = 0`` exactly at
    every ``t`` (zero-mean trigonometric mode on the periodic domain).

    The rectangle rule on an endpoint-exclusive periodic grid is
    spectrally accurate on this integrand — the discrete integral
    equals the analytical integral (zero) up to float64 roundoff
    from the FFT-adjacent accumulation, empirically observed at
    ``~1e-18`` relative to the `L^1`-norm scale of ``~1``.

    Parameters
    ----------
    fixture : HeatSnapshotFixture
        Currently only ``"cos_2pi_2d"`` is supported.
    nx : int
        Spatial grid size per axis. Periodic endpoint-exclusive
        (``nx`` points in ``[0, 1)``).
    nt : int
        Number of time snapshots (``>= 3`` per the rule's
        ``_MIN_TIME_STEPS_FOR_GRADIENT`` contract).
    kappa : float, default 1.0
        Heat diffusivity. Must be positive.
    t_end : float, default 0.1
        Final time of the snapshot window ``[0, t_end]``.

    Returns
    -------
    (u, h) : tuple[np.ndarray, tuple[float, float, float]]
        ``u`` has shape ``(nx, nx, nt)`` (ndim=3 matching the rule's
        ``_MIN_TIME_STEPS_FOR_GRADIENT``-compatible time-dependent
        contract). ``h = (hx, hy, ht)`` is the spacing tuple for the
        ``GridField`` constructor.
    """
    if fixture != "cos_2pi_2d":
        raise ValueError(f"unknown fixture={fixture!r}; expected 'cos_2pi_2d'")
    if nx < 4:
        raise ValueError(f"nx must be >= 4; got {nx}")
    if nt < 3:
        raise ValueError(f"nt must be >= 3 (rule needs 3 time samples); got {nt}")
    if kappa <= 0:
        raise ValueError(f"kappa must be positive; got {kappa}")
    if t_end <= 0:
        raise ValueError(f"t_end must be positive; got {t_end}")

    xs = np.linspace(0.0, 1.0, nx, endpoint=False)
    ys = np.linspace(0.0, 1.0, nx, endpoint=False)
    ts = np.linspace(0.0, t_end, nt)
    mesh_x, mesh_y = np.meshgrid(xs, ys, indexing="ij")
    spatial_mode = np.cos(2.0 * np.pi * mesh_x) * np.cos(2.0 * np.pi * mesh_y)
    time_decay = np.exp(-8.0 * np.pi**2 * kappa * ts)

    u = np.empty((nx, nx, nt), dtype=np.float64)
    for k, factor in enumerate(time_decay):
        u[..., k] = spatial_mode * factor

    hx = 1.0 / nx
    ht = ts[1] - ts[0] if nt > 1 else t_end
    return u, (hx, hx, ht)
