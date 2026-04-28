"""Conservation-law energy helpers (Task 8 mass + Task 9 wave energy).

Populated by Task 8 (PH-CON-001 heat mass conservation) and Task 9
(PH-CON-002 wave energy conservation).

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

from physics_lint.norms import integrate_over_domain

HeatSnapshotFixture = Literal["cos_2pi_2d"]
WaveSnapshotFixture = Literal["sin_kx_cos_ckt_yindep"]


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


def analytical_wave_snapshot_2d_yindep(
    *,
    fixture: WaveSnapshotFixture = "sin_kx_cos_ckt_yindep",
    nx: int,
    nt: int,
    c: float = 1.0,
    k: int = 1,
    t_end: float | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[float, float, float],
]:
    """Analytical-snapshot wave solution on 2D periodic `[0, 2 pi]^2`, y-independent.

    Fixture ``"sin_kx_cos_ckt_yindep"``:
        ``u(x, y, t) = sin(k x) cos(c k t)``
        (y-independent — the 1D wave solution lifted to 2D by trivial y extension
        so the field satisfies the rule's ``ndim >= 3`` contract.)

    Exact solution of the 1D wave equation ``u_tt = c^2 u_xx`` with
    periodic boundary on ``[0, 2 pi]`` and zero y dependence. Analytical
    energy density:

        u_t(x, y, t) = -c k sin(k x) sin(c k t)
        u_x(x, y, t) =  k cos(k x) cos(c k t)
        u_y(x, y, t) =  0

        E(t) = 0.5 integral_Omega (u_t^2 + c^2 (u_x^2 + u_y^2)) dV
             = pi^2 c^2 k^2            (constant in t)

    (Derivation: ``int_0^{2pi} sin^2(k x) dx = int_0^{2pi} cos^2(k x) dx
    = pi`` for integer ``k``; cross terms ``sin^2(c k t) + cos^2(c k t)
    = 1``; y integral over ``[0, 2 pi]`` contributes a factor of ``2 pi``.)

    Parameters
    ----------
    fixture : WaveSnapshotFixture
        Currently only ``"sin_kx_cos_ckt_yindep"`` is supported.
    nx : int
        Spatial grid size per axis. Periodic endpoint-exclusive.
    nt : int
        Number of time snapshots (``>= 3`` per the rule's
        ``_MIN_TIME_STEPS_FOR_GRADIENT`` contract).
    c : float, default 1.0
        Wave speed. Must be positive.
    k : int, default 1
        Spatial wavenumber (integer so the fixture stays periodic on
        ``[0, 2 pi]``). ``k <= nx / 2`` to stay below Nyquist.
    t_end : float or None
        Final time of the snapshot window. Defaults to one full wave
        period ``2 pi / (c k)``.

    Returns
    -------
    (u, u_t, u_x, u_y, h) : tuple of arrays plus the spacing tuple.
        Each array has shape ``(nx, nx, nt)``; ``h = (hx, hy, ht)``
        with ``hx = hy = 2 pi / nx`` for ``GridField`` construction.
        ``u_y`` is an all-zeros array by construction of this fixture.
    """
    if fixture != "sin_kx_cos_ckt_yindep":
        raise ValueError(f"unknown fixture={fixture!r}; expected 'sin_kx_cos_ckt_yindep'")
    if nx < 4:
        raise ValueError(f"nx must be >= 4; got {nx}")
    if nt < 3:
        raise ValueError(f"nt must be >= 3 (rule needs 3 time samples); got {nt}")
    if c <= 0:
        raise ValueError(f"c must be positive; got {c}")
    if k < 1 or 2 * k > nx:
        raise ValueError(f"k must satisfy 1 <= 2 * k <= nx (Nyquist); got k={k}, nx={nx}")
    if t_end is None:
        t_end = 2.0 * np.pi / (c * k)
    if t_end <= 0:
        raise ValueError(f"t_end must be positive; got {t_end}")

    xs = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
    ys = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
    ts = np.linspace(0.0, t_end, nt)
    mesh_x, _mesh_y = np.meshgrid(xs, ys, indexing="ij")

    u = np.empty((nx, nx, nt), dtype=np.float64)
    u_t = np.empty((nx, nx, nt), dtype=np.float64)
    u_x = np.empty((nx, nx, nt), dtype=np.float64)
    u_y = np.zeros((nx, nx, nt), dtype=np.float64)

    sin_kx = np.sin(k * mesh_x)
    cos_kx = np.cos(k * mesh_x)
    for ki, tk in enumerate(ts):
        u[..., ki] = sin_kx * np.cos(c * k * tk)
        u_t[..., ki] = -c * k * sin_kx * np.sin(c * k * tk)
        u_x[..., ki] = k * cos_kx * np.cos(c * k * tk)

    hx = 2.0 * np.pi / nx
    ht = ts[1] - ts[0] if nt > 1 else t_end
    return u, u_t, u_x, u_y, (hx, hx, ht)


def wave_energy_from_analytical_fields(
    *,
    u_t: np.ndarray,
    u_x: np.ndarray,
    u_y: np.ndarray,
    c: float,
    h: tuple[float, float, float],
    periodic: bool = True,
) -> np.ndarray:
    """Compute ``E(t) = 0.5 integral (u_t^2 + c^2 (u_x^2 + u_y^2)) dV`` per time step.

    Authoritative F2 layer for Task 9 PH-CON-002: computes energy from
    analytical field components directly (no finite-difference
    approximation of ``u_t``, ``u_x``, ``u_y``). Drift across time
    snapshots is expected at float64 roundoff (~1e-16 relative) on
    analytically controlled conservative snapshots.

    Parameters
    ----------
    u_t, u_x, u_y : np.ndarray of shape (nx, ny, nt)
        Analytical time-derivative and spatial-gradient components.
    c : float
        Wave speed (appears squared in the energy density).
    h : (hx, hy, ht)
        Grid spacing tuple; time spacing ht is unused here (energies
        are per-time-snapshot, not time-integrated).
    periodic : bool, default True
        Selects rectangle-rule (endpoint-exclusive periodic) vs
        trapezoidal quadrature in ``integrate_over_domain``.

    Returns
    -------
    np.ndarray of shape (nt,)
        Energy at each time snapshot.
    """
    if u_t.shape != u_x.shape or u_x.shape != u_y.shape:
        raise ValueError(
            f"u_t, u_x, u_y must have matching shape; got {u_t.shape}, {u_x.shape}, {u_y.shape}"
        )
    hx, hy, _ = h
    density = 0.5 * (u_t**2 + (c**2) * (u_x**2 + u_y**2))
    nt = density.shape[-1]
    energies = np.empty(nt)
    for ki in range(nt):
        energies[ki] = integrate_over_domain(density[..., ki], (hx, hy), periodic=periodic)
    return energies
