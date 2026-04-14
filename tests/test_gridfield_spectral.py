"""GridField Fourier spectral backend tests.

Spectral derivatives on smooth periodic inputs are exact to roughly
machine precision; the tolerance here reflects floating-point noise
plus FFT backend drift (see design doc §6.3).
"""

import numpy as np

from physics_lint.field import GridField


def test_spectral_laplacian_sine_machine_precision():
    # -∆ of sin(2*pi*x)*sin(2*pi*y) = 8*pi^2 * sin*sin at machine precision.
    N = 64  # noqa: N806  (N is grid resolution; math convention)
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    h = 1.0 / N
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    f = GridField(u, h=h, periodic=True)  # auto-selects spectral
    lap = f.laplacian().values()
    expected = -8 * np.pi**2 * u

    assert np.max(np.abs(lap - expected)) < 1e-11


def test_spectral_laplacian_multimode():
    # Higher modes: -∆ sin(k*pi*x) sin(k*pi*y) = 2*(k*pi)^2 * sin*sin
    #
    # Tolerance note: FFT floating-point noise scales with the expected
    # signal amplitude, which grows as k^2 (from the Laplacian eigenvalue).
    # An *absolute* tolerance therefore tightens against the highest mode.
    # We use a *relative* tolerance pegged a few orders of magnitude above
    # machine epsilon (empirically ~1e-13 rel). This checks the intrinsic
    # invariant — "spectral Laplacian is exact up to FFT roundoff" — rather
    # than an amplitude-coupled bound.
    N = 128  # noqa: N806  (N is grid resolution; math convention)
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806  (meshgrid coords)
    for k in (1, 3, 7):
        u = np.sin(2 * k * np.pi * X) * np.sin(2 * k * np.pi * Y)
        f = GridField(u, h=h, periodic=True)
        lap = f.laplacian().values()
        expected = -2 * (2 * k * np.pi) ** 2 * u
        rel_err = np.max(np.abs(lap - expected)) / np.max(np.abs(expected))
        assert rel_err < 1e-12, f"k={k} failed rel_err={rel_err:.3e}"


def test_spectral_laplacian_3d():
    N = 32  # noqa: N806  (N is grid resolution; math convention)
    h = 1.0 / N
    axis = np.linspace(0.0, 1.0, N, endpoint=False)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")  # noqa: N806  (meshgrid coords)
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) * np.sin(2 * np.pi * Z)
    f = GridField(u, h=h, periodic=True)
    lap = f.laplacian().values()
    expected = -12 * np.pi**2 * u
    assert np.max(np.abs(lap - expected)) < 1e-10
