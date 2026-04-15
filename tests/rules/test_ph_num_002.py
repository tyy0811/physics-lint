"""PH-NUM-002 — refinement convergence rate rule."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_num_002


def _non_periodic_spec(n: int) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [n, n],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def _harmonic_expcos(n: int) -> GridField:
    """u = exp(x) cos(y) on [0, 1]^2: analytically harmonic.

    The FD residual -Delta_h u is pure operator truncation error —
    4th-order in the interior, 2nd-order along the outer boundary band,
    giving a measured L^2 rate of ~2.5 per doubling on fd4.
    """
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    u = np.exp(X) * np.cos(Y)
    return GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False, backend="fd")


def test_ph_num_002_skipped_without_refined_field():
    spec = _non_periodic_spec(64)
    field = _harmonic_expcos(64)
    result = ph_num_002.check(field, spec)
    assert result.status == "SKIPPED"
    assert "refined_field" in (result.reason or "")


def test_ph_num_002_fd4_converges_on_harmonic():
    spec = _non_periodic_spec(64)
    result = ph_num_002.check(
        _harmonic_expcos(64),
        spec,
        refined_field=_harmonic_expcos(128),
    )
    assert result.rule_id == "PH-NUM-002"
    assert result.refinement_rate is not None
    # Measured 64 -> 128 on exp(x)cos(y) non-periodic fd4 is ~2.5.
    assert 1.8 <= result.refinement_rate <= 3.5
    assert result.status == "PASS"


def test_ph_num_002_warns_on_non_converging_field():
    # White Gaussian noise on a random grid does not converge under
    # refinement: ||Delta_h noise|| is dominated by the h^-2 pixel-scale
    # noise amplification, which shrinks by a constant under doubling,
    # not by O(h^2). The measured rate stays near 0 and we expect WARN.
    spec = _non_periodic_spec(64)
    rng = np.random.default_rng(7)
    coarse_values = rng.normal(0.0, 1.0, size=(64, 64))
    fine_values = rng.normal(0.0, 1.0, size=(128, 128))
    coarse_field = GridField(
        coarse_values,
        h=(1.0 / 63, 1.0 / 63),
        periodic=False,
        backend="fd",
    )
    fine_field = GridField(
        fine_values,
        h=(1.0 / 127, 1.0 / 127),
        periodic=False,
        backend="fd",
    )
    result = ph_num_002.check(coarse_field, spec, refined_field=fine_field)
    assert result.status == "WARN"
    assert result.refinement_rate is not None
    assert result.refinement_rate < 1.8


def test_ph_num_002_rejects_non_grid_refined_field():
    spec = _non_periodic_spec(64)
    field = _harmonic_expcos(64)
    try:
        ph_num_002.check(field, spec, refined_field=object())  # type: ignore[arg-type]
    except TypeError as e:
        assert "refined_field" in str(e)
        return
    raise AssertionError("expected TypeError for non-GridField refined_field")


def _poisson_periodic_spec(n: int) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [n, n],
            "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )


def _poisson_exact_sin_sin(n: int) -> GridField:
    """u = sin(x) sin(y), -Laplace(u) = 2 sin(x) sin(y); exact solution."""
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    u = np.sin(X) * np.sin(Y)
    return GridField(u, h=(2 * np.pi / n, 2 * np.pi / n), periodic=True, backend="spectral")


def test_ph_num_002_poisson_is_skipped():
    # Review regression: previously this WARN-ed with rate ~0 because the
    # rule treated ||Delta u|| as the residual and never subtracted the
    # Poisson source. The V1 scope is now Laplace-only and this path must
    # SKIP with a clear reason.
    spec = _poisson_periodic_spec(64)
    result = ph_num_002.check(
        _poisson_exact_sin_sin(64),
        spec,
        refined_field=_poisson_exact_sin_sin(128),
    )
    assert result.status == "SKIPPED"
    assert "laplace" in (result.reason or "").lower()
    assert "poisson" in (result.reason or "").lower()


def _heat_periodic_spec(n: int, nt: int) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [n, n, nt],
            "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi], "t": [0.0, 0.5]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "diffusivity": 0.01,
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )


def _heat_exact(n: int, nt: int) -> GridField:
    """u = cos(x) cos(y) exp(-2 * kappa * t), an exact heat solution."""
    kappa = 0.01
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    t = np.linspace(0.0, 0.5, nt)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    pred = np.stack(
        [np.cos(X) * np.cos(Y) * np.exp(-2 * kappa * ti) for ti in t],
        axis=-1,
    )
    return GridField(
        pred,
        h=(2 * np.pi / n, 2 * np.pi / n, 0.5 / (nt - 1)),
        periodic=True,
        backend="spectral",
    )


def test_ph_num_002_saturated_residual_passes_without_noise_rate():
    # Harmonic polynomial x^2 - y^2 has FD Laplacian at machine roundoff
    # (~1e-12) on both 64^2 and 128^2 grids, so the log2 ratio of the two
    # residuals is pure floating-point noise (e.g., -1.94) and cannot be
    # interpreted as a convergence rate. The rule should clamp the rate
    # to 'inf' via the saturation floor and PASS.
    spec = _non_periodic_spec(64)

    def _harmonic_polynomial(n: int) -> GridField:
        x = np.linspace(0.0, 1.0, n)
        y = np.linspace(0.0, 1.0, n)
        X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
        u = X**2 - Y**2
        return GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False, backend="fd")

    result = ph_num_002.check(
        _harmonic_polynomial(64),
        spec,
        refined_field=_harmonic_polynomial(128),
    )
    assert result.status == "PASS"
    assert result.refinement_rate == float("inf")


def test_ph_num_002_heat_is_skipped():
    # Review regression: previously this WARN-ed because field.laplacian()
    # differentiated over the time axis too, producing garbage. The V1
    # scope is Laplace-only so heat must SKIP.
    spec = _heat_periodic_spec(64, 16)
    result = ph_num_002.check(
        _heat_exact(64, 16),
        spec,
        refined_field=_heat_exact(128, 16),
    )
    assert result.status == "SKIPPED"
    assert "heat" in (result.reason or "").lower()
