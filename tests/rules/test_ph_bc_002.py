"""PH-BC-002 — boundary flux imbalance via divergence theorem."""

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.loader import load_target
from physics_lint.rules import ph_bc_002


def _laplace_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [64, 64],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def test_ph_bc_002_harmonic_has_zero_net_flux():
    # u = x^2 - y^2, harmonic. Net flux around the boundary = 0 (up to FD error).
    spec = _laplace_spec()
    n = 64
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    u = mesh_x**2 - mesh_y**2
    field = GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)
    result = ph_bc_002.check(field, spec)
    assert result.rule_id == "PH-BC-002"
    assert result.status == "PASS"
    assert result.raw_value is not None
    assert abs(result.raw_value) < 0.01  # small FD edge contribution


def test_ph_bc_002_non_harmonic_has_nonzero_net_flux():
    spec = _laplace_spec()
    n = 64
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(xg, yg, indexing="ij")
    # u = x^2 + y^2 has Laplacian = 4, net flux integral = 4 (genuinely non-harmonic).
    # NOTE: The plan originally used exp(x)*sin(y), but that IS harmonic
    # (Laplacian = exp(x)*sin(y) + exp(x)*(-sin(y)) = 0), so the net flux is
    # ~0 and the test incorrectly passes. x^2 + y^2 is the fix.
    u = mesh_x**2 + mesh_y**2
    field = GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)
    result = ph_bc_002.check(field, spec)
    assert result.status in {"WARN", "FAIL"}
    assert result.raw_value is not None and abs(result.raw_value) > 0.01


def test_ph_bc_002_heat_pde_is_skipped():
    spec = DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [16, 16, 4],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.1]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
            "diffusivity": 0.01,
        }
    )
    field = GridField(np.zeros((16, 16)), h=(1.0 / 15, 1.0 / 15), periodic=False)
    result = ph_bc_002.check(field, spec)
    assert result.status == "SKIPPED"
    assert "laplace/poisson only" in result.reason


def test_ph_bc_002_accepts_callable_field_adapter_mode():
    """Adapter-mode: PH-BC-002 materializes the callable and runs the
    divergence-theorem check against the sampled values. Harmonic
    u = x^2 - y^2 has zero net flux, so the rule should PASS."""
    import torch

    from physics_lint import CallableField

    n = 64
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, n),
            torch.linspace(0.0, 1.0, n),
            indexing="ij",
        ),
        dim=-1,
    )
    field = CallableField(
        lambda x: (x[..., 0] ** 2 - x[..., 1] ** 2).unsqueeze(-1),
        sampling_grid=grid,
        h=(1.0 / (n - 1), 1.0 / (n - 1)),
        periodic=False,
    )
    spec = _laplace_spec()
    result = ph_bc_002.check(field, spec)
    assert result.status == "PASS"
    assert result.raw_value is not None and abs(result.raw_value) < 0.01


def test_ph_bc_002_metadata():
    assert ph_bc_002.__rule_id__ == "PH-BC-002"
    assert ph_bc_002.__default_severity__ == "warning"


def _poisson_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [64, 64],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def _grid_xy(n: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.linspace(0.0, 1.0, n)
    return np.meshgrid(g, g, indexing="ij")


def test_ph_bc_002_poisson_consistent_field_passes():
    # u = x^2 + y^2  =>  Laplacian(u) = 4.  Poisson convention -Lap(u) = f
    # => f = -4 (constant). A consistent (u, f) pair has imbalance ~ 0.
    n = 64
    mesh_x, mesh_y = _grid_xy(n)
    u = mesh_x**2 + mesh_y**2
    spec = _poisson_spec()
    source = np.full((n, n), -4.0)
    # The loader injects the source array as spec._source_array via
    # object.__setattr__; mirror that here for the unit test.
    object.__setattr__(spec, "_source_array", source)
    field = GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)
    result = ph_bc_002.check(field, spec)
    assert result.status == "PASS", f"expected PASS, got {result.status} ({result.reason})"
    assert result.raw_value is not None
    # u = x^2+y^2 is degree-2: the FD4 Laplacian is exact, so Delta u = 4
    # everywhere and the trapezoidal integral of a constant is exact. The
    # imbalance is pure float64 roundoff (~7e-13), not FD truncation error
    # -- a loose bound like 0.05 would not distinguish a correct arm from a
    # moderately broken one.
    assert abs(result.raw_value) < 1e-12


def test_ph_bc_002_poisson_inconsistent_field_warns_or_fails():
    # u = x^2 + y^2  =>  Laplacian(u) = 4, so the consistent source is
    # f = -4. Feed f = 0 instead: imbalance = integral(Lap u) + 0 ~ 4,
    # which is large relative to ||u||, so the rule must WARN or FAIL.
    n = 64
    mesh_x, mesh_y = _grid_xy(n)
    u = mesh_x**2 + mesh_y**2
    spec = _poisson_spec()
    object.__setattr__(spec, "_source_array", np.zeros((n, n)))
    field = GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)
    result = ph_bc_002.check(field, spec)
    assert result.status in {"WARN", "FAIL"}, f"expected WARN/FAIL, got {result.status}"
    assert result.raw_value is not None and abs(result.raw_value) > 0.01
    assert result.reason is not None  # non-PASS verdicts carry a reason


def test_ph_bc_002_poisson_no_source_skips_with_reason():
    # Poisson with no source array plumbed: the rule SKIPs (it must not
    # guess f), with a reason that names the two ways to provide a source.
    n = 16
    spec = _poisson_spec()
    field = GridField(np.zeros((n, n)), h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)
    result = ph_bc_002.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason is not None
    assert "source" in result.reason.lower()


def test_ph_bc_002_poisson_source_shape_mismatch_skips():
    # A source array whose shape disagrees with the field => SKIP, not crash.
    n = 32
    spec = _poisson_spec()
    object.__setattr__(spec, "_source_array", np.zeros((8, 8)))
    field = GridField(np.zeros((n, n)), h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)
    result = ph_bc_002.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.reason is not None
    assert "shape" in result.reason.lower()


def test_ph_bc_002_poisson_warn_band():
    # A source offset just off the consistent value lands the imbalance
    # ratio in the WARN band [0.01, 0.1). The consistent source is f = -4;
    # feeding f = -3.96 makes the imbalance ~ 0.04 and the ratio ~ 0.05 =>
    # WARN (neither PASS nor FAIL). Pins the boundary between the tristate
    # thresholds so a threshold regression that keeps the case non-PASS
    # cannot slip through.
    n = 64
    mesh_x, mesh_y = _grid_xy(n)
    u = mesh_x**2 + mesh_y**2
    spec = _poisson_spec()
    object.__setattr__(spec, "_source_array", np.full((n, n), -3.96))
    field = GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)
    result = ph_bc_002.check(field, spec)
    assert result.status == "WARN", f"expected WARN, got {result.status}"
    assert result.violation_ratio is not None
    assert 0.01 <= result.violation_ratio < 0.1


def test_ph_bc_002_poisson_end_to_end_via_loader(tmp_path):
    # End-to-end: an .npz dump carrying an embedded `source` key is loaded
    # by load_target, which plumbs spec._source_array. PH-BC-002's Poisson
    # arm must see that source and emit a real verdict -- this exercises
    # the advertised .npz-source path, not just a hand-injected attribute.
    n = 64
    mesh_x, mesh_y = _grid_xy(n)
    u = mesh_x**2 + mesh_y**2  # consistent with f = -4
    metadata = {
        "pde": "poisson",
        "grid_shape": [n, n],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "field": {"type": "grid", "backend": "fd"},
    }
    path = tmp_path / "poisson_pred.npz"
    np.savez(path, prediction=u, metadata=metadata, source=np.full((n, n), -4.0))
    target = load_target(path, cli_overrides={}, toml_path=None)
    result = ph_bc_002.check(target.field, target.spec)
    assert result.status == "PASS", f"expected PASS, got {result.status} ({result.reason})"
    assert result.raw_value is not None and abs(result.raw_value) < 1e-12
