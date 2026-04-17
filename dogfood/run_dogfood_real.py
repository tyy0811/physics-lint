"""Real-model dogfood for physics-lint Criterion 3 (A1 scope).

See docs/superpowers/specs/2026-04-17-week-2.5-dogfood-a1-design.md.
"""

from physics_lint import DomainSpec
from physics_lint.spec import BCSpec, FieldSourceSpec, GridDomain, SymmetrySpec


def build_a1_spec() -> DomainSpec:
    """DomainSpec for the Week 2½ A1 configuration.

    64x64 unit square, Laplace, non-homogeneous Dirichlet BCs.
    """
    return DomainSpec(
        pde="laplace",
        grid_shape=(64, 64),
        domain=GridDomain(x=(0.0, 1.0), y=(0.0, 1.0)),
        periodic=False,
        boundary_condition=BCSpec(kind="dirichlet"),
        symmetries=SymmetrySpec(declared=[]),
        field=FieldSourceSpec(type="grid", backend="fd", dump_path="unused"),
    )
