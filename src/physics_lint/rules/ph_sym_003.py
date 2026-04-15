"""PH-SYM-003: SO(2) Lie derivative equivariance (adapter-only).

**V1 scope:** adapter mode only. Dump mode emits ``SKIPPED`` because
SO(2) Lie derivative requires forward-mode AD on a live model, which
a frozen dumped tensor cannot supply. Also requires a 2D grid centered
at the origin (domain in ``[-L/2, L/2]`` by convention).

Single-generator Lie-derivative-based equivariance error following
Gruver et al. 2023 (ICLR): compute ``d/dtheta`` of ``rho^-1 f(R_theta x)``
at ``theta = 0`` via forward-mode AD (torch.autograd.functional.jvp with
the tangent passed via ``v=``).

For a scalar-valued invariant field the Lie derivative should be
identically zero; nonzero values measure the degree of broken equivariance.
"""

from __future__ import annotations

import torch

from physics_lint.field import CallableField, Field
from physics_lint.report import RuleResult
from physics_lint.rules._symmetry_helpers import is_symmetry_declared
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-SYM-003"
__rule_name__ = "SO(2) Lie derivative equivariance violation"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-SYM-003"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    if not is_symmetry_declared(spec.symmetries, "SO2"):
        return _skip("SO2 not declared in SymmetrySpec")
    if not isinstance(field, CallableField):
        return _skip(
            "SO(2) Lie derivative requires a callable model; dump mode provides "
            "only a frozen tensor"
        )

    model = field._model  # internal — CallableField API exposes it for rules
    grid = field._grid  # shape (..., 2) for 2D
    if grid.shape[-1] != 2:
        return _skip("SO(2) LEE requires a 2D spatial grid")
    if grid.numel() == 0:
        return _skip("SO(2) LEE requires a non-empty sampling grid")

    def rotated_model(theta_param: torch.Tensor) -> torch.Tensor:
        c = torch.cos(theta_param)
        s = torch.sin(theta_param)
        # Rotate coords about origin; for physics_lint convention the grid is
        # assumed centered at origin (domain in [-L/2, L/2]).
        x = grid[..., 0]
        y = grid[..., 1]
        x_rot = c * x - s * y
        y_rot = s * x + c * y
        rotated_grid = torch.stack([x_rot, y_rot], dim=-1)
        return model(rotated_grid).squeeze(-1)

    theta0 = torch.zeros(1)  # jvp is forward-mode; no requires_grad needed
    tangent = torch.ones_like(theta0)

    from torch.autograd.functional import jvp

    _, lie_deriv = jvp(rotated_model, (theta0,), v=(tangent,))

    lie_norm = float(torch.norm(lie_deriv).item() / max(float(lie_deriv.numel()), 1.0) ** 0.5)

    if lie_norm <= 1e-6:
        status = "PASS"
    elif lie_norm <= 0.05:
        status = "WARN"
    else:
        status = "FAIL"

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=lie_norm,
        violation_ratio=lie_norm / 0.05,
        mode=None,
        reason=(
            None
            if status == "PASS"
            else f"SO(2) Lie derivative norm {lie_norm:.2e} exceeds "
            f"{'WARN threshold 1e-6' if status == 'WARN' else 'FAIL threshold 5e-2'}"
        ),
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="per-point L^2 of the scalar Lie derivative",
        citation="Gruver et al. 2023 (ICLR)",
        doc_url=_DOC_URL,
    )


def _skip(reason: str) -> RuleResult:
    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status="SKIPPED",
        raw_value=None,
        violation_ratio=None,
        mode=None,
        reason=reason,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="",
        citation="",
        doc_url=_DOC_URL,
    )
