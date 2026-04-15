"""PH-NUM-002: Refinement convergence rate below expected.

Per design doc §7.5. Given a predicted homogeneous-Laplace field at
two resolutions (``field`` at grid h and ``refined_field`` at grid
h/2), the rule measures how fast the L^2 norm of the strong-form
residual -Delta_h u shrinks under refinement. If the user's
prediction is an approximate Laplace solution, the residual is
dominated by the FD operator truncation error and converges at a
resolution-dependent rate:

- spectral (periodic): saturates at machine precision almost
  immediately, so the measured rate is effectively infinite.
- fd4 interior-dominated (periodic): ~4 per doubling.
- fd4 boundary-dominated (non-periodic): ~2-2.5 per doubling because
  the outer band of stencils is 2nd-order.

The shipped PASS threshold is 1.8 per doubling — below 2.0 to leave
margin for the non-periodic boundary regime, so a smooth Laplace
prediction on any backend/BC combination passes. Non-converging
predictions (rate << 2) land in WARN; the rule's severity is
'warning' by default so PR gates are not blocked on a single
convergence rate signal.

**V1 scope: homogeneous Laplace only.** PH-NUM-002 uses
``field.laplacian()`` directly as the residual, which is only correct
for ``spec.pde == "laplace"``:

- Poisson would need the source term subtracted — ``||Delta_h u||``
  alone measures the source magnitude, not the residual error.
- Heat/wave have time-dependent residual constructions with spatial
  slice Laplacians and time derivatives (see PH-RES-001), and a
  GridField-wide ``laplacian()`` would differentiate the time axis.

Both cases are explicitly SKIPPED with a reason string. Extending the
rule to delegate residual computation to PH-RES-001 per PDE is
straightforward but needs refined-source / refined-initial-condition
plumbing on the ``refined_field`` contract; that lands in a future
task.

For the refined field we intentionally keep a narrow contract — the
caller passes a GridField at denser spacing — because adapter-mode
CallableFields do not carry a natural notion of a 'refined' grid
without the caller stating what grid to sample on. Adapter mode is
still accepted for the coarse ``field`` via ensure_grid_field().
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field, GridField
from physics_lint.norms import l2_grid
from physics_lint.report import RuleResult
from physics_lint.rules._helpers import ensure_grid_field
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-NUM-002"
__rule_name__ = "Refinement convergence rate below expected"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-NUM-002"
_CITATION = "Fornberg 1988; Trefethen 2000"

# Minimum acceptable refinement rate per doubling for a homogeneous
# Laplace prediction. 1.8 is the non-periodic boundary-dominated fd4
# floor — our measured 2.5 on a harmonic exp(x)cos(y) test minus 0.7
# margin. Periodic interior-dominated fd4 sits near 4 and spectral
# saturates at machine precision, so the same 1.8 threshold passes
# every well-behaved Laplace refinement pair regardless of backend or
# BC. Poisson/heat/wave are SKIPPED at the top of check(); adding a
# per-PDE expected rate is future work.
_DEFAULT_EXPECTED_RATE = 1.8


def check(
    field: Field,
    spec: DomainSpec,
    *,
    refined_field: GridField | None = None,
) -> RuleResult:
    if spec.pde != "laplace":
        return _skipped(
            f"PH-NUM-002 V1 scope is homogeneous Laplace only; got pde={spec.pde!r}. "
            f"Poisson/heat/wave residual construction requires refined source or "
            f"initial-condition plumbing and is deferred to a future task."
        )
    if refined_field is None:
        return _skipped(
            "PH-NUM-002 needs a refined_field (same physical domain, "
            "denser grid) to estimate the convergence rate"
        )
    if not isinstance(refined_field, GridField):
        raise TypeError(
            f"PH-NUM-002 refined_field must be a GridField; got {type(refined_field).__name__}"
        )

    field = ensure_grid_field(field, spec)

    lap_coarse = field.laplacian().values()
    lap_fine = refined_field.laplacian().values()
    r_coarse = l2_grid(lap_coarse, field.h)
    r_fine = l2_grid(lap_fine, refined_field.h)

    if r_fine <= 0.0:
        # Fine residual already at machine-precision floor; treat as
        # super-converged. Covers both the constant/polynomial degenerate
        # case and the usual spectral saturation path.
        rate = float("inf")
    elif r_coarse <= 0.0:
        rate = 0.0
    else:
        rate = float(np.log2(r_coarse / r_fine))

    if rate >= _DEFAULT_EXPECTED_RATE:
        status = "PASS"
        reason = None
    else:
        status = "WARN"
        reason = (
            f"refinement rate {rate:.2f} < expected {_DEFAULT_EXPECTED_RATE} "
            f"(backend={field.backend}, periodic={spec.periodic})"
        )

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=rate,
        violation_ratio=None,
        mode=None,
        reason=reason,
        refinement_rate=rate,
        spatial_map=None,
        recommended_norm="log2 ratio of L^2 residual between coarse and refined grids",
        citation=_CITATION,
        doc_url=_DOC_URL,
    )


def _skipped(reason: str) -> RuleResult:
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
        citation=_CITATION,
        doc_url=_DOC_URL,
    )
