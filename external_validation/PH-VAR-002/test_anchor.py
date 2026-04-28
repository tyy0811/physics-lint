"""PH-VAR-002 external-validation anchor - hyperbolic norm-equivalence info-flag
diagnostic (F3-absent-by-structure).

Task 13 of the complete-v1.0 plan. Verifies the rule's diagnostic contract:
PASS + info severity + literature-pointer reason on wave PDE; SKIPPED with
per-PDE reason on every other PDE kind; PhysicsLintReport aggregation
invariant (info-severity PASS does not move exit_code).

External validation separates:

    F1   Mathematical-legitimacy anchor: multi-paper DPG + variational-
         correctness stack (Bachmayr-Dahmen-Oster 2024, Ernst-Sprungk-
         Tamellini 2025, Gopalakrishnan-Sepulveda 2019, Ernesti-Wieners
         2019, Henning-Palitta-Simoncini-Urban 2022, Demkowicz-
         Gopalakrishnan 2010/2011). Structural argument identifies that
         the parabolic variational-correctness framework physics-lint's
         shipped rules rely on does not extend to hyperbolic problems
         without specialized DPG machinery; V1's wave-equation residual
         norms are therefore a conjectural extension.

    F2   Correctness-fixture: diagnostic-contract verification. PASS +
         info severity + literature-pointer reason on wave;
         SKIPPED + per-PDE reason on laplace / poisson / heat;
         PhysicsLintReport aggregation invariant checked explicitly.

    F3   Borrowed-credibility: absent by structure. Info-flag rules emit
         no numerical quantity against the field; nothing to reproduce
         against a published baseline. Per complete-v1.0 plan section
         1.2, F3-absent is structural for this anchor class.

    Supplementary: Demkowicz-Gopalakrishnan 2025 Acta Numerica DOI
         10.1017/S0962492924000102 as theoretical framing only.

Wording discipline (CITATION.md + README + this file):
    "PH-VAR-002 emits an info-severity diagnostic on wave-equation
    problems pointing users to DPG hyperbolic norm-equivalence
    literature; it does not certify hyperbolic norm-equivalence and
    does not compute against the field."
"""

from __future__ import annotations

import pytest

from physics_lint import DomainSpec
from physics_lint.field import GridField
from physics_lint.report import PhysicsLintReport
from physics_lint.rules import ph_var_002


def _wave_spec() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "wave",
            "grid_shape": [8, 8, 4],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "wave_speed": 1.0,
            "field": {"type": "grid", "dump_path": "p.npz"},
        }
    )


def _parabolic_spec(pde: str) -> DomainSpec:
    payload: dict = {
        "pde": pde,
        "grid_shape": [8, 8, 4] if pde == "heat" else [8, 8],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "field": {"type": "grid", "dump_path": "p.npz"},
    }
    if pde == "heat":
        payload["domain"]["t"] = [0.0, 1.0]
        payload["diffusivity"] = 1.0
    return DomainSpec.model_validate(payload)


def _dummy_field(spec: DomainSpec) -> GridField:
    import numpy as np

    shape = spec.grid_shape
    return GridField(np.zeros(shape, dtype=np.float64), h=0.1, periodic=False)


# =========================================================================
# F2: diagnostic-contract verification on wave (PASS + info + reason)
# =========================================================================


def test_wave_pde_returns_pass_info_with_literature_pointer():
    """Wave PDE -> PASS status, info severity, literature-pointer reason."""
    spec = _wave_spec()
    field = _dummy_field(spec)
    result = ph_var_002.check(field, spec)
    assert result.status == "PASS"
    assert result.severity == "info"
    assert result.reason is not None
    # Literature-pointer fragments (rule's _MESSAGE).
    assert "Bachmayr-Ernst variational framework" in result.reason
    assert "diagnostic" in result.reason
    # Rule must not claim to compute a numerical quantity.
    assert result.raw_value is None
    assert result.violation_ratio is None
    # Recommended-norm field carries the conjectural flag.
    assert "conjectural" in result.recommended_norm.lower()


# =========================================================================
# F2: SKIPPED path on parabolic / elliptic PDEs
# =========================================================================


@pytest.mark.parametrize("pde", ["laplace", "poisson", "heat"])
def test_non_wave_pdes_are_skipped_with_per_pde_reason(pde):
    """Non-wave PDE -> SKIPPED with reason naming the current PDE."""
    spec = _parabolic_spec(pde)
    field = _dummy_field(spec)
    result = ph_var_002.check(field, spec)
    assert result.status == "SKIPPED"
    assert result.severity == "info"
    assert result.reason is not None
    assert "wave only" in result.reason
    assert pde in result.reason
    assert result.raw_value is None


# =========================================================================
# F2: PhysicsLintReport aggregation invariant
# =========================================================================


def test_info_severity_pass_does_not_move_report_exit_code():
    """Because the rule is info-severity, its PASS must not move
    PhysicsLintReport.exit_code. This guards against a future commit
    that flips the severity to warning or error — which would alter
    the aggregate report status on every wave-equation invocation."""
    spec = _wave_spec()
    field = _dummy_field(spec)
    result = ph_var_002.check(field, spec)
    report = PhysicsLintReport(pde=spec.pde, grid_shape=spec.grid_shape, rules=[result])
    assert report.exit_code == 0, (
        f"info-severity PASS must yield exit_code=0, got {report.exit_code} "
        f"(severity={result.severity!r}, status={result.status!r})"
    )


# =========================================================================
# Wording-discipline guard: CITATION.md and README avoid overclaim
# =========================================================================


def test_citation_md_does_not_claim_certification():
    """CITATION.md must not describe PH-VAR-002 as certifying hyperbolic
    norm-equivalence; only affirmative appearances of the forbidden
    phrase would be in the 'Avoid:' block (at most one occurrence)."""
    from pathlib import Path

    citation = Path(__file__).parent / "CITATION.md"
    text = citation.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    # Required scope language.
    assert "does not certify hyperbolic norm-equivalence" in normalized
    assert "emits an info-severity diagnostic" in normalized
    # Forbidden overclaim phrasings — allowed at most once each (inside
    # the "Avoid:" block).
    for forbidden in (
        "validates hyperbolic norm-equivalence",
        "certifies wave-equation residuals",
        "reproduces Demkowicz-Gopalakrishnan 2025",
    ):
        occurrences = text.count(forbidden)
        assert occurrences <= 1, (
            f"forbidden overclaim phrase `{forbidden}` appears {occurrences} "
            f"times in CITATION.md; expected at most 1 (inside 'Avoid:' block)"
        )


def test_citation_md_f3_absent_by_structure_is_documented():
    """F3 absent-by-structure must be explicitly stated in the Borrowed-
    credibility subsection so the three-function-structure check passes
    AND downstream readers see the rationale."""
    from pathlib import Path

    citation = Path(__file__).parent / "CITATION.md"
    text = citation.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    # Both the structural marker and the rationale must appear.
    assert "F3 absent by structure" in normalized or "absent by structure" in normalized
    assert "no numerical quantity" in normalized or ("emit no numerical quantity" in normalized)
