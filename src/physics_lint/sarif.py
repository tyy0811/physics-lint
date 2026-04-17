"""SARIF 2.1.0 emission for PhysicsLintReport.

Per design doc §13. Key discipline points:
1. Only WARN and FAIL rules emit run.results entries. PASS rules do NOT
   — SARIF results are findings, and GitHub code scanning treats every
   result as an alert regardless of the SARIF `level` field (an error-
   severity rule's PASS would surface as an `error` alert). PASS is
   visible in text/JSON output; in SARIF it is the absence of a result.
2. SKIPPED rules go into run.invocations[0].toolExecutionNotifications
   (level: note), NOT into run.results. Prevents Security-tab noise.
3. category parameter propagates to run.automationDetails.id AND should
   match the workflow's category: input on codeql-action/upload-sarif.
4. Artifact-only is the default location mode; source-mapped triggers when
   report.metadata['sarif_source'] carries a source_file + line info.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from physics_lint.report import PhysicsLintReport, RuleResult


_SCHEMA_URI = "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json"

_SEVERITY_LEVEL = {
    "error": "error",
    "warning": "warning",
    "info": "note",
}


def to_sarif(report: PhysicsLintReport, category: str = "physics-lint") -> dict[str, Any]:
    """Emit SARIF 2.1.0 JSON for a PhysicsLintReport."""
    from physics_lint import __version__

    target_path = report.metadata.get("target_path", "unknown")
    sarif_source = report.metadata.get("sarif_source")
    source_mapped = isinstance(sarif_source, dict) and sarif_source.get("source_file")

    from physics_lint.rules._registry import list_rules as _list_rules

    # No try/except here: registry listing failures should propagate. If
    # the rule registry is broken, we do NOT want to ship SARIF with a
    # silently empty driver.rules metadata block — that's a
    # silent-correctness-failure pattern. Same class as the Codex review
    # finding on the PASS-as-result bug: loud failure beats a false-green
    # signal. (The registry is deterministic; if _list_rules raises, the
    # install is broken and the user needs to know.)
    registry_entries = _list_rules()

    descriptors = [
        {
            "id": entry.rule_id,
            "name": entry.rule_name,
            "shortDescription": {"text": entry.rule_name},
            "defaultConfiguration": {"level": _SEVERITY_LEVEL.get(entry.default_severity, "note")},
            "properties": {
                "input_modes": sorted(entry.input_modes),
            },
        }
        for entry in registry_entries
    ]

    results: list[dict[str, Any]] = []
    notifications: list[dict[str, Any]] = []
    for r in report.rules:
        if r.status == "SKIPPED":
            notifications.append(_skipped_notification(r))
            continue
        if r.status == "PASS":
            # PASS rules do not emit SARIF results — see module docstring.
            # GitHub code scanning treats every result as an alert.
            continue
        results.append(_result_object(r, target_path, sarif_source if source_mapped else None))

    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": "physics-lint",
                "version": __version__,
                "informationUri": "https://physics-lint.readthedocs.io",
                "rules": descriptors,
            }
        },
        "automationDetails": {"id": category},
        "results": results,
        "invocations": [
            {
                "executionSuccessful": report.exit_code == 0,
                "toolExecutionNotifications": notifications,
            }
        ],
    }

    return {
        "version": "2.1.0",
        "$schema": _SCHEMA_URI,
        "runs": [run],
    }


def _result_object(
    r: RuleResult,
    target_path: str,
    sarif_source: dict[str, Any] | None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "violation_ratio": r.violation_ratio,
        "raw_value": r.raw_value,
        "doc_url": r.doc_url,
        "mode": r.mode,
    }
    if sarif_source:
        properties["location_mode"] = "source-mapped"
        properties["model_artifact"] = target_path
        location = {
            "physicalLocation": {
                "artifactLocation": {"uri": sarif_source["source_file"]},
                "region": _build_region(r, sarif_source),
            }
        }
    else:
        properties["location_mode"] = "artifact-only"
        location = {
            "physicalLocation": {
                "artifactLocation": {"uri": target_path},
            }
        }

    return {
        "ruleId": r.rule_id,
        "level": _SEVERITY_LEVEL.get(r.severity, "note"),
        "message": {"text": _message_text(r)},
        "locations": [location],
        "properties": {k: v for k, v in properties.items() if v is not None},
    }


def _build_region(r: RuleResult, sarif_source: dict[str, Any]) -> dict[str, int]:
    """Pick the source line matching the rule category."""
    category = r.rule_id.split("-")[1]
    line_key = {
        "RES": "pde_line",
        "BC": "bc_line",
        "CON": "pde_line",
        "POS": "pde_line",
        "SYM": "symmetry_line",
        "VAR": "pde_line",
        "NUM": "pde_line",
    }.get(category, "pde_line")
    line = sarif_source.get(line_key) or sarif_source.get("pde_line") or 1
    return {"startLine": int(line), "endLine": int(line)}


def _message_text(r: RuleResult) -> str:
    parts = [f"{r.rule_name}"]
    if r.raw_value is not None:
        parts.append(f"raw={r.raw_value:.3e}")
    if r.violation_ratio is not None:
        parts.append(f"ratio={r.violation_ratio:.2f}")
    if r.mode:
        parts.append(f"mode={r.mode}")
    if r.reason:
        parts.append(r.reason)
    return "; ".join(parts)


def _skipped_notification(r: RuleResult) -> dict[str, Any]:
    return {
        "level": "note",
        "message": {"text": f"{r.rule_id} skipped: {r.reason or 'unknown reason'}"},
        "descriptor": {"id": r.rule_id},
    }
