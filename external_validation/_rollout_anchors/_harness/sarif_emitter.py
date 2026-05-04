"""Shared SARIF emitter for `_rollout_anchors/_harness`.

Both `particle_rollout_adapter.py` and `mesh_rollout_adapter.py` consume
this module to produce SARIF in the same schema as physics-lint's public
emitter. The properties surface is documented in `SCHEMA.md` §3; the
literal-string `"rollout-anchor-harness"` value of `properties.source`
distinguishes harness-emitted results from public-API-emitted results
in downstream tooling.

Day-0 scope: enough surface to be invoked by the controlled-fixture
test (`tests/fixtures/test_harness_vs_public_api.py`). The full
public-API mirroring lands when Day 1 / Day 2 rollouts populate
case-study `outputs/lint.sarif`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# SARIF v2.1.0 schema URI. Matches what physics-lint's public emitter writes.
_SARIF_SCHEMA_URI = "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json"
_SARIF_VERSION = "2.1.0"

# Literal source-tag values. See SCHEMA.md §3.1 for the namespace rationale.
SourceTag = Literal["rollout-anchor-harness", "physics-lint-public-api"]


@dataclass
class HarnessResult:
    """One harness-emitted SARIF result row.

    Mirrors the structure of physics_lint.report.RuleResult only on the
    fields SARIF surfaces; this is deliberately a separate dataclass so
    that the public RuleResult contract is not coupled to the harness.
    """

    rule_id: str
    level: Literal["note", "warning", "error"]
    message: str
    raw_value: float | None
    case_study: str
    dataset: str
    model: str
    ckpt_hash: str
    source: SourceTag = "rollout-anchor-harness"
    harness_validation_passed: bool | None = None
    harness_vs_public_epsilon: float | None = None
    extra_properties: dict[str, Any] = field(default_factory=dict)

    def to_sarif_result(self) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "source": self.source,
            "harness_validation_passed": self.harness_validation_passed,
            "harness_vs_public_epsilon": self.harness_vs_public_epsilon,
            "case_study": self.case_study,
            "dataset": self.dataset,
            "model": self.model,
            "ckpt_hash": self.ckpt_hash,
        }
        if self.raw_value is not None:
            properties["raw_value"] = self.raw_value
        properties.update(self.extra_properties)
        return {
            "ruleId": self.rule_id,
            "level": self.level,
            "message": {"text": self.message},
            "properties": properties,
        }


def emit_sarif(
    results: list[HarnessResult],
    *,
    output_path: Path | str,
    tool_name: str = "physics-lint-rollout-anchor-harness",
    tool_version: str = "0.1.0",
) -> Path:
    """Write `results` to `output_path` in SARIF v2.1.0 format.

    Returns the absolute path written.
    """
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    sarif: dict[str, Any] = {
        "$schema": _SARIF_SCHEMA_URI,
        "version": _SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": tool_name,
                        "version": tool_version,
                        "informationUri": "https://github.com/tyy0811/physics-lint",
                    }
                },
                "results": [r.to_sarif_result() for r in results],
            }
        ],
    }
    out.write_text(json.dumps(sarif, indent=2, sort_keys=True))
    return out
