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

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# SARIF v2.1.0 schema URI. Matches what physics-lint's public emitter writes.
_SARIF_SCHEMA_URI = "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json"
_SARIF_VERSION = "2.1.0"

# Literal source-tag values. See SCHEMA.md §3.1 for the namespace rationale.
SourceTag = Literal["rollout-anchor-harness", "physics-lint-public-api"]

# Repo-root-relative location of the harness package. SARIF
# `artifactLocation.uri` values must be relative to the checkout root for
# GitHub Code Scanning to resolve (display) them; this is the stable on-disk
# location of `_harness/`.
_HARNESS_URI_PREFIX = "external_validation/_rollout_anchors/_harness"

# rule_id -> the harness adapter module that *implements* the check. A
# physics-lint harness finding has no source-line location (it describes
# model behavior on a rollout, not a code defect at a file:line), so its
# physical location points at the committed adapter the finding causally
# originates from. GitHub Code Scanning rejects results with no `locations`
# at all ("locationFromSarifResult: expected at least one location") and only
# *displays* results whose location is a file path, so a logical-only
# location is insufficient on its own — see the round-code-1 absorption
# writeup for the physical-vs-logical rationale.
_RULE_TO_HARNESS_MODULE: dict[str, str] = {
    "PH-SYM-001": "symmetry_rollout_adapter.py",
    "PH-SYM-003": "symmetry_rollout_adapter.py",
    "harness:mass_conservation_defect": "particle_rollout_adapter.py",
    "harness:energy_drift": "particle_rollout_adapter.py",
    "harness:dissipation_sign_violation": "particle_rollout_adapter.py",
}

# Placeholder region attached to every result's physical location: the
# finding is about the whole adapter module, not a specific line. Documented
# as a placeholder in the round-code-1 writeup.
_PLACEHOLDER_REGION: dict[str, int] = {"startLine": 1, "startColumn": 1, "endColumn": 2}


def _harness_module_for_rule(rule_id: str) -> str:
    """Return the basename of the harness adapter module that implements `rule_id`.

    Known rule ids are mapped explicitly; unmapped ids fall back by family
    (``PH-SYM-*`` -> symmetry adapter, ``PH-MESH-*``/``mesh:*`` -> mesh
    adapter, ``PH-CON-*``/``harness:*`` -> particle adapter) and finally to
    `sarif_emitter.py` itself — the one committed file every harness result
    is constructed in — so the physical-location URI is always a real file.
    """
    explicit = _RULE_TO_HARNESS_MODULE.get(rule_id)
    if explicit is not None:
        return explicit
    if rule_id.startswith("PH-SYM-"):
        return "symmetry_rollout_adapter.py"
    if rule_id.startswith(("PH-MESH-", "mesh:", "harness:mesh")):
        return "mesh_rollout_adapter.py"
    if rule_id.startswith(("PH-CON-", "harness:")):
        return "particle_rollout_adapter.py"
    return "sarif_emitter.py"


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

    def _fully_qualified_name(self) -> str:
        """Slash-delimited logical name uniquely identifying this result row.

        ``<case_study>/<model>/<dataset>/<rule_id>`` followed by
        ``/<transform_kind>[_<transform_param>]`` and ``/traj<NN>`` segments
        when those keys are present in `extra_properties` (eps SARIFs carry
        both; rung-4a/4c conservation SARIFs carry only `traj_index`). Empty
        segments (e.g. when a case-study driver leaves model/dataset blank)
        are dropped so the name is always non-empty. This doubles as the
        per-row identity used to derive a stable partialFingerprint.
        """
        parts: list[str] = [self.case_study, self.model, self.dataset, self.rule_id]

        transform_kind = self.extra_properties.get("transform_kind")
        if transform_kind:
            segment = str(transform_kind)
            transform_param = self.extra_properties.get("transform_param")
            if transform_param:
                segment = f"{segment}_{transform_param}"
            parts.append(segment)

        traj_index = self.extra_properties.get("traj_index")
        if traj_index is not None:
            parts.append(f"traj{int(traj_index):02d}")

        parts = [str(p) for p in parts if p]
        if not parts:
            # Defensive: a HarnessResult with no rule_id and no metadata at
            # all. rule_id is required by the dataclass so this is unreachable
            # in practice, but a non-empty name is mandatory either way.
            parts = ["harness-result"]
        return "/".join(parts)

    def _locations(self) -> list[dict[str, Any]]:
        """SARIF `locations` for this result: one physical + one logical.

        The physical location points at the committed harness adapter module
        that implements the rule (with a placeholder region — see
        `_PLACEHOLDER_REGION`) so GitHub Code Scanning can resolve and display
        the result; the logical location carries the detailed per-row
        fully-qualified name. Many results legitimately share the same
        adapter:1 physical location — `partialFingerprints` (set in
        `to_sarif_result`) keeps them distinct in the Security tab.
        """
        module = _harness_module_for_rule(self.rule_id)
        fqn = self._fully_qualified_name()
        leaf = fqn.rsplit("/", 1)[-1]
        return [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": f"{_HARNESS_URI_PREFIX}/{module}"},
                    "region": dict(_PLACEHOLDER_REGION),
                },
                "logicalLocations": [{"name": leaf, "fullyQualifiedName": fqn}],
            }
        ]

    def _partial_fingerprints(self) -> dict[str, str]:
        """Stable per-row fingerprint so results sharing an adapter:1 physical
        location are not collapsed into one Security-tab alert. Keyed in the
        conventional ``<name>/<version>`` form; valued by a hash of the
        fully-qualified name (the per-row identity).
        """
        digest = hashlib.sha256(self._fully_qualified_name().encode("utf-8")).hexdigest()[:16]
        return {"physicsLintResultFqnHash/v1": digest}

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
            "locations": self._locations(),
            "partialFingerprints": self._partial_fingerprints(),
            "properties": properties,
        }


def emit_sarif(
    results: list[HarnessResult],
    *,
    output_path: Path | str,
    tool_name: str = "physics-lint-rollout-anchor-harness",
    tool_version: str = "0.1.0",
    run_properties: dict[str, Any] | None = None,
) -> Path:
    """Write `results` to `output_path` in SARIF v2.1.0 format.

    `run_properties` (D0-19): optional dict written verbatim to
    `runs[0].properties`. When None, runs[0].properties is omitted.
    Callers that emit harness SARIF for rung 4a+ pass the 10 D0-19
    run-level fields here; pre-D0-19 call sites omit and continue to work.

    Returns the absolute path written.
    """
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": tool_name,
                "version": tool_version,
                "informationUri": "https://github.com/tyy0811/physics-lint",
            }
        },
        "results": [r.to_sarif_result() for r in results],
    }
    if run_properties is not None:
        run["properties"] = run_properties

    sarif: dict[str, Any] = {
        "$schema": _SARIF_SCHEMA_URI,
        "version": _SARIF_VERSION,
        "runs": [run],
    }
    out.write_text(json.dumps(sarif, indent=2, sort_keys=True))
    return out
