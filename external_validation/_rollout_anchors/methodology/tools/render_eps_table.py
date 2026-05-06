"""Rung 4b sibling renderer — schema_version v1.1 only.

Per design §5.1: focused on one schema version. v1.0 SARIFs cannot be
rendered by this tool; the renderer raises SchemaVersionMismatchError
with a clear message. Rung 4a's render_cross_stack_table.py handles
v1.0 unchanged.

Output: tripartite-grouped markdown table (architectural-evidence /
construction-trivial / substrate-incompatible-SKIP) per design §5.2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA_VERSION = "1.1"
EXPECTED_SOURCE_TAG = "rollout-anchor-harness"


class SchemaVersionMismatchError(Exception):
    """Raised when a SARIF's harness_sarif_schema_version differs from
    EXPECTED_SCHEMA_VERSION. Mirrors rung-4a's renderer error class.
    """


class SourceTagMismatchError(Exception):
    """Raised when a SARIF's source tag is not 'rollout-anchor-harness'."""


class MissingRunLevelFieldError(Exception):
    """Raised when expected run-level fields are absent."""


def _classify_evidence(rule_id: str, transform_kind: str) -> str:
    """Map (rule_id, transform_kind) to one of three evidence classes per design §3.2."""
    if transform_kind == "skip":
        return "substrate-incompatible-skip"
    if transform_kind == "identity":
        return "construction-trivial"
    if transform_kind == "translation":
        return "construction-trivial"
    return "architectural"


def _verdict_label(eps: float | None, evidence_class: str) -> str:
    """Per design §3.3 threshold band; SKIP gets its own label."""
    if evidence_class == "substrate-incompatible-skip":
        return "SKIP"
    assert eps is not None
    if eps <= 1e-5:
        return "PASS"
    if eps <= 1e-2:
        return "APPROXIMATE"
    return "FAIL"


def _format_eps(eps: float | None) -> str:
    if eps is None:
        return "—"
    if eps == 0.0:
        return "0.0e+00"
    return f"{eps:.3e}"


def _read_sarif(path: Path) -> dict[str, Any]:
    sarif = json.loads(Path(path).read_text())
    run_props = sarif["runs"][0].get("properties", {})

    for required in ("source", "harness_sarif_schema_version"):
        if required not in run_props:
            raise MissingRunLevelFieldError(
                f"{path}: missing required run-level field {required!r}."
            )

    if run_props["source"] != EXPECTED_SOURCE_TAG:
        raise SourceTagMismatchError(
            f"{path}: expected source={EXPECTED_SOURCE_TAG!r}, got {run_props['source']!r}."
        )

    if run_props["harness_sarif_schema_version"] != EXPECTED_SCHEMA_VERSION:
        raise SchemaVersionMismatchError(
            f"render_eps_table: harness_sarif_schema_version mismatch in {path}: "
            f"expected {EXPECTED_SCHEMA_VERSION}, got {run_props['harness_sarif_schema_version']}"
        )
    return sarif


def render_eps_table(*, segnn_sarif_path: Path, gns_sarif_path: Path) -> str:
    """Read both stacks' v1.1 SARIFs, render tripartite-grouped markdown table.

    Returns the markdown content as a string. The caller decides where
    to write it (typically into the rung 4b table writeup at
    `methodology/docs/2026-05-05-rung-4b-equivariance-table.md`).
    """
    segnn_sarif = _read_sarif(Path(segnn_sarif_path))
    gns_sarif = _read_sarif(Path(gns_sarif_path))

    rows: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "architectural": [],
        "construction-trivial": [],
        "substrate-incompatible-skip": [],
    }
    for model, sarif in (("segnn", segnn_sarif), ("gns", gns_sarif)):
        for result in sarif["runs"][0].get("results", []):
            props = result["properties"]
            evidence_class = _classify_evidence(result["ruleId"], props["transform_kind"])
            rows[evidence_class].append((model, result))

    lines: list[str] = []

    def _emit_class_table(title: str, class_key: str) -> None:
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Rule | Stack | transform_param | traj_index | eps | Verdict |")
        lines.append("|---|---|---|---|---|---|")
        for model, result in sorted(
            rows[class_key],
            key=lambda mr: (
                mr[1]["ruleId"],
                mr[1]["properties"]["transform_param"],
                mr[0],
                mr[1]["properties"]["traj_index"],
            ),
        ):
            props = result["properties"]
            eps = props.get("eps_pos_rms")
            verdict = _verdict_label(eps, class_key)
            extra = ""
            if class_key == "substrate-incompatible-skip":
                extra = f" — {props['skip_reason']}"
            lines.append(
                f"| {result['ruleId']} | {model.upper()} | "
                f"{props['transform_param']} | {props['traj_index']} | "
                f"{_format_eps(eps)} | {verdict}{extra} |"
            )
        lines.append("")

    _emit_class_table("Architectural-evidence rows", "architectural")
    _emit_class_table("Construction-trivial rows", "construction-trivial")
    _emit_class_table("Substrate-incompatible SKIP", "substrate-incompatible-skip")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: render to stdout."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--segnn-sarif", type=Path, required=True)
    parser.add_argument("--gns-sarif", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        render_eps_table(
            segnn_sarif_path=args.segnn_sarif,
            gns_sarif_path=args.gns_sarif,
        )
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
