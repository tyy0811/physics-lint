"""Render the cross-stack conservation table from harness SARIF artifacts.

Per DECISIONS.md D0-20 + the rung-4a design at
`methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-design.md`:

Reads N harness SARIFs (one per stack), asserts schema_version +
source-tag + run-level field presence, aggregates per-traj rows per
(rule, stack) -- detecting "all N identical" specially -- and emits a
markdown table to stdout.

Generator-vs-consumer separation: this module imports nothing from
`_harness/` or `01-lagrangebench/`. The SARIF schema is the wire
protocol; harness_sarif_schema_version is asserted on read.

INVOKE FROM REPO ROOT:

    python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \
        --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/

The rendered table is what the rung-4a writeup includes via copy-paste,
plus the rederivability footer that records the exact command + sha.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Pinned by D0-19. Bump when SCHEMA.md §3.x bumps.
EXPECTED_SCHEMA_VERSION = "1.0"
EXPECTED_SOURCE_TAG = "rollout-anchor-harness"

# Required run-level fields per D0-19 §3.1.
REQUIRED_RUN_LEVEL_FIELDS: tuple[str, ...] = (
    "source",
    "harness_sarif_schema_version",
    "physics_lint_sha_pkl_inference",
    "physics_lint_sha_npz_conversion",
    "physics_lint_sha_sarif_emission",
    "lagrangebench_sha",
    "checkpoint_id",
    "model_name",
    "dataset_name",
    "rollout_subdir",
)


class SchemaVersionMismatchError(Exception):
    """Raised when a SARIF's harness_sarif_schema_version doesn't match
    EXPECTED_SCHEMA_VERSION. The renderer's contract is bound to the
    expected version; mismatch means the renderer might silently emit a
    wrong table on a schema-bumped artifact. Fail loud.
    """


class SourceTagMismatchError(Exception):
    """Raised when a SARIF's source-tag is not 'rollout-anchor-harness'.
    Distinguishes harness SARIF from public-API SARIF reaching the
    renderer by accident.
    """


class MissingRunLevelFieldError(Exception):
    """Raised when a SARIF is missing one or more of the 10 required
    D0-19 run-level fields. No defaulting.
    """


def _assert_run_level(sarif: dict[str, Any], src_path: Path) -> dict[str, Any]:
    """Apply the three D0-20 fail-loud assertions on a SARIF.

    Returns the run-level properties dict.
    """
    runs = sarif.get("runs", [])
    if not runs:
        raise MissingRunLevelFieldError(
            f"{src_path}: SARIF has no runs[]; D0-19 requires runs[0] with properties."
        )
    properties = runs[0].get("properties", {})

    missing = [f for f in REQUIRED_RUN_LEVEL_FIELDS if f not in properties]
    if missing:
        raise MissingRunLevelFieldError(
            f"{src_path}: missing required D0-19 run-level fields: {missing}. See SCHEMA.md §3.x."
        )

    if properties["source"] != EXPECTED_SOURCE_TAG:
        raise SourceTagMismatchError(
            f"{src_path}: expected source={EXPECTED_SOURCE_TAG!r}, got {properties['source']!r}."
        )

    if properties["harness_sarif_schema_version"] != EXPECTED_SCHEMA_VERSION:
        raise SchemaVersionMismatchError(
            f"{src_path}: expected harness_sarif_schema_version="
            f"{EXPECTED_SCHEMA_VERSION!r}, got {properties['harness_sarif_schema_version']!r}. "
            f"See SCHEMA.md §3.x."
        )

    return properties


def render_cross_stack_table(sarif_paths: Iterable[Path | str]) -> str:
    """Read each SARIF in sarif_paths, assert D0-19 contract, aggregate,
    return a markdown table string.
    """
    paths = [Path(p) for p in sarif_paths]
    if not paths:
        raise MissingRunLevelFieldError("render_cross_stack_table: no SARIF paths provided.")

    stacks: list[tuple[Path, dict[str, Any], list[dict[str, Any]]]] = []
    for path in paths:
        sarif = json.loads(path.read_text())
        run_props = _assert_run_level(sarif, path)
        results = sarif["runs"][0].get("results", [])
        stacks.append((path, run_props, results))

    rule_ids = (
        "harness:mass_conservation_defect",
        "harness:energy_drift",
        "harness:dissipation_sign_violation",
    )

    cells: dict[tuple[str, str], str] = {}
    stack_labels: list[str] = []
    sha_lines: list[str] = []
    for _path, run_props, results in stacks:
        stack_label = f"{run_props['model_name']}-{run_props['dataset_name']}"
        stack_labels.append(stack_label)
        sha_lines.append(
            f"- **{stack_label}**: pkl_inference={run_props['physics_lint_sha_pkl_inference']}, "
            f"npz_conversion={run_props['physics_lint_sha_npz_conversion']}, "
            f"sarif_emission={run_props['physics_lint_sha_sarif_emission']}"
        )
        for rule_id in rule_ids:
            rule_rows = [r for r in results if r["ruleId"] == rule_id]
            if not rule_rows:
                cells[(rule_id, stack_label)] = "(no rows)"
                continue
            n = len(rule_rows)
            raw_values = [r["properties"].get("raw_value") for r in rule_rows]
            skip_present = [
                r["properties"].get("skip_reason") is not None
                or r["properties"].get("raw_value") is None
                for r in rule_rows
            ]
            if all(rv is None for rv in raw_values) and all(skip_present):
                cells[(rule_id, stack_label)] = f"SKIP (x{n}, D0-18)"
            elif all(rv is not None for rv in raw_values):
                vals = [float(rv) for rv in raw_values]
                if all(abs(v - vals[0]) < 1e-15 for v in vals):
                    cells[(rule_id, stack_label)] = f"{vals[0]:.3e} (x{n} identical)"
                else:
                    cells[(rule_id, stack_label)] = (
                        f"min={min(vals):.3e}, max={max(vals):.3e}, n={n}"
                    )
            else:
                cells[(rule_id, stack_label)] = f"MIXED (n={n})"

    header = ["Rule", *stack_labels]
    rows: list[list[str]] = [header]
    for rule_id in rule_ids:
        short = rule_id.replace("harness:", "")
        row = [f"`{short}`"]
        for label in stack_labels:
            row.append(cells.get((rule_id, label), "(missing)"))
        rows.append(row)

    md_lines = [
        "| " + " | ".join(rows[0]) + " |",
        "|" + "|".join(["---"] * len(rows[0])) + "|",
    ]
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")

    md_lines.append("")
    md_lines.append("**Provenance (D0-19 three-sha):**")
    md_lines.append("")
    md_lines.extend(sha_lines)

    return "\n".join(md_lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sarif-dir",
        type=Path,
        required=True,
        help="Directory containing the harness SARIF files (e.g., outputs/sarif/).",
    )
    args = parser.parse_args(argv)
    sarif_paths = sorted(args.sarif_dir.glob("*.sarif"))
    if not sarif_paths:
        print(f"No .sarif files found in {args.sarif_dir}", file=sys.stderr)
        return 2
    print(render_cross_stack_table(sarif_paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
