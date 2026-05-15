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

INVOKE FROM REPO ROOT (rung-4a LB-only — backwards compatible):

    python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \
        --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/

INVOKE FOR THE UNIFIED CS01 + CS02 CROSS-STACK TABLE (Phase 3):

    python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \
        --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/ \
        --include-glob '*_tgv2d_8e49339469.sarif' \
        --sarif-dir external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/

The rendered table is what the rung-4a / Phase-3 writeup includes via
copy-paste, plus the rederivability footer that records the exact
command + sha. Multiple ``--sarif-dir`` values accumulate; the
``--include-glob`` filter applies positionally to the corresponding
``--sarif-dir`` (with ``*.sarif`` filling any unspecified positions).
Control-arm SARIFs (``arm == 'gt-control'``; CS02-onwards convention)
are filtered out so they do NOT become cross-stack columns.
"""

from __future__ import annotations

import argparse
import json
import re
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


class DuplicateStackLabelError(Exception):
    """Raised when two input SARIFs map to the same ``model_name-dataset_name``.

    Pre-round-codex-3, the renderer accepted duplicate stack labels and
    silently overwrote cells under one key while emitting duplicate
    columns in the header. This is a realistic workflow failure because
    ``emit_sarif.py`` writes sha-named SARIFs without removing older
    ones, so a re-emission leaves both old and new SARIFs in the same
    directory; ``--include-glob '*dam2d*.sarif'`` then picks up both.
    Fail-loud rather than produce an ambiguous table.

    Callers should either (a) clean up stale SARIFs before rendering,
    (b) use a sha-specific glob like ``--include-glob '*<sha>*.sarif'``,
    or (c) pass exact file paths.
    """


class ResultRowInvariantError(Exception):
    """Raised when D0-19 §3.4's guaranteed-identical-across-rows
    invariant is violated for a (rule, stack) group: a SKIP row missing
    `properties.skip_reason`, divergent `skip_reason` strings within the
    group, divergent `message.text`, or a mix of SKIP and raw rows
    within a single (rule, stack) (HarnessDefect's "one or the other"
    invariant). Fail loud — the writeup's "20 identical fires" claim
    binds on this.
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
        # Filter out per-case-study control-arm SARIFs (CS02-onwards
        # convention). Control-arm SARIFs (`arm == 'gt-control'`) carry
        # ground-truth-vs-rule-floor evidence for the case study itself;
        # they are NOT cross-stack columns. Model-under-test SARIFs
        # (`arm == 'mgn-rollout'`) and LB-side legacy SARIFs (no `arm`
        # field; pre-CS02 convention) flow through to the cross-stack
        # table. Phase 3 Task 2 plan literal was `arm == 'control'`;
        # the actual CS02 emission uses `'gt-control'` (and
        # `'mgn-rollout'` for the model arm), so the filter binds on
        # that concrete value.
        arm = run_props.get("arm")
        if arm == "gt-control":
            continue
        results = sarif["runs"][0].get("results", [])
        stacks.append((path, run_props, results))

    # Phase-3-round-codex-2 (second-pass artifact-mode review) Finding 1:
    # if every input SARIF was filtered as a control arm (e.g., caller
    # passed only gt.sarif via a glob that matched only the control), the
    # rest of the renderer would produce a syntactically valid but empty
    # cross-stack table with only a `Rule` header column. Fail-loud
    # instead — the renderer's whole contract is "N upstream-rollout
    # stacks under D0-19/D0-20 enforcement"; an empty stack set after
    # filtering is a no-input case, parallel to the up-front
    # `MissingRunLevelFieldError` for an empty `paths` list.
    if not stacks:
        raise MissingRunLevelFieldError(
            "render_cross_stack_table: all input SARIFs were filtered as "
            "control-arm (arm == 'gt-control'); no model-under-test SARIF "
            "remains to render. Pass at least one model-arm or arm-absent "
            "SARIF (LB-side legacy or CS02-onwards mgn-rollout)."
        )

    rule_ids = (
        "harness:mass_conservation_defect",
        "harness:energy_drift",
        "harness:dissipation_sign_violation",
    )

    # Round-codex-3 finding 2: detect duplicate (model_name, dataset_name)
    # stack labels before building the table. Two SARIFs for the same
    # stack (e.g., post-re-emission with old SARIFs still in the dir)
    # would silently produce duplicate header columns and overwriting
    # cells. Fail-loud with the specific file paths so the caller can
    # disambiguate via a tighter --include-glob or explicit paths.
    seen_labels: dict[str, Path] = {}
    for path, run_props, _results in stacks:
        label = f"{run_props['model_name']}-{run_props['dataset_name']}"
        if label in seen_labels:
            raise DuplicateStackLabelError(
                f"Two SARIFs map to the same stack label {label!r}: "
                f"{seen_labels[label].name} and {path.name}. The renderer "
                "cannot produce a coherent table with duplicate stack "
                "columns. Disambiguate by passing exact file paths or by "
                "tightening --include-glob to a sha-specific pattern such "
                f"as '*{label.replace('-', '_')}_<sarif_emission_sha>.sarif' "
                "(round-codex-3 finding 2)."
            )
        seen_labels[label] = path

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

            all_skip = all(rv is None for rv in raw_values)
            all_raw = all(rv is not None for rv in raw_values)
            if not (all_skip or all_raw):
                # HarnessDefect emits one of value or skip_reason consistently
                # per rule per row; mixing within a (rule, stack) violates D0-19 §3.4.
                raise ResultRowInvariantError(
                    f"{rule_id} on {stack_label}: mixed SKIP and raw rows "
                    f"within a single (rule, stack). D0-19 §3.4 requires "
                    f"HarnessDefect's one-of-value-or-skip_reason invariant."
                )

            if all_skip:
                # D0-19 §3.4: SKIP rows MUST carry properties.skip_reason and
                # the value MUST be identical across rows. message.text MUST
                # also be identical (it's a co-variate of skip_reason).
                skip_reasons = [r["properties"].get("skip_reason") for r in rule_rows]
                if any(sr is None for sr in skip_reasons):
                    raise ResultRowInvariantError(
                        f"{rule_id} on {stack_label}: SKIP row(s) missing "
                        f"properties.skip_reason. D0-19 §3.4 requires it on "
                        f"every SKIP row. See SCHEMA.md §3.x."
                    )
                if len(set(skip_reasons)) != 1:
                    raise ResultRowInvariantError(
                        f"{rule_id} on {stack_label}: skip_reason divergence "
                        f"across rows. D0-19 §3.4 requires guaranteed-identical "
                        f"skip_reason within (rule, stack); got {len(set(skip_reasons))} "
                        f"distinct values."
                    )
                messages = [r.get("message", {}).get("text") for r in rule_rows]
                if len(set(messages)) != 1:
                    raise ResultRowInvariantError(
                        f"{rule_id} on {stack_label}: message.text divergence "
                        f"across SKIP rows. D0-19 §3.4 requires guaranteed-"
                        f"identical message.text co-varying with skip_reason."
                    )
                # Extract the cited D-entry from the skip_reason for the cell
                # label. Pre-rung-4c the only SKIP path was D0-18 so the label
                # was hardcoded; post-rung-4c skip_reasons cite D0-08 (KE-rest),
                # D0-18 (dissipative-monotone), D0-22 (open-driven on
                # dissipation_sign_violation), or D0-22 (amendment 1) (open-
                # driven on energy_drift). Regex requires the "DECISIONS.md "
                # prefix so it doesn't pick up incidental D0-NN mentions
                # (e.g., "D0-19 §3.4" referencing the schema invariant).
                d_entry_match = re.search(
                    r"DECISIONS\.md\s+(D0-\d+(?:\s+\(amendment\s+\d+\))?)",
                    skip_reasons[0],
                )
                d_entry = d_entry_match.group(1) if d_entry_match else "?"
                cells[(rule_id, stack_label)] = f"SKIP (x{n}, {d_entry})"
            else:
                # all_raw: per-row raw_value may legitimately vary across trajs
                # (e.g., conservation defect differs by initial conditions).
                # Renderer surfaces uniformity vs distribution; does not raise.
                vals = [float(rv) for rv in raw_values]
                if all(abs(v - vals[0]) < 1e-15 for v in vals):
                    cells[(rule_id, stack_label)] = f"{vals[0]:.3e} (x{n} identical)"
                else:
                    cells[(rule_id, stack_label)] = (
                        f"min={min(vals):.3e}, max={max(vals):.3e}, n={n}"
                    )

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

    # Optional inference-run-status section (rung-4c §9 review-gate fold-in).
    # Each stack's SARIF may carry an optional `inference_run_status`
    # property in run-level properties (one of `from_completed_inference`,
    # `from_aborted_inference`, `from_unknown_inference`); when at least
    # one stack carries it, render the section with explicit values.
    # Stacks without the field (rung-4a/4b legacy SARIFs, pre-fold-in)
    # render as `n/a (pre-salvage-tag-schema)` — honest absence rather
    # than assumed-clean default, parallel to the gate's refuse-by-default
    # posture. If ALL stacks lack the field, the section is omitted
    # entirely so legacy outputs render byte-identically to pre-fold-in.
    status_entries: list[tuple[str, str | None]] = []
    for _path, run_props, _results in stacks:
        stack_label = f"{run_props['model_name']}-{run_props['dataset_name']}"
        status_entries.append((stack_label, run_props.get("inference_run_status")))
    if any(status is not None for _label, status in status_entries):
        md_lines.append("")
        md_lines.append("**Inference run status (rung-4c §9 review-gate fold-in):**")
        md_lines.append("")
        for stack_label, status in status_entries:
            display = status if status is not None else "n/a (pre-salvage-tag-schema)"
            md_lines.append(f"- **{stack_label}**: {display}")

    return "\n".join(md_lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sarif-dir",
        type=Path,
        action="append",
        required=True,
        help=(
            "Directory containing the harness SARIF files (e.g., outputs/sarif/). "
            "May be specified multiple times to ingest from multiple case-study "
            "directories in a single invocation (Phase 3 unified CS01+CS02 cross-stack "
            "table). Each --sarif-dir pairs positionally with the corresponding "
            "--include-glob; unpaired positions default to '*.sarif'."
        ),
    )
    parser.add_argument(
        "--include-glob",
        type=str,
        action="append",
        default=None,
        help=(
            "Glob pattern (relative to --sarif-dir) for files to include. Defaults to "
            "'*.sarif'. Use to render a subset of SARIFs in a directory that mixes "
            "schema versions (e.g., '--include-glob \"*tgv2d*.sarif\"' to skip "
            "rung-4b eps SARIFs at v1.1, or '--include-glob \"*dam2d*.sarif\"' for "
            "rung-4c dam-break-only). The fail-loud schema-mismatch assertion still "
            "applies to the filtered set. With multiple --sarif-dir values, each "
            "--include-glob pairs positionally with the corresponding --sarif-dir."
        ),
    )
    args = parser.parse_args(argv)
    dirs: list[Path] = list(args.sarif_dir)
    globs: list[str] = list(args.include_glob) if args.include_glob else []
    if len(globs) > len(dirs):
        print(
            f"--include-glob specified {len(globs)} times but --sarif-dir only "
            f"{len(dirs)} times; cannot pair globs positionally.",
            file=sys.stderr,
        )
        return 2
    # Pad unpaired positions with the default glob.
    globs.extend(["*.sarif"] * (len(dirs) - len(globs)))

    sarif_paths: list[Path] = []
    for d, glob in zip(dirs, globs, strict=True):
        matched = sorted(d.glob(glob))
        if not matched:
            print(
                f"No SARIF files matching glob {glob!r} in {d}",
                file=sys.stderr,
            )
            return 2
        sarif_paths.extend(matched)
    print(render_cross_stack_table(sarif_paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
