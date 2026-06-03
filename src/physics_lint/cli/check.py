"""physics-lint check subcommand."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import typer

from physics_lint.loader import LoadedTarget, LoaderError, load_target
from physics_lint.report import PhysicsLintReport, RuleResult
from physics_lint.rules import _registry


def _extra_required_params(check_fn) -> list[str]:
    """Return required keyword-only parameters of check_fn beyond (field, spec).

    Used to skip rules that need kwargs we can't provide from the CLI
    (boundary_target, boundary_values, refined_field) without swallowing
    TypeErrors raised from inside the rule body.
    """
    try:
        sig = inspect.signature(check_fn)
    except (TypeError, ValueError):
        return []
    extras: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("field", "spec"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty:
            extras.append(name)
    return extras


def _autoextract_kwargs(
    extras: list[str], loaded: LoadedTarget
) -> tuple[dict[str, Any], list[str]]:
    """Fill rule kwargs the CLI knows how to derive, return (filled, remaining).

    V1.0 scope covers two kwargs:
    - `boundary_target` (PH-BC-001): from `loaded.boundary_target` if the
      dump shipped it; else zeros on the boundary when BC is
      `dirichlet_homogeneous`; else remains in `remaining`.
    - `boundary_values` (PH-POS-002): same sources as `boundary_target`.

    `refined_field` and any other kwargs stay in `remaining` — they need
    loader/adapter contract extensions tracked in docs/backlog/v1.2.md.
    """
    filled: dict[str, Any] = {}
    remaining: list[str] = []
    bc_kind = loaded.spec.boundary_condition.kind
    for name in extras:
        if name in ("boundary_target", "boundary_values"):
            if loaded.boundary_target is not None:
                filled[name] = loaded.boundary_target
            elif bc_kind == "dirichlet_homogeneous":
                filled[name] = np.zeros_like(loaded.field.values_on_boundary())
            else:
                remaining.append(name)
        else:
            remaining.append(name)
    return filled, remaining


def _skipped_for_missing_kwargs(entry, extras: list[str]) -> RuleResult:
    """Emit a SKIPPED RuleResult for a rule the CLI can't invoke.

    Visible in text summary (⊘ glyph), JSON dump, and SARIF
    toolExecutionNotifications. Prevents the silent-correctness-failure
    pattern where a user runs `physics-lint check model.pt`, sees green,
    and doesn't realize 3/N rules never fired.

    V1 limitation; V1.1 auto-extraction tracked in docs/backlog/v1.2.md.
    """
    joined = ", ".join(extras)
    return RuleResult(
        rule_id=entry.rule_id,
        rule_name=entry.rule_name,
        severity=entry.default_severity,
        status="SKIPPED",
        raw_value=None,
        violation_ratio=None,
        mode=None,
        reason=f"requires {joined} (CLI V1 limitation; V1.1 auto-extracts)",
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="",
        citation="",
        doc_url="",
    )


def _field_type_applies(entry, field_type: str) -> bool:
    return field_type in entry.field_types


def _skipped_for_field_type(entry, field_type: str) -> RuleResult:
    """Emit a SKIPPED RuleResult for a rule that does not accept this field type.

    The central applicability filter runs before check() so a grid rule never
    reaches a MeshVectorField (where ensure_grid_field raises TypeError) and a
    mesh_vector rule never reaches a grid field. The reason names the substrate
    mismatch so the SKIP list reads as a substrate-applicability map.
    """
    return RuleResult(
        rule_id=entry.rule_id,
        rule_name=entry.rule_name,
        severity=entry.default_severity,
        status="SKIPPED",
        raw_value=None,
        violation_ratio=None,
        mode=None,
        reason=(
            f"{entry.rule_id} does not apply to {field_type} fields "
            f"(accepts: {', '.join(sorted(entry.field_types))})"
        ),
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="",
        citation="",
        doc_url="",
    )


def _run_rules_for_test(loaded: LoadedTarget, disable: set[str] | None = None) -> list[RuleResult]:
    """Run all applicable rules against a loaded target; return their results.

    Shared by check_cmd and the CLI tests so the end-to-end dispatch (the
    central field-type filter, then kwarg auto-extraction, then check()) is
    exercised by both. The field-type filter is the FIRST gate: it SKIPs a rule
    whose declared __field_types__ does not include the target's field type,
    closing the latent crash surface where a grid rule's ensure_grid_field
    raises TypeError on a MeshVectorField. check() is still called without a
    try/except, so rule-internal TypeErrors propagate (not silently swallowed).
    """
    disabled = disable or set()
    entries = [e for e in _registry.list_rules() if e.rule_id not in disabled]
    field_type = loaded.spec.field.type
    results: list[RuleResult] = []
    for entry in entries:
        if not _field_type_applies(entry, field_type):
            results.append(_skipped_for_field_type(entry, field_type))
            continue
        check_fn = _registry.load_check(entry)
        extras = _extra_required_params(check_fn)
        auto_kwargs: dict[str, Any] = {}
        if extras:
            auto_kwargs, remaining = _autoextract_kwargs(extras, loaded)
            if remaining:
                results.append(_skipped_for_missing_kwargs(entry, remaining))
                continue
        result = check_fn(loaded.field, loaded.spec, **auto_kwargs)
        if result is not None:
            results.append(result)
    return results


def check_cmd(
    target: Path = typer.Argument(..., help="Adapter .py or dump .npz/.npy"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to pyproject.toml"),
    format: str = typer.Option("text", "--format", help="text | json | sarif"),
    category: str = typer.Option("physics-lint", "--category", help="SARIF automationDetails.id"),
    output: Optional[Path] = typer.Option(None, "--output", help="Write output to file"),
    disable: list[str] = typer.Option([], "--disable", help="Disable a rule by ID"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Run physics-lint rules against a target model artifact."""
    try:
        loaded = load_target(target, cli_overrides={}, toml_path=config)
    except LoaderError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=3) from e

    # Dispatch is centralized in _run_rules_for_test (also the CLI test seam):
    # field-type applicability filter first, then kwarg auto-extraction, then
    # check(). The verbose echo surfaces each SKIP (type-mismatch or missing
    # kwargs) on stderr without changing the report payload.
    results = _run_rules_for_test(loaded, disable=set(disable))
    if verbose:
        for r in results:
            if r.status == "SKIPPED":
                typer.echo(f"  (skipped {r.rule_id}: {r.reason})", err=True)

    metadata: dict[str, object] = {"target_path": str(target)}
    # Plumb [tool.physics-lint.sarif] into SARIF metadata so source-mapped
    # emission activates from the CLI.
    if loaded.spec.sarif is not None and loaded.spec.sarif.source_file:
        metadata["sarif_source"] = {
            "source_file": loaded.spec.sarif.source_file,
            "pde_line": loaded.spec.sarif.pde_line,
            "bc_line": loaded.spec.sarif.bc_line,
            "symmetry_line": loaded.spec.sarif.symmetry_line,
        }

    report = PhysicsLintReport(
        pde=loaded.spec.pde,
        grid_shape=loaded.spec.grid_shape,
        rules=results,
        metadata=metadata,
    )

    if format == "text":
        payload = report.summary()
    elif format == "json":
        payload = report.to_json()
    elif format == "sarif":
        payload = json.dumps(report.to_sarif(category=category), indent=2)
    else:
        typer.echo(f"unknown format: {format}", err=True)
        raise typer.Exit(code=2)

    if output is not None:
        output.write_text(payload)
    else:
        typer.echo(payload)

    raise typer.Exit(code=report.exit_code)
