"""physics-lint check subcommand."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Optional

import typer

from physics_lint.loader import LoaderError, load_target
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


def _skipped_for_missing_kwargs(entry, extras: list[str]) -> RuleResult:
    """Emit a SKIPPED RuleResult for a rule the CLI can't invoke.

    Visible in text summary (⊘ glyph), JSON dump, and SARIF
    toolExecutionNotifications. Prevents the silent-correctness-failure
    pattern where a user runs `physics-lint check model.pt`, sees green,
    and doesn't realize 3/N rules never fired.

    V1 limitation; V1.1 auto-extraction tracked in docs/backlog/v1.1.md.
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

    disabled = set(disable)
    entries = [e for e in _registry.list_rules() if e.rule_id not in disabled]
    results: list[RuleResult] = []
    for entry in entries:
        check_fn = _registry.load_check(entry)
        extras = _extra_required_params(check_fn)
        if extras:
            # Known V1 limitation: rules taking extra required kwargs
            # (boundary_target, boundary_values, refined_field) are skipped.
            # V1.1 adds auto-extraction. Detect by signature rather than
            # catching TypeError — that way rule-internal TypeErrors still
            # propagate and fail loudly. Emit a SKIPPED RuleResult so the
            # skip is visible in the summary, not silently absent.
            results.append(_skipped_for_missing_kwargs(entry, extras))
            if verbose:
                typer.echo(f"  (skipping {entry.rule_id}: needs kwargs {extras})", err=True)
            continue
        result = check_fn(loaded.field, loaded.spec)
        if result is not None:
            results.append(result)

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
