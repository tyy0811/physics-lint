"""physics-lint check subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from physics_lint.loader import LoaderError, load_target
from physics_lint.report import PhysicsLintReport, RuleResult
from physics_lint.rules import _registry


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
        try:
            result = check_fn(loaded.field, loaded.spec)
        except TypeError:
            if verbose:
                typer.echo(f"  (skipping {entry.rule_id}: needs extra kwargs)", err=True)
            continue
        if result is not None:
            results.append(result)

    report = PhysicsLintReport(
        pde=loaded.spec.pde,
        grid_shape=loaded.spec.grid_shape,
        rules=results,
        metadata={"target_path": str(target)},
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
