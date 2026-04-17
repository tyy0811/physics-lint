"""physics-lint rules list/show subcommands."""

from __future__ import annotations

import typer

from physics_lint.rules import _registry

rules_app = typer.Typer(help="Browse the rule catalog.")


@rules_app.command("list")
def list_cmd() -> None:
    """Print rule id, name, severity, input modes (metadata only; <50 ms)."""
    entries = _registry.list_rules()
    if not entries:
        typer.echo("no rules registered")
        return
    width = max(len(e.rule_id) for e in entries)
    for e in entries:
        modes = "+".join(sorted(e.input_modes))
        typer.echo(f"  {e.rule_id:<{width}}  {e.default_severity:<7}  {modes:<16}  {e.rule_name}")


@rules_app.command("show")
def show_cmd(rule_id: str) -> None:
    """Print a rule's docstring, metadata, and doc URL."""
    import importlib

    entries = _registry.list_rules()
    entry = next((e for e in entries if e.rule_id == rule_id), None)
    if entry is None:
        typer.echo(f"unknown rule: {rule_id}", err=True)
        raise typer.Exit(code=2)
    module = importlib.import_module(entry.module_name)
    doc = module.__doc__ or "(no docstring)"
    typer.echo(f"Rule: {entry.rule_id}")
    typer.echo(f"Name: {entry.rule_name}")
    typer.echo(f"Severity: {entry.default_severity}")
    typer.echo(f"Input modes: {sorted(entry.input_modes)}")
    typer.echo("")
    typer.echo(doc)
