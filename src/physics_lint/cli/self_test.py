"""self-test subcommand stub. Task 3 implements the real logic."""

from __future__ import annotations

import typer


def self_test_cmd() -> None:
    """Run physics-lint's analytical self-test battery."""
    typer.echo("self-test not yet implemented (Task 3)")
    raise typer.Exit(code=1)
