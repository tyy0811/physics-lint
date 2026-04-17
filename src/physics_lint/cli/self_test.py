"""physics-lint self-test subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer


def self_test_cmd(
    verbose: bool = typer.Option(False, "--verbose"),
    rule: Optional[str] = typer.Option(None, "--rule", help="Run a single rule"),
    write_report: Optional[Path] = typer.Option(None, "--write-report"),
) -> None:
    """Run the analytical battery against the full rule set.

    Release criterion 1: exit 0 iff every rule hits its calibrated floor
    within tolerance on every analytical input. Delegates to
    physics_lint.selftest which lives in the installable package so
    this subcommand works from a wheel install as well as a repo clone.
    """
    from physics_lint.selftest import run

    code, text = run(verbose=verbose)
    typer.echo(text, nl=False)
    if write_report is not None:
        write_report.write_text(text)
    raise typer.Exit(code=code)
