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
    within tolerance on every analytical input.
    """
    import subprocess
    import sys

    script = Path(__file__).parent.parent.parent.parent / "scripts" / "smoke_self_test.py"
    if not script.is_file():
        typer.echo(f"self-test script not found: {script}", err=True)
        raise typer.Exit(code=2)
    cmd = [sys.executable, str(script)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    typer.echo(result.stdout)
    if result.stderr:
        typer.echo(result.stderr, err=True)
    if write_report is not None:
        write_report.write_text(result.stdout)
    raise typer.Exit(code=result.returncode)
