"""physics-lint CLI entry point.

Typer app. Task 2 ships the `check` subcommand. self-test / rules / config
live as stubs until Task 3 fills them in; they are registered so typer
routes subcommands by name rather than collapsing the single command to root.
"""

from __future__ import annotations

import typer

from physics_lint.cli import check as check_mod
from physics_lint.cli import self_test as self_test_mod

app = typer.Typer(
    name="physics-lint",
    help="Linter for trained neural PDE surrogates.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("check")(check_mod.check_cmd)
app.command("self-test")(self_test_mod.self_test_cmd)


if __name__ == "__main__":
    app()
