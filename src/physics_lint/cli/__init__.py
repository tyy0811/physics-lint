"""physics-lint CLI entry point.

Typer app with four subcommands: check, self-test, rules, config.
Each subcommand lives in its own module; this file stitches them
together.
"""

from __future__ import annotations

import typer

from physics_lint.cli import check as check_mod
from physics_lint.cli import config_cmd as config_mod
from physics_lint.cli import rules as rules_mod
from physics_lint.cli import self_test as self_test_mod

app = typer.Typer(
    name="physics-lint",
    help="Linter for trained neural PDE surrogates.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("check")(check_mod.check_cmd)
app.command("self-test")(self_test_mod.self_test_cmd)
app.add_typer(rules_mod.rules_app, name="rules")
app.add_typer(config_mod.config_app, name="config")


if __name__ == "__main__":
    app()
