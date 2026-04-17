"""physics-lint config init/show subcommands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

config_app = typer.Typer(help="Scaffold and inspect configuration.")


_GENERIC_SKELETON = """\
[tool.physics-lint]
pde = "laplace"                    # "laplace" | "poisson" | "heat" | "wave"
grid_shape = [64, 64]              # [Nx, Ny] or [Nx, Ny, Nz] or [Nx, Ny, Nt]
domain = { x = [0.0, 1.0], y = [0.0, 1.0] }
periodic = false
boundary_condition = "dirichlet"   # "periodic" | "dirichlet" | "dirichlet_homogeneous" | "neumann" | "neumann_homogeneous"
# diffusivity = 0.01               # required for heat
# wave_speed = 1.0                 # required for wave
symmetries = []                    # e.g. ["D4", "translation_x"]

[tool.physics-lint.field]
type = "grid"                      # "grid" | "callable" | "mesh"
backend = "auto"                   # "fd" | "spectral" | "auto"

[tool.physics-lint.rules]
# "PH-BC-001" = { abs_threshold = 1e-10, abs_tol_fail = 1e-3 }
# "PH-SYM-003" = { enabled = false }

# [tool.physics-lint.sarif]
# source_file = "train_model.py"
# pde_line = 42
# bc_line = 58
"""


_HEAT_SKELETON = """\
[tool.physics-lint]
pde = "heat"
grid_shape = [64, 64, 32]
domain = { x = [0.0, 1.0], y = [0.0, 1.0], t = [0.0, 1.0] }
periodic = false
boundary_condition = "dirichlet_homogeneous"
diffusivity = 0.01
symmetries = ["D4"]

[tool.physics-lint.field]
type = "grid"
backend = "fd"
"""


_WAVE_SKELETON = """\
[tool.physics-lint]
pde = "wave"
grid_shape = [64, 64, 32]
domain = { x = [0.0, 1.0], y = [0.0, 1.0], t = [0.0, 1.0] }
periodic = false
boundary_condition = "dirichlet_homogeneous"
wave_speed = 1.0
symmetries = ["D4"]

[tool.physics-lint.field]
type = "grid"
backend = "fd"
"""


@config_app.command("init")
def init_cmd(
    pde: Optional[str] = typer.Option(None, "--pde", help="heat | wave | generic"),
) -> None:
    if pde is None or pde == "generic":
        typer.echo(_GENERIC_SKELETON)
    elif pde == "heat":
        typer.echo(_HEAT_SKELETON)
    elif pde == "wave":
        typer.echo(_WAVE_SKELETON)
    else:
        typer.echo(_GENERIC_SKELETON)


@config_app.command("show")
def show_cmd(
    config: Path = typer.Option(..., "--config", help="Path to pyproject.toml"),
) -> None:
    """Read, merge, and validate config WITHOUT loading a target (partial view)."""
    from physics_lint.config import load_spec_from_toml, merge_into_spec
    from physics_lint.spec import DomainSpec

    try:
        raw = load_spec_from_toml(config)
        merged = merge_into_spec(raw, adapter_spec=None, cli_overrides={})
        merged.setdefault("field", {}).setdefault("dump_path", "(placeholder)")
        spec = DomainSpec.model_validate(merged)
    except Exception as e:
        typer.echo(f"config invalid: {e}", err=True)
        raise typer.Exit(code=2) from e

    typer.echo("(adapter domain_spec() not applied; partial view)")
    typer.echo("")
    typer.echo(spec.model_dump_json(indent=2))
