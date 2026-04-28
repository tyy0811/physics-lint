"""CLI tests for rules and config subcommands."""

from pathlib import Path

from typer.testing import CliRunner

from physics_lint.cli import app

runner = CliRunner()


def test_cli_rules_list_contains_week1_rules():
    result = runner.invoke(app, ["rules", "list"])
    assert result.exit_code == 0
    assert "PH-RES-001" in result.stdout
    assert "PH-BC-001" in result.stdout
    assert "PH-POS-001" in result.stdout


def test_cli_rules_show_returns_docstring():
    result = runner.invoke(app, ["rules", "show", "PH-RES-001"])
    assert result.exit_code == 0
    assert "residual" in result.stdout.lower()


def test_cli_rules_show_unknown_id_errors():
    result = runner.invoke(app, ["rules", "show", "PH-NONEXISTENT-999"])
    assert result.exit_code == 2


def test_cli_config_init_generic():
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 0
    assert "[tool.physics-lint]" in result.stdout
    assert "pde =" in result.stdout


def test_cli_config_init_pde_heat_uncomments_diffusivity():
    result = runner.invoke(app, ["config", "init", "--pde", "heat"])
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    for line in lines:
        if "diffusivity" in line and not line.strip().startswith("#"):
            break
    else:
        raise AssertionError("diffusivity not uncommented in heat template")


def test_cli_self_test_delegates_to_package_module(monkeypatch):
    """Regression: self-test used to resolve scripts/smoke_self_test.py by
    walking out of the package tree, which fails from an installed wheel
    because scripts/ is not packaged. The CLI must delegate to
    physics_lint.selftest.run (a package module)."""
    from physics_lint.selftest import run as real_run

    call_log: list[bool] = []

    def _fake_run(*, verbose: bool = False) -> tuple[int, str]:
        call_log.append(verbose)
        return 0, "PASS\n"

    monkeypatch.setattr("physics_lint.selftest.run", _fake_run)
    result = runner.invoke(app, ["self-test"])
    assert result.exit_code == 0
    assert "PASS" in result.stdout
    assert call_log == [False]
    # Sanity: restore and confirm the real entry point is importable
    assert callable(real_run)


def test_cli_config_show_without_target(tmp_path: Path):
    cfg = tmp_path / "pyproject.toml"
    cfg.write_text("""
[tool.physics-lint]
pde = "laplace"
grid_shape = [64, 64]
domain = { x = [0.0, 1.0], y = [0.0, 1.0] }
periodic = false
boundary_condition = "dirichlet"

[tool.physics-lint.field]
type = "grid"
backend = "fd"
dump_path = "pred.npz"
""")
    result = runner.invoke(app, ["config", "show", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "laplace" in result.stdout
    assert "adapter domain_spec() not applied" in result.stdout


def test_cli_config_show_missing_dump_path_uses_unset_sentinel(tmp_path: Path):
    """Regression: when user TOML omits dump_path, config show used to inject
    '(placeholder)' into the model dump — a string that could be mistaken
    for a real config value. Now displays '<unset>' with an advisory."""
    cfg = tmp_path / "pyproject.toml"
    cfg.write_text("""
[tool.physics-lint]
pde = "laplace"
grid_shape = [64, 64]
domain = { x = [0.0, 1.0], y = [0.0, 1.0] }
periodic = false
boundary_condition = "dirichlet"

[tool.physics-lint.field]
type = "grid"
backend = "fd"
""")
    result = runner.invoke(app, ["config", "show", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "<unset>" in result.stdout
    assert "(placeholder)" not in result.stdout
    assert '"<unset>"' in result.stdout or "unset" in result.stdout.lower()
