"""CLI check subcommand tests — file dispatch, formats, exit codes."""

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from physics_lint.cli import app

runner = CliRunner()


def _write_good_dump(path: Path) -> Path:
    N = 32  # noqa: N806  (N is grid resolution; math convention)
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    u = X**2 - Y**2
    np.savez(
        path,
        prediction=u,
        metadata={
            "pde": "laplace",
            "grid_shape": [N, N],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": "dirichlet",
            "field": {"type": "grid", "backend": "fd"},
        },
    )
    return path


def test_cli_check_dump_text_output(tmp_path):
    dump = _write_good_dump(tmp_path / "pred.npz")
    result = runner.invoke(app, ["check", str(dump), "--format", "text"])
    assert result.exit_code == 0, result.stdout
    assert "physics-lint report" in result.stdout
    assert "PH-RES-001" in result.stdout


def test_cli_check_json_format(tmp_path):
    import json

    dump = _write_good_dump(tmp_path / "pred.npz")
    result = runner.invoke(app, ["check", str(dump), "--format", "json"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["pde"] == "laplace"
    assert "rules" in parsed


def test_cli_check_sarif_format(tmp_path):
    import json

    dump = _write_good_dump(tmp_path / "pred.npz")
    result = runner.invoke(
        app,
        ["check", str(dump), "--format", "sarif", "--category", "physics-lint-test"],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["version"] == "2.1.0"
    assert parsed["runs"][0]["automationDetails"]["id"] == "physics-lint-test"


def test_cli_check_output_flag(tmp_path):
    dump = _write_good_dump(tmp_path / "pred.npz")
    out = tmp_path / "out.sarif"
    result = runner.invoke(app, ["check", str(dump), "--format", "sarif", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    text = out.read_text()
    assert "2.1.0" in text


def test_cli_check_pt_file_errors(tmp_path):
    p = tmp_path / "model.pt"
    p.write_bytes(b"\x80\x04")
    result = runner.invoke(app, ["check", str(p)])
    assert result.exit_code == 3
    assert "adapter" in result.stdout.lower() or "adapter" in result.output.lower()


def test_cli_check_disable_rule(tmp_path):
    dump = _write_good_dump(tmp_path / "pred.npz")
    result = runner.invoke(app, ["check", str(dump), "--format", "text", "--disable", "PH-RES-003"])
    assert result.exit_code == 0
    # PH-RES-003 should not appear in the output at all
    assert "PH-RES-003" not in result.stdout
