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


def test_cli_check_sarif_config_produces_source_mapped(tmp_path):
    """[tool.physics-lint.sarif] must reach SARIF output as source-mapped."""
    import json

    dump = _write_good_dump(tmp_path / "pred.npz")
    cfg = tmp_path / "pyproject.toml"
    cfg.write_text("""
[tool.physics-lint]
pde = "laplace"
grid_shape = [32, 32]
domain = { x = [0.0, 1.0], y = [0.0, 1.0] }
periodic = false
boundary_condition = "dirichlet"

[tool.physics-lint.field]
type = "grid"
backend = "fd"

[tool.physics-lint.sarif]
source_file = "train_model.py"
pde_line = 42
bc_line = 58
""")
    result = runner.invoke(
        app,
        [
            "check",
            str(dump),
            "--config",
            str(cfg),
            "--format",
            "sarif",
            "--category",
            "physics-lint-cli-src-mapped",
        ],
    )
    # Exit may be 0 or 1 depending on rule outcomes on this synthetic field;
    # the contract is that any emitted result (if any) is source-mapped.
    parsed = json.loads(result.stdout)
    for res in parsed["runs"][0]["results"]:
        loc = res["locations"][0]["physicalLocation"]
        assert loc["artifactLocation"]["uri"] == "train_model.py"
        assert res["properties"]["location_mode"] == "source-mapped"


def test_cli_check_does_not_swallow_rule_internal_type_error(tmp_path, monkeypatch):
    """Regression: catching TypeError blanket-hid rule-internal bugs and
    produced false-green exits. Signature-based skip must let internal
    TypeErrors propagate (the CLI reports it as a crash, not a silent skip)."""
    from physics_lint.rules import _registry

    real_load_check = _registry.load_check

    def _fake_load_check(entry):
        if entry.rule_id != "PH-RES-001":
            return real_load_check(entry)

        def broken_check(field, spec):
            # Simulate a rule-internal bug: an int.__add__ with a str.
            # Raises TypeError inside the rule body, unrelated to missing kwargs.
            return 1 + "two"  # type: ignore[operator]

        return broken_check

    monkeypatch.setattr(_registry, "load_check", _fake_load_check)

    dump = _write_good_dump(tmp_path / "pred.npz")
    result = runner.invoke(app, ["check", str(dump), "--format", "text"])
    # The rule must not be silently skipped. Either the CLI crashes
    # (exception bubbled up — exit_code != 0, exception stored on result),
    # or it reports an error path. What must NOT happen: exit 0 with
    # PH-RES-001 absent from the report.
    silent_skip = result.exit_code == 0 and "PH-RES-001" not in result.stdout
    assert not silent_skip, (
        f"Rule-internal TypeError was silently swallowed. "
        f"exit={result.exit_code}, stdout preview: {result.stdout[:300]!r}"
    )
