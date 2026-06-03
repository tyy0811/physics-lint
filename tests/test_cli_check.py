"""CLI check subcommand tests — file dispatch, formats, exit codes."""

from pathlib import Path

import numpy as np
import pytest
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


def _write_dirichlet_homogeneous_dump(path: Path) -> Path:
    """Dump with dirichlet_homogeneous BC; prediction has non-zero boundary
    so PH-BC-001 auto-extracts zeros and fires FAIL."""
    N = 32  # noqa: N806
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    # sin(pi*x)sin(pi*y) vanishes on the boundary (dirichlet_homogeneous).
    # Add a bump to make the boundary non-zero so PH-BC-001 FAILs.
    target = np.sin(np.pi * X) * np.sin(np.pi * Y)
    bump = 0.02 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.3**2))
    pred = target + bump
    np.savez(
        path,
        prediction=pred,
        metadata={
            "pde": "laplace",
            "grid_shape": [N, N],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": "dirichlet_homogeneous",
            "field": {"type": "grid", "backend": "fd"},
        },
    )
    return path


def test_cli_check_dirichlet_homogeneous_auto_extracts_boundary_target(tmp_path):
    """Regression: on a dirichlet_homogeneous dump, PH-BC-001 must auto-
    extract boundary_target = zeros and RUN (not SKIP). The Task 4 workflow
    + README hero depend on this path — without it, SARIF never contains a
    PH-BC-001 result and the Security-tab screenshot cannot be captured."""
    import json

    dump = _write_dirichlet_homogeneous_dump(tmp_path / "dh_pred.npz")
    result = runner.invoke(app, ["check", str(dump), "--format", "json"])
    parsed = json.loads(result.stdout)
    rule_map = {r["rule_id"]: r for r in parsed["rules"]}
    assert "PH-BC-001" in rule_map
    # PH-BC-001 must NOT be SKIPPED on a dirichlet_homogeneous dump
    assert rule_map["PH-BC-001"]["status"] != "SKIPPED", (
        f"PH-BC-001 skipped unexpectedly: {rule_map['PH-BC-001']}"
    )
    # The bump leaks ~3.6e-3 onto the boundary; PH-BC-001 absolute mode
    # FAILs against a 1e-3 default threshold.
    assert rule_map["PH-BC-001"]["status"] == "FAIL"


def test_cli_check_boundary_target_from_dump(tmp_path):
    """Regression: dumps that ship a `boundary_target` array (Task 4's
    make_ci_dumps.py convention) get it plumbed through the loader and
    used by PH-BC-001 auto-extraction."""
    import json

    N = 32  # noqa: N806
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, _Y = np.meshgrid(x, y, indexing="ij")  # noqa: N806
    # Prediction matches a linear BC (boundary value = X on the boundary)
    pred = X.copy()
    # boundary_target is the TRUE BC (also X on the boundary) — so PH-BC-001
    # should PASS because the prediction matches the true BC.
    from physics_lint.field import GridField

    temp_field = GridField(pred, h=(1 / (N - 1), 1 / (N - 1)), periodic=False)
    true_boundary = temp_field.values_on_boundary()
    path = tmp_path / "shipped_bt.npz"
    np.savez(
        path,
        prediction=pred,
        boundary_target=true_boundary,
        metadata={
            "pde": "laplace",
            "grid_shape": [N, N],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": "dirichlet",  # generic, not homogeneous
            "field": {"type": "grid", "backend": "fd"},
        },
    )
    result = runner.invoke(app, ["check", str(path), "--format", "json"])
    parsed = json.loads(result.stdout)
    rule_map = {r["rule_id"]: r for r in parsed["rules"]}
    # Even for non-homogeneous dirichlet, PH-BC-001 runs because the dump
    # shipped boundary_target.
    assert rule_map["PH-BC-001"]["status"] != "SKIPPED"


def test_cli_check_signature_skipped_rules_surface_in_summary(tmp_path):
    """Rules that need extra kwargs (PH-BC-001 boundary_target, PH-POS-002
    boundary_values, PH-NUM-002 refined_field) must emit a SKIPPED RuleResult
    visible in the text summary — not silently disappear. Guards the
    silent-correctness-failure pattern on the CLI dispatch."""
    dump = _write_good_dump(tmp_path / "pred.npz")
    result = runner.invoke(app, ["check", str(dump), "--format", "text"])
    # At least one of the three known-limitation rules is present in the
    # registry; the one whose kwargs cannot be derived should appear with
    # the ⊘ SKIPPED glyph and a reason mentioning "requires".
    assert "⊘" in result.stdout, result.stdout
    assert "PH-BC-001" in result.stdout
    # The reason text must explain WHY (what kwarg is missing), not just
    # that the rule didn't run.
    assert "requires" in result.stdout.lower()


def test_cli_check_signature_skipped_rules_in_json(tmp_path):
    """JSON output must also surface signature-skipped rules."""
    import json

    dump = _write_good_dump(tmp_path / "pred.npz")
    result = runner.invoke(app, ["check", str(dump), "--format", "json"])
    parsed = json.loads(result.stdout)
    skipped_ids = {r["rule_id"] for r in parsed["rules"] if r["status"] == "SKIPPED"}
    # PH-BC-001 requires boundary_target — always in the skip set from CLI
    assert "PH-BC-001" in skipped_ids
    for r in parsed["rules"]:
        if r["rule_id"] == "PH-BC-001":
            assert r["status"] == "SKIPPED"
            assert r["reason"] is not None
            assert "requires" in r["reason"].lower()
            break


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


def test_mesh_vector_target_yields_one_active_eighteen_skips(tmp_path):
    """End-to-end: a mesh_vector dump yields exactly 1 active rule (PH-CON-005)
    + 18 type-specific SKIPs through the real dispatch."""
    pytest.importorskip("skfem")
    from physics_lint.cli.check import _run_rules_for_test
    from physics_lint.loader import load_target

    np.savez(
        tmp_path / "r.npz",
        node_positions=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        cells=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        velocity=np.zeros((1, 4, 2), dtype=np.float32),
        metadata=np.array({"pde": "incompressible_ns"}, dtype=object),
    )
    loaded = load_target(tmp_path / "r.npz", cli_overrides={}, toml_path=None)
    results = _run_rules_for_test(loaded)
    active = [r for r in results if r.status != "SKIPPED"]
    skipped = [r for r in results if r.status == "SKIPPED"]
    assert [r.rule_id for r in active] == ["PH-CON-005"]
    assert len(skipped) == 18
    assert all("mesh_vector" in r.reason for r in skipped)  # type-specific


def _write_mesh_vector_cli_dump(path):
    np.savez(
        path,
        node_positions=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        cells=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        velocity=np.zeros((1, 4, 2), dtype=np.float32),
        metadata=np.array({"pde": "incompressible_ns"}, dtype=object),
    )


def test_mesh_vector_target_renders_all_formats_without_crash(tmp_path):
    """End-to-end via the CLI: a mesh_vector dump (grid_shape=None) must render
    text/json/sarif without crashing. Regression: report.summary()/to_dict()
    iterated self.grid_shape, which is None for a mesh_vector target."""
    pytest.importorskip("skfem")
    import json as _json

    dump = tmp_path / "r.npz"
    _write_mesh_vector_cli_dump(dump)
    for fmt in ("text", "json", "sarif"):
        result = runner.invoke(app, ["check", str(dump), "--format", fmt])
        assert result.exit_code == 0, (
            f"{fmt}: exit={result.exit_code} exc={result.exception!r} "
            f"stdout={result.stdout[:300]!r}"
        )
        assert "PH-CON-005" in result.stdout, fmt
    result = runner.invoke(app, ["check", str(dump), "--format", "json"])
    parsed = _json.loads(result.stdout)
    assert parsed["grid_shape"] is None  # mesh_vector target carries no grid shape
    assert "PH-CON-005" in {r["rule_id"] for r in parsed["rules"]}
