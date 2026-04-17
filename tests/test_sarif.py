"""SARIF emission tests — tier framing, category, SKIPPED-to-notifications."""

from physics_lint.report import PhysicsLintReport, RuleResult


def _rr(rule_id, status, severity="error", **kw):
    return RuleResult(
        rule_id=rule_id,
        rule_name=f"{rule_id} name",
        severity=severity,
        status=status,
        raw_value=kw.get("raw_value"),
        violation_ratio=kw.get("violation_ratio"),
        mode=kw.get("mode"),
        reason=kw.get("reason"),
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="H-1",
        citation="test",
        doc_url="https://x",
    )


def _base_report(**kw) -> PhysicsLintReport:
    return PhysicsLintReport(
        pde="laplace",
        grid_shape=(64, 64),
        rules=kw.get("rules", [_rr("PH-RES-001", "PASS", raw_value=1e-6)]),
        metadata=kw.get("metadata", {"target_path": "models/unet.pt"}),
    )


def test_sarif_minimal_schema():
    sarif = _base_report().to_sarif(category="physics-lint")
    assert sarif["version"] == "2.1.0"
    assert sarif["$schema"].endswith("2.1.0.json")
    assert len(sarif["runs"]) == 1
    run = sarif["runs"][0]
    assert run["tool"]["driver"]["name"] == "physics-lint"
    assert run["automationDetails"]["id"] == "physics-lint"
    assert isinstance(run["results"], list)


def test_sarif_skipped_goes_to_notifications_not_results():
    rules = [
        _rr("PH-RES-001", "PASS", raw_value=1e-6),
        _rr("PH-RES-002", "SKIPPED", severity="warning", reason="dump mode"),
        _rr("PH-SYM-003", "SKIPPED", severity="warning", reason="dump mode"),
    ]
    sarif = _base_report(rules=rules).to_sarif(category="physics-lint-fno")
    run = sarif["runs"][0]
    result_ids = {r["ruleId"] for r in run["results"]}
    # PASS goes into results; SKIPPED does NOT
    assert "PH-RES-001" in result_ids
    assert "PH-RES-002" not in result_ids
    assert "PH-SYM-003" not in result_ids

    notifications = run["invocations"][0]["toolExecutionNotifications"]
    notif_rule_ids = {n["descriptor"]["id"] for n in notifications}
    assert "PH-RES-002" in notif_rule_ids
    assert "PH-SYM-003" in notif_rule_ids
    for n in notifications:
        assert n["level"] == "note"


def test_sarif_category_propagates():
    sarif = _base_report().to_sarif(category="physics-lint-unet")
    assert sarif["runs"][0]["automationDetails"]["id"] == "physics-lint-unet"


def test_sarif_artifact_only_location_default():
    rules = [_rr("PH-POS-001", "FAIL", raw_value=-0.03, violation_ratio=300)]
    sarif = _base_report(rules=rules, metadata={"target_path": "models/fno.pt"}).to_sarif()
    result = sarif["runs"][0]["results"][0]
    loc = result["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uri"] == "models/fno.pt"
    assert "region" not in loc
    assert result["properties"]["location_mode"] == "artifact-only"


def test_sarif_source_mapped_when_metadata_has_sarif_spec():
    rules = [_rr("PH-POS-001", "FAIL", raw_value=-0.03, violation_ratio=300)]
    metadata = {
        "target_path": "models/fno.pt",
        "sarif_source": {
            "source_file": "train_fno.py",
            "pde_line": 42,
            "bc_line": 58,
        },
    }
    sarif = _base_report(rules=rules, metadata=metadata).to_sarif()
    result = sarif["runs"][0]["results"][0]
    loc = result["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uri"] == "train_fno.py"
    assert loc["region"]["startLine"] == 42
    assert result["properties"]["location_mode"] == "source-mapped"
    assert result["properties"]["model_artifact"] == "models/fno.pt"


def test_sarif_severity_level_mapping():
    rules = [
        _rr("PH-RES-001", "FAIL", severity="error", raw_value=1.0, violation_ratio=1000),
        _rr("PH-SYM-001", "WARN", severity="warning", raw_value=0.1, violation_ratio=50),
        _rr("PH-VAR-001", "WARN", severity="info", raw_value=None),
    ]
    sarif = _base_report(rules=rules).to_sarif()
    levels = {r["ruleId"]: r["level"] for r in sarif["runs"][0]["results"]}
    assert levels["PH-RES-001"] == "error"
    assert levels["PH-SYM-001"] == "warning"
    assert levels["PH-VAR-001"] == "note"
