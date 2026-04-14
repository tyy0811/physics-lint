"""Report schema tests — RuleResult + PhysicsLintReport with SKIPPED handling."""

from physics_lint.report import PhysicsLintReport, RuleResult


def _rr(rule_id: str, status: str, severity: str = "error", **kw) -> RuleResult:
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
        recommended_norm="H^-1",
        citation="Week 1 plan Task 9",
        doc_url="https://physics-lint.readthedocs.io/rules/",
    )


def test_overall_status_all_pass():
    r = PhysicsLintReport(
        pde="laplace",
        grid_shape=(64, 64),
        metadata={},
        rules=[_rr("PH-RES-001", "PASS"), _rr("PH-BC-001", "PASS")],
    )
    assert r.overall_status == "PASS"
    assert r.exit_code == 0


def test_overall_status_with_skipped_is_not_warn():
    r = PhysicsLintReport(
        pde="laplace",
        grid_shape=(64, 64),
        metadata={},
        rules=[
            _rr("PH-RES-001", "PASS"),
            _rr("PH-SYM-003", "SKIPPED", reason="dump mode"),
            _rr("PH-BC-001", "PASS"),
        ],
    )
    assert r.overall_status == "PASS"  # SKIPPED rank == PASS rank
    assert r.exit_code == 0
    counts = r.status_counts
    assert counts == {"PASS": 2, "WARN": 0, "FAIL": 0, "SKIPPED": 1}


def test_overall_status_warn_beats_pass():
    r = PhysicsLintReport(
        pde="laplace",
        grid_shape=(64, 64),
        metadata={},
        rules=[_rr("PH-RES-001", "PASS"), _rr("PH-SYM-001", "WARN", severity="warning")],
    )
    assert r.overall_status == "WARN"
    assert r.exit_code == 0  # WARN does not trigger non-zero exit


def test_overall_status_fail_beats_warn():
    r = PhysicsLintReport(
        pde="laplace",
        grid_shape=(64, 64),
        metadata={},
        rules=[
            _rr("PH-RES-001", "PASS"),
            _rr("PH-SYM-001", "WARN", severity="warning"),
            _rr("PH-POS-001", "FAIL"),
        ],
    )
    assert r.overall_status == "FAIL"
    assert r.exit_code == 1


def test_fail_of_warning_severity_does_not_trigger_exit_code():
    r = PhysicsLintReport(
        pde="laplace",
        grid_shape=(64, 64),
        metadata={},
        rules=[_rr("PH-SYM-001", "FAIL", severity="warning")],
    )
    assert r.overall_status == "FAIL"
    assert r.exit_code == 0


def test_empty_report_is_pass():
    r = PhysicsLintReport(pde="laplace", grid_shape=(64, 64), metadata={}, rules=[])
    assert r.overall_status == "PASS"
    assert r.exit_code == 0
    assert r.status_counts == {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIPPED": 0}


def test_error_severity_fail_triggers_exit_code():
    r = PhysicsLintReport(
        pde="laplace",
        grid_shape=(64, 64),
        metadata={},
        rules=[_rr("PH-RES-001", "FAIL", severity="error")],
    )
    assert r.overall_status == "FAIL"
    assert r.exit_code == 1
