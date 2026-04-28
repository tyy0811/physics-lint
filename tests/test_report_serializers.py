"""Report serializer tests — summary text, to_dict, to_json."""

import json

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


def test_summary_contains_all_statuses_with_glyphs():
    r = PhysicsLintReport(
        pde="heat",
        grid_shape=(64, 64, 32),
        metadata={},
        rules=[
            _rr("PH-RES-001", "PASS", raw_value=1e-6, violation_ratio=2.3),
            _rr("PH-RES-002", "SKIPPED", reason="dump mode", severity="warning"),
            _rr("PH-BC-001", "PASS", raw_value=1e-5, mode="absolute"),
            _rr("PH-POS-001", "FAIL", raw_value=-0.034, violation_ratio=340),
            _rr("PH-CON-001", "FAIL", raw_value=0.0017, violation_ratio=240),
            _rr(
                "PH-CON-003",
                "WARN",
                severity="warning",
                raw_value=0.5,
                violation_ratio=45,
            ),
        ],
    )
    summary = r.summary()
    # Each status glyph appears at least once
    assert "✓" in summary  # PASS
    assert "✗" in summary  # FAIL
    assert "⚠" in summary  # WARN
    assert "⊘" in summary  # SKIPPED
    # Mode tag appears on BC line
    assert "[absolute mode]" in summary
    # Overall header
    assert "FAIL" in summary
    # status_counts line
    assert "2 fail" in summary
    assert "1 warn" in summary
    assert "2 pass" in summary
    assert "1 skip" in summary


def test_to_dict_roundtrips_via_json():
    r = PhysicsLintReport(
        pde="laplace",
        grid_shape=(64, 64),
        metadata={"platform": "darwin"},
        rules=[_rr("PH-RES-001", "PASS", raw_value=1.0, violation_ratio=0.1)],
    )
    d = r.to_dict()
    payload = json.dumps(d)
    parsed = json.loads(payload)
    assert parsed["pde"] == "laplace"
    assert parsed["overall_status"] == "PASS"
    assert parsed["rules"][0]["rule_id"] == "PH-RES-001"
    assert parsed["rules"][0]["raw_value"] == 1.0


def test_to_json_is_valid_json():
    r = PhysicsLintReport(
        pde="heat",
        grid_shape=(64, 64, 32),
        metadata={},
        rules=[_rr("PH-RES-001", "PASS", raw_value=1e-6, violation_ratio=0.5)],
    )
    # Should not raise
    parsed = json.loads(r.to_json())
    assert parsed["pde"] == "heat"
