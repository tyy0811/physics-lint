"""Shared rule helpers: _tristate and _load_floor."""

import pytest

from physics_lint.rules._helpers import _load_floor, _tristate


def test_tristate_pass():
    assert _tristate(ratio=5.0, pass_=10.0, fail_=100.0) == "PASS"


def test_tristate_warn():
    assert _tristate(ratio=50.0, pass_=10.0, fail_=100.0) == "WARN"


def test_tristate_fail():
    assert _tristate(ratio=500.0, pass_=10.0, fail_=100.0) == "FAIL"


def test_tristate_boundary_inclusive():
    # ratio == pass_ is still PASS; ratio == fail_ is WARN (not FAIL)
    assert _tristate(10.0, 10.0, 100.0) == "PASS"
    assert _tristate(100.0, 10.0, 100.0) == "WARN"


def test_load_floor_returns_shipped_defaults_before_calibration():
    # Week 1 Day 4: floors.toml is empty; helper should return a conservative
    # default so rules still compute violation_ratio without KeyError.
    floor = _load_floor(
        rule="PH-RES-001",
        pde="laplace",
        grid_shape=(64, 64),
        method="fd4",
        norm="H-1",
    )
    assert floor.value > 0
    assert floor.tolerance >= 1.0


def test_load_floor_calibrated_path_uses_file(tmp_path, monkeypatch):
    # Redirect _helpers._FLOORS_PATH at a temp file, write a matching entry,
    # assert _load_floor reads it instead of falling back to shipped defaults.
    from physics_lint.rules import _helpers

    fake_floors = tmp_path / "floors.toml"
    fake_floors.write_text(
        "[[floor]]\n"
        'rule = "PH-RES-001"\n'
        'pde = "laplace"\n'
        "grid_shape = [64, 64]\n"
        'method = "fd4"\n'
        'norm = "H-1"\n'
        "value = 3.14e-6\n"
        "tolerance = 5.0\n"
    )
    monkeypatch.setattr(_helpers, "_FLOORS_PATH", fake_floors)
    floor = _load_floor(
        rule="PH-RES-001",
        pde="laplace",
        grid_shape=(64, 64),
        method="fd4",
        norm="H-1",
    )
    assert floor.source == "calibrated"
    assert floor.value == pytest.approx(3.14e-6)
    assert floor.tolerance == pytest.approx(5.0)


def test_load_floor_calibrated_path_no_match_falls_through(tmp_path, monkeypatch):
    from physics_lint.rules import _helpers

    fake_floors = tmp_path / "floors.toml"
    fake_floors.write_text(
        "[[floor]]\n"
        'rule = "PH-BC-001"\n'  # different rule, no match for PH-RES-001
        'pde = "laplace"\n'
        "grid_shape = [64, 64]\n"
        'method = "fd4"\n'
        'norm = "H-1"\n'
        "value = 1e-9\n"
    )
    monkeypatch.setattr(_helpers, "_FLOORS_PATH", fake_floors)
    floor = _load_floor(
        rule="PH-RES-001",
        pde="laplace",
        grid_shape=(64, 64),
        method="fd4",
        norm="H-1",
    )
    assert floor.source == "shipped"


def test_load_floor_missing_rule_uses_fallback_default():
    # An unknown rule/pde combination that's NOT in _SHIPPED_DEFAULTS falls
    # through to the final `default = 1e-5` path.
    floor = _load_floor(
        rule="PH-NOT-A-RULE",
        pde="laplace",
        grid_shape=(64, 64),
        method="fd4",
        norm="H-1",
    )
    assert floor.source == "shipped"
    assert floor.value == 1e-5
