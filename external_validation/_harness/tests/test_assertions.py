"""Assertion-primitive tests - 100% coverage target per spec section 1.5."""

from __future__ import annotations

import numpy as np
import pytest

from external_validation._harness.assertions import (
    assert_ranking_matches,
    assert_rule_passes,
    assert_slope_in_range,
    assert_within,
)

# -- assert_within ---------------------------------------------------------


def test_assert_within_passes_within_rel_tol():
    assert_within(measured=1.01, expected=1.0, rel_tol=0.02, abs_tol=0.0)


def test_assert_within_passes_within_abs_tol():
    assert_within(measured=0.001, expected=0.0, rel_tol=0.0, abs_tol=0.01)


def test_assert_within_fails_when_outside_both_tols():
    with pytest.raises(AssertionError, match="outside"):
        assert_within(measured=2.0, expected=1.0, rel_tol=0.05, abs_tol=0.01)


# -- assert_slope_in_range -------------------------------------------------


def test_assert_slope_in_range_passes_on_known_4th_order():
    # E = C * h^4; log E = log C + 4 * log h -> slope 4.0
    hs = np.array([1 / 16, 1 / 32, 1 / 64, 1 / 128])
    errs = 1e-3 * hs**4
    assert_slope_in_range(hs=hs, errs=errs, expected_slope=4.0, tolerance=0.2)


def test_assert_slope_in_range_fails_when_slope_off():
    hs = np.array([1 / 16, 1 / 32, 1 / 64, 1 / 128])
    errs = 1e-3 * hs**2  # actually O(h^2)
    with pytest.raises(AssertionError, match="slope"):
        assert_slope_in_range(hs=hs, errs=errs, expected_slope=4.0, tolerance=0.2)


def test_assert_slope_in_range_rejects_nonpositive_inputs():
    hs = np.array([1 / 16, 1 / 32, 1 / 64, 1 / 128])
    errs = np.array([1e-3, 1e-4, 0.0, 1e-6])  # contains zero
    with pytest.raises(ValueError, match="strictly positive"):
        assert_slope_in_range(hs=hs, errs=errs, expected_slope=4.0, tolerance=0.2)


# -- assert_ranking_matches ------------------------------------------------


def test_assert_ranking_matches_identical():
    assert_ranking_matches(measured_ranking=["a", "b", "c"], expected_ranking=["a", "b", "c"])


def test_assert_ranking_matches_fails_on_swap():
    with pytest.raises(AssertionError, match="ranking"):
        assert_ranking_matches(measured_ranking=["a", "c", "b"], expected_ranking=["a", "b", "c"])


# -- assert_rule_passes ----------------------------------------------------


def test_assert_rule_passes_on_ok_rule():
    def _rule(x: float) -> str:
        return "PASS" if x > 0 else "FAIL"

    assert_rule_passes(rule_fn=lambda: _rule(1.0))


def test_assert_rule_passes_fails_on_fail_rule():
    def _rule(x: float) -> str:
        return "PASS" if x > 0 else "FAIL"

    with pytest.raises(AssertionError, match="expected PASS"):
        assert_rule_passes(rule_fn=lambda: _rule(-1.0))


def test_assert_rule_passes_accepts_status_bearing_object():
    class _Result:
        def __init__(self, status: str) -> None:
            self.status = status

    assert_rule_passes(rule_fn=lambda: _Result("PASS"))
