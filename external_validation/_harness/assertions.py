"""Shared assertion primitives (spec section 1.2).

Four patterns reused across all external-validation anchors:
    assert_within           - single-value tolerance check
    assert_slope_in_range   - log-log convergence-rate check
    assert_ranking_matches  - ordinal agreement check
    assert_rule_passes      - rule-on-known-correct-input
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def assert_within(*, measured: float, expected: float, rel_tol: float, abs_tol: float) -> None:
    """Pass iff |measured - expected| <= max(rel_tol * |expected|, abs_tol)."""
    diff = abs(measured - expected)
    bound = max(rel_tol * abs(expected), abs_tol)
    if diff > bound:
        raise AssertionError(
            f"measured={measured!r} is outside tolerance "
            f"(diff={diff:.3e}, bound={bound:.3e}, "
            f"rel_tol={rel_tol}, abs_tol={abs_tol}) of expected={expected!r}"
        )


def assert_slope_in_range(
    *,
    hs: Sequence[float],
    errs: Sequence[float],
    expected_slope: float,
    tolerance: float,
) -> None:
    """Fit log-log slope of errs vs hs; raise if outside +/-tolerance.

    Precondition: all `errs > 0` and all `hs > 0` (log-log requires positives).
    The caller is responsible for clipping a noise floor before passing in.
    """
    hs_arr = np.asarray(hs, dtype=float)
    errs_arr = np.asarray(errs, dtype=float)
    if np.any(hs_arr <= 0) or np.any(errs_arr <= 0):
        raise ValueError(
            "assert_slope_in_range requires strictly positive hs and errs; "
            "clip the noise floor or drop non-positive points before calling."
        )
    log_h = np.log(hs_arr)
    log_e = np.log(errs_arr)
    slope, _intercept = np.polyfit(log_h, log_e, 1)
    low = expected_slope - tolerance
    high = expected_slope + tolerance
    if not (low <= slope <= high):
        raise AssertionError(
            f"log-log slope={slope:.3f} outside [{low:.3f}, {high:.3f}] "
            f"(expected {expected_slope} +/- {tolerance})"
        )


def assert_ranking_matches(
    *,
    measured_ranking: Sequence[str],
    expected_ranking: Sequence[str],
) -> None:
    if list(measured_ranking) != list(expected_ranking):
        raise AssertionError(
            f"ranking mismatch: measured={list(measured_ranking)!r}, "
            f"expected={list(expected_ranking)!r}"
        )


def assert_rule_passes(*, rule_fn: Callable[[], Any]) -> None:
    """Call rule_fn and assert its return is/has PASS status.

    Accepts either a plain string "PASS" or an object with a `.status`
    attribute (matching `physics_lint.report.RuleResult`).
    """
    result = rule_fn()
    status = getattr(result, "status", result)
    if status != "PASS":
        raise AssertionError(
            f"expected PASS from rule_fn; got status={status!r} (result={result!r})"
        )
