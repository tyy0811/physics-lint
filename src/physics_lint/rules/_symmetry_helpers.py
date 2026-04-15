"""Shared helpers for the PH-SYM-* rules."""

from __future__ import annotations

import numpy as np

from physics_lint.spec import SymmetryLiteral, SymmetrySpec


def equivariance_error_np(lhs: np.ndarray, rhs: np.ndarray, *, eps: float = 1e-12) -> float:
    """Relative L^2 difference between two tensors.

    Computes ``||lhs - rhs|| / max(||lhs||, eps)``.

    For the finite-transformation equivariance tests in PH-SYM-001/002/004,
    pass ``lhs = f(T_in(x))`` (model applied to transformed input) and
    ``rhs = T_out(f(x))`` (transformation applied to model output). For
    norm-preserving ``T_out`` and an exactly equivariant ``f``, the denominator
    ``||lhs||`` equals ``||f(x)||`` and the formula reduces to
    design doc §9.2's ``|| T_out(f(x)) - f(T_in(x)) || / max(||f(x)||, eps)``.
    """
    diff = lhs - rhs
    denom = float(np.linalg.norm(lhs))
    return float(np.linalg.norm(diff) / max(denom, eps))


def is_symmetry_declared(spec: SymmetrySpec, target: SymmetryLiteral) -> bool:
    """Return True if `target` is in the declared list, or implied by a superset.

    Current implications (V1):
        D4 implies C4 + reflection_x + reflection_y
    """
    declared = set(spec.declared)
    if target in declared:
        return True
    return "D4" in declared and target in {"C4", "reflection_x", "reflection_y"}
