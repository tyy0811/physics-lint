"""Symmetry helpers: equivariance error and declared-symmetry gate."""

import numpy as np

from physics_lint.rules._symmetry_helpers import (
    equivariance_error_np,
    is_symmetry_declared,
)
from physics_lint.spec import SymmetrySpec


def test_equivariance_error_zero_on_identity():
    u = np.sin(np.linspace(0, np.pi, 16))[:, None] * np.sin(np.linspace(0, np.pi, 16))[None, :]
    err = equivariance_error_np(u, u)
    assert err == 0.0


def test_equivariance_error_nonzero_on_noise():
    u = np.random.default_rng(0).normal(size=(16, 16))
    err = equivariance_error_np(u, u + 0.1)
    assert err > 0.0


def test_is_symmetry_declared_d4_implies_c4():
    # D4 implies C4 (D4 includes rotations)
    assert is_symmetry_declared(SymmetrySpec(declared=["D4"]), "C4")


def test_is_symmetry_declared_explicit():
    assert is_symmetry_declared(SymmetrySpec(declared=["C4"]), "C4")
    assert not is_symmetry_declared(SymmetrySpec(declared=["reflection_x"]), "C4")


def test_is_symmetry_declared_reflection():
    assert is_symmetry_declared(SymmetrySpec(declared=["D4"]), "reflection_x")
    assert is_symmetry_declared(SymmetrySpec(declared=["reflection_x"]), "reflection_x")
