"""Field ABC contract tests.

The ABC is not instantiable on its own; this test asserts the required
abstract methods are declared so subclasses that forget one fail at
instantiation time rather than at first use.
"""

import pytest

from physics_lint.field import Field


def test_field_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        Field()  # type: ignore[abstract]


def test_field_abstract_method_names():
    expected = {"values", "at", "grad", "laplacian", "integrate", "values_on_boundary"}
    assert set(Field.__abstractmethods__) == expected
