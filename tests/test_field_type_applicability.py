"""Central field-type applicability filter: default grid+callable, PH-CON-005
mesh_vector-only, and the 18-rule type-specific SKIP audit on a mesh_vector
target."""

from physics_lint.cli.check import _field_type_applies
from physics_lint.rules import _registry


def test_default_field_types_are_grid_callable():
    entries = {e.rule_id: e for e in _registry.list_rules()}
    # A representative grid rule rides the default.
    assert entries["PH-RES-001"].field_types == frozenset({"grid", "callable"})
    # PH-CON-005 declares mesh_vector.
    assert entries["PH-CON-005"].field_types == frozenset({"mesh_vector"})


def test_grid_rules_skip_on_mesh_vector_with_type_specific_reason():
    entries = [e for e in _registry.list_rules() if e.rule_id != "PH-CON-005"]
    assert len(entries) == 18  # registry-verified denominator
    for e in entries:
        assert not _field_type_applies(e, "mesh_vector"), e.rule_id


def test_ph_con_005_applies_to_mesh_vector_only():
    entries = {e.rule_id: e for e in _registry.list_rules()}
    assert _field_type_applies(entries["PH-CON-005"], "mesh_vector")
    assert not _field_type_applies(entries["PH-CON-005"], "grid")
