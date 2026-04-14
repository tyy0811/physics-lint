"""Rule registry tests — lazy metadata discovery.

The registry reads __rule_id__, __rule_name__, __default_severity__, and
__input_modes__ from rule modules WITHOUT importing their check() functions,
so `physics-lint rules list` stays under 50 ms.
"""

import pytest

from physics_lint.rules import _registry


def test_registry_discovers_week_1_rules():
    rules = _registry.list_rules()
    ids = {r.rule_id for r in rules}
    expected = {"PH-RES-001", "PH-RES-002", "PH-RES-003"}
    assert expected.issubset(ids), f"missing rules: {expected - ids}"


def test_registry_lazy_check_not_imported():
    # Iterating metadata should NOT import the check() functions.
    rules = _registry.list_rules()
    for r in rules:
        assert r.check_fn is None, f"{r.rule_id} check was eagerly imported"


def test_registry_materialize_check():
    rules = _registry.list_rules()
    first = next(iter(rules))
    check = _registry.load_check(first)
    assert callable(check)


def test_load_check_caches_on_entry():
    rules = _registry.list_rules()
    first = next(iter(rules))
    check_first = _registry.load_check(first)
    check_second = _registry.load_check(first)  # second call hits the cache
    assert check_first is check_second


def test_load_check_raises_when_module_missing_check(tmp_path, monkeypatch):
    # Synthesize a rules module at runtime that has the metadata attrs
    # but no check() function. Temporarily inject it into physics_lint.rules
    # via sys.modules + a stub attribute on the rules package path.
    import sys
    import types

    fake = types.ModuleType("physics_lint.rules._fake_metadata_only")
    fake.__rule_id__ = "PH-FAKE-999"
    fake.__rule_name__ = "fake"
    fake.__default_severity__ = "info"
    fake.__input_modes__ = frozenset({"adapter"})
    sys.modules["physics_lint.rules._fake_metadata_only"] = fake
    entry = _registry.RegistryEntry(
        rule_id="PH-FAKE-999",
        rule_name="fake",
        default_severity="info",
        input_modes=frozenset({"adapter"}),
        module_name="physics_lint.rules._fake_metadata_only",
    )
    try:
        with pytest.raises(AttributeError, match="check"):
            _registry.load_check(entry)
    finally:
        sys.modules.pop("physics_lint.rules._fake_metadata_only", None)
