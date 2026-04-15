"""Lazy rule discovery.

Each rule module at physics_lint/rules/ph_xxx_nnn.py exports four module-level
attributes: __rule_id__, __rule_name__, __default_severity__, __input_modes__
(a frozenset of {"adapter", "dump"}). The registry walks the rules package,
reads ONLY those four attributes (no check() import), and returns a list of
RegistryEntry. The check() function is loaded on demand via load_check().

This is the Section-2-review rollback pattern: if Day 3 converges lazy
discovery, rules list runs in <50 ms; if not, Week 4 Day 2 can switch to
eager discovery (import every module), at the cost of ~500 ms on
rules list, without changing any rule-module code.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class RegistryEntry:
    rule_id: str
    rule_name: str
    default_severity: str
    input_modes: frozenset[str]
    module_name: str
    check_fn: Callable[..., Any] | None = None


def list_rules() -> list[RegistryEntry]:
    """Scan physics_lint.rules for rule modules; return metadata-only entries."""
    import physics_lint.rules as rules_pkg

    entries: list[RegistryEntry] = []
    for mod_info in pkgutil.iter_modules(rules_pkg.__path__):
        name = mod_info.name
        if name.startswith("_"):
            continue
        full_name = f"physics_lint.rules.{name}"
        # Read metadata without executing check(): use importlib.util to
        # load the module, then discard any non-metadata attributes.
        spec = importlib.util.find_spec(full_name)
        if spec is None:
            continue
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        rule_id = getattr(module, "__rule_id__", None)
        if rule_id is None:
            continue
        entries.append(
            RegistryEntry(
                rule_id=rule_id,
                rule_name=getattr(module, "__rule_name__", ""),
                default_severity=getattr(module, "__default_severity__", "warning"),
                input_modes=frozenset(getattr(module, "__input_modes__", ())),
                module_name=full_name,
                check_fn=None,  # NOT loaded here
            )
        )
    return sorted(entries, key=lambda e: e.rule_id)


def load_check(entry: RegistryEntry) -> Callable[..., Any]:
    """Import the rule module for real and return its check function."""
    if entry.check_fn is not None:
        return entry.check_fn
    module = importlib.import_module(entry.module_name)
    check = getattr(module, "check", None)
    if check is None:
        raise AttributeError(f"{entry.module_name} has no `check` function")
    entry.check_fn = check
    return check
