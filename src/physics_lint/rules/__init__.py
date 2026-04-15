"""Rule catalog — lazy-discovered at import time.

Do NOT eagerly import rule check() functions here; the _registry module
scans for __rule_id__, __rule_name__, __default_severity__, __input_modes__
at the module level and defers check() imports until a rule is actually
invoked. This keeps `physics-lint rules list` fast.
"""

from physics_lint.rules._registry import list_rules, load_check

__all__ = ["list_rules", "load_check"]
