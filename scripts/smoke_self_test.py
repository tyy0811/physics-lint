"""physics-lint self-test entry point (thin wrapper).

The self-test logic lives in the installable package at
`physics_lint.selftest` so that `physics-lint self-test` also works
from a wheel / PyPI install where `scripts/` is not shipped. This
wrapper is retained for historical reference and for reproducing the
self-test from a clone without installing (``python
scripts/smoke_self_test.py``). Both entry points call the same
``physics_lint.selftest.main`` function.
"""

from __future__ import annotations

import sys

from physics_lint.selftest import main

if __name__ == "__main__":
    sys.exit(main())
