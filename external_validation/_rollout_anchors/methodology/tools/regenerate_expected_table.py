"""Regenerate methodology/tests/fixtures/expected_table.md from the fixtures.

The golden test (`test_renderer_golden_output_matches_expected_table`) calls
`render_cross_stack_table([SEGNN_FIXTURE, GNS_FIXTURE])` with that explicit
argument order. The CLI's `--sarif-dir` mode globs files alphabetically, which
would produce `[GNS, SEGNN]` (the reverse) and corrupt the golden if used to
regenerate.

This helper exists to keep the regen procedure aligned with the test's order.
Invoke when the renderer's output format changes intentionally:

    PATH=".venv/bin:$PATH" python \\
        external_validation/_rollout_anchors/methodology/tools/regenerate_expected_table.py

Run from the physics-lint repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from external_validation._rollout_anchors.methodology.tools.render_cross_stack_table import (  # noqa: E402
    render_cross_stack_table,
)

FIXTURES_DIR = _REPO_ROOT / "external_validation/_rollout_anchors/methodology/tests/fixtures"


def main() -> int:
    output = render_cross_stack_table(
        [
            FIXTURES_DIR / "segnn_tgv2d_fixture.sarif",
            FIXTURES_DIR / "gns_tgv2d_fixture.sarif",
        ]
    )
    expected_path = FIXTURES_DIR / "expected_table.md"
    expected_path.write_text(output)
    print(f"Wrote {expected_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
