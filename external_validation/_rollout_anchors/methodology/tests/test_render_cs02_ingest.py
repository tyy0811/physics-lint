"""Pre-flight verification: CS02 SARIFs satisfy the existing D0-19 v1.0
required-field contract without per-source parameterization.

If this test ever starts failing (e.g., a future SARIF emission drops one
of the 10 required fields, or a schema bump lands without re-emission),
the renderer extension's "no parameterization needed" premise is broken
and Phase 3 Task 2 should re-open the per-source parameterization decision.
"""

from pathlib import Path

import pytest

from external_validation._rollout_anchors.methodology.tools.render_cross_stack_table import (
    _assert_run_level,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CS02_SARIF_DIR = (
    REPO_ROOT
    / "external_validation"
    / "_rollout_anchors"
    / "02-physicsnemo-mgn"
    / "outputs"
    / "sarif"
)


@pytest.mark.parametrize("sarif_name", ["gt.sarif", "mgn.sarif"])
def test_cs02_sarifs_pass_existing_assert_run_level(sarif_name: str) -> None:
    """CS02 SARIFs (post-F1-absorption) satisfy the renderer's existing
    10-required-field contract without per-source parameterization. The
    Q2 brainstorming initially scoped a per-source parameterization
    refactor; pre-flight inspection found this is YAGNI because CS02
    SARIFs already carry the LB-side required fields via sentinel
    values (lagrangebench_sha = 'n/a_cs02_physicsnemo').
    """
    import json

    path = CS02_SARIF_DIR / sarif_name
    sarif = json.loads(path.read_text())
    run_props = _assert_run_level(sarif, path)

    assert run_props["source"] == "rollout-anchor-harness"
    assert run_props["harness_sarif_schema_version"] == "1.0"
    assert run_props["lagrangebench_sha"] == "n/a_cs02_physicsnemo"
    assert run_props["dataset_name"] == "vortex_shedding_2d"
