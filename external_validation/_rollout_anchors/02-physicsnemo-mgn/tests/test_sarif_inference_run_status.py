"""Phase 2 Task 8 — design §2.5 SARIF ``inference_run_status`` regression test.

The field documents whether the rollout that produced the SARIF came from
a clean inference fire, a salvage path, or no inference at all
(GT control arm). Salvage paths (``from_truncated_inference``,
``from_post_oom_salvage``, etc.) are design-anticipated; Phase 2's N=1
clean path is supposed to be uniformly ``from_completed_inference`` for
the MGN arm and ``n/a_gt_control_arm`` for the GT arm.

This test asserts the pin holds on the committed SARIFs. If a Phase 3
re-emission picks up a salvage classification, the test FAILs and the
methodology routing (D-entry amendment vs code change) decides the next
step per the design's salvage-tag forward-flag.
"""

from __future__ import annotations

import json
import pathlib

import pytest

OUTPUTS = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "sarif"


@pytest.mark.parametrize(
    "arm,expected",
    [
        ("gt", "n/a_gt_control_arm"),
        ("mgn", "from_completed_inference"),
    ],
)
def test_sarif_inference_run_status_present_and_pinned(arm: str, expected: str) -> None:
    """The committed gt.sarif + mgn.sarif each carry the
    ``inference_run_status`` run-level property at the design-§2.5-pinned
    value. The GT arm is ``n/a_gt_control_arm`` (no inference fire); the
    MGN arm is ``from_completed_inference`` (Task 6's A10G fire was clean).
    """
    sarif_path = OUTPUTS / f"{arm}.sarif"
    assert sarif_path.exists(), f"{sarif_path} missing — re-emit via Task 5 / Task 7."
    sarif = json.loads(sarif_path.read_text())
    props = sarif["runs"][0]["properties"]
    assert "inference_run_status" in props, (
        f"{arm}.sarif missing 'inference_run_status' run-level property; "
        f"design §2.5 requires the field on every Phase 2 SARIF emission."
    )
    actual = props["inference_run_status"]
    assert actual == expected, (
        f"{arm}.sarif inference_run_status = {actual!r}; expected {expected!r}. "
        f"A salvage-marker value here (e.g. 'from_truncated_inference', "
        f"'from_post_oom_salvage') is a Phase-3 writeup forward-flag, not a "
        f"code-patch — methodology routes via D-entry amendment per the "
        f"salvage-tag design."
    )


def test_sarif_inference_run_status_is_a_well_known_value() -> None:
    """Defense-in-depth: the value must be one of the design-§2.5
    enumerated states. Catches a typo or rogue value that bypasses the
    pin assertion above (which only fires on the GT / MGN arms).
    """
    valid = {
        "from_completed_inference",
        "from_truncated_inference",
        "from_post_oom_salvage",
        "from_aborted_inference",
        "n/a_gt_control_arm",
    }
    for arm in ("gt", "mgn"):
        sarif_path = OUTPUTS / f"{arm}.sarif"
        if not sarif_path.exists():
            continue
        sarif = json.loads(sarif_path.read_text())
        status = sarif["runs"][0]["properties"].get("inference_run_status")
        assert status in valid, (
            f"{arm}.sarif inference_run_status = {status!r} is not one of the "
            f"design-§2.5 enumerated values {sorted(valid)}."
        )
