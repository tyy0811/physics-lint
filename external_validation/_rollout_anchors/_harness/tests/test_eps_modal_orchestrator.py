"""Unit tests for eps_modal_orchestrator (rung 4b T7)."""

from __future__ import annotations


def test_build_main_sweep_transforms_yields_120_entries_for_n_trajs_20():
    """Main sweep: 4 PH-SYM-001 angles + 1 PH-SYM-002 reflection +
    1 PH-SYM-004 translation = 6 transforms x 20 trajs = 120 entries.
    PH-SYM-003 SKIP not in this list (handled separately)."""
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        build_main_sweep_transforms,
    )

    transforms = build_main_sweep_transforms(n_trajs=20)
    assert len(transforms) == 120, f"expected 120, got {len(transforms)}"

    rule_counts: dict[str, int] = {}
    for entry in transforms:
        rule_counts[entry["rule_id"]] = rule_counts.get(entry["rule_id"], 0) + 1
    assert rule_counts == {"PH-SYM-001": 80, "PH-SYM-002": 20, "PH-SYM-004": 20}

    # No PH-SYM-003 in main sweep (SKIP shortcut).
    assert "PH-SYM-003" not in rule_counts


def test_build_main_sweep_transforms_ph_sym_001_has_4_angles():
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        build_main_sweep_transforms,
    )

    transforms = build_main_sweep_transforms(n_trajs=20)
    angles = sorted({e["transform_param"] for e in transforms if e["rule_id"] == "PH-SYM-001"})
    assert angles == ["0", "3pi_2", "pi", "pi_2"]

    # Identity (theta=0) has transform_kind=identity; others have rotation.
    identity_kinds = {
        e["transform_kind"]
        for e in transforms
        if e["rule_id"] == "PH-SYM-001" and e["transform_param"] == "0"
    }
    assert identity_kinds == {"identity"}

    rotation_params = {
        e["transform_param"]
        for e in transforms
        if e["rule_id"] == "PH-SYM-001" and e["transform_kind"] == "rotation"
    }
    assert rotation_params == {"pi_2", "pi", "3pi_2"}


def test_build_main_sweep_transforms_each_rule_covers_all_traj_indices():
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        build_main_sweep_transforms,
    )

    transforms = build_main_sweep_transforms(n_trajs=20)
    # For each (rule, transform_param) pair, original_traj_index covers 0..19.
    by_pair: dict[tuple, list[int]] = {}
    for e in transforms:
        key = (e["rule_id"], e["transform_param"])
        by_pair.setdefault(key, []).append(e["original_traj_index"])
    for key, idxs in by_pair.items():
        assert sorted(idxs) == list(range(20)), f"{key} has {sorted(idxs)}"


def test_build_main_sweep_transforms_transform_fn_is_callable():
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        build_main_sweep_transforms,
    )

    transforms = build_main_sweep_transforms(n_trajs=20)
    for e in transforms:
        assert callable(e["transform_fn"]), f"{e['rule_id']}/{e['transform_param']}"


def test_build_figure_sweep_transforms_yields_3_entries():
    """Figure sweep: 1 angle (pi/2) x 3 trajs (0, 7, 14) = 3 entries."""
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        build_figure_sweep_transforms,
    )

    transforms = build_figure_sweep_transforms()
    assert len(transforms) == 3
    assert all(e["rule_id"] == "PH-SYM-001" for e in transforms)
    assert all(e["transform_kind"] == "rotation" for e in transforms)
    assert all(e["transform_param"] == "pi_2" for e in transforms)
    assert sorted(e["original_traj_index"] for e in transforms) == [0, 7, 14]


def test_interpret_sanity_probe_verdict_pass_at_floor():
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        interpret_sanity_probe_verdict,
    )

    verdict = interpret_sanity_probe_verdict(eps=1e-7)
    assert verdict["status"] == "PASS"
    assert verdict["abort"] is False
    assert "1.0e-07" in verdict["message"]


def test_interpret_sanity_probe_verdict_pass_at_threshold_boundary():
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        interpret_sanity_probe_verdict,
    )

    # Exactly at 1e-5 should still pass (gate is <=).
    verdict = interpret_sanity_probe_verdict(eps=1e-5)
    assert verdict["status"] == "PASS"
    assert verdict["abort"] is False


def test_interpret_sanity_probe_verdict_concerning_band():
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        interpret_sanity_probe_verdict,
    )

    verdict = interpret_sanity_probe_verdict(eps=1e-4)
    assert verdict["status"] == "ABORT"
    assert verdict["abort"] is True
    assert "concerning" in verdict["message"].lower()
    # Diagnostic mentions borderline FP and partial-bug per design §6.
    assert "borderline" in verdict["message"].lower() or "partial" in verdict["message"].lower()


def test_interpret_sanity_probe_verdict_clear_bug_band():
    from external_validation._rollout_anchors._harness.eps_modal_orchestrator import (
        interpret_sanity_probe_verdict,
    )

    verdict = interpret_sanity_probe_verdict(eps=1e-2)
    assert verdict["status"] == "ABORT"
    assert verdict["abort"] is True
    assert "clear bug" in verdict["message"].lower()
    # Diagnostic lists the four candidate causes per design §6.
    msg = verdict["message"].lower()
    assert "coordinate" in msg
    assert "frame" in msg
    assert "normalization" in msg
    assert "manifest" in msg
