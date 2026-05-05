"""Tests for lint_npz_dir.py: generic npz-dir → HarnessResults bridge.

Per DECISIONS.md D0-19, this module reads a directory of
particle_rollout_traj{NN}.npz files, invokes the three conservation
defects (mass_conservation_defect, energy_drift, dissipation_sign_violation)
on each, and returns a list[HarnessResult] suitable for emit_sarif.

Energy_drift SKIP rows get ke_initial / ke_final attached to
extra_properties (the per-row varying values that D0-19's
template-constant skip_reason no longer interpolates).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from external_validation._rollout_anchors._harness.lint_npz_dir import (
    EmptyNpzDirectoryError,
    lint_npz_dir,
)
from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    ParticleRollout,
    save_rollout_npz,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dissipative_rollout(dataset_name: str = "tgv2d") -> ParticleRollout:
    """Decaying-KE rollout — fires the D0-18 dissipative SKIP path."""
    rng = np.random.default_rng(42)
    n_t, n_p = 8, 4
    dt = 0.01
    positions = np.zeros((n_t, n_p, 2), dtype=float)
    velocities = np.zeros((n_t, n_p, 2), dtype=float)
    v0 = rng.normal(scale=1.0, size=(n_p, 2))
    for t in range(n_t):
        velocities[t] = v0 * np.exp(-0.5 * t * dt)
        positions[t] = 0.5
    return ParticleRollout(
        positions=positions,
        velocities=velocities,
        particle_type=np.zeros(n_p, dtype=np.int32),
        particle_mass=np.ones(n_p, dtype=np.float64),
        dt=dt,
        domain_box=np.array([[0.0, 0.0], [1.0, 1.0]]),
        metadata={"dataset": dataset_name},
    )


def _save_n_npzs(tmp_path: Path, n: int = 3, dataset_name: str = "tgv2d") -> Path:
    """Persist `n` rollouts as particle_rollout_traj{NN}.npz files in tmp_path."""
    for i in range(n):
        rollout = _make_dissipative_rollout(dataset_name=dataset_name)
        save_rollout_npz(rollout, tmp_path / f"particle_rollout_traj{i:02d}.npz")
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Directory presence / emptiness
# ---------------------------------------------------------------------------


def test_lint_npz_dir_raises_on_empty_dir(tmp_path: Path) -> None:
    """An empty directory raises EmptyNpzDirectoryError rather than returning
    an empty list. Silent empty SARIF is a methodology hazard
    (writeup table renders blank with no error).
    """
    with pytest.raises(EmptyNpzDirectoryError):
        lint_npz_dir(tmp_path)


# ---------------------------------------------------------------------------
# 2. Happy-path row construction
# ---------------------------------------------------------------------------


def test_lint_npz_dir_yields_three_rows_per_npz(tmp_path: Path) -> None:
    """For 3 npz files x 3 defects, expect 9 HarnessResult rows in
    deterministic ordering (sorted by traj_index, then by defect emission
    order from _DEFECTS).
    """
    _save_n_npzs(tmp_path, n=3)
    results = lint_npz_dir(tmp_path)
    assert len(results) == 9

    # Verify ordering: 3 rows per traj_index, ascending traj_index.
    # Row 0..2 from traj 0; row 3..5 from traj 1; row 6..8 from traj 2.
    for traj_idx in range(3):
        rows = results[traj_idx * 3 : (traj_idx + 1) * 3]
        assert rows[0].rule_id == "harness:mass_conservation_defect"
        assert rows[1].rule_id == "harness:energy_drift"
        assert rows[2].rule_id == "harness:dissipation_sign_violation"
        for row in rows:
            assert row.extra_properties["traj_index"] == traj_idx
            assert (
                row.extra_properties["npz_filename"] == f"particle_rollout_traj{traj_idx:02d}.npz"
            )


# ---------------------------------------------------------------------------
# 3. ke_initial / ke_final attachment (D0-19 contract)
# ---------------------------------------------------------------------------


def test_energy_drift_skip_rows_have_ke_initial_and_ke_final(tmp_path: Path) -> None:
    """D0-19: harness:energy_drift SKIP rows MUST carry ke_initial and
    ke_final in extra_properties (the per-row varying values that the
    template-constant skip_reason no longer interpolates).
    """
    _save_n_npzs(tmp_path, n=2, dataset_name="tgv2d")  # tgv2d → dissipative SKIP
    results = lint_npz_dir(tmp_path)
    energy_drift_rows = [r for r in results if r.rule_id == "harness:energy_drift"]
    assert len(energy_drift_rows) == 2
    for row in energy_drift_rows:
        # Defect SKIPped → raw_value is None
        assert row.raw_value is None
        # ke_initial / ke_final present and finite
        assert "ke_initial" in row.extra_properties
        assert "ke_final" in row.extra_properties
        ke_i = row.extra_properties["ke_initial"]
        ke_f = row.extra_properties["ke_final"]
        assert isinstance(ke_i, float) and isinstance(ke_f, float)
        # Dissipative rollout: KE(end) < KE(0)
        assert ke_f < ke_i


def test_non_energy_drift_rows_do_not_have_ke_fields(tmp_path: Path) -> None:
    """ke_initial / ke_final live ONLY on harness:energy_drift rows
    (other rules don't get them per D0-19's result-level field table).
    """
    _save_n_npzs(tmp_path, n=2, dataset_name="tgv2d")
    results = lint_npz_dir(tmp_path)
    non_energy_rows = [r for r in results if r.rule_id != "harness:energy_drift"]
    for row in non_energy_rows:
        assert "ke_initial" not in row.extra_properties
        assert "ke_final" not in row.extra_properties


# ---------------------------------------------------------------------------
# 4. Wrapper preserves D0-18 signal end-to-end
# ---------------------------------------------------------------------------


def test_lint_npz_dir_preserves_d0_18_skip_signal(tmp_path: Path) -> None:
    """Wrapper-preserves-contract test: lint_npz_dir doesn't muck up the
    D0-18 SKIP signal as it traverses defect → HarnessResult. A
    dissipative-system npz produces an energy_drift row with skip_reason
    set (not raw_value), confirming the wrapper preserves what the
    defect emitted.
    """
    _save_n_npzs(tmp_path, n=1, dataset_name="tgv2d")  # known dissipative
    results = lint_npz_dir(tmp_path)
    energy_drift_rows = [r for r in results if r.rule_id == "harness:energy_drift"]
    assert len(energy_drift_rows) == 1
    row = energy_drift_rows[0]
    assert row.raw_value is None  # SKIP signal preserved
    # Message documents the SKIP path
    assert "SKIP" in row.message


def test_lint_npz_dir_fires_raw_on_unknown_dataset(tmp_path: Path) -> None:
    """Regression guard: a non-LB dataset name does NOT trigger D0-18
    SKIP — wrapper must not silently classify any synthetic dataset as
    dissipative. raw_value is set; skip path not taken.
    """
    _save_n_npzs(tmp_path, n=1, dataset_name="synthetic-non-lb-name")
    results = lint_npz_dir(tmp_path)
    energy_drift_rows = [r for r in results if r.rule_id == "harness:energy_drift"]
    assert len(energy_drift_rows) == 1
    row = energy_drift_rows[0]
    assert row.raw_value is not None  # raw value emitted


# ---------------------------------------------------------------------------
# 5. Non-npz files ignored
# ---------------------------------------------------------------------------


def test_lint_npz_dir_ignores_non_npz_files(tmp_path: Path) -> None:
    """Files not matching particle_rollout_traj*.npz (e.g., the .pkl
    files persisted alongside, the metrics .pkl, README files) MUST be
    ignored. lint_npz_dir's glob is `particle_rollout_traj*.npz`.
    """
    _save_n_npzs(tmp_path, n=2)
    # Add some red-herring files
    (tmp_path / "rollout_0.pkl").write_text("not an npz")
    (tmp_path / "metrics2026_05_04.pkl").write_text("not an npz")
    (tmp_path / "README.md").write_text("ignored")
    (tmp_path / "particle_rollout_traj99_extra.txt").write_text("doesn't match glob")
    results = lint_npz_dir(tmp_path)
    # Only the 2 npzs x 3 defects = 6 rows.
    assert len(results) == 6
