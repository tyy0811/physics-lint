"""Unit test for the P2.1 multi-trajectory gap tabulation."""

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).parents[1] / "tabulate_p2_multi_trajectory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("tabulate_p2", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_sarif(path: Path, raw_value: float) -> None:
    """Write a minimal harness SARIF carrying one mass_conservation_defect result."""
    doc = {
        "runs": [
            {
                "results": [
                    {
                        "ruleId": "harness:mass_conservation_defect",
                        "level": "note",
                        "properties": {"raw_value": raw_value},
                    }
                ]
            }
        ]
    }
    path.write_text(json.dumps(doc))


def test_tabulate_computes_gap_median_and_range(tmp_path):
    mod = _load_module()
    # Three trajectories with known GT/MGN pairs:
    #   t=1: gt=0.10  mgn=0.11  -> gap +10.00% of GT
    #   t=2: gt=0.10  mgn=0.10  -> gap   0.00%
    #   t=3: gt=0.10  mgn=0.16  -> gap +60.00%  (FAIL band)
    for t, (gt, mgn) in {1: (0.10, 0.11), 2: (0.10, 0.10), 3: (0.10, 0.16)}.items():
        _write_sarif(tmp_path / f"traj{t}_gt.sarif", gt)
        _write_sarif(tmp_path / f"traj{t}_mgn.sarif", mgn)

    report = mod.tabulate(results_dir=tmp_path, trajectories=[1, 2, 3])
    rows = {r["traj_idx"]: r for r in report["per_trajectory"]}
    # gap_pct is a derived float; compare with tolerance, not exact ==.
    assert abs(rows[1]["gap_pct_of_gt"] - 10.0) < 1e-9
    assert abs(rows[2]["gap_pct_of_gt"] - 0.0) < 1e-9
    assert abs(rows[3]["gap_pct_of_gt"] - 60.0) < 1e-9
    assert rows[1]["d0_24_v2_band"] == "PASS"
    assert rows[3]["d0_24_v2_band"] == "FAIL"
    assert rows[1]["gap_within_fe_floor"] is True  # 0.01 gap < 0.10 GT floor

    summary = report["summary"]
    assert summary["n_trajectories"] == 3
    assert abs(summary["median_gap_pct"] - 10.0) < 1e-9  # median of {0, 10, 60}
    assert abs(summary["min_gap_pct"] - 0.0) < 1e-9
    assert abs(summary["max_gap_pct"] - 60.0) < 1e-9
    assert summary["all_d0_24_v2_pass"] is False


def test_d0_24_v2_band_partition():
    """The D0-24 v2 band partitions |gap| with no gap or overlap at the edges."""
    band = _load_module()._d0_24_v2_band
    # PASS: |gap| <= 20
    assert band(0.0) == "PASS"
    assert band(19.999) == "PASS"
    assert band(20.0) == "PASS"  # boundary lands in the lower band
    # MARGINAL: 20 < |gap| <= 50
    assert band(20.001) == "MARGINAL"
    assert band(35.0) == "MARGINAL"
    assert band(49.999) == "MARGINAL"
    assert band(50.0) == "MARGINAL"  # boundary lands in the lower band
    # FAIL: |gap| > 50
    assert band(50.001) == "FAIL"
    assert band(75.0) == "FAIL"
    # negative gaps classify on magnitude (abs)
    assert band(-20.0) == "PASS"
    assert band(-35.0) == "MARGINAL"
    assert band(-60.0) == "FAIL"


def test_tabulate_negative_gaps_and_summary_fields(tmp_path):
    """A mix of negative/positive gaps; check the marginal medians + summary flags."""
    mod = _load_module()
    #   t=1: gt=0.040 mgn=0.030 -> gap -25.00% MARGINAL, |gap_abs| 0.010 < 0.040
    #   t=2: gt=0.050 mgn=0.052 -> gap  +4.00% PASS,     |gap_abs| 0.002 < 0.050
    #   t=3: gt=0.060 mgn=0.030 -> gap -50.00% MARGINAL, |gap_abs| 0.030 < 0.060
    for t, (gt, mgn) in {1: (0.040, 0.030), 2: (0.050, 0.052), 3: (0.060, 0.030)}.items():
        _write_sarif(tmp_path / f"traj{t}_gt.sarif", gt)
        _write_sarif(tmp_path / f"traj{t}_mgn.sarif", mgn)

    report = mod.tabulate(results_dir=tmp_path, trajectories=[1, 2, 3])
    rows = {r["traj_idx"]: r for r in report["per_trajectory"]}
    assert abs(rows[1]["gap_pct_of_gt"] - (-25.0)) < 1e-9
    assert rows[1]["d0_24_v2_band"] == "MARGINAL"  # negative gap, MARGINAL band
    assert abs(rows[3]["gap_pct_of_gt"] - (-50.0)) < 1e-9
    assert rows[3]["d0_24_v2_band"] == "MARGINAL"  # negative gap at the 50% edge
    assert all(r["gap_within_fe_floor"] for r in rows.values())

    summary = report["summary"]
    assert abs(summary["median_gap_pct"] - (-25.0)) < 1e-9  # median of {-50, -25, 4}
    assert abs(summary["median_gt_raw_value"] - 0.050) < 1e-9  # median of {.04, .05, .06}
    assert abs(summary["median_mgn_raw_value"] - 0.030) < 1e-9  # median of {.030, .030, .052}
    assert summary["all_gaps_within_fe_floor"] is True
    assert summary["all_d0_24_v2_pass"] is False


def test_tabulate_gap_outside_fe_floor(tmp_path):
    """A gap larger than the GT defect itself flips gap_within_fe_floor false."""
    mod = _load_module()
    # gt=0.10 mgn=0.30 -> gap_abs 0.20 >= gt 0.10, gap +200% -> FAIL
    _write_sarif(tmp_path / "traj7_gt.sarif", 0.10)
    _write_sarif(tmp_path / "traj7_mgn.sarif", 0.30)
    report = mod.tabulate(results_dir=tmp_path, trajectories=[7])
    row = report["per_trajectory"][0]
    assert row["gap_within_fe_floor"] is False
    assert row["d0_24_v2_band"] == "FAIL"
    assert report["summary"]["all_gaps_within_fe_floor"] is False
