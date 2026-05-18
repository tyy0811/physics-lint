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
