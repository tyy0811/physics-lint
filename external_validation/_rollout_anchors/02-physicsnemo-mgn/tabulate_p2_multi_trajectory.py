"""Tabulate the P2.1 multi-trajectory CS02 mass-conservation results.

Reads the per-trajectory GT + MGN SARIF files produced by the Phase-2
Modal lint entrypoints re-run across the 5 P2.1 trajectories, extracts the
``harness:mass_conservation_defect`` raw_value from each, and reports the
per-trajectory MGN/GT gap plus the median and range across trajectories.
Writes results.json next to the SARIFs and prints a Markdown table for the
CS02 README.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

TRAJECTORIES = [88, 48, 44, 38, 60]
RESULTS_DIR = Path(__file__).parent / "outputs" / "p2_multi_trajectory"
MASS_RULE_ID = "harness:mass_conservation_defect"


def _mass_defect(sarif_path: Path) -> float:
    """Return the mass_conservation_defect raw_value from a harness SARIF."""
    doc = json.loads(sarif_path.read_text())
    for result in doc["runs"][0]["results"]:
        if result.get("ruleId") == MASS_RULE_ID:
            raw = result.get("properties", {}).get("raw_value")
            if raw is None:
                raise ValueError(f"{sarif_path}: {MASS_RULE_ID} has no raw_value")
            return float(raw)
    raise ValueError(f"{sarif_path}: no {MASS_RULE_ID} result")


def _d0_24_v2_band(gap_pct: float) -> str:
    """D0-24 v2 band on the MGN/GT gap as a percentage of GT."""
    magnitude = abs(gap_pct)
    if magnitude <= 20.0:
        return "PASS"
    if magnitude <= 50.0:
        return "MARGINAL"
    return "FAIL"


def tabulate(results_dir: Path = RESULTS_DIR, trajectories: list[int] | None = None) -> dict:
    """Build the per-trajectory + summary report from the SARIF files."""
    trajectories = list(TRAJECTORIES if trajectories is None else trajectories)
    rows = []
    for t in trajectories:
        gt = _mass_defect(results_dir / f"traj{t}_gt.sarif")
        mgn = _mass_defect(results_dir / f"traj{t}_mgn.sarif")
        gap_abs = mgn - gt
        gap_pct = 100.0 * gap_abs / gt
        rows.append(
            {
                "traj_idx": t,
                "gt_raw_value": gt,
                "mgn_raw_value": mgn,
                "gap_abs": gap_abs,
                "gap_pct_of_gt": gap_pct,
                "gap_within_fe_floor": abs(gap_abs) < gt,
                "d0_24_v2_band": _d0_24_v2_band(gap_pct),
            }
        )
    gaps = [r["gap_pct_of_gt"] for r in rows]
    summary = {
        "n_trajectories": len(rows),
        "median_gap_pct": statistics.median(gaps),
        "min_gap_pct": min(gaps),
        "max_gap_pct": max(gaps),
        "median_gt_raw_value": statistics.median([r["gt_raw_value"] for r in rows]),
        "median_mgn_raw_value": statistics.median([r["mgn_raw_value"] for r in rows]),
        "all_gaps_within_fe_floor": all(r["gap_within_fe_floor"] for r in rows),
        "all_d0_24_v2_pass": all(r["d0_24_v2_band"] == "PASS" for r in rows),
    }
    return {"per_trajectory": rows, "summary": summary}


def markdown_table(report: dict) -> str:
    """Render the per-trajectory table + summary line for the CS02 README."""
    lines = [
        "| `traj_idx` | GT (FE-on-P1 floor) | MGN | gap (% of GT) | D0-24 v2 band |",
        "|---|---|---|---|---|",
    ]
    for r in report["per_trajectory"]:
        lines.append(
            f"| {r['traj_idx']} | {r['gt_raw_value']:.3e} | {r['mgn_raw_value']:.3e} "
            f"| {r['gap_pct_of_gt']:+.2f} % | {r['d0_24_v2_band']} |"
        )
    s = report["summary"]
    lines.append("")
    lines.append(
        f"Median gap = {s['median_gap_pct']:+.2f} % of GT; "
        f"range [{s['min_gap_pct']:+.2f} %, {s['max_gap_pct']:+.2f} %] "
        f"across N = {s['n_trajectories']} trajectories."
    )
    return "\n".join(lines)


def main() -> None:
    report = tabulate()
    out_path = RESULTS_DIR / "results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(markdown_table(report))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
