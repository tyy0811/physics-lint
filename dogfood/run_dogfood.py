"""Run physics-lint's PH-RES-001 against the Week-2 dogfood dumps.

This is the fallback D' harness (see dogfood/laplace_uq_bench/README.md).
The four "surrogates" are the laplace-uq-bench FD solver plus three
synthetic defect variants. Criterion 3 (modified under fallback D) is
"physics-lint produces a ranking table on >= 3 surrogates," and we add
the stronger assertion that the ranking is monotone in defect severity:

    oracle < {coarsened, smoothed} < noisy

If laplace-uq-bench publishes a H^1 ranking of the six trained
surrogates in a future session, this orchestrator can grow an
``EXPECTED_RANKING`` list the same way the plan's Task 9 sketches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from physics_lint.loader import load_target
from physics_lint.rules import ph_res_001

HERE = Path(__file__).parent
LQB = HERE / "laplace_uq_bench"

# Defect severity ordering we expect physics-lint to reproduce.
# Oracle first (best); ties are allowed between the two mid-tier defects.
EXPECTED_SEVERITY_BANDS: list[set[str]] = [
    {"oracle"},
    {"coarsened", "smoothed"},
    {"noisy"},
]


def _run_one(name: str) -> dict[str, Any]:
    path = LQB / f"{name}_pred.npz"
    loaded = load_target(path, cli_overrides={}, toml_path=None)
    result = ph_res_001.check(loaded.field, loaded.spec)
    return {
        "name": name,
        "status": result.status,
        "raw_value": float(result.raw_value) if result.raw_value is not None else float("nan"),
        "violation_ratio": float(result.violation_ratio)
        if result.violation_ratio is not None
        else float("nan"),
        "recommended_norm": result.recommended_norm,
    }


def _ordered(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(results, key=lambda r: r["raw_value"])


def _check_severity_bands(ordering: list[str]) -> tuple[bool, str]:
    """Return (ok, reason). The ranking is valid iff, read left to right,
    every surrogate in band k appears before every surrogate in band k+1.
    Within a band the order is irrelevant.
    """
    position = {name: idx for idx, name in enumerate(ordering)}
    prev_max = -1
    for i, band in enumerate(EXPECTED_SEVERITY_BANDS):
        positions = sorted(position[name] for name in band)
        if positions[0] <= prev_max:
            return False, (
                f"band {i} ({sorted(band)}) overlaps previous band: "
                f"positions {positions} vs prev_max {prev_max}"
            )
        prev_max = positions[-1]
    return True, "ok"


def main() -> int:
    surrogates = ["oracle", "noisy", "coarsened", "smoothed"]
    results = [_run_one(name) for name in surrogates]
    ordered = _ordered(results)

    print("physics-lint PH-RES-001 dogfood — fallback D' (4 synthetic surrogates)")
    print()
    print(f"{'rank':<6}{'surrogate':<14}{'status':<10}{'raw':<14}{'ratio':<14}{'norm'}")
    print("-" * 80)
    for i, r in enumerate(ordered, start=1):
        print(
            f"{i:<6}{r['name']:<14}{r['status']:<10}{r['raw_value']:<14.3e}"
            f"{r['violation_ratio']:<14.2e}{r['recommended_norm']}"
        )

    ranking_order = [r["name"] for r in ordered]
    ok, why = _check_severity_bands(ranking_order)
    print()
    if ok:
        print(f"criterion 3 (fallback D', monotone defect ordering): PASS — {ranking_order}")
    else:
        print(f"criterion 3 (fallback D', monotone defect ordering): FAIL — {why}")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    table = out_dir / "week2-dogfood-table.md"
    lines = [
        "# Week 2 dogfood — fallback D' on laplace-uq-bench",
        "",
        "PH-RES-001 run against four synthetic surrogates built from the",
        "laplace-uq-bench `LaplaceSolver`. Oracle is the FD solve on a",
        "canonical BC; the other three apply a known defect (Gaussian",
        "noise, coarse-grid resolve + bilinear upsample, Gaussian blur)",
        "so there is an analytical severity ranking physics-lint must",
        "reproduce.",
        "",
        "| rank | surrogate | status | raw residual | violation ratio | norm |",
        "|------|-----------|--------|--------------|-----------------|------|",
    ]
    for i, r in enumerate(ordered, start=1):
        lines.append(
            f"| {i} | {r['name']} | {r['status']} | "
            f"{r['raw_value']:.3e} | {r['violation_ratio']:.2e} | {r['recommended_norm']} |"
        )
    lines.append("")
    lines.append(f"**expected severity bands:** {[sorted(b) for b in EXPECTED_SEVERITY_BANDS]}")
    lines.append(f"**physics-lint ranking:** {ranking_order}")
    lines.append(f"**criterion 3:** {'PASS' if ok else 'FAIL'}")
    lines.append("")
    lines.append(
        "Note: every status is FAIL because the PH-RES-001 Laplace L2 floor is "
        "calibrated on the analytical `harmonic_polynomial_square` MMS "
        "(`~1e-12`), and even a mathematically-correct FD Laplace solve on "
        "a 64x64 grid with O(1) Dirichlet BCs has O(h^2) truncation residual "
        "~10 orders of magnitude above that floor. The dogfood criterion is "
        "the ranking, not the pass/fail statuses."
    )
    table.write_text("\n".join(lines) + "\n")
    print(f"wrote {table}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
