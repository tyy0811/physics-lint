"""Real-model dogfood for physics-lint Criterion 3 (A1 scope).

See docs/superpowers/specs/2026-04-17-week-2.5-dogfood-a1-design.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from physics_lint.field import GridField
from physics_lint.rules import ph_bc_001, ph_pos_002, ph_res_001
from physics_lint.spec import (
    BCSpec,
    DomainSpec,
    FieldSourceSpec,
    GridDomain,
    SymmetrySpec,
)

# Grid spacing: 64-point endpoint-inclusive unit grid → h = 1/63.
# Matches upstream metrics.py:17. Do NOT change — any other h breaks
# PH-RES-001 / pde_residual quadrature comparability.
H = (1.0 / 63.0, 1.0 / 63.0)


def build_a1_spec() -> DomainSpec:
    """DomainSpec for the Week 2½ A1 configuration.

    64x64 unit square, Laplace, non-homogeneous Dirichlet BCs.
    """
    return DomainSpec(
        pde="laplace",
        grid_shape=(64, 64),
        domain=GridDomain(x=(0.0, 1.0), y=(0.0, 1.0)),
        periodic=False,
        boundary_condition=BCSpec(kind="dirichlet"),
        symmetries=SymmetrySpec(declared=[]),
        field=FieldSourceSpec(type="grid", backend="fd", dump_path="unused"),
    )


def apply_rules_to_prediction(
    *,
    prediction: np.ndarray,
    truth: np.ndarray,
    spec: DomainSpec,
) -> dict[str, float]:
    """Run PH-RES-001, PH-BC-001, PH-POS-002 on one (64, 64) prediction.

    Args:
        prediction: (64, 64) float array — model output for one test problem.
        truth: (64, 64) float array — ground-truth solution; used to build
            boundary_target for PH-BC-001.
        spec: DomainSpec from build_a1_spec().

    Returns:
        Mapping rule_id -> raw_value. Each raw_value is a scalar per §6 of
        the design doc; NaN if the rule returned SKIPPED.
    """
    pred_field = GridField(prediction, h=H, periodic=False, backend="fd")
    truth_field = GridField(truth, h=H, periodic=False, backend="fd")
    boundary_target = truth_field.values_on_boundary()
    pred_boundary = pred_field.values_on_boundary()

    res_result = ph_res_001.check(pred_field, spec)
    bc_result = ph_bc_001.check(pred_field, spec, boundary_target=boundary_target)
    pos_result = ph_pos_002.check(pred_field, spec, boundary_values=pred_boundary)

    def _raw(rr):
        return float(rr.raw_value) if rr.raw_value is not None else float("nan")

    return {
        "PH-RES-001": _raw(res_result),
        "PH-BC-001": _raw(bc_result),
        "PH-POS-002": _raw(pos_result),
    }


def check_ordinal_axis(
    *,
    axis_name: str,
    upstream_ranking: list[str],
    physlint_scores: dict[str, dict[str, float]],
    rule_id: str,
) -> dict:
    """Ordinal ranking match between upstream ranking and physics-lint scores.

    Sorts models by physics-lint raw_value ascending (smallest = best per
    upstream's convention on these three metrics) and compares to upstream's
    ranking.
    """
    physlint_ranking = sorted(
        physlint_scores.keys(),
        key=lambda m: physlint_scores[m][rule_id],
    )
    return {
        "axis": axis_name,
        "mode": "ordinal",
        "upstream": list(upstream_ranking),
        "physlint": physlint_ranking,
        "match": physlint_ranking == list(upstream_ranking),
    }


def check_binary_axis(
    *,
    axis_name: str,
    expected_violators: set[str],
    physlint_scores: dict[str, dict[str, float]],
    rule_id: str,
    threshold: float,
) -> dict:
    """Binary-mode check: which models have raw_value > threshold?

    Used for PH-POS-002 where upstream's max_viol splits into {FNO violates,
    others don't}. Under the threshold-mismatch edge case (§6.5), physics-lint
    may report universal violation — this returns match=False and lets the
    caller record the finding in the report.
    """
    physlint_violators = {m for m, s in physlint_scores.items() if s[rule_id] > threshold}
    return {
        "axis": axis_name,
        "mode": "binary",
        "expected_violators": set(expected_violators),
        "physlint_violators": physlint_violators,
        "match": physlint_violators == set(expected_violators),
    }


def load_predictions(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load predictions and truth from a .npz dumped by _extract_predictions.py.

    Returns:
        (predictions, truth) each shaped (N, 64, 64) float32.
    """
    data = np.load(npz_path)
    predictions = data["predictions"].astype(np.float32)
    truth = data["truth"].astype(np.float32)
    return predictions, truth


def aggregate_over_problems(
    *,
    predictions: np.ndarray,
    truth: np.ndarray,
    spec: DomainSpec,
) -> dict[str, float]:
    """Apply the three rules per problem and return per-rule mean raw_value.

    SKIPPED rules contribute NaN; the mean treats NaN as missing (nanmean).
    Ranking downstream breaks ties deterministically by model-name sort.
    """
    assert predictions.shape == truth.shape
    assert predictions.shape[1:] == (64, 64)

    per_problem = []
    for i in range(predictions.shape[0]):
        scores = apply_rules_to_prediction(
            prediction=predictions[i],
            truth=truth[i],
            spec=spec,
        )
        per_problem.append(scores)

    rule_ids = ["PH-RES-001", "PH-BC-001", "PH-POS-002"]
    return {rid: float(np.nanmean([p[rid] for p in per_problem])) for rid in rule_ids}


def compute_verdict(*, sanity_match: bool, real_axis_matches: list[bool]) -> str:
    """Four-way Criterion 3 verdict per §7 of the design doc.

    Ordering of branches matters: sanity failure dominates (BUG) even if
    real axes match, because L2-vs-L2 disagreement indicates a discretization
    bug to fix, not a Criterion 3 deferral.
    """
    if not sanity_match:
        return "BUG"
    real_passes = sum(1 for m in real_axis_matches if m)
    if real_passes == len(real_axis_matches):
        return "PASS (scoped)"
    if real_passes >= 1:
        return "PASS (scoped, MIXED)"
    return "FAIL"


UPSTREAM_VALUES = {
    "pde_residual": {"unet_regressor": 20.58, "fno": 24.52, "ddpm": 4.22},
    "bc_err": {"unet_regressor": 0.0067, "fno": 0.2088, "ddpm": 0.0014},
    "max_viol": {"unet_regressor": 0.0, "fno": 0.006, "ddpm": 0.0},
}

UPSTREAM_RANKINGS = {
    "pde_residual": ["ddpm", "unet_regressor", "fno"],
    "bc_err": ["ddpm", "unet_regressor", "fno"],
}

EXPECTED_BINARY_VIOLATORS = {"fno"}

POS_002_THRESHOLD = 1e-10

MODELS = [
    # (name, config_rel, checkpoint_rel)
    (
        "unet_regressor",
        "configs/unet_regressor.yaml",
        "experiments/unet_regressor/best.pt",
    ),
    ("fno", "configs/fno.yaml", "experiments/fno/best.pt"),
    (
        "ddpm",
        None,  # resolved at runtime from P1 reproduction
        "experiments/ddpm/best.pt",
    ),
]


@dataclass
class RunContext:
    diffphys_root: Path
    diffphys_python: Path
    diffphys_sha: str
    ddpm_config: str  # relative path that reproduced 4.22 at P1
    ddpm_reproduced: float
    n_samples: int = 300
    n_samples_ddpm: int = 5


def format_report(
    *,
    verdict: str,
    axis_results: list[dict],
    physlint_scores: dict[str, dict[str, float]],
    ddpm_config: str,
    ddpm_reproduced: float,
    upstream_ddpm: float,
    n_samples: int,
    floor_status: str,
    diffphys_sha: str,
) -> str:
    """Produce the dogfood_real_results.md report body.

    Always includes: verdict, per-axis ranking tables, reproduction
    provenance line, floor-status note, threshold-mismatch note if
    PH-POS-002 binary axis shows universal violation.
    """
    lines: list[str] = []
    lines.append("# Week 2½ real-model dogfood — results\n")
    lines.append(f"**Verdict (Criterion 3, scoped):** {verdict}\n")
    lines.append(f"**Pinned diffusion-physics commit:** `{diffphys_sha}`\n")
    lines.append(f"**Test subset:** first {n_samples} of 5000 samples in `test_in.npz`\n")
    lines.append(
        f"**DDPM reproduction provenance:** reproduced `pde_residual.mean = "
        f"{ddpm_reproduced:.3f}` (upstream table records {upstream_ddpm:.2f}) "
        f"using `{ddpm_config}`.\n"
    )
    lines.append(
        f"**Floor status:** {floor_status} (affects pass/warn/fail status "
        "in the per-rule report only; ranking uses raw_value and is "
        "unaffected).\n"
    )
    lines.append("")
    lines.append("## Scores\n")
    lines.append(
        "| Model | PH-RES-001 (L² trap.) | upstream pde_residual (L²) | "
        "PH-BC-001 (L² rel) | upstream bc_err (L¹ abs) | "
        "PH-POS-002 (overshoot mag) | upstream max_viol (count) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in ("ddpm", "unet_regressor", "fno"):
        lines.append(
            f"| {m} | {physlint_scores[m]['PH-RES-001']:.4g} | "
            f"{UPSTREAM_VALUES['pde_residual'][m]:.4g} | "
            f"{physlint_scores[m]['PH-BC-001']:.4g} | "
            f"{UPSTREAM_VALUES['bc_err'][m]:.4g} | "
            f"{physlint_scores[m]['PH-POS-002']:.4g} | "
            f"{UPSTREAM_VALUES['max_viol'][m]:.4g} |"
        )
    lines.append("")
    lines.append("## Axis comparisons\n")
    for ar in axis_results:
        match_str = "MATCH" if ar["match"] else "MISMATCH"
        lines.append(f"### {ar['axis']} ({ar['mode']}): {match_str}\n")
        if ar["mode"] == "ordinal":
            lines.append(f"- Upstream ranking: `{ar['upstream']}`")
            lines.append(f"- Physics-lint ranking: `{ar['physlint']}`")
        else:
            lines.append(f"- Expected violators: `{sorted(ar['expected_violators'])}`")
            lines.append(f"- Physics-lint violators: `{sorted(ar['physlint_violators'])}`")
            if ar["physlint_violators"] == {"unet_regressor", "fno", "ddpm"}:
                lines.append(
                    "- **Threshold-mismatch finding:** physics-lint's PH-POS-002 "
                    "uses a stricter 1e-10 tolerance than upstream's 1e-6. The "
                    "universal-violation outcome here indicates FP32 boundary "
                    "noise registers in physics-lint but not upstream. This is a "
                    "finding, not a bug (design doc §6.5)."
                )
        lines.append("")
    lines.append("## Scope caveats\n")
    lines.append(
        "- **n=3 models:** DPS, ensemble, OT-CFM, improved DDPM, flow-matching "
        "deferred to v1.1 (see `docs/backlog/v1.2.md`)."
    )
    lines.append(
        "- **L² baselines:** both physics-lint PH-RES-001 and upstream "
        "pde_residual compute L² on Dirichlet Laplace. H⁻¹ requires periodicity."
    )
    lines.append(
        "- **Not 1:1 reimplementations:** rules measure quantities in the "
        "same spirit as upstream columns but with different norms, scopes, "
        "or thresholds. The axes are informative comparisons, not sanity "
        "checks of upstream's implementation (except PH-RES-001, which is)."
    )
    lines.append("")
    return "\n".join(lines)


def run_extract(
    ctx: RunContext,
    model_name: str,
    config_rel: str,
    checkpoint_rel: str,
    output_npz: Path,
) -> None:
    """Invoke _extract_predictions.py in the diffphys venv."""
    cmd = [
        str(ctx.diffphys_python),
        "dogfood/_extract_predictions.py",
        "--model-name",
        model_name,
        "--config",
        str(ctx.diffphys_root / config_rel),
        "--checkpoint",
        str(ctx.diffphys_root / checkpoint_rel),
        "--test-npz",
        str(ctx.diffphys_root / "data/test_in.npz"),
        "--max-samples",
        str(ctx.n_samples),
        "--n-samples",
        str(ctx.n_samples_ddpm),
        "--output",
        str(output_npz),
        "--device",
        "cpu",
    ]
    print(f"[extract] {model_name}: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"extraction failed for {model_name} (returncode {result.returncode})")


def main() -> int:
    diffphys_root = (
        Path(os.environ.get("DIFFPHYS_ROOT", Path.home() / "Desktop/diffusion-physics"))
        .expanduser()
        .absolute()
    )
    # .absolute() — not .resolve() — because .venv-diffphys/bin/python is a
    # symlink to the base interpreter. Following it bypasses venv discovery
    # (pyvenv.cfg lookup needs argv[0] inside the venv) and silently runs
    # the subprocess in the wrong environment.
    diffphys_python = (
        Path(os.environ.get("DIFFPHYS_PYTHON", diffphys_root / ".venv-diffphys/bin/python"))
        .expanduser()
        .absolute()
    )
    ddpm_config = os.environ.get("DDPM_CONFIG_REL", "configs/ddpm_phase2.yaml")
    ddpm_reproduced = float(os.environ.get("DDPM_REPRODUCED", "0.0"))
    n_samples = int(os.environ.get("N_SAMPLES", "300"))
    n_samples_ddpm = int(os.environ.get("N_SAMPLES_DDPM", "5"))

    if ddpm_reproduced == 0.0:
        print(
            "FATAL: DDPM_REPRODUCED must be set (from P1 output).",
            file=sys.stderr,
        )
        return 2

    # Guard against silent wrong-env bug: $DIFFPHYS_PYTHON must point into a
    # virtualenv (parent dir contains pyvenv.cfg). Otherwise the subprocess
    # runs in the base interpreter with whatever site-packages happens to be
    # on $PATH, tainting results without warning.
    if not (diffphys_python.parent.parent / "pyvenv.cfg").exists():
        print(
            f"FATAL: DIFFPHYS_PYTHON={diffphys_python} is not inside a venv "
            f"(expected pyvenv.cfg at {diffphys_python.parent.parent}). "
            "Refusing to run subprocess in an unpinned environment.",
            file=sys.stderr,
        )
        return 3

    diffphys_sha = subprocess.check_output(
        ["git", "-C", str(diffphys_root), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()

    ctx = RunContext(
        diffphys_root=diffphys_root,
        diffphys_python=diffphys_python,
        diffphys_sha=diffphys_sha,
        ddpm_config=ddpm_config,
        ddpm_reproduced=ddpm_reproduced,
        n_samples=n_samples,
        n_samples_ddpm=n_samples_ddpm,
    )

    spec = build_a1_spec()

    tmpdir = Path(os.environ.get("DOGFOOD_TMPDIR", "/tmp/dogfood_real")).resolve()
    tmpdir.mkdir(parents=True, exist_ok=True)

    physlint_scores: dict[str, dict[str, float]] = {}
    for model_name, cfg_rel, ckpt_rel in MODELS:
        # For DDPM, cfg_rel is None → use the P1 reproduction result.
        cfg = ddpm_config if model_name == "ddpm" else cfg_rel
        npz_out = tmpdir / f"{model_name}_preds.npz"
        if not npz_out.exists():
            run_extract(ctx, model_name, cfg, ckpt_rel, npz_out)
        predictions, truth = load_predictions(npz_out)
        physlint_scores[model_name] = aggregate_over_problems(
            predictions=predictions,
            truth=truth,
            spec=spec,
        )
        print(
            f"[scores] {model_name}: {physlint_scores[model_name]}",
            file=sys.stderr,
        )

    # Ranking checks.
    sanity = check_ordinal_axis(
        axis_name="pde_residual",
        upstream_ranking=UPSTREAM_RANKINGS["pde_residual"],
        physlint_scores=physlint_scores,
        rule_id="PH-RES-001",
    )
    bc = check_ordinal_axis(
        axis_name="bc_err",
        upstream_ranking=UPSTREAM_RANKINGS["bc_err"],
        physlint_scores=physlint_scores,
        rule_id="PH-BC-001",
    )
    pos = check_binary_axis(
        axis_name="max_viol",
        expected_violators=EXPECTED_BINARY_VIOLATORS,
        physlint_scores=physlint_scores,
        rule_id="PH-POS-002",
        threshold=POS_002_THRESHOLD,
    )

    verdict = compute_verdict(
        sanity_match=sanity["match"],
        real_axis_matches=[bc["match"], pos["match"]],
    )

    report = format_report(
        verdict=verdict,
        axis_results=[sanity, bc, pos],
        physlint_scores=physlint_scores,
        ddpm_config=ddpm_config,
        ddpm_reproduced=ddpm_reproduced,
        upstream_ddpm=4.22,
        n_samples=n_samples,
        floor_status="calibrated (floors.toml entries verified pre-H0)",
        diffphys_sha=diffphys_sha,
    )

    report_path = Path("dogfood/dogfood_real_results.md")
    report_path.write_text(report, encoding="utf-8")
    print(f"[report] wrote {report_path}", file=sys.stderr)
    print(f"[verdict] {verdict}", file=sys.stderr)

    return 0 if verdict in ("PASS (scoped)", "PASS (scoped, MIXED)") else 1


if __name__ == "__main__":
    sys.exit(main())
