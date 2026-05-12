"""eps(t) npz dir -> HarnessResult rows (rung 4b consumer).

Sibling to lint_npz_dir.py; reads eps(t) npzs (SCHEMA.md §1.5) and emits
SARIF v1.1 result rows. Single artifact tier (uniform schema for both
T_steps=1 and T_steps=100 npzs) per design §3.4.

Per design §3.6 trigger-vs-emission separation, the SO(2) substrate
trigger has already fired upstream in symmetry_rollout_adapter.py;
this module reads `transform_kind == "skip"` from the npz and emits
the skip_reason via the shared D0-19 §3.4 emission machinery (same
shape as PH-CON-002 dissipative SKIP).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from external_validation._rollout_anchors._harness.sarif_emitter import HarnessResult
from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
    read_eps_t_npz,
)

# Rung-4b design §3.3 + writeup §6 item 3 thresholds. Band → SARIF level:
#   PASS         (ε ≤ EPS_PASS_THRESHOLD)                                 → note
#   APPROXIMATE  (EPS_PASS_THRESHOLD < ε ≤ EPS_APPROXIMATE_THRESHOLD)     → warning
#   FAIL         (ε > EPS_APPROXIMATE_THRESHOLD)                          → error
#
# Pinned against design §3.3 by test_lint_eps_dir_band_thresholds_match_design;
# any future amendment to these values must update the design doc in lockstep.
# Landed at round-code-1 (2026-05-12); pre-round-code-1, level was hardcoded
# to "note" for every row regardless of eps band — the rung-4b writeup §6
# item 3 forward-flag had a writeup-vs-code drift that round-code-1 absorbs.
EPS_PASS_THRESHOLD = 1e-5
EPS_APPROXIMATE_THRESHOLD = 1e-2


def _band_to_level(eps: float) -> str:
    """Map active-row eps to SARIF level per rung-4b design §3.3 bands.

    SKIP rows have no eps to classify and keep level="note" at the call site
    (lint_eps_dir), where the skip_reason carries the methodology signal.
    """
    if eps <= EPS_PASS_THRESHOLD:
        return "note"
    if eps <= EPS_APPROXIMATE_THRESHOLD:
        return "warning"
    return "error"


class EmptyEpsDirectoryError(Exception):
    """Raised when lint_eps_dir is invoked on a directory with no eps_*.npz files.

    Silent empty SARIF is a methodology hazard. Same fail-loud pattern as
    rung 4a's EmptyNpzDirectoryError in lint_npz_dir.py.
    """


def lint_eps_dir(
    *,
    eps_dir: Path | str,
    case_study: str,
    dataset: str,
    model: str,
    ckpt_hash: str,
) -> list[HarnessResult]:
    """Read all eps_*.npz files from `eps_dir`, emit one HarnessResult per file.

    Active rows: raw_value = eps_t[0] (first-step eps); message includes
    the eps_pos_rms scalar and transform_param.

    SKIP rows (PH-SYM-003): raw_value = None; message = "SKIP:
    <skip_reason>"; extra_properties.skip_reason populated per
    D0-19 §3.4.
    """
    eps_dir = Path(eps_dir)
    npz_paths = sorted(eps_dir.glob("eps_*.npz"))
    if not npz_paths:
        raise EmptyEpsDirectoryError(
            f"No eps_*.npz files found in {eps_dir}. "
            "Expected at least one eps(t) npz; run the Modal entrypoint to populate."
        )

    results: list[HarnessResult] = []
    for npz_path in npz_paths:
        record = read_eps_t_npz(npz_path)
        rule_id = record["rule_id"]
        transform_kind = record["transform_kind"]

        extra: dict[str, Any] = {
            "transform_kind": transform_kind,
            "transform_param": record["transform_param"],
            "traj_index": record["traj_index"],
            "eps_t_npz_filename": npz_path.name,
        }

        level: str = "note"
        raw_value: float | None
        if transform_kind == "skip":
            extra["skip_reason"] = record["skip_reason"] or "(no reason)"
            extra["eps_pos_rms"] = None
            message = f"SKIP: {record['skip_reason'] or '(no reason)'}"
            raw_value = None
        else:
            eps_first = float(record["eps_t"][0])
            extra["eps_pos_rms"] = eps_first
            message = (
                f"eps_pos_rms={eps_first:.3e} "
                f"(transform={transform_kind} {record['transform_param']})"
            )
            raw_value = eps_first
            level = _band_to_level(eps_first)

        results.append(
            HarnessResult(
                rule_id=rule_id,
                level=level,  # type: ignore[arg-type]
                message=message,
                raw_value=raw_value,
                case_study=case_study,
                dataset=dataset,
                model=model,
                ckpt_hash=ckpt_hash,
                extra_properties=extra,
            )
        )
    return results
