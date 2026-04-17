"""Report schema: RuleResult and PhysicsLintReport dataclasses.

Design doc §11. _STATUS_RANK is module-level (not inline) so overall_status
and any future sort/filter operations share one source of truth. SKIPPED
has rank 0 (same as PASS) — a skipped rule never moves overall status,
and status_counts always reports all four keys with explicit zeros.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Status = Literal["PASS", "WARN", "FAIL", "SKIPPED"]
Severity = Literal["error", "warning", "info"]

_STATUS_RANK: dict[Status, int] = {"SKIPPED": 0, "PASS": 0, "WARN": 1, "FAIL": 2}
_STATUS_ORDER: tuple[Status, ...] = ("PASS", "WARN", "FAIL", "SKIPPED")


@dataclass
class RuleResult:
    rule_id: str
    rule_name: str
    severity: Severity
    status: Status
    raw_value: float | None
    violation_ratio: float | None
    mode: str | None
    reason: str | None
    refinement_rate: float | None
    spatial_map: np.ndarray | None
    recommended_norm: str
    citation: str
    doc_url: str


@dataclass
class PhysicsLintReport:
    pde: str
    grid_shape: tuple[int, ...]
    rules: list[RuleResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_status(self) -> Status:
        if not self.rules:
            return "PASS"
        return max(self.rules, key=lambda r: _STATUS_RANK[r.status]).status

    @property
    def status_counts(self) -> dict[Status, int]:
        counts = Counter(r.status for r in self.rules)
        return {s: counts.get(s, 0) for s in _STATUS_ORDER}

    @property
    def exit_code(self) -> int:
        """Non-zero iff any error-severity rule has status FAIL. SKIPPED is ignored."""
        return int(any(r.status == "FAIL" and r.severity == "error" for r in self.rules))

    def summary(self) -> str:
        """Human-readable text summary with status glyphs."""
        lines: list[str] = []
        counts = self.status_counts
        header_grid = "×".join(str(d) for d in self.grid_shape)  # noqa: RUF001 — intentional MULTIPLICATION SIGN
        exit_int = self.exit_code
        overall = self.overall_status
        lines.append(
            f"physics-lint report — {self.pde} on {header_grid} grid — "
            f"overall: {overall} (exit {exit_int})"
        )
        lines.append(
            f"   {counts['FAIL']} fail · {counts['WARN']} warn · "
            f"{counts['PASS']} pass · {counts['SKIPPED']} skip"
        )
        lines.append("")

        glyph_map = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIPPED": "⊘"}
        for r in self.rules:
            glyph = glyph_map.get(r.status, "?")
            mode_tag = f" [{r.mode} mode]" if r.mode else ""
            if r.status == "SKIPPED":
                value_str = r.reason or "skipped"
            elif r.raw_value is None:
                value_str = ""
            else:
                value_str = f"raw={r.raw_value:.3e}"
                if r.violation_ratio is not None:
                    value_str += f" ratio={r.violation_ratio:.2f}"
            lines.append(
                f"  {glyph} {r.rule_id}  {r.status:<7} {r.rule_name}{mode_tag}  {value_str}"
            )
            if r.status in {"FAIL", "WARN"} and r.doc_url:
                lines.append(f"                      → {r.doc_url}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pde": self.pde,
            "grid_shape": list(self.grid_shape),
            "overall_status": self.overall_status,
            "exit_code": self.exit_code,
            "status_counts": self.status_counts,
            "metadata": dict(self.metadata),
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "rule_name": r.rule_name,
                    "severity": r.severity,
                    "status": r.status,
                    "raw_value": r.raw_value,
                    "violation_ratio": r.violation_ratio,
                    "mode": r.mode,
                    "reason": r.reason,
                    "refinement_rate": r.refinement_rate,
                    "recommended_norm": r.recommended_norm,
                    "citation": r.citation,
                    "doc_url": r.doc_url,
                }
                for r in self.rules
            ],
        }

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, default=str)

    def plot(self, figsize: tuple[float, float] = (10, 6)) -> Any:
        """Matplotlib bar chart of violation_ratio per rule, coloured by status."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        labels = [r.rule_id for r in self.rules]
        ratios = [r.violation_ratio or 0.0 for r in self.rules]
        color_map = {
            "PASS": "#55a868",
            "WARN": "#dd8452",
            "FAIL": "#c44e52",
            "SKIPPED": "#8c8c8c",
        }
        colors = [color_map.get(r.status, "#444") for r in self.rules]
        ax.barh(labels, ratios, color=colors)
        ax.set_xlabel("violation_ratio")
        ax.set_title(
            f"physics-lint report — {self.pde} — overall: {self.overall_status} "
            f"(exit {self.exit_code})"
        )
        ax.axvline(1.0, linestyle="--", color="black", alpha=0.4)
        ax.axvline(10.0, linestyle=":", color="orange", alpha=0.4)
        ax.axvline(100.0, linestyle=":", color="red", alpha=0.4)
        plt.tight_layout()
        return fig
