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
