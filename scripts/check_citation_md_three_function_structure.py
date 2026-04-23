"""Enforce §1.3 three-function-labeled structure in per-rule CITATION.md files.

Per complete-v1.0 plan §1.3:

- Every external-validation rule's CITATION.md must contain a
  "## Function-labeled citation stack" section with three subsections:
  "### Mathematical-legitimacy", "### Correctness-fixture", "### Borrowed-credibility".
- The Borrowed-credibility subsection must be either populated with at least
  one identifier-bearing citation (arxiv/doi/isbn/url) or contain an explicit
  "F3 absent" / "absent with justification" / "Function 3: absent" marker.
- An optional "### Supplementary calibration context" subsection is permitted
  under the Function-labeled citation stack section for calibration-only
  references that do not qualify for F3.

Usage:

    python scripts/check_citation_md_three_function_structure.py <path> [path ...]

Exit code 0 on full compliance across all provided paths; exit code 1 on any
violation (violations printed to stderr).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REQUIRED_SECTION = "## Function-labeled citation stack"
_REQUIRED_SUBSECTIONS = (
    "### Mathematical-legitimacy",
    "### Correctness-fixture",
    "### Borrowed-credibility",
)
_OPTIONAL_SUBSECTION = "### Supplementary calibration context"

_F3_ABSENT_MARKERS = (
    "F3 absent",
    "F3: absent",
    "Function 3 absent",
    "Function 3: absent",
    "F3-absent",
    "absent with justification",
    "absent-with-justification",
    "absent by structure",
)

_IDENTIFIER_PATTERNS = (
    re.compile(r"arXiv:\d{4}\.\d{4,5}", re.IGNORECASE),
    re.compile(r"arxiv\.org/abs/\d{4}\.\d{4,5}", re.IGNORECASE),
    re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE),
    re.compile(r"978[- ]?\d[- ]?\d{3}[- ]?\d{5}[- ]?\d", re.IGNORECASE),
    re.compile(r"https?://\S+", re.IGNORECASE),
)


def _extract_section(text: str, heading: str) -> str | None:
    """Return the text under `heading` up to the next equal-or-higher heading.

    Returns None if `heading` is not present.
    """
    lines = text.splitlines()
    start = None
    heading_level = heading.count("#")
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i + 1
            break
    if start is None:
        return None
    end = len(lines)
    for i in range(start, len(lines)):
        line = lines[i]
        stripped = line.lstrip("#")
        level = len(line) - len(stripped)
        if 0 < level <= heading_level and stripped.startswith(" "):
            end = i
            break
    return "\n".join(lines[start:end])


def check_file(path: Path) -> list[str]:
    """Return a list of violation messages; empty list means compliant."""
    violations: list[str] = []
    text = path.read_text(encoding="utf-8")

    if _REQUIRED_SECTION not in text:
        violations.append(f"{path}: missing top-level section `{_REQUIRED_SECTION}`")
        return violations

    for subsection in _REQUIRED_SUBSECTIONS:
        if subsection not in text:
            violations.append(f"{path}: missing subsection `{subsection}`")

    bc_body = _extract_section(text, "### Borrowed-credibility")
    if bc_body is not None:
        has_citation = any(p.search(bc_body) for p in _IDENTIFIER_PATTERNS)
        has_absent_marker = any(m in bc_body for m in _F3_ABSENT_MARKERS)
        if not has_citation and not has_absent_marker:
            violations.append(
                f"{path}: Borrowed-credibility subsection must contain at least "
                "one identifier-bearing citation (arxiv / doi / isbn / url) OR "
                f"an explicit F3-absent marker (one of: {_F3_ABSENT_MARKERS})"
            )

    return violations


def main(argv: list[str]) -> int:
    if len(argv) <= 1:
        print(__doc__, file=sys.stderr)
        return 2
    paths = [Path(a) for a in argv[1:]]
    any_violation = False
    for path in paths:
        if not path.is_file():
            print(f"ERROR: {path} does not exist", file=sys.stderr)
            any_violation = True
            continue
        violations = check_file(path)
        if violations:
            any_violation = True
            for v in violations:
                print(v, file=sys.stderr)
        else:
            print(f"OK: {path}")
    return 1 if any_violation else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
