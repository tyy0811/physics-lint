"""Enforce §6.4 textbook-primary-source-verification framing discipline.

Per complete-v1.0 plan §6.4: a CITATION.md citation of a textbook whose
verification status in `external_validation/_harness/TEXTBOOK_AVAILABILITY.md`
is ⚠ (secondary-source-confirmed only) must use section-level framing
("Hall 2015 §2.5, theorem number pending local copy") rather than tight
theorem-number framing ("Hall 2015 Theorem 2.14"). The script reads the
availability file's summary table, extracts per-textbook status, and
scans the target CITATION.md file(s) for tight-theorem-number patterns
that cite ⚠-flagged textbooks.

Usage:

    python scripts/check_theorem_number_framing.py <path> [path ...]

Exit code 0 on full compliance; exit code 1 on any framing violation.

Optional env var `PHYSICS_LINT_TEXTBOOK_AVAILABILITY_PATH` overrides the
default availability-file path (useful for tests).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_DEFAULT_AVAILABILITY_PATH = (
    Path(__file__).resolve().parent.parent
    / "external_validation"
    / "_harness"
    / "TEXTBOOK_AVAILABILITY.md"
)

_STATUS_GLYPHS = {"OK": "✅", "WARN": "⚠", "FAIL": "❌", "PEND": "\U0001f17f"}
# Note: the PEND glyph 🅿 is U+1F17F (NEGATIVE SQUARED LATIN CAPITAL LETTER P).

_ROW_RE = re.compile(r"^\|\s*(?P<textbook>[^|]+?)\s*\|\s*(?P<status>[^|]+?)\s*\|")

_TIGHT_THEOREM_RE = re.compile(
    r"\b(Theorem|Thm\.?|Corollary|Cor\.?|Proposition|Prop\.?|Lemma)"
    r"\s+\d+(?:\.\d+)*\b"
)

_SECTION_LEVEL_HINTS = (
    "theorem number pending",
    "pending local copy",
    "section-level",
    "secondary-source",
    "secondary source",
)


def _short_key(textbook_cell: str) -> str:
    """Extract a short identifier like `Hall 2015` or `Evans 2010` from a row."""
    # Normalize: drop markdown emphasis and leading/trailing whitespace.
    cell = textbook_cell.strip()
    cell = re.sub(r"\*+", "", cell)  # drop ** bold markers
    # First three tokens typically carry "Author Year Title..."; we take the first
    # two tokens when the second is a 4-digit year, otherwise first two tokens.
    tokens = cell.split()
    if len(tokens) >= 2 and re.match(r"\d{4}", tokens[1]):
        return f"{tokens[0]} {tokens[1]}"
    # Some author chains have hyphenated surnames; fall back to first two tokens.
    return " ".join(tokens[:2]) if len(tokens) >= 2 else cell


def _row_status(status_cell: str) -> str:
    """Classify the row status. Returns one of {'OK', 'WARN', 'FAIL', 'PEND', 'MIXED', 'UNKNOWN'}."""
    if "MIXED" in status_cell.upper():
        return "MIXED"
    has_ok = _STATUS_GLYPHS["OK"] in status_cell
    has_warn = _STATUS_GLYPHS["WARN"] in status_cell
    has_fail = _STATUS_GLYPHS["FAIL"] in status_cell
    has_pend = _STATUS_GLYPHS["PEND"] in status_cell
    # Rows with both ✅ and ⚠ use MIXED semantics: some theorems verified, others not.
    # For the framing check, we treat MIXED as permissive for ✅-marked theorems — the
    # CITATION.md caller is responsible for mapping each cited theorem to its row status.
    if has_ok and has_warn:
        return "MIXED"
    if has_ok:
        return "OK"
    if has_warn:
        return "WARN"
    if has_fail:
        return "FAIL"
    if has_pend:
        return "PEND"
    return "UNKNOWN"


def parse_availability(path: Path) -> dict[str, str]:
    """Parse TEXTBOOK_AVAILABILITY.md summary table; return {short_key: status}."""
    result: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if set(line) <= set("| -:"):
            continue  # separator row
        m = _ROW_RE.match(line)
        if not m:
            continue
        textbook = m.group("textbook")
        if textbook.lower().startswith("textbook"):
            continue  # header row
        key = _short_key(textbook)
        status = _row_status(m.group("status"))
        result[key] = status
    return result


def check_file(citation_path: Path, status_map: dict[str, str]) -> list[str]:
    """Return a list of violation messages; empty list means compliant."""
    violations: list[str] = []
    text = citation_path.read_text(encoding="utf-8")

    warn_keys = [k for k, v in status_map.items() if v in {"WARN", "FAIL", "PEND"}]
    if not warn_keys:
        return violations

    for match in _TIGHT_THEOREM_RE.finditer(text):
        window_start = max(0, match.start() - 200)
        window = text[window_start : match.end() + 20]
        window_lower = window.lower()
        if any(hint in window_lower for hint in _SECTION_LEVEL_HINTS):
            continue
        for key in warn_keys:
            if key in window:
                violations.append(
                    f"{citation_path}: tight theorem-number framing "
                    f"`{match.group(0)}` near `{key}` (status={status_map[key]} "
                    f"in TEXTBOOK_AVAILABILITY.md); §6.4 requires section-level "
                    f"framing for ⚠/❌/🅿 textbook rows."
                )
                break
    return violations


def main(argv: list[str]) -> int:
    if len(argv) <= 1:
        print(__doc__, file=sys.stderr)
        return 2
    availability_path_env = os.environ.get("PHYSICS_LINT_TEXTBOOK_AVAILABILITY_PATH")
    availability_path = (
        Path(availability_path_env) if availability_path_env else _DEFAULT_AVAILABILITY_PATH
    )
    if not availability_path.is_file():
        print(f"ERROR: availability file not found at {availability_path}", file=sys.stderr)
        return 2
    status_map = parse_availability(availability_path)
    if not status_map:
        print(f"ERROR: no textbook rows parsed from {availability_path}", file=sys.stderr)
        return 2

    paths = [Path(a) for a in argv[1:]]
    any_violation = False
    for path in paths:
        if not path.is_file():
            print(f"ERROR: {path} does not exist", file=sys.stderr)
            any_violation = True
            continue
        violations = check_file(path, status_map)
        if violations:
            any_violation = True
            for v in violations:
                print(v, file=sys.stderr)
        else:
            print(f"OK: {path}")
    return 1 if any_violation else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
