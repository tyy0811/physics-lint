"""Tests for the canonical-section extraction in docs/sphinx/conf.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "docs" / "sphinx"))

from conf import extract_canonical_section  # type: ignore[import-not-found]  # noqa: E402


def test_extracts_rule_reference_section(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# PH-XXX-NNN\n"
        "\n"
        "Intro paragraph.\n"
        "\n"
        "## Rule reference\n"
        "\n"
        "Body paragraph 1.\n"
        "\n"
        "Body paragraph 2.\n"
        "\n"
        "## Validation-side content\n"
        "\n"
        "Should NOT appear in extracted output.\n"
    )
    extracted = extract_canonical_section(readme, heading="## Rule reference")
    assert "Body paragraph 1." in extracted
    assert "Body paragraph 2." in extracted
    assert "Should NOT appear" not in extracted
    assert "Intro paragraph" not in extracted


def test_extracts_section_at_end_of_file(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# title\n\n## Rule reference\n\nBody at EOF.\n")
    extracted = extract_canonical_section(readme, heading="## Rule reference")
    assert "Body at EOF." in extracted


def test_missing_canonical_section_raises(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# title\n\nSome other content without the canonical heading.\n")
    with pytest.raises(RuntimeError, match="missing canonical section"):
        extract_canonical_section(readme, heading="## Rule reference")


def test_extracts_case_study_reference(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Case Study Title\n"
        "\n"
        "## Case study reference\n"
        "\n"
        "Case-study body.\n"
        "\n"
        "## Internal validation details\n"
        "\n"
        "Internal content.\n"
    )
    extracted = extract_canonical_section(readme, heading="## Case study reference")
    assert "Case-study body." in extracted
    assert "Internal content." not in extracted
