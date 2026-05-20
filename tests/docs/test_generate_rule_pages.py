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


def test_ignores_h2_inside_code_fence(tmp_path: Path) -> None:
    """A `## ` line inside a fenced code block must not terminate extraction."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "# title\n"
        "\n"
        "## Rule reference\n"
        "\n"
        "First paragraph.\n"
        "\n"
        "Example markdown to author in your README:\n"
        "\n"
        "```markdown\n"
        "## Rule reference\n"
        "\n"
        "Body content here.\n"
        "```\n"
        "\n"
        "Closing paragraph after the fence.\n"
        "\n"
        "## Validation harness\n"
        "\n"
        "Should not appear.\n"
    )
    extracted = extract_canonical_section(readme, heading="## Rule reference")
    assert "First paragraph." in extracted
    assert "Closing paragraph after the fence." in extracted
    assert "Should not appear" not in extracted


def test_ignores_h2_inside_tilde_fence(tmp_path: Path) -> None:
    """Same as code-fence test but with tilde-delimited fence."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "# title\n"
        "\n"
        "## Rule reference\n"
        "\n"
        "Before fence.\n"
        "\n"
        "~~~markdown\n"
        "## Inside fence — must not split\n"
        "~~~\n"
        "\n"
        "After fence.\n"
        "\n"
        "## Validation harness\n"
        "\n"
        "Not included.\n"
    )
    extracted = extract_canonical_section(readme, heading="## Rule reference")
    assert "Before fence." in extracted
    assert "After fence." in extracted
    assert "Not included" not in extracted


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
