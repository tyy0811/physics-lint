"""Tests for scripts/check_theorem_number_framing.py.

Uses synthetic TEXTBOOK_AVAILABILITY.md + CITATION.md fixtures to cover the
pass / fail / section-level-hint paths of the Task 0 closeout gate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "check_theorem_number_framing.py"


_SYNTHETIC_AVAILABILITY = """\
# Textbook availability — synthetic fixture

## Summary table

| Textbook | Status | Sections | Consumer tasks |
|----------|--------|----------|----------------|
| Hall 2015 2nd ed. | ⚠ section-level only | Theorem 2.14 | Tasks 1, 6 |
| Evans 2010 | ✅ §2.2.3 Theorem 4 verified | verified | Tasks 1, 2 |
| Varadarajan 1984 | ⚠ secondary-source only | §2.9-2.10 | Tasks 1, 6 |
| Trefethen 2000 | \U0001f17f pending | pending | Task 3 |
"""


def _run(script_args: list[str], availability_path: Path) -> tuple[int, str, str]:
    env = {
        "PHYSICS_LINT_TEXTBOOK_AVAILABILITY_PATH": str(availability_path),
        "PATH": "/usr/bin:/bin",
    }
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), *script_args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _write_fixture(tmp_path: Path, availability_text: str, citation_text: str) -> tuple[Path, Path]:
    avail = tmp_path / "TEXTBOOK_AVAILABILITY.md"
    avail.write_text(availability_text, encoding="utf-8")
    cit = tmp_path / "CITATION.md"
    cit.write_text(citation_text, encoding="utf-8")
    return avail, cit


def test_tight_framing_on_warn_textbook_fails(tmp_path: Path):
    avail, cit = _write_fixture(
        tmp_path,
        _SYNTHETIC_AVAILABILITY,
        "# Rule\n\nHall 2015 Theorem 2.14 asserts one-parameter subgroups.\n",
    )
    code, _, err = _run([str(cit)], avail)
    assert code == 1
    assert "Hall 2015" in err
    assert "Theorem 2.14" in err


def test_section_level_framing_on_warn_textbook_passes(tmp_path: Path):
    avail, cit = _write_fixture(
        tmp_path,
        _SYNTHETIC_AVAILABILITY,
        "# Rule\n\nHall 2015 §2.5, theorem number pending local copy.\n",
    )
    code, out, err = _run([str(cit)], avail)
    assert code == 0, f"stdout={out!r} stderr={err!r}"


def test_tight_framing_on_ok_textbook_passes(tmp_path: Path):
    avail, cit = _write_fixture(
        tmp_path,
        _SYNTHETIC_AVAILABILITY,
        "# Rule\n\nEvans 2010 §2.2.3 Theorem 4 gives the strong max principle.\n",
    )
    code, out, err = _run([str(cit)], avail)
    assert code == 0, f"stdout={out!r} stderr={err!r}"


def test_tight_framing_with_pending_hint_passes(tmp_path: Path):
    # Secondary-source hint in the surrounding window suppresses the violation.
    text = (
        "# Rule\n\n"
        "Hall 2015 Theorem 2.14 (secondary-source-confirmed; theorem number "
        "pending local copy per TEXTBOOK_AVAILABILITY.md).\n"
    )
    avail, cit = _write_fixture(tmp_path, _SYNTHETIC_AVAILABILITY, text)
    code, out, err = _run([str(cit)], avail)
    assert code == 0, f"stdout={out!r} stderr={err!r}"


def test_tight_framing_on_unknown_textbook_passes(tmp_path: Path):
    # Protter-Weinberger isn't in the availability file; theorem-number citation
    # for textbooks not tracked should not be flagged.
    avail, cit = _write_fixture(
        tmp_path,
        _SYNTHETIC_AVAILABILITY,
        "# Rule\n\nProtter-Weinberger Chapter 2 Theorem 1 cross-reference.\n",
    )
    code, out, err = _run([str(cit)], avail)
    assert code == 0, f"stdout={out!r} stderr={err!r}"


def test_tight_framing_on_pending_textbook_fails(tmp_path: Path):
    avail, cit = _write_fixture(
        tmp_path,
        _SYNTHETIC_AVAILABILITY,
        "# Rule\n\nTrefethen 2000 Theorem 4 establishes spectral accuracy.\n",
    )
    code, _, err = _run([str(cit)], avail)
    assert code == 1
    assert "Trefethen 2000" in err


def test_missing_citation_file_fails(tmp_path: Path):
    avail = tmp_path / "TEXTBOOK_AVAILABILITY.md"
    avail.write_text(_SYNTHETIC_AVAILABILITY, encoding="utf-8")
    code, _, err = _run([str(tmp_path / "nonexistent.md")], avail)
    assert code == 1
    assert "does not exist" in err


def test_missing_availability_file_fails(tmp_path: Path):
    cit = tmp_path / "CITATION.md"
    cit.write_text("# Rule\n\nEvans 2010 Theorem 4.\n", encoding="utf-8")
    avail = tmp_path / "missing.md"  # does not exist
    code, _, err = _run([str(cit)], avail)
    assert code == 2
    assert "availability file not found" in err


def test_script_reports_usage_when_no_args(tmp_path: Path):
    avail = tmp_path / "TEXTBOOK_AVAILABILITY.md"
    avail.write_text(_SYNTHETIC_AVAILABILITY, encoding="utf-8")
    code, _, err = _run([], avail)
    assert code == 2
    # Script docstring should be emitted as usage.
    assert "check_theorem_number_framing" in err or "Usage" in err


@pytest.mark.parametrize(
    "body,expected_ok",
    [
        ("Hall 2015 Corollary 3.50 is the continuous-to-smooth result.", False),
        ("Hall 2015 Proposition 1 appears in §X.Y.", False),
        (
            "Hall 2015 §2.5 one-parameter subgroups (theorem number pending local copy).",
            True,
        ),
    ],
)
def test_framing_patterns_for_varied_claims(tmp_path: Path, body: str, expected_ok: bool):
    avail, cit = _write_fixture(tmp_path, _SYNTHETIC_AVAILABILITY, f"# Rule\n\n{body}\n")
    code, _, _ = _run([str(cit)], avail)
    if expected_ok:
        assert code == 0
    else:
        assert code == 1
