"""Sphinx configuration for physics-lint documentation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from physics_lint import __version__

project = "physics-lint"
author = "tyy0811"
copyright = "2026, tyy0811"
version = __version__
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "amsmath",
    "deflist",
    "dollarmath",
    "colon_fence",
    "smartquotes",
    "fieldlist",
]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

html_theme = "furo"
html_title = f"physics-lint {version}"
html_static_path: list[str] = []


def extract_canonical_section(readme_path: Path, heading: str) -> str:
    """Extract content under a literal canonical heading from a Markdown README.

    Returns content from the line after the matching heading up to the next
    ``## `` heading at the same level (or EOF). The heading match is exact:
    ``heading`` must equal the line, leading/trailing whitespace aside.

    Raises ``RuntimeError`` if the canonical heading is not present in the
    file. This fails the sphinx build by design (inclusion model — missing
    canonical section is a documentation defect, not a silent stub).
    """
    text = readme_path.read_text()
    lines = text.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start_idx = i + 1
            break

    if start_idx is None:
        raise RuntimeError(f"missing canonical section {heading!r} in {readme_path}")

    end_idx = len(lines)
    for j in range(start_idx, len(lines)):
        if lines[j].startswith("## "):
            end_idx = j
            break

    body = "\n".join(lines[start_idx:end_idx])
    return body.strip("\n").strip()


def generate_rule_pages(app):
    """Walk the lazy registry and write one .md file per rule on build start.

    Each rule page is built by extracting the ``## Rule reference`` section
    from the rule's corresponding ``external_validation/<rule_id>/README.md``
    and prepending a registry-derived header block. Missing canonical
    section raises ``RuntimeError``, failing the sphinx build.
    """
    from physics_lint.rules import _registry

    repo_root = Path(__file__).parent.parent.parent
    out = Path(__file__).parent / "rules"
    out.mkdir(exist_ok=True)

    entries = _registry.list_rules()

    by_category: dict[str, list] = {}
    for entry in entries:
        category = entry.rule_id.split("-")[1]
        by_category.setdefault(category, []).append(entry)

    index_lines = [
        "# Rule Catalog",
        "",
        "```{toctree}",
        ":hidden:",
        ":maxdepth: 1",
        "",
    ]
    index_lines.extend(entry.rule_id for entry in entries)
    index_lines.extend(["```", ""])

    for category in sorted(by_category):
        index_lines.append(f"## PH-{category}")
        index_lines.append("")
        index_lines.append("| Rule | Name | Severity | Input modes |")
        index_lines.append("|------|------|----------|-------------|")
        for entry in by_category[category]:
            modes = "+".join(sorted(entry.input_modes))
            index_lines.append(
                f"| [{entry.rule_id}]({entry.rule_id}.md) | "
                f"{entry.rule_name} | {entry.default_severity} | {modes} |"
            )
        index_lines.append("")

    for entry in entries:
        readme = repo_root / "external_validation" / entry.rule_id / "README.md"
        body = extract_canonical_section(readme, heading="## Rule reference")
        modes = "+".join(sorted(entry.input_modes))
        page = (
            f"# {entry.rule_id}\n"
            "\n"
            f"**Name:** {entry.rule_name}\n"
            "\n"
            f"**Severity:** {entry.default_severity}\n"
            "\n"
            f"**Input modes:** {modes}\n"
            "\n"
            f"{body}\n"
        )
        (out / f"{entry.rule_id}.md").write_text(page)

    (out / "index.md").write_text("\n".join(index_lines))


def generate_case_study_pages(app):
    """Emit case-study pages by extracting canonical sections from the
    rollout-anchor READMEs. Missing canonical section raises RuntimeError.
    """
    repo_root = Path(__file__).parent.parent.parent
    out = Path(__file__).parent / "case-studies"
    out.mkdir(exist_ok=True)

    case_studies = [
        ("01-lagrangebench", "Case Study 01 — LagrangeBench"),
        ("02-physicsnemo-mgn", "Case Study 02 — PhysicsNeMo MGN"),
    ]

    for slug, title in case_studies:
        readme = repo_root / "external_validation" / "_rollout_anchors" / slug / "README.md"
        body = extract_canonical_section(readme, heading="## Case study reference")
        page = f"# {title}\n\n{body}\n"
        (out / f"{slug}.md").write_text(page)


def setup(app):
    app.connect("builder-inited", generate_rule_pages)
    app.connect("builder-inited", generate_case_study_pages)
