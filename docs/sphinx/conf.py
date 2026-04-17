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


def generate_rule_pages(app):
    """Walk the lazy registry and write one .md file per rule on build start."""
    import importlib

    from physics_lint.rules import _registry

    out = Path(__file__).parent / "rules"
    out.mkdir(exist_ok=True)

    entries = _registry.list_rules()
    index_lines = [
        "# Rule Catalog",
        "",
        "```{toctree}",
        ":hidden:",
        ":maxdepth: 1",
        "",
    ]
    index_lines.extend(entry.rule_id for entry in entries)
    index_lines.extend(
        [
            "```",
            "",
            "| Rule | Name | Severity | Input modes |",
            "|------|------|----------|-------------|",
        ]
    )
    for entry in entries:
        modes = "+".join(sorted(entry.input_modes))
        index_lines.append(
            f"| [{entry.rule_id}]({entry.rule_id}.md) | {entry.rule_name} | "
            f"{entry.default_severity} | {modes} |"
        )
        module = importlib.import_module(entry.module_name)
        doc = module.__doc__ or "(no docstring provided)"
        page = [
            f"# {entry.rule_id}",
            "",
            f"**Name:** {entry.rule_name}",
            "",
            f"**Severity:** {entry.default_severity}",
            "",
            f"**Input modes:** {sorted(entry.input_modes)}",
            "",
            doc,
        ]
        (out / f"{entry.rule_id}.md").write_text("\n".join(page))
    (out / "index.md").write_text("\n".join(index_lines))


def setup(app):
    app.connect("builder-inited", generate_rule_pages)
