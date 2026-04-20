"""Citation dataclass validation tests."""

from __future__ import annotations

import pytest

from external_validation._harness.citations import Citation


def _base_citation_kwargs() -> dict:
    return dict(
        paper_title="Test paper",
        authors="Author, A.",
        venue="Test venue",
        arxiv_id="1234.56789",
        doi=None,
        isbn=None,
        url=None,
        section="§1.1",
        artifact="Theorem 1",
        pinned_value="0.5 ± 0.1",
        verification_date="2026-04-20",
        verification_protocol="analytical derivation",
    )


def test_citation_constructs_with_minimal_valid_fields():
    c = Citation(**_base_citation_kwargs())
    assert c.paper_title == "Test paper"


def test_citation_rejects_empty_required_string_fields():
    kw = _base_citation_kwargs()
    kw["paper_title"] = ""
    with pytest.raises(ValueError, match="paper_title"):
        Citation(**kw)


def test_citation_requires_at_least_one_identifier():
    kw = _base_citation_kwargs()
    kw["arxiv_id"] = None
    kw["doi"] = None
    kw["isbn"] = None
    kw["url"] = None
    with pytest.raises(ValueError, match="at least one"):
        Citation(**kw)


def test_citation_accepts_isbn_only_book():
    kw = _base_citation_kwargs()
    kw["arxiv_id"] = None
    kw["isbn"] = "978-0-8218-4974-3"
    c = Citation(**kw)
    assert c.isbn == "978-0-8218-4974-3"


def test_citation_accepts_url_only_source():
    kw = _base_citation_kwargs()
    kw["arxiv_id"] = None
    kw["url"] = "https://www.osti.gov/biblio/759450"
    c = Citation(**kw)
    assert c.url == "https://www.osti.gov/biblio/759450"


def test_citation_rejects_bad_verification_date():
    kw = _base_citation_kwargs()
    kw["verification_date"] = "April 20, 2026"
    with pytest.raises(ValueError, match="ISO 8601"):
        Citation(**kw)


def test_citation_accepts_both_arxiv_and_doi():
    # Hansen 2024, Bachmayr-Dahmen-Oster 2025: journal articles with both.
    kw = _base_citation_kwargs()
    kw["doi"] = "10.1016/j.physd.2023.133952"
    # arxiv_id already set in base kwargs
    c = Citation(**kw)
    assert c.arxiv_id is not None and c.doi is not None
