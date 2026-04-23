"""Citation dataclass for external-validation anchors.

See design spec §1.3 and complete-v1.0 plan §1.3 (three-function labeling).
Validation runs in `__post_init__`:
- All required string fields must be non-empty.
- At least one of {arxiv_id, doi, isbn, url} must be non-None.
- `verification_date` must be valid ISO 8601 (YYYY-MM-DD).
- `function_tag` (optional) must be one of the four accepted labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

_REQUIRED_NONEMPTY = (
    "paper_title",
    "authors",
    "venue",
    "section",
    "artifact",
    "pinned_value",
    "verification_date",
    "verification_protocol",
)

_IDENTIFIER_FIELDS = ("arxiv_id", "doi", "isbn", "url")

FUNCTION_TAGS = frozenset(
    {
        "mathematical_legitimacy",
        "correctness_fixture",
        "borrowed_credibility",
        "supplementary_calibration_context",
    }
)


@dataclass
class Citation:
    paper_title: str
    authors: str
    venue: str
    arxiv_id: str | None
    doi: str | None
    isbn: str | None
    url: str | None
    section: str
    artifact: str
    pinned_value: str
    verification_date: str
    verification_protocol: str
    function_tag: str | None = None

    def __post_init__(self) -> None:
        for field_name in _REQUIRED_NONEMPTY:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Citation.{field_name} must be a non-empty string; got {value!r}")

        if not any(getattr(self, f) for f in _IDENTIFIER_FIELDS):
            raise ValueError(
                f"Citation requires at least one of {_IDENTIFIER_FIELDS} to be non-None"
            )

        try:
            date.fromisoformat(self.verification_date)
        except ValueError as exc:
            raise ValueError(
                f"Citation.verification_date must be ISO 8601 (YYYY-MM-DD); "
                f"got {self.verification_date!r}"
            ) from exc

        if self.function_tag is not None and self.function_tag not in FUNCTION_TAGS:
            raise ValueError(
                f"Citation.function_tag must be one of {sorted(FUNCTION_TAGS)} or None; "
                f"got {self.function_tag!r}"
            )
