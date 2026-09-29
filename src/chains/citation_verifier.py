"""Deterministic citation-integrity checks for generated legal answers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_CANONICAL_CITATION = re.compile(r"\[S(?P<number>[1-9]\d*)\]")
_SOURCE_LIKE_CITATION = re.compile(
    r"\[(?:Source\s+(?P<source_number>[1-9]\d*)|S(?P<s_number>[1-9]\d*))\]",
    re.IGNORECASE,
)


class CitationVerifier:
    """Verify that answer citation markers bind only to supplied retrieval sources.

    This is a structural integrity check. It does not claim that a cited passage
    semantically entails every sentence that references it.
    """

    @staticmethod
    def source_id(index: int) -> str:
        if index < 1:
            raise ValueError("source index must be >= 1")
        return f"S{index}"

    def verify(self, answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
        available = [str(source.get("source_id") or "") for source in sources]
        available = [source_id for source_id in available if source_id]
        available_set = set(available)

        canonical_ids = [
            self.source_id(int(match.group("number")))
            for match in _CANONICAL_CITATION.finditer(answer)
        ]
        counts = Counter(canonical_ids)
        cited_ids = list(dict.fromkeys(canonical_ids))
        invalid_ids = [source_id for source_id in cited_ids if source_id not in available_set]

        noncanonical_markers = []
        for match in _SOURCE_LIKE_CITATION.finditer(answer):
            marker = match.group(0)
            if not _CANONICAL_CITATION.fullmatch(marker):
                noncanonical_markers.append(marker)
        noncanonical_markers = list(dict.fromkeys(noncanonical_markers))

        if not available:
            status = "not_applicable" if not cited_ids and not noncanonical_markers else "failed"
        elif invalid_ids or noncanonical_markers:
            status = "failed"
        elif not cited_ids:
            status = "uncited"
        else:
            status = "verified"

        cited_valid_ids = [source_id for source_id in cited_ids if source_id in available_set]
        references = []
        source_by_id = {
            str(source.get("source_id")): source
            for source in sources
            if source.get("source_id")
        }
        for source_id in cited_valid_ids:
            source = source_by_id[source_id]
            references.append(
                {
                    "source_id": source_id,
                    "count": counts[source_id],
                    "type": source.get("type", "unknown"),
                    "citation": source.get("citation", "Unknown source"),
                    "document_id": source.get("document_id", ""),
                    "chunk_id": source.get("chunk_id", ""),
                    "page": source.get("page", ""),
                }
            )

        return {
            "status": status,
            "structurally_verified": status == "verified",
            "semantic_support_verified": False,
            "citation_count": len(canonical_ids),
            "available_source_ids": available,
            "cited_source_ids": cited_ids,
            "valid_source_ids": cited_valid_ids,
            "invalid_source_ids": invalid_ids,
            "noncanonical_markers": noncanonical_markers,
            "references": references,
            "message": self._message(status),
        }

    def annotate_sources(
        self,
        sources: list[dict[str, Any]],
        verification: dict[str, Any],
    ) -> list[dict[str, Any]]:
        cited = set(verification.get("valid_source_ids", []))
        return [
            {
                **source,
                "cited": source.get("source_id") in cited,
            }
            for source in sources
        ]

    @staticmethod
    def _message(status: str) -> str:
        messages = {
            "verified": (
                "Every answer citation marker maps to a retrieved source. "
                "This verifies citation integrity, not semantic entailment."
            ),
            "uncited": "Retrieved sources were available, but the answer cited none of them.",
            "failed": (
                "The answer contains an invalid or noncanonical source citation marker."
            ),
            "not_applicable": "No retrieved sources or answer citations were present.",
        }
        return messages[status]
