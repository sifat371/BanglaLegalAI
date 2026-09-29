"""Stable document identity helpers shared by indexing and retrieval."""

from __future__ import annotations

from hashlib import sha256

from langchain_core.documents import Document


def _content_digest(text: str, length: int = 20) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:length]


def get_document_id(document: Document) -> str:
    """Return a deterministic identifier for a LangChain document.

    BanglaLegalIngest chunk IDs are authoritative for ingested judgments.
    Existing act/case-study data receives deterministic compatibility IDs.
    """
    metadata = document.metadata

    chunk_id = str(metadata.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id

    source_type = str(metadata.get("source_type") or "").strip()

    if source_type == "act":
        act_no = str(metadata.get("act_no") or "").strip()
        act_year = str(metadata.get("act_year") or "").strip()
        section_number = str(metadata.get("section_number") or "").strip()
        chunk_index = str(metadata.get("chunk_index", 0))
        stable_key = "|".join(
            [act_no, act_year, section_number, chunk_index, document.page_content]
        )
        return f"act:{_content_digest(stable_key)}"

    case_id = str(
        metadata.get("case_id")
        or metadata.get("case_number")
        or metadata.get("document_id")
        or ""
    ).strip()
    if case_id:
        return f"case:{_content_digest(case_id + '|' + document.page_content)}"

    return f"doc:{_content_digest(document.page_content)}"
