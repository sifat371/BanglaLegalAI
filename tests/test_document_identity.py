from langchain_core.documents import Document

from src.vectorstore.document_identity import get_document_id


def test_ingested_chunk_id_is_authoritative():
    doc = Document(
        page_content="A judgment passage",
        metadata={
            "source_type": "judgment",
            "chunk_id": "abc123:p7:c2",
            "document_id": "abc123",
        },
    )

    assert get_document_id(doc) == "abc123:p7:c2"


def test_fallback_identity_is_deterministic():
    first = Document(page_content="same text", metadata={"source_type": "unknown"})
    second = Document(page_content="same text", metadata={"source_type": "unknown"})

    assert get_document_id(first) == get_document_id(second)
