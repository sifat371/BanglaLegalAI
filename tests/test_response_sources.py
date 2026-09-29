import os

os.environ.setdefault("MISTRAL_API_KEY", "test-key")
os.environ.setdefault("HUGGINGFACE_API_KEY", "test-key")

from langchain_core.documents import Document

from src.chains.response_chain import ResponseChain


def test_response_source_for_judgment_is_page_grounded():
    chain = ResponseChain.__new__(ResponseChain)
    document = Document(
        page_content="Grounded passage",
        metadata={
            "source_type": "judgment",
            "case_number": "Death Reference No. 106 of 2018",
            "court": "High Court Division",
            "page_start": 12,
            "document_id": "sha",
            "chunk_id": "sha:p12:c1",
            "source_filename": "death_reference.pdf",
        },
    )

    source = chain._format_sources([document])[0]

    assert source["type"] == "judgment"
    assert source["page"] == "12"
    assert source["citation"] == (
        "Death Reference No. 106 of 2018, High Court Division, p. 12"
    )


def test_response_context_exposes_judgment_page():
    chain = ResponseChain.__new__(ResponseChain)
    document = Document(
        page_content="Grounded passage",
        metadata={
            "source_type": "judgment",
            "case_number": "Writ Petition No. 42 of 2024",
            "court": "High Court Division",
            "page_start": 3,
            "source_filename": "writ.pdf",
        },
    )

    context = chain._format_documents([document])

    assert "Writ Petition No. 42 of 2024" in context
    assert "Page: 3" in context
    assert "Grounded passage" in context
