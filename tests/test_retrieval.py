import os

os.environ.setdefault("MISTRAL_API_KEY", "test-key")
os.environ.setdefault("HUGGINGFACE_API_KEY", "test-key")

from langchain_core.documents import Document

from src.chains.retrieval_chain import RetrievalChain


def make_chain_without_backends() -> RetrievalChain:
    chain = RetrievalChain.__new__(RetrievalChain)
    chain.alpha = 0.7
    return chain


def test_case_search_routes_to_shared_case_index():
    chain = make_chain_without_backends()
    plan = chain._plan_retrieval(
        {
            "intent": "CASE_SEARCH",
            "search_strategy": "HYBRID",
            "metadata_filters": {"source_type": "case"},
            "specific_references": {},
            "reformulated_query": "property appeal",
        },
        k=5,
        user_type="public",
    )

    assert plan["stores"] == ["cases"]
    assert plan["metadata_filters"] is None


def test_judgment_source_keeps_page_and_provenance_ids():
    chain = make_chain_without_backends()
    document = Document(
        page_content="A cited passage",
        metadata={
            "source_type": "judgment",
            "case_number": "Civil Revision No. 205 of 2021",
            "court": "Supreme Court of Bangladesh",
            "page_start": 7,
            "document_id": "docsha",
            "chunk_id": "docsha:p7:c0",
            "source_filename": "civil_revision.pdf",
        },
    )

    source = chain._extract_sources([document])[0]

    assert source["type"] == "judgment"
    assert source["page"] == 7
    assert source["document_id"] == "docsha"
    assert source["chunk_id"] == "docsha:p7:c0"
    assert source["citation"] == (
        "Civil Revision No. 205 of 2021, Supreme Court of Bangladesh, p. 7"
    )


def test_range_filter_is_not_wrapped_in_equality():
    chain = make_chain_without_backends()

    converted = chain._convert_to_chromadb_filter(
        {
            "source_type": "act",
            "act_year": {"$gte": 1950, "$lte": 1980},
        }
    )

    assert converted == {"act_year": {"$gte": 1950, "$lte": 1980}}
