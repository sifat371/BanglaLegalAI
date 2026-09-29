from src.chains.citation_verifier import CitationVerifier


def sources():
    return [
        {
            "source_id": "S1",
            "type": "judgment",
            "citation": "Criminal Appeal No. 1 of 2024, High Court Division, p. 7",
            "document_id": "doc1",
            "chunk_id": "doc1:p7:c0",
            "page": "7",
        },
        {
            "source_id": "S2",
            "type": "act",
            "citation": "The Evidence Act, 1872, Section 45",
        },
    ]


def test_valid_citations_bind_to_supplied_sources():
    verifier = CitationVerifier()

    result = verifier.verify(
        "The judgment discusses the issue [S1]. The statute is also relevant [S2].",
        sources(),
    )

    assert result["status"] == "verified"
    assert result["structurally_verified"] is True
    assert result["semantic_support_verified"] is False
    assert result["cited_source_ids"] == ["S1", "S2"]
    assert result["invalid_source_ids"] == []
    assert result["references"][0]["chunk_id"] == "doc1:p7:c0"


def test_unknown_source_marker_fails_closed():
    result = CitationVerifier().verify("Unsupported citation [S9].", sources())

    assert result["status"] == "failed"
    assert result["invalid_source_ids"] == ["S9"]
    assert result["structurally_verified"] is False


def test_noncanonical_source_marker_is_rejected():
    result = CitationVerifier().verify("Legacy marker [Source 1].", sources())

    assert result["status"] == "failed"
    assert result["noncanonical_markers"] == ["[Source 1]"]


def test_sources_without_answer_citations_are_uncited():
    result = CitationVerifier().verify("An answer without markers.", sources())

    assert result["status"] == "uncited"
    assert result["citation_count"] == 0


def test_no_sources_and_no_citations_is_not_applicable():
    result = CitationVerifier().verify("No relevant material was retrieved.", [])

    assert result["status"] == "not_applicable"


def test_annotate_sources_marks_only_referenced_sources():
    verifier = CitationVerifier()
    available = sources()
    verification = verifier.verify("Only the case is used [S1].", available)

    annotated = verifier.annotate_sources(available, verification)

    assert annotated[0]["cited"] is True
    assert annotated[1]["cited"] is False


def test_compound_noncanonical_marker_is_rejected():
    result = CitationVerifier().verify("Combined marker [S1, S2].", sources())

    assert result["status"] == "failed"
    assert result["noncanonical_markers"] == ["[S1, S2]"]
