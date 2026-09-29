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

    assert source["source_id"] == "S1"
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

    assert "[S1]" in context
    assert "Writ Petition No. 42 of 2024" in context
    assert "Page: 3" in context
    assert "Grounded passage" in context


def test_generate_response_exposes_citation_verification(monkeypatch):
    from src.chains.citation_verifier import CitationVerifier

    chain = ResponseChain.__new__(ResponseChain)
    chain.citation_verifier = CitationVerifier()
    monkeypatch.setattr(
        chain,
        "_generate_answer",
        lambda query, context: "The court addressed the issue [S1].",
    )

    document = Document(
        page_content="The court addressed the issue.",
        metadata={
            "source_type": "judgment",
            "case_number": "Civil Revision No. 205 of 2021",
            "court": "High Court Division",
            "page_start": 7,
            "document_id": "doc",
            "chunk_id": "doc:p7:c0",
            "source_filename": "case.pdf",
        },
    )

    result = chain.generate_response(
        "What did the court address?",
        [document],
        include_followups=False,
        include_confidence=False,
    )

    assert result["citation_verification"]["status"] == "verified"
    assert result["citation_verification"]["semantic_support_verified"] is False
    assert result["sources"][0]["source_id"] == "S1"
    assert result["sources"][0]["cited"] is True


def test_failed_citation_integrity_caps_confidence_low():
    chain = ResponseChain.__new__(ResponseChain)
    document = Document(page_content="A source", metadata={"source_type": "act"})

    confidence = chain._assess_confidence(
        "Question",
        "A sufficiently long answer that would otherwise avoid the brief-answer downgrade. " * 3,
        [document, document, document],
        citation_verification={"status": "failed"},
    )

    assert confidence["level"] == "LOW"
    assert "Citation integrity check failed" in confidence["reasoning"]


def test_generate_response_exposes_claim_support_verification(monkeypatch):
    from src.chains.citation_verifier import CitationVerifier

    class FakeSupportVerifier:
        def verify(self, answer, documents, sources, citation_verification):
            return {
                "status": "supported",
                "experimental": True,
                "claims_total": 1,
                "claims_evaluated": 1,
                "counts": {"supported": 1, "contradicted": 0, "insufficient": 0},
                "claims": [],
                "truncated": False,
                "independently_validated": False,
                "message": "fixture",
            }

    chain = ResponseChain.__new__(ResponseChain)
    chain.citation_verifier = CitationVerifier()
    chain.claim_support_verifier = FakeSupportVerifier()
    monkeypatch.setattr(
        chain,
        "_generate_answer",
        lambda query, context: "The court addressed the issue [S1].",
    )

    document = Document(
        page_content="The court addressed the issue.",
        metadata={
            "source_type": "judgment",
            "case_number": "Civil Revision No. 205 of 2021",
            "court": "High Court Division",
            "page_start": 7,
            "document_id": "doc",
            "chunk_id": "doc:p7:c0",
            "source_filename": "case.pdf",
        },
    )

    result = chain.generate_response(
        "What did the court address?",
        [document],
        include_followups=False,
        include_confidence=False,
    )

    assert result["claim_support_verification"]["status"] == "supported"
    assert result["claim_support_verification"]["independently_validated"] is False


def test_contradicted_claim_support_caps_confidence_low():
    chain = ResponseChain.__new__(ResponseChain)
    document = Document(page_content="A source", metadata={"source_type": "act"})

    confidence = chain._assess_confidence(
        "Question",
        "A sufficiently long answer that would otherwise be high confidence. " * 3,
        [document, document, document],
        citation_verification={"status": "verified"},
        claim_support_verification={"status": "contradicted"},
    )

    assert confidence["level"] == "LOW"
    assert "assessed as contradicted" in " ".join(confidence["reasoning"])


def _verification_bundle(*, acceptable, citation_status="verified", support_status="supported"):
    return {
        "sources": [
            {
                "source_id": "S1",
                "type": "judgment",
                "citation": "Example case, p. 1",
                "cited": True,
            }
        ],
        "citation_verification": {
            "status": citation_status,
            "invalid_source_ids": [],
            "noncanonical_markers": [],
        },
        "claim_support_verification": {
            "status": support_status,
            "coverage_complete": support_status == "supported",
            "truncated": False,
            "claims": [],
        },
        "decision": {
            "acceptable": acceptable,
            "reasons": [] if acceptable else [f"claim_support:{support_status}"],
        },
    }


def test_failed_candidate_is_repaired_and_reverified(monkeypatch):
    from src.chains.grounding_enforcer import GroundingEnforcer

    chain = ResponseChain.__new__(ResponseChain)
    chain.grounding_enforcer = GroundingEnforcer()
    chain.enable_answer_repair = True
    chain.answer_repair_max_attempts = 1
    chain.fail_closed_on_grounding_failure = True

    checks = iter(
        [
            _verification_bundle(
                acceptable=False,
                support_status="insufficient",
            ),
            _verification_bundle(acceptable=True),
        ]
    )
    monkeypatch.setattr(
        chain,
        "_verify_answer_candidate",
        lambda answer, documents: next(checks),
    )
    monkeypatch.setattr(
        chain,
        "_repair_answer",
        lambda **kwargs: "Repaired source-backed answer [S1].",
    )

    result = chain._finalize_grounded_answer(
        query="Question",
        documents=[Document(page_content="source", metadata={})],
        context="[S1] source",
        initial_answer="Unsupported draft [S1].",
    )

    assert result["answer"] == "Repaired source-backed answer [S1]."
    assert result["grounding_enforcement"]["status"] == "repaired"
    assert result["grounding_enforcement"]["repair_attempts"] == 1


def test_failed_repair_is_blocked_fail_closed(monkeypatch):
    from src.chains.grounding_enforcer import GroundingEnforcer

    chain = ResponseChain.__new__(ResponseChain)
    chain.grounding_enforcer = GroundingEnforcer()
    chain.enable_answer_repair = True
    chain.answer_repair_max_attempts = 1
    chain.fail_closed_on_grounding_failure = True

    failed = _verification_bundle(
        acceptable=False,
        support_status="contradicted",
    )
    monkeypatch.setattr(
        chain,
        "_verify_answer_candidate",
        lambda answer, documents: failed,
    )
    monkeypatch.setattr(
        chain,
        "_repair_answer",
        lambda **kwargs: "Still contradicted [S1].",
    )
    monkeypatch.setattr(
        chain,
        "_format_sources",
        lambda documents: [
            {
                "source_id": "S1",
                "type": "judgment",
                "citation": "Example case, p. 1",
            }
        ],
    )

    result = chain._finalize_grounded_answer(
        query="Question",
        documents=[Document(page_content="source", metadata={})],
        context="[S1] source",
        initial_answer="Bad draft [S1].",
    )

    assert result["grounding_enforcement"]["status"] == "blocked"
    assert "fully source-supported answer" in result["answer"]
    assert result["sources"][0]["cited"] is False


def test_repair_can_explicitly_signal_insufficient_retrieved_support(monkeypatch):
    from src.chains.grounding_enforcer import GroundingEnforcer

    chain = ResponseChain.__new__(ResponseChain)
    chain.grounding_enforcer = GroundingEnforcer()
    chain.enable_answer_repair = True
    chain.answer_repair_max_attempts = 1
    chain.fail_closed_on_grounding_failure = True

    failed = _verification_bundle(
        acceptable=False,
        support_status="insufficient",
    )
    monkeypatch.setattr(
        chain,
        "_verify_answer_candidate",
        lambda answer, documents: failed,
    )
    monkeypatch.setattr(
        chain,
        "_repair_answer",
        lambda **kwargs: "INSUFFICIENT_RETRIEVED_SUPPORT",
    )
    monkeypatch.setattr(
        chain,
        "_format_sources",
        lambda documents: [
            {
                "source_id": "S1",
                "type": "judgment",
                "citation": "Example case, p. 1",
            }
        ],
    )

    result = chain._finalize_grounded_answer(
        query="Question",
        documents=[Document(page_content="source", metadata={})],
        context="[S1] source",
        initial_answer="Too broad [S1].",
    )

    assert result["grounding_enforcement"]["status"] == "blocked"
    assert result["grounding_enforcement"]["attempts"][0]["outcome"] == (
        "insufficient_retrieved_support"
    )


def test_blocked_grounding_forces_confidence_low():
    chain = ResponseChain.__new__(ResponseChain)
    document = Document(page_content="source", metadata={"source_type": "act"})

    confidence = chain._assess_confidence(
        "Question",
        "Fallback text",
        [document, document, document],
        citation_verification={"status": "verified"},
        claim_support_verification={
            "status": "supported",
            "coverage_complete": True,
        },
        grounding_enforcement={"status": "blocked"},
    )

    assert confidence["level"] == "LOW"
    assert "blocked by grounding enforcement" in " ".join(confidence["reasoning"])
