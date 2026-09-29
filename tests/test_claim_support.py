from langchain_core.documents import Document

from src.chains.claim_support import ClaimSupportVerifier


class FakeEvaluator:
    name = "fake"

    def __init__(self, labels):
        self.labels = iter(labels)
        self.calls = []

    def evaluate(self, claim, cited_sources):
        self.calls.append((claim, cited_sources))
        return {
            "label": next(self.labels),
            "reason": "fixture",
            "evidence": "fixture evidence",
            "evaluator_error": None,
        }


def source(source_id, page="1"):
    return {
        "source_id": source_id,
        "type": "judgment",
        "citation": f"Example case, p. {page}",
        "document_id": "doc",
        "chunk_id": f"doc:p{page}:c0",
        "page": page,
    }


def test_extracts_cited_claims_and_keeps_multiple_sources():
    claims = ClaimSupportVerifier.extract_cited_claims(
        "First proposition [S1]. Second proposition uses two passages [S1] [S2]."
    )

    assert [claim["claim_id"] for claim in claims] == ["C1", "C2"]
    assert claims[0]["source_ids"] == ["S1"]
    assert claims[1]["source_ids"] == ["S1", "S2"]


def test_supported_claim_keeps_chunk_provenance():
    evaluator = FakeEvaluator(["supported"])
    verifier = ClaimSupportVerifier(evaluator)
    documents = [
        Document(page_content="The court allowed the appeal.", metadata={}),
    ]
    sources = [source("S1", page="7")]

    result = verifier.verify(
        "The court allowed the appeal [S1].",
        documents,
        sources,
        {"status": "verified"},
    )

    assert result["status"] == "supported"
    assert result["counts"]["supported"] == 1
    assert result["claims"][0]["provenance"][0]["chunk_id"] == "doc:p7:c0"
    assert evaluator.calls[0][1][0]["text"] == "The court allowed the appeal."


def test_contradiction_has_priority_in_overall_status():
    verifier = ClaimSupportVerifier(FakeEvaluator(["supported", "contradicted"]))

    result = verifier.verify(
        "Claim one [S1]. Claim two [S2].",
        [
            Document(page_content="source one", metadata={}),
            Document(page_content="source two", metadata={}),
        ],
        [source("S1"), source("S2")],
        {"status": "verified"},
    )

    assert result["status"] == "contradicted"
    assert result["counts"] == {
        "supported": 1,
        "contradicted": 1,
        "insufficient": 0,
    }


def test_insufficient_claim_makes_overall_insufficient():
    verifier = ClaimSupportVerifier(FakeEvaluator(["insufficient"]))

    result = verifier.verify(
        "Compound claim [S1].",
        [Document(page_content="partial evidence", metadata={})],
        [source("S1")],
        {"status": "verified"},
    )

    assert result["status"] == "insufficient"


def test_structural_citation_failure_prevents_semantic_run():
    evaluator = FakeEvaluator(["supported"])
    verifier = ClaimSupportVerifier(evaluator)

    result = verifier.verify(
        "Claim [S9].",
        [Document(page_content="source", metadata={})],
        [source("S1")],
        {"status": "failed"},
    )

    assert result["status"] == "not_run"
    assert evaluator.calls == []


def test_max_claims_caps_cost_and_reports_truncation():
    evaluator = FakeEvaluator(["supported"])
    verifier = ClaimSupportVerifier(evaluator, max_claims=1)

    result = verifier.verify(
        "One [S1]. Two [S1].",
        [Document(page_content="source", metadata={})],
        [source("S1")],
        {"status": "verified"},
    )

    assert result["claims_total"] == 2
    assert result["claims_evaluated"] == 1
    assert result["truncated"] is True


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("Message", (), {"content": self.content})()


def test_llm_evaluator_parses_json_and_uses_only_cited_text():
    from src.chains.claim_support import LLMClaimSupportEvaluator

    llm = FakeLLM(
        '{"label":"supported","reason":"directly stated","evidence":"allowed the appeal"}'
    )
    evaluator = LLMClaimSupportEvaluator(llm)

    result = evaluator.evaluate(
        "The court allowed the appeal.",
        [{"source_id": "S1", "text": "The court allowed the appeal."}],
    )

    assert result["label"] == "supported"
    assert result["evaluator_error"] is None
    assert "The court allowed the appeal." in llm.prompts[0]
    assert "[S1]" in llm.prompts[0]


def test_llm_evaluator_fails_conservatively_on_invalid_output():
    from src.chains.claim_support import LLMClaimSupportEvaluator

    evaluator = LLMClaimSupportEvaluator(FakeLLM("not json"))

    result = evaluator.evaluate(
        "Claim",
        [{"source_id": "S1", "text": "source"}],
    )

    assert result["label"] == "insufficient"
    assert result["evaluator_error"]


def test_trailing_citation_after_period_stays_bound_to_claim():
    claims = ClaimSupportVerifier.extract_cited_claims(
        "The appeal was allowed. [S1]"
    )

    assert len(claims) == 1
    assert claims[0]["text"] == "The appeal was allowed."
    assert claims[0]["source_ids"] == ["S1"]


def test_uncited_segments_are_reported_as_coverage_warning():
    verifier = ClaimSupportVerifier(FakeEvaluator(["supported"]))

    result = verifier.verify(
        "The appeal was allowed [S1]. This additional sentence has no citation at all.",
        [Document(page_content="The appeal was allowed.", metadata={})],
        [source("S1")],
        {"status": "verified"},
    )

    assert result["status"] == "supported"
    assert result["coverage_complete"] is False
    assert result["uncited_segments_count"] == 1
    assert "additional sentence" in result["uncited_segments"][0]
