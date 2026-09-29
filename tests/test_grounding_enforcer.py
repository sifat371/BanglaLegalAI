from src.chains.grounding_enforcer import GroundingEnforcer


def valid_citation():
    return {
        "status": "verified",
        "invalid_source_ids": [],
        "noncanonical_markers": [],
    }


def supported_claims(**overrides):
    value = {
        "status": "supported",
        "coverage_complete": True,
        "truncated": False,
        "claims": [],
        "uncited_segments": [],
    }
    value.update(overrides)
    return value


def test_grounding_gate_accepts_fully_checked_supported_answer():
    result = GroundingEnforcer.evaluate(
        valid_citation(),
        supported_claims(),
    )

    assert result == {"acceptable": True, "reasons": []}


def test_grounding_gate_rejects_contradiction():
    result = GroundingEnforcer.evaluate(
        valid_citation(),
        supported_claims(status="contradicted"),
    )

    assert result["acceptable"] is False
    assert "claim_support:contradicted" in result["reasons"]


def test_grounding_gate_rejects_incomplete_coverage():
    result = GroundingEnforcer.evaluate(
        valid_citation(),
        supported_claims(coverage_complete=False),
    )

    assert result["acceptable"] is False
    assert "semantic_coverage:incomplete_or_unknown" in result["reasons"]


def test_grounding_gate_rejects_truncated_verification():
    result = GroundingEnforcer.evaluate(
        valid_citation(),
        supported_claims(truncated=True),
    )

    assert result["acceptable"] is False
    assert "semantic_coverage:truncated_or_unknown" in result["reasons"]


def test_grounding_gate_rejects_evaluator_error():
    result = GroundingEnforcer.evaluate(
        valid_citation(),
        supported_claims(
            claims=[
                {
                    "claim_id": "C2",
                    "label": "insufficient",
                    "evaluator_error": "timeout",
                }
            ]
        ),
    )

    assert result["acceptable"] is False
    assert "evaluator_error:C2" in result["reasons"]


def test_repair_feedback_contains_only_problem_signals():
    feedback = GroundingEnforcer.repair_feedback(
        {
            "status": "failed",
            "invalid_source_ids": ["S9"],
            "noncanonical_markers": ["[Source 1]"],
        },
        {
            "claims": [
                {
                    "claim_id": "C1",
                    "label": "supported",
                    "text": "fine",
                    "source_ids": ["S1"],
                    "reason": "fine",
                },
                {
                    "claim_id": "C2",
                    "label": "insufficient",
                    "text": "too broad",
                    "source_ids": ["S2"],
                    "reason": "only partially supported",
                },
            ],
            "uncited_segments": ["This sentence has no citation."],
            "truncated": False,
        },
    )

    assert "S9" in feedback
    assert "[Source 1]" in feedback
    assert "C2 is insufficient" in feedback
    assert "C1 is supported" not in feedback
    assert "This sentence has no citation." in feedback


def test_grounding_gate_rejects_missing_coverage_fields():
    support = {
        "status": "supported",
        "claims": [],
    }

    result = GroundingEnforcer.evaluate(valid_citation(), support)

    assert result["acceptable"] is False
    assert "semantic_coverage:incomplete_or_unknown" in result["reasons"]
    assert "semantic_coverage:truncated_or_unknown" in result["reasons"]
