from src.agents.base_agent import BaseAgent, ConversationHistory


class FakeRetrievalChain:
    def retrieve(self, **kwargs):
        return {
            "documents": [object()],
            "num_results": 1,
            "classification": None,
        }


class FakeResponseChain:
    def __init__(self):
        self.stream_answer_called = False

    def stream_answer(self, *args, **kwargs):
        self.stream_answer_called = True
        raise AssertionError("unsafe first-draft streaming must not be used")

    def generate_response(self, **kwargs):
        return {
            "answer": "Final repaired answer [S1].",
            "sources": [
                {
                    "source_id": "S1",
                    "citation": "Example source",
                    "cited": True,
                }
            ],
            "num_sources": 1,
            "citation_verification": {"status": "verified"},
            "claim_support_verification": {
                "status": "supported",
                "coverage_complete": True,
            },
            "grounding_enforcement": {
                "status": "repaired",
                "repair_attempts": 1,
            },
            "confidence": {"level": "MEDIUM", "reasoning": []},
        }

    def _generate_followups(self, query, answer):
        return ["What happened next?"]


def test_chat_stream_only_emits_finalized_answer():
    agent = BaseAgent.__new__(BaseAgent)
    agent.user_type = "public"
    agent.session_id = "test"
    agent.default_k = 5
    agent.retrieval_chain = FakeRetrievalChain()
    agent.response_chain = FakeResponseChain()
    agent.conversation = ConversationHistory()

    events = list(
        agent.chat_stream(
            "Question",
            include_followups=True,
            include_confidence=True,
        )
    )

    answer = "".join(
        event["content"]
        for event in events
        if event["type"] == "answer_chunk"
    )
    metadata = events[-1]

    assert answer == "Final repaired answer [S1]."
    assert metadata["grounding_enforcement"]["status"] == "repaired"
    assert agent.response_chain.stream_answer_called is False
    assert agent.conversation.history[0]["answer"] == answer
