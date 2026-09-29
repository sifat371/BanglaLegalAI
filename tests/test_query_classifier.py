import os

os.environ.setdefault("MISTRAL_API_KEY", "test-key")
os.environ.setdefault("HUGGINGFACE_API_KEY", "test-key")

from src.chains.query_classifier import SimpleQueryClassifier


def test_real_bangladesh_case_number_is_exact_case_search():
    result = SimpleQueryClassifier().classify(
        "Find Criminal Appeal No. 3346 of 2022"
    )

    assert result["intent"] == "CASE_SEARCH"
    assert result["search_strategy"] == "EXACT_MATCH"
    assert result["metadata_filters"]["source_type"] == "case"
    assert result["specific_references"]["case_ids"] == [
        "Criminal Appeal No. 3346 of 2022"
    ]
