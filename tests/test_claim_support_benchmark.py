from pathlib import Path

from src.evaluation.claim_support_benchmark import (
    load_jsonl,
    load_sources,
    score_predictions,
    validate_gold,
)


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "benchmarks" / "claim_support" / "gold_v0.1.jsonl"
SOURCES = ROOT / "benchmarks" / "claim_support" / "sources_v0.1.json"


def test_stage6_gold_set_is_valid_and_balanced():
    sources = load_sources(SOURCES)
    items = load_jsonl(GOLD)

    result = validate_gold(items, sources)

    assert result["valid"] is True
    assert result["items"] == 30
    assert result["label_counts"] == {
        "supported": 10,
        "contradicted": 10,
        "insufficient": 10,
    }
    assert result["language_counts"]["bn"] >= 3
    assert result["language_counts"]["mixed"] >= 3
    assert result["difficulty_counts"]["hard"] >= 3


def test_every_benchmark_source_has_auditable_provenance():
    sources = load_sources(SOURCES)

    for source in sources.values():
        assert source["source_type"] in {"judgment", "statute"}
        assert source["title"]
        assert source["text"]
        if source["source_type"] == "judgment":
            assert source.get("url")
            assert source.get("page")
        else:
            assert source.get("repo_path")


def test_perfect_predictions_score_one():
    gold = load_jsonl(GOLD)
    predictions = [
        {
            "item_id": item["item_id"],
            "predicted_label": item["gold_label"],
        }
        for item in gold
    ]

    result = score_predictions(gold, predictions)

    assert result["complete"] is True
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["errors"] == []


def test_scoring_detects_missing_and_wrong_predictions():
    gold = load_jsonl(GOLD)[:3]
    predictions = [
        {
            "item_id": gold[0]["item_id"],
            "predicted_label": gold[0]["gold_label"],
        },
        {
            "item_id": gold[1]["item_id"],
            "predicted_label": "insufficient",
        },
    ]

    result = score_predictions(gold, predictions)

    assert result["complete"] is False
    assert result["scored_items"] == 2
    assert result["missing_predictions"] == [gold[2]["item_id"]]
    assert len(result["errors"]) == 1


def test_gold_never_claims_human_or_legal_professional_review():
    items = load_jsonl(GOLD)

    for item in items:
        adjudication = item["adjudication"]
        assert adjudication["human_reviewer"] is False
        assert adjudication["legal_professional_review"] is False
        assert adjudication["review_mode"] == "manual_source_bound"
