"""Validate or run the Stage 6 claim-support benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_mistralai import ChatMistralAI

from src.chains.claim_support import LLMClaimSupportEvaluator
from src.config import get_settings
from src.evaluation.claim_support_benchmark import (
    load_jsonl,
    load_sources,
    score_predictions,
    validate_gold,
)


DEFAULT_GOLD = Path("benchmarks/claim_support/gold_v0.1.jsonl")
DEFAULT_SOURCES = Path("benchmarks/claim_support/sources_v0.1.json")


def _write_json(path: Path | None, payload: object) -> None:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered)


def run_model(
    gold_items: list[dict],
    sources: dict[str, dict],
    output_path: Path,
) -> list[dict]:
    settings = get_settings()
    llm = ChatMistralAI(
        model=settings.mistral_model_small,
        temperature=0.0,
        api_key=settings.mistral_api_key,
    )
    evaluator = LLMClaimSupportEvaluator(llm)
    predictions = []

    for item in gold_items:
        cited_sources = [
            {
                "source_id": source_id,
                "text": sources[source_id]["text"],
            }
            for source_id in item["source_ids"]
        ]
        assessment = evaluator.evaluate(item["claim"], cited_sources)
        predictions.append(
            {
                "item_id": item["item_id"],
                "predicted_label": assessment["label"],
                "reason": assessment["reason"],
                "evidence": assessment["evidence"],
                "evaluator_error": assessment["evaluator_error"],
                "model": settings.mistral_model_small,
                "temperature": 0.0,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(prediction, ensure_ascii=False) + "\n"
            for prediction in predictions
        ),
        encoding="utf-8",
    )
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate or score the Stage 6 claim-support benchmark."
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=Path("benchmark_results/claim_support_predictions.jsonl"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("benchmark_results/claim_support_report.json"),
    )
    args = parser.parse_args()

    sources = load_sources(args.sources)
    gold_items = load_jsonl(args.gold)
    validation = validate_gold(gold_items, sources)

    if not validation["valid"]:
        _write_json(args.report_output, {"validation": validation})
        return 1

    if args.run_model and args.predictions:
        parser.error("Use either --run-model or --predictions, not both.")

    predictions = None
    if args.run_model:
        predictions = run_model(
            gold_items,
            sources,
            args.prediction_output,
        )
    elif args.predictions:
        predictions = load_jsonl(args.predictions)

    if predictions is None:
        _write_json(args.report_output, {"validation": validation})
        return 0

    scoring = score_predictions(gold_items, predictions)
    report = {
        "validation": validation,
        "scoring": scoring,
        "benchmark_status": {
            "gold_adjudicator": "ChatGPT GPT-5.6 Sol",
            "human_reviewed": False,
            "legal_professional_reviewed": False,
            "suitable_for_internal_model_iteration": True,
            "suitable_for_claiming_human_validated_legal_accuracy": False,
        },
    }
    _write_json(args.report_output, report)

    return 0 if scoring["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
