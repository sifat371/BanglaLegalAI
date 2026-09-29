"""Utilities for validating and scoring the Stage 6 claim-support benchmark."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LABELS = ("supported", "contradicted", "insufficient")
LANGUAGES = ("en", "bn", "mixed")
DIFFICULTIES = ("easy", "medium", "hard")


def load_sources(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = payload.get("sources", [])
    return {source["source_id"]: source for source in sources}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        items.append(item)
    return items


def validate_gold(
    items: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    seen_ids: set[str] = set()

    for item in items:
        item_id = item.get("item_id")
        if not item_id:
            errors.append("item missing item_id")
            continue
        if item_id in seen_ids:
            errors.append(f"duplicate item_id: {item_id}")
        seen_ids.add(item_id)

        label = item.get("gold_label")
        if label not in LABELS:
            errors.append(f"{item_id}: invalid gold_label {label!r}")

        language = item.get("language")
        if language not in LANGUAGES:
            errors.append(f"{item_id}: invalid language {language!r}")

        difficulty = item.get("difficulty")
        if difficulty not in DIFFICULTIES:
            errors.append(f"{item_id}: invalid difficulty {difficulty!r}")

        source_ids = item.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            errors.append(f"{item_id}: source_ids must be a non-empty list")
        else:
            missing = [source_id for source_id in source_ids if source_id not in sources]
            if missing:
                errors.append(
                    f"{item_id}: unknown source_ids: {', '.join(missing)}"
                )

        if not str(item.get("claim", "")).strip():
            errors.append(f"{item_id}: empty claim")
        if not str(item.get("reviewer_rationale", "")).strip():
            errors.append(f"{item_id}: empty reviewer_rationale")

        adjudication = item.get("adjudication", {})
        if adjudication.get("human_reviewer") is not False:
            errors.append(f"{item_id}: human_reviewer must be false")
        if adjudication.get("legal_professional_review") is not False:
            errors.append(f"{item_id}: legal_professional_review must be false")

    label_counts = Counter(item.get("gold_label") for item in items)
    language_counts = Counter(item.get("language") for item in items)
    difficulty_counts = Counter(item.get("difficulty") for item in items)

    return {
        "valid": not errors,
        "errors": errors,
        "items": len(items),
        "label_counts": dict(label_counts),
        "language_counts": dict(language_counts),
        "difficulty_counts": dict(difficulty_counts),
    }


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _label_metrics(
    confusion: dict[str, dict[str, int]],
    label: str,
) -> dict[str, float | int]:
    tp = confusion[label][label]
    fp = sum(
        confusion[gold][label]
        for gold in LABELS
        if gold != label
    )
    fn = sum(
        confusion[label][predicted]
        for predicted in LABELS
        if predicted != label
    )
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def score_predictions(
    gold_items: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_by_id = {item["item_id"]: item for item in gold_items}
    pred_by_id: dict[str, dict[str, Any]] = {}
    duplicate_predictions: list[str] = []

    for prediction in predictions:
        item_id = prediction.get("item_id")
        if item_id in pred_by_id:
            duplicate_predictions.append(str(item_id))
        pred_by_id[str(item_id)] = prediction

    missing = sorted(set(gold_by_id) - set(pred_by_id))
    extra = sorted(set(pred_by_id) - set(gold_by_id))
    invalid_labels = sorted(
        item_id
        for item_id, prediction in pred_by_id.items()
        if prediction.get("predicted_label") not in LABELS
    )

    confusion = {
        gold: {predicted: 0 for predicted in LABELS}
        for gold in LABELS
    }
    correct = 0
    scored = 0

    slice_totals: dict[str, dict[str, list[int]]] = {
        "language": defaultdict(lambda: [0, 0]),
        "difficulty": defaultdict(lambda: [0, 0]),
    }

    errors = []
    for item_id, gold in gold_by_id.items():
        prediction = pred_by_id.get(item_id)
        if not prediction:
            continue
        predicted_label = prediction.get("predicted_label")
        if predicted_label not in LABELS:
            continue

        gold_label = gold["gold_label"]
        confusion[gold_label][predicted_label] += 1
        scored += 1
        is_correct = predicted_label == gold_label
        correct += int(is_correct)

        for dimension in ("language", "difficulty"):
            key = gold[dimension]
            slice_totals[dimension][key][1] += 1
            slice_totals[dimension][key][0] += int(is_correct)

        if not is_correct:
            errors.append(
                {
                    "item_id": item_id,
                    "gold_label": gold_label,
                    "predicted_label": predicted_label,
                    "language": gold["language"],
                    "difficulty": gold["difficulty"],
                    "phenomenon": gold["phenomenon"],
                    "claim": gold["claim"],
                    "reviewer_rationale": gold["reviewer_rationale"],
                    "prediction_reason": prediction.get("reason", ""),
                }
            )

    per_label = {
        label: _label_metrics(confusion, label)
        for label in LABELS
    }
    macro_f1 = sum(
        metrics["f1"] for metrics in per_label.values()
    ) / len(LABELS)

    slice_accuracy = {}
    for dimension, values in slice_totals.items():
        slice_accuracy[dimension] = {
            key: {
                "correct": pair[0],
                "total": pair[1],
                "accuracy": _safe_div(pair[0], pair[1]),
            }
            for key, pair in sorted(values.items())
        }

    return {
        "complete": not (
            missing
            or extra
            or invalid_labels
            or duplicate_predictions
        ),
        "gold_items": len(gold_items),
        "scored_items": scored,
        "accuracy": _safe_div(correct, scored),
        "macro_f1": macro_f1,
        "per_label": per_label,
        "confusion_matrix": confusion,
        "slice_accuracy": slice_accuracy,
        "missing_predictions": missing,
        "extra_predictions": extra,
        "invalid_prediction_labels": invalid_labels,
        "duplicate_predictions": duplicate_predictions,
        "errors": errors,
    }
