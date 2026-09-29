# Claim-support benchmark v0.1

This benchmark evaluates the `supported / contradicted / insufficient` classifier used by
BanglaLegalAI's experimental claim-support layer.

## Review status

The gold labels in v0.1 were manually adjudicated by **ChatGPT GPT-5.6 Sol acting as a
source-bound reviewer** on 2026-09-29.

They are **not human-reviewed** and **not reviewed by a Bangladesh legal professional**.

The benchmark is appropriate for:

- internal model iteration;
- regression testing;
- finding obvious support-classification failures;
- comparing verifier prompts/models on a fixed source set.

It is not sufficient, by itself, for claims such as:

- "human-validated legal accuracy";
- "lawyer-reviewed benchmark";
- "clinically/legally certified";
- production reliability guarantees.

A later lawyer/human review can reuse the same items and add independent adjudications rather than
rebuilding the benchmark.

## Composition

v0.1 contains **30 items**, exactly balanced:

| Gold label | Items |
| --- | ---: |
| supported | 10 |
| contradicted | 10 |
| insufficient | 10 |

The source registry contains real Bangladesh Supreme Court judgment passages and statute passages
already present in the repository.

The set includes:

- direct holdings and statutory rules;
- reversed/opposite propositions;
- missing facts and invented deadlines;
- compound claims;
- multi-source claims;
- English examples;
- Bangla examples;
- mixed Bangla-English examples;
- judgment and statute sources.

## Label policy

### supported

All material parts of the claim are explicitly stated or directly entailed by the cited source
passage(s).

For a compound claim, every material part must be supported.

### contradicted

At least one material part of the claim directly conflicts with the cited source passage(s).

This label requires positive conflicting evidence. Merely failing to find support is not enough.

### insufficient

The source does not establish the full claim and does not directly establish its opposite.

Typical cases include:

- invented time limits;
- invented fees;
- missing remedies;
- missing factual details;
- added procedural requirements;
- partially supported compound claims.

## Adjudication method

For each item the reviewer recorded:

- claim;
- source IDs;
- gold label;
- language;
- difficulty;
- test phenomenon;
- short source-bound rationale;
- reviewer confidence;
- adjudicator metadata.

Gold rationale is for audit and error analysis only. It must not be passed to the model being
evaluated.

## Source provenance

`sources_v0.1.json` stores each selected source passage once.

Judgment sources include the public Supreme Court URL and source page. Statute sources point to the
repository JSON file from which the passage was selected.

## Running validation only

```bash
uv run python scripts/evaluate_claim_support_benchmark.py
```

This validates the benchmark schema and writes a report without invoking an LLM.

## Scoring existing predictions

Predictions use JSONL records such as:

```json
{"item_id":"CS001","predicted_label":"supported","reason":"..."}
```

Run:

```bash
uv run python scripts/evaluate_claim_support_benchmark.py \
  --predictions path/to/predictions.jsonl
```

The report includes:

- accuracy;
- macro-F1;
- per-label precision/recall/F1;
- confusion matrix;
- language-slice accuracy;
- difficulty-slice accuracy;
- item-level errors.

## Running the current Mistral verifier

With a real `MISTRAL_API_KEY` configured:

```bash
uv run python scripts/evaluate_claim_support_benchmark.py --run-model
```

The benchmark runner feeds the model only:

- the claim;
- the cited source passage(s).

It does **not** feed the gold label or reviewer rationale.

The model is run at temperature 0 for this evaluation.

## Why v0.1 is not a CI quality gate yet

The benchmark itself is checked in CI for schema, balance, provenance, and scoring correctness.

Model accuracy is not yet a required CI threshold because:

1. the gold labels have not received independent human/legal review;
2. live model calls would require a secret and introduce provider/network variance;
3. an appropriate acceptance threshold should be chosen after independent adjudication.

Once a human legal reviewer validates the set, a versioned benchmark (for example v1.0) can become a
stronger release gate.

## Current live-evaluation status

A safe repository-secret probe was run on 2026-09-29. The repository did not have a
`MISTRAL_API_KEY` Actions secret configured, so the current Mistral verifier was **not** scored and
no accuracy/F1 figure is reported.

This is intentional: the benchmark does not substitute the gold adjudicator's own labels as model
predictions, because doing so would be circular.

Configure the repository secret and run the **Claim Support Benchmark** workflow to produce the
first independent system-under-test report.
