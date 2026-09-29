# Claim-level source support verification

Stage 4 adds an experimental semantic-support layer after deterministic citation integrity.

## Pipeline

```text
generated answer
   |
   v
citation integrity
[S1] -> retrieved source?
   |
   | structurally valid
   v
claim extraction
C1, C2, C3...
   |
   v
claim -> cited source passages only
   |
   v
support evaluator
   |
   +-- supported
   +-- contradicted
   +-- insufficient
```

The support evaluator does not receive uncited documents, and its prompt explicitly prohibits use of
outside legal knowledge.

## Deterministic responsibilities

`ClaimSupportVerifier` deterministically:

- segments cited answer sentences into claim units;
- records the `[S#]` IDs attached to each claim;
- maps those IDs back to the exact retrieved document passages;
- retains judgment `document_id`, `chunk_id`, and page provenance;
- caps the number of evaluated claims;
- aggregates per-claim labels into an answer-level status;
- prevents semantic assessment from running when citation integrity has already failed.

## Model-assessed responsibility

`LLMClaimSupportEvaluator` receives one claim and only its cited source passages. It must choose:

- `supported`: all material parts are established by the cited text;
- `contradicted`: the cited text directly conflicts with a material part;
- `insufficient`: the full claim is not established, is ambiguous, or only partially supported.

The evaluator is instructed to choose `insufficient` for partially supported compound claims.

Invalid evaluator output also fails conservatively to `insufficient`.

## Result contract

A typical result is:

```json
{
  "status": "supported",
  "experimental": true,
  "evaluator": "llm_source_only_v1",
  "claims_total": 3,
  "claims_evaluated": 3,
  "counts": {
    "supported": 3,
    "contradicted": 0,
    "insufficient": 0
  },
  "truncated": false,
  "independently_validated": false
}
```

The answer-level status is conservative:

1. any contradicted claim -> `contradicted`;
2. otherwise any insufficient claim -> `insufficient`;
3. otherwise all evaluated cited claims -> `supported`.

The verifier also reports semantic-check coverage separately:

- `coverage_complete=true` means every nontrivial answer segment carried a canonical `[S#]` marker;
- `coverage_complete=false` means one or more nontrivial segments had no citation and were not
  semantically checked;
- `uncited_segments_count` and a capped sample of `uncited_segments` make that gap visible.

An answer can therefore have `status=supported` while `coverage_complete=false`. In that case the
correct interpretation is **all evaluated cited claims were assessed as supported, but the whole
answer was not checked**.

## Important interpretation

A `supported` result means:

> the configured experimental verifier assessed the evaluated cited claims as supported by their
> cited retrieved passages.

It does **not** establish that:

- the underlying legal source is authoritative or current;
- the retrieval corpus is complete;
- the model-based verifier is always correct;
- the answer is legally correct overall;
- uncited practical advice has been independently validated.

For that reason the result always reports `experimental=true` and
`independently_validated=false` until a reviewed benchmark justifies stronger language.

## Confidence interaction

Claim support can only lower the existing heuristic confidence indicator:

- `contradicted` forces LOW;
- `insufficient` prevents HIGH;
- incomplete citation/support coverage prevents HIGH;
- `supported` does not increase confidence.

## Cost control

`CLAIM_SUPPORT_MAX_CLAIMS` defaults to 12. Longer answers are truncated for verification and the
result reports `truncated=true`.

The feature can be disabled with:

```text
ENABLE_CLAIM_SUPPORT_VERIFICATION=false
```

## Benchmark requirement

This stage intentionally does not label a synthetic test fixture as a legal-quality benchmark.

Before this verifier is used for strong product claims, create a manually reviewed Bangladesh-law
evaluation set containing at least:

- directly supported claims;
- partially supported compound claims;
- contradicted claims;
- sources that are topically related but insufficient;
- multi-source claims;
- statute and judgment examples;
- English, Bangla, and mixed-language examples.

Measure per-label precision/recall and disagreement with legal reviewers before changing the
`experimental` / `independently_validated` flags.
