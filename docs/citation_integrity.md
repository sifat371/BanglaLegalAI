# Answer citation integrity

BanglaLegalAI uses a closed source-ID protocol for generated legal answers.

## Source IDs

Every retrieved document supplied to the response model receives a response-local ID:

```text
[S1]
[S2]
[S3]
...
```

For judgments, the source object behind an ID still contains the stable ingestion provenance,
including `document_id`, `chunk_id`, source filename, court, case number, and source page.

The response model is instructed to cite factual legal claims using only exact markers such as
`[S1]`. The UI maps those markers back to the human-readable legal citation.

## Deterministic verification

After generation, `CitationVerifier` parses the answer and compares every source marker with the
retrieved source set that was actually supplied to the model.

The verifier reports:

- `verified`: at least one canonical citation is present and every cited source ID exists;
- `uncited`: sources were supplied, but the answer used no canonical `[S#]` markers;
- `failed`: the answer used an unknown or noncanonical source marker;
- `not_applicable`: no retrieved sources and no answer citations were present.

It also returns the cited source IDs and, for valid references, the corresponding source metadata.

## What "verified" means

`verified` means **citation binding/integrity is valid**:

> every citation marker in the answer resolves to a source that was actually retrieved for that
> answer.

It does **not** mean that the cited passage semantically proves every claim in the sentence.

For this reason the response object explicitly contains:

```json
{
  "structurally_verified": true,
  "semantic_support_verified": false
}
```

The Streamlit interface uses the same wording and does not present source-ID validation as
independent legal-fact verification.

## Closed-world answer rule

The response prompt now treats the retrieved documents as the only factual legal evidence available
for the answer. It instructs the model to:

- use only IDs present in the supplied context;
- avoid invented statutes, cases, sections, pages, holdings, dates, or quotations;
- say when the retrieved material is insufficient rather than fill gaps from model memory;
- avoid claiming that a rule is current unless the retrieved corpus establishes that;
- keep practical suggestions distinguishable from sourced statements of law.

This reduces unsupported generation, but the deterministic verifier can only enforce source-marker
integrity.

## Streaming behavior

Streaming answers are shown as they are generated. After the stream completes, the full answer is
verified and the final metadata event contains `citation_verification`.

If a streamed answer is uncited or uses an invalid source marker, the UI shows a warning/error.
Because already-streamed text cannot be retracted, strict fail-closed generation or automatic
citation repair would require buffering/regeneration and is intentionally deferred.

## Confidence interaction

The existing heuristic confidence indicator is capped by citation integrity:

- a failed citation check forces LOW;
- an uncited sourced answer cannot remain HIGH.

This remains a heuristic display signal, not a probability that the legal answer is correct.

## Next verification layer

A later stage can evaluate **claim-level semantic support**, for example:

1. segment the answer into legal factual claims;
2. bind each claim to its cited chunks;
3. test whether the cited passage supports, contradicts, or is insufficient for the claim;
4. refuse, repair, or flag unsupported claims;
5. benchmark that verifier on a manually reviewed Bangladesh-law evaluation set.

That layer should not be labeled reliable until it has measured precision/recall on such a set.
