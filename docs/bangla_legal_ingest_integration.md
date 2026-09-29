# BanglaLegalIngest integration contract

This document defines the boundary between BanglaLegalAI and BanglaLegalIngest.

## Ownership

BanglaLegalIngest owns:

- PDF validation and extraction;
- page preservation;
- Bangla encoding detection and safe normalization;
- deterministic legal metadata parsing;
- metadata evidence/provenance;
- extraction diagnostics and warnings;
- source SHA-256 identity;
- page-grounded retrieval chunk creation.

BanglaLegalAI owns:

- conversion of retrieval chunks to its retrieval-document type;
- embeddings;
- ChromaDB and BM25 persistence;
- hybrid retrieval and ranking;
- query classification;
- answer generation;
- answer/source presentation;
- future answer-level citation verification;
- user interface and application state.

BanglaLegalIngest must not depend on BanglaLegalAI.

## Dependency policy

BanglaLegalAI currently pins BanglaLegalIngest to a specific Git commit through `tool.uv.sources`.
Do not switch this to an unpinned `main` dependency.

Before advancing the pin:

1. inspect BanglaLegalIngest's schema/chunk contract;
2. run BanglaLegalIngest CI and validation;
3. run BanglaLegalAI integration tests;
4. verify one real judgment end-to-end;
5. update this document when contract semantics change.

## Canonical judgment identity

The BanglaLegalIngest `RetrievalChunk.chunk_id` is authoritative.

BanglaLegalAI must not replace it with:

- random UUIDs;
- Python `hash()`;
- filename-only IDs;
- embedding-store-generated identities.

`document_id` remains the SHA-256-based source identity produced by BanglaLegalIngest.

## Chunking rule

Do not re-split judgment chunks in BanglaLegalAI.

The current ingestion contract is page-local. Re-splitting after ingestion can break:

- page citation semantics;
- page-relative character offsets;
- deterministic chunk identity;
- debugging and source verification.

Alternative retrieval chunk strategies should be implemented and evaluated at the ingestion boundary
or introduced as an explicit versioned downstream transformation that preserves provenance.

## Metadata mapping

`JudgmentProcessor` maps each retrieval chunk to a LangChain `Document`.

The primary fields are:

| BanglaLegalIngest | BanglaLegalAI metadata |
| --- | --- |
| chunk_id | chunk_id |
| document_id | document_id |
| source_filename | source_filename |
| source_sha256 | source_sha256 |
| page_start | page_start |
| page_end | page_end |
| chunk_index | chunk_index |
| char_start | char_start |
| char_end | char_end |
| case_number | case_number |
| case_type | case_type |
| court | court |
| district | district |
| judges | judges (scalar joined form) |
| citations | citations (scalar joined form) |

The adapter additionally exposes temporary compatibility aliases:

- `case_id`;
- `case_title`;
- `court_level`.

## Diagnostics

Extraction warnings are operational information and must not be silently discarded.

The ingestion CLI currently reports the number of warnings across processed judgments. A future
manifest layer should persist per-document diagnostics outside the vector metadata rather than
duplicating warning lists into every chunk.

## Indexing contract

Both ChromaDB and BM25 use the shared `get_document_id()` helper.

For judgments this resolves directly to `chunk_id`.

BM25 ingestion is idempotent by stable identity: re-ingesting the same chunk replaces its stored
document and rebuilds the sparse index.

## Retrieval routing

The case-law index may contain both:

- `source_type=judgment` for real PDF judgments;
- `source_type=case_study` for the legacy structured Markdown data.

Query routing treats `source_type=case` as the generic request for the shared case-law index. It
must not force a query to only the legacy case-study source type.

## Citation contract

When a retrieved judgment contains a source page, the response layer should preserve it in the source
display, for example:

```text
Criminal Appeal No. 3346 of 2022, Supreme Court of Bangladesh, p. 7
```

The source object also retains `document_id` and `chunk_id` so future citation verification can
trace the answer back to the exact indexed chunk.

## Raw PDF policy

Raw judgment PDFs under `data/judgments/` are ignored by Git by default. This avoids accidentally
committing large files or source material with unclear redistribution terms. Use local files or
explicitly curated redistributable fixtures for tests.
