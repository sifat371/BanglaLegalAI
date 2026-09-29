# Public judgment end-to-end validation

BanglaLegalAI has an external smoke workflow for validating the complete downstream path on real
public Bangladesh Supreme Court judgments.

The PDFs are downloaded only during the workflow and are not committed to this repository.

## Corpus

The validation uses the same four public judgments already exercised by BanglaLegalIngest:

1. Death Reference No 106 of 2018
2. Death Reference No.117 OF 2017
3. Civil Revision No.205 of 2021
4. Criminal Appeal No. 3346 of 2022

Using the same corpus isolates the integration question: when ingestion succeeds, does
BanglaLegalAI preserve and use the result correctly?

## Path under test

```text
Supreme Court PDF
  -> BanglaLegalIngest
  -> LegalDocument
  -> RetrievalChunk[]
  -> JudgmentProcessor
  -> LangChain Document[]
  -> deterministic chunk IDs
  -> ChromaDB + BM25
  -> HybridRetriever
  -> page-grounded source citation
```

## Checks

The validator fails if any of the following occur:

- an expected PDF is missing or cannot be ingested;
- the parsed case number no longer matches the established smoke fixture;
- pages, text, or retrieval chunks are missing;
- chunk IDs are duplicated;
- page or source-SHA provenance is lost;
- ChromaDB or BM25 stores a different number of logical chunks;
- indexing the same chunks twice changes collection size;
- an exact case-number query cannot retrieve its source judgment within the top 5;
- the retrieved judgment source loses its page citation.

The generated JSON report records document-level ingestion diagnostics, index counts, retrieval
ranks, chunk IDs, pages, and rendered source citations.

## Why the workflow uses local deterministic embeddings

The workflow is intended to validate **integration plumbing**, not embedding-model quality. It uses
a deterministic token-hash embedding implementation only inside the validator. This has three
advantages:

- no Hugging Face or other AI API secret is required;
- results are reproducible;
- a provider outage cannot be mistaken for an ingestion/indexing regression.

Production BanglaLegalAI continues to use the configured embedding service.

The retrieval result from this workflow should therefore be interpreted as an exact-case retrieval
smoke test, not as a semantic retrieval benchmark.

## Running manually

Download the four PDFs using the URLs in the workflow and run:

```bash
uv run python scripts/validate_public_judgments.py \
  path/to/pdfs \
  --output public-judgment-report.json
```

Or use the GitHub Actions workflow **Public Judgment E2E**.
