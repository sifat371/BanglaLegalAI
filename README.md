# BanglaLegalAI

**Grounded legal retrieval and AI assistance for Bangladesh law**

BanglaLegalAI is the downstream search, retrieval, and answer-generation application in the
BanglaLegal ecosystem. Structured statutes and case-law sources are indexed with hybrid retrieval,
while real court PDFs are normalized by
[BanglaLegalIngest](https://github.com/sifat371/BanglaLegalIngest) before they enter the retrieval
layer.

> **Project status:** active research/engineering prototype. This repository is not yet a validated
> production legal service, and generated answers are not a substitute for advice from a qualified
> legal professional.

## Architecture

```text
Structured acts (JSON) -----------------------> ActProcessor
                                                   |
Court judgments (PDF) -> BanglaLegalIngest -> JudgmentProcessor
                                                   |
Legacy case-study Markdown -> CaseProcessor ------+
                                                   |
                                                   v
                                        LangChain Document[]
                                                   |
                                  +----------------+----------------+
                                  |                                 |
                                  v                                 v
                              ChromaDB                             BM25
                                  |                                 |
                                  +----------------+----------------+
                                                   v
                                           HybridRetriever
                                                   |
                                                   v
                                     Query + response chains
                                                   |
                                                   v
                                      Public / research agents
                                                   |
                                                   v
                                             Streamlit UI
```

BanglaLegalIngest owns PDF extraction, Bangla encoding handling, deterministic legal metadata,
source provenance, diagnostics, and page-grounded retrieval chunks. BanglaLegalAI owns indexing,
embeddings, retrieval/ranking, query understanding, answer generation, citation presentation, and
application behavior.

## Key capabilities

- Hybrid dense + BM25 retrieval over Bangladesh legal material.
- Structured statute ingestion from the existing acts corpus.
- Real judgment PDF ingestion through BanglaLegalIngest.
- Deterministic judgment chunk IDs for idempotent indexing.
- Page-aware judgment citations carrying document and chunk provenance.
- Rule-based and LLM-assisted query classification.
- Separate public-facing and legal-research response modes.
- Streamlit application for interactive use.

## Requirements

- Python 3.13+
- `uv`
- Mistral API key for LLM-backed query/response features
- Hugging Face API key for the configured embedding service

## Installation

```bash
git clone https://github.com/sifat371/BanglaLegalAI.git
cd BanglaLegalAI
uv sync --group dev
cp .env.example .env
```

Add your API credentials to `.env`.

BanglaLegalIngest is currently resolved directly from its repository and pinned to a tested commit
in `pyproject.toml`. This keeps the integration contract reproducible while BanglaLegalIngest is
still pre-1.0.

## Data ingestion

### Structured acts

```bash
uv run python scripts/ingest_data.py --acts-only --yes
```

For a small development run:

```bash
uv run python scripts/ingest_data.py --acts-only --limit 10 --yes
```

### Legacy Markdown case studies

```bash
uv run python scripts/ingest_data.py --cases-only --yes
```

These remain supported for compatibility but are distinct from real court judgments.

### Real court judgments

Place PDFs under `data/judgments/` or provide another directory:

```bash
uv run python scripts/ingest_data.py --judgments-only --yes
```

```bash
uv run python scripts/ingest_data.py \
  --judgments-only \
  --judgments-dir /path/to/judgments \
  --recursive \
  --yes
```

The judgment path is:

```text
PDF
 -> BanglaLegalIngest
 -> LegalDocument
 -> provenance-preserving RetrievalChunk[]
 -> JudgmentProcessor
 -> LangChain Document[]
 -> ChromaDB + BM25
```

BanglaLegalAI does **not** re-split BanglaLegalIngest judgment chunks. This preserves page numbers,
page-relative offsets, source hashes, and deterministic chunk IDs.

### Full ingestion

```bash
uv run python scripts/ingest_data.py --yes
```

If `data/judgments/` does not exist, judgment ingestion is skipped.

## Judgment retrieval metadata

A judgment chunk entering the retrieval layer carries scalar metadata suitable for ChromaDB,
including:

```text
source_type = judgment
chunk_id
document_id
source_filename
source_sha256
page_start / page_end
chunk_index
char_start / char_end
case_number
case_type
court
district
judges
citations
encoding_kind
```

For current compatibility, `case_id`, `case_title`, and `court_level` aliases are also emitted.
They can be removed later after the old case-study assumptions are fully migrated.

## Deterministic identity

For BanglaLegalIngest judgments, `chunk_id` is the canonical retrieval identity. ChromaDB and BM25
now use the same stable identity contract, so ingesting the same judgment chunk again replaces the
existing BM25 record rather than silently creating a duplicate logical document.

## Running the application

```bash
uv run streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501`.

## Testing

```bash
uv run pytest
```

The integration tests currently protect:

- BanglaLegalIngest-to-LangChain metadata conversion;
- page/document/chunk provenance;
- deterministic retrieval IDs;
- idempotent BM25 ingestion;
- Chroma-style filter behavior in BM25;
- case-law routing;
- real Bangladesh case-number detection;
- page-grounded judgment source formatting.

CI also performs a Python syntax check and focused linting of the new integration modules.

## Repository structure

```text
BanglaLegalAI/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── acts/
│   ├── case_studies/
│   └── judgments/
├── docs/
│   └── bangla_legal_ingest_integration.md
├── scripts/
│   └── ingest_data.py
├── src/
│   ├── agents/
│   ├── chains/
│   ├── data_processing/
│   │   ├── act_processor.py
│   │   ├── case_processor.py
│   │   └── judgment_processor.py
│   ├── prompts/
│   ├── vectorstore/
│   │   ├── bm25_store.py
│   │   ├── chroma_store.py
│   │   ├── document_identity.py
│   │   └── hybrid_retriever.py
│   └── config.py
└── tests/
```

## Real-judgment validation

BanglaLegalAI includes an external end-to-end smoke workflow using four public Bangladesh Supreme
Court judgments. The first measured run ingested all four documents into **135 unique chunks**,
kept both ChromaDB and BM25 at **135 records after re-indexing**, and retrieved the correct judgment
at **rank 1 for all four exact case-number queries** while preserving source pages.

See `docs/public_judgment_validation.md` for the validation boundary and measured results.

## Current limitations

- The UI and much of the prompt layer are still English-first.
- Existing case-study Markdown remains a legacy compatibility source.
- Real judgment coverage is limited by the PDFs explicitly ingested.
- BanglaLegalIngest currently does not provide a general OCR baseline for image-only PDFs.
- Answer quality depends on retrieval quality and the supplied corpus.
- Page-grounded provenance is available for judgments, but full answer-level citation verification is
  still a downstream maturity task.
- Authentication, caching, user workspaces, and production deployment controls are not yet complete.

## Ecosystem boundary

**BanglaLegalIngest:** document ingestion infrastructure.

**BanglaLegalAI:** legal retrieval and grounded AI application.

Keeping these repositories independent means BanglaLegalIngest can remain reusable by other search,
RAG, research, and legal-document systems while BanglaLegalAI can evolve its retrieval and product
layers independently.
