"""End-to-end validation on public Bangladesh Supreme Court judgments.

This validates the downstream BanglaLegalAI handoff:
PDF -> BanglaLegalIngest -> judgment adapter -> Chroma + BM25 -> hybrid retrieval -> citation.
It intentionally uses deterministic local embeddings so the smoke test needs no external AI API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path

from langchain_core.embeddings import Embeddings
from legal_ingest import LegalDocumentPipeline, PipelineConfig

from src.chains.retrieval_chain import RetrievalChain
from src.data_processing.judgment_processor import JudgmentProcessor
from src.vectorstore.bm25_store import BM25Store
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.document_identity import get_document_id
from src.vectorstore.hybrid_retriever import HybridRetriever


CASES = [
    {
        "filename": "death_ref_106_2018.pdf",
        "case_number": "Death Reference No 106 of 2018",
    },
    {
        "filename": "death_ref_117_2017.pdf",
        "case_number": "Death Reference No.117 OF 2017",
    },
    {
        "filename": "civil_revision_205_2021.pdf",
        "case_number": "Civil Revision No.205 of 2021",
    },
    {
        "filename": "criminal_appeal_3346_2022.pdf",
        "case_number": "Criminal Appeal No. 3346 of 2022",
    },
]


def _normalize(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.casefold().replace(".", "").split())


class DeterministicHashEmbeddings(Embeddings):
    """Small local lexical embedding used only to validate retrieval plumbing."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _citation_for(document) -> dict:
    chain = RetrievalChain.__new__(RetrievalChain)
    return chain._extract_sources([document])[0]


def run(input_dir: Path, work_dir: Path) -> dict:
    failures: list[str] = []
    records: list[dict] = []
    all_documents = []
    expected_document_ids: dict[str, str] = {}

    pipeline = LegalDocumentPipeline(
        PipelineConfig(
            extractor="auto",
            min_extracted_characters=100,
            parse_metadata=True,
            preserve_page_text=True,
        )
    )
    processor = JudgmentProcessor(
        pipeline=pipeline,
        chunk_size=1000,
        chunk_overlap=200,
    )

    for case in CASES:
        path = input_dir / case["filename"]
        if not path.exists():
            failures.append(f"missing file: {path}")
            continue

        try:
            processed = processor.process_pdf(path)
        except Exception as exc:
            failures.append(
                f"{case['filename']}: {type(exc).__name__}: {exc}"
            )
            continue

        ingestion = processed.ingestion
        metadata = ingestion.document.metadata
        actual_case_number = metadata.case_number
        document_id = ingestion.document.document_id
        expected_document_ids[case["case_number"]] = document_id

        checks = {
            "case_number": _normalize(actual_case_number)
            == _normalize(case["case_number"]),
            "has_pages": bool(ingestion.document.pages),
            "has_text": bool(ingestion.document.text.strip()),
            "has_chunks": bool(processed.documents),
            "chunk_ids_unique": len(
                {get_document_id(document) for document in processed.documents}
            )
            == len(processed.documents),
            "page_provenance": all(
                int(document.metadata["page_start"]) >= 1
                and int(document.metadata["page_end"])
                >= int(document.metadata["page_start"])
                for document in processed.documents
            ),
            "document_identity": all(
                document.metadata["document_id"] == document_id
                and document.metadata["source_sha256"] == ingestion.document.source_sha256
                for document in processed.documents
            ),
        }

        for name, passed in checks.items():
            if not passed:
                failures.append(f"{case['filename']}: failed check: {name}")

        all_documents.extend(processed.documents)
        records.append(
            {
                "filename": case["filename"],
                "expected_case_number": case["case_number"],
                "case_number": actual_case_number,
                "document_id": document_id,
                "pages": len(ingestion.document.pages),
                "chunks": len(processed.documents),
                "extractor": ingestion.diagnostics.extractor,
                "fallback_used": ingestion.diagnostics.fallback_used,
                "warnings": ingestion.diagnostics.warnings,
                "checks": checks,
            }
        )

    if not all_documents:
        failures.append("no retrieval documents were produced")
        return {
            "documents_requested": len(CASES),
            "documents_ingested": len(records),
            "chunks": 0,
            "failures": failures,
            "documents": records,
            "retrieval": [],
        }

    ids = [get_document_id(document) for document in all_documents]
    if len(set(ids)) != len(ids):
        failures.append("chunk IDs are not globally unique")

    chroma = ChromaStore(
        "public_judgment_validation",
        embeddings=DeterministicHashEmbeddings(),
        persist_directory=work_dir / "chroma",
    )
    bm25 = BM25Store(
        "public_judgment_validation",
        persist_directory=work_dir / "bm25",
    )

    chroma.add_documents(all_documents, ids=ids)
    bm25.add_documents(all_documents)

    initial_chroma_count = chroma.get_collection_stats()["document_count"]
    initial_bm25_count = bm25.get_stats()["document_count"]

    if initial_chroma_count != len(all_documents):
        failures.append(
            f"Chroma count mismatch: {initial_chroma_count} != {len(all_documents)}"
        )
    if initial_bm25_count != len(all_documents):
        failures.append(
            f"BM25 count mismatch: {initial_bm25_count} != {len(all_documents)}"
        )

    # Re-index the same logical chunks to validate stable IDs/idempotency.
    chroma.add_documents(all_documents, ids=ids)
    bm25.add_documents(all_documents)

    final_chroma_count = chroma.get_collection_stats()["document_count"]
    final_bm25_count = bm25.get_stats()["document_count"]
    idempotent = (
        final_chroma_count == initial_chroma_count
        and final_bm25_count == initial_bm25_count
    )
    if not idempotent:
        failures.append(
            "re-indexing changed collection size "
            f"(Chroma {initial_chroma_count}->{final_chroma_count}, "
            f"BM25 {initial_bm25_count}->{final_bm25_count})"
        )

    retriever = HybridRetriever(chroma, bm25, alpha=0.5)
    retrieval_records = []

    for case in CASES:
        expected_id = expected_document_ids.get(case["case_number"])
        if not expected_id:
            continue

        results = retriever.retrieve_with_scores(
            query=case["case_number"],
            k=5,
            method="rrf",
        )

        hit_rank = None
        hit_document = None
        hit_score = None
        for rank, (document, score) in enumerate(results, start=1):
            if document.metadata.get("document_id") == expected_id:
                hit_rank = rank
                hit_document = document
                hit_score = score
                break

        passed = hit_document is not None
        if not passed:
            failures.append(
                f"retrieval miss@5 for exact case query: {case['case_number']}"
            )

        citation = _citation_for(hit_document) if hit_document else None
        if citation and citation.get("page") in {None, "", "N/A"}:
            failures.append(
                f"citation missing page for: {case['case_number']}"
            )

        retrieval_records.append(
            {
                "query": case["case_number"],
                "expected_document_id": expected_id,
                "hit_at_5": passed,
                "rank": hit_rank,
                "score": hit_score,
                "chunk_id": (
                    hit_document.metadata.get("chunk_id")
                    if hit_document
                    else None
                ),
                "page": (
                    hit_document.metadata.get("page_start")
                    if hit_document
                    else None
                ),
                "citation": citation,
            }
        )

    retrieval_hits = sum(
        1 for record in retrieval_records if record["hit_at_5"]
    )
    retrieval_total = len(retrieval_records)

    return {
        "documents_requested": len(CASES),
        "documents_ingested": len(records),
        "chunks": len(all_documents),
        "unique_chunk_ids": len(set(ids)),
        "index_counts": {
            "chroma_initial": initial_chroma_count,
            "chroma_after_reindex": final_chroma_count,
            "bm25_initial": initial_bm25_count,
            "bm25_after_reindex": final_bm25_count,
            "idempotent": idempotent,
        },
        "retrieval": {
            "hits_at_5": retrieval_hits,
            "queries": retrieval_total,
            "recall_at_5": (
                retrieval_hits / retrieval_total if retrieval_total else 0.0
            ),
            "records": retrieval_records,
        },
        "failures": failures,
        "documents": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()

    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        report = run(args.input_dir, args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="banglalegalai-validation-") as tmp:
            report = run(args.input_dir, Path(tmp))

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
