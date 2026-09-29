"""Adapter from BanglaLegalIngest outputs to Law Buddy retrieval documents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from legal_ingest import (
    IngestionResult,
    LegalDocumentPipeline,
    RetrievalChunk,
    to_retrieval_chunks,
)


@dataclass(slots=True)
class ProcessedJudgment:
    """A source PDF together with its retrieval documents and ingestion result."""

    source_path: Path
    documents: list[Document]
    ingestion: IngestionResult


class JudgmentProcessor:
    """Process real Bangladesh legal PDFs through BanglaLegalIngest."""

    def __init__(
        self,
        pipeline: LegalDocumentPipeline | None = None,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        chunker: Callable[..., list[RetrievalChunk]] = to_retrieval_chunks,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

        self.pipeline = pipeline or LegalDocumentPipeline()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = chunker

    @staticmethod
    def _scalar(value: Any) -> str:
        return "" if value is None else str(value)

    @staticmethod
    def _joined(values: list[str]) -> str:
        return " | ".join(value for value in values if value)

    def _to_document(
        self,
        chunk: RetrievalChunk,
        *,
        encoding_kind: str,
        schema_version: str,
    ) -> Document:
        case_number = self._scalar(chunk.case_number)
        court = self._scalar(chunk.court)

        return Document(
            page_content=chunk.text,
            metadata={
                "source_type": "judgment",
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source_filename": chunk.source_filename,
                "source_sha256": chunk.source_sha256,
                "schema_version": schema_version,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chunk_index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "case_number": case_number,
                "case_type": self._scalar(chunk.case_type),
                "court": court,
                "district": self._scalar(chunk.district),
                "judges": self._joined(chunk.judges),
                "citations": self._joined(chunk.citations),
                "encoding_kind": encoding_kind,
                # Compatibility aliases for the current case-oriented UI/retrieval code.
                "case_id": case_number or chunk.document_id,
                "case_title": case_number or chunk.source_filename,
                "court_level": court,
            },
        )

    def process_pdf(self, file_path: str | Path) -> ProcessedJudgment:
        """Ingest one PDF and convert its page-grounded chunks for Law Buddy."""
        path = Path(file_path)
        ingestion = self.pipeline.ingest(path)

        chunks = self.chunker(
            ingestion.document,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

        encoding_kind = getattr(ingestion.document.encoding.kind, "value", None)
        if encoding_kind is None:
            encoding_kind = str(ingestion.document.encoding.kind)

        documents = [
            self._to_document(
                chunk,
                encoding_kind=encoding_kind,
                schema_version=ingestion.document.schema_version,
            )
            for chunk in chunks
        ]

        return ProcessedJudgment(
            source_path=path,
            documents=documents,
            ingestion=ingestion,
        )

    def process_directory(
        self,
        input_dir: str | Path,
        *,
        recursive: bool = False,
    ) -> list[ProcessedJudgment]:
        """Process all PDFs in a directory in deterministic filename order."""
        directory = Path(input_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Judgment directory does not exist: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Judgment path is not a directory: {directory}")

        candidates = directory.rglob("*") if recursive else directory.iterdir()
        pdfs = sorted(
            (path for path in candidates if path.is_file() and path.suffix.lower() == ".pdf"),
            key=lambda path: str(path).lower(),
        )

        return [self.process_pdf(path) for path in pdfs]
