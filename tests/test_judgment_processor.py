from pathlib import Path
from types import SimpleNamespace

from src.data_processing.judgment_processor import JudgmentProcessor


class FakePipeline:
    def __init__(self):
        self.paths = []

    def ingest(self, path):
        self.paths.append(Path(path))
        document = SimpleNamespace(
            schema_version="1.0",
            encoding=SimpleNamespace(kind=SimpleNamespace(value="unicode_bangla")),
        )
        return SimpleNamespace(document=document, diagnostics=SimpleNamespace(warnings=[]))


def fake_chunker(document, *, chunk_size, overlap):
    assert chunk_size == 900
    assert overlap == 120
    return [
        SimpleNamespace(
            chunk_id="deadbeef:p3:c0",
            document_id="deadbeef",
            source_filename="appeal.pdf",
            source_sha256="deadbeef",
            page_start=3,
            page_end=3,
            chunk_index=0,
            char_start=10,
            char_end=110,
            text="Relevant judgment passage",
            case_number="Criminal Appeal No. 3346 of 2022",
            case_type="Criminal Appeal",
            court="Supreme Court of Bangladesh",
            district=None,
            judges=["Justice A", "Justice B"],
            citations=["75 DLR 123"],
        )
    ]


def test_process_pdf_preserves_ingestion_provenance():
    pipeline = FakePipeline()
    processor = JudgmentProcessor(
        pipeline=pipeline,
        chunk_size=900,
        chunk_overlap=120,
        chunker=fake_chunker,
    )

    processed = processor.process_pdf("appeal.pdf")
    doc = processed.documents[0]

    assert pipeline.paths == [Path("appeal.pdf")]
    assert doc.page_content == "Relevant judgment passage"
    assert doc.metadata["source_type"] == "judgment"
    assert doc.metadata["chunk_id"] == "deadbeef:p3:c0"
    assert doc.metadata["document_id"] == "deadbeef"
    assert doc.metadata["page_start"] == 3
    assert doc.metadata["char_start"] == 10
    assert doc.metadata["case_number"] == "Criminal Appeal No. 3346 of 2022"
    assert doc.metadata["case_id"] == "Criminal Appeal No. 3346 of 2022"
    assert doc.metadata["judges"] == "Justice A | Justice B"
    assert doc.metadata["citations"] == "75 DLR 123"
    assert doc.metadata["encoding_kind"] == "unicode_bangla"


def test_process_directory_accepts_pdf_suffix_case_insensitively(tmp_path):
    (tmp_path / "b.PDF").write_bytes(b"placeholder")
    (tmp_path / "a.pdf").write_bytes(b"placeholder")
    (tmp_path / "ignore.txt").write_text("not a pdf")

    pipeline = FakePipeline()
    processor = JudgmentProcessor(
        pipeline=pipeline,
        chunk_size=900,
        chunk_overlap=120,
        chunker=fake_chunker,
    )

    processed = processor.process_directory(tmp_path)

    assert [item.source_path.name for item in processed] == ["a.pdf", "b.PDF"]
