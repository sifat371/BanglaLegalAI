"""Data ingestion commands for BanglaLegalAI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

# Preserve the repository's current import layout when invoked as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.data_processing.act_processor import ActProcessor
from src.data_processing.case_processor import CaseProcessor
from src.data_processing.judgment_processor import JudgmentProcessor
from src.vectorstore.bm25_store import get_acts_bm25_store, get_cases_bm25_store
from src.vectorstore.chroma_store import get_acts_store, get_cases_store
from src.vectorstore.document_identity import get_document_id


def ingest_acts(limit: int | None = None, clear_existing: bool = False) -> None:
    """Ingest structured legal acts into the retrieval stores."""
    print("\n" + "=" * 80)
    print("INGESTING LEGAL ACTS")
    print("=" * 80)

    chroma_store = get_acts_store()
    bm25_store = get_acts_bm25_store()

    if clear_existing:
        print("Clearing existing act data...")
        chroma_store.delete_collection()
        bm25_store.clear()
        chroma_store = get_acts_store()

    processor = ActProcessor()
    documents = processor.process_all_acts(limit=limit)

    if not documents:
        print("No act documents to ingest.")
        return

    print(f"Generated {len(documents)} act documents")
    batch_size = 50
    for index in tqdm(range(0, len(documents), batch_size), desc="ChromaDB batches"):
        batch = documents[index : index + batch_size]
        chroma_store.add_documents(batch)

    bm25_store.add_documents(documents)
    bm25_store.save()
    _print_store_stats(chroma_store, bm25_store)


def ingest_case_studies(clear_existing: bool = False) -> None:
    """Ingest the repository's legacy structured case-study Markdown."""
    print("\n" + "=" * 80)
    print("INGESTING LEGACY CASE STUDIES")
    print("=" * 80)

    chroma_store = get_cases_store()
    bm25_store = get_cases_bm25_store()

    if clear_existing:
        print("Clearing the shared case-law indexes...")
        chroma_store.delete_collection()
        bm25_store.clear()
        chroma_store = get_cases_store()

    processor = CaseProcessor()
    documents = processor.process_all_cases()

    if not documents:
        print("No case-study documents to ingest.")
        return

    chroma_store.add_documents(documents)
    bm25_store.add_documents(documents)
    bm25_store.save()
    _print_store_stats(chroma_store, bm25_store)


def ingest_judgments(
    input_dir: Path | None = None,
    *,
    clear_existing: bool = False,
    recursive: bool = False,
) -> None:
    """Ingest real legal PDFs through BanglaLegalIngest."""
    print("\n" + "=" * 80)
    print("INGESTING COURT JUDGMENTS WITH BANGLALEGALINGEST")
    print("=" * 80)

    settings = get_settings()
    judgment_dir = input_dir or settings.judgments_dir

    if not judgment_dir.exists():
        print(f"Judgment directory does not exist; skipping: {judgment_dir}")
        return

    chroma_store = get_cases_store()
    bm25_store = get_cases_bm25_store()

    if clear_existing:
        print("Clearing the shared case-law indexes...")
        chroma_store.delete_collection()
        bm25_store.clear()
        chroma_store = get_cases_store()

    processor = JudgmentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=min(settings.chunk_overlap, settings.chunk_size - 1),
    )
    processed = processor.process_directory(judgment_dir, recursive=recursive)

    documents = [
        document
        for judgment in processed
        for document in judgment.documents
    ]

    if not documents:
        print(f"No PDF judgments found in {judgment_dir}")
        return

    ids = [get_document_id(document) for document in documents]
    chroma_store.add_documents(documents, ids=ids)
    bm25_store.add_documents(documents)
    bm25_store.save()

    warnings = sum(
        len(judgment.ingestion.diagnostics.warnings)
        for judgment in processed
    )
    print(
        f"Processed {len(processed)} PDFs into {len(documents)} provenance-preserving chunks "
        f"({warnings} ingestion warnings)."
    )
    _print_store_stats(chroma_store, bm25_store)


def _print_store_stats(chroma_store, bm25_store) -> None:
    chroma_stats = chroma_store.get_collection_stats()
    bm25_stats = bm25_store.get_stats()
    print(f"ChromaDB: {chroma_stats['document_count']} documents")
    print(f"BM25: {bm25_stats['document_count']} documents")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Bangladesh legal data")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--acts-only", action="store_true")
    source_group.add_argument(
        "--cases-only",
        action="store_true",
        help="Ingest only the legacy Markdown case-study dataset",
    )
    source_group.add_argument(
        "--judgments-only",
        action="store_true",
        help="Ingest only real court PDFs through BanglaLegalIngest",
    )

    parser.add_argument(
        "--judgments-dir",
        type=Path,
        help="PDF directory for judgment ingestion (default: data/judgments)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search the judgment directory for PDFs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of act files processed",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the relevant retrieval index before ingestion",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the destructive-operation confirmation",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("BANGLALEGALAI - DATA INGESTION")
    print("=" * 80)

    if args.clear and not args.yes:
        try:
            response = input(
                "\nWARNING: --clear deletes the selected persisted index. Continue? (yes/no): "
            )
        except EOFError:
            print("No input available. Use --yes to skip confirmation.")
            return
        if response.lower() != "yes":
            print("Aborted.")
            return

    try:
        if args.judgments_only:
            ingest_judgments(
                args.judgments_dir,
                clear_existing=args.clear,
                recursive=args.recursive,
            )
        elif args.cases_only:
            ingest_case_studies(clear_existing=args.clear)
        elif args.acts_only:
            ingest_acts(limit=args.limit, clear_existing=args.clear)
        else:
            ingest_acts(limit=args.limit, clear_existing=args.clear)
            ingest_case_studies(clear_existing=args.clear)
            ingest_judgments(
                args.judgments_dir,
                clear_existing=False,
                recursive=args.recursive,
            )

        print("\n" + "=" * 80)
        print("ALL REQUESTED INGESTION COMPLETE")
        print("=" * 80 + "\n")
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user.")
    except Exception as exc:
        print(f"\nError during ingestion: {exc}")
        raise


if __name__ == "__main__":
    main()
