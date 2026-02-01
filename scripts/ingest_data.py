"""
Data ingestion script for Law Buddy.
Processes legal acts and case studies, generates embeddings, and populates vector stores.
"""

import sys
from pathlib import Path
from typing import Optional
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.data_processing.act_processor import ActProcessor
from src.data_processing.case_processor import CaseProcessor
from src.vectorstore.chroma_store import get_acts_store, get_cases_store
from src.vectorstore.bm25_store import get_acts_bm25_store, get_cases_bm25_store


def ingest_acts(limit: Optional[int] = None, clear_existing: bool = False):
    """
    Ingest legal acts into vector stores.
    
    Args:
        limit: Limit number of act files to process
        clear_existing: Whether to clear existing data
    """
    print("\n" + "="*80)
    print("INGESTING LEGAL ACTS")
    print("="*80)
    
    # Initialize stores
    print("\n1. Initializing vector stores...")
    chroma_store = get_acts_store()
    bm25_store = get_acts_bm25_store()
    
    if clear_existing:
        print("Clearing existing data...")
        chroma_store.delete_collection()
        bm25_store.clear()
        chroma_store = get_acts_store()  # Reinitialize
    
    # Process acts
    print("\n2. Processing act files...")
    processor = ActProcessor()
    documents = processor.process_all_acts(limit=limit)
    
    if not documents:
        print("No documents to ingest!")
        return
    
    print(f"\n3. Generated {len(documents)} documents from acts")
    
    # Add to ChromaDB with progress bar
    print("\n4. Adding to ChromaDB (with embeddings)...")
    print("   This will take time as embeddings are generated via API...")
    
    batch_size = 50
    for i in tqdm(range(0, len(documents), batch_size), desc="ChromaDB batches"):
        batch = documents[i:i + batch_size]
        chroma_store.add_documents(batch)
    
    # Add to BM25
    print("\n5. Adding to BM25 store...")
    bm25_store.add_documents(documents)
    bm25_store.save()
    
    # Show stats
    print("\n6. Ingestion complete!")
    chroma_stats = chroma_store.get_collection_stats()
    bm25_stats = bm25_store.get_stats()
    
    print(f"\nChromaDB: {chroma_stats['document_count']} documents")
    print(f"BM25: {bm25_stats['document_count']} documents")
    print("="*80)


def ingest_cases(clear_existing: bool = False):
    """
    Ingest case studies into vector stores.
    
    Args:
        clear_existing: Whether to clear existing data
    """
    print("\n" + "="*80)
    print("INGESTING CASE STUDIES")
    print("="*80)
    
    # Initialize stores
    print("\n1. Initializing vector stores...")
    chroma_store = get_cases_store()
    bm25_store = get_cases_bm25_store()
    
    if clear_existing:
        print("Clearing existing data...")
        chroma_store.delete_collection()
        bm25_store.clear()
        chroma_store = get_cases_store()  # Reinitialize
    
    # Process cases
    print("\n2. Processing case study files...")
    processor = CaseProcessor()
    documents = processor.process_all_cases()
    
    if not documents:
        print("No case documents to ingest!")
        return
    
    print(f"\n3. Generated {len(documents)} case documents")
    
    # Add to ChromaDB
    print("\n4. Adding to ChromaDB (with embeddings)...")
    chroma_store.add_documents(documents)
    
    # Add to BM25
    print("\n5. Adding to BM25 store...")
    bm25_store.add_documents(documents)
    bm25_store.save()
    
    # Show stats
    print("\n6. Ingestion complete!")
    chroma_stats = chroma_store.get_collection_stats()
    bm25_stats = bm25_store.get_stats()
    
    print(f"\nChromaDB: {chroma_stats['document_count']} documents")
    print(f"BM25: {bm25_stats['document_count']} documents")
    print("="*80)


def main():
    """Main ingestion function."""
    parser = argparse.ArgumentParser(description="Ingest legal data into vector stores")
    parser.add_argument(
        '--acts-only',
        action='store_true',
        help='Ingest only legal acts'
    )
    parser.add_argument(
        '--cases-only',
        action='store_true',
        help='Ingest only case studies'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of act files to process (for testing)'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing data before ingestion'
    )
    parser.add_argument(
        '--yes',
        '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("LAW BUDDY - DATA INGESTION")
    print("="*80)
    
    if args.clear and not args.yes:
        print("\n⚠️  WARNING: This will clear existing data!")
        try:
            response = input("Continue? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted.")
                return
        except EOFError:
            print("No input available. Use --yes to skip confirmation.")
            return
    
    try:
        if args.cases_only:
            ingest_cases(clear_existing=args.clear)
        elif args.acts_only:
            ingest_acts(limit=args.limit, clear_existing=args.clear)
        else:
            # Ingest both
            ingest_acts(limit=args.limit, clear_existing=args.clear)
            ingest_cases(clear_existing=args.clear)
        
        print("\n" + "="*80)
        print("✓ ALL INGESTION COMPLETE")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nIngestion interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
