"""
Test script for hybrid retrieval (ChromaDB + BM25).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore.chroma_store import get_acts_store, get_cases_store
from src.vectorstore.bm25_store import get_acts_bm25_store, get_cases_bm25_store
from src.vectorstore.hybrid_retriever import get_hybrid_retriever


def print_results(results, title, with_scores=False):
    """Print retrieval results."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    if not results:
        print("No results found.")
        return
    
    for idx, item in enumerate(results, 1):
        if with_scores:
            doc, score = item
            print(f"\n{idx}. Score: {score:.4f}")
        else:
            doc = item
            print(f"\n{idx}.")
        
        # Print metadata
        metadata = doc.metadata
        print(f"   Source: {metadata.get('source_type', 'unknown')}")
        
        if metadata.get('source_type') == 'act':
            print(f"   Act: {metadata.get('act_title', 'N/A')}")
            print(f"   Section: {metadata.get('section_number', 'N/A')} - {metadata.get('section_title', 'N/A')}")
        elif metadata.get('source_type') == 'case_study':
            print(f"   Case: {metadata.get('case_title', 'N/A')}")
            print(f"   Court: {metadata.get('court_level', 'N/A')}")
            print(f"   Verdict: {metadata.get('verdict', 'N/A')}")
        
        # Print content preview
        content_preview = doc.page_content[:200].replace('\n', ' ')
        print(f"   Content: {content_preview}...")


def test_acts_hybrid_retrieval():
    """Test hybrid retrieval on acts."""
    print("\n" + "="*80)
    print("TESTING ACTS HYBRID RETRIEVAL")
    print("="*80)
    
    # Initialize stores
    chroma_store = get_acts_store()
    bm25_store = get_acts_bm25_store()
    
    # Get stats
    chroma_stats = chroma_store.get_collection_stats()
    bm25_stats = bm25_store.get_stats()
    
    print(f"\nChromaDB Acts: {chroma_stats['document_count']} documents")
    print(f"BM25 Acts: {bm25_stats['document_count']} documents")
    
    if chroma_stats['document_count'] == 0:
        print("\n⚠️  No acts in vector store. Run ingestion first:")
        print("   uv run python scripts/ingest_data.py --acts-only --limit 10 --yes")
        return
    
    # Create hybrid retriever
    hybrid_retriever = get_hybrid_retriever(chroma_store, bm25_store, alpha=0.7)
    
    # Test queries
    test_queries = [
        "What are the penalties for theft?",
        "Section 420 fraud",
        "Laws about property rights",
        "Contract formation requirements"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        # Test weighted fusion
        print("\n--- Weighted Fusion (alpha=0.7) ---")
        results_weighted = hybrid_retriever.retrieve_with_scores(
            query=query,
            k=3,
            method="weighted"
        )
        print_results(results_weighted, "Hybrid Results (Weighted)", with_scores=True)
        
        # Test RRF fusion
        print("\n--- Reciprocal Rank Fusion ---")
        results_rrf = hybrid_retriever.retrieve_with_scores(
            query=query,
            k=3,
            method="rrf"
        )
        print_results(results_rrf, "Hybrid Results (RRF)", with_scores=True)
        
        # Compare with pure dense
        print("\n--- Pure Dense (ChromaDB only) ---")
        dense_only = chroma_store.similarity_search_with_score(query, k=3)
        dense_only_formatted = [(doc, 1.0 / (1.0 + dist)) for doc, dist in dense_only]
        print_results(dense_only_formatted, "Dense Only", with_scores=True)
        
        # Compare with pure sparse
        print("\n--- Pure Sparse (BM25 only) ---")
        sparse_only = bm25_store.search_with_scores(query, k=3)
        print_results(sparse_only, "Sparse Only", with_scores=True)


def test_cases_hybrid_retrieval():
    """Test hybrid retrieval on case studies."""
    print("\n" + "="*80)
    print("TESTING CASE STUDIES HYBRID RETRIEVAL")
    print("="*80)
    
    # Initialize stores
    chroma_store = get_cases_store()
    bm25_store = get_cases_bm25_store()
    
    # Get stats
    chroma_stats = chroma_store.get_collection_stats()
    bm25_stats = bm25_store.get_stats()
    
    print(f"\nChromaDB Cases: {chroma_stats['document_count']} documents")
    print(f"BM25 Cases: {bm25_stats['document_count']} documents")
    
    if chroma_stats['document_count'] == 0:
        print("\n⚠️  No cases in vector store. Run ingestion first:")
        print("   uv run python scripts/ingest_data.py --cases-only --limit 10 --yes")
        return
    
    # Create hybrid retriever
    hybrid_retriever = get_hybrid_retriever(chroma_store, bm25_store, alpha=0.7)
    
    # Test queries
    test_queries = [
        "Cases about property disputes",
        "Murder conviction precedents",
        "Family court custody battles"
    ]
    
    for query in test_queries:
        results = hybrid_retriever.retrieve_with_scores(
            query=query,
            k=3,
            method="weighted"
        )
        print_results(results, f"Query: {query}", with_scores=True)


def test_metadata_filtering():
    """Test hybrid retrieval with metadata filters."""
    print("\n" + "="*80)
    print("TESTING HYBRID RETRIEVAL WITH METADATA FILTERS")
    print("="*80)
    
    # Initialize stores
    chroma_store = get_acts_store()
    bm25_store = get_acts_bm25_store()
    
    chroma_stats = chroma_store.get_collection_stats()
    if chroma_stats['document_count'] == 0:
        print("\n⚠️  No data in vector store. Run ingestion first.")
        return
    
    hybrid_retriever = get_hybrid_retriever(chroma_store, bm25_store)
    
    # Test with language filter
    query = "What are the penalties?"
    
    print(f"\nQuery: {query}")
    print("\n--- Filter: English only ---")
    results_en = hybrid_retriever.retrieve_with_scores(
        query=query,
        k=3,
        filter={"language": "english"}
    )
    print_results(results_en, "English Results", with_scores=True)
    
    print("\n--- Filter: Bengali only ---")
    results_bn = hybrid_retriever.retrieve_with_scores(
        query=query,
        k=3,
        filter={"language": "bengali"}
    )
    print_results(results_bn, "Bengali Results", with_scores=True)


def test_alpha_comparison():
    """Test different alpha values."""
    print("\n" + "="*80)
    print("TESTING DIFFERENT ALPHA VALUES")
    print("="*80)
    
    chroma_store = get_acts_store()
    bm25_store = get_acts_bm25_store()
    
    chroma_stats = chroma_store.get_collection_stats()
    if chroma_stats['document_count'] == 0:
        print("\n⚠️  No data in vector store. Run ingestion first.")
        return
    
    query = "Section 420 fraud penalties"
    
    alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for alpha in alpha_values:
        hybrid_retriever = get_hybrid_retriever(chroma_store, bm25_store, alpha=alpha)
        results = hybrid_retriever.retrieve_with_scores(query, k=3)
        
        print(f"\n{'='*80}")
        print(f"Alpha = {alpha} ({'Pure Sparse' if alpha == 0 else 'Pure Dense' if alpha == 1 else f'{int(alpha*100)}% Dense, {int((1-alpha)*100)}% Sparse'})")
        print(f"{'='*80}")
        print_results(results, f"Results", with_scores=True)


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("HYBRID RETRIEVAL TEST SUITE")
    print("="*80)
    
    try:
        # Test acts
        test_acts_hybrid_retrieval()
        
        # Test cases
        test_cases_hybrid_retrieval()
        
        # Test metadata filtering
        test_metadata_filtering()
        
        # Test alpha comparison
        test_alpha_comparison()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
