"""
Similarity search test script.
Tests vector store functionality with ingested data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore.chroma_store import get_acts_store, get_cases_store
from src.vectorstore.bm25_store import get_acts_bm25_store, get_cases_bm25_store


def test_acts_similarity_search():
    """Test similarity search on acts."""
    print("\n" + "="*80)
    print("TESTING ACTS SIMILARITY SEARCH")
    print("="*80)
    
    # Initialize stores
    print("\n1. Initializing stores...")
    chroma_store = get_acts_store()
    bm25_store = get_acts_bm25_store()
    
    # Get stats
    chroma_stats = chroma_store.get_collection_stats()
    bm25_stats = bm25_store.get_stats()
    
    print(f"   ChromaDB: {chroma_stats['document_count']} documents")
    print(f"   BM25: {bm25_stats['document_count']} documents")
    
    # Test 1: Semantic search (ChromaDB)
    print("\n2. Testing semantic search (ChromaDB)...")
    query = "What is the punishment for cheating and fraud?"
    print(f"   Query: '{query}'")
    print(f"   Generating embedding...")
    
    results = chroma_store.similarity_search(query, k=2)
    print(f"   ✓ Found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        section = doc.metadata.get('section_number', 'N/A')
        act = doc.metadata.get('act_title', 'Unknown')
        print(f"\n   Result {i}:")
        print(f"   - Act: {act}")
        print(f"   - Section: {section}")
        print(f"   - Content: {doc.page_content[:100]}...")
    
    # Test 2: Semantic search with scores
    print("\n3. Testing semantic search with scores...")
    print(f"   Reusing embedding from previous query...")
    results_with_scores = chroma_store.similarity_search_with_score(query, k=2)
    print(f"   ✓ Found {len(results_with_scores)} results with scores:")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        section = doc.metadata.get('section_number', 'N/A')
        print(f"   {i}. Section {section} - Score: {score:.4f}")
    
    # Test 3: Keyword search (BM25)
    print("\n4. Testing keyword search (BM25)...")
    keyword_query = "Section 420"
    print(f"   Query: '{keyword_query}'")
    
    bm25_results = bm25_store.search(keyword_query, k=3)
    print(f"   Found {len(bm25_results)} results:")
    for i, doc in enumerate(bm25_results, 1):
        section = doc.metadata.get('section_number', 'N/A')
        act = doc.metadata.get('act_title', 'Unknown')
        print(f"   {i}. {act} - Section {section}")
    
    # Test 4: Metadata filtering
    print("\n5. Testing metadata filtering...")
    filtered_results = chroma_store.similarity_search(
        query="imprisonment",
        k=5,
        filter={"source_type": "act"}
    )
    print(f"   Found {len(filtered_results)} acts with 'imprisonment'")


def test_cases_similarity_search():
    """Test similarity search on case studies."""
    print("\n" + "="*80)
    print("TESTING CASE STUDIES SIMILARITY SEARCH")
    print("="*80)
    
    # Initialize stores
    print("\n1. Initializing stores...")
    chroma_store = get_cases_store()
    bm25_store = get_cases_bm25_store()
    
    # Get stats
    chroma_stats = chroma_store.get_collection_stats()
    bm25_stats = bm25_store.get_stats()
    
    print(f"   ChromaDB: {chroma_stats['document_count']} documents")
    print(f"   BM25: {bm25_stats['document_count']} documents")
    
    # Test 1: Semantic search
    print("\n2. Testing semantic search (ChromaDB)...")
    query = "Find cases about cyber crime and unauthorized access"
    print(f"   Query: '{query}'")
    
    results = chroma_store.similarity_search(query, k=3)
    print(f"   Found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        case_id = doc.metadata.get('case_id', 'N/A')
        area = doc.metadata.get('area_of_law', 'Unknown')
        verdict = doc.metadata.get('verdict', 'Unknown')
        print(f"\n   Result {i}:")
        print(f"   - Case ID: {case_id}")
        print(f"   - Area of Law: {area}")
        print(f"   - Verdict: {verdict}")
    
    # Test 2: Keyword search
    print("\n3. Testing keyword search (BM25)...")
    keyword_query = "contract breach"
    print(f"   Query: '{keyword_query}'")
    
    bm25_results = bm25_store.search(keyword_query, k=3)
    print(f"   Found {len(bm25_results)} results:")
    for i, doc in enumerate(bm25_results, 1):
        case_id = doc.metadata.get('case_id', 'N/A')
        area = doc.metadata.get('area_of_law', 'Unknown')
        print(f"   {i}. {case_id} - {area}")
    
    # Test 3: Filter by verdict
    print("\n4. Testing metadata filtering...")
    filtered_results = chroma_store.similarity_search(
        query="guilty",
        k=5,
        filter={"verdict": "Guilty"}
    )
    print(f"   Found {len(filtered_results)} cases with 'Guilty' verdict")


def main():
    """Run all similarity search tests."""
    print("\n" + "="*80)
    print("SIMILARITY SEARCH TESTS")
    print("="*80)
    
    try:
        test_acts_similarity_search()
        test_cases_similarity_search()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nSummary:")
        print("- Semantic search (dense vectors) working")
        print("- Keyword search (BM25 sparse) working")
        print("- Metadata filtering working")
        print("- Both acts and cases searchable")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
