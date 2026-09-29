"""
BM25 sparse retrieval store for Law Buddy.
Provides exact keyword matching for legal terms and section numbers.
"""

from typing import List, Dict, Any, Optional, Tuple
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from src.config import get_settings
from src.vectorstore.document_identity import get_document_id


class BM25Store:
    """BM25 sparse retrieval store."""
    
    def __init__(self, collection_name: str):
        """
        Initialize BM25 store.
        
        Args:
            collection_name: Name of the collection (for file persistence)
        """
        settings = get_settings()
        self.collection_name = collection_name
        self.persist_directory = settings.bm25_persist_dir
        self.persist_path = self.persist_directory / f"{collection_name}_bm25.pkl"
        
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Document] = []
        self.tokenized_corpus: List[List[str]] = []
        
        # Load if exists
        if self.persist_path.exists():
            self.load()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization + lowercase
        # For legal text, we want to preserve things like "Section 420"
        tokens = text.lower().split()
        return tokens
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add or replace documents using stable document identities."""
        if not documents:
            return

        positions = {
            get_document_id(document): index
            for index, document in enumerate(self.documents)
        }
        added = 0
        replaced = 0

        for document in documents:
            document_id = get_document_id(document)
            if document_id in positions:
                self.documents[positions[document_id]] = document
                replaced += 1
            else:
                positions[document_id] = len(self.documents)
                self.documents.append(document)
                added += 1

        self.tokenized_corpus = [
            self._tokenize(document.page_content) for document in self.documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

        print(
            f"BM25 indexed {added} new documents and replaced {replaced}. "
            f"Total: {len(self.documents)}"
        )

    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search documents using BM25.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter (applied after retrieval)
            
        Returns:
            List of relevant documents
        """
        if self.bm25 is None or not self.documents:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k * 2]
        
        # Apply metadata filter if provided
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            
            # Apply filter
            if filter:
                if all(doc.metadata.get(key) == value for key, value in filter.items()):
                    results.append(doc)
            else:
                results.append(doc)
            
            if len(results) >= k:
                break
        
        return results
    
    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search documents with BM25 scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        if self.bm25 is None or not self.documents:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices with scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k * 2]
        
        # Apply metadata filter if provided
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = float(scores[idx])
            
            # Apply filter
            if filter:
                if all(doc.metadata.get(key) == value for key, value in filter.items()):
                    results.append((doc, score))
            else:
                results.append((doc, score))
            
            if len(results) >= k:
                break
        
        return results
    
    def save(self) -> None:
        """Save BM25 index to disk."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        data = {
            'documents': self.documents,
            'tokenized_corpus': self.tokenized_corpus,
            'bm25': self.bm25
        }
        
        with open(self.persist_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved BM25 index to {self.persist_path}")
    
    def load(self) -> None:
        """Load BM25 index from disk."""
        if not self.persist_path.exists():
            print(f"No saved BM25 index found at {self.persist_path}")
            return
        
        with open(self.persist_path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.tokenized_corpus = data['tokenized_corpus']
        self.bm25 = data['bm25']
        
        print(f"Loaded BM25 index with {len(self.documents)} documents")
    
    def clear(self) -> None:
        """Clear the index."""
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []
        
        if self.persist_path.exists():
            self.persist_path.unlink()
        
        print(f"Cleared BM25 index for {self.collection_name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        return {
            "collection_name": self.collection_name,
            "document_count": len(self.documents),
            "is_indexed": self.bm25 is not None
        }


def get_acts_bm25_store() -> BM25Store:
    """Get BM25Store for legal acts."""
    settings = get_settings()
    return BM25Store(collection_name=settings.acts_collection_name)


def get_cases_bm25_store() -> BM25Store:
    """Get BM25Store for case studies."""
    settings = get_settings()
    return BM25Store(collection_name=settings.cases_collection_name)
