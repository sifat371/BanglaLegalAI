"""
Hybrid retriever combining dense (ChromaDB) and sparse (BM25) retrieval.
Implements Reciprocal Rank Fusion (RRF) for score combination.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from langchain_core.documents import Document

from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.document_identity import get_document_id
from src.vectorstore.bm25_store import BM25Store
from src.config import get_settings


class HybridRetriever:
    """
    Hybrid retriever combining dense vector search and sparse BM25 search.
    
    Uses weighted score fusion to combine results from both retrievers.
    """
    
    def __init__(
        self,
        chroma_store: ChromaStore,
        bm25_store: BM25Store,
        alpha: float = 0.7
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            chroma_store: Dense vector store (ChromaDB)
            bm25_store: Sparse keyword store (BM25)
            alpha: Weight for dense retrieval (0-1). 
                   alpha=1.0 means only dense, alpha=0.0 means only sparse
        """
        self.chroma_store = chroma_store
        self.bm25_store = bm25_store
        self.alpha = alpha
        
        # Validate alpha
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            scores: List of raw scores
            
        Returns:
            List of normalized scores
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        # Avoid division by zero
        if max_score - min_score == 0:
            return [1.0] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    def _reciprocal_rank_fusion(
        self,
        doc_scores: Dict[str, List[int]],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score = sum(1 / (k + rank_i)) for each retriever i
        
        Args:
            doc_scores: Dict mapping doc_id to list of ranks from each retriever
            k: Constant for RRF (default 60 as per paper)
            
        Returns:
            List of (doc_id, rrf_score) tuples sorted by score
        """
        rrf_scores = {}
        
        for doc_id, ranks in doc_scores.items():
            rrf_score = sum(1.0 / (k + rank) for rank in ranks)
            rrf_scores[doc_id] = rrf_score
        
        # Sort by RRF score descending
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_docs
    
    def _weighted_score_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        Combine dense and sparse results using weighted score fusion.
        
        Args:
            dense_results: Results from ChromaDB with scores
            sparse_results: Results from BM25 with scores
            
        Returns:
            Combined and sorted results
        """
        # Extract scores and normalize
        dense_scores = [score for _, score in dense_results]
        sparse_scores = [score for _, score in sparse_results]
        
        # Normalize scores to [0, 1]
        norm_dense_scores = self._normalize_scores(dense_scores)
        norm_sparse_scores = self._normalize_scores(sparse_scores)
        
        # Create doc_id to (document, score) mapping
        doc_map: Dict[str, Tuple[Document, float, float]] = {}
        
        # Add dense results
        for doc, norm_score in zip([d for d, _ in dense_results], norm_dense_scores):
            doc_id = self._get_doc_id(doc)
            doc_map[doc_id] = (doc, norm_score, 0.0)
        
        # Add/update with sparse results
        for doc, norm_score in zip([d for d, _ in sparse_results], norm_sparse_scores):
            doc_id = self._get_doc_id(doc)
            if doc_id in doc_map:
                existing_doc, dense_score, _ = doc_map[doc_id]
                doc_map[doc_id] = (existing_doc, dense_score, norm_score)
            else:
                doc_map[doc_id] = (doc, 0.0, norm_score)
        
        # Calculate weighted combined scores
        combined_results = []
        for doc_id, (doc, dense_score, sparse_score) in doc_map.items():
            combined_score = self.alpha * dense_score + (1 - self.alpha) * sparse_score
            combined_results.append((doc, combined_score))
        
        # Sort by combined score descending
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results
    
    def _get_doc_id(self, doc: Document) -> str:
        """Return the shared deterministic ID used by all retrieval backends."""
        return get_document_id(doc)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        method: str = "weighted"
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter to apply
            method: Fusion method - "weighted" or "rrf" (Reciprocal Rank Fusion)
            
        Returns:
            List of retrieved documents
        """
        results_with_scores = self.retrieve_with_scores(
            query=query,
            k=k,
            filter=filter,
            method=method
        )
        
        return [doc for doc, _ in results_with_scores]
    
    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        method: str = "weighted"
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with scores using hybrid search.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter to apply
            method: Fusion method - "weighted" or "rrf"
            
        Returns:
            List of (document, score) tuples
        """
        # Retrieve from both stores (get more than k for fusion)
        retrieval_k = k * 2
        
        # Dense retrieval (ChromaDB)
        # Note: ChromaDB returns distance, lower is better
        # We need to convert to similarity score
        dense_results = self.chroma_store.similarity_search_with_score(
            query=query,
            k=retrieval_k,
            filter=filter
        )
        
        # Convert ChromaDB distance to similarity (invert)
        # ChromaDB uses L2 distance, convert to similarity score
        dense_results = [
            (doc, 1.0 / (1.0 + distance))
            for doc, distance in dense_results
        ]
        
        # Sparse retrieval (BM25)
        sparse_results = self.bm25_store.search_with_scores(
            query=query,
            k=retrieval_k,
            filter=filter
        )
        
        # Combine results based on method
        if method == "rrf":
            combined_results = self._rrf_fusion(dense_results, sparse_results)
        else:  # weighted
            combined_results = self._weighted_score_fusion(dense_results, sparse_results)
        
        # Return top k
        return combined_results[:k]
    
    def _rrf_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            
        Returns:
            Combined results with RRF scores
        """
        # Build doc_id to document mapping and rank lists
        doc_map: Dict[str, Document] = {}
        doc_ranks: Dict[str, List[int]] = {}
        
        # Process dense results (rank 0 is best)
        for rank, (doc, _) in enumerate(dense_results):
            doc_id = self._get_doc_id(doc)
            doc_map[doc_id] = doc
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = []
            doc_ranks[doc_id].append(rank)
        
        # Process sparse results
        for rank, (doc, _) in enumerate(sparse_results):
            doc_id = self._get_doc_id(doc)
            doc_map[doc_id] = doc
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = []
            doc_ranks[doc_id].append(rank)
        
        # Calculate RRF scores
        rrf_scores = self._reciprocal_rank_fusion(doc_ranks, k=60)
        
        # Build result list
        results = [(doc_map[doc_id], score) for doc_id, score in rrf_scores]
        
        return results
    
    def update_alpha(self, new_alpha: float):
        """
        Update the weight for dense retrieval.
        
        Args:
            new_alpha: New alpha value (0-1)
        """
        if not 0 <= new_alpha <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {new_alpha}")
        self.alpha = new_alpha


def get_hybrid_retriever(
    chroma_store: ChromaStore,
    bm25_store: BM25Store,
    alpha: Optional[float] = None
) -> HybridRetriever:
    """
    Create a hybrid retriever with given stores.
    
    Args:
        chroma_store: ChromaDB vector store
        bm25_store: BM25 sparse store
        alpha: Weight for dense retrieval (defaults to config value)
        
    Returns:
        Configured HybridRetriever
    """
    settings = get_settings()
    if alpha is None:
        alpha = settings.hybrid_alpha
    
    return HybridRetriever(
        chroma_store=chroma_store,
        bm25_store=bm25_store,
        alpha=alpha
    )
