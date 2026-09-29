"""
RAG Retrieval Chain for BanglaLegalAI.
Combines query classification, hybrid retrieval, and context building.
"""

from typing import Dict, Any, List, Optional
from langchain_core.documents import Document

from src.vectorstore.chroma_store import get_acts_store, get_cases_store
from src.vectorstore.bm25_store import get_acts_bm25_store, get_cases_bm25_store
from src.vectorstore.hybrid_retriever import get_hybrid_retriever
from src.chains.query_classifier import classify_query
from src.config import get_settings


class RetrievalChain:
    """
    RAG retrieval chain that orchestrates:
    1. Query classification
    2. Hybrid retrieval
    3. Context building
    """
    
    def __init__(
        self,
        use_llm_classifier: bool = True,
        alpha: Optional[float] = None
    ):
        """
        Initialize retrieval chain.
        
        Args:
            use_llm_classifier: Whether to use LLM for query classification
            alpha: Weight for dense retrieval (None = use config default)
        """
        settings = get_settings()
        self.use_llm_classifier = use_llm_classifier
        self.alpha = alpha or settings.hybrid_alpha
        
        # Initialize stores
        self.acts_chroma = get_acts_store()
        self.acts_bm25 = get_acts_bm25_store()
        self.cases_chroma = get_cases_store()
        self.cases_bm25 = get_cases_bm25_store()
        
        # Initialize hybrid retrievers
        self.acts_retriever = get_hybrid_retriever(
            self.acts_chroma,
            self.acts_bm25,
            alpha=self.alpha
        )
        self.cases_retriever = get_hybrid_retriever(
            self.cases_chroma,
            self.cases_bm25,
            alpha=self.alpha
        )
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        user_type: str = "public",
        include_classification: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User's natural language query
            k: Number of documents to retrieve
            user_type: "public" or "lawyer" (affects retrieval strategy)
            include_classification: Whether to include classification in result
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        # Step 1: Classify the query
        classification = classify_query(query, use_llm=self.use_llm_classifier)
        
        # Step 2: Determine retrieval strategy based on classification
        retrieval_plan = self._plan_retrieval(classification, k, user_type)
        
        # Step 3: Execute retrieval
        results = self._execute_retrieval(query, retrieval_plan)
        
        # Step 4: Build context
        context = self._build_context(results, classification)
        
        # Prepare response
        response = {
            "query": query,
            "documents": results["documents"],
            "context": context,
            "num_results": len(results["documents"]),
            "sources": self._extract_sources(results["documents"])
        }
        
        if include_classification:
            response["classification"] = classification
            response["retrieval_plan"] = retrieval_plan
        
        return response
    
    def _convert_to_chromadb_filter(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert flat metadata filters to ChromaDB's filter format.
        
        ChromaDB requires filters with operators like $eq, $and, etc.
        Input: {'source_type': 'act', 'area_of_law': 'Property'}
        Output: {'$and': [{'source_type': {'$eq': 'act'}}, {'area_of_law': {'$eq': 'Property'}}]}
        
        Args:
            filters: Flat dictionary of metadata filters
            
        Returns:
            ChromaDB-compatible filter dictionary or None
        """
        if not filters:
            return None
        
        # Remove source_type from filters as it's used to determine stores
        filters_copy = {k: v for k, v in filters.items() if k != "source_type"}
        
        if not filters_copy:
            return None
        
        def condition(key: str, value: Any) -> Dict[str, Any]:
            if isinstance(value, dict) and any(
                str(operator).startswith("$") for operator in value
            ):
                return {key: value}
            return {key: {"$eq": value}}

        # If only one filter, return simple format
        if len(filters_copy) == 1:
            key, value = list(filters_copy.items())[0]
            return condition(key, value)

        # Multiple filters - use $and
        conditions = [condition(key, value) for key, value in filters_copy.items()]
        return {"$and": conditions}
    
    def _plan_retrieval(
        self,
        classification: Dict[str, Any],
        k: int,
        user_type: str
    ) -> Dict[str, Any]:
        """
        Plan retrieval strategy based on classification.
        
        Args:
            classification: Query classification result
            k: Number of documents to retrieve
            user_type: User type
            
        Returns:
            Retrieval plan dictionary
        """
        intent = classification["intent"]
        search_strategy = classification["search_strategy"]
        metadata_filters = classification["metadata_filters"]
        specific_refs = classification["specific_references"]
        
        # Determine which stores to query
        source_type = metadata_filters.get("source_type")
        if source_type == "act":
            stores = ["acts"]
        elif source_type in {"case", "case_study", "judgment"} or intent == "CASE_SEARCH":
            stores = ["cases"]
        else:
            # Query both by default
            stores = ["acts", "cases"]
        
        # Convert metadata filters to ChromaDB format
        chromadb_filters = self._convert_to_chromadb_filter(metadata_filters)
        
        # Adjust k based on user type
        if user_type == "lawyer":
            # Lawyers want more comprehensive results
            retrieve_k = k * 2
        else:
            # Public users want focused results
            retrieve_k = k
        
        # Choose fusion method
        if search_strategy == "EXACT_MATCH":
            fusion_method = "weighted"
            alpha = 0.3  # Favor sparse (keyword) matching
        elif search_strategy == "SEMANTIC":
            fusion_method = "weighted"
            alpha = 0.9  # Favor dense (semantic) matching
        else:  # HYBRID
            fusion_method = "weighted"
            alpha = self.alpha  # Use default
        
        return {
            "stores": stores,
            "k": retrieve_k,
            "fusion_method": fusion_method,
            "alpha": alpha,
            "metadata_filters": chromadb_filters,
            "specific_references": specific_refs,
            "reformulated_query": classification.get("reformulated_query", "")
        }
    
    def _execute_retrieval(
        self,
        query: str,
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute retrieval based on plan.
        
        Args:
            query: User query
            plan: Retrieval plan
            
        Returns:
            Retrieved documents with scores
        """
        all_results = []
        
        # Use reformulated query if available
        search_query = plan.get("reformulated_query") or query
        
        # Update retriever alpha if needed
        if plan["alpha"] != self.alpha:
            if "acts" in plan["stores"]:
                self.acts_retriever.update_alpha(plan["alpha"])
            if "cases" in plan["stores"]:
                self.cases_retriever.update_alpha(plan["alpha"])
        
        # Retrieve from acts store
        if "acts" in plan["stores"]:
            acts_results = self.acts_retriever.retrieve_with_scores(
                query=search_query,
                k=plan["k"],
                filter=plan["metadata_filters"] or None,
                method=plan["fusion_method"]
            )
            all_results.extend([
                {"document": doc, "score": score, "source_store": "acts"}
                for doc, score in acts_results
            ])
        
        # Retrieve from cases store
        if "cases" in plan["stores"]:
            cases_results = self.cases_retriever.retrieve_with_scores(
                query=search_query,
                k=plan["k"],
                filter=plan["metadata_filters"] or None,
                method=plan["fusion_method"]
            )
            all_results.extend([
                {"document": doc, "score": score, "source_store": "cases"}
                for doc, score in cases_results
            ])
        
        # Sort by score and take top k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_results[:plan["k"]]
        
        # Reset alpha to default
        if plan["alpha"] != self.alpha:
            if "acts" in plan["stores"]:
                self.acts_retriever.update_alpha(self.alpha)
            if "cases" in plan["stores"]:
                self.cases_retriever.update_alpha(self.alpha)
        
        return {
            "documents": [r["document"] for r in top_results],
            "scores": [r["score"] for r in top_results],
            "source_stores": [r["source_store"] for r in top_results]
        }
    
    def _build_context(
        self,
        results: Dict[str, Any],
        classification: Dict[str, Any]
    ) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            results: Retrieval results
            classification: Query classification
            
        Returns:
            Formatted context string
        """
        documents = results["documents"]
        scores = results["scores"]
        
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        
        for idx, (doc, score) in enumerate(zip(documents, scores), 1):
            metadata = doc.metadata
            source_type = metadata.get("source_type", "unknown")
            
            # Format document based on type
            if source_type == "act":
                doc_header = f"[Document {idx}] ACT - {metadata.get('act_title', 'Unknown Act')}"
                doc_header += f"\nSection {metadata.get('section_number', 'N/A')}"
                if metadata.get('section_title'):
                    doc_header += f": {metadata['section_title']}"
                doc_header += f"\n(Relevance: {score:.3f})"
                
            elif source_type == "judgment":
                doc_header = f"[Document {idx}] JUDGMENT - {metadata.get('case_number', metadata.get('source_filename', 'Unknown Judgment'))}"
                doc_header += f"\nCourt: {metadata.get('court', 'N/A')}"
                doc_header += f"\nPage: {metadata.get('page_start', 'N/A')}"
                doc_header += f"\nSource: {metadata.get('source_filename', 'N/A')}"
                doc_header += f"\n(Relevance: {score:.3f})"

            elif source_type == "case_study":
                doc_header = f"[Document {idx}] CASE - {metadata.get('case_title', 'Unknown Case')}"
                doc_header += f"\nCase ID: {metadata.get('case_id', 'N/A')}"
                doc_header += f"\nCourt: {metadata.get('court_level', 'N/A')}"
                doc_header += f"\nVerdict: {metadata.get('verdict', 'N/A')}"
                doc_header += f"\n(Relevance: {score:.3f})"
            else:
                doc_header = f"[Document {idx}] (Relevance: {score:.3f})"
            
            # Add content
            content = f"\n\nContent:\n{doc.page_content}\n"
            
            context_parts.append(doc_header + content)
            context_parts.append("-" * 80)
        
        return "\n\n".join(context_parts)
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract source citations from documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            List of source dictionaries
        """
        sources = []
        
        for doc in documents:
            metadata = doc.metadata
            source_type = metadata.get("source_type", "unknown")
            
            if source_type == "act":
                source = {
                    "type": "act",
                    "title": metadata.get("act_title", "Unknown Act"),
                    "act_no": metadata.get("act_no", "N/A"),
                    "act_year": metadata.get("act_year", "N/A"),
                    "section": metadata.get("section_number", "N/A"),
                    "section_title": metadata.get("section_title", ""),
                    "citation": f"{metadata.get('act_title', 'Unknown')}, Section {metadata.get('section_number', 'N/A')}"
                }
            elif source_type == "judgment":
                case_number = metadata.get("case_number") or metadata.get("source_filename", "Unknown Judgment")
                page = metadata.get("page_start", "N/A")
                court = metadata.get("court", "N/A")
                source = {
                    "type": "judgment",
                    "title": case_number,
                    "case_number": metadata.get("case_number", ""),
                    "court": court,
                    "page": page,
                    "document_id": metadata.get("document_id", ""),
                    "chunk_id": metadata.get("chunk_id", ""),
                    "source_filename": metadata.get("source_filename", ""),
                    "citation": f"{case_number}, {court}, p. {page}",
                }
            elif source_type == "case_study":
                source = {
                    "type": "case",
                    "title": metadata.get("case_title", "Unknown Case"),
                    "case_id": metadata.get("case_id", "N/A"),
                    "court": metadata.get("court_level", "N/A"),
                    "verdict": metadata.get("verdict", "N/A"),
                    "area_of_law": metadata.get("area_of_law", "N/A"),
                    "citation": f"{metadata.get('case_title', 'Unknown')} ({metadata.get('case_id', 'N/A')})"
                }
            else:
                source = {
                    "type": "unknown",
                    "citation": "Unknown source"
                }
            
            sources.append(source)
        
        return sources
    
    def retrieve_by_section(
        self,
        section_number: str,
        act_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve a specific section by number.
        
        Args:
            section_number: Section number (e.g., "420")
            act_name: Optional act name to filter
            
        Returns:
            Retrieval result
        """
        filter_dict = {"section_number": section_number}
        if act_name:
            filter_dict["act_title"] = act_name
        
        results = self.acts_chroma.similarity_search(
            query=f"Section {section_number}",
            k=5,
            filter=filter_dict
        )
        
        return {
            "query": f"Section {section_number}" + (f" of {act_name}" if act_name else ""),
            "documents": results,
            "num_results": len(results),
            "sources": self._extract_sources(results)
        }
    
    def retrieve_by_case_id(self, case_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific case by ID.
        
        Args:
            case_id: Case ID (e.g., "BD-CR-001")
            
        Returns:
            Retrieval result
        """
        results = self.cases_chroma.similarity_search(
            query=case_id,
            k=5,
            filter={"case_id": case_id}
        )
        
        return {
            "query": f"Case {case_id}",
            "documents": results,
            "num_results": len(results),
            "sources": self._extract_sources(results)
        }


def get_retrieval_chain(
    use_llm_classifier: bool = True,
    alpha: Optional[float] = None
) -> RetrievalChain:
    """
    Get a configured retrieval chain.
    
    Args:
        use_llm_classifier: Whether to use LLM for classification
        alpha: Weight for dense retrieval
        
    Returns:
        RetrievalChain instance
    """
    return RetrievalChain(
        use_llm_classifier=use_llm_classifier,
        alpha=alpha
    )


# Quick test
if __name__ == "__main__":
    print("Testing RAG Retrieval Chain")
    print("=" * 80)
    
    # Initialize chain
    chain = get_retrieval_chain(use_llm_classifier=False)  # Use rule-based for speed
    
    # Test queries
    test_queries = [
        ("What are the penalties for theft?", "public"),
        ("Show me property dispute cases", "lawyer"),
        ("Section 11 of Societies Registration Act", "public"),
    ]
    
    for query, user_type in test_queries:
        print(f"\nQuery: {query}")
        print(f"User Type: {user_type}")
        print("-" * 80)
        
        result = chain.retrieve(
            query=query,
            k=3,
            user_type=user_type,
            include_classification=True
        )
        
        print(f"Classification: {result['classification']['intent']}")
        print(f"Retrieved: {result['num_results']} documents")
        print(f"\nSources:")
        for idx, source in enumerate(result['sources'], 1):
            print(f"  {idx}. {source['citation']}")
        
        print("\n" + "=" * 80)
