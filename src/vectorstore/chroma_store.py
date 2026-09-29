"""
ChromaDB vector store for Law Buddy.
Handles storage and retrieval of document embeddings.
"""

from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

from src.config import get_settings
from src.vectorstore.document_identity import get_document_id
from src.vectorstore.embeddings import get_embedding_service


class ChromaStore:
    """ChromaDB vector store manager."""
    
    def __init__(self, collection_name: str):
        """
        Initialize ChromaDB store.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        settings = get_settings()
        self.collection_name = collection_name
        self.persist_directory = str(settings.chroma_persist_dir)
        
        # Get embedding service
        self.embedding_service = get_embedding_service()
        self.embeddings = self.embedding_service.get_langchain_embeddings()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize LangChain Chroma wrapper
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Documents to add
            ids: Optional list of IDs for the documents
            batch_size: Number of documents to process at once
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Filter complex metadata from documents
        filtered_docs = filter_complex_metadata(documents)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [get_document_id(doc) for doc in filtered_docs]
        
        # Add in batches
        all_ids = []
        for i in range(0, len(filtered_docs), batch_size):
            batch_docs = filtered_docs[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            added_ids = self.vectorstore.add_documents(
                documents=batch_docs,
                ids=batch_ids
            )
            all_ids.extend(added_ids)
            
            print(f"Added batch {i // batch_size + 1}: {len(batch_docs)} documents")
        
        return all_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter
            
        Returns:
            List of similar documents
        """
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        return results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            result = collection.get(ids=[doc_id])
            
            if result['documents']:
                return Document(
                    page_content=result['documents'][0],
                    metadata=result['metadatas'][0] if result['metadatas'] else {}
                )
            return None
        except Exception as e:
            print(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "error": str(e)
            }
    
    def as_retriever(self, **kwargs):
        """
        Get a LangChain retriever interface.
        
        Args:
            **kwargs: Arguments to pass to the retriever
            
        Returns:
            LangChain retriever
        """
        return self.vectorstore.as_retriever(**kwargs)


def get_acts_store() -> ChromaStore:
    """Get ChromaStore for legal acts."""
    settings = get_settings()
    return ChromaStore(collection_name=settings.acts_collection_name)


def get_cases_store() -> ChromaStore:
    """Get ChromaStore for case studies."""
    settings = get_settings()
    return ChromaStore(collection_name=settings.cases_collection_name)


def get_summaries_store() -> ChromaStore:
    """Get ChromaStore for act summaries."""
    settings = get_settings()
    return ChromaStore(collection_name=settings.summaries_collection_name)
