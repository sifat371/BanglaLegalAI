"""
Embeddings module for Law Buddy.
Handles text embeddings using HuggingFace Inference API.
"""

import os
from typing import List, Optional
import time

from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from src.config import get_settings


class EmbeddingService:
    """Service for generating embeddings using HuggingFace API."""
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the embedding service.
        
        Args:
            model_name: HuggingFace model to use for embeddings
            api_key: HuggingFace API key
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.api_key = api_key or settings.huggingface_api_key
        self.dimension = settings.embedding_dimension
        
        # Initialize HuggingFace embeddings
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=self.model_name,
            huggingfacehub_api_token=self.api_key,
        )
        
        # Initialize inference client for direct API calls
        self.client = InferenceClient(token=self.api_key)
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            print(f"Error embedding text: {e}")
            raise
    
    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            
        Returns:
            List of embeddings
        """
        try:
            # Process in batches to avoid rate limits
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(embeddings)
                
                # Small delay to avoid rate limiting
                if i + batch_size < len(texts):
                    time.sleep(0.5)
            
            return all_embeddings
        except Exception as e:
            print(f"Error embedding documents: {e}")
            raise
    
    def get_langchain_embeddings(self) -> HuggingFaceEndpointEmbeddings:
        """
        Get the LangChain embeddings instance for direct use.
        
        Returns:
            HuggingFaceEndpointEmbeddings instance
        """
        return self.embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.dimension


def get_embedding_service() -> EmbeddingService:
    """Get a singleton embedding service instance."""
    if not hasattr(get_embedding_service, "_instance"):
        get_embedding_service._instance = EmbeddingService()
    return get_embedding_service._instance
