"""
Configuration module for Law Buddy application.
Handles environment variables and application settings.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys
    mistral_api_key: str
    huggingface_api_key: str
    
    # Project paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    acts_dir: Path = data_dir / "acts"
    case_studies_dir: Path = data_dir / "case_studies"
    judgments_dir: Path = data_dir / "judgments"
    
    # Vector store settings
    chroma_persist_dir: Path = project_root / "chroma_db"
    bm25_persist_dir: Path = project_root / "bm25_index"
    
    # Embedding settings
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dimension: int = 1024
    
    # Chunk settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval settings
    top_k: int = 5
    hybrid_alpha: float = 0.7  # Weight for dense retrieval
    
    # LLM settings
    mistral_model: str = "mistral-large-latest"
    mistral_model_small: str = "mistral-small-latest"
    temperature: float = 0.1
    max_tokens: int = 2000
    
    # ChromaDB collection names
    acts_collection_name: str = "acts_sections"
    cases_collection_name: str = "case_studies"
    summaries_collection_name: str = "act_summaries"
    
    # Logging
    log_level: str = "INFO"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_persist_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


# Convenience accessors
def get_settings() -> Settings:
    """Get application settings."""
    return settings


def get_api_key(service: str) -> str:
    """Get API key for a specific service."""
    if service.lower() == "mistral":
        return settings.mistral_api_key
    elif service.lower() == "huggingface":
        return settings.huggingface_api_key
    else:
        raise ValueError(f"Unknown service: {service}")
