"""
Act processor for parsing Bangladesh legal acts from JSON files.
Handles chunking, metadata extraction, and document preparation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings


class ActProcessor:
    """Processor for legal acts JSON files."""
    
    def __init__(self):
        """Initialize the act processor."""
        self.settings = get_settings()
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_act_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load a single act JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Parsed JSON data or None if error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def extract_section_number(self, section_content: str) -> Optional[str]:
        """
        Extract section number from section content.
        
        Args:
            section_content: Section text
            
        Returns:
            Section number if found
        """
        # Look for patterns like "420." or "420:" at start
        match = re.match(r'^(\d+[A-Za-z]?)\s*[.:]', section_content.strip())
        if match:
            return match.group(1)
        return None
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection for Bengali vs English.
        
        Args:
            text: Text to analyze
            
        Returns:
            "bengali", "english", or "mixed"
        """
        if not text:
            return "english"
        
        # Count Bengali unicode characters
        bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        total_chars = len(text.strip())
        
        if total_chars == 0:
            return "english"
        
        bengali_ratio = bengali_chars / total_chars
        
        if bengali_ratio > 0.5:
            return "bengali"
        elif bengali_ratio > 0.1:
            return "mixed"
        else:
            return "english"
    
    def process_act(self, act_data: Dict[str, Any], file_path: Path) -> List[Document]:
        """
        Process a single act into chunked documents.
        
        Args:
            act_data: Parsed act JSON data
            file_path: Original file path
            
        Returns:
            List of LangChain Documents
        """
        documents = []
        
        # Extract act-level metadata
        act_title = act_data.get('act_title', 'Unknown')
        act_no = act_data.get('act_no', '')
        act_year = act_data.get('act_year', '')
        
        # Try to parse year as integer
        try:
            act_year_int = int(str(act_year))
        except (ValueError, TypeError):
            act_year_int = 0
        
        # Extract government and legal context
        gov_context = act_data.get('government_context', {})
        legal_context = act_data.get('legal_system_context', {})
        
        # Detect language
        language = self.detect_language(act_title)
        
        # Process sections
        sections = act_data.get('sections', [])
        
        for idx, section in enumerate(sections):
            section_title = section.get('section_title', '')
            section_content = section.get('section_content', '')
            
            if not section_content:
                continue
            
            # Extract section number
            section_number = self.extract_section_number(section_content)
            if not section_number and section_title:
                section_number = self.extract_section_number(section_title)
            
            # Combine title and content
            full_content = ""
            if section_title:
                full_content = f"{section_title}\n\n{section_content}"
            else:
                full_content = section_content
            
            # Check if chunking is needed
            if len(full_content) > self.settings.chunk_size:
                # Split into chunks
                chunks = self.text_splitter.split_text(full_content)
                
                for chunk_idx, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source_type": "act",
                            "act_title": act_title,
                            "act_no": act_no,
                            "act_year": act_year_int,
                            "section_number": section_number or f"section_{idx}",
                            "section_title": section_title,
                            "language": language,
                            "chunk_index": chunk_idx,
                            "total_chunks": len(chunks),
                            "file_path": str(file_path),
                            "government_name": gov_context.get('government_name', ''),
                            "legal_framework": legal_context.get('legal_framework', ''),
                        }
                    )
                    documents.append(doc)
            else:
                # Single document
                doc = Document(
                    page_content=full_content,
                    metadata={
                        "source_type": "act",
                        "act_title": act_title,
                        "act_no": act_no,
                        "act_year": act_year_int,
                        "section_number": section_number or f"section_{idx}",
                        "section_title": section_title,
                        "language": language,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "file_path": str(file_path),
                        "government_name": gov_context.get('government_name', ''),
                        "legal_framework": legal_context.get('legal_framework', ''),
                    }
                )
                documents.append(doc)
        
        return documents
    
    def process_all_acts(self, limit: Optional[int] = None) -> List[Document]:
        """
        Process all act JSON files in the data directory.
        
        Args:
            limit: Optional limit on number of files to process
            
        Returns:
            List of all documents
        """
        acts_dir = self.settings.acts_dir
        json_files = sorted(acts_dir.glob('*.json'))
        
        if limit:
            json_files = json_files[:limit]
        
        all_documents = []
        
        print(f"Processing {len(json_files)} act files...")
        
        for idx, file_path in enumerate(json_files, 1):
            act_data = self.load_act_file(file_path)
            
            if act_data:
                docs = self.process_act(act_data, file_path)
                all_documents.extend(docs)
                
                if idx % 100 == 0:
                    print(f"Processed {idx}/{len(json_files)} files, {len(all_documents)} documents so far")
        
        print(f"✓ Completed: {len(json_files)} acts → {len(all_documents)} documents")
        
        return all_documents


def get_act_processor() -> ActProcessor:
    """Get an ActProcessor instance."""
    return ActProcessor()
