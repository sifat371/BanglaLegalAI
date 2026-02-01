"""
Case processor for parsing case study markdown files.
Extracts structured metadata and prepares documents for vector storage.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from src.config import get_settings


class CaseProcessor:
    """Processor for case study markdown files."""
    
    def __init__(self):
        """Initialize the case processor."""
        self.settings = get_settings()
    
    def parse_case_markdown(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse markdown content to extract individual cases.
        
        Args:
            content: Markdown file content
            
        Returns:
            List of case dictionaries
        """
        cases = []
        
        # Split by case headers (## Case)
        case_blocks = re.split(r'^##\s+Case\s+\d+', content, flags=re.MULTILINE)
        
        for block in case_blocks:
            if not block.strip():
                continue
            
            case_data = {}
            
            # Extract fields using regex
            patterns = {
                'case_id': r'\*\*case_id:\*\*\s*(.+?)$',
                'case_title': r'\*\*case_title:\*\*\s*(.+?)$',
                'jurisdiction': r'\*\*jurisdiction:\*\*\s*(.+?)$',
                'court_level': r'\*\*court_level:\*\*\s*(.+?)$',
                'area_of_law': r'\*\*area_of_law:\*\*\s*(.+?)$',
                'facts': r'\*\*facts:\*\*\s*(.+?)(?=\n\n|\*\*)',
                'legal_issues': r'\*\*legal_issues:\*\*\s*\n\n(.+?)(?=\n\n\*\*)',
                'laws_cited': r'\*\*laws_cited:\*\*\s*\n\n(.+?)(?=\n\n\*\*)',
                'arguments_plaintiff': r'\*\*arguments_plaintiff:\*\*\s*(.+?)(?=\n\n|\*\*)',
                'arguments_defendant': r'\*\*arguments_defendant:\*\*\s*(.+?)(?=\n\n|\*\*)',
                'verdict': r'\*\*verdict:\*\*\s*(.+?)$',
                'reasoning': r'\*\*reasoning:\*\*\s*(.+?)(?=\n\n|\*\*)',
                'penalty_or_relief': r'\*\*penalty_or_relief:\*\*\s*(.+?)$',
                'precedent_value': r'\*\*precedent_value:\*\*\s*(.+?)$',
                'decision_date': r'\*\*decision_date:\*\*\s*(.+?)$',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, block, re.MULTILINE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    # Clean up list items
                    if key in ['legal_issues', 'laws_cited']:
                        # Extract list items
                        items = re.findall(r'\*\s+(.+?)$', value, re.MULTILINE)
                        case_data[key] = items if items else [value]
                    else:
                        case_data[key] = value
            
            if case_data.get('case_id'):
                cases.append(case_data)
        
        return cases
    
    def process_case_file(self, file_path: Path) -> List[Document]:
        """
        Process a single case study markdown file.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            List of LangChain Documents
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
        cases = self.parse_case_markdown(content)
        documents = []
        
        for case in cases:
            # Build comprehensive case content
            content_parts = []
            
            # Case header
            case_title = case.get('case_title', 'Unknown')
            case_id = case.get('case_id', '')
            content_parts.append(f"Case: {case_title} ({case_id})")
            
            # Facts
            if case.get('facts'):
                content_parts.append(f"\nFacts: {case['facts']}")
            
            # Legal Issues
            if case.get('legal_issues'):
                issues = case['legal_issues']
                if isinstance(issues, list):
                    content_parts.append(f"\nLegal Issues:\n" + "\n".join(f"- {issue}" for issue in issues))
                else:
                    content_parts.append(f"\nLegal Issues: {issues}")
            
            # Laws Cited
            if case.get('laws_cited'):
                laws = case['laws_cited']
                if isinstance(laws, list):
                    content_parts.append(f"\nLaws Cited:\n" + "\n".join(f"- {law}" for law in laws))
                else:
                    content_parts.append(f"\nLaws Cited: {laws}")
            
            # Arguments
            if case.get('arguments_plaintiff'):
                content_parts.append(f"\nPlaintiff Arguments: {case['arguments_plaintiff']}")
            if case.get('arguments_defendant'):
                content_parts.append(f"\nDefendant Arguments: {case['arguments_defendant']}")
            
            # Verdict and Reasoning
            if case.get('verdict'):
                content_parts.append(f"\nVerdict: {case['verdict']}")
            if case.get('reasoning'):
                content_parts.append(f"\nReasoning: {case['reasoning']}")
            if case.get('penalty_or_relief'):
                content_parts.append(f"\nPenalty/Relief: {case['penalty_or_relief']}")
            
            # Combine all parts
            full_content = "\n".join(content_parts)
            
            # Extract laws_cited as list for metadata
            laws_cited_list = case.get('laws_cited', [])
            if isinstance(laws_cited_list, str):
                laws_cited_list = [laws_cited_list]
            
            # Create document
            doc = Document(
                page_content=full_content,
                metadata={
                    "source_type": "case_study",
                    "case_id": case_id,
                    "case_title": case_title,
                    "jurisdiction": case.get('jurisdiction', ''),
                    "court_level": case.get('court_level', ''),
                    "area_of_law": case.get('area_of_law', ''),
                    "verdict": case.get('verdict', ''),
                    "precedent_value": case.get('precedent_value', ''),
                    "decision_date": case.get('decision_date', ''),
                    "laws_cited": ', '.join(laws_cited_list),
                    "file_path": str(file_path),
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_all_cases(self) -> List[Document]:
        """
        Process all case study markdown files.
        
        Returns:
            List of all case documents
        """
        cases_dir = self.settings.case_studies_dir
        md_files = sorted(cases_dir.glob('*.md'))
        
        all_documents = []
        
        print(f"Processing {len(md_files)} case study files...")
        
        for file_path in md_files:
            docs = self.process_case_file(file_path)
            all_documents.extend(docs)
            print(f"Processed {file_path.name}: {len(docs)} cases")
        
        print(f"✓ Completed: {len(md_files)} files → {len(all_documents)} case documents")
        
        return all_documents


def get_case_processor() -> CaseProcessor:
    """Get a CaseProcessor instance."""
    return CaseProcessor()
