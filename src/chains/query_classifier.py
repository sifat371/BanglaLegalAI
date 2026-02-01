"""
Query classifier for extracting intent and metadata from user queries.
Uses LLM to intelligently parse natural language into structured filters.
"""

import json
import re
from typing import Dict, Any, Optional, List
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.config import get_settings
from src.prompts.system_prompts import QUERY_CLASSIFIER_PROMPT


class QueryClassifier:
    """
    Classifies user queries and extracts structured information.
    
    Identifies:
    - User intent (specific law, case search, general advice, etc.)
    - Search strategy (exact match, semantic, hybrid)
    - Metadata filters (year, court, language, etc.)
    - Specific references (section numbers, case IDs)
    """
    
    def __init__(self):
        """Initialize query classifier."""
        settings = get_settings()
        
        # Use smaller, faster model for classification
        self.llm = ChatMistralAI(
            model=settings.mistral_model_small,
            temperature=0.0,  # Deterministic classification
            api_key=settings.mistral_api_key
        )
        
        # Parser for JSON output
        self.json_parser = JsonOutputParser()
        
        # Prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_CLASSIFIER_PROMPT),
            ("user", "Query: {query}")
        ])
        
        # Create chain
        self.chain = self.prompt | self.llm | self.json_parser
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify a user query and extract structured information.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Use LLM to classify
            result = self.chain.invoke({"query": query})
            
            # Validate and clean the result
            result = self._validate_classification(result)
            
            return result
            
        except Exception as e:
            print(f"Error during LLM classification: {e}")
            # Fallback to rule-based classification
            return self._fallback_classification(query)
    
    def _validate_classification(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean the classification result.
        
        Args:
            result: Raw classification result from LLM
            
        Returns:
            Validated and cleaned result
        """
        # Ensure required fields exist
        validated = {
            "intent": result.get("intent", "GENERAL_ADVICE"),
            "search_strategy": result.get("search_strategy", "HYBRID"),
            "metadata_filters": result.get("metadata_filters", {}),
            "specific_references": result.get("specific_references", {
                "section_numbers": [],
                "case_ids": [],
                "act_names": []
            }),
            "reformulated_query": result.get("reformulated_query", "")
        }
        
        # Clean metadata filters (remove null values)
        validated["metadata_filters"] = {
            k: v for k, v in validated["metadata_filters"].items()
            if v is not None
        }
        
        return validated
    
    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """
        Rule-based fallback classification when LLM fails.
        
        Args:
            query: User query
            
        Returns:
            Basic classification result
        """
        query_lower = query.lower()
        
        # Detect section numbers
        section_numbers = self._extract_section_numbers(query)
        
        # Detect case IDs
        case_ids = self._extract_case_ids(query)
        
        # Determine intent
        intent = "GENERAL_ADVICE"
        if section_numbers or "section" in query_lower:
            intent = "SPECIFIC_LAW"
        elif "case" in query_lower or "precedent" in query_lower or case_ids:
            intent = "CASE_SEARCH"
        elif any(word in query_lower for word in ["penalty", "punishment", "fine", "jail", "imprisonment"]):
            intent = "PENALTIES"
        elif any(word in query_lower for word in ["right", "can i", "allowed", "legal"]):
            intent = "RIGHTS"
        elif any(word in query_lower for word in ["how to", "procedure", "process", "file", "apply"]):
            intent = "PROCEDURE"
        
        # Determine search strategy
        search_strategy = "HYBRID"
        if section_numbers or case_ids:
            search_strategy = "EXACT_MATCH"
        
        # Basic metadata filters
        metadata_filters = {}
        
        # Detect source type
        if intent == "CASE_SEARCH":
            metadata_filters["source_type"] = "case_study"
        elif intent == "SPECIFIC_LAW":
            metadata_filters["source_type"] = "act"
        
        # Detect court level
        for court in ["high court", "supreme court", "family court", "district court", "labor court"]:
            if court in query_lower:
                metadata_filters["court_level"] = court.title()
                break
        
        # Detect language preference
        # Check for Bengali characters
        if re.search(r'[\u0980-\u09FF]', query):
            metadata_filters["language"] = "bengali"
        
        return {
            "intent": intent,
            "search_strategy": search_strategy,
            "metadata_filters": metadata_filters,
            "specific_references": {
                "section_numbers": section_numbers,
                "case_ids": case_ids,
                "act_names": []
            },
            "reformulated_query": query
        }
    
    def _extract_section_numbers(self, query: str) -> List[str]:
        """
        Extract section numbers from query.
        
        Args:
            query: User query
            
        Returns:
            List of section numbers
        """
        # Pattern: "section 420", "sec 420", "section 302A"
        # Only match when explicitly preceded by "section" or "sec"
        patterns = [
            r'section\s+(\d+[A-Za-z]?)',
            r'sec\.?\s+(\d+[A-Za-z]?)',
        ]
        
        sections = []
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            sections.extend(matches)
        
        # Remove duplicates and return
        return list(set(sections))
    
    def _extract_case_ids(self, query: str) -> List[str]:
        """
        Extract case IDs from query.
        
        Args:
            query: User query
            
        Returns:
            List of case IDs
        """
        # Pattern: BD-XX-NNN (e.g., BD-CR-001)
        pattern = r'BD-[A-Z]{2}-\d{3}'
        case_ids = re.findall(pattern, query, re.IGNORECASE)
        return [cid.upper() for cid in case_ids]
    
    def extract_year_range(self, query: str) -> Optional[Dict[str, int]]:
        """
        Extract year range from query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with $gte and $lte or None
        """
        query_lower = query.lower()
        
        # Pattern: "from 1950 to 1960"
        range_match = re.search(r'from\s+(\d{4})\s+to\s+(\d{4})', query_lower)
        if range_match:
            return {
                "$gte": int(range_match.group(1)),
                "$lte": int(range_match.group(2))
            }
        
        # Pattern: "between 1950 and 1960"
        range_match = re.search(r'between\s+(\d{4})\s+and\s+(\d{4})', query_lower)
        if range_match:
            return {
                "$gte": int(range_match.group(1)),
                "$lte": int(range_match.group(2))
            }
        
        # Pattern: "before 2020"
        before_match = re.search(r'before\s+(\d{4})', query_lower)
        if before_match:
            return {
                "$lte": int(before_match.group(1))
            }
        
        # Pattern: "after 1980"
        after_match = re.search(r'after\s+(\d{4})', query_lower)
        if after_match:
            return {
                "$gte": int(after_match.group(1))
            }
        
        # Pattern: "in 1990"
        year_match = re.search(r'in\s+(\d{4})', query_lower)
        if year_match:
            year = int(year_match.group(1))
            return {
                "$gte": year,
                "$lte": year
            }
        
        return None


class SimpleQueryClassifier:
    """
    Lightweight rule-based query classifier.
    
    Use this when you don't need LLM-based classification
    or want to avoid API calls for testing.
    """
    
    def __init__(self):
        """Initialize simple classifier."""
        pass
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify query using simple rules.
        
        Args:
            query: User query
            
        Returns:
            Classification result
        """
        classifier = QueryClassifier()
        return classifier._fallback_classification(query)


def classify_query(query: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Convenience function to classify a query.
    
    Args:
        query: User's natural language query
        use_llm: Whether to use LLM or rule-based classification
        
    Returns:
        Classification result dictionary
    """
    if use_llm:
        classifier = QueryClassifier()
    else:
        classifier = SimpleQueryClassifier()
    
    return classifier.classify(query)


# Quick test
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "What is Section 420?",
        "Show me murder cases from 2020",
        "Can my landlord evict me without notice?",
        "Laws about property rights between 1950 and 1980",
        "Family court custody cases"
    ]
    
    print("Testing Query Classification")
    print("=" * 80)
    
    # Test with rule-based classifier (fast)
    classifier = SimpleQueryClassifier()
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = classifier.classify(query)
        print(json.dumps(result, indent=2))
        print("-" * 80)
