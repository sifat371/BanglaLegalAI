"""
Research Agent for Law Buddy.
Provides comprehensive legal research for lawyers and legal professionals.
"""

from typing import Dict, Any, Optional, List

from src.agents.base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    """
    Agent for lawyers and legal professionals.
    
    Features:
    - Comprehensive analysis
    - More documents (extensive research)
    - Detailed citations
    - Case law and precedents
    - No simplified language
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize research agent.
        
        Args:
            session_id: Optional session identifier
        """
        super().__init__(
            user_type="lawyer",
            use_llm_classifier=True,
            session_id=session_id
        )
        
        # Override default k for lawyers (more comprehensive results)
        self.default_k = 10
    
    def chat(
        self,
        query: str,
        k: Optional[int] = None,
        include_followups: bool = True,
        include_confidence: bool = False,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Process a query with comprehensive research settings.
        
        Args:
            query: Research question
            k: Number of documents to retrieve (defaults to 10)
            include_followups: Whether to generate follow-up questions
            include_confidence: Whether to assess confidence (usually not needed for lawyers)
            verbose: Whether to include debug information
            
        Returns:
            Comprehensive research response
        """
        k = k or self.default_k
        
        response = super().chat(
            query=query,
            k=k,
            include_followups=include_followups,
            include_confidence=include_confidence,
            verbose=verbose
        )
        
        # Add research metadata
        response['research_depth'] = 'comprehensive' if k >= 10 else 'standard'
        
        return response
    
    def research_statute(
        self,
        topic: str,
        year_range: Optional[tuple] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Research statutes on a specific topic.
        
        Args:
            topic: Legal topic to research
            year_range: Optional tuple of (start_year, end_year)
            language: Optional language filter ("english", "bengali")
            
        Returns:
            Comprehensive statute research
        """
        query = f"Find all statutes and legal provisions related to {topic}"
        
        if year_range:
            query += f" from {year_range[0]} to {year_range[1]}"
        
        if language:
            query += f" in {language}"
        
        return self.chat(query, k=15)
    
    def find_precedents(
        self,
        legal_issue: str,
        court_level: Optional[str] = None,
        verdict: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find case law precedents for a legal issue.
        
        Args:
            legal_issue: The legal issue to research
            court_level: Optional court level filter
            verdict: Optional verdict filter
            
        Returns:
            Case law research results
        """
        query = f"Find case law precedents related to {legal_issue}"
        
        if court_level:
            query += f" from {court_level}"
        
        if verdict:
            query += f" with {verdict} verdict"
        
        return self.chat(query, k=10)
    
    def analyze_legal_question(
        self,
        question: str,
        include_both_sides: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive legal analysis of a question.
        
        Args:
            question: Legal question to analyze
            include_both_sides: Whether to include arguments for both sides
            
        Returns:
            Detailed legal analysis
        """
        if include_both_sides:
            query = f"Provide a comprehensive legal analysis of: {question}. Include arguments for both sides."
        else:
            query = f"Provide a comprehensive legal analysis of: {question}"
        
        return self.chat(query, k=10, include_followups=True)
    
    def compare_sections(
        self,
        section_numbers: List[str],
        act_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple legal sections.
        
        Args:
            section_numbers: List of section numbers to compare
            act_names: Optional list of act names
            
        Returns:
            Comparative analysis
        """
        sections_str = ", ".join(section_numbers)
        query = f"Compare and analyze the following legal sections: {sections_str}"
        
        if act_names:
            acts_str = ", ".join(act_names)
            query += f" from the following acts: {acts_str}"
        
        return self.chat(query, k=len(section_numbers) * 3)
    
    def trace_legal_history(
        self,
        topic: str,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Trace the legal history and evolution of a topic.
        
        Args:
            topic: Legal topic to trace
            from_year: Starting year
            to_year: Ending year
            
        Returns:
            Historical legal analysis
        """
        query = f"Trace the legal history and evolution of {topic}"
        
        if from_year and to_year:
            query += f" from {from_year} to {to_year}"
        elif from_year:
            query += f" from {from_year} onwards"
        elif to_year:
            query += f" up to {to_year}"
        
        return self.chat(query, k=15)
    
    def draft_legal_argument(
        self,
        position: str,
        supporting_facts: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Draft a legal argument with supporting authorities.
        
        Args:
            position: The legal position to argue
            supporting_facts: Optional supporting facts
            
        Returns:
            Legal argument with citations
        """
        query = f"Draft a legal argument for the following position: {position}"
        
        if supporting_facts:
            query += f"\n\nBased on these facts: {supporting_facts}"
        
        query += "\n\nProvide relevant statutes, case law, and legal reasoning."
        
        return self.chat(query, k=10)


def create_research_agent(session_id: Optional[str] = None) -> ResearchAgent:
    """
    Create a research agent for lawyers.
    
    Args:
        session_id: Optional session identifier
        
    Returns:
        Configured ResearchAgent
    """
    return ResearchAgent(session_id=session_id)


# Quick test
if __name__ == "__main__":
    print("Testing Research Agent")
    print("=" * 80)
    
    agent = create_research_agent()
    
    # Test comprehensive research
    print("\n1. Statute Research")
    print("-" * 80)
    response = agent.research_statute(
        topic="property rights and ownership",
        year_range=(1850, 1950)
    )
    print(f"Retrieved: {response['num_sources']} sources")
    print(f"Research depth: {response.get('research_depth', 'N/A')}")
    
    # Test precedent finding
    print("\n2. Precedent Research")
    print("-" * 80)
    response = agent.find_precedents(
        legal_issue="property disputes",
        court_level="High Court"
    )
    print(f"Retrieved: {response['num_sources']} sources")
    
    # Test legal analysis
    print("\n3. Legal Analysis")
    print("-" * 80)
    response = agent.analyze_legal_question(
        question="Can a society expel a member without following proper procedure?",
        include_both_sides=True
    )
    print(f"Answer: {response['answer'][:200]}...")
    print(f"Sources: {len(response['sources'])}")
    
    # Test section comparison
    print("\n4. Section Comparison")
    print("-" * 80)
    response = agent.compare_sections(
        section_numbers=["9", "11"],
        act_names=["The Societies Registration Act, 1860"]
    )
    print(f"Answer: {response['answer'][:200]}...")
    
    print("\n" + "=" * 80)
