"""
Public User Agent for Law Buddy.
Provides simplified legal guidance for general public.
"""

from typing import Dict, Any, Optional

from src.agents.base_agent import BaseAgent


class PublicAgent(BaseAgent):
    """
    Agent for public users (non-lawyers).
    
    Features:
    - Simplified language
    - Practical advice
    - Fewer documents (focused results)
    - Always includes follow-up questions
    - Confidence warnings for low confidence
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize public agent.
        
        Args:
            session_id: Optional session identifier
        """
        super().__init__(
            user_type="public",
            use_llm_classifier=True,
            session_id=session_id
        )
        
        # Override default k for public users (fewer, more focused results)
        self.default_k = 3
    
    def chat(
        self,
        query: str,
        k: Optional[int] = None,
        include_followups: bool = True,
        include_confidence: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Process a user query with public-friendly settings.
        
        Args:
            query: User's question
            k: Number of documents to retrieve (defaults to 3)
            include_followups: Whether to generate follow-up questions (default True)
            include_confidence: Whether to assess confidence (default True)
            verbose: Whether to include debug information
            
        Returns:
            Response with answer, sources, and follow-ups
        """
        k = k or self.default_k
        
        # Always include follow-ups and confidence for public users
        response = super().chat(
            query=query,
            k=k,
            include_followups=include_followups,
            include_confidence=include_confidence,
            verbose=verbose
        )
        
        # Add disclaimer for low confidence
        if response.get('confidence', {}).get('level') == 'LOW':
            response['disclaimer'] = (
                "⚠️ Please note: The information found is limited. "
                "We recommend consulting with a qualified lawyer for accurate legal advice."
            )
        
        # Add general disclaimer
        if 'disclaimer' not in response:
            response['disclaimer'] = (
                "This is general legal information, not legal advice. "
                "For specific situations, please consult with a qualified lawyer."
            )
        
        return response
    
    def ask_about_rights(self, situation: str) -> Dict[str, Any]:
        """
        Specialized method for asking about legal rights.
        
        Args:
            situation: Description of the situation
            
        Returns:
            Response focused on rights and protections
        """
        query = f"What are my legal rights in this situation: {situation}"
        return self.chat(query)
    
    def ask_about_procedure(self, process: str) -> Dict[str, Any]:
        """
        Specialized method for asking about legal procedures.
        
        Args:
            process: The legal process to ask about
            
        Returns:
            Response with step-by-step guidance
        """
        query = f"What is the procedure for {process}? Please explain step by step."
        return self.chat(query)
    
    def understand_penalty(self, offense: str) -> Dict[str, Any]:
        """
        Specialized method for understanding penalties.
        
        Args:
            offense: The offense to ask about
            
        Returns:
            Response explaining penalties and consequences
        """
        query = f"What are the penalties or punishments for {offense}?"
        return self.chat(query)


def create_public_agent(session_id: Optional[str] = None) -> PublicAgent:
    """
    Create a public user agent.
    
    Args:
        session_id: Optional session identifier
        
    Returns:
        Configured PublicAgent
    """
    return PublicAgent(session_id=session_id)


# Quick test
if __name__ == "__main__":
    print("Testing Public Agent")
    print("=" * 80)
    
    agent = create_public_agent()
    
    # Test general question
    print("\n1. General Question")
    print("-" * 80)
    response = agent.chat(
        "Can my landlord evict me without notice?",
        verbose=False
    )
    print(f"Answer: {response['answer'][:200]}...")
    print(f"Sources: {len(response['sources'])}")
    print(f"Disclaimer: {response.get('disclaimer', 'None')}")
    
    # Test rights question
    print("\n\n2. Rights Question")
    print("-" * 80)
    response = agent.ask_about_rights(
        "My employer fired me without any reason after 5 years of service"
    )
    print(f"Answer: {response['answer'][:200]}...")
    
    # Test procedure question
    print("\n\n3. Procedure Question")
    print("-" * 80)
    response = agent.ask_about_procedure("filing a property dispute case")
    print(f"Answer: {response['answer'][:200]}...")
    
    # Test penalty question
    print("\n\n4. Penalty Question")
    print("-" * 80)
    response = agent.understand_penalty("theft from a registered society")
    print(f"Answer: {response['answer'][:200]}...")
    print(f"Confidence: {response.get('confidence', {}).get('level', 'N/A')}")
    
    print("\n" + "=" * 80)
