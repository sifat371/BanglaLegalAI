"""
Base Agent for BanglaLegalAI.
Provides core RAG functionality with conversation history.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from src.chains.retrieval_chain import get_retrieval_chain
from src.chains.response_chain import get_response_chain
from src.config import get_settings


class ConversationHistory:
    """Manages conversation history for context."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation history.
        
        Args:
            max_history: Maximum number of turns to keep
        """
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
    
    def add_turn(
        self,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a conversation turn.
        
        Args:
            query: User query
            answer: System answer
            sources: List of sources used
            metadata: Additional metadata
        """
        turn = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "sources": sources,
            "metadata": metadata or {}
        }
        
        self.history.append(turn)
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_recent_context(self, n: int = 3) -> str:
        """
        Get recent conversation context.
        
        Args:
            n: Number of recent turns to include
            
        Returns:
            Formatted context string
        """
        if not self.history:
            return ""
        
        recent = self.history[-n:]
        context_parts = []
        
        for turn in recent:
            context_parts.append(f"User: {turn['query']}")
            context_parts.append(f"Assistant: {turn['answer'][:200]}...")  # Truncate
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """Export history as dictionary."""
        return self.history.copy()


class BaseAgent:
    """
    Base legal assistant agent.
    
    Provides core RAG functionality:
    - Query processing
    - Document retrieval
    - Answer generation
    - Conversation history
    """
    
    def __init__(
        self,
        user_type: str = "public",
        use_llm_classifier: bool = True,
        session_id: Optional[str] = None
    ):
        """
        Initialize base agent.
        
        Args:
            user_type: "public" or "lawyer"
            use_llm_classifier: Whether to use LLM for query classification
            session_id: Optional session identifier
        """
        self.user_type = user_type
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize chains
        self.retrieval_chain = get_retrieval_chain(use_llm_classifier=use_llm_classifier)
        self.response_chain = get_response_chain(user_type=user_type)
        
        # Initialize conversation history
        self.conversation = ConversationHistory()
        
        # Settings
        settings = get_settings()
        self.default_k = settings.top_k
    
    def chat(
        self,
        query: str,
        k: Optional[int] = None,
        include_followups: bool = True,
        include_confidence: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: User's question
            k: Number of documents to retrieve (None = use default)
            include_followups: Whether to generate follow-up questions
            include_confidence: Whether to assess confidence
            verbose: Whether to include debug information
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        k = k or self.default_k
        
        # Step 1: Retrieve relevant documents
        if verbose:
            print(f"[Agent] Retrieving documents for: {query}")
        
        retrieval_result = self.retrieval_chain.retrieve(
            query=query,
            k=k,
            user_type=self.user_type,
            include_classification=verbose
        )
        
        if verbose:
            print(f"[Agent] Retrieved {retrieval_result['num_results']} documents")
            if retrieval_result.get('classification'):
                print(f"[Agent] Intent: {retrieval_result['classification']['intent']}")
        
        # Step 2: Generate response
        if verbose:
            print(f"[Agent] Generating response...")
        
        response_result = self.response_chain.generate_response(
            query=query,
            documents=retrieval_result["documents"],
            include_followups=include_followups,
            include_confidence=include_confidence
        )
        
        # Step 3: Add to conversation history
        self.conversation.add_turn(
            query=query,
            answer=response_result["answer"],
            sources=response_result["sources"],
            metadata={
                "num_sources": response_result["num_sources"],
                "confidence": response_result.get("confidence"),
                "citation_verification": response_result.get("citation_verification"),
                "claim_support_verification": response_result.get("claim_support_verification"),
                "grounding_enforcement": response_result.get("grounding_enforcement"),
                "classification": retrieval_result.get("classification")
            }
        )
        
        # Step 4: Build response
        response = {
            "query": query,
            "answer": response_result["answer"],
            "sources": response_result["sources"],
            "num_sources": response_result["num_sources"],
            "citation_verification": response_result["citation_verification"],
            "claim_support_verification": response_result["claim_support_verification"],
            "grounding_enforcement": response_result["grounding_enforcement"],
            "session_id": self.session_id,
            "user_type": self.user_type
        }
        
        if include_followups and response_result.get("followup_questions"):
            response["followup_questions"] = response_result["followup_questions"]
        
        if include_confidence and response_result.get("confidence"):
            response["confidence"] = response_result["confidence"]
        
        if verbose and retrieval_result.get("classification"):
            response["debug"] = {
                "classification": retrieval_result["classification"],
                "retrieval_plan": retrieval_result.get("retrieval_plan"),
                "num_documents_retrieved": retrieval_result["num_results"]
            }
        
        return response
    
    @staticmethod
    def _display_chunks(text: str, chunk_size: int = 96):
        """Yield display chunks only after the answer has passed grounding policy."""
        for start in range(0, len(text), chunk_size):
            yield text[start : start + chunk_size]

    def chat_stream(
        self,
        query: str,
        k: Optional[int] = None,
        include_followups: bool = True,
        include_confidence: bool = True,
        verbose: bool = False,
    ):
        """Generate privately, verify/repair, then stream only the finalized answer."""
        k = k or self.default_k

        if verbose:
            print(f"[Agent] Retrieving documents for: {query}")

        retrieval_result = self.retrieval_chain.retrieve(
            query=query,
            k=k,
            user_type=self.user_type,
            include_classification=verbose,
        )

        if verbose:
            print(f"[Agent] Retrieved {retrieval_result['num_results']} documents")

        # Fail-closed streaming: no candidate text is yielded before verification.
        response_result = self.response_chain.generate_response(
            query=query,
            documents=retrieval_result["documents"],
            include_followups=False,
            include_confidence=include_confidence,
        )
        full_answer = response_result["answer"]

        for chunk in self._display_chunks(full_answer):
            yield {
                "type": "answer_chunk",
                "content": chunk,
            }

        followup_questions = []
        if (
            include_followups
            and response_result["grounding_enforcement"]["status"] != "blocked"
        ):
            followup_questions = self.response_chain._generate_followups(
                query,
                full_answer,
            )

        response_result["followup_questions"] = followup_questions

        self.conversation.add_turn(
            query=query,
            answer=full_answer,
            sources=response_result["sources"],
            metadata={
                "num_sources": response_result["num_sources"],
                "confidence": response_result.get("confidence"),
                "citation_verification": response_result.get(
                    "citation_verification"
                ),
                "claim_support_verification": response_result.get(
                    "claim_support_verification"
                ),
                "grounding_enforcement": response_result.get(
                    "grounding_enforcement"
                ),
                "classification": retrieval_result.get("classification"),
            },
        )

        yield {
            "type": "metadata",
            "query": query,
            "answer": full_answer,
            "sources": response_result["sources"],
            "num_sources": response_result["num_sources"],
            "citation_verification": response_result["citation_verification"],
            "claim_support_verification": response_result[
                "claim_support_verification"
            ],
            "grounding_enforcement": response_result["grounding_enforcement"],
            "followup_questions": followup_questions,
            "confidence": response_result.get("confidence"),
            "session_id": self.session_id,
            "user_type": self.user_type,
        }

    def get_section(self, section_number: str, act_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a specific legal section.
        
        Args:
            section_number: Section number (e.g., "420")
            act_name: Optional act name to filter
            
        Returns:
            Response with section details
        """
        retrieval_result = self.retrieval_chain.retrieve_by_section(
            section_number=section_number,
            act_name=act_name
        )
        
        if not retrieval_result["documents"]:
            return {
                "query": f"Section {section_number}",
                "answer": f"Section {section_number} not found in the database.",
                "sources": [],
                "num_sources": 0
            }
        
        response_result = self.response_chain.generate_response(
            query=f"Explain Section {section_number}" + (f" of {act_name}" if act_name else ""),
            documents=retrieval_result["documents"],
            include_followups=True,
            include_confidence=False
        )
        
        return response_result
    
    def get_case(self, case_id: str) -> Dict[str, Any]:
        """
        Get a specific case by ID.
        
        Args:
            case_id: Case ID (e.g., "BD-CR-001")
            
        Returns:
            Response with case details
        """
        retrieval_result = self.retrieval_chain.retrieve_by_case_id(case_id)
        
        if not retrieval_result["documents"]:
            return {
                "query": f"Case {case_id}",
                "answer": f"Case {case_id} not found in the database.",
                "sources": [],
                "num_sources": 0
            }
        
        response_result = self.response_chain.generate_response(
            query=f"Summarize case {case_id}",
            documents=retrieval_result["documents"],
            include_followups=True,
            include_confidence=False
        )
        
        return response_result
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.conversation.to_dict()
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation.clear()
    
    def update_user_type(self, user_type: str):
        """
        Update user type.
        
        Args:
            user_type: "public" or "lawyer"
        """
        self.user_type = user_type
        self.response_chain.update_user_type(user_type)


def create_agent(user_type: str = "public", session_id: Optional[str] = None) -> BaseAgent:
    """
    Factory function to create an agent.
    
    Args:
        user_type: "public" or "lawyer"
        session_id: Optional session identifier
        
    Returns:
        Configured BaseAgent
    """
    return BaseAgent(
        user_type=user_type,
        use_llm_classifier=True,  # Use LLM classification by default
        session_id=session_id
    )


# Quick test
if __name__ == "__main__":
    print("Testing Base Agent")
    print("=" * 80)
    
    # Create agent for public user
    agent = create_agent(user_type="public")
    
    # Test query
    query = "What are the penalties for theft in a registered society?"
    
    print(f"\nQuery: {query}")
    print("-" * 80)
    
    response = agent.chat(
        query=query,
        k=3,
        include_followups=True,
        include_confidence=True,
        verbose=True
    )
    
    print(f"\nAnswer:\n{response['answer']}")
    
    print(f"\n\nSources ({response['num_sources']}):")
    for idx, source in enumerate(response['sources'], 1):
        print(f"  {idx}. {source['citation']}")
    
    if response.get('followup_questions'):
        print(f"\nFollow-up Questions:")
        for idx, q in enumerate(response['followup_questions'], 1):
            print(f"  {idx}. {q}")
    
    if response.get('confidence'):
        conf = response['confidence']
        print(f"\nConfidence: {conf['level']}")
        print(f"Reasoning: {', '.join(conf['reasoning'])}")
    
    # Show conversation history
    history = agent.get_conversation_history()
    print(f"\n\nConversation History: {len(history)} turns")
    
    print("\n" + "=" * 80)
