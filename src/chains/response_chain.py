"""
Response Generation Chain for Law Buddy.
Generates answers from retrieved documents with proper citations.
"""

from typing import Dict, Any, List, Optional
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.config import get_settings
from src.prompts.system_prompts import (
    get_system_prompt,
    RESPONSE_WITH_SOURCES_PROMPT,
    FOLLOWUP_QUESTIONS_PROMPT,
    SUMMARIZATION_PROMPT
)


class ResponseChain:
    """
    Generates responses from retrieved documents.
    
    Handles:
    - Answer generation with citations
    - Follow-up question generation
    - Document summarization
    - Confidence assessment
    """
    
    def __init__(self, user_type: str = "public"):
        """
        Initialize response chain.
        
        Args:
            user_type: "public" or "lawyer"
        """
        settings = get_settings()
        self.user_type = user_type
        
        # Use large model for response generation (better quality)
        self.llm = ChatMistralAI(
            model=settings.mistral_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            api_key=settings.mistral_api_key
        )
        
        # Small model for quick tasks (summarization, follow-ups)
        self.llm_small = ChatMistralAI(
            model=settings.mistral_model_small,
            temperature=settings.temperature,
            api_key=settings.mistral_api_key
        )
        
        # Get system prompt for user type
        self.system_prompt = get_system_prompt(user_type)
        
        # Output parser
        self.parser = StrOutputParser()
    
    def generate_response(
        self,
        query: str,
        documents: List[Document],
        include_followups: bool = True,
        include_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a response from retrieved documents.
        
        Args:
            query: User's question
            documents: Retrieved documents
            include_followups: Whether to generate follow-up questions
            include_confidence: Whether to assess confidence
            
        Returns:
            Dictionary with response and metadata
        """
        # Build context from documents
        context = self._format_documents(documents)
        
        # Generate answer
        answer = self._generate_answer(query, context)
        
        # Build response
        response = {
            "query": query,
            "answer": answer,
            "sources": self._format_sources(documents),
            "num_sources": len(documents)
        }
        
        # Generate follow-up questions
        if include_followups:
            followups = self._generate_followups(query, answer)
            response["followup_questions"] = followups
        
        # Assess confidence
        if include_confidence:
            confidence = self._assess_confidence(query, answer, documents)
            response["confidence"] = confidence
        
        return response
    
    def _generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using LLM.
        
        Args:
            query: User query
            context: Formatted document context
            
        Returns:
            Generated answer
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", RESPONSE_WITH_SOURCES_PROMPT)
        ])
        
        chain = prompt | self.llm | self.parser
        
        answer = chain.invoke({
            "query": query,
            "documents": context
        })
        
        return answer.strip()
    
    def stream_answer(self, query: str, context: str):
        """
        Stream answer generation using LLM.
        
        Args:
            query: User query
            context: Formatted document context
            
        Yields:
            Chunks of the generated answer
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", RESPONSE_WITH_SOURCES_PROMPT)
        ])
        
        chain = prompt | self.llm | self.parser
        
        for chunk in chain.stream({
            "query": query,
            "documents": context
        }):
            yield chunk
    
    def _format_documents(self, documents: List[Document]) -> str:
        """
        Format documents into context string.
        
        Args:
            documents: List of documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        
        for idx, doc in enumerate(documents, 1):
            metadata = doc.metadata
            source_type = metadata.get("source_type", "unknown")
            
            # Format header based on type
            if source_type == "act":
                header = f"[Source {idx}] {metadata.get('act_title', 'Unknown Act')}"
                header += f"\nSection {metadata.get('section_number', 'N/A')}"
                if metadata.get('section_title'):
                    header += f": {metadata['section_title']}"
                header += f"\nYear: {metadata.get('act_year', 'N/A')}"
                
            elif source_type == "case_study":
                header = f"[Source {idx}] {metadata.get('case_title', 'Unknown Case')}"
                header += f"\nCase ID: {metadata.get('case_id', 'N/A')}"
                header += f"\nCourt: {metadata.get('court_level', 'N/A')}"
                header += f"\nVerdict: {metadata.get('verdict', 'N/A')}"
                header += f"\nArea of Law: {metadata.get('area_of_law', 'N/A')}"
            else:
                header = f"[Source {idx}]"
            
            # Add content
            content = f"\n\n{doc.page_content}\n"
            
            context_parts.append(header + content)
            context_parts.append("=" * 80)
        
        return "\n".join(context_parts)
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, str]]:
        """
        Format sources for citation.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            List of formatted sources
        """
        sources = []
        
        for doc in documents:
            metadata = doc.metadata
            source_type = metadata.get("source_type", "unknown")
            
            if source_type == "act":
                citation = f"{metadata.get('act_title', 'Unknown Act')}, "
                citation += f"Section {metadata.get('section_number', 'N/A')}"
                if metadata.get('act_year'):
                    citation += f" ({metadata['act_year']})"
                
                sources.append({
                    "type": "act",
                    "citation": citation,
                    "title": metadata.get('act_title', 'Unknown'),
                    "section": metadata.get('section_number', 'N/A'),
                    "year": str(metadata.get('act_year', 'N/A'))
                })
                
            elif source_type == "case_study":
                citation = f"{metadata.get('case_title', 'Unknown Case')}, "
                citation += f"{metadata.get('case_id', 'N/A')}"
                if metadata.get('court_level'):
                    citation += f", {metadata['court_level']}"
                
                sources.append({
                    "type": "case",
                    "citation": citation,
                    "title": metadata.get('case_title', 'Unknown'),
                    "case_id": metadata.get('case_id', 'N/A'),
                    "court": metadata.get('court_level', 'N/A')
                })
            else:
                sources.append({
                    "type": "unknown",
                    "citation": "Unknown source"
                })
        
        return sources
    
    def _generate_followups(self, query: str, answer: str) -> List[str]:
        """
        Generate follow-up questions.
        
        Args:
            query: Original query
            answer: Generated answer
            
        Returns:
            List of 3 follow-up questions
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that generates relevant follow-up questions."),
                ("user", FOLLOWUP_QUESTIONS_PROMPT)
            ])
            
            chain = prompt | self.llm_small | self.parser
            
            result = chain.invoke({
                "query": query,
                "answer": answer
            })
            
            # Parse questions (assuming they're numbered)
            lines = [line.strip() for line in result.strip().split('\n') if line.strip()]
            questions = []
            for line in lines:
                # Remove numbering (1., 2., etc.)
                cleaned = line.lstrip('0123456789.-) ')
                if cleaned and '?' in cleaned:
                    questions.append(cleaned)
            
            return questions[:3]  # Return max 3 questions
            
        except Exception as e:
            print(f"Error generating follow-ups: {e}")
            return []
    
    def _assess_confidence(
        self,
        query: str,
        answer: str,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Assess confidence in the answer.
        
        Args:
            query: Original query
            answer: Generated answer
            documents: Retrieved documents
            
        Returns:
            Confidence assessment
        """
        # Simple heuristic-based confidence
        confidence_score = "MEDIUM"
        reasoning = []
        
        # Factor 1: Number of documents
        if len(documents) >= 3:
            reasoning.append("Multiple relevant sources found")
            confidence_score = "HIGH"
        elif len(documents) == 0:
            reasoning.append("No sources found")
            confidence_score = "LOW"
        else:
            reasoning.append("Limited sources available")
        
        # Factor 2: Answer length (very short answers might indicate uncertainty)
        if len(answer) < 100:
            reasoning.append("Brief answer may indicate limited information")
            if confidence_score == "HIGH":
                confidence_score = "MEDIUM"
        
        # Factor 3: Check for hedging language
        hedging_words = ["may", "might", "possibly", "unclear", "uncertain", "not sure"]
        if any(word in answer.lower() for word in hedging_words):
            reasoning.append("Answer contains uncertainty language")
            if confidence_score == "HIGH":
                confidence_score = "MEDIUM"
        
        return {
            "level": confidence_score,
            "reasoning": reasoning,
            "num_sources": len(documents)
        }
    
    def summarize_document(self, document: Document) -> str:
        """
        Summarize a legal document.
        
        Args:
            document: Document to summarize
            
        Returns:
            Summary text
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal document summarizer."),
            ("user", SUMMARIZATION_PROMPT)
        ])
        
        chain = prompt | self.llm_small | self.parser
        
        summary = chain.invoke({"document": document.page_content})
        
        return summary.strip()
    
    def update_user_type(self, user_type: str):
        """
        Update the user type and system prompt.
        
        Args:
            user_type: "public" or "lawyer"
        """
        self.user_type = user_type
        self.system_prompt = get_system_prompt(user_type)


def get_response_chain(user_type: str = "public") -> ResponseChain:
    """
    Get a configured response chain.
    
    Args:
        user_type: "public" or "lawyer"
        
    Returns:
        ResponseChain instance
    """
    return ResponseChain(user_type=user_type)


# Quick test
if __name__ == "__main__":
    from langchain_core.documents import Document
    
    print("Testing Response Generation Chain")
    print("=" * 80)
    
    # Mock documents
    mock_docs = [
        Document(
            page_content="Any member of the society who shall steal, purloin or embezzle any money or other property, or wilfully and maliciously destroy or injure any property of such society, or shall forge any deed, bond, security for money, receipt or other instrument, whereby the funds of the society may be exposed to loss, shall be punishable as if he had been guilty of such act in respect of the property of an individual.",
            metadata={
                "source_type": "act",
                "act_title": "The Societies Registration Act, 1860",
                "section_number": "11",
                "section_title": "Penalties for certain offences",
                "act_year": 1860
            }
        ),
        Document(
            page_content="Whenever by any bye-law duly made in accordance with the rules and regulations of the society, or, if the rules do not provide for the making of bye-laws, by any bye-law made at a general meeting of the society, a fine or penalty is imposed for any act or omission, such fine or penalty may be sued for and recovered, and shall be applied in such manner as the rules of the society prescribe.",
            metadata={
                "source_type": "act",
                "act_title": "The Societies Registration Act, 1860",
                "section_number": "9",
                "section_title": "Recovery of fines",
                "act_year": 1860
            }
        )
    ]
    
    # Test for public user
    print("\n--- PUBLIC USER RESPONSE ---")
    chain_public = get_response_chain(user_type="public")
    response_public = chain_public.generate_response(
        query="What are the penalties for theft in a society?",
        documents=mock_docs,
        include_followups=True,
        include_confidence=True
    )
    
    print(f"\nQuery: {response_public['query']}")
    print(f"\nAnswer:\n{response_public['answer']}")
    print(f"\nSources ({response_public['num_sources']}):")
    for idx, source in enumerate(response_public['sources'], 1):
        print(f"  {idx}. {source['citation']}")
    
    if response_public.get('followup_questions'):
        print(f"\nFollow-up Questions:")
        for idx, q in enumerate(response_public['followup_questions'], 1):
            print(f"  {idx}. {q}")
    
    if response_public.get('confidence'):
        conf = response_public['confidence']
        print(f"\nConfidence: {conf['level']}")
        print(f"Reasoning: {', '.join(conf['reasoning'])}")
    
    print("\n" + "=" * 80)
    
    # Test for lawyer
    print("\n--- LAWYER RESPONSE ---")
    chain_lawyer = get_response_chain(user_type="lawyer")
    response_lawyer = chain_lawyer.generate_response(
        query="What are the penalties for theft in a society?",
        documents=mock_docs,
        include_followups=False,
        include_confidence=False
    )
    
    print(f"\nQuery: {response_lawyer['query']}")
    print(f"\nAnswer:\n{response_lawyer['answer']}")
    print(f"\nSources ({response_lawyer['num_sources']}):")
    for idx, source in enumerate(response_lawyer['sources'], 1):
        print(f"  {idx}. {source['citation']}")
    
    print("\n" + "=" * 80)
