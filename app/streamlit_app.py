"""
Streamlit UI for BanglaLegalAI
"""

import streamlit as st
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.public_agent import create_public_agent
from src.agents.research_agent import create_research_agent


# Page configuration
st.set_page_config(
    page_title="BanglaLegalAI - Bangladesh Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'user_type' not in st.session_state:
        st.session_state.user_type = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")


def display_header():
    """Display the app header."""
    st.markdown('<div class="main-header">⚖️ BanglaLegalAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Your AI-Powered Legal Assistant for Bangladesh Law</div>',
        unsafe_allow_html=True
    )


def display_sidebar():
    """Display sidebar with user type selection and settings."""
    with st.sidebar:
        st.title("Settings")
        
        # User type selection
        st.subheader("👤 Who are you?")
        user_type_display = st.radio(
            "Select your user type:",
            options=["General Public", "Legal Professional"],
            help="This determines how answers are presented to you"
        )
        
        user_type = "public" if user_type_display == "General Public" else "lawyer"
        
        # Initialize or update agent if user type changed
        if st.session_state.user_type != user_type:
            st.session_state.user_type = user_type
            if user_type == "public":
                st.session_state.agent = create_public_agent(
                    session_id=st.session_state.session_id
                )
            else:
                st.session_state.agent = create_research_agent(
                    session_id=st.session_state.session_id
                )
            st.session_state.chat_history = []
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        
        if user_type == "public":
            num_results = st.slider(
                "Number of sources",
                min_value=1,
                max_value=5,
                value=3,
                help="More sources = more comprehensive but longer answers"
            )
        else:
            num_results = st.slider(
                "Number of sources",
                min_value=5,
                max_value=20,
                value=10,
                help="Number of legal sources to retrieve"
            )
        
        include_followups = st.checkbox(
            "Generate follow-up questions",
            value=True,
            help="AI will suggest related questions"
        )
        
        show_confidence = st.checkbox(
            "Show confidence assessment",
            value=(user_type == "public"),
            help="Display AI's confidence in the answer"
        )
        
        st.divider()
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.agent:
                st.session_state.agent.clear_history()
            st.rerun()
        
        # Session info
        st.divider()
        st.caption(f"Session ID: {st.session_state.session_id}")
        st.caption(f"User Type: {user_type_display}")
        
        return num_results, include_followups, show_confidence


def display_chat_message(message: dict):
    """Display a chat message."""
    role = message["role"]
    
    if role == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    
    else:  # assistant
        with st.chat_message("assistant"):
            # Main answer
            st.markdown(message["content"])
            
            # Sources
            if message.get("sources"):
                with st.expander(f"📚 Sources ({len(message['sources'])})"):
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{idx}.** {source['citation']}")
            
            # Confidence
            if message.get("confidence"):
                conf = message["confidence"]
                level = conf["level"]
                
                if level == "HIGH":
                    conf_class = "confidence-high"
                    icon = "✅"
                elif level == "MEDIUM":
                    conf_class = "confidence-medium"
                    icon = "⚠️"
                else:
                    conf_class = "confidence-low"
                    icon = "❗"
                
                st.markdown(
                    f'{icon} <span class="{conf_class}">Confidence: {level}</span>',
                    unsafe_allow_html=True
                )
                
                if conf.get("reasoning"):
                    with st.expander("Why this confidence level?"):
                        for reason in conf["reasoning"]:
                            st.write(f"• {reason}")
            
            # Disclaimer
            if message.get("disclaimer"):
                st.markdown(
                    f'<div class="disclaimer-box">ℹ️ {message["disclaimer"]}</div>',
                    unsafe_allow_html=True
                )
            
            # Follow-up questions
            if message.get("followup_questions"):
                st.markdown("**💡 Related Questions:**")
                for q in message["followup_questions"]:
                    if st.button(q, key=f"followup_{message['timestamp']}_{q[:20]}"):
                        st.session_state.followup_query = q
                        st.rerun()


def process_query(query: str, num_results: int, include_followups: bool, show_confidence: bool):
    """Process a user query and display the response."""
    if not st.session_state.agent:
        st.error("Please select a user type in the sidebar first.")
        return
    
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": query,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(query)
    
    # Get response from agent with streaming
    with st.chat_message("assistant"):
        # Create placeholder for streaming text
        answer_placeholder = st.empty()
        full_answer = ""
        response = None
        
        try:
            with st.spinner("Searching legal database..."):
                # Stream the response
                for chunk in st.session_state.agent.chat_stream(
                    query=query,
                    k=num_results,
                    include_followups=include_followups,
                    include_confidence=show_confidence,
                    verbose=False
                ):
                    if chunk["type"] == "answer_chunk":
                        # Accumulate answer text
                        full_answer += chunk["content"]
                        # Display with cursor while streaming
                        answer_placeholder.markdown(full_answer + "▌")
                    elif chunk["type"] == "metadata":
                        # Got all metadata (sources, confidence, etc.)
                        response = chunk
            
            # Remove cursor and show final answer
            answer_placeholder.markdown(full_answer)
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            # Fallback to non-streaming if streaming fails
            with st.spinner("Retrying..."):
                response = st.session_state.agent.chat(
                    query=query,
                    k=num_results,
                    include_followups=include_followups,
                    include_confidence=show_confidence,
                    verbose=False
                )
                full_answer = response["answer"]
                st.markdown(full_answer)
        
        # Display sources
        if response.get("sources"):
            with st.expander(f"📚 Sources ({len(response['sources'])})"):
                for idx, source in enumerate(response["sources"], 1):
                    st.markdown(f"**{idx}.** {source['citation']}")
        
        # Display confidence
        if response.get("confidence"):
            conf = response["confidence"]
            level = conf["level"]
            
            if level == "HIGH":
                conf_class = "confidence-high"
                icon = "✅"
            elif level == "MEDIUM":
                conf_class = "confidence-medium"
                icon = "⚠️"
            else:
                conf_class = "confidence-low"
                icon = "❗"
            
            st.markdown(
                f'{icon} <span class="{conf_class}">Confidence: {level}</span>',
                unsafe_allow_html=True
            )
            
            if conf.get("reasoning"):
                with st.expander("Why this confidence level?"):
                    for reason in conf["reasoning"]:
                        st.write(f"• {reason}")
        
        # Display disclaimer
        if response.get("disclaimer"):
            st.markdown(
                f'<div class="disclaimer-box">ℹ️ {response["disclaimer"]}</div>',
                unsafe_allow_html=True
            )
        
        # Display follow-up questions
        if response.get("followup_questions"):
            st.markdown("**💡 Related Questions:**")
            cols = st.columns(1)
            for q in response["followup_questions"]:
                if cols[0].button(q, key=f"followup_{datetime.now().timestamp()}_{q[:20]}", use_container_width=True):
                    st.session_state.followup_query = q
                    st.rerun()
    
    # Add assistant message to history
    if response:  # Only add if we got a response
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_answer,  # Use streamed answer
            "sources": response.get("sources", []),
            "confidence": response.get("confidence"),
            "disclaimer": response.get("disclaimer"),
            "followup_questions": response.get("followup_questions", []),
            "timestamp": datetime.now().isoformat()
        })


def main():
    """Main app function."""
    initialize_session_state()
    display_header()
    
    # Sidebar
    num_results, include_followups, show_confidence = display_sidebar()
    
    # Check if agent is initialized
    if not st.session_state.agent:
        st.info("👈 Please select your user type in the sidebar to get started.")
        return
    
    # Display example questions
    if not st.session_state.chat_history:
        st.markdown("### 💬 Ask me anything about Bangladesh law!")
        
        if st.session_state.user_type == "public":
            st.markdown("**Example questions:**")
            examples = [
                "What are my rights as a tenant?",
                "Can my employer fire me without notice?",
                "What is the penalty for theft?",
                "How do I file a property dispute case?"
            ]
        else:
            st.markdown("**Example research queries:**")
            examples = [
                "Find precedents on property ownership disputes",
                "Analyze Section 11 of The Societies Registration Act",
                "Compare penalties for fraud across different acts",
                "Trace the legal history of tenant rights from 1850-1950"
            ]
        
        cols = st.columns(2)
        for idx, example in enumerate(examples):
            if cols[idx % 2].button(example, key=f"example_{idx}", use_container_width=True):
                st.session_state.followup_query = example
                st.rerun()
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message)
    
    # Handle follow-up query from button click
    if hasattr(st.session_state, 'followup_query'):
        query = st.session_state.followup_query
        delattr(st.session_state, 'followup_query')
        process_query(query, num_results, include_followups, show_confidence)
        st.rerun()
    
    # Chat input
    query = st.chat_input("Type your legal question here...")
    if query:
        process_query(query, num_results, include_followups, show_confidence)
        st.rerun()


if __name__ == "__main__":
    main()
