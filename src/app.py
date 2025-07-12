import os
import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from pathlib import Path

# App title and config
st.set_page_config(
    page_title="Complaint Analysis Assistant",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #f0f2f6;
        margin-left: 20%;
        border-bottom-right-radius: 0;
    }
    .assistant-message {
        background-color: #e3f2fd;
        margin-right: 20%;
        border-bottom-left-radius: 0;
    }
    .source-doc {
        padding: 0.75rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
        border-left: 3px solid #4285f4;
        font-size: 0.9em;
    }
    .source-header {
        font-weight: bold;
        color: #5f6368;
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

class RAGSystem:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_store_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                                   "data", "faiss_consumer_vector_store")):
        self.model = SentenceTransformer(model_name)
        self.vector_store_path = vector_store_path
        self.index = None
        self.metadata = None
        self.initialize_components()

    def initialize_components(self):
        """Load FAISS index and metadata."""
        try:
            # Load FAISS index
            index_path = os.path.join(self.vector_store_path, "faiss.index")
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
                
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            self.index = None
            self.metadata = []

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        if self.index is None or not self.metadata:
            return []
            
        try:
            # Encode query and search
            query_embedding = self.model.encode([query])
            query_embedding = np.array(query_embedding).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search the index
            distances, indices = self.index.search(query_embedding, k)
            
            # Prepare results
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['similarity_score'] = float(1.0 - distances[0][i])
                    results.append(result)
            
            return results
            
        except Exception as e:
            st.error(f"Error during retrieval: {str(e)}")
            return []

# Initialize RAG system
@st.cache_resource
def load_rag_system():
    with st.spinner("Loading AI model and documents..."):
        return RAGSystem()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.rag = load_rag_system()

# Sidebar with app info
with st.sidebar:
    st.title("📄 Complaint Analysis Assistant")
    st.markdown("""
    This assistant helps you analyze consumer complaints using AI-powered search.
    
    ### How to use:
    1. Type your question in the chat box below
    2. The system will find relevant complaint records
    3. Review the answer and sources
    
    ### Example questions:
    - What are common credit card issues?
    - How do I report unauthorized transactions?
    - What's the process for disputing charges?
    """)
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Display the response
            st.markdown(message["content"])
            
            # Display sources if available
            if "sources" in message:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"][:3], 1):
                        st.markdown(f"""
                        <div class="source-doc">
                            <div class="source-header">
                                Source {i} | Product: {source.get('product', 'Unknown')} | 
                                Score: {source.get('similarity_score', 0):.2f}
                            </div>
                            {source.get('chunk_text', 'No text available')[:300]}...
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about consumer complaints..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching for relevant information..."):
            # Retrieve relevant documents
            results = st.session_state.rag.retrieve(prompt, k=3)
            
            if not results:
                response = "I couldn't find any relevant information in our records."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": []
                })
                st.markdown(response)
            else:
                # Format response with sources
                response = "Here's what I found in our records:\n\n"
                for i, doc in enumerate(results, 1):
                    text = doc.get('chunk_text', '')
                    # Find the first period after 100 characters
                    first_period = text.find('.', 100)
                    if first_period > 0:
                        text = text[:first_period + 1]  # Include the period
                    response += f"{i}. {text}\n\n"
                response += "\nWould you like me to help you with anything specific from this information?"
                
                # Display response
                st.markdown(response)
                
                # Add to chat history with sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": results
                })
    
    # Rerun to update the display
    st.rerun()