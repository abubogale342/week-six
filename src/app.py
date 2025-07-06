import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# App title
st.set_page_config(page_title="RAG Chatbot Demo")
st.title("RAG Chatbot Demo")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- RAG Setup ---
# 1. Load the vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. Initialize InferenceClient
client = InferenceClient(
    model="deepseek-ai/DeepSeek-V3-0324",  # Corrected model ID
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# --- Custom RAG Logic ---
def run_rag_query(prompt):
    # Retrieve relevant documents
    retriever = vector_store.as_retriever()
    docs = retriever.invoke(prompt)
    
    # Combine documents into context
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create prompt with context
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the following context to answer the question:\n" + context
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    # Call InferenceClient with streaming enabled
    stream = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=messages,
        max_tokens=1024,
        temperature=0.5,
        stream=True
    )
    
    return stream, docs

# --- Chat Input and Response ---
if prompt := st.chat_input("Ask a question about the document"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Get the stream and source documents
        stream, source_docs = run_rag_query(prompt)
        
        # Use st.write_stream to display the response as it comes in
        # and capture the full response for chat history.
        full_response = st.write_stream(
            (chunk.choices[0].delta.content for chunk in stream if chunk.choices[0].delta.content)
        )
        
        # Display source documents
        with st.expander("Sources"):
            for doc in source_docs:
                st.info(f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
                st.markdown(f"> {doc.page_content[:250]}...")
        
        # Add the full response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})