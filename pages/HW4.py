import streamlit as st
import os
import sys
from bs4 import BeautifulSoup
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic

# ==============================================================================
# 1. INITIALIZATION AND ENVIRONMENT FIXES
# ==============================================================================

# This workaround forces Python to use a newer, compatible version of SQLite3
# required by ChromaDB. It MUST be placed before the chromadb import.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    st.warning("pysqlite3 not found. ChromaDB might face issues if the system's sqlite3 is outdated.")

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ==============================================================================
# 2. CONSTANTS AND CONFIGURATION
# ==============================================================================

CHROMA_DB_PATH = "./ChromaDB_RAG"
SOURCE_DIR = "su_orgs"
CHROMA_COLLECTION_NAME = "MultiDocCollection"

# NEW: Nested dictionary to hold model options for each provider
MODEL_OPTIONS = {
    "OpenAI": ["gpt-5-nano", "gpt-5-chat-latest", "gpt-5-mini"],
    "Google": ["gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-2.5-flash"],
    "Anthropic": [ "claude-sonnet-4-20250514", "claude-3-haiku-20240307"]
}

# ==============================================================================
# 3. CACHED RESOURCES (EFFICIENT INITIALIZATION)
# ==============================================================================

@st.cache_resource
def get_api_clients():
    """Initializes and returns all API clients in a dictionary."""
    try:
        # UPDATED: For Google, we just need the configured module to be flexible
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        clients = {
            "OpenAI": OpenAI(api_key=st.secrets["OPENAI_API_KEY"]),
            "Google": genai,
            "Anthropic": Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        }
        return clients
    except Exception as e:
        st.error(f"Failed to initialize API clients. Please check your API keys in Streamlit secrets. Error: {e}")
        st.stop()

@st.cache_resource
def get_chroma_collection():
    """Initializes a persistent ChromaDB client and returns the collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# ==============================================================================
# 4. CORE LOGIC FUNCTIONS
# ==============================================================================

def extract_text_from_html(file_path):
    """Extracts clean text content from an HTML file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
            return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        st.error(f"Error reading {os.path.basename(file_path)}: {e}")
        return None

def setup_vector_db(collection, openai_client, force_rebuild=False):
    """Builds the vector database from HTML files using efficient batch embedding."""
    if collection.count() > 0 and not force_rebuild:
        st.sidebar.info(f"Vector DB contains {collection.count()} chunks.")
        return

    st.sidebar.warning("Building Vector DB. Please wait...")
    with st.spinner("Processing files and creating embeddings..."):
        source_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".html")]
        if not source_files:
            st.sidebar.error(f"No HTML files found in '{SOURCE_DIR}'.")
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        all_chunks, all_ids = [], []

        for filename in source_files:
            doc_text = extract_text_from_html(os.path.join(SOURCE_DIR, filename))
            if doc_text:
                chunks = text_splitter.split_text(doc_text)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_ids.append(f"{filename}_chunk_{i+1}")

        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]
            try:
                response = openai_client.embeddings.create(input=batch_chunks, model="text-embedding-3-small")
                embeddings = [item.embedding for item in response.data]
                collection.add(documents=batch_chunks, ids=batch_ids, embeddings=embeddings)
            except Exception as e:
                st.error(f"Failed to embed batch starting at index {i}: {e}")

    st.sidebar.success(f"Vector DB built with {collection.count()} chunks.", icon="✅")

def query_vector_db(collection, openai_client, prompt, n_results=4):
    """Queries the database to find contextually relevant document chunks."""
    try:
        query_embedding = openai_client.embeddings.create(input=[prompt], model="text-embedding-3-small").data[0].embedding
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return "\n---\n".join(results['documents'][0]) if results.get('documents') else "No relevant context found."
    except Exception as e:
        st.error(f"Vector DB query failed: {e}")
        return "Error retrieving context."

def get_llm_response(clients, llm_provider, model_name, prompt, context, chat_history):
    """Generates a response from the selected Language Model."""
    system_prompt = "You are an expert assistant. Answer the user's question based on the provided context and conversation history. If the answer is not found, state that clearly."
    user_prompt = f"CONTEXT:\n{context}\n\nHISTORY:\n{chat_history}\n\nQUESTION:\n{prompt}"
    
    try:
        with st.spinner(f"Asking {model_name}..."):
            client = clients[llm_provider]
            if llm_provider == "OpenAI":
                response = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], max_completion_tokens=2048)
                return response.choices[0].message.content
            
            elif llm_provider == "Google":
                # UPDATED: Initialize the specific model chosen by the user
                model_obj = client.GenerativeModel(model_name)
                response = model_obj.generate_content(f"{system_prompt}\n\n{user_prompt}")
                return response.text
            
            elif llm_provider == "Anthropic":
                response = client.messages.create(model=model_name, system=system_prompt, messages=[{"role": "user", "content": user_prompt}], max_tokens=2048)
                return response.content[0].text
    except Exception as e:
        st.error(f"Error with {llm_provider}: {e}")
        return "Sorry, an error occurred with the AI model."

# ==============================================================================
# 5. STREAMLIT UI AND MAIN APPLICATION FLOW
# ==============================================================================

def main():
    st.set_page_config(page_title="Multi-LLM RAG Chat", page_icon="🧠")
    st.title("🧠 Multi-LLM RAG Chat Application")
    st.write(f"Ask questions about documents in the '{SOURCE_DIR}' folder.")

    clients = get_api_clients()
    collection = get_chroma_collection()

    # --- Sidebar UI ---
    st.sidebar.header("Settings")
    
    # UPDATED: Two-step dropdown selection for provider and model
    selected_llm_provider = st.sidebar.selectbox(
        "Choose an LLM Provider:",
        list(MODEL_OPTIONS.keys())
    )
    
    available_models = MODEL_OPTIONS[selected_llm_provider]
    
    selected_model = st.sidebar.selectbox(
        "Choose a Model:",
        available_models
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("Management")
    if st.sidebar.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

    if st.sidebar.button("Re-Build Vector DB", key="rebuild_db"):
        with st.spinner("Deleting existing collection and rebuilding..."):
            try:
                get_chroma_collection.clear()
                chromadb.PersistentClient(path=CHROMA_DB_PATH).delete_collection(name=CHROMA_COLLECTION_NAME)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error during rebuild: {e}")
    
    setup_vector_db(collection, clients["OpenAI"])

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        context = query_vector_db(collection, clients["OpenAI"], prompt)
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-11:-1]])
        
        response = get_llm_response(clients, selected_llm_provider, selected_model, prompt, context, chat_history)
        
        full_response = f"**Answer from `{selected_model}`:**\n\n{response}"
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        with st.chat_message("assistant"):
            st.markdown(full_response)

if __name__ == "__main__":
    main()