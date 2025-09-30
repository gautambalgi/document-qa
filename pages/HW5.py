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

MODEL_OPTIONS = {
    "OpenAI": ["gpt-5-nano", "gpt-5-chat-latest", "gpt-5-mini"],
    "Google": ["gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-2.5-flash"],
    "Anthropic": ["claude-sonnet-4-20250514", "claude-3-haiku-20240307"]
}

# ==============================================================================
# 3. CACHED RESOURCES
# ==============================================================================

@st.cache_resource
def get_api_clients():
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        return {
            "OpenAI": OpenAI(api_key=st.secrets["OPENAI_API_KEY"]),
            "Google": genai,
            "Anthropic": Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        }
    except Exception as e:
        st.error(f"Failed to initialize API clients. Check API keys. Error: {e}")
        st.stop()

@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# ==============================================================================
# 4. CORE LOGIC FUNCTIONS
# ==============================================================================

def extract_text_from_html(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
            return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        st.error(f"Error reading {os.path.basename(file_path)}: {e}")
        return None

def setup_vector_db(collection, openai_client, force_rebuild=False):
    if collection.count() > 0 and not force_rebuild:
        st.sidebar.info(f"Vector DB contains {collection.count()} chunks.")
        return

    st.sidebar.warning("Building Vector DB...")
    with st.spinner("Processing HTML files and creating embeddings..."):
        source_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".html")]
        if not source_files:
            st.sidebar.error(f"No HTML files found in '{SOURCE_DIR}'.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks, ids = [], []

        for filename in source_files:
            text = extract_text_from_html(os.path.join(SOURCE_DIR, filename))
            if text:
                doc_chunks = splitter.split_text(text)
                for i, chunk in enumerate(doc_chunks):
                    chunks.append(chunk)
                    ids.append(f"{filename}_chunk_{i+1}")

        for i in range(0, len(chunks), 100):
            try:
                response = openai_client.embeddings.create(
                    input=chunks[i:i+100], model="text-embedding-3-small"
                )
                embeddings = [item.embedding for item in response.data]
                collection.add(documents=chunks[i:i+100], ids=ids[i:i+100], embeddings=embeddings)
            except Exception as e:
                st.error(f"Embedding failed for batch starting at {i}: {e}")

    st.sidebar.success(f"Vector DB built with {collection.count()} chunks ✅")

def get_relevant_info(collection, openai_client, query, n_results=4):
    try:
        query_embedding = openai_client.embeddings.create(
            input=[query], model="text-embedding-3-small"
        ).data[0].embedding
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        docs = results.get('documents', [[]])[0]
        return "\n---\n".join(docs) if docs else "No relevant information found."
    except Exception as e:
        st.error(f"Vector search failed: {e}")
        return "Error retrieving information."

def get_llm_response(clients, llm_provider, model_name, prompt, context, chat_history):
    system_prompt = "You are a helpful assistant. Use the provided CONTEXT and HISTORY to answer accurately. If unsure, say you don't know."
    user_prompt = f"CONTEXT:\n{context}\n\nHISTORY:\n{chat_history}\n\nQUESTION:\n{prompt}"

    try:
        with st.spinner(f"Generating answer with {model_name}..."):
            client = clients[llm_provider]
            if llm_provider == "OpenAI":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}],
                    max_completion_tokens=2048
                )
                return response.choices[0].message.content

            elif llm_provider == "Google":
                model_obj = client.GenerativeModel(model_name)
                return model_obj.generate_content(f"{system_prompt}\n\n{user_prompt}").text

            elif llm_provider == "Anthropic":
                response = client.messages.create(
                    model=model_name, system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=2048
                )
                return response.content[0].text
    except Exception as e:
        st.error(f"LLM error: {e}")
        return "Sorry, the AI model encountered an error."

# ==============================================================================
# 5. STREAMLIT APP (ISOLATED MEMORY)
# ==============================================================================

def main():
    st.set_page_config(page_title="HW5: Smart Memory Chatbot", page_icon="🤖")
    st.title("🤖 HW5: Smart Memory Chatbot")
    st.write("Ask questions about your documents. The bot retrieves relevant info and answers intelligently.")

    clients = get_api_clients()
    collection = get_chroma_collection()

    # 🧠 Use a separate memory key just for HW5
    if "hw5_messages" not in st.session_state:
        st.session_state.hw5_messages = []

    # Sidebar controls
    st.sidebar.header("Settings")
    provider = st.sidebar.selectbox("LLM Provider", list(MODEL_OPTIONS.keys()))
    model = st.sidebar.selectbox("Model", MODEL_OPTIONS[provider])

    # Add clear chat button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.hw5_messages = []
        st.rerun()

    if st.sidebar.button("Rebuild Vector DB"):
        chromadb.PersistentClient(path=CHROMA_DB_PATH).delete_collection(name=CHROMA_COLLECTION_NAME)
        st.rerun()

    setup_vector_db(collection, clients["OpenAI"])

    # Display conversation
    for msg in st.session_state.hw5_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_query := st.chat_input("Ask a question..."):
        st.session_state.hw5_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        context = get_relevant_info(collection, clients["OpenAI"], user_query)
        chat_history = "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.hw5_messages[-6:-1]]
        )

        response = get_llm_response(clients, provider, model, user_query, context, chat_history)
        full_response = f"**Answer from `{model}`:**\n\n{response}"
        st.session_state.hw5_messages.append({"role": "assistant", "content": full_response})

        with st.chat_message("assistant"):
            st.markdown(full_response)

if __name__ == "__main__":
    main()
