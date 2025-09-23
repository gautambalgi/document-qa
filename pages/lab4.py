import streamlit as st
import os
import fitz  # PyMuPDF

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

# Vendor SDKs
from openai import OpenAI as OpenAIClient
import anthropic
import google.generativeai as genai


# ==============================
# Setup VectorDB (DuckDB backend)
# ==============================
def setup_lab4_vectorDB(pdf_folder="pdfs"):
    if "Lab4_vectorDB" in st.session_state:
        return st.session_state.Lab4_vectorDB

    # ✅ Use DuckDB to avoid sqlite issues on Streamlit Cloud
    client = chromadb.Client(Settings(anonymized_telemetry=False, persist_directory=".chromadb"))

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )

    collection = client.get_or_create_collection(
        name="Lab4Collection",
        embedding_function=openai_ef
    )

    if collection.count() == 0:
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
        for fname in pdf_files:
            path = os.path.join(pdf_folder, fname)
            doc = fitz.open(path)
            for i, page in enumerate(doc):
                text = page.get_text().strip()
                if text:
                    collection.add(
                        ids=[f"{fname}_p{i}"],
                        documents=[text],
                        metadatas=[{"filename": fname}]
                    )
            doc.close()

    st.session_state.Lab4_vectorDB = collection
    return collection


# ==============================
# Streamlit Page
# ==============================
st.title("💬 Lab 4b — Course Info Chatbot")

# Sidebar: model selection
st.sidebar.header("Model Vendor & Variant")
vendor = st.sidebar.selectbox("Vendor", ["OpenAI", "Anthropic", "Google (Gemini)"])
if vendor == "OpenAI":
    model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
elif vendor == "Anthropic":
    model = st.sidebar.selectbox("Model", ["claude-3-5-haiku-20241022", "claude-sonnet-4-20250514"])
else:
    model = st.sidebar.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])

# Load VectorDB
collection = setup_lab4_vectorDB("pdfs")

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_q := st.chat_input("Ask me about the course materials..."):
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # 1. Retrieve from VectorDB
    results = collection.query(query_texts=[user_q], n_results=3)
    retrieved_chunks = results["documents"][0]
    retrieved_files = [md["filename"] for md in results["metadatas"][0]]
    context_text = "\n\n".join(retrieved_chunks)

    # 2. Build system prompt
    system_prompt = (
        "You are a helpful course assistant. "
        "Answer clearly and simply. "
        "If you use retrieved knowledge, say: 'Based on the course PDFs...' "
        "If not found, say: 'I don’t know from the PDFs, but here's what I think...'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {user_q}\n\nRelevant course material:\n{context_text}"}
    ]

    # 3. Choose vendor
    answer = ""
    if vendor == "OpenAI":
        client = OpenAIClient(api_key=st.secrets["OPENAI_API_KEY"])
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=500)
        answer = resp.choices[0].message.content
    elif vendor == "Anthropic":
        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        resp = client.messages.create(model=model, max_tokens=400, messages=messages)
        answer = "".join([b.text for b in resp.content if b.type == "text"])
    else:  # Gemini
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model_obj = genai.GenerativeModel(model)
        resp = model_obj.generate_content(f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {user_q}")
        answer = resp.text

    # 4. Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
