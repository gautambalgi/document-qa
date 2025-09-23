import streamlit as st
import os
import chromadb
from chromadb.utils import embedding_functions
import fitz  # from PyMuPDF
from openai import OpenAI

# ==============================
# Setup VectorDB
# ==============================
def setup_lab4_vectorDB(pdf_folder="pdfs"):
    if "Lab4_vectorDB" in st.session_state:
        return st.session_state.Lab4_vectorDB

    client = chromadb.Client()

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )

    # ✅ safer: get or create collection
    collection = client.get_or_create_collection(
        name="Lab4Collection",
        embedding_function=openai_ef
    )

    # Only load PDFs if collection is empty
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

# Initialize
collection = setup_lab4_vectorDB("pdfs")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_q := st.chat_input("Ask me about the course materials..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # 1. Retrieve from VectorDB
    results = collection.query(query_texts=[user_q], n_results=3)
    retrieved_chunks = [doc for doc in results["documents"][0]]
    context_text = "\n\n".join(retrieved_chunks)

    # 2. Build prompt for LLM
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

    # 3. Call OpenAI LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # can switch to gpt-4o if needed
        messages=messages,
        max_tokens=500
    )

    answer = response.choices[0].message.content

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
