# HW4.py — iSchool Student Organizations Chatbot (RAG) with Multiple AI Providers

import sys

# --- Fix sqlite for Chroma on Streamlit Cloud ---
try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os, re, time
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic
import google.generativeai as genai

# ==============================
# App Config
# ==============================
st.set_page_config(
    page_title="HW4 — iSchool Orgs Chatbot (RAG)",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.title("🎓 HW4 — iSchool Student Orgs Chatbot (RAG)")

# ==============================
# Sidebar Settings
# ==============================
st.sidebar.header("⚙️ Settings")

PROVIDERS = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-4.1"],
    "Gemini": ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-2.0-flash-thinking-exp-01-21"],
    "Claude": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"],
}

provider = st.sidebar.selectbox("Choose AI provider", list(PROVIDERS.keys()))
model_version = st.sidebar.selectbox("Choose model version", PROVIDERS[provider])

show_retrieval = st.sidebar.checkbox("Show retrieved context", value=False)

st.sidebar.markdown(
    """
**Note**: API keys must be set in `.streamlit/secrets.toml`:  
- `OPENAI_API_KEY`  
- `ANTHROPIC_API_KEY`  
- `GOOGLE_API_KEY`
"""
)

# ==============================
# Helpers
# ==============================
def read_html_as_text(path: str) -> str:
    """Extract visible text from HTML file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            text = soup.get_text(separator="\n")
    except Exception:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
            text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

def chunk_text(text: str, max_chars=1200, overlap=200):
    """Split text into overlapping chunks for better retrieval."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max_chars - overlap
    return chunks if chunks else [text]

def setup_vector_db(html_folder="su_orgs", persist_dir=".chroma_hw4"):
    """Load or build vector DB from HTMLs."""
    if "HW4_vector_collection" in st.session_state:
        return st.session_state.HW4_vector_collection

    client = chromadb.PersistentClient(path=persist_dir)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )

    collection = client.get_or_create_collection(
        name="HW4_iSchool_Orgs",
        embedding_function=ef,
    )

    if collection.count() == 0:
        if not os.path.exists(html_folder):
            st.error(f"Folder `{html_folder}` not found.")
            st.stop()

        html_files = sorted([f for f in os.listdir(html_folder) if f.lower().endswith(".html")])
        if not html_files:
            st.error(f"No HTML files found in `{html_folder}`.")
            st.stop()

        for fname in html_files:
            path = os.path.join(html_folder, fname)
            try:
                raw_text = read_html_as_text(path)
                chunks = chunk_text(raw_text)
                ids = [f"{fname}::{i}" for i in range(len(chunks))]
                metas = [{"filename": fname, "chunk": i} for i in range(len(chunks))]
                collection.add(ids=ids, documents=chunks, metadatas=metas)
                time.sleep(0.01)
            except Exception as e:
                print(f"Skipping {fname}: {e}")

    st.session_state.HW4_vector_collection = collection
    return collection

def trim_memory(messages, max_turns=5):
    cap = max_turns * 2
    if len(messages) > cap:
        return messages[-cap:]
    return messages

def build_messages(system_prompt, history, user_q, retrieved_text):
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(history)
    user_block = (
        f"Question:\n{user_q}\n\n"
        "Retrieved context (from iSchool org HTML documents):\n"
        f"{retrieved_text if retrieved_text.strip() else '[No relevant context found]'}"
    )
    msgs.append({"role": "user", "content": user_block})
    return msgs

def call_model(provider, model, messages):
    """Dispatch to correct provider."""
    if provider == "OpenAI":
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )
        return resp.choices[0].message.content

    elif provider == "Claude":
        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        content = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        resp = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text

    elif provider == "Gemini":
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model_obj = genai.GenerativeModel(model)
        content = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        resp = model_obj.generate_content(content)
        return resp.text

    else:
        return "Provider not supported."

# ==============================
# Init
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

collection = setup_vector_db("su_orgs", ".chroma_hw4")

# ==============================
# Chat UI
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("Ask about iSchool student organizations…")

if user_q:
    # Save and show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    try:
        if collection.count() == 0:
            st.error("Vector DB is empty. Please check your su_orgs folder.")
            retrieved_text = ""
        else:
            results = collection.query(query_texts=[user_q], n_results=5)
            retrieved = results.get("documents", [[]])[0] if results else []
            retrieved_text = "\n\n---\n".join(retrieved) if retrieved else ""
    except Exception as e:
        st.warning(f"Vector DB query failed: {e}")
        retrieved_text = ""

    if show_retrieval and retrieved_text:
        with st.expander("🔎 Retrieved context"):
            st.write(retrieved_text)

    system_prompt = (
        "You are a helpful iSchool assistant. Use the retrieved context as much as possible. "
        "If you see partial matches, still try to answer from them. "
        "Only say 'not found' if there is truly no relevant info in the org pages."
    )

    history = trim_memory(st.session_state.messages[:-1], 5)
    messages = build_messages(system_prompt, history, user_q, retrieved_text)

    try:
        answer = call_model(provider, model_version, messages)
    except Exception as e:
        st.error(f"Model call failed: {e}")
        answer = "Sorry — the model call failed."

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(f"_Answer generated by **{provider} / {model_version}**_")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.messages = trim_memory(st.session_state.messages, 5)

# ==============================
# Footer
# ==============================
st.divider()
st.markdown("""
**Deployment tips:**
- Add API keys to `.streamlit/secrets.toml`:
""")