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
    "Gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
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

def two_chunk_split(text: str):
    if not text:
        return ["", ""]
    mid = max(1, len(text) // 2)
    return [text[:mid].strip(), text[mid:].strip()]

def setup_vector_db(html_folder="htmls", persist_dir=".chroma_hw4"):
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
        html_files = sorted([f for f in os.listdir(html_folder) if f.lower().endswith(".html")])
        for fname in html_files:
            path = os.path.join(html_folder, fname)
            try:
                raw_text = read_html_as_text(path)
                chunk_a, chunk_b = two_chunk_split(raw_text)
                docs = [chunk_a, chunk_b]
                ids = [f"{fname}::A", f"{fname}::B"]
                metas = [{"filename": fname, "chunk": "A"},
                         {"filename": fname, "chunk": "B"}]
                collection.add(ids=ids, documents=docs, metadatas=metas)
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
    """Dispatch to the right provider/model."""
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

collection = setup_vector_db("htmls", ".chroma_hw4")

# ==============================
# Chat UI
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("Ask about iSchool student organizations…")

if user_q:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_q})

    # Show it immediately
    with st.chat_message("user"):
        st.markdown(user_q)

    try:
        results = collection.query(query_texts=[user_q], n_results=3)
        retrieved = results.get("documents", [[]])[0] if results else []
        retrieved_text = "\n\n---\n".join(retrieved) if retrieved else ""
    except Exception as e:
        st.warning(f"Vector DB query failed: {e}")
        retrieved_text = ""

    if show_retrieval and retrieved_text:
        with st.expander("🔎 Retrieved context"):
            st.write(retrieved_text)

    system_prompt = (
        "You are a helpful iSchool assistant. Answer clearly and accurately. "
        "If you used retrieved info, begin with: 'Based on the iSchool org pages…' "
        "If nothing relevant was found, say: 'I couldn’t find this in the org pages, but…' "
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