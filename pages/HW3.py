# --- HW3: Multi-doc, Multi-model, Multi-memory Guided Chatbot ---
# Extends your Lab 3 file with:
# - URL1/URL2 ingestion (requests+BeautifulSoup)
# - Per-URL summaries and relevant passage extraction
# - Vendor + model selection (OpenAI / Anthropic / Google)
# - Memory strategies: Turn buffer (6), Conversation summary, Token buffer (~2000)
# - Grounded answers with simple source tags [URL1]/[URL2]
# - Keeps your yes/no → "DO YOU WANT MORE INFO?" flow

import os
import re
import time
from typing import List, Dict, Any, Tuple

import streamlit as st
import requests
from bs4 import BeautifulSoup

# Vendor SDKs
from openai import OpenAI as OpenAIClient
import anthropic
import google.generativeai as genai

import tiktoken

# =========================
# Token counting (approx.)
# =========================
def num_tokens_from_messages(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> int:
    """Estimate number of tokens used by a list of messages (OpenAI-style)."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = 0
    for msg in messages:
        tokens += 4  # base per-message cost (approx)
        tokens += len(encoding.encode(msg.get("content", "")))
    return tokens

# =========================
# Page setup
# =========================
st.set_page_config(page_title="HW3 — Multi-Doc Guided Chatbot", page_icon="🧠")
st.title("🧠 HW3 — Multi-Doc Guided Chatbot")
st.caption("Answers are simplified so a 10-year-old can understand. Uses your URLs as sources.")

# =========================
# API keys & clients
# =========================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

# Clients are created lazily only if needed
_openai_client = None
_anthropic_client = None
_gemini_ready = False

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            st.sidebar.warning("OpenAI key missing. Set OPENAI_API_KEY in secrets/env.")
        _openai_client = OpenAIClient(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    return _openai_client

def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_API_KEY:
            st.sidebar.warning("Anthropic key missing. Set ANTHROPIC_API_KEY in secrets/env.")
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
    return _anthropic_client

def ensure_gemini():
    global _gemini_ready
    if not _gemini_ready:
        if not GOOGLE_API_KEY:
            st.sidebar.warning("Google key missing. Set GOOGLE_API_KEY in secrets/env.")
        else:
            genai.configure(api_key=GOOGLE_API_KEY)
            _gemini_ready = True
    return _gemini_ready

# =========================
# Session state
# =========================
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []  # whole chat history
if "mode" not in st.session_state:
    st.session_state.mode = "awaiting_question"  # or "offer_more_pending"
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_short_answer" not in st.session_state:
    st.session_state.last_short_answer = None
if "summary" not in st.session_state:
    st.session_state.summary = ""  # rolling conversation summary (for the memory mode)
if "docs" not in st.session_state:
    # {"url1": {"url": str, "text": str, "summary": str, "ts": float},
    #  "url2": {...}}
    st.session_state.docs: Dict[str, Dict[str, Any]] = {}

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Sources")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2 (optional)")

st.sidebar.header("Model Vendor & Variant")
vendor = st.sidebar.selectbox("Vendor", ["OpenAI", "Anthropic", "Google (Gemini)"])
if vendor == "OpenAI":
    model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
elif vendor == "Anthropic":
    model = st.sidebar.selectbox("Model", ["claude-3-5-haiku-20241022", "claude-sonnet-4-20250514"])
else:
    model = st.sidebar.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])

st.sidebar.header("Memory Strategy")
memory_mode = st.sidebar.radio(
    "Choose memory:",
    ("Turn buffer (last 6)", "Conversation summary", "Token buffer (~2000)")
)
max_ctx_tokens = st.sidebar.slider("Max tokens (for Token buffer)", 500, 4000, 2000, 100)

colA, colB = st.sidebar.columns(2)
with colA:
    process_btn = st.button("Process URLs")
with colB:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.mode = "awaiting_question"
        st.session_state.last_question = None
        st.session_state.last_short_answer = None
        st.session_state.summary = ""
        st.toast("Chat cleared.", icon="🧹")

# =========================
# Helpers: URL fetching & cleaning
# =========================
def fetch_url(url: str) -> str:
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove script/style/nav elements
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "form"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        # Normalize
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()
        return text
    except Exception as e:
        return f"[ERROR fetching {url}: {e}]"

def update_docs(url_key: str, url_val: str):
    if not url_val:
        st.session_state.docs.pop(url_key, None)
        return
    text = fetch_url(url_val)
    st.session_state.docs[url_key] = {
        "url": url_val,
        "text": text,
        "summary": "",  # filled later
        "ts": time.time(),
    }

def chunk_text(text: str, chunk_chars: int = 1400) -> List[str]:
    chunks, buf = [], []
    total = 0
    for para in text.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        if total + len(p) > chunk_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, total = [], 0
        buf.append(p)
        total += len(p)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks if chunks else ([text] if text else [])

def extract_relevant_passages(question: str, text: str, k: int = 3) -> List[str]:
    """Very simple keyword overlap scoring to pick k paragraphs."""
    if not question or not text:
        return []
    q_words = set(re.findall(r"\w+", question.lower()))
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    scored = []
    for p in paras:
        p_words = set(re.findall(r"\w+", p.lower()))
        score = len(q_words & p_words)
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for s, p in scored[:k] if s > 0] or paras[:k]

# =========================
# LLM plumbing (unified)
# =========================
def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Flatten role-based messages to a simple text prompt for Anthropic/Gemini."""
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            lines.append(f"System:\n{content}\n")
        elif role == "assistant":
            lines.append(f"Assistant:\n{content}\n")
        else:
            lines.append(f"User:\n{content}\n")
    lines.append("Assistant:")
    return "\n".join(lines)

def openai_stream(messages: List[Dict[str, str]], model_name: str):
    client = get_openai_client()
    if client is None:
        yield "[OpenAI key missing]"
        return
    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta:
            txt = chunk.choices[0].delta.content
            if txt:
                yield txt


def openai_complete_text(messages: List[Dict[str, str]], model_name: str, max_tokens: int = 800) -> str:
    client = get_openai_client()
    if client is None:
        return "[OpenAI key missing]"
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""

def anthropic_complete_text(messages: List[Dict[str, str]], model_name: str, max_tokens: int = 800) -> str:
    client = get_anthropic_client()
    if client is None:
        return "[Anthropic key missing]"
    prompt = messages_to_prompt(messages)
    resp = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        system="You are a helpful tutor.",
    )
    out = []
    for block in resp.content:
        # text blocks have .type == "text" and .text field
        if getattr(block, "type", "") == "text":
            out.append(getattr(block, "text", ""))
    return "".join(out).strip()

def gemini_complete_text(messages: List[Dict[str, str]], model_name: str, max_tokens: int = 800) -> str:
    if not ensure_gemini():
        return "[Google key missing]"
    prompt = messages_to_prompt(messages)
    model_obj = genai.GenerativeModel(model_name)
    resp = model_obj.generate_content(prompt)
    return (resp.text or "").strip()

def llm_complete_stream(vendor: str, model_name: str, messages: List[Dict[str, str]]):
    """Yield strings for Streamlit streaming. Streaming for OpenAI; single-chunk for others."""
    if vendor == "OpenAI":
        yield from openai_stream(messages, model_name)
    elif vendor == "Anthropic":
        yield anthropic_complete_text(messages, model_name)
    else:
        yield gemini_complete_text(messages, model_name)

def llm_complete_text(vendor: str, model_name: str, messages: List[Dict[str, str]], max_tokens: int = 800) -> str:
    if vendor == "OpenAI":
        return openai_complete_text(messages, model_name, max_tokens=max_tokens)
    elif vendor == "Anthropic":
        return anthropic_complete_text(messages, model_name, max_tokens=max_tokens)
    else:
        return gemini_complete_text(messages, model_name, max_tokens=max_tokens)

# =========================
# Memory builders
# =========================
def last_n_pairs(messages: List[Dict[str, str]], n_pairs: int) -> List[Dict[str, str]]:
    # Take last 2*n messages (user+assistant pairs)
    return messages[-2 * n_pairs:] if len(messages) > 0 else []

def trim_to_token_limit(messages: List[Dict[str, str]], limit: int, model_hint: str = "gpt-4o-mini") -> List[Dict[str, str]]:
    buf = []
    total = 0
    for m in reversed(messages):
        t = num_tokens_from_messages([m], model=model_hint)
        if total + t <= limit:
            buf.insert(0, m)
            total += t
        else:
            break
    return buf

def build_memory(messages: List[Dict[str, str]], mode: str) -> List[Dict[str, str]]:
    if mode == "Turn buffer (last 6)":
        return last_n_pairs(messages, 6)
    if mode == "Conversation summary":
        mem = []
        if st.session_state.summary:
            mem.append({
                "role": "system",
                "content": f"Conversation summary so far (use as memory, don't repeat it): {st.session_state.summary}"
            })
        # Keep last 2 turns verbatim to preserve recency
        mem.extend(last_n_pairs(messages, 2))
        return mem
    # Token buffer
    return trim_to_token_limit(messages, limit=max_ctx_tokens)

def update_conversation_summary(latest_user: str, latest_assistant: str):
    """Keep a rolling short synopsis to help long chats."""
    if not latest_assistant:
        return
    sys = "Summarize the conversation so far in 3-4 concise bullet points. Keep important facts and user preferences."
    prior = st.session_state.summary or "(no summary yet)"
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Previous summary:\n{prior}"},
        {"role": "user", "content": f"New turn:\nUser: {latest_user}\nAssistant: {latest_assistant}\n\nUpdate the summary."},
    ]
    # Use the currently selected vendor/model for the summary (falls back to OpenAI if others not configured)
    try:
        new_sum = llm_complete_text(vendor, model, messages, max_tokens=240)
    except Exception:
        new_sum = prior  # don't break the app on summary error
    st.session_state.summary = (new_sum or prior).strip()

# =========================
# Build grounded prompt
# =========================
def build_context_blocks(question: str) -> Tuple[str, str]:
    """Return (summaries_block, passages_block) from URL1/URL2 if available."""
    summaries, passages = [], []
    # Summaries
    for key, label in (("url1", "URL1"), ("url2", "URL2")):
        doc = st.session_state.docs.get(key)
        if doc and doc.get("summary"):
            summaries.append(f"[{label}] {doc['summary']}")
    # Relevant passages
    for key, label in (("url1", "URL1"), ("url2", "URL2")):
        doc = st.session_state.docs.get(key)
        if doc and doc.get("text"):
            tops = extract_relevant_passages(question, doc["text"], k=3)
            if tops:
                joined = "\n\n".join(tops[:3])
                passages.append(f"[{label} excerpts]\n{joined}")
    return ("\n\n".join(summaries), "\n\n".join(passages))

def compose_messages_for_answer(user_question: str) -> List[Dict[str, str]]:
    sys = (
        "You are a kind tutor explaining concepts so a 10-year-old can understand. "
        "Answer ONLY using the provided URL summaries and excerpts. "
        "If something is not in the URLs, say you don't know. "
        "After facts, add simple source tags like [URL1] or [URL2] where appropriate."
    )
    summaries_block, passages_block = build_context_blocks(user_question)
    context = "No context provided."
    if summaries_block or passages_block:
        context = f"Summaries:\n{summaries_block or '(none)'}\n\nExcerpts:\n{passages_block or '(none)'}"

    base_msgs = [
        {"role": "system", "content": sys},
        {"role": "system", "content": f"Use this context:\n{context}"},
    ]
    # Inject memory per selected strategy
    mem_msgs = build_memory(st.session_state.messages, memory_mode)
    base_msgs.extend(mem_msgs)
    # Current user question (short answer stage)
    base_msgs.append({"role": "user", "content": f"Question: {user_question}\nGive a short, clear answer (3–5 sentences)."})
    return base_msgs

def compose_messages_for_more_info(user_question: str, prior_answer: str) -> List[Dict[str, str]]:
    sys = (
        "You are a tutor for a 10-year-old. Expand with a simple analogy and one concrete example. "
        "Stay grounded ONLY in the provided URL content; if unknown, say so. "
        "Use light source tags like [URL1]/[URL2] after factual claims."
    )
    summaries_block, passages_block = build_context_blocks(user_question)
    context = f"Summaries:\n{summaries_block or '(none)'}\n\nExcerpts:\n{passages_block or '(none)'}"
    base_msgs = [
        {"role": "system", "content": sys},
        {"role": "system", "content": f"Use this context:\n{context}"},
    ]
    # Memory
    mem_msgs = build_memory(st.session_state.messages, memory_mode)
    base_msgs.extend(mem_msgs)
    # Expansion turn
    user_content = (
        f"Original question: {user_question}\n"
        f"Short answer already given: {prior_answer}\n\n"
        "Now expand with more detail a child can understand."
    )
    base_msgs.append({"role": "user", "content": user_content})
    return base_msgs

# =========================
# Summarize documents (once per URL load)
# =========================
def summarize_text_for_doc(raw_text: str, label: str) -> str:
    if not raw_text or raw_text.startswith("[ERROR"):
        return raw_text or ""
    # Heuristic pre-shrink to keep prompt reasonable
    first_chunks = "\n\n".join(chunk_text(raw_text, chunk_chars=1200)[:2])
    sys = "Summarize the text in 250–400 words, kid-friendly, factual, no fluff."
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Text from {label}:\n{first_chunks}"},
    ]
    try:
        return llm_complete_text(vendor, model, msgs, max_tokens=420)
    except Exception as e:
        return f"[Summary error: {e}]"

def process_urls_if_needed():
    changed = False
    if url1 and (st.session_state.docs.get("url1", {}).get("url") != url1):
        update_docs("url1", url1); changed = True
    elif not url1 and "url1" in st.session_state.docs:
        st.session_state.docs.pop("url1"); changed = True

    if url2 and (st.session_state.docs.get("url2", {}).get("url") != url2):
        update_docs("url2", url2); changed = True
    elif not url2 and "url2" in st.session_state.docs:
        st.session_state.docs.pop("url2"); changed = True

    # Summarize when changed
    if changed:
        with st.status("Processing URLs…", expanded=True) as status:
            for key, label in (("url1", "URL1"), ("url2", "URL2")):
                doc = st.session_state.docs.get(key)
                if not doc:
                    continue
                st.write(f"Fetching {label}…")
                if doc["text"]:
                    st.write(f"Summarizing {label}…")
                    doc["summary"] = summarize_text_for_doc(doc["text"], label)
                    st.success(f"{label} ready.")
                else:
                    st.warning(f"{label} has no text or failed to fetch.")
            status.update(label="Done", state="complete")

# Handle sidebar button
if process_btn:
    process_urls_if_needed()

# =========================
# Conversation history
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# Yes/No helpers (unchanged)
# =========================
def is_yes(text: str) -> bool:
    return text.strip().lower() in {"yes", "y", "yeah", "yep", "sure", "ok", "okay"}

def is_no(text: str) -> bool:
    return text.strip().lower() in {"no", "n", "nope", "nah"}

def ask_more_prompt():
    with st.chat_message("assistant"):
        st.markdown("**DO YOU WANT MORE INFO?** (yes / no)")

def prompt_for_new_question():
    with st.chat_message("assistant"):
        st.markdown("What question can I help you with?")

# =========================
# Streaming answer wrappers
# =========================
def stream_short_answer(question: str) -> str:
    # Compose grounded prompt
    msgs = compose_messages_for_answer(question)
    with st.chat_message("assistant"):
        answer = st.write_stream(llm_complete_stream(vendor, model, msgs))
    return answer

def stream_more_info(question: str, prior_answer: str) -> str:
    msgs = compose_messages_for_more_info(question, prior_answer)
    with st.chat_message("assistant"):
        answer = st.write_stream(llm_complete_stream(vendor, model, msgs))
    return answer

# =========================
# Input box + state machine
# =========================
placeholder = "Ask a question or reply yes/no…"
if not (url1 or url2):
    placeholder = "Enter URLs in the sidebar first (you can still ask general questions)…"

if user_input := st.chat_input(placeholder):
    # Record user msg
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Ensure URLs are processed if user pasted them in the sidebar just now
    process_urls_if_needed()

    # State machine logic preserved from Lab 3
    if st.session_state.mode == "awaiting_question":
        st.session_state.last_question = user_input
        short = stream_short_answer(user_input)
        st.session_state.messages.append({"role": "assistant", "content": short})
        st.session_state.last_short_answer = short

        # Update summary memory if selected
        if memory_mode == "Conversation summary":
            update_conversation_summary(user_input, short)

        ask_more_prompt()
        st.session_state.messages.append({"role": "assistant", "content": "DO YOU WANT MORE INFO? (yes/no)"})
        st.session_state.mode = "offer_more_pending"

    elif st.session_state.mode == "offer_more_pending":
        if is_yes(user_input):
            more = stream_more_info(st.session_state.last_question, st.session_state.last_short_answer or "")
            st.session_state.messages.append({"role": "assistant", "content": more})

            if memory_mode == "Conversation summary":
                update_conversation_summary(st.session_state.last_question, more)

            ask_more_prompt()
            st.session_state.messages.append({"role": "assistant", "content": "DO YOU WANT MORE INFO? (yes/no)"})
        elif is_no(user_input):
            prompt_for_new_question()
            st.session_state.messages.append({"role": "assistant", "content": "What question can I help you with?"})
            st.session_state.mode = "awaiting_question"
        else:
            # Treat as a new question
            st.session_state.last_question = user_input
            short = stream_short_answer(user_input)
            st.session_state.messages.append({"role": "assistant", "content": short})
            st.session_state.last_short_answer = short

            if memory_mode == "Conversation summary":
                update_conversation_summary(user_input, short)

            ask_more_prompt()
            st.session_state.messages.append({"role": "assistant", "content": "DO YOU WANT MORE INFO? (yes/no)"})
            st.session_state.mode = "offer_more_pending"
# =========================
# Sidebar: show buffer status
# =========================
st.sidebar.header("Buffer Status")
if memory_mode == "Token buffer (~2000)":
    token_count = num_tokens_from_messages(st.session_state.messages, model="gpt-4o-mini")
    st.sidebar.write(f"Current tokens in chat: **{token_count} / {max_ctx_tokens}**")
else:
    st.sidebar.write("Token count shown only in Token buffer mode.")