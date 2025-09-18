import os
import streamlit as st
from openai import OpenAI
import tiktoken

# --- Helper: count tokens ---
def num_tokens_from_messages(messages, model="gpt-4o-mini"):
    """Estimate number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = 0
    for msg in messages:
        tokens += 4  # base cost per message (approx)
        tokens += len(encoding.encode(msg["content"]))
    return tokens

# --- Page setup ---
st.set_page_config(page_title="Lab 3 — Gautam's Guided Chatbot", page_icon="💬")
st.title("💬 Lab 3 — Gautam's Guided Chatbot")
st.caption("Answers are simplified so a 10-year-old can understand them.")

# --- API key ---
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not openai_api_key:
    st.error("Missing OpenAI API key. Please set it in .streamlit/secrets.toml or Streamlit Cloud secrets.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "awaiting_question"  # or "offer_more_pending"
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_short_answer" not in st.session_state:
    st.session_state.last_short_answer = None

# --- Sidebar controls ---
st.sidebar.header("Buffer Settings")
buffer_type = st.sidebar.radio(
    "Choose buffer type:",
    ("Last 20 exchanges", "Token-based"),
)
max_tokens = st.sidebar.slider("Max tokens (for token-based buffer)", 200, 4000, 1500, 100)

# --- Display conversation history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Helper functions ---
def is_yes(text: str) -> bool:
    return text.strip().lower() in {"yes", "y", "yeah", "yep", "sure", "ok"}

def is_no(text: str) -> bool:
    return text.strip().lower() in {"no", "n", "nope", "nah"}

def build_buffer(messages):
    """Return buffer based on selection."""
    if buffer_type == "Last 20 exchanges":
        return messages[-40:]  # 20 user+assistant pairs
    else:
        buffer_messages = []
        total_tokens = 0
        for msg in reversed(messages):
            msg_tokens = num_tokens_from_messages([msg])
            if total_tokens + msg_tokens <= max_tokens:
                buffer_messages.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break
        return buffer_messages

def stream_short_answer(question: str) -> str:
    sys = "You are a tutor explaining things to a 10-year-old. Be simple, clear, and kind. Keep it short (3–5 sentences)."
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": f"Question: {question}\nAnswer in simple terms for a 10-year-old."},
            ],
            stream=True,
        )
        answer = st.write_stream(stream)
    return answer

def stream_more_info(question: str, prior_answer: str) -> str:
    sys = "You are a tutor for a 10-year-old. Add more detail with an example and a simple analogy, but still be easy to understand."
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": (
                    f"Original question: {question}\n"
                    f"Short answer already given: {prior_answer}\n\n"
                    "Now expand with more info a child can understand."
                )},
            ],
            stream=True,
        )
        answer = st.write_stream(stream)
    return answer

def ask_more_prompt():
    with st.chat_message("assistant"):
        st.markdown("**DO YOU WANT MORE INFO?** (yes / no)")

def prompt_for_new_question():
    with st.chat_message("assistant"):
        st.markdown("What question can I help you with?")

# --- User input handling ---
if prompt := st.chat_input("Ask a question or reply yes/no…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # State machine
    if st.session_state.mode == "awaiting_question":
        st.session_state.last_question = prompt
        short = stream_short_answer(prompt)
        st.session_state.messages.append({"role": "assistant", "content": short})
        st.session_state.last_short_answer = short
        ask_more_prompt()
        st.session_state.messages.append({"role": "assistant", "content": "DO YOU WANT MORE INFO? (yes/no)"})
        st.session_state.mode = "offer_more_pending"

    elif st.session_state.mode == "offer_more_pending":
        if is_yes(prompt):
            more = stream_more_info(st.session_state.last_question, st.session_state.last_short_answer or "")
            st.session_state.messages.append({"role": "assistant", "content": more})
            ask_more_prompt()
            st.session_state.messages.append({"role": "assistant", "content": "DO YOU WANT MORE INFO? (yes/no)"})
        elif is_no(prompt):
            prompt_for_new_question()
            st.session_state.messages.append({"role": "assistant", "content": "What question can I help you with?"})
            st.session_state.mode = "awaiting_question"
        else:
            # Treat as a new question
            st.session_state.last_question = prompt
            short = stream_short_answer(prompt)
            st.session_state.messages.append({"role": "assistant", "content": short})
            st.session_state.last_short_answer = short
            ask_more_prompt()
            st.session_state.messages.append({"role": "assistant", "content": "DO YOU WANT MORE INFO? (yes/no)"})
            st.session_state.mode = "offer_more_pending"
