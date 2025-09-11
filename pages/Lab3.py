import streamlit as st
from openai import OpenAI
import os
import tiktoken  # for token counting

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
st.set_page_config(page_title="Lab 3B — Chatbot with Buffer", page_icon="💬")
st.title(" Lab 3 — Gautam's Chatbot")
st.write(
    """
    This chatbot remembers the last 2 exchanges by default.  
    Advanced mode: limit messages so total tokens ≤ max_tokens.
    """
)

# --- API key ---
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not openai_api_key:
    st.error("Missing OpenAI API key. Please set it in .streamlit/secrets.toml or Streamlit Cloud secrets.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar controls ---
st.sidebar.header("Buffer Settings")
buffer_type = st.sidebar.radio(
    "Choose buffer type:",
    ("Last 2 exchanges", "Token-based"),
)

max_tokens = st.sidebar.slider("Max tokens (for token-based buffer)", 200, 2000, 800, 100)

# --- Display conversation ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User input ---
if prompt := st.chat_input("Type your message here..."):
    # Add user msg to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show user msg
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---- Conversation buffer ----
    if buffer_type == "Last 2 exchanges":
        buffer_messages = st.session_state.messages[-4:]
    else:
        # Token-based buffer
        buffer_messages = []
        total_tokens = 0
        # Traverse backwards until token budget is exceeded
        for msg in reversed(st.session_state.messages):
            msg_tokens = num_tokens_from_messages([msg])
            if total_tokens + msg_tokens <= max_tokens:
                buffer_messages.insert(0, msg)  # prepend
                total_tokens += msg_tokens
            else:
                break

    # ---- Assistant reply ----
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=buffer_messages,
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
