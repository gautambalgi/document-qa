import streamlit as st

# Page setup
st.set_page_config(
    page_title="Gautam's Homework Manager",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Title
st.title("📚 Gautam's Homework Manager")

# Welcome text
st.write(
    """
    Welcome! This is the landing page.  
    Use the sidebar (☰ in the top-left) to navigate to:
    - 📄 HW1 — Document Q&A  
    - 📑 HW2 — URL Summarizer  
    - 💬 Lab3 — Streaming Chatbot
    """
)

st.divider()

# Optional section with direct page links (if your Streamlit version supports st.page_link)
st.subheader("Open a Homework or Lab")

st.page_link("pages/HW1.py", label="📄 HW1 — Document Q&A", icon="📄")
st.page_link("pages/HW2.py", label="📑 HW2 — URL Summarizer", icon="📑")
st.page_link("Labs/Lab3.py", label="💬 Lab3 — Chatbot", icon="💬")

st.divider()

