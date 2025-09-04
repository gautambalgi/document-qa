import streamlit as st
from openai import OpenAI
import os

# Get API key from Streamlit secrets or environment variable
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not openai_api_key:
    st.error("No API key found. Please set it in .streamlit/secrets.toml or as an environment variable.")
else:
    client = OpenAI(api_key=openai_api_key)

    # 👇 Define uploaded_file first
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    # 👇 Then use uploaded_file inside text_area
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document}\n\n---\n\n{question}",
            }
        ]

        stream = client.chat.completions.create(
            model="gpt-5-chat-latest",
            messages=messages,
            stream=True,
        )

        st.write_stream(stream)
