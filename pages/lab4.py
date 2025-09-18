import streamlit as st
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
import chromadb
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF for PDF reading
import os





# Function to load PDFs, embed, and create Chroma collection
def setup_lab4_vectorDB(pdf_folder="pdfs"):
    if "Lab4_vectorDB" in st.session_state:
        return st.session_state.Lab4_vectorDB

    # 1. Initialize Chroma client
    client = chromadb.Client()

    # 2. Create collection
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name="text-embedding-3-small"
    )
    collection = client.get_or_create_collection(
    name="Lab4Collection",
    embedding_function=openai_ef
)


    # 3. Read all PDFs from folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    for fname in pdf_files:
        full_path = os.path.join(pdf_folder, fname)
        doc = fitz.open(full_path)
        text_chunks = []
        for page in doc:
            text = page.get_text().strip()
            if text:
                text_chunks.append(text)
        doc.close()

        # 4. Add each page/chunk to vector DB
        for i, chunk in enumerate(text_chunks):
            collection.add(
                ids=[f"{fname}_p{i}"],
                documents=[chunk],
                metadatas=[{"filename": fname}]
            )

    # Save in session state
    st.session_state.Lab4_vectorDB = collection
    return collection

# ==============================
# Lab 4 Streamlit Page
# ==============================
st.title("📚 Lab 4 – Part A: Vector Database Setup")

# Create vector DB if not exists
collection = setup_lab4_vectorDB("pdfs")  # make sure you have a folder 'pdfs' with 7 PDFs

# Test queries
test_queries = ["Generative AI", "Text Mining", "Data Science Overview"]

for query in test_queries:
    st.subheader(f"Results for: {query}")
    results = collection.query(query_texts=[query], n_results=3)
    filenames = [md["filename"] for md in results["metadatas"][0]]
    st.write(filenames)
