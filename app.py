import streamlit as st
import os

from backend.ingest import load_documents, chunk_text
from backend.embed import generate_embeddings
from backend.store import create_index, add_embeddings
from backend.reasoning import reasoning_pipeline

# -------------------- CONFIG --------------------
st.set_page_config(page_title="R-RAG", layout="wide")
st.title("🧠 Reasoning-Aware Retrieval-Augmented Generation (R-RAG)")
st.caption("Ask questions grounded strictly in your uploaded documents")

UPLOAD_DIR = "data/uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------- SESSION STATE INIT --------------------
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "kb_built" not in st.session_state:
    st.session_state.kb_built = False

# -------------------- SIDEBAR: UPLOAD --------------------
st.sidebar.header("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# 🔥 Only reset when NEW files are uploaded
if uploaded_files and not st.session_state.kb_built:
    # Clear old files
    for f in os.listdir(UPLOAD_DIR):
        os.remove(os.path.join(UPLOAD_DIR, f))

    # Save new files
    for file in uploaded_files:
        with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
            f.write(file.getbuffer())

    st.sidebar.success("Files uploaded. Ready to build knowledge base.")

# -------------------- BUILD KB --------------------
if st.sidebar.button("🔧 Build Knowledge Base"):
    with st.spinner("Building knowledge base..."):
        docs = load_documents(UPLOAD_DIR)

        if not docs:
            st.sidebar.error("No documents found.")
        else:
            chunks = []
            for doc in docs:
                chunks.extend(chunk_text(doc))

            embeddings = generate_embeddings(chunks)
            index = create_index(embeddings.shape[1])
            add_embeddings(index, embeddings)

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.kb_built = True

            st.sidebar.success("Knowledge base built successfully!")

# -------------------- QUESTION --------------------
st.header("❓ Ask a Question")

question = st.text_input("Enter your question")

if st.button("🚀 Generate Answer"):
    if not st.session_state.kb_built:
        st.error("Please upload documents and build the knowledge base first.")
    elif not question.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Reasoning..."):
            result = reasoning_pipeline(
                question=question,
                index=st.session_state.index,
                documents=st.session_state.chunks
            )

        st.subheader("✅ Answer")
        st.write(result["answer"])

        st.subheader("📌 Sources")
        for i, src in enumerate(result["sources"], start=1):
            st.markdown(f"**Source {i}:** {src[:300]}...")