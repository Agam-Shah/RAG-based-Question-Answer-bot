
import hashlib
import streamlit as st
from config.settings import MODEL_NAME, UPLOAD_DIR, CHROMA_DIR
from state.session import init_session
from ui.sidebar import render_sidebar
from ui.qa import render_qa
from services.loader import load_pdf_parallel
from services.splitter import split_documents
from services.embeddings import add_to_vectorstore, get_vectorstore
from services.llm import build_llm
from services.pipeline import RAGPipeline

st.set_page_config(page_title="PDF Q&A (Local RAG)", page_icon="📄", layout="wide")

init_session()
st.title("PDF Q&A — Local RAG (OOP + LCEL)")
render_sidebar()

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    file_path = UPLOAD_DIR / f"{file_hash}.pdf"

    # Save PDF if new
    if not file_path.exists():
        with open(file_path, "wb") as f:
            f.write(pdf_bytes)

        # Load and split PDF into chunks
        docs = load_pdf_parallel(str(file_path))
        chunks = split_documents(docs)

        # Add chunks to vectorstore
        vectorstore = add_to_vectorstore(chunks, str(CHROMA_DIR), file_hash)

    else:
        # Load existing vectorstore if PDF already processed
        vectorstore = get_vectorstore(str(CHROMA_DIR))

    # Build LLM and RAG pipeline
    llm = build_llm(MODEL_NAME)
    rag = RAGPipeline(vectorstore, llm, k=2)  # k=2 reduces repeated chunks

    st.session_state.vectorstore = vectorstore
    st.session_state.rag = rag

# QA panel
rag = st.session_state.get("rag")  # your pipeline object
if rag:
    result = render_qa(rag)
    if result:
        st.session_state.history.append(result)
