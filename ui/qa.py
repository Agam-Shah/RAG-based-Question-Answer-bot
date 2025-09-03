import streamlit as st
from services.pipeline import RAGPipeline  # ✅ import your OOP pipeline

def render_qa(rag: RAGPipeline):
    """
    Streamlit UI for Q&A using our RAGPipeline.
    """
    with st.form("qa_form"):
        query = st.text_input("Your question:")
        ask = st.form_submit_button("Ask")

    if ask and query.strip():
        with st.spinner("Generating answer... ⚡"):
            result = rag.ask(query)  # ✅ use OOP method

        st.write("### 🧠 Answer")
        st.write(result["answer"])  # ✅ changed from result["result"]

        if result["source_documents"]:
            st.write("### 📚 Sources")
            for doc in result["source_documents"]:
                st.write(f"- {doc.metadata.get('source', 'Unknown source')}")

        return result
