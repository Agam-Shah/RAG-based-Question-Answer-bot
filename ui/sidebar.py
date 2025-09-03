import streamlit as st
from state.session import reset_session
from config.settings import MODEL_NAME, CHROMA_DIR, UPLOAD_DIR

def render_sidebar():
    st.sidebar.subheader("⚙️ Settings")
    st.sidebar.write(f"**Model**: `{MODEL_NAME}`")
    st.sidebar.text(f"Chroma: {CHROMA_DIR}")
    st.sidebar.text(f"Uploads: {UPLOAD_DIR}")

    if st.sidebar.button("🔄 Reset session"):
        reset_session()
        st.sidebar.success("Session state cleared.")
