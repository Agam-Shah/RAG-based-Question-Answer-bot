import streamlit as st

def init_session():
    st.session_state.setdefault("vectorstore", None)
    st.session_state.setdefault("rag", None)
    st.session_state.setdefault("history", [])

def reset_session():
    st.session_state.clear()
    init_session()
