import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import process_pdf, load_chain

load_dotenv()

st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Healthcare Assistant")
st.markdown("Upload medical PDFs and ask healthcare-related questions.")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chain" not in st.session_state:
    st.session_state.chain = None

# Sidebar
with st.sidebar:

    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Upload Healthcare PDF",
        type=["pdf"]
    )

    if uploaded_file:

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:

            temp_file.write(uploaded_file.read())

            temp_path = temp_file.name

        with st.spinner("Processing PDF..."):

            process_pdf(temp_path)

            st.session_state.chain = load_chain()

        st.success("PDF processed successfully!")

# Main Chat UI
question = st.chat_input(
    "Ask a medical question..."
)

if question:

    if st.session_state.chain is None:

        st.warning("Please upload a PDF first.")

    else:

        with st.spinner("Generating response..."):

            response = st.session_state.chain.invoke({
                "question": question
            })

            answer = response["answer"]

            st.session_state.chat_history.append(
                ("You", question)
            )

            st.session_state.chat_history.append(
                ("AI", answer)
            )

# Display Chat
for sender, message in st.session_state.chat_history:

    if sender == "You":

        with st.chat_message("user"):
            st.write(message)

    else:

        with st.chat_message("assistant"):
            st.write(message)

# Footer
st.markdown("---")

st.caption(
    "⚠️ This assistant is for educational purposes only. Consult a doctor for professional medical advice."
)
