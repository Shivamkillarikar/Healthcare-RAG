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
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# Sidebar
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Upload Healthcare PDF", type=["pdf"])

    if uploaded_file:
        # ✅ Fix 1: only re-process if it's a new file
        if uploaded_file.name != st.session_state.processed_file:
            # ✅ Fix 2: suffix + flush + cleanup
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp.flush()
                temp_path = tmp.name

            try:
                with st.spinner("Processing PDF..."):
                    process_pdf(temp_path)
                    st.session_state.chain = load_chain()
                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.chat_history = []  # clear old chat
                st.success(f"✓ {uploaded_file.name} processed!")
            except Exception as e:
                st.error(f"Failed to process PDF: {e}")
            finally:
                os.unlink(temp_path)  # always clean up temp file
        else:
            st.info(f"✓ {uploaded_file.name} already loaded.")

    if st.session_state.chain:
        if st.button("Clear chat"):
            st.session_state.chat_history = []

# ✅ Fix 3: display history BEFORE input
for sender, message in st.session_state.chat_history:
    role = "user" if sender == "You" else "assistant"
    with st.chat_message(role):
        if isinstance(message, dict):
            st.write(message["answer"])
            if message.get("sources"):
                with st.expander(f"📄 Sources ({len(message['sources'])})"):
                    for doc in message["sources"]:
                        page = doc.metadata.get("page", "?")
                        st.caption(f"Page {page + 1}: {doc.page_content[:300]}…")
        else:
            st.write(message)

# Input
question = st.chat_input("Ask a medical question...")

if question:
    if st.session_state.chain is None:
        st.warning("Please upload a PDF first.")
    else:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    response = st.session_state.chain.invoke({"question": question})
                    answer = response["answer"]
                    # ✅ Fix 5: capture sources
                    sources = response.get("source_documents", [])

                    st.write(answer)
                    if sources:
                        with st.expander(f"📄 Sources ({len(sources)})"):
                            for doc in sources:
                                page = doc.metadata.get("page", "?")
                                st.caption(f"Page {page + 1}: {doc.page_content[:300]}…")

                    st.session_state.chat_history.append(("You", question))
                    st.session_state.chat_history.append(
                        ("AI", {"answer": answer, "sources": sources})
                    )
                except Exception as e:
                    st.error(f"Error generating response: {e}")

st.markdown("---")
st.caption("⚠️ For educational purposes only. Always consult a qualified doctor.")
