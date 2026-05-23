import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

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
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# Sidebar
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Upload Healthcare PDF", type=["pdf"])

    if uploaded_file:
        if uploaded_file.name != st.session_state.processed_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp.flush()
                temp_path = tmp.name

            try:
                with st.spinner("Processing PDF..."):
                    process_pdf(temp_path)
                    retriever, chain = load_chain()
                    st.session_state.retriever = retriever
                    st.session_state.chain = chain
                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.chat_history = []
                st.success(f"✓ {uploaded_file.name} processed!")
            except Exception as e:
                st.error(f"Failed to process PDF: {e}")
            finally:
                os.unlink(temp_path)
        else:
            st.info(f"✓ {uploaded_file.name} already loaded.")

    if st.session_state.chain:
        if st.button("Clear chat"):
            st.session_state.chat_history = []

# Display chat history
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

# Chat input
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
                    # Build message history for prompt
                    history = []
                    for sender, msg in st.session_state.chat_history:
                        if sender == "You":
                            history.append(HumanMessage(content=msg))
                        else:
                            content = msg["answer"] if isinstance(msg, dict) else msg
                            history.append(AIMessage(content=content))

                    # Retrieve relevant chunks
                    docs = st.session_state.retriever.invoke(question)
                    context = "\n\n".join(d.page_content for d in docs)

                    # Run chain
                    answer = st.session_state.chain.invoke({
                        "context": context,
                        "chat_history": history,
                        "question": question
                    })

                    st.write(answer)

                    if docs:
                        with st.expander(f"📄 Sources ({len(docs)})"):
                            for doc in docs:
                                page = doc.metadata.get("page", "?")
                                st.caption(f"Page {page + 1}: {doc.page_content[:300]}…")

                    st.session_state.chat_history.append(("You", question))
                    st.session_state.chat_history.append(
                        ("AI", {"answer": answer, "sources": docs})
                    )

                except Exception as e:
                    st.error(f"Error generating response: {e}")

st.markdown("---")
st.caption("⚠️ For educational purposes only. Always consult a qualified doctor.")
