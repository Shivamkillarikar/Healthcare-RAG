import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS          # ← changed
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

FAISS_PATH = "faiss_db"                                     # ← changed

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    vectordb = FAISS.from_documents(chunks, embeddings)     # ← changed
    vectordb.save_local(FAISS_PATH)                         # ← changed (no .persist())
    return vectordb

def load_chain():
    vectordb = FAISS.load_local(                            # ← changed
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True                # ← required for FAISS
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return chain
