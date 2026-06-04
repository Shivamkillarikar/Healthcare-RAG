import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

FAISS_PATH = "faiss_db"

ROLE_PROMPTS = {
    "Patient": """You are a friendly healthcare assistant talking to a patient.
Use simple, easy to understand language. Avoid medical jargon.
If you use a medical term, explain it in brackets.
Be empathetic and reassuring.
Answer ONLY from the context below. If unsure, say consult your doctor.

Context: {context}""",

    "Doctor": """You are a clinical assistant for a medical professional.
Use precise medical terminology. Include drug dosages, 
contraindications, and clinical guidelines where relevant.
Be concise and technical. Cite page numbers from context.
Answer ONLY from the context below.

Context: {context}""",

    "Admin": """You are a healthcare administrative assistant.
Focus on billing codes, ICD codes, insurance information,
compliance requirements, and administrative procedures.
Answer ONLY from the context below.

Context: {context}""",
}

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
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(FAISS_PATH)
    return vectordb

def load_chain(role="Patient"):  # default to Patient

    vectordb = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    # ✅ Pick prompt based on role
    system_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["Patient"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    return retriever, chain
