# 🧬 MediQ — Intelligent Healthcare Document Assistant

> Upload any healthcare PDF and have a role-aware, source-cited conversation with it — powered by LangChain, FAISS, and Google Gemini 2.5 Flash.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange?logo=google)](https://aistudio.google.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [How It Works](#-how-it-works)
- [Role-Based System](#-role-based-system)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)
- [Limitations](#-limitations)

---

## 🔍 Overview

MediQ is a **Retrieval-Augmented Generation (RAG)** application designed for the healthcare domain. It allows users to upload medical PDFs — clinical guidelines, research papers, drug references, or patient reports — and ask natural language questions about them.

Unlike a general-purpose chatbot, MediQ:
- **Only answers from your document** — never fabricates from general knowledge
- **Adapts its response** based on who is asking (Patient, Doctor, or Admin)
- **Cites its sources** — showing exactly which page the answer came from
- **Remembers context** — maintains conversation history across multi-turn sessions

---

## ✨ Features

- 🔐 **Role-Based Responses** — Patient, Doctor, and Admin each receive tailored answers in appropriate language and depth
- 📄 **Source Citations** — every answer surfaces the exact page and passage it was derived from
- 🧠 **Conversational Memory** — remembers previous questions within the session for follow-up queries
- 🔎 **MMR Retrieval** — Max Marginal Relevance search returns diverse, relevant chunks rather than redundant ones
- 🏠 **Local Embeddings** — HuggingFace model runs entirely on your machine — no data sent to third-party embedding APIs
- 🛡️ **Hallucination Mitigation** — strict prompt grounding and `temperature=0` keep answers factually anchored
- ☁️ **Cloud Ready** — one-click deploy to Streamlit Cloud with secure secrets management

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   UPLOAD PHASE                      │
│                                                     │
│  PDF File                                           │
│     │                                               │
│     ▼                                               │
│  PyPDFLoader ──► RecursiveCharacterTextSplitter     │
│                  (chunk_size=1000, overlap=200)      │
│                         │                           │
│                         ▼                           │
│              HuggingFace Embeddings                 │
│              (all-MiniLM-L6-v2, local)              │
│                         │                           │
│                         ▼                           │
│                 FAISS Vector Store                  │
│                 (persisted to disk)                 │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   QUERY PHASE                       │
│                                                     │
│  User Question                                      │
│     │                                               │
│     ▼                                               │
│  HuggingFace Embeddings (same model)                │
│     │                                               │
│     ▼                                               │
│  FAISS MMR Search (k=3, fetch_k=10)                 │
│     │                                               │
│     ▼                                               │
│  Top 3 Relevant Chunks                              │
│     │                                               │
│     ▼                                               │
│  ChatPromptTemplate                                 │
│  (Role System Prompt + Chat History + Question)     │
│     │                                               │
│     ▼                                               │
│  Google Gemini 2.5 Flash (temperature=0)            │
│     │                                               │
│     ▼                                               │
│  Answer + Source Citations                          │
└─────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Streamlit | Web UI, file upload, chat interface |
| Orchestration | LangChain 0.3.x (LCEL) | Pipeline composition |
| LLM | Google Gemini 2.5 Flash | Answer generation |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Local semantic embeddings |
| Vector Store | FAISS (Facebook AI) | Similarity search |
| Document Loader | PyPDFLoader | PDF text extraction |
| Text Splitting | RecursiveCharacterTextSplitter | Semantic chunking |
| Memory | LangChain MessagesPlaceholder | Conversation history |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- Google Gemini API key → [Get one free at aistudio.google.com](https://aistudio.google.com/apikey)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/mediq.git
cd mediq

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
```

Open `.env` and add your API key:
```env
GOOGLE_API_KEY=your-gemini-api-key-here
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ How It Works

### 1. Document Ingestion
When you upload a PDF, MediQ:
- Extracts text page-by-page using `PyPDFLoader`
- Splits it into 1000-character chunks with 200-character overlap using `RecursiveCharacterTextSplitter`
- Converts each chunk into a 384-dimensional vector using `all-MiniLM-L6-v2` locally
- Stores all vectors in a FAISS index persisted to `faiss_db/`

### 2. Retrieval
When you ask a question:
- Your question is embedded using the same model
- FAISS performs **MMR search** — fetching 10 candidates then selecting the 3 most relevant AND diverse chunks
- MMR prevents returning 3 nearly identical chunks — ensuring the answer draws from different parts of the document

### 3. Generation
The retrieved chunks are injected into a role-specific prompt:
```
System:  [Role instructions] + [Retrieved context]
History: [Previous conversation turns]
Human:   [Current question]
```
Gemini 2.5 Flash generates the answer at `temperature=0` — fully deterministic, no creativity, no hallucination.

### 4. Hallucination Mitigation
Three-layer approach:
- **Prompt grounding** — system prompt explicitly restricts answers to provided context only
- **Temperature = 0** — eliminates randomness and fabrication
- **Source display** — user can verify every answer against the original page

---

## 👤 Role-Based System

MediQ adapts its responses based on the selected role:

| Role | Language | Focus | Example Response Style |
|------|----------|-------|----------------------|
| **Patient** | Simple, empathetic | Symptoms, lifestyle, next steps | "Metformin is a pill that helps control your blood sugar..." |
| **Doctor** | Clinical, precise | Dosages, contraindications, guidelines | "Metformin (biguanide) — first-line T2DM therapy. CI: eGFR <30..." |
| **Admin** | Professional, structured | ICD codes, billing, compliance | "Metformin maps to ICD-10 E11.9. Covered under standard formulary..." |

Switching roles rebuilds the chain instantly — no need to re-upload the document.

---

## 📁 Project Structure

```
mediq/
├── app.py                  ← Streamlit frontend
├── rag_pipeline.py         ← LangChain RAG pipeline
├── requirements.txt        ← Python dependencies
├── .env.example            ← API key template
├── .gitignore              ← Excludes .env and faiss_db/
├── README.md
└── faiss_db/               ← Auto-generated after first upload
    ├── index.faiss         ← Vector index (binary)
    └── index.pkl           ← Chunk text + metadata
```

---

## ☁️ Deployment

### Streamlit Cloud (Recommended — Free)

1. Push your repo to GitHub (ensure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo and set `app.py` as the entry point
4. Go to **Settings → Secrets** and add:
```toml
GOOGLE_API_KEY = "your-gemini-api-key-here"
```
5. Click Deploy

> ⚠️ Use Python 3.11 on Streamlit Cloud. Add a `.python-version` file containing `3.11` to your repo root to enforce this.

---

## ⚠️ Limitations

- **Session only** — FAISS index resets on Streamlit Cloud restart; re-upload required
- **PDF only** — currently supports PDF format; DOCX and TXT support can be added
- **Single document** — one PDF at a time; multi-document support would require index merging
- **No authentication** — all users share the same session on a public deployment
- **Educational use only** — not a substitute for professional medical advice

---

## 🔮 Future Improvements

- [ ] Persistent vector store using Pinecone or Supabase pgvector
- [ ] Multi-document support
- [ ] RAGAS evaluation framework for measuring retrieval and answer quality
- [ ] User authentication for private deployments
- [ ] Support for DOCX, TXT, and CSV formats
- [ ] Export chat history as PDF report

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## ⚕️ Disclaimer

> MediQ is built for educational and demonstration purposes only. It is not a certified medical device and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions.

---

<p align="center">Built with LangChain · FAISS · Google Gemini · Streamlit</p>
