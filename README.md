# ⚖️ LegalSaathi AI — Indian Legal Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/LangGraph-Agent_Architecture-blue?style=flat-square" alt="LangGraph">
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=flat-square" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Reranker-Qwen3--0.6B-6366F1?style=flat-square" alt="Qwen3 Reranker">
  <img src="https://img.shields.io/badge/LLM-Groq_Cloud-F55036?style=flat-square" alt="Groq">
  <img src="https://img.shields.io/badge/PostgreSQL-PostgresSaver-4169E1?style=flat-square&logo=postgresql&logoColor=white" alt="PostgreSQL">
</p>

> ⚠️ **Disclaimer**: For educational and research purposes only. Does not constitute formal legal advice. Always cross-check against official government gazettes.

---

## 📌 Overview

**LegalSaathi AI** is an intelligent legal assistant built specifically for Indian statutory law and personal legal document review. 

It uses a stateful **LangGraph Agent** paired with **Dual-Source Hybrid Retrieval** and **Cross-Encoder Reranking** to deliver accurate, citation-backed answers with zero hallucinations.

## ✨ Features

1. **Autonomous LangGraph Agent & Dynamic Tool Calling**: The agent autonomously reasons over incoming user queries—answering chit-chat and greetings directly with **zero retrieval** (saving latency and tokens), while invoking the `search_legal_documents` tool for substantive legal queries.
2. **Dual-Source Retrieval (Public Statutes + Private Vault)**: In a single prompt, cross-examine **22 core Indian central statutes** alongside user-uploaded contracts, agreements, or legal notices (`.pdf` / `.docx`) isolated by user ID.
3. **High-Precision Hybrid Search & Cross-Encoder Reranking**: Blends dense semantic retrieval (`Octen-0.6B`) with sparse lexical matching (`BM25`) via Reciprocal Rank Fusion (`RRF`), followed by `Qwen3-Reranker-0.6B` cross-encoder reranking to achieve a **0.977 Hit Rate @ 5**.
4. **Stateful Multi-Session Memory & Context Compaction**: Persists multi-turn conversations across sessions via PostgreSQL checkpointing (`PostgresSaver`), utilizing LangChain `SummarizationMiddleware` to compress history after 20 messages while preserving recent context.
5. **Real-Time Token Streaming with Exact Page Citations**: Streams answers in real-time over Server-Sent Events (SSE) alongside verified statutory section numbers and PDF page-level citations for transparent legal auditing.

---

## 🏗️ System Architecture

### 1. Ingestion Pipeline
```mermaid
flowchart LR
    A[22 Indian Acts<br/>PDF] --> B[Docling<br/>parse + chunk] --> C[Octen<br/>embeddings] --> D[(Statutes<br/>ChromaDB)]
    E[User upload<br/>PDF / DOCX] --> F[Parse + chunk] --> G[Octen<br/>embeddings] --> H[(User vault<br/>ChromaDB)]
```

### 2. Query Runtime (Stateful LangGraph Agent)
```mermaid
flowchart TD
    U([User query]) --> API[FastAPI<br/>/chat/stream]
    API --> AG{LangGraph agent<br/>+ Postgres memory}
    AG -->|chit-chat| R[Direct reply]
    AG -->|legal query| T[search tool]
    T --> S1[Statutes<br/>dense + BM25 + RRF]
    T --> S2[User vault<br/>filtered by user_id]
    S1 --> RR[Qwen3 reranker<br/>top 5 + page citations]
    S2 --> RR
    RR --> AG
    AG --> ANS[Grounded answer]
    R --> OUT([SSE stream to UI])
    ANS --> OUT
```


### ⚡ End-to-End Query Lifecycle
1. **Agent State & Memory**: User query arrives via Streamlit $\rightarrow$ FastAPI loads conversation thread state using `PostgresSaver`. If the thread exceeds 20 messages, `SummarizationMiddleware` compresses older context.
2. **Autonomous Tool Routing**:
   - **Chit-chat / Greeting**: The agent responds directly without querying vector databases.
   - **Legal Question**: The agent invokes `search_legal_documents` with 2 concurrent workers (Statutory Hybrid Search + Personal Vault Search).
3. **Candidate Merging & Cross-Encoder Reranking**: Retrieved chunks from both sources are pooled and scored using `Qwen3-Reranker-0.6B` to isolate the Top 5 most authoritative passages.
4. **Grounded Generation**: Groq LLM synthesizes the verified answer strictly using the top passages and streams tokens + PDF page citations over SSE.

## 📚 Document Coverage & Knowledge Base

### 1. Indexed Indian Statutes (Public Vault)
LegalSaathi indexes **22 core Indian central statutes**:
* **Criminal Law**: *Bharatiya Nyaya Sanhita (BNS) 2023*, *BNSS 2023*, *Bharatiya Sakshya Adhiniyam (BSA) 2023*
* **Constitution & Transparency**: *Constitution of India*, *Right to Information (RTI) Act 2005*
* **Contracts & Commerce**: *Indian Contract Act 1872*, *RERA 2016*, *Consumer Protection Act 2019*
* **Cyber & Digital Law**: *DPDP Act 2023*, *Information Technology Act 2000*, *Telecommunications Act 2023*
* **Labour & Workplace**: *Code on Wages 2019*, *Code on Social Security 2020*, *Industrial Relations Code 2020*, *OSH Code 2020*, *POSH Act 2013*
* **Welfare & Family Law**: *Domestic Violence Act 2005*, *Dowry Prohibition Act 1961*, *Child Labour Act 1986*, *Motor Vehicles Act 1988*

### 2. User Personal Document Vault (Private Vault)
* Users can upload private contracts, rental agreements, employment letters, or legal notices (`.pdf` and `.docx`).
* Stored per-user in isolated ChromaDB collections.
* Directly queryable alongside statutory provisions.

---

## 📊 Production Evaluation Benchmarks

Benchmarked using **DeepEval** (`deepseek-v4-flash` judge, `groq` generator) on 45 curated statutory test cases with the production Octen-0.6B + Qwen3-Reranker-0.6B architecture (1,200-character chunks, $K=5$, `min_score=0.30`):

| Category | Metric | Score | Description |
|---|---|:---:|---|
| **Retrieval** | **Hit Rate @ 5** | **0.977** | 44 of 45 queries retrieve exact statutory evidence in top 5 chunks |
| **Retrieval** | **MRR @ 5** | **0.841** | Key authoritative clause positioned at rank 1 or 2 on average |
| **Retrieval** | **Recall @ 5** | **0.881** | Captures multi-hop clauses and provisos with tuned K=5 |
| **LLM Judge** | **Answer Relevancy** | **0.954** | Direct, clear takeaway in plain English without conversational filler |
| **LLM Judge** | **Faithfulness** | **0.910** | Strict adherence to retrieved context with zero hallucinations |
| **LLM Judge** | **Contextual Precision** | **0.839** | Filtered by min_score=0.30; relevant nodes placed at the top |
| **LLM Judge** | **Contextual Recall** | **0.881** | Complete statutory evidence retrieved for 38 of 45 questions |
| **LLM Judge** | **Contextual Relevancy** | **0.365** | Complete 1,200-char statutory sections preserved to prevent legal distortion |

### ⏱️ Retrieval Pipeline Stage Profiling

Benchmarked over 45 statutory queries on **NVIDIA RTX 4050 Laptop GPU (6 GB VRAM) + Intel Core 5 210H + 24 GB RAM**:

| Stage | Hardware | Avg. Latency (ms) | Share (%) |
|---|:---:|:---:|:---:|
| **1. Dense Search** (`Octen-0.6B`) | GPU (RTX 4050) | 315.1 ms | 30.2% |
| **2. BM25 Lexical Search** | CPU / RAM | 57.1 ms | 5.5% |
| **3. RRF Fusion & Deduplication** | CPU | 0.1 ms | < 0.1% |
| **4. Qwen3 Cross-Encoder Reranker** | GPU (RTX 4050) | 671.7 ms | 64.3% |
| **Total Retrieval Pipeline** | **Local Hardware** | **1,044.0 ms** | **100.0%** |

> 🔍 **Engineering Trade-off & Optimization:**
> * **The Trade-off:** The Qwen3 reranker adds **672 ms** (64% of local retrieval), increasing latency from 372 ms to 1,044 ms, but boosts Hit Rate @ 5 by **+45.4%** (from **52.3% to 97.7%**) an essential trade-off where citing the wrong legal section is unacceptable.
> * **Roadmap:** Converting the reranker to **ONNX Runtime with  quantization** is our top priority to compress model size and cut latency under 200 ms.

---

## 🚀 Quickstart Guide

### Option A: Run with Docker Compose (Recommended)

Run both the FastAPI backend and Streamlit frontend in containerized environments:

```bash
# 1. Clone the repository
git clone https://github.com/rajgurubhosale/LegalSathiAI.git
cd LegalSathiAI

# 2. Configure your environment variables (.env)
# Create a .env file with your API keys and database credentials (see template below)

# 3. Build and launch all services
docker compose up --build
```

* 🌐 **Frontend UI**: [http://localhost:8501](http://localhost:8501)
* 📖 **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option B: Local Development Setup (`uv`)

#### 1. Install Dependencies
```bash
# Install dependencies using uv
uv sync
```

#### 2. Configure Environment (`.env`)
Create a `.env` file in the project root:
```dotenv
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://postgres:password@localhost:5432/legalsathi
SECRET_KEY=your_secret_key_hex_at_least_32_characters
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

#### 3. Run Backend & Frontend

**Terminal 1 — Start FastAPI Backend:**
```bash
uv run python -m uvicorn src.backend.main:app --port 8000 --reload
```

**Terminal 2 — Start Streamlit Frontend:**
```bash
uv run streamlit run frontend/app.py --server.port 8501
```
---

## 🔮 Future Scope & Roadmap

- [ ] **More Legal Coverage**: Add State laws, corporate acts (Companies Act, IBC), and Supreme Court / High Court case laws.
- [ ] **ONNX & Model Optimization**: Convert embedding and reranker models to ONNX / INT8 quantization to cut Docker image size and boost CPU inference.
- [ ] **Legal Drafting & Redlining**: Auto-draft agreements and notices, with automated contract risk analysis and clause comparison.
- [ ] **Scanned Documents & OCR**: Integrate multimodal vision models to extract text from mobile camera-clicked PDFs, skewed/shadowed phone scans, physical stamped agreements, and handwritten police complaints (FIRs).
