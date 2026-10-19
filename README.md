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

### Key Capabilities
1. **Agentic Tool Calling**: Uses a LangGraph agent that decides when to search the law, handles multi-turn reasoning, and knows when simple conversation doesn't need retrieval.
2. **Dual-Retrieval (Statutes + Personal Vault)**: In a single prompt, you can compare 22 central Indian statutes with your own uploaded contracts, notices, or agreements (PDF/DOCX).
3. **Hybrid Search + Re-ranking**: Blends keyword matching (`BM25`) with deep semantic embeddings (`Octen-0.6B`) via Reciprocal Rank Fusion (`RRF`), followed by precision re-ranking with `Qwen3-Reranker-0.6B`.
4. **Smart Conversational Memory**: Remembers your chat across sessions using PostgreSQL checkpointing, while automatic summarization middleware keeps long conversations lightning-fast and token-efficient.
5. **Real-time SSE Streaming**: Answers stream in token-by-token alongside exact statutory section and PDF page citations.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph OFF["1. Statutory ingestion (offline)"]
        A1["22 Indian Acts (PDF)"] --> A2["Docling parsing + chunking"]
        A2 --> A3["Octen-0.6B embeddings"]
        A3 --> A4[("Statutory ChromaDB")]
    end

    subgraph UP["2. Personal vault ingestion (online)"]
        U1["User upload (PDF / DOCX)"] --> U2["PyMuPDF / python-docx + chunking"]
        U2 --> U3["Octen-0.6B embeddings"]
        U3 --> U4[("Vault ChromaDB<br/>filtered by user_id")]
    end

    subgraph RT["3. Agentic RAG (runtime)"]
        Q(["User query<br/>Streamlit + FastAPI"]) --> AG["LangGraph agent<br/>Groq LLM + tool calling"]
        AG <-->|"long chats"| SM["Summarization middleware"]
        AG -->|"legal / document question"| T["search_legal_documents"]
        AG -->|"greeting / chat history"| OUT

        T --> W1["Statutory hybrid search<br/>dense + BM25, RRF"]
        T -->|"user has documents"| W2["Vault search<br/>user_id from JWT"]
        W1 --> R["Qwen3-Reranker-0.6B<br/>one pass over combined pool"]
        W2 --> R
        R -->|"top chunks + sources"| AG
        AG --> OUT(["Streamed answer + citations"])
    end

    A4 -.-> W1
    U4 -.-> W2
```

### ⚡ End-to-End Query Lifecycle
1. **Query & Agent State**: User query arrives from Streamlit → FastAPI loads conversation thread via `PostgresSaver` → Summarizer compresses history if > 12 messages.
2. **Parallel Tool Retrieval**: Agent invokes `search_legal_documents` with 2 concurrent workers:
   - **Worker 1**: Executes Statutory Hybrid Search (ChromaDB dense vectors + in-memory BM25 lexical search fused via RRF).
   - **Worker 2**: Conditionally executes Personal Vault Search in ChromaDB (only if user documents exist).
3. **Candidate Merging & Reranking**: Documents from both streams are merged (`docs = main_docs + vault_docs`) and fed into `Qwen3-Reranker-0.6B` to filter the Top 4 most authoritative passages.
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

Benchmarked using **DeepEval** (`deepseek-v4-flash` judge, `gpt-oss-120b` generator) on 44 curated statutory test cases:

| Category | Metric | Score | Description |
|---|---|:---:|---|
| **Retrieval** | **Hit Rate @ 5** | **0.977** | 43 of 44 queries retrieve exact statutory evidence in top 5 chunks |
| **Retrieval** | **Recall @ 5** | **0.867** | High coverage of required statutory provisions |
| **Retrieval** | **MRR @ 5** | **0.841** | Key authoritative clause positioned at rank 1 or 2 on average |
| **Retrieval** | **nDCG @ 5** | **0.802** | High ranking quality accounting for graded statutory relevance |
| **Retrieval** | **Precision @ 5** | **0.328** | Proportion of selected chunks directly matching ground-truth clauses |
| **LLM Judge** | **Answer Relevancy** | **0.940** | Comprehensive, accurate, and intent-aligned legal answer |
| **LLM Judge** | **Contextual Precision** | **0.826** | Clean, noise-free statutory context in prompt |
| **LLM Judge** | **Contextual Recall** | **0.883** | All required legal nuances and provisions captured in context |
| **LLM Judge** | **Contextual Relevancy** | **0.385** | Ratio of retrieved statutory sentences directly relevant to query |
| **LLM Judge** | **Faithfulness** | **0.948** | Strict adherence to text with zero hallucinations |

---

## 🚀 Quickstart Guide

### 1. Clone & Install
```powershell
# Clone the repository
git clone https://github.com/rajgurubhosale/LegalSathiAI.git
cd LegalSathiAI

# Install dependencies using uv
uv sync
```

### 2. Configure Environment
Create a `.env` file in the project root:
```dotenv
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://postgres:password@localhost:5432/legalsaathi
SECRET_KEY=your_secret_key_hex
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 3. Start PostgreSQL
Ensure your PostgreSQL instance is running:
```powershell
docker-compose up -d db
```

### 4. Run the Application

**Start the FastAPI Backend:**
```powershell
uv run python -m uvicorn src.backend.main:app --port 8000 --reload
```

**Start the Streamlit Web UI:**
```powershell
uv run streamlit run frontend/app.py --server.port 8501
```

* 🌐 **Frontend UI**: [http://localhost:8501](http://localhost:8501)
* 📖 **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔮 Future Scope & Roadmap

- [ ] **More Legal Coverage**: Add State laws, corporate acts (Companies Act, IBC), and Supreme Court / High Court case laws.
- [ ] **ONNX & Model Optimization**: Convert embedding and reranker models to ONNX / INT8 quantization to cut Docker image size and boost CPU inference.
- [ ] **Legal Drafting & Redlining**: Auto-draft agreements and notices, with automated contract risk analysis and clause comparison.
- [ ] **Scanned Documents & OCR**: Add vision models to process stamped papers, handwritten FIRs, and physical court documents.
- [ ] **Legal Citation Graph**: Build a reference graph linking related legal sections and judicial precedents.
