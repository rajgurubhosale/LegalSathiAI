# ⚖️ LegalSaathi AI — Grounded Indian Legal Intelligence Platform

<p align="center">
  <img src="Architecture.svg" alt="LegalSaathi AI Architecture" width="850">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=flat-square" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Reranker-Qwen3--0.6B-6366F1?style=flat-square" alt="Qwen3 Reranker">
  <img src="https://img.shields.io/badge/LLM-GPT--OSS--120B_via_Groq-F55036?style=flat-square" alt="Groq">
  <img src="https://img.shields.io/badge/Evaluation-DeepEval-10B981?style=flat-square" alt="DeepEval">
  <img src="https://img.shields.io/badge/Package_Manager-uv-blueviolet?style=flat-square" alt="uv">
</p>

> ⚠️ **Legal Disclaimer**: LegalSaathi AI is an informational research assistant for educational purposes only. It does not provide formal legal advice or substitute an advocate licensed by the Bar Council of India. Always verify statutory provisions against official gazettes before taking legal action.

---

## 📌 Executive Summary

**LegalSaathi AI** is an end-to-end legal decision-support system engineered to bridge the accessibility gap in Indian jurisprudence:

* **Dual-Channel Hybrid Retrieval**: Combines semantic dense embeddings (`Octen-0.6B`) with keyword-exact sparse search (`BM25`) fused via Reciprocal Rank Fusion (`RRF`).
* **Cross-Encoder Reranking**: Uses `Qwen3-Reranker-0.6B` with dynamic logit filtering to score and retain only high-confidence statutory passages.
* **Strict Statutory Grounding**: Generates hallucination-resistant answers strictly grounded in authoritative Indian Acts (*Bharatiya Nyaya Sanhita 2023*, *Constitution of India*, *RERA*, *IT Act*, *DPDP Act*, *Labour Codes*, etc.).
* **Verifiable Source Attribution**: Every response returns exact statutory section references and source PDF page numbers, allowing direct verification in the built-in Document Vault reader.
* **Production Authentication & Security**: Complete user isolation with PostgreSQL storage, Argon2 password hashing, and 30-minute sliding JWT session tokens.

---

## 🏗️ Multi-Stage Retrieval & Generation Pipeline

```text
1. Ingestion Pipeline (Offline Indexing)
   Legal PDFs → Docling Parsing → Recursive Chunking (1200 chars / 100 overlap) → Octen-0.6B Embeddings → ChromaDB

2. Retrieval & Generation Pipeline (Online Inference)
   User Query + History → Sub-Query Rewriting → Dense (40) + BM25 (40) → RRF Fusion (40)
                        → Qwen3-Reranker (Cutoff >= 0.0) → Top 4 Chunks → Groq (GPT-OSS-120B) → Streamed Answer + Sources
```

### ⚙️ Pipeline Specifications

| Pipeline Stage | Technology / Model | Configuration Details |
|---|---|---|
| **Document Parsing** | `Docling` | High-fidelity extraction of legal tables, sections, and statutory hierarchies |
| **Chunking Strategy** | Page-Aware Recursive Splitting | `chunk_size: 1200`, `chunk_overlap: 100`, preserving page metadata |
| **Dense Embeddings** | `Octen/Octen-Embedding-0.6B` | High-dimensional semantic representation fine-tuned for structured text |
| **Vector Index** | `ChromaDB` (`vectorstore_octen_1200`) | HNSW cosine similarity search |
| **Lexical Search** | `BM25 (Rank-BM25)` | Exact match token indexing for specific Act names, section numbers, and legal keywords |
| **Ensemble Fusion** | Reciprocal Rank Fusion (`RRF`) | Merges Dense (40) + BM25 (40) into 40 fused candidates per query |
| **Cross-Encoder Reranking** | `Qwen/Qwen3-Reranker-0.6B` | Evaluates full query-passage cross-attention; top 6 per sub-query; logit cutoff `0.0` |
| **Context Assembler** | Deduplicated Balance Selector | Selects up to 4 unique high-confidence chunks across sub-queries |
| **LLM Inference** | `openai/gpt-oss-120b` via Groq | Low-latency token streaming, zero-temperature statutory grounding |
| **Authentication** | PostgreSQL + Argon2 + JWT | Stateless bearer token authentication (30-min sliding expiry) |

---

## 📊 Empirical Evaluation & Benchmarks

The pipeline is benchmarked using **DeepEval** with `deepseek/deepseek-v4-flash` as the judge LLM over a verified dataset of 44 complex statutory queries (`evalution_set/final_evalution_set.csv`).

### 🎯 Current Retrieval Metrics (All Selected Chunks)

| Metric | Score | Interpretation |
|---|---:|---|
| **Hit Rate @ 5** | **0.977** | In 97.7% of queries, ground-truth legal passage appears in top-5 reranked candidates |
| **Recall @ 5** | **0.867** | Captures 86.7% of all relevant statutory evidence within top 5 passages |
| **MRR @ 5** (Mean Reciprocal Rank) | **0.841** | The most critical statutory clause is positioned at rank 1 or 2 on average |
| **nDCG @ 5** | **0.802** | High ranking quality weighted by statutory relevance |
| **Precision (Selected Chunks)** | **0.328** | Deliberately balanced to maximize recall before LLM synthesis |

### 📈 Progression Across Experimental Iterations

| Version | Configuration / Retrieval Strategy | Answer Relevancy | Context Precision | Context Recall | Context Relevancy | Faithfulness |
|---|---|---:|---:|---:|---:|---:|
| **v1** | Baseline: Dense only, no BM25, no sub-query rewriting | 0.935 | 0.713 | 0.634 | **0.416** | **0.978** |
| **v2** | Granite Small + Nyaya Reranker | 0.918 | 0.762 | 0.703 | 0.407 | 0.954 |
| **v3** | Granite Small + Nyaya; Hybrid BM25 + Query Rewriting | 0.915 | 0.730 | 0.682 | 0.408 | 0.963 |
| **v4** | Granite English + Nyaya; 1500 chars; Act Title Injection | 0.952 | 0.835 | 0.819 | 0.370 | 0.935 |
| **v5** | Octen (0.6B) + Qwen3 Reranker; 1200/100 chunks; Hybrid 35 | 0.943 | 0.833 | 0.843 | 0.361 | 0.945 |
| **GTE+±1** | GTE ModernBERT + Neighbor expansion (±1 page chunks) | **0.953** | 0.382 | **0.954** | 0.266 | 0.952 |
| **Production Setup** | **Octen + Qwen3 Reranker + GPT-OSS; Hybrid 40; Top-4 Deduplicated** | **0.940** | **0.826** | **0.883** | **0.385** | **0.948** |

---

## 📚 Document Vault Knowledge Base

The platform features an indexed library of **22 foundational statutes** categorized for quick discovery:

| Category | Acts & Statutes Included |
|---|---|
| **Criminal Law & Procedure** | `Bharatiya Nyaya Sanhita (BNS), 2023`, `Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023`, `Bharatiya Sakshya Adhiniyam (BSA), 2023` |
| **Constitutional Law** | `The Constitution of India (Complete Diglot)` |
| **Commercial & Contracts** | `The Indian Contract Act, 1872`, `The Real Estate (Regulation and Development) Act (RERA), 2016` |
| **Cyber & Emerging Tech** | `The Digital Personal Data Protection Act (DPDP), 2023`, `The Information Technology Act, 2000`, `The Telecommunications Act, 2023` |
| **Consumer & Public Rights** | `The Consumer Protection Act, 2019`, `Right to Information Act (RTI), 2005` |
| **Women & Child Protection** | `Protection of Women from Domestic Violence Act, 2005`, `POSH Act, 2013`, `Dowry Prohibition Act, 1961`, `Child & Adolescent Labour Act, 1986` |
| **Labour & Employment Codes** | `Code on Wages, 2019`, `Code on Social Security, 2020`, `Industrial Relations Code, 2020`, `Occupational Safety & Health (OSH) Code, 2020` |
| **Public Interest & Media** | `The Legal Services Authorities Act, 1987 (NALSA/Lok Adalats)`, `The Motor Vehicles Act, 1988`, `The Press Council Act, 1978` |

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (Fast Python package manager)
- **PostgreSQL 14+** (running locally or via container)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/rajgurubhosale/LegalSathiAI.git
cd LegalSathiAI

# Sync dependencies using uv
uv sync
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```dotenv
# LLM & Reranking Keys
GROQ_API_KEY=your_groq_api_key_here
MESH_DEEP_SEEK_FLASH=your_mesh_api_key_here  # Required only for running evaluations

# JWT Authentication
SECRET_KEY=generate_with_python_secrets_token_hex_32
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# PostgreSQL Database Connection
DATABASE_URL=postgresql://postgres:password@localhost:5432/legalsaathi
```

Generate a secure `SECRET_KEY`:
```powershell
uv run python -c "import secrets; print(secrets.token_hex(32))"
```

### 4. Database Initialization
Ensure your PostgreSQL database exists, then execute the initial schema:
```sql
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chats (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### 5. Index Building (Optional if ChromaDB is pre-populated)
```powershell
uv run python -m src.data_ingestion.ingest_and_chunk
uv run python -m src.data_ingestion.embed_and_store
```

### 6. Launch Applications
Run the FastAPI backend and Streamlit frontend in separate terminals:

**Backend:**
```powershell
uv run python -m uvicorn src.backend.main:app --env-file .env --port 8000 --reload
```

**Frontend:**
```powershell
uv run streamlit run frontend/app.py --server.port 8501
```

Access the UI at [http://localhost:8501](http://localhost:8501) and Swagger docs at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🛣️ Engineering Roadmap

* **Two-Stage RAG Pipeline with Statute Router**: Skip retrieval if the query doesn't require it (chitchat, greetings, or out-of-scope) via zero-cost routing inside query decomposition.
* **Dual-Stream Grounding (User Uploads + Statutory Law)**: Allow users to upload their own case files, contracts, or notices to get answers grounded in their uploaded document combined with applicable Indian statutes.
* **Structural Hierarchical Legal Chunking**: Transition from character splitting to legal AST parsing (`Chapter` → `Section` → `Sub-section` → `Proviso` → `Explanation`).
* **Auto-Cascade Ephemeral Document Deletion on Chat Removal**: Automatically purge uploaded user files and temporary vector index records whenever a chat thread is deleted.
* **Real-Time IST Timezone Synchronization**: Native IST timestamps across session clocks, audit logs, and message history.
* **Production Tracing & Observability**: Integrate Langfuse / Arize Phoenix for latency and retrieval stage tracing.
---
