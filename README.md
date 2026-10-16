# ⚖️ LegalSaathi AI — Indian Legal Intelligence Platform

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
</p>

> ⚠️ **Disclaimer**: For educational and research purposes only. Does not constitute formal legal advice. Always cross-check against official gazettes.

---

## 📌 Overview

**LegalSaathi AI** is a production-grade legal research and decision-support system grounded in Indian statutory law. It combines hybrid dense-sparse retrieval, cross-encoder reranking, and citation-backed LLM generation to answer complex legal questions without hallucination.

### Key Capabilities
* **Dual-Channel Hybrid Search**: Fuses semantic embeddings (`Octen-0.6B`) with keyword search (`BM25`) via Reciprocal Rank Fusion (`RRF`).
* **Cross-Encoder Reranking**: Uses `Qwen3-Reranker-0.6B` with dynamic logit filtering to select the top 4 most authoritative statutory passages.
* **Strict Statutory Grounding**: Powered by `GPT-OSS-120B` (via Groq) to synthesize answers with exact statutory sections and PDF page citations.
* **Secure Multi-User Backend**: FastAPI backend with PostgreSQL, Argon2 password hashing, and JWT session authentication.

---

## 🏗️ Architecture Flow

```text
User Query → Sub-Query Decomposition → Dense (Octen) + Sparse (BM25) → RRF Fusion (Top 40)
           → Qwen3-Reranker (Top 4) → Groq (GPT-OSS-120B) → Streamed Answer + Citations
```

---

## 📊 Production Benchmarks

Evaluated with **DeepEval** (`deepseek-v4-flash` judge) on a test suite of 44 statutory queries ([`final_evalution_set.csv`](src/evalutions/final_evalution_set.csv)).

| Category | Metric | Score | Target | Focus |
|---|---|:---:|:---:|---|
| **Retrieval** | Hit Rate @ 5 | **0.977** | > 0.90 | 43 of 44 queries retrieve exact statutory evidence |
| **Retrieval** | Recall @ 5 | **0.867** | > 0.80 | High coverage of required legal provisions |
| **Retrieval** | MRR @ 5 | **0.841** | > 0.75 | Key clause positioned at rank 1 or 2 on average |
| **LLM Judge** | Answer Relevancy | **0.940** | > 0.85 | Comprehensive and intent-aligned legal answer |
| **LLM Judge** | Contextual Precision | **0.826** | > 0.80 | Clean, noise-free statutory context in prompt |
| **LLM Judge** | Contextual Recall | **0.883** | > 0.80 | All required legal nuances captured in context |
| **LLM Judge** | Faithfulness | **0.948** | > 0.90 | Strict adherence to text with zero hallucinations |

> ℹ️ *For detailed comparative experiments (E1 to E4), refer to the [Evaluation Benchmark Report](src/evalutions/eval.md).*

---

## 📚 Indexed Legal Statutes (Document Vault)

The platform indexes **22 core Indian statutes**, including:
* **Criminal Law**: *Bharatiya Nyaya Sanhita (BNS) 2023*, *BNSS 2023*, *Bharatiya Sakshya Adhiniyam (BSA) 2023*
* **Constitutional & Rights**: *Constitution of India*, *Right to Information (RTI) Act 2005*
* **Commercial & Property**: *Indian Contract Act 1872*, *RERA 2016*, *Consumer Protection Act 2019*
* **Cyber & Data Privacy**: *DPDP Act 2023*, *Information Technology Act 2000*, *Telecommunications Act 2023*
* **Labour & Workplace**: *Code on Wages 2019*, *Code on Social Security 2020*, *Industrial Relations Code 2020*, *OSH Code 2020*, *POSH Act 2013*
* **Protection & Public Welfare**: *Domestic Violence Act 2005*, *Dowry Prohibition Act 1961*, *Child Labour Act 1986*, *Motor Vehicles Act 1988*

---

## 🚀 Quickstart

### 1. Installation
```powershell
# Clone the repository
git clone https://github.com/rajgurubhosale/LegalSathiAI.git
cd LegalSathiAI

# Install dependencies using uv
uv sync
```

### 2. Environment Setup
Create a `.env` file in the root directory:
```dotenv
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://postgres:password@localhost:5432/legalsaathi
SECRET_KEY=your_secret_key_hex
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 3. Launch Services

**Backend (FastAPI):**
```powershell
uv run python -m uvicorn src.backend.main:app --port 8000 --reload
```

**Frontend (Streamlit):**
```powershell
uv run streamlit run frontend/app.py --server.port 8501
```

* API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
* Web Interface: [http://localhost:8501](http://localhost:8501)

---

## 🔮 Future Scope

* **Dual-Stream Grounding**: Enable users to upload private contracts and notices for comparative analysis against applicable statutes.
* **Statute Intent Router**: Zero-cost query classification to bypass retrieval for non-statutory queries or direct legal definitions.
* **Hierarchical AST Chunking**: Transition from fixed character chunks to legal syntax parsing (`Act` → `Chapter` → `Section` → `Proviso`).
* **Production Observability**: Integrate Langfuse/Arize Phoenix for real-time latency, token usage, and retrieval tracing.
