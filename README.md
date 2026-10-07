# LegalSaathi

A RAG chatbot for Indian legal information, with conversation history and PDF source pages. It retrieves passages from local legal PDFs and generates answers grounded in those passages.

LegalSaathi provides general information. Check important answers against the source documents.

## Pipeline structure

```text
PDFs → extraction → chunking → embeddings → Chroma
Question + history → query rewriting → dense + BM25 → RRF
                  → reranking → filtering → up to 4 chunks → answer + source pages
```

| Component | Setting |
|---|---|
| PDF extraction | Docling |
| Chunking | Page-based recursive splitting: 1200 characters, 100 overlap |
| Embeddings | `Octen/Octen-Embedding-0.6B` |
| Vector database | Chroma (`vectorstore_octen_1200`) |
| Retrieval | Dense 40 + BM25 40 per query; RRF keeps 40 candidates |
| Reranking | `Qwen/Qwen3-Reranker-0.6B`; top 6 per query; cutoff `0.0` |
| Final context | Up to 4 unique chunks, selected across queries |
| Generation | `openai/gpt-oss-120b` through Groq |

If every reranked candidate falls below the cutoff, the top candidate is retained. Scores are raw logits. Selected chunk text is joined with blank lines in selection order; source pages are returned separately to the UI.

## Architecture

![LegalSaathi RAG architecture](Architecture.svg)

The Streamlit frontend calls FastAPI. The backend handles query rewriting, retrieval, reranking and answer generation.

## Evaluation

DeepEval uses `deepseek/deepseek-v4-flash` through MeshAPI. Questions, expected answers and evidence IDs are stored in `evalution_set/final_evalution_set.csv`.

| Metric | Score |
|---|---:|
| Precision, all selected chunks | 0.328 |
| Recall, all selected chunks | 0.867 |
| Hit Rate@5 | 0.977 |
| Recall@5 | 0.867 |
| MRR@5 | 0.841 |
| nDCG@5 | 0.802 |


###  evaluations


| Version | Approach | Answer Relevancy | Contextual Precision | Contextual Recall | Contextual Relevancy | Faithfulness |
|---|---|---:|---:|---:|---:|---:|
| v1 | Original pipeline; no BM25 or neighbour expansion | 0.935 | 0.713 | 0.634 | **0.416** | **0.978** |
| v2 | Granite Small + Nyaya | 0.918 | 0.762 | 0.703 | 0.407 | 0.954 |
| v3 | Granite Small + Nyaya; hybrid retrieval and query rewriting | 0.915 | 0.730 | 0.682 | 0.408 | 0.963 |
| v4 | Granite English + Nyaya; 1500-character chunks; Act name before reranking | 0.952 | 0.835 | 0.819 | 0.370 | 0.935 |
| v5 | Octen + Qwen; 1200/100 chunks; hybrid 35; no cutoff | 0.943 | 0.833 | 0.843 | 0.361 | 0.945 |
| v6 | Octen + Qwen; 1200/100 chunks; hybrid 40; rerank 6; cutoff 0.0 | 0.935 | **0.902** | 0.826 | 0.363 | 0.930 |
| GTE + ±1 | GTE ModernBERT embedding + reranker; 850/100 chunks; hybrid 40; rerank 4; cutoff 0.1; ±1 neighbours | **0.953** | 0.382 | **0.954** | 0.266 | 0.952 |
| **Final setup (current)** | Octen + Qwen reranker + GPT-OSS; reviewed dataset; 1200/100; hybrid 40; rerank 6; cutoff 0.0; up to 4 chunks | 0.940 | 0.826 | 0.883 | 0.385 | 0.948 |

Scores are averaged over available metric results. Retrieval metrics cover 44 questions with labelled evidence. Settings and golden labels differ across experiments.

Run evaluation:

```powershell
uv run python -m src.evalutions.main_evalution_pipeline
```

Set dataset and output paths in [config_file.yaml](src/config/config_file.yaml). Each batch saves metric scores, retrieval scores and diagnostics. Use a new output filename to preserve previous runs.

## Setup

Install Python 3.12 and `uv`, then:

```powershell
git clone https://github.com/rajgurubhosale/LegalSathiAI.git
cd LegalSathiAI
uv sync
```

Add your keys to `.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key
MESH_DEEP_SEEK_FLASH=your_mesh_api_key
```

Put PDFs in `PDF_DATA/` and set paths in [config_file.yaml](src/config/config_file.yaml).

### Build the index

Skip this if your database is already populated.

```powershell
uv run python -m src.data_ingestion.ingest_and_chunk
uv run python -m src.data_ingestion.embed_and_store
```

### Run the app

Complete the backend authentication setup first. Then run these in separate terminals:

```powershell
uv run python -m uvicorn src.backend.main:app --env-file .env --port 8000
```

```powershell
uv run streamlit run frontend/app.py
```

Open [localhost:8501](http://localhost:8501).

### Docker

Once authentication and Docker settings are configured:

```powershell
docker compose up --build
```
