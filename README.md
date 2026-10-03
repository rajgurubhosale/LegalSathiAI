# LegalSaathi

An Indian legal information assistant built with Retrieval-Augmented Generation (RAG). LegalSaathi retrieves evidence from supplied legal PDFs and uses it to answer questions.

## Scope

The current local corpus includes criminal law, evidence, consumer protection, RTI, RERA, labour laws, information technology and other Indian Acts.

Coverage depends on the PDFs available in the configured corpus. The assistant is designed to:

- Explain legal provisions in clear language.
- Answer multiple legal issues within a question.
- Preserve relevant conditions and exceptions.
- State when the retrieved evidence is insufficient.
- Avoid guaranteeing outcomes or inventing compensation amounts.

LegalSaathi provides general legal information, not a substitute for advice from a qualified advocate. Answers may be incomplete or incorrect and should be checked against the source documents.

## How it works

```text
Question
   ↓
Query decomposition and focused rewriting
   ↓
Dense retrieval + BM25
   ↓
Reciprocal Rank Fusion (RRF)
   ↓
Reranking with Act names
   ↓
Minimum-score filtering
   ↓
Deduplication and balanced context selection
   ↓
Answer generation
```

Each search query is retrieved and reranked separately. Final context selection takes candidates from the queries in turns and keeps up to four unique chunks.

Act names are added to the reranker's input. Generation and evaluation receive the original chunk text.

## Current working baseline — v6

| Component | Setting |
|---|---|
| PDF extraction | Docling |
| Chunking | Page-based recursive splitting |
| Chunk size | 1200 characters |
| Chunk overlap | 100 characters |
| Embedding | `Octen/Octen-Embedding-0.6B` |
| Vector database | Chroma |
| Retrieval | Dense 40 + BM25 40 per search query |
| Fusion | RRF, retaining 40 candidates |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` |
| Ranked candidates | Top 6 per search query |
| Score cutoff | Raw `min_score=0.0` |
| Final context | Up to 4 unique chunks |
| Generator | `openai/gpt-oss-120b` through Groq |
| Evaluation | DeepEval |
| Evaluation judge | `deepseek/deepseek-v4-flash` through Mesh |

Scores below the cutoff are removed. If every ranked candidate for a query is below the cutoff, its top candidate is retained.

Reranker scores are raw logits, not percentages. Setting `min_score` to `null` disables filtering.

Automatic previous/next expansion is not part of this baseline.

## Evaluation history

Higher scores are better. Bold values are the highest reported score for each metric.

| Version | Approach | Answer Relevancy | Contextual Precision | Contextual Recall | Contextual Relevancy | Faithfulness |
|---|---|---:|---:|---:|---:|---:|
| v1 | Original pipeline; no BM25 or neighbour expansion | 0.935 | 0.713 | 0.634 | **0.416** | **0.978** |
| v2 | Granite Small + Nyaya | 0.918 | 0.762 | 0.703 | 0.407 | 0.954 |
| v3 | Granite Small + Nyaya; hybrid retrieval and query rewriting | 0.915 | 0.730 | 0.682 | 0.408 | 0.963 |
| v4 | Granite English + Nyaya; 1500-character chunks; Act name before reranking | **0.952** | 0.835 | 0.819 | 0.370 | 0.935 |
| v5 | Octen + Qwen; 1200/100 chunks; hybrid 35; no cutoff | 0.943 | 0.833 | **0.843** | 0.361 | 0.945 |
| v6 | Octen + Qwen; 1200/100 chunks; hybrid 40; rerank 6; cutoff 0.0 | 0.935 | **0.902** | 0.826 | 0.363 | 0.930 |

The v6 run used a 45-question golden dataset and recorded **1 metric error**. Failed metric results do not contribute valid scores to the averages.

These are experiment results, not guarantees of legal accuracy. Several settings changed between versions, so differences cannot always be attributed to one component.

## Project structure

```text
src/
├── config/
│   └── config_file.yaml
├── data_ingestion/
│   ├── ingest_and_chunk.py
│   └── chunking_experiment.py
├── rag_retrieval/
│   └── retrival.py
├── genration_pipeline/
│   ├── pipeline.py
│   ├── model.py
│   ├── new_prompts.py
│   └── schemas.py
├── evalutions/
│   └── main_evalution_pipeline.py
├── utils/
│   └── main_utils.py
├── logger/
└── exception/
```

`chunking_experiment.py` is a separate section-aware chunking experiment. It is not used by the v6 baseline.

## Setup

### 1. Requirements

- Python 3.12
- `uv`
- Source PDFs
- A Groq API key for query rewriting and answer generation
- A Mesh API key for evaluation

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

The project configures PyTorch through a CUDA 13.0 wheel index. Adjust that dependency configuration if your environment needs a different build.

### 2. Clone and install

```powershell
git clone https://github.com/rajgurubhosale/LegalSathiAI.git
cd LegalSathiAI
uv sync
```

### 3. Configure environment variables

Create `.env` in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
MESH_DEEP_SEEK_FLASH=your_mesh_api_key
```

Keep `.env` out of version control.

### 4. Configure local paths

The implementation currently contains Windows paths under:

```text
D:\LegalSaathi AI
```

If your checkout is elsewhere, update the paths in:

- `src/config/config_file.yaml`
- `src/utils/main_utils.py`
- `src/rag_retrieval/retrival.py`
- `src/data_ingestion/ingest_and_chunk.py`

Place your source PDFs in the configured `PDF_DATA` directory.

PDFs, vector stores, evaluation datasets and result files are local artifacts and are not included in the repository. A fresh checkout requires these inputs and a newly built vector store.

## Build the vector store

Run from the project root:

```powershell
uv run -m src.data_ingestion.ingest_and_chunk
```

The ingestion pipeline extracts PDF text, creates chunks, embeds them and stores them in the configured Chroma collection.

Use the same embedding model for ingestion and retrieval. Changing the embedding model requires a new collection and re-embedding the documents, even when the vector dimensions match.

Use separate collections for chunking and embedding experiments to preserve the baseline.

## Ask a question

Run this in a notebook or Python script using the project environment:

```python
from src.genration_pipeline.pipeline import LegalSaathiPipeline

pipeline = LegalSaathiPipeline(
    top_n=40,
    rerank_k=6,
    min_score=0.0,
)

result = pipeline.run(
    "What remedies are available under RERA when a builder delays possession?"
)

print(result["answer"])
```

Reuse the pipeline instance for subsequent questions to avoid repeatedly loading the models.

## Run evaluation

### Dataset

Configure `evaluation.dataset_path` in `src/config/config_file.yaml`.

The current evaluator requires these CSV columns:

| Column | Content |
|---|---|
| `question` | User question |
| `updated_golden_answer` | Verified expected answer |

The golden dataset may also contain `id` and `references` for source verification. References should use exact PDF filenames and 1-based physical PDF page numbers.

### Command

```powershell
uv run -m src.evalutions.main_evalution_pipeline
```

The evaluator:

- Generates answers in batches of two.
- Passes individual context chunks to DeepEval.
- Measures five evaluation metrics.
- Saves each completed batch to CSV.
- Prints metric averages and the number of metric errors.

The first batch overwrites the configured output CSV. Later batches append to it. Use a different `evaluation.output_path` for each experiment.

`evaluation.threshold` is the DeepEval pass threshold. It is separate from the reranker’s `evaluation.min_score`.

## Current limitations

- Fixed-size chunks can split provisions or include unrelated provisions.
- Retrieval and reranking can select the wrong Act or miss essential evidence.
- Page metadata is inherited from extracted page documents and may be broader than an individual chunk's text.
- The best-scoring chunk is retained even when its score is below the cutoff.
- Model-generated evaluation scores require inspection alongside source evidence.
- Laws, amendments and commencement status are only represented when present in the supplied corpus.
- The application currently uses local Windows paths rather than portable path configuration.

## Roadmap

- [ ] Improve Contextual Relevancy while preserving Recall and Faithfulness.
- [ ] Compare chunking strategies in separate collections.
- [ ] Improve provision completeness and page provenance.
- [ ] Expand and review the golden dataset.
- [ ] Resolve metric errors and inspect individual question failures.
- [ ] Make paths and configuration portable.
- [ ] Add a maintained application interface.