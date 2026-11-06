"""
Baseline Latency Benchmark for LegalSaathi AI (GPU Accelerated)
---------------------------------------------------------------
Measures end-to-end and component latencies across ground-truth legal queries
from final_evalution_set.csv using GPU acceleration (CUDA).

Metrics Measured:
  1. Router (decision_node via Groq API)
  2. ChromaDB (Dense Vector Search)
  3. BM25 (Sparse Keyword Search)
  4. Qwen3 Cross-Encoder Reranker (GPU inference via CUDA)
  5. Time-to-First-Token (TTFT via streaming generation)
  6. Total End-to-End Time (Question to complete streamed answer)
  7. Peak RAM (RSS) & Peak GPU VRAM (CUDA Allocated)
"""

import os
import sys
import time
import argparse
import asyncio
import numpy as np
import pandas as pd
import psutil
import torch

from src.genration_pipeline.pipeline import LegalSaathiPipeline
from src.genration_pipeline.prompts import DECISION_MODEL_PROMPT, RAG_SYSTEM_PROMPT
from src.genration_pipeline.model import router


def format_ms(val_ms: float) -> str:
    """Format milliseconds cleanly with thousand separators and seconds."""
    if val_ms >= 1000:
        return f"{val_ms:,.0f} ms ({val_ms / 1000:.2f} s)"
    return f"{val_ms:.1f} ms"


def print_table(metrics_summary: list, peak_ram_mb: float, peak_vram_mb: float | None, num_queries: int, device_name: str):
    """Print clean ASCII summary table of benchmark results."""
    col1_w = 34
    col2_w = 20
    col3_w = 20
    col4_w = 20
    total_w = col1_w + col2_w + col3_w + col4_w + 3

    print("\n" + "=" * total_w)
    title = f"LATENCY BENCHMARK BASELINE SUMMARY ({num_queries} QUERIES)"
    print(f"{title:^{total_w}}")
    print(f"{f'Device: {device_name}':^{total_w}}")
    print("=" * total_w)
    header = f"{'Metric':<{col1_w}} | {'Mean (Avg)':<{col2_w}} | {'P50 (Median)':<{col3_w}} | {'P95 (Tail)':<{col4_w}}"
    print(header)
    print("-" * total_w)

    for item in metrics_summary:
        name = item["name"]
        avg_str = format_ms(item["avg"])
        p50_str = format_ms(item["p50"])
        p95_str = format_ms(item["p95"])
        print(f"{name:<{col1_w}} | {avg_str:<{col2_w}} | {p50_str:<{col3_w}} | {p95_str:<{col4_w}}")

    print("-" * total_w)
    ram_line = f"Peak System Memory (RAM): {peak_ram_mb:,.1f} MB"
    if peak_vram_mb is not None:
        ram_line += f" | Peak GPU VRAM: {peak_vram_mb:,.1f} MB"
    print(f"{ram_line:<{total_w}}")
    print("=" * total_w + "\n")


async def run_benchmark(
    num_queries: int = 10,
    top_n: int = None,
    dataset_path: str = "src/evalutions/final_evalution_set.csv",
    save_csv: str = None
):
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset not found at: {dataset_path}")
        return

    use_cuda = torch.cuda.is_available()
    device_name = f"CUDA ({torch.cuda.get_device_name(0)})" if use_cuda else "CPU"

    print("=" * 80)
    print(" LegalSaathi AI - Latency Benchmark Baseline (GPU Accelerated)")
    print("=" * 80)
    print(f"Device: {device_name}")
    print(f"Loading queries from: {dataset_path}")

    df = pd.read_csv(dataset_path)
    queries = df["question"].dropna().head(num_queries).tolist()
    print(f"Selected {len(queries)} evaluation questions.\n")

    # Initialize RAG Pipeline (Loads Embeddings & Qwen3 Reranker onto GPU)
    print("[1/3] Initializing LegalSaathiPipeline on GPU...")
    pipeline = LegalSaathiPipeline(vault=None, top_n=top_n)
    print(f"Pipeline initialized (top_n = {pipeline.top_n}, rerank_k = {pipeline.rerank_k})")

    # Warmup pass: Preload models and HTTP connections
    print("[2/3] Performing Warmup pass (eliminates cold-start import overhead)...")
    try:
        pipeline.warmup()
        # Warmup router and generator
        await router.ainvoke([("system", DECISION_MODEL_PROMPT), ("human", "ping")])
        async for _ in pipeline.model.astream([("system", "Reply 'ok'."), ("human", "ping")]):
            pass
        print("Warmup complete!\n")
    except Exception as e:
        print(f"Warmup notice: {e}\n")

    # Metric accumulators
    m_router = []
    m_dense = []
    m_sparse = []
    m_rerank = []
    m_ttft = []
    m_total = []
    records = []

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    print(f"[3/3] Running {len(queries)} Benchmark Queries (Zero-Duplicate Single Pass)...")
    print("-" * 80)

    for i, q in enumerate(queries, start=1):
        clean_q = (q[:70] + "...") if len(q) > 73 else q
        print(f"\n[Query {i}/{len(queries)}] \"{clean_q}\"")

        # Start of end-to-end timing for this query
        t_query_start = time.perf_counter()

        # -------------------------------------------------------------
        # 1. Router Latency (Groq API structured routing decision)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        decision = await router.ainvoke([
            ("system", DECISION_MODEL_PROMPT),
            ("human", q),
        ])
        t_router = (time.perf_counter() - t0) * 1000
        m_router.append(t_router)
        print(f"  ├─ 1. Router ({decision.destination}): {t_router:.1f} ms")

        # -------------------------------------------------------------
        # 2. Dense Search (ChromaDB Vector Store)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        dense_docs = await asyncio.to_thread(pipeline.retriever.retrieve_invoke, q)
        t_dense = (time.perf_counter() - t0) * 1000
        m_dense.append(t_dense)
        print(f"  ├─ 2. ChromaDB Dense Search: {t_dense:.1f} ms")

        # -------------------------------------------------------------
        # 3. Sparse Search (BM25 Keyword Matching)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        sparse_docs = await asyncio.to_thread(pipeline.retriever.bm25_retriever.invoke, q)
        t_sparse = (time.perf_counter() - t0) * 1000
        m_sparse.append(t_sparse)
        print(f"  ├─ 3. BM25 Sparse Search: {t_sparse:.1f} ms")

        # -------------------------------------------------------------
        # 4. Hybrid Reciprocal Rank Fusion + Qwen3 Reranker on GPU
        # Combines the already-retrieved dense & sparse docs directly
        # -------------------------------------------------------------
        candidate_docs = pipeline.retriever.hybrid_retriever.weighted_reciprocal_rank(
            [dense_docs, sparse_docs]
        )[:pipeline.top_n]
        num_candidates = len(candidate_docs)

        t0 = time.perf_counter()
        ranked = await asyncio.to_thread(pipeline.reranker.rerank, docs=candidate_docs, query=q)
        t_rerank = (time.perf_counter() - t0) * 1000
        m_rerank.append(t_rerank)
        print(f"  ├─ 4. Qwen3 Reranker ({num_candidates} candidates on GPU): {format_ms(t_rerank)}")

        # -------------------------------------------------------------
        # 5 & 6. LLM Stream: True TTFT and Total End-to-End Latency
        # Uses the ranked docs directly without re-running retrieval!
        # -------------------------------------------------------------
        top_docs = [doc for _, doc in ranked[:pipeline.max_chunks]]
        chunks = []
        for d in top_docs:
            m = d.metadata
            name = m.get("doc_name") or m.get("act_name", "Statute")
            try:
                pages = [m.get("page_no", 1)] if "doc_name" in m else pipeline.document_pages(d)
            except Exception:
                pages = [m.get("page", m.get("page_no", 1))]
            kind = "Your document" if "doc_name" in m else "Act"
            chunks.append(f"SOURCE: {kind} | {name} | PDF page {pages[0]}\n{d.page_content}")
        context = "\n\n".join(chunks)

        system_prompt = f"{RAG_SYSTEM_PROMPT}\n\nRELEVANT LEGAL SOURCES:\n{context}"

        first_token_time = None
        token_count = 0

        async for chunk in pipeline.model.astream([
            ("system", system_prompt),
            ("human", q)
        ]):
            if chunk.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1

        t_query_end = time.perf_counter()
        t_ttft = ((first_token_time - t_query_start) * 1000) if first_token_time else ((t_query_end - t_query_start) * 1000)
        t_total = (t_query_end - t_query_start) * 1000

        m_ttft.append(t_ttft)
        m_total.append(t_total)
        print(f"  ├─ 5. TTFT (First Token): {format_ms(t_ttft)}")
        print(f"  └─ 6. Total End-to-End Time: {format_ms(t_total)} ({token_count} tokens streamed)")

        records.append({
            "query_idx": i,
            "question": q,
            "router_ms": t_router,
            "dense_ms": t_dense,
            "sparse_ms": t_sparse,
            "rerank_ms": t_rerank,
            "ttft_ms": t_ttft,
            "total_ms": t_total,
            "tokens_generated": token_count,
        })

    # Memory Footprints
    peak_ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if use_cuda else None

    # Compute Statistics
    def stats_dict(name: str, values: list):
        arr = np.array(values)
        return {
            "name": name,
            "avg": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    summary_data = [
        stats_dict("1. Router (Groq API)", m_router),
        stats_dict("2. ChromaDB (Dense)", m_dense),
        stats_dict("3. BM25 (Sparse)", m_sparse),
        stats_dict(f"4. Qwen3 Reranker ({'GPU' if use_cuda else 'CPU'})", m_rerank),
        stats_dict("5. Time to First Token (TTFT)", m_ttft),
        stats_dict("6. Total End-to-End Time", m_total),
    ]

    # Print Summary Table
    print_table(summary_data, peak_ram_mb, peak_vram_mb, len(queries), device_name)

    # Optional: Save CSV
    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        pd.DataFrame(records).to_csv(save_csv, index=False)
        print(f"Detailed per-query metrics saved to: {save_csv}")


def main():
    parser = argparse.ArgumentParser(description="LegalSaathi AI - Latency Benchmark Baseline (GPU Accelerated)")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of benchmark queries (default: 10)")
    parser.add_argument("--top-n", type=int, default=None, help="Override top_n candidates for retrieval/rerank (default from config: 40)")
    parser.add_argument("--dataset", type=str, default="src/evalutions/final_evalution_set.csv", help="Path to evaluation CSV")
    parser.add_argument("--save-csv", type=str, default="src/latency_evaluation/baseline_results.csv", help="CSV path to save per-query metrics")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        num_queries=args.num_queries,
        top_n=args.top_n,
        dataset_path=args.dataset,
        save_csv=args.save_csv,
    ))


if __name__ == "__main__":
    main()
