"""Threshold & Cutoff Tuner for LegalSaathi AI

Evaluates the exact trade-off of min_score thresholds across K = 4, 5, 6:
  - Tests thresholds: 0.0, 0.10, 0.15, 0.20, 0.25, 0.30
  - Measures: Avg Chunks, Precision, Recall, Hit Rate, and F1-Score
  - Runs in-memory: 1 retrieval pass, 0 extra LLM calls!

Usage:
  python src/evalutions/threshold_tuner.py
"""

import json
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.genration_pipeline.pipeline import LegalSaathiPipeline
from src.logger import logger
from src.utils.main_utils import read_config_file

load_dotenv()
warnings.filterwarnings("ignore")

K_VALUES = (4, 5, 6)
THRESHOLDS = (0.00, 0.10, 0.15, 0.20, 0.25, 0.30)


def run_threshold_tuning():
    config = read_config_file()
    cfg = config["evaluation"]

    out_dir = Path(cfg.get("output_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing LegalSaathi pipeline...")
    pipeline = LegalSaathiPipeline(
        top_n=cfg.get("top_n", 40),
        rerank_k=max(K_VALUES),
        min_score=None,
    )

    df = pd.read_csv(cfg["dataset_path"], converters={"relevant_chunk_ids": json.loads})
    logger.info(f"Loaded {len(df)} questions from {cfg['dataset_path']}")
    logger.info(f"Starting single retrieval pass for threshold tuning...")

    tuning_records = []
    question_records = []

    for idx, row in df.iterrows():
        question = row["question"]
        gold = set(row.get("relevant_chunk_ids") or [])

        docs, diag = pipeline.retrieve_context(question, return_docs=True, return_diagnostics=True)
        first_query = diag["queries"][0]
        reranked = first_query["reranked"]  

        if (idx + 1) % 5 == 0 or (idx + 1) == len(df):
            logger.info(f"Progress: Evaluated {idx + 1}/{len(df)} questions...")

        row_data = {"index": idx, "question": question}

        for k in K_VALUES:
            candidates = reranked[:k]

            for th in THRESHOLDS:

                passing = [r["chunk_id"] for r in candidates if r["score"] >= th]
                chosen = passing if passing else [candidates[0]["chunk_id"]]

                hits = len(set(chosen) & gold)
                prec = hits / len(chosen) if chosen else 0.0
                rec = hits / len(gold) if gold else 0.0
                hit = 1.0 if hits > 0 else 0.0

                tag = f"K{k}_min{int(th * 100):02d}"
                row_data[f"{tag}_Count"] = len(chosen)
                row_data[f"{tag}_Precision"] = prec
                row_data[f"{tag}_Recall"] = rec

                tuning_records.append({
                    "K": k,
                    "min_score": th,
                    "chunk_count": len(chosen),
                    "precision": prec,
                    "recall": rec,
                    "hit_rate": hit,
                })

        question_records.append(row_data)

    detailed_df = pd.DataFrame(question_records)
    detailed_path = out_dir / "threshold_tuning_detailed.csv"
    detailed_df.to_csv(detailed_path, index=False)

    summary_df = pd.DataFrame(tuning_records).groupby(["K", "min_score"]).agg({
        "chunk_count": "mean",
        "precision": "mean",
        "recall": "mean",
        "hit_rate": "mean",
    }).reset_index()

    summary_df["F1_Score"] = (
        2 * (summary_df["precision"] * summary_df["recall"])
        / (summary_df["precision"] + summary_df["recall"])
    ).fillna(0.0).round(4)

    summary_df = summary_df.round({
        "chunk_count": 2,
        "precision": 4,
        "recall": 4,
        "hit_rate": 4,
    })

    summary_path = out_dir / "threshold_tuning_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("        THRESHOLD TUNING MATRIX (K = 4, 5, 6 vs min_score)")
    print("=" * 78)
    print(f"{'K':^5} | {'min_score':^10} | {'Avg Chunks':^12} | {'Precision':^11} | {'Recall':^10} | {'Hit Rate':^10} | {'F1-Score':^10}")
    print("-" * 78)

    best_f1_idx = summary_df["F1_Score"].idxmax()
    for i, row in summary_df.iterrows():
        star = " ⭐ (BEST)" if i == best_f1_idx else ""
        print(
            f"{int(row['K']):^5} | "
            f"{row['min_score']:^10.2f} | "
            f"{row['chunk_count']:^12.2f} | "
            f"{row['precision'] * 100:^10.1f}% | "
            f"{row['recall'] * 100:^9.1f}% | "
            f"{row['hit_rate'] * 100:^9.1f}% | "
            f"{row['F1_Score']:^9.4f}{star}"
        )

    print("=" * 78)
    logger.info(f"Detailed results saved to: {detailed_path}")
    logger.info(f"Summary table saved to:    {summary_path}\n")


if __name__ == "__main__":
    run_threshold_tuning()

