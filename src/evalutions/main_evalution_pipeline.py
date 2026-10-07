import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import json
from deepeval import metrics
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.utils.main_utils import read_config_file

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from src.genration_pipeline.pipeline import LegalSaathiPipeline
load_dotenv()
warnings.filterwarnings("ignore")
from pathlib import Path
import time
import math


def retrieval_metrics(docs, relevant_chunk_ids):

    # Preserve ranking while removing duplicate IDs.
    retrieved = list(dict.fromkeys(
        doc.metadata["chunk_id"] for doc in docs
    ))
    relevant = set(relevant_chunk_ids)

    # Recall and nDCG are undefined without relevant evidence.
    if not relevant:
        return {}

    matches = len(set(retrieved) & relevant)
    scores = {
        "Precision (all chunks)": matches / len(retrieved) if retrieved else 0.0,
        "Recall (all chunks)": matches / len(relevant),
    }

    for k in (5, 20):
        hits = [
            int(chunk_id in relevant)
            for chunk_id in retrieved[:k]
        ]

        scores[f"Hit Rate@{k}"] = float(any(hits))
        scores[f"Recall@{k}"] = sum(hits) / len(relevant)

        if k == 5:
            scores["MRR@5"] = next(
                (1 / rank for rank, hit in enumerate(hits, 1) if hit),
                0.0,
            )

            dcg = sum(
                hit / math.log2(rank + 1)
                for rank, hit in enumerate(hits, 1)
            )
            ideal_dcg = sum(
                1 / math.log2(rank + 1)
                for rank in range(1, min(k, len(relevant)) + 1)
            )
            scores["nDCG@5"] = dcg / ideal_dcg

    return scores


class MeshJudge(DeepEvalBaseLLM):
    """LangChain ChatOpenAI client (meshapi) wrapped as a deepeval judge."""

    def __init__(self, model_name, base_url, api_key):
        self.model_name = model_name
        # max_retries helps with rate limits when many requests run at once
        
        self.client = ChatOpenAI(
            model=model_name, base_url=base_url, api_key=api_key,
            temperature=0, max_retries=5,
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        return self.client.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return (await self.client.ainvoke(prompt)).content

    def get_model_name(self) -> str:
        return self.model_name

class LegalSaathiEvaluator:

    def __init__(self,config):
            self.config = config
            
            evalution_config = config['evaluation']

            judge_config = evalution_config['judge']
            
            api_key = os.getenv("MESH_DEEP_SEEK_FLASH")
            if not api_key:
                raise ValueError("MESH_DEEP_SEEK_FLASH is missing from .env")
    
            self.pipeline = LegalSaathiPipeline(
                top_n=evalution_config["top_n"],
                rerank_k=evalution_config["rerank_k"],
                min_score=evalution_config.get("min_score"),
            )

            self.judge = MeshJudge(
                judge_config["model_name"],
                judge_config["base_url"],
                api_key,
            )
    
            self.threshold = evalution_config["threshold"]
            self.gen_workers = evalution_config["gen_workers"]
            self.max_concurrent = evalution_config["max_concurrent"]

            self.metric_classes = {
                "relevancy": ContextualRelevancyMetric,
                "recall": ContextualRecallMetric,
                "precision": ContextualPrecisionMetric,
                "answer_relevancy": AnswerRelevancyMetric,
                "faithfulness": FaithfulnessMetric,
                }
            
    def _generate(self, item):
        index, row = item
        question = row["question"]

        docs, diagnostics = self.pipeline.retrieve_context(
            question, return_docs=True, return_diagnostics=True,
        )
        retrieval_scores = retrieval_metrics(docs, row["relevant_chunk_ids"])
        diagnostics.update({
            "index": int(index),
            "question": question,
            "expected_answer": row["expected_answer"],
            "relevant_chunk_ids": row["relevant_chunk_ids"],
            "retrieval_scores": retrieval_scores,
            "embedding": getattr(self, "config", {}).get("embedding"),
            "rerank": getattr(self, "config", {}).get("rerank"),
            "vectorstore": getattr(self, "config", {}).get("vectorstore"),
        })
        # Persist retrieval even when answer generation subsequently hits a limit.
        if getattr(self, "diagnostics_path", None) is not None:
            with self.diagnostics_lock:
                with self.diagnostics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(diagnostics, ensure_ascii=False) + "\n")
        chunks = [doc.page_content for doc in docs]
        context = self.pipeline.format_context(docs)

        output = self.pipeline.generate_answer(
            question=question,
            context=context,
            chat_history=[],
        )

        return index, output["answer"], chunks, retrieval_scores

    @staticmethod
    def _build_cases(df, generated, expected_col):
        return [
            LLMTestCase(
                name=str(index),
                input=df.loc[index, "question"],
                actual_output=answer,
                expected_output=df.loc[index, expected_col],
                retrieval_context=context,
            )
            for index, answer, context, _ in generated
        ]
    
    def run(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
        min_batch_interval: float = 70.0,
        stop_on_errors: bool = False,
        batch_size: int = 2,
    ):
        if df.empty:
            raise ValueError("The evaluation dataset is empty.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_results = []
        retrieval_rows = []
        retrieval_path = output_path.with_name(f"{output_path.stem}_retrieval.csv")
        self.diagnostics_path = output_path.with_name(f"{output_path.stem}_diagnostics.jsonl")
        self.diagnostics_path.write_text("", encoding="utf-8")
        self.diagnostics_lock = Lock()
        print(f"Saving retrieval diagnostics to {self.diagnostics_path}")
        for start in range(0, len(df), batch_size):
            batch_started = time.monotonic()
            batch = df.iloc[start:start + batch_size]
            print(f"Evaluating questions {start + 1}–{start + len(batch)} of {len(df)}")

            with ThreadPoolExecutor(max_workers=self.gen_workers) as executor:
                generated = list(executor.map(self._generate, batch.iterrows()))

            for index, _, _, scores in generated:
                if scores:
                    retrieval_rows.append({
                        "index": index,
                        "question": df.loc[index, "question"],
                        **scores,
                    })
            if retrieval_rows:
                pd.DataFrame(retrieval_rows).to_csv(retrieval_path, index=False)

            test_cases = self._build_cases(df, generated, "expected_answer")
            metric_options = {
                "threshold": self.threshold,
                "model": self.judge,
                "async_mode": True,
            }
            metrics = [
                metric_class(**metric_options)
                for metric_class in self.metric_classes.values()
            ]

            evaluation = evaluate(
                test_cases=test_cases,
                metrics=metrics,
                async_config=AsyncConfig(
                    run_async=True,
                    max_concurrent=self.max_concurrent,
                ),
                display_config=DisplayConfig(print_results=False),
                error_config=ErrorConfig(ignore_errors=True),
            )

            rows = []
            for test_result in evaluation.test_results:
                index = int(test_result.name)
                for metric in test_result.metrics_data or []:
                    rows.append({
                        "index": index,
                        "question": df.loc[index, "question"],
                        "metric": metric.name,
                        "score": metric.score,
                        "reason": metric.reason,
                        "error": metric.error,
                    })

            batch_results = pd.DataFrame(rows)
            if batch_results.empty:
                raise RuntimeError(f"Batch starting at question {start + 1} returned no results.")

            batch_results = batch_results.sort_values(["index", "metric"])
            batch_results.to_csv(
                output_path,
                mode="w" if start == 0 else "a",
                header=start == 0,
                index=False,
            )
            all_results.append(batch_results)
            print(f"Saved batch to {output_path}")

            if stop_on_errors:
                scores = pd.to_numeric(batch_results["score"], errors="coerce")
                errors = batch_results["error"].fillna("").astype(str).str.strip()
                counts = batch_results.groupby("index")["metric"].nunique()
                if (
                    len(batch_results) != len(batch) * len(metrics)
                    or batch_results.duplicated(["index", "metric"]).any()
                    or set(counts.index) != set(batch.index)
                    or not counts.eq(len(metrics)).all()
                    or not scores.between(0, 1).all()
                    or errors.ne("").any()
                ):
                    failures = errors[errors.ne("")].drop_duplicates().tolist()
                    raise RuntimeError(
                        f"Incomplete or failed metrics; partial results saved to {output_path}. "
                        + " | ".join(failures)
                    )

            if start + batch_size < len(df):
                elapsed = time.monotonic() - batch_started
                pause = max(0.0, min_batch_interval - elapsed)
                print(f"Waiting {pause:.1f} seconds before the next batch...")
                time.sleep(pause)

        results = pd.concat(all_results, ignore_index=True)
        print(results.groupby("metric")["score"].mean().round(3))
        print(f"Metric errors: {results['error'].notna().sum()}")
        if retrieval_rows:
            print("\nRetrieval metrics (final combined context):")
            print(pd.DataFrame(retrieval_rows).drop(columns=["index", "question"]).mean().round(3))
            print(f"Saved retrieval scores to {retrieval_path}")
        return results

    
if __name__ == "__main__":
    config = read_config_file()
    settings = config["evaluation"]

    
    df = pd.read_csv(
        settings["dataset_path"],
        converters={
        "relevant_chunk_ids": json.loads,
        "relevant_pages": json.loads,
        "relevant_acts": json.loads,},)
    

    evaluator = LegalSaathiEvaluator(config)
    output_path = Path(settings["output_path"])
    evaluator.run(df, output_path=settings["output_path"])
