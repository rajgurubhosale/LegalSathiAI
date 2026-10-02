import os
import warnings
from concurrent.futures import ThreadPoolExecutor

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
            
            evalution_config = config['evaluation']

            judge_config = evalution_config['judge']
            
            api_key = os.getenv("MESH_DEEP_SEEK_FLASH")
            if not api_key:
                raise ValueError("MESH_DEEP_SEEK_FLASH is missing from .env")
    
            self.pipeline = LegalSaathiPipeline(top_n=35, rerank_k=4)
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

        chunks = self.pipeline.retrieve_context(question, return_chunks=True)
        context = "\n\n".join(chunks)

        output = self.pipeline.generate_answer(
            question=question,
            context=context,
            chat_history=[],
        )

        return index, output["answer"], chunks

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
            for index, answer, context in generated
        ]
    
    def run(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
        min_batch_interval: float = 70.0,
    ):
        if df.empty:
            raise ValueError("The evaluation dataset is empty.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_results = []
        batch_size = 2

        for start in range(0, len(df), batch_size):
            batch_started = time.monotonic()
            batch = df.iloc[start:start + batch_size]
            print(f"Evaluating questions {start + 1}–{start + len(batch)} of {len(df)}")

            with ThreadPoolExecutor(max_workers=self.gen_workers) as executor:
                generated = list(executor.map(self._generate, batch.iterrows()))

            test_cases = self._build_cases(df, generated, "updated_golden_answer")
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

            if start + batch_size < len(df):
                elapsed = time.monotonic() - batch_started
                pause = max(0.0, min_batch_interval - elapsed)
                print(f"Waiting {pause:.1f} seconds before the next batch...")
                time.sleep(pause)

        results = pd.concat(all_results, ignore_index=True)
        print(results.groupby("metric")["score"].mean().round(3))
        print(f"Metric errors: {results['error'].notna().sum()}")
        return results

    
if __name__ == "__main__":
    config = read_config_file()
    settings = config["evaluation"]

    df = pd.read_csv(settings["dataset_path"])
    evaluator = LegalSaathiEvaluator(config)
    output_path = Path(settings["output_path"])
    evaluator.run(df, output_path=settings["output_path"])
