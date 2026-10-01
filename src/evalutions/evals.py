import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv

from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
from src.genration_pipeline.pipeline import LegalSaathiPipeline


load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)


class DeepseekMeshJudge(DeepEvalBaseLLM):
    """
    Wraps a LangChain ChatOpenAI client pointed at meshapi so
    deepeval's metrics can call it as a judge.
    """

    def __init__(self, model_name, base_url, api_key, temperature=0):
        self.model_name = model_name
        self.client = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        return self.client.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        response = await self.client.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return self.model_name


class LegalSaathiEvaluator:
    def __init__(self, threshold=0.7, output_path=None, max_workers=5):
        self.pipeline = LegalSaathiPipeline(top_n=30, rerank_k=7)
        self.output_path = output_path
        self.threshold = threshold
        self.max_workers = max_workers

        self.judge = DeepseekMeshJudge(
            model_name="deepseek/deepseek-v4-flash",
            base_url="https://api.meshapi.ai/v1",
            api_key=os.getenv("MESH_DEEP_SEEK_FLASH"),
        )

        # metric name -> metric class
        self.metric_classes = {
            "relevancy": ContextualRelevancyMetric,
            "recall": ContextualRecallMetric,
            "precision": ContextualPrecisionMetric,
            "answer_relevancy": AnswerRelevancyMetric,
            "faithfulness": FaithfulnessMetric,
        }

    def _evaluate_row(self, index, row):
        """Same per-row logic as before, just pulled into its own method so it
        can be submitted to a thread pool."""
        print('#############################')
        print(f"I:{index}")
        question = row["question"]
        golden_answer = row["golden_answer"]
        golden_answer_clean = row["without_disclaimer_ans"]

        # recall + precision need the disclaimer-stripped golden answer,
        # everything else uses the normal golden answer
        RECALL_PRECISION_METRICS = {"recall", "precision"}

        output = self.pipeline.genration_answer(question, chat_history=[])
        answer = output["answer"]
        context = output["context"]

        # ensure context is a list of plain strings
        if isinstance(context, str):
            context = [context]

        row_results = []
        # run each metric one at a time, in order (unchanged)
        for metric_name, metric_cls in self.metric_classes.items():
            expected_output = golden_answer_clean if metric_name in RECALL_PRECISION_METRICS else golden_answer

            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=expected_output,
                retrieval_context=context,
            )

            metric = metric_cls(threshold=self.threshold, model=self.judge, async_mode=False, verbose_mode=True)
            metric.measure(test_case)

            print(f"  -> {metric_name.capitalize()}: {metric.score} | Reason: {metric.reason}")

            row_results.append({
                "index": index,
                "question": question,
                "golden_answer": expected_output,
                "metric": metric_name,
                "score": metric.score,
                "reason": metric.reason,
            })

        return row_results

    def evaluate(self, df):
        all_rows = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._evaluate_row, index, row): index
                for index, row in df.iterrows()
            }

            for future in as_completed(futures):
                index = futures[future]
                try:
                    row_results = future.result()
                    all_rows.extend(row_results)
                except Exception as e:
                    print(f"Row {index} failed: {e}")

        results_df = pd.DataFrame(all_rows)
        # keep output ordered by original row index, same as sequential version
        results_df = results_df.sort_values("index").reset_index(drop=True)

        if self.output_path:
            results_df.to_csv(self.output_path, index=False)
            print(f"\nSaved all results -> {self.output_path}")

        return results_df


if __name__ == "__main__":
    df = pd.read_csv(r"D:\LegalSaathi AI\data\evalutions\gd.csv")

    evaluator = LegalSaathiEvaluator(
        output_path=r"D:\LegalSaathi AI\data\evalutions\eval_results.csv",
        max_workers=5,
    )

    results_df = evaluator.evaluate(df)