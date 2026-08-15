from src.logger import *
from src.generation.genrator import generate_answer
from src.retrieval.retriever import *
from src.retrieval.reranker import *
from src.retrieval.parent_store import ParentStore
from src.generation.llm import *
import json

class LegalSaathiPipeline:
    """
    Reusable RAG pipeline for LegalSaathi.
    Import this anywhere — CLI, eval scripts, notebooks — instead of
    re-wiring retriever/reranker/generator each time.
    """

    def __init__(self, top_n: int = 50, rerank_k: int = 3):
        self.runner = Retrieval(top_n=top_n)
        self.reranker = Reranker(rerank_k=rerank_k)
        self.model = get_model()
        
        
    def retrieve_context(self, question: str):
        sub_questions = self.query_decomposition(question)
        
        # 1. Pool all raw retrieved documents globally
        all_raw_docs = []
        seen = set()

        for sub_q in sub_questions:
            docs = self.runner.retrieve_invoke(sub_q)
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    all_raw_docs.append(doc)

        if not all_raw_docs:
            return "", []

        # 2. Rerank the entire pooled list against the ORIGINAL question
        # (or you can use a cross-encoder to score them and pick the top K overall)
        str_output, final_context_list = self.reranker.rerank_invoke(all_raw_docs, question)
        
        return str_output, final_context_list




    def extract_text_from_response(self, response):
        """Always return a clean string"""
        
        content = response.content if hasattr(response, "content") else response

        # If already string
        if isinstance(content, str):
            return content.strip()

        # If list → convert to string safely
        if isinstance(content, list):
            return " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            ).strip()

        # If dict → convert to JSON string
        if isinstance(content, dict):
            return json.dumps(content)

        # fallback
        return str(content).strip()


    def query_decomposition(self, question: str) -> list:

        try:
            prompt = f"""You split legal questions into standalone sub-questions,
            one per distinct legal topic/offense. If the question already covers only
            one topic, return it unchanged as a single-item list.

            Return ONLY a JSON list of strings. No preamble, no explanation, no markdown.

            Example 1:
            Question: "What are the punishments for murder and theft?"
            Output: ["What is the punishment for murder?", "What is the punishment for theft?"]

            Example 2:
            Question: "What is the punishment for theft?"
            Output: ["What is the punishment for theft?"]

            Question: "{question}"
            Output:"""

            response = self.model.invoke(prompt)

            raw_text = self.extract_text_from_response(response)

            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

            try:
                sub_queries = json.loads(raw_text)
            except:
                # fallback if model gives plain text
                return [question]

            if not isinstance(sub_queries, list) or not sub_queries:
                return [question]

            return sub_queries

        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [question]
            
    
    def genration_answer(self, question: str, chat_history: list = None):
        """
        Full pipeline: retrieve -> rerank -> generate.
        Returns a dict: {"answer": str, "context": ..., "chat_history": [...]}
        Use `answer` as `actual_output` and `context` as `retrieval_context`
        in DeepEval test cases.
        """
        if chat_history is None:
            chat_history = []

        
        
        
        str_output,context = self.retrieve_context(question)
        
        result = generate_answer(question, str_output, chat_history)

        return {
            "answer": result["answer"],
            'context':context,
            "chat_history": result["chat_history"],
        }


def run_cli():
    pipeline = LegalSaathiPipeline(top_n=50, rerank_k=3)

    print("=" * 60)
    print("LegalSaathi — Ask me about Indian criminal law (BNS/BNSS)")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("=" * 60)

    chat_history = []
    while True:
        question = input("\nYou: ").strip()
        print(f'User: {question}')

        if question.lower() in ("exit", "quit"):
            print("LegalSaathi: Goodbye!")
            break
        if not question:
            continue

        try:
            result = pipeline.genration_answer(question, chat_history)
            chat_history = result["chat_history"]
            print(f"\nLegalSaathi: {result['answer']}")
        except Exception as e:
            print(f"\n[Error] Something went wrong: {e}")


if __name__ == "__main__":
    run_cli()