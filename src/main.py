from src.logger import *
from src.generation.genrator import generate_answer
from src.retrieval.retriever import *
from src.retrieval.reranker import *
from src.retrieval.parent_store import ParentStore

runner = Retrieval(top_n=50)
reranker = Reranker(rerank_k=3)
parent_store = ParentStore()


def run_cli():
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
            docs = runner.retrieve_invoke(question)
            context = reranker.rerank_invoke(docs, question)

            # NEW: resolve parent chunks instead of using child text directly

            result = generate_answer(question, context, chat_history)
            chat_history = result["chat_history"]
            print(f"\nLegalSaathi: {result['answer']}")
        except Exception as e:
            print(f"\n[Error] Something went wrong: {e}")


if __name__ == "__main__":
    run_cli()