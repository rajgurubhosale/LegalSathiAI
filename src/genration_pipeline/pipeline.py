from src.rag_retrieval.retrival import *
from src.genration_pipeline.model import *
from src.genration_pipeline.new_prompts import *
from src.genration_pipeline.pipeline import *
from src.rag_retrieval.retrival import *
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

class LegalSaathiPipeline:
    """
    Reusable RAG pipeline for LegalSaathi.
    Import this anywhere — CLI, eval scripts, notebooks — instead of
    re-wiring retriever/reranker/generator each time.
    """

    def __init__(self, top_n: int = 50, rerank_k: int = 3):
        self.retriever = Retrieval(top_n=top_n)
        self.reranker = Reranker(rerank_k=rerank_k)
        self.model = get_model()
        self.chain = system_msg | self.model | StrOutputParser()

    def retrieve_context(self, query: str) -> str:
        docs = self.retriever.retrieve_invoke(query)

        scored_docs = self.reranker.rerank(
            docs=docs,
            query=query,
            top_n=self.rerank_k,
        )

        context = "\n\n".join(
            f"Act: {doc.metadata.get('act_name', 'Unknown')}\n"
            f"PDF pages: {doc.metadata.get('pages', doc.metadata.get('page', 'Unknown'))}\n"
            f"Text:\n{doc.page_content}"
            for score, doc in scored_docs
        )

        return context


    def generate_answer(self,question: str, context: str, chat_history: list = None) -> dict:
        """Generate an answer using retrieved context + prior conversation history."""
        if chat_history is None:
            chat_history = []
            
        
    
        answer = self.chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": question,
        })

        updated_history = chat_history + [
            HumanMessage(content=question),
            AIMessage(content=answer),
        ]

        return {"answer": answer, "question": question, "chat_history": updated_history}


    def run(self, question: str, chat_history: list = None) -> dict:
        context = self.retrieve_context(question)
        return self.generate_answer(question, context, chat_history)