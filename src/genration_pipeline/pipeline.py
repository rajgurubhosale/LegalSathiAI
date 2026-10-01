from src.rag_retrieval.retrival import *
from src.genration_pipeline.model import *
from src.genration_pipeline.new_prompts import *
from src.rag_retrieval.retrival import *
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import json
from src.genration_pipeline.schemas import *

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
        self.decomposition_model = self.model.with_structured_output(
            SubQueriesSchema,
            method="json_schema",)
        
    def query_decomposition(self, question: str) -> list:

        try:
            query_decomposition_prompt = f"""You split legal questions into standalone sub-questions,
            one per distinct legal topic/offense. If the question already covers only
            one topic, return it unchanged as a single-item list.

            Return ONLY a JSON list of strings. No preamble, no explanation, no markdown.

            Example 1:
            Question: "What are the punishments for murder and theft?"
            Output: ["What is the punishment for murder?", "What is the punishment for theft?"]

            Example 2:
            Question: "What is the punishment for theft?"
            Output: ["What is the punishment for theft?"]

            """
            prompt = ChatPromptTemplate.from_messages([
                ("system",query_decomposition_prompt),  
                ("human", "Question:\n{question}"),
            ])

            chain = prompt | self.decomposition_model

            result = chain.invoke({"question": question})

            sub_questions = result.questions

            if not isinstance(sub_questions, list) or not sub_questions:
                return [question]
            
            
            return sub_questions

        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [question]
        
    def retrieve_context(self, question: str) -> str:
        sub_questions = self.query_decomposition(question)

        all_docs = []
        seen = set()

        for sub_question in sub_questions:
            docs = self.retriever.retrieve_invoke(sub_question)

            for doc in docs:
                key = (
                    doc.metadata.get("act_name"),
                    doc.metadata.get("chunk_id"),
                    doc.page_content,
                )

                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        if not all_docs:
            return ""

        scored_docs = self.reranker.rerank(
            docs=all_docs,
            query=question,
        )

        return "\n\n".join(
            f"Act: {doc.metadata.get('act_name', 'Unknown')}\n"
            f"PDF pages: {doc.metadata.get('pages', doc.metadata.get('page', 'Unknown'))}\n"
            f"Text:\n{doc.page_content}"
            for score, doc in scored_docs
        )    
   

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