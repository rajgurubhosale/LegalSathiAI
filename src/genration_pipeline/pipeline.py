from src.genration_pipeline.model import *
from src.rag_retrieval.retrival import *
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from src.genration_pipeline.prompts import RAG_SYSTEM_PROMPT
from src.genration_pipeline.schemas import *
from src.logger import logger
from .prompts import QUERY_DECOMPOSITION_PROMPT


class LegalSaathiPipeline:
    """
    Reusable RAG pipeline for LegalSaathi.
    Import this anywhere — CLI, eval scripts, notebooks — instead of
    re-wiring retriever/reranker/generator each time.
    """
    def __init__(self, top_n=None, rerank_k=None, min_score=None,groq_api_key=None,vault = None):

        self.vault = vault or None
        self.config = read_config_file()
        cfg = self.config["retrieval"]
        self.top_n = cfg["top_n"] if top_n is None else top_n
        self.rerank_k = cfg["rerank_k"] if rerank_k is None else rerank_k
        self.min_score = cfg["min_score"] if min_score is None else min_score
        self.max_chunks = cfg["rerank_k"]
        
        self.retriever = Retrieval(top_n=self.top_n)
        self.reranker = Reranker(rerank_k=self.rerank_k)
        
        self.model = get_model(api_key=groq_api_key)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "Context:\n<context>\n{context}\n</context>\n\nQuestion:\n{question}"),
        ])
        self.chain = self.prompt | self.model | StrOutputParser()

        self.decomposition_model = self.model.with_structured_output(
            SubQueriesSchema,
            method="json_schema",
        )
    
    def warmup(self, include_llm: bool = False) -> None:
        self.retriever.warmup()
        self.reranker.warmup()
        if include_llm:
            self.model.invoke("ping")    


    def query_decomposition(self, question: str, chat_history: list) -> list:

        try:
            prompt = ChatPromptTemplate.from_messages([
                    ("system", QUERY_DECOMPOSITION_PROMPT),
                    MessagesPlaceholder("chat_history"),
                    ("human", "Question:\n{question}"),
                ])


            chain = prompt | self.decomposition_model

            result = chain.invoke({"question": question, "chat_history": chat_history or []})

            sub_questions = result.questions

            if not isinstance(sub_questions, list) or not sub_questions:
                return [question]
            
            return sub_questions

        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [question]

    async def aquery_decomposition(self, question: str, chat_history: list | None = None) -> list:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", QUERY_DECOMPOSITION_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "Question:\n{question}"),
            ])
            chain = prompt | self.decomposition_model
            result = await chain.ainvoke({"question": question, "chat_history": chat_history or []})
            return result.questions or [question]
        except Exception as e:
            logger.error(f"Async query decomposition failed: {e}")
            return [question]


    def retrive_context(self, question: str, chat_history: list | None = None) -> list:


        search_queries = self.query_decomposition(question, chat_history)    

        ranked_groups = []
        for query in search_queries:
            docs = self.retriever.hybrid_retrieve_invoke(query)
            ranked = self.reranker.rerank(docs=docs, query=query)
            ranked_groups.append(ranked)

        all_scored = sorted(
            [item for group in ranked_groups for item in group],
            key=lambda x: x[0],
            reverse=True
        )


        selected_docs = []
        seen = set()
        for _, doc in all_scored:   

            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id not in seen:
                seen.add(chunk_id)
                selected_docs.append(doc)
                if len(selected_docs) == self.max_chunks:
                    break

            
        return selected_docs
    

    @staticmethod
    def document_pages(doc):
        return doc.metadata.get("pages") or [doc.metadata["page"]] or [doc.metadata["page_no"]]

    def format_context(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)


    def generate_answer(self,question: str, context: str, chat_history: list = None) -> dict:
        """
        Generate an answer using retrieved context + prior conversation history.
        method: invoke for the evalution and checks
        """
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

