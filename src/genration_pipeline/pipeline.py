from src.rag_retrieval.retrival import *
from src.genration_pipeline.model import *
from src.genration_pipeline.new_prompts import *
from src.rag_retrieval.retrival import *
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from itertools import zip_longest
import json
from src.genration_pipeline.schemas import *

class LegalSaathiPipeline:
    """
    Reusable RAG pipeline for LegalSaathi.
    Import this anywhere — CLI, eval scripts, notebooks — instead of
    re-wiring retriever/reranker/generator each time.
    """
    def __init__(self, top_n=None, rerank_k=None, min_score=None):
        self.config = read_config_file()
        cfg = self.config["retrieval"]
        self.top_n = cfg["top_n"] if top_n is None else top_n
        self.rerank_k = cfg["rerank_k"] if rerank_k is None else rerank_k
        self.min_score = cfg["min_score"] if min_score is None else min_score
        self.max_chunks = cfg["max_chunks"]
        
        self.retriever = Retrieval(top_n=self.top_n)
        self.reranker = Reranker(rerank_k=self.rerank_k)
        self.model = get_model()
        self.chain = system_msg | self.model | StrOutputParser()
        self.decomposition_model = self.model.with_structured_output(
            SubQueriesSchema,
            method="json_schema",
        )
    
    def warmup(self, include_llm: bool = False) -> None:
        self.retriever.warmup()
        self.reranker.warmup()
        if include_llm:
            self.model.invoke("ping")    


    def query_decomposition(self, question: str, chat_history: list = None) -> list:

        try:
            query_decomposition_prompt = """
                Convert the user's question into concise, standalone legal search queries.
                STRICTLY Do not answer the question.

                STRICT Rules:
                - Use conversation history only to resolve references in the latest question.
                - Make follow-up queries standalone, preserving the user's intended topic.
                - Previous answers are not verified legal evidence; do not adopt their claims as facts.
                - Split only genuinely distinct legal issues.
                - Return one focused query for a single issue.
                - Preserve the user's meaning, relevant facts, named laws, dates, amounts,
                thresholds, conditions, exceptions and negations.
                - Repeat shared facts only where needed to make each query standalone.
                - Remove conversational filler and requests for reassurance.
                - For guaranteed outcomes, focus on entitlement, remedies and the conditions
                determining the outcome or amount. Do not assume the outcome is available.
                - Keep related conditions, exceptions and remedies together.
                - Do not invent facts, laws, section numbers or legal conclusions.
                - Do not add legal issues the user did not ask about.
                - Keep an already clear search query unchanged.
                - Return unrelated or insufficiently clear requests unchanged.

                Examples:

                User: I'm confused. Under the RTI Act, how long do they have to reply
                to my information request?
                Queries: ["Under the RTI Act, what time limit applies to replying to an information request?"]

                User: My landlord changed the locks while my belongings were inside.
                What can I do?
                Queries: ["What legal remedies are available to a tenant whose landlord changed the locks while the tenant's belongings remained inside?"]

                User: Can you promise I will get ₹50,000 compensation for a defective
                bicycle I bought for personal use?
                Queries: ["What remedies and conditions determine compensation for a defective bicycle bought for personal use, including whether ₹50,000 can be awarded?"]

                User: My employer has not paid my salary for two months, and an online
                seller refuses to refund a cancelled order. What can I do about both?
                Queries: ["What remedies are available when an employer has not paid salary for two months?", "What remedies are available when an online seller refuses to refund a cancelled order?"]

                User: Does a tenant need permission to sublet when the written rental
                agreement prohibits subletting?
                Queries: ["Does a tenant need permission to sublet when the written rental agreement prohibits subletting?"]

                User: Suggest a birthday cake flavour.
                Queries: ["Suggest a birthday cake flavour."]

                                
                Question: What are the punishments for murder and theft?
                Queries: ["What punishment applies to murder?", "What punishment applies to theft?"]

                Question: Is every threat a criminal-intimidation offence under the Bharatiya Nyaya Sanhita?
                Queries: ["Under the Bharatiya Nyaya Sanhita, what elements and intent are required for a threat to constitute criminal intimidation?"]

                Return only valid JSON in this format: {{"questions": ["standalone query"]}}.
                No answers, explanations or extra fields.

                """

            prompt = ChatPromptTemplate.from_messages([
                    ("system", query_decomposition_prompt),
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
   
    def retrieve_context(self, question, return_chunks=False, return_docs=False,
                     chat_history=None, return_diagnostics=False,
                     search_queries: list[str] | None = None):
        
        if search_queries is None:
            search_queries = self.query_decomposition(question, chat_history)

        ranked_groups = []
        diagnostics = {"search_queries": search_queries, "queries": []}
        for search_query in search_queries:
            docs = self.retriever.hybrid_retrieve_invoke(search_query)
            if return_diagnostics:
                all_ranked = self.reranker.rerank(docs=docs, query=search_query, return_all=True)
                ranked_docs = all_ranked[:self.rerank_k]
            else:
                ranked_docs = self.reranker.rerank(docs=docs, query=search_query)
            fallback = False

            if self.min_score is not None and ranked_docs:
                # keep the best chunk even if nothing passes the threshold
                passing = [
                    item for item in ranked_docs if float(item[0]) >= self.min_score
                ]
                fallback = not passing
                ranked_docs = passing or ranked_docs[:1]

            if return_diagnostics:
                diagnostics["queries"].append({
                    "query": search_query,
                    "hybrid_chunk_ids": [doc.metadata["chunk_id"] for doc in docs],
                    "reranked": [
                        {"rank": rank, "chunk_id": doc.metadata["chunk_id"], "score": float(score)}
                        for rank, (score, doc) in enumerate(all_ranked, 1)
                    ],
                    "after_filter_chunk_ids": [doc.metadata["chunk_id"] for _, doc in ranked_docs],
                    "fallback_used": fallback,
                })

            ranked_groups.append(ranked_docs)

        # Take chunks from each query in turn, keeping the configured number of unique chunks.
        selected_docs = []
        seen = set()

        for group in zip_longest(*ranked_groups):
            for item in group:
                if item is None:
                    continue
                _, doc = item
                key = (
                    doc.metadata.get("act_name"),
                    doc.metadata.get("chunk_id") or doc.page_content,
                )
                if key in seen:
                    continue
                seen.add(key)
                selected_docs.append(doc)
                if len(selected_docs) == self.max_chunks:
                    break
            if len(selected_docs) == self.max_chunks:
                break

        if return_diagnostics:
            diagnostics.update({
                "top_n": self.top_n,
                "rerank_k": self.rerank_k,
                "min_score": self.min_score,
                "max_chunks": self.max_chunks,
                "final_context": [{
                    "chunk_id": doc.metadata["chunk_id"],
                    "act_name": doc.metadata["act_name"],
                    "pages": self.document_pages(doc),
                    "text": doc.page_content,
                } for doc in selected_docs],
            })

        if return_docs:
            result = selected_docs
        elif return_chunks:
            result = [doc.page_content for doc in selected_docs]
        else:
            result = self.format_context(selected_docs)
        return (result, diagnostics) if return_diagnostics else result
    

    @staticmethod
    def document_pages(doc):
        return doc.metadata.get("pages") or [doc.metadata["page"]]

    def format_context(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
        
    def generate_stream(self, question: str, context: str, chat_history: list | None = None):
        """
        answer in streaming for chatbot
        chat/stream
        """
        yield from self.chain.stream({
            "context": context,
            "chat_history": chat_history or [],
            "question": question,
        })

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


    def run(self, question: str, chat_history: list | None = None) -> dict:
        docs = self.retrieve_context(question, return_docs=True, chat_history=chat_history)
        result = self.generate_answer(question, self.format_context(docs), chat_history)
        sources = [
            {"act_name": d.metadata.get("act_name"),
            "pages": self.document_pages(d),}
            for d in docs
        ]
        return {"answer": result["answer"], "sources": sources, "chat_history": result["chat_history"]}
