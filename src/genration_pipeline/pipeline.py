from src.rag_retrieval.retrival import *
from src.genration_pipeline.model import *
from src.genration_pipeline.new_prompts import *
from src.rag_retrieval.retrival import *
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from itertools import zip_longest
import json
from src.genration_pipeline.schemas import *

class LegalSaathiPipeline:
    """
    Reusable RAG pipeline for LegalSaathi.
    Import this anywhere — CLI, eval scripts, notebooks — instead of
    re-wiring retriever/reranker/generator each time.
    """

    def __init__(self, top_n: int = 40, rerank_k: int = 6, min_score=None):
        self.min_score = min_score
        self.retriever = Retrieval(top_n=top_n)
        self.reranker = Reranker(rerank_k=rerank_k)
        self.model = get_model()
        self.chain = system_msg | self.model | StrOutputParser()
        self.decomposition_model = self.model.with_structured_output(
            SubQueriesSchema,
            method="json_schema",)
        
    def query_decomposition(self, question: str) -> list:

        try:
            query_decomposition_prompt = """
                Convert the user's question into concise, standalone legal search queries.
                STRICTLY Do not answer the question.

                STRICT Rules:
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

                Return only the required structured output with the questions list.
                No answers, explanations or extra fields.

                """

            prompt = ChatPromptTemplate.from_messages([
                    ("system", query_decomposition_prompt),
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
        
    def retrieve_context(self, question: str, return_chunks: bool = False):
        search_queries = self.query_decomposition(question)

        selected_docs = []
        seen = set()
        ranked_groups = []

        for search_query in search_queries:
            docs = self.retriever.hybrid_retrieve_invoke(search_query)

            ranked_docs = self.reranker.rerank(
                docs=docs,
                query=search_query,
            )

            if self.min_score is not None:
                ranked_docs = [
                    item for item in ranked_docs
                    if float(item[0]) >= self.min_score
                ] or ranked_docs[:1]

            ranked_groups.append(ranked_docs)

        # Take candidates from each query in turns, keeping four unique chunks.
        for group in zip_longest(*ranked_groups):
            for item in group:
                if item is None:
                    continue
                _, doc = item
                key = (
                    doc.metadata.get("act_name"),
                    doc.metadata.get("chunk_id") or doc.page_content,
                )

                if key not in seen:
                    seen.add(key)
                    selected_docs.append(doc)
                if len(selected_docs) == 4:
                    break
            if len(selected_docs) == 4:
                break

        chunks = [doc.page_content for doc in selected_docs]

        return chunks if return_chunks else "\n\n".join(chunks)
    

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
