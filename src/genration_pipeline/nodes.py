import asyncio
from typing import TypedDict, List,  Literal, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages, RemoveMessage, REMOVE_ALL_MESSAGES
from src.logger import logger
from .prompts import (
    DECISION_MODEL_PROMPT, 
    CHIT_CHAT_PROMPT, 
    REFUSAL_MSG, 
    SUMMARY_PROMPT, 
    RAG_SYSTEM_PROMPT 
)

from .model import amos_client, chit_chat_model, summarizer 
from typesafe_sdk import Choice,TypeSafeClient

SUMMARIZE_AFTER = 20
KEEP_RECENT = 8

class LegalState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    route_decision: Literal["guardrail", "chit_chat", "rag_retrieval"]
    sources: List[dict]
    



async def decision_node(state: LegalState):
    user_query = state["messages"][-1].content
    result = amos_client.system_one(
        state=f"User query in Indian Legal Assistant: \"{user_query}\"",
        questions={
            "route": Choice(
               instructions=(
                    "Classify this message for LegalSaathi, an Indian Legal Assistant (BNS, BNSS, BSA, Labour Codes, RTI, Contracts). "
                    "1. Asking ABOUT a crime, penalty, section, or citizen right is strictly 'rag_retrieval'. "
                    "2. Flag 'guardrail' ONLY if the user asks HOW TO commit fraud, forge documents, destroy evidence, or evade police. "
                    "3. Short legal follow-ups (e.g., 'what about bail?', 'and penalty for second offence?') belong to 'rag_retrieval'. "
                    "4. If in doubt or borderline between casual conversation and a legal issue, always favor 'rag_retrieval'."
                ),
                criteria={
                    "rag_retrieval": (
                        "Legal questions, laws (BNS, Labour Codes, RTI, Contracts), rights, "
                        "punishments, court procedures, or questions on uploaded documents."
                    ),
                    "chit_chat": (
                        "Greetings (hi, hello), small talk, 'who are you', compliments, "
                        "or questions about what the assistant can do."
                    ),
                    "guardrail": (
                        "Instructions or requests for help to commit fraud, forge documents, "
                        "evade police, hack, or harm someone."
                    ),
                },
            )
        },
    )
    route_obj = result.answers['route']
    
    return {"route_decision": route_obj.choice}
    


async def chit_chat_node(state: LegalState):
    user_message = state["messages"][-1].content
    
    res = await chit_chat_model.ainvoke([
        ("system", CHIT_CHAT_PROMPT),
        ("human", user_message)
    ])
    return {"messages": [res], "sources": []}



async def guardrail_node(state: LegalState):
    
    return {"messages": [AIMessage(content=REFUSAL_MSG)],"sources": []}


async def summarization_trim_middleware(state:LegalState):
    """
    Summarize the conversation if it exceeds 8 messages.
    """

    messages = state["messages"]
    if len(messages) <= SUMMARIZE_AFTER:
        return {}  

    older = messages[:-KEEP_RECENT]
    recent = messages[-KEEP_RECENT:]

    conversation_text = "\n".join(
        f"{getattr(m, 'type', 'message')}: {getattr(m, 'content', '')}"
        for m in older
        if getattr(m, "content", None)
    )
    

    summary = await summarizer.ainvoke([
        ("system",SUMMARY_PROMPT),
        ("human",f"conversation summry so far:\n{conversation_text}")])  

    logger.info(f"Summarization Triggered - pruned {len(older)} older messages.")

    return {
        "messages": [
            *[RemoveMessage(id=REMOVE_ALL_MESSAGES)],
            SystemMessage(content=f"Summary of earlier conversation:\n{summary.content}"),
            *recent
        ]
    }



def make_app_node(pipeline):
    """Factory function so the node has access to pipeline cleanly."""

    async def app_node(state: LegalState, config: RunnableConfig):
        
        query = state["messages"][-1].content
        user_id = (config or {}).get("configurable", {}).get("user_id")
        vault = getattr(pipeline, "vault", None)
        use_vault = bool(user_id and vault)


        async def _empty():  
            return []

        main_task = asyncio.to_thread(pipeline.retriever.hybrid_retrieve_invoke, query)
        vault_task = vault.asearch(query, user_id, 10) if use_vault else _empty()
        main_docs, vault_docs = await asyncio.gather(main_task, vault_task)
        docs = (main_docs or []) + (vault_docs or [])
        if not docs:
            no_docs_msg = AIMessage(
                content="I could not find relevant statutory provisions or vault documents matching your query."
            )
            return {"messages": [no_docs_msg], "sources": []}


        ranked = await asyncio.to_thread(pipeline.reranker.rerank, docs=docs, query=query)
        top = [doc for _, doc in ranked[:pipeline.max_chunks]]

        chunks, sources = [], []
        for d in top:
            m = d.metadata
            if "doc_name" in m:
                name, pages, kind = m["doc_name"], [m.get("page_no", 1)], "Your document"
            else:
                name, pages, kind = m.get("act_name", "Statute"), pipeline.document_pages(d), "Act"
            chunks.append(f"SOURCE: {kind} | {name} | PDF page {pages[0]}\n{d.page_content}")
            sources.append({"act_name": name, "pages": pages})
        context = "\n\n".join(chunks)

        system_instruction = (
                f"{RAG_SYSTEM_PROMPT}\n\n"
                f"RELEVANT LEGAL SOURCES:\n{context}"
            )

        response = await pipeline.model.ainvoke([
            ("system", system_instruction),
            *state["messages"]
        ])

        return {
            "messages": [response],
            "sources": sources
        }
    return app_node
