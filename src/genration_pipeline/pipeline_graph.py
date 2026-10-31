from concurrent.futures import ThreadPoolExecutor
from typing import Annotated
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from src.genration_pipeline.new_prompts import SYSTEM_PROMPT as AGENT_SYSTEM_PROMPT

SUMMARIZE_AFTER = 20   
KEEP_RECENT = 8        

SUMMARY_PROMPT = """You are a conversation summarizer. Summarize the key points, user questions, and legal context discussed in the conversation so far in concise, plain text:
<messages>
{messages}
</messages>
Write only a plain, clear summary. Strictly do NOT use section headers like 'SESSION INTENT', 'SUMMARY', or 'ARTIFACTS'."""        

# ---------- State ----------
def collect_sources(old, new):
    if not new:                    # [] = reset for a new question
        return []
    return (old or []) + new       # several searches in one turn add up     


class LegalState(AgentState):
    sources: Annotated[list[dict], collect_sources]


def create_search_tool(pipeline):
    @tool
    def search_legal_documents(query: str, config: RunnableConfig,tool_call_id: Annotated[str, InjectedToolCallId]):
        """Search Indian statutory acts (BNS, BSA, BNSS, Labour Codes, RTI, RERA)
        and the user's uploaded private documents (contracts, agreements, notices).
        Use for ANY question about laws, rights, punishments, procedures, or the user's files,
        including follow-up legal questions (rewrite them as a standalone query).
        Pass one focused query; call again for a separate legal issue.
        Do NOT use for greetings, thanks, or questions about earlier messages in this chat."""

        # 1. Safely extract user_id injected from backend config
        user_id = config.get("configurable", {}).get("user_id")
        vault = getattr(pipeline, "vault", None)
        use_vault = bool(user_id and vault)

        # 2. Retrieve main acts and vault files in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            main_f = pool.submit(pipeline.retriever.hybrid_retrieve_invoke, query)
            vault_f = pool.submit(vault.search, query, user_id, 10) if use_vault else None
            docs = main_f.result() + (vault_f.result() if vault_f else [])

        if not docs:
            return Command(update={
                "sources": [],
                "messages": [ToolMessage(
                    content="No relevant legal or vault documents found.",
                    tool_call_id=tool_call_id,
                )],
            })
        ranked = pipeline.reranker.rerank(docs=docs, query=query)
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
        return Command(update={
            "sources": sources,
            "messages": [ToolMessage(content=context, tool_call_id=tool_call_id)],
        })

    return search_legal_documents


# ---------- Agent (build ONCE at startup) ----------
def build_agent(pipeline, summarizer, checkpointer=None):
    return create_agent(
        model=pipeline.model,
        tools=[create_search_tool(pipeline)],
        system_prompt=AGENT_SYSTEM_PROMPT,
        state_schema=LegalState,
        middleware=[
            SummarizationMiddleware(
                model=summarizer,
                trigger=("messages", SUMMARIZE_AFTER),
                keep=("messages", KEEP_RECENT),
                summary_prompt=SUMMARY_PROMPT,
            )
        ],
        checkpointer=checkpointer,
    )


# ---------- Ask & Stream ----------
async def stream_ask(agent, question: str, session_id: str, user_id: int):
    """Yields ("token", text) for each token, then ("sources", list)."""
    config = {
        "configurable": {"thread_id": session_id, "user_id": user_id},
        "recursion_limit": 10,
    }
    sources = []
    async for mode, data in agent.astream(
        {"messages": [HumanMessage(content=question)], "sources": []},
        config=config,
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            chunk, _ = data
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield ("token", chunk.content)
        elif mode == "values":
            sources = data.get("sources", [])

    yield ("sources", sources)


def ask(agent, question: str, session_id: str, user_id: int) -> dict:
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)], "sources": []},  # [] resets sources
        config={
            "configurable": {"thread_id": session_id, "user_id": user_id},
            "recursion_limit": 10,
        },
    )
    return {
        "answer": result["messages"][-1].content,
        "sources": result.get("sources", []),  # [] when the agent never searched
    }


# Backward-compatible alias
build_graph = build_agent
