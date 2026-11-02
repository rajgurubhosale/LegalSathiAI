from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from .nodes import (
    LegalState,
    decision_node,
    chit_chat_node,
    guardrail_node,
    summarization_trim_middleware,
    make_app_node
)


def build_graph_workflow(pipeline, checkpointer=None):

    graph = StateGraph(LegalState)


    graph.add_node("summarize", summarization_trim_middleware)
    graph.add_node("decision_node", decision_node)
    graph.add_node("chit_chat", chit_chat_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("legal_rag", make_app_node(pipeline))


    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", "decision_node")


    def route_condition(state: LegalState) -> str:

        decision = state.get("route_decision")

        if not decision:
            return "rag_retrieval"

        return decision

    graph.add_conditional_edges(
        "decision_node",
        route_condition,
        {
            "guardrail": "guardrail",
            "chit_chat": "chit_chat",
            "rag_retrieval": "legal_rag",
        }
    )
    graph.add_edge("guardrail", END)
    graph.add_edge("chit_chat", END)
    graph.add_edge("legal_rag", END)


    return graph.compile(checkpointer=checkpointer)


async def stream_ask(app, question: str, session_id: str, user_id: int):
    """Yields ("token", text) for each token, then ("sources", list)."""

    config = {"configurable": {"thread_id": session_id, "user_id": user_id},}
    
    sources = []

    async for mode, data in app.astream(
        {"messages": [HumanMessage(content=question)], "sources": []},
        config=config,
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            chunk,metadata = data

            if metadata.get("langgraph_node") in {"chit_chat", "guardrail", "legal_rag"}:
                
                if isinstance(chunk, (AIMessage, AIMessageChunk)) and chunk.content:
                    yield ("token", chunk.content)

        elif mode == "values":
            sources = data.get("sources", [])

    yield ("sources", sources)



def ask(app, question: str, session_id: str, user_id: int) -> dict:

    result = app.invoke(
        {"messages": [HumanMessage(content=question)], "sources": []},  
        config = {"configurable": {"thread_id": session_id, "user_id": user_id},}
    )
    return {
        "answer": result["messages"][-1].content,
        "sources": result.get("sources", []),  
    }