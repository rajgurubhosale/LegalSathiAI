from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

SUMMARIZE_AFTER = 8   
KEEP_RECENT = 4        


class GraphState(TypedDict, total=False):
    question: str
    messages: Annotated[list, add_messages]  
    summary: str                             
    summarized_count: int                    
    sub_queries: list[str]
    context: str
    sources: list[dict]
    answer: str


def build_graph(pipeline, summarizer) -> StateGraph:
    """summarizer: a small, cheap chat model. Returns an uncompiled graph."""

    def summarize(state: GraphState):
        msgs = state.get("messages", [])
        start = state.get("summarized_count", 0)

        # runs the summarizer only when enough new messages have piled up
        if len(msgs) - start <= SUMMARIZE_AFTER:
            return {}

        cut = len(msgs) - KEEP_RECENT
        text = "".join(
            f"{getattr(m, 'type', 'unknown')}: {getattr(m, 'content', '')}\n"
            for m in msgs[start:cut]
        )
        summary = summarizer.invoke(
            "Update the summary of this conversation in 2-3 lines. Keep names, numbers, "
            "decisions, and Act or section numbers exactly as written.\n\n"
            f"Existing summary:\n{state.get('summary') or '(none)'}\n\n"
            f"New messages:\n{text}"
        ).content
        return {"summary": summary, "summarized_count": cut}

    def decompose(state: GraphState):
        history = state.get("messages", [])[-KEEP_RECENT:]          
        return {"sub_queries": pipeline.query_decomposition(state["question"], history)}

    def retrieve(state: GraphState):
        docs = pipeline.retrieve_context(
            state["question"],
            return_docs=True,                     
            search_queries=state["sub_queries"],  
        )
        return {
            "context": pipeline.format_context(docs),
            "sources": [{"act_name": d.metadata.get("act_name"),
                         "pages": pipeline.document_pages(d)} for d in docs],
        }

        
    def generate(state: GraphState, config: RunnableConfig):
        history = state.get("messages", [])[state.get("summarized_count", 0):]
        if state.get("summary"):
            history = [HumanMessage(content=f"Summary of the earlier conversation:\n{state['summary']}")] + history
        answer = pipeline.chain.invoke({
            "context": state["context"],
            "chat_history": history,
            "question": state["question"],
        }, config)
        return {
            "answer": answer,
            "messages": [HumanMessage(content=state["question"]), AIMessage(content=answer)],
        }

    graph = StateGraph(GraphState)
    graph.add_node("summarize", summarize)
    graph.add_node("decompose", decompose)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", "decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph