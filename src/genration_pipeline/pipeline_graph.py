from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

HISTORY_WINDOW = 8


class GraphState(TypedDict, total=False):
    question: str
    messages: Annotated[list, add_messages]
    sub_queries: list[str]
    context: str
    sources: list[dict]
    answer: str


def build_graph(pipeline) -> StateGraph:
    """Nodes use the pipeline passed in here. Returns an uncompiled graph."""

    def decompose(state: GraphState):
        history = state.get("messages", [])[-HISTORY_WINDOW:]
        return {"sub_queries": pipeline.query_decomposition(state["question"], history)}


    def retrieve(state):
        docs = pipeline.retrieve_context(
            state["question"],
            return_docs=True,                    # returns Document objects, not a string
            search_queries=state["sub_queries"], # skips decomposing a second time
        )
        return {
            "context": pipeline.format_context(docs),
            "sources": [{"act_name": d.metadata.get("act_name"),
                        "pages": pipeline.document_pages(d)} for d in docs],
        }
        
    def generate(state, config: RunnableConfig):

            

        answer = pipeline.chain.invoke({
            "context": state["context"],
            "chat_history": state.get("messages", [])[-HISTORY_WINDOW:],
            "question": state["question"],
        }, config)

        return {"answer": answer,
                "messages": [HumanMessage(content=state["question"]), AIMessage(content=answer)]}


    graph = StateGraph(GraphState)
    graph.add_node("decompose", decompose)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph