from langchain_core.messages import AIMessageChunk


def make_config(user_id, chat_id: str) -> dict:
    return {"configurable": {"thread_id": f"{user_id}:{chat_id}"}}

def stream_answer(rag_app, user_id, chat_id: str, question: str):
    """Yields ("token", text) for each token, then ("sources", list)."""
    config = make_config(user_id, str(chat_id))
    sources = []

    for mode, data in rag_app.stream(
        {"question": question}, config, stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            chunk, meta = data
            if meta["langgraph_node"] == "generate" and isinstance(chunk, AIMessageChunk):
                if chunk.content:
                    yield "token", chunk.content
        elif mode == "updates" and "retrieve" in data:
            sources = data["retrieve"]["sources"]

    yield "sources", sources