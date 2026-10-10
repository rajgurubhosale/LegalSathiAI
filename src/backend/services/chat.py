from langchain_core.messages import AIMessageChunk


def make_config(user_id, chat_id: str) -> dict:
    return {"configurable": {"thread_id": f"{user_id}:{chat_id}"}}

def stream_answer(rag_app, user_id, chat_id: str, question: str):
    """Yields ("token", text) for each token, then ("sources", list)."""
    
    config = make_config(user_id, str(chat_id))
    sources = []

    for mode, data in rag_app.stream({"question": question}, config, stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk, meta = data
            if meta["langgraph_node"] == "generate" and isinstance(chunk, AIMessageChunk):
                if chunk.content:
                    yield "token", chunk.content
        elif mode == "updates" and "retrieve" in data:
            sources = data["retrieve"]["sources"]

    yield "sources", sources

def list_chats(db, user_id):
    return db.execute(
        "SELECT id::text AS chat_id, title, created_at FROM chats "
        "WHERE user_id = %s ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()


def owns_chat(db, chat_id, user_id) -> bool:
    return db.execute(
        "SELECT 1 FROM chats WHERE id = %s AND user_id = %s",
        (chat_id, user_id),
    ).fetchone() is not None

def save_chat_if_needed(db, chat_id, user_id, question) -> bool:
    db.execute(
        "INSERT INTO chats (id, user_id, title) VALUES (%s, %s, %s) "
        "ON CONFLICT (id) DO NOTHING",
        (chat_id, user_id, " ".join(question.split())[:50]),
    )
    return db.execute(
        "SELECT 1 FROM chats WHERE id = %s AND user_id = %s",
        (chat_id, user_id),
    ).fetchone() is not None

