from src.genration_pipeline.pipeline_graph import stream_ask


async def stream_answer(rag_app, user_id, chat_id: str, question: str):
    """Yields ("token", text) for each token, then ("sources", list)."""
    async for item in stream_ask(rag_app, question, session_id=f"{user_id}:{chat_id}", user_id=user_id):
        yield item


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
