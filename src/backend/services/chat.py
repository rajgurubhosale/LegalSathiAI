from src.backend.db.users import fetch_all, fetch_one
from src.genration_pipeline.pipeline_graph import stream_ask


def make_config(user_id, chat_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": f"{user_id}:{chat_id}",
            "user_id": user_id,
        }
    }


async def stream_answer(rag_app, user_id, chat_id: str, question: str):
    """Yields ("token", text) for each token, then ("sources", list)."""
    async for item in stream_ask(rag_app, question, session_id=f"{user_id}:{chat_id}", user_id=user_id):
        yield item


async def list_chats(db, user_id):
    return await fetch_all(
        db,
        """SELECT id::text AS chat_id, title, created_at FROM chats
        WHERE user_id = %s ORDER BY created_at DESC""",
        (user_id,),
    )


async def owns_chat(db, chat_id, user_id) -> bool:
    row = await fetch_one(
        db,
        "SELECT 1 FROM chats WHERE id = %s AND user_id = %s",
        (chat_id, user_id),
    )
    return row is not None


async def save_chat_if_needed(db, chat_id, user_id, question) -> bool:
    await db.execute(
        "INSERT INTO chats (id, user_id, title) VALUES (%s, %s, %s) "
        "ON CONFLICT (id) DO NOTHING",
        (chat_id, user_id, " ".join(question.split())[:50]),
    )
    row = await fetch_one(
        db,
        "SELECT 1 FROM chats WHERE id = %s AND user_id = %s",
        (chat_id, user_id),
    )
    return row is not None
