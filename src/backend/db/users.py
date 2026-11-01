from pathlib import Path
from psycopg import AsyncConnection


async def fetch_one(db: AsyncConnection, sql: str, params: tuple = ()) -> dict | None:
    cur = await db.execute(sql, params)
    return await cur.fetchone()
 
 
async def fetch_all(db: AsyncConnection, sql: str, params: tuple = ()) -> list[dict]:
    cur = await db.execute(sql, params)
    return await cur.fetchall()
 


async def get_user_by_email(db: AsyncConnection, email: str) -> dict | None:
    return await fetch_one(
        db,
        "SELECT id, hashed_password, is_active FROM users WHERE email = %s",
        (email,),
    )

async def get_user_by_id(db: AsyncConnection, user_id: int) -> dict | None:
    return await fetch_one(
        db,
        "SELECT id, username, email, role, is_active FROM users WHERE id = %s",
        (user_id,),
    )
 

async def create_user(
    db: AsyncConnection, username: str, email: str, hashed_password: str
) -> dict | None:
    return await fetch_one(
        db,
        """
        INSERT INTO users (username, email, hashed_password)
        VALUES (%s, %s, %s)
        RETURNING id, username, email, role
        """,
        (username, email, hashed_password),
    )


async def savefiles_metadata_in_db(
    db: AsyncConnection, user_id: int, file_content: dict
) -> str:
    """Save a file record. If the name is taken, use 'name (1).ext', 'name (2).ext'..."""
    original_name = file_content["original_name"]
    path = Path(original_name)
 
    rows = await fetch_all(
        db, "SELECT display_name FROM user_documents WHERE user_id = %s", (user_id,)
    )
    taken = {row["display_name"] for row in rows}
 
    display_name, n = original_name, 1
    while display_name in taken:
        display_name = f"{path.stem} ({n}){path.suffix}"
        n += 1
 
    await db.execute(
        """
        INSERT INTO user_documents (user_id, display_name, stored_path, file_size, file_type)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            user_id,
            display_name,
            file_content["stored_path"],
            file_content["file_size"],
            path.suffix.lower().lstrip(".") or "unknown",
        ),
    )
    return display_name
 
 
async def list_user_documents(db: AsyncConnection, user_id: int) -> list[dict]:
    return await fetch_all(
        db,
        """
        SELECT id::text AS id, display_name, file_size, file_type, uploaded_at
        FROM user_documents
        WHERE user_id = %s
        ORDER BY uploaded_at DESC
        """,
        (user_id,),
    )
 
 
async def get_user_document(db: AsyncConnection, user_id: int, doc_id: str) -> dict | None:
    return await fetch_one(
        db,
        """
        SELECT id::text AS id, display_name, stored_path, file_type
        FROM user_documents
        WHERE id = %s AND user_id = %s
        """,
        (doc_id, user_id),
    )
 
 
async def delete_user_document(db: AsyncConnection, user_id: int, doc_id: str) -> dict | None:
    return await fetch_one(
        db,
        """
        DELETE FROM user_documents
        WHERE id = %s AND user_id = %s
        RETURNING id::text AS id, display_name, stored_path
        """,
        (doc_id, user_id),
    )
 
 
async def clear_user_vault(db: AsyncConnection, user_id: int) -> list[dict]:
    return await fetch_all(
        db,
        """
        DELETE FROM user_documents
        WHERE user_id = %s
        RETURNING id::text AS id, display_name, stored_path
        """,
        (user_id,),
    )
 