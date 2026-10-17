from pathlib import Path
from psycopg import Connection


def get_user_by_email(db: Connection, email: str) -> dict | None:
    return db.execute(
        "SELECT id, hashed_password, is_active FROM users WHERE email = %s",
        (email,),
    ).fetchone()

def get_user_by_id(db: Connection, user_id: int) -> dict | None:
    return db.execute(
        "SELECT id, username, email, role, is_active FROM users WHERE id = %s",
        (user_id,),


    ).fetchone()

def create_user(db: Connection,username: str,email: str,hashed_password: str,) -> dict | None:
    
    return db.execute(
        """
        INSERT INTO users (username, email, hashed_password)
        VALUES (%s, %s, %s)
        RETURNING id, username, email, role
        """, (username, email, hashed_password),
    ).fetchone()


def savefiles_metadata_in_db(db: Connection, user_id: int, file_content: dict) -> str:
    original_name = file_content["original_name"]
    stored_path = file_content["stored_path"]
    file_size = file_content["file_size"]

    path_obj = Path(original_name)
    base_name = path_obj.stem
    extension = path_obj.suffix.lower()                   
    file_type = extension.lstrip(".") or "unknown"       

    existing = db.execute(
        "SELECT display_name FROM user_documents WHERE user_id = %s AND display_name LIKE %s",
        (user_id, f"{base_name}%{extension}"),
    ).fetchall()

    if existing:
        clean_display_name = f"{base_name} ({len(existing)}){extension}"
    else:
        clean_display_name = original_name

    db.execute(
        """
        INSERT INTO user_documents (user_id, display_name, stored_path, file_size, file_type)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (user_id, clean_display_name, stored_path, file_size, file_type),
    )

    return clean_display_name


def list_user_documents(db: Connection, user_id: int) -> list[dict]:
    return db.execute(
        """
        SELECT id::text AS id, display_name, file_size, file_type, uploaded_at
        FROM user_documents
        WHERE user_id = %s
        ORDER BY uploaded_at DESC
        """,
        (user_id,),
    ).fetchall()


def get_user_document(db: Connection, user_id: int, doc_id: str) -> dict | None:
    return db.execute(
        """
        SELECT id::text AS id, display_name, stored_path, file_type
        FROM user_documents
        WHERE id = %s AND user_id = %s
        """,
        (doc_id, user_id),
    ).fetchone()


def delete_user_document(db: Connection, user_id: int, doc_id: str) -> dict | None:
    return db.execute(
        """
        DELETE FROM user_documents
        WHERE id = %s AND user_id = %s
        RETURNING id::text AS id, display_name, stored_path
        """,
        (doc_id, user_id),
    ).fetchone()


def clear_user_vault(db: Connection, user_id: int) -> list[dict]:
    return db.execute(
        """
        DELETE FROM user_documents
        WHERE user_id = %s
        RETURNING id::text AS id, display_name, stored_path
        """,
        (user_id,),
    ).fetchall()

