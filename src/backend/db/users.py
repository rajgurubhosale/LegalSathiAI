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

def create_user(db: Connection,username: str,email: str,hashed_password: str,) -> dict:
    
    return db.execute(
        """
        INSERT INTO users (username, email, hashed_password)
        VALUES (%s, %s, %s)
        RETURNING id, username, email, role
        """, (username, email, hashed_password),
    ).fetchone()




