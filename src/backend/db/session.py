from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from src.backend.core.config import DATABASE_URL

pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=10,
                      kwargs={"row_factory": dict_row}, open=False)

def get_db():
    with pool.connection() as conn:
        yield conn
