from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from src.backend.core.config import DATABASE_URL

pool = AsyncConnectionPool(DATABASE_URL, min_size=1, max_size=10,
                      kwargs={"row_factory": dict_row}, open=False)

async def get_db():
    async with pool.connection() as conn:
        yield conn
 