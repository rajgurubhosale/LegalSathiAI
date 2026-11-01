import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.backend.api.routers import auth, routes
from src.backend.core.config import DATABASE_URL
from src.backend.db.session import pool
from src.genration_pipeline.pipeline import LegalSaathiPipeline
from src.logger import logger
from src.genration_pipeline.pipeline_graph import build_graph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_groq import ChatGroq
from src.backend.services.vault import UserVault


@asynccontextmanager
async def lifespan(app: FastAPI):
    await pool.open(wait=True)
    try:
        pipeline = LegalSaathiPipeline()
        pipeline.vault = UserVault(embedding_model=pipeline.retriever.embedding_model)  

        pipeline.warmup()

        summarizer = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=300, max_retries=3)

        async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as saver:
            await saver.setup()
            app.state.pipeline = pipeline
            app.state.rag_app = build_graph(pipeline, summarizer, checkpointer=saver)

            logger.info("LegalSaathi API is ready.")
            yield
    finally:
        await pool.close()


app = FastAPI(title="legal-sathi-api", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(routes.router)
app.include_router(routes.protected)
