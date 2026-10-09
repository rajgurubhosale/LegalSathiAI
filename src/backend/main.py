from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.backend.api.routers import auth, routes
from src.backend.db.session import pool
from src.genration_pipeline.pipeline import LegalSaathiPipeline
from src.logger import logger
from langgraph.checkpoint.memory import InMemorySaver
from src.genration_pipeline.pipeline_graph import *




@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        pool.open(wait=True)

        pipeline = LegalSaathiPipeline()      # created once, here only
        pipeline.warmup()                     # models load here, not in compile

        app.state.rag_app = build_graph(pipeline).compile(checkpointer=InMemorySaver())


        logger.info("LegalSaathi API is ready.")
        yield
    finally:
        pool.close()

app = FastAPI(title="legal-sathi-api", lifespan=lifespan)

app.include_router(auth.router)       # /auth/register, /auth/login, /auth/me
app.include_router(routes.router)     # public:  /health
app.include_router(routes.protected)  # needs token: /documents, /chat, /chat/stream
