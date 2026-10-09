from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.backend.api.routers import auth, routes
from src.backend.db.session import pool
from src.genration_pipeline.pipeline import LegalSaathiPipeline
from src.logger import logger



@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Opening database pool...")
        pool.open(wait=True)

        logger.info("Initializing LegalSaathi RAG pipeline...")
        pipeline = LegalSaathiPipeline()
        pipeline.warmup()
        app.state.pipeline = pipeline

        logger.info("LegalSaathi API is ready.")
        yield

    finally:
        logger.info("Shutting down LegalSaathi API.")
        pool.close()

app = FastAPI(title="legal-sathi-api", lifespan=lifespan)

app.include_router(auth.router)       # /auth/register, /auth/login, /auth/me
app.include_router(routes.router)     # public:  /health
app.include_router(routes.protected)  # needs token: /documents, /chat, /chat/stream
