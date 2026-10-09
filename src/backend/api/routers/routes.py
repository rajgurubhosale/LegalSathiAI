from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from src.backend.api.dependencies import get_current_user, get_pipeline
from src.backend.api.schemas import ChatRequest, ChatResponse
from src.backend.api.sse import sse
from src.backend.services.chat import build_sources, to_langchain
from src.genration_pipeline.pipeline import LegalSaathiPipeline
from src.logger import logger

PDF_DIR = Path(__file__).resolve().parents[4] / "PDF_DATA"

router = APIRouter()                                              # public
protected = APIRouter(dependencies=[Depends(get_current_user)])   # needs a token

# public
@router.get("/")
def welcome():
    return {"msg": "welcome to the legalsathi api"}


@router.get("/health")
def health():
    return {"status": "ready"}


#protected
@protected.get("/documents", response_model=list[str])
def documents():
    return sorted(pdf.name for pdf in PDF_DIR.glob("*.pdf"))


@protected.get("/documents/{filename}")
def read_document(filename: str):
    document = (PDF_DIR / filename).resolve()
    if (
        "/" in filename or "\\" in filename
        or document.parent != PDF_DIR.resolve()
        or document.suffix.lower() != ".pdf"
        or not document.is_file()
    ):
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(document, media_type="application/pdf", filename=filename, content_disposition_type="inline")

@protected.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, pipeline: LegalSaathiPipeline = Depends(get_pipeline)):
    history = to_langchain(request.chat_history)
    result = pipeline.run(request.question, chat_history=history)
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        chat_history=[
            {"role": "user" if message.type == "human" else "assistant", "content": message.content}
            for message in result["chat_history"]
        ],
    )


@protected.post("/chat/stream")
def chat_stream(request: ChatRequest, pipeline: LegalSaathiPipeline = Depends(get_pipeline)):
    history = to_langchain(request.chat_history)

    def events():
        try:
            docs = pipeline.retrieve_context(request.question, return_docs=True, chat_history=history)
            context = pipeline.format_context(docs)
            for token in pipeline.generate_stream(request.question, context, history):
                yield sse(token)
            yield sse(build_sources(pipeline, docs), event="sources")
            yield sse("done", event="done")
        except Exception as e:
            logger.error(f"Stream failed: {e}")
            yield sse("Something went wrong. Please try again.", event="error")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
