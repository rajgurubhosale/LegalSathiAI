from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException,Request
from fastapi.responses import FileResponse, StreamingResponse
from src.backend.services.chat import stream_answer
from src.backend.api.dependencies import get_current_user, get_pipeline
from src.backend.api.schemas import ChatRequest, ChatResponse
from src.backend.api.sse import sse
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



@protected.post("/chat/stream")
def chat_stream(request: ChatRequest, http: Request, user=Depends(get_current_user)):
    rag_app = http.app.state.rag_app

    def events():
        try:
            for kind, payload in stream_answer(rag_app,user["id"],  str(request.chat_id), request.question):
                if kind == "token":
                    yield sse(payload)
                else:
                    yield sse(payload, event="sources")
            yield sse("done", event="done")
        except Exception as e:
            logger.error(f"Stream failed: {e}")
            yield sse("Something went wrong. Please try again.", event="error")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )