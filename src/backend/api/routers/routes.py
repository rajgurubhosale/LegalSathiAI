from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException,Request
from fastapi.responses import FileResponse, StreamingResponse
from src.backend.services.chat import stream_answer
from src.backend.api.dependencies import get_current_user, get_pipeline,get_db
from src.backend.api.sse import sse

from src.logger import logger
from src.backend.api.schemas import ChatRequest
from uuid import UUID, uuid4
from src.backend.services.chat import save_chat_if_needed, list_chats, owns_chat
from src.backend.services.chat import make_config
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


@protected.get("/documents", response_model=list[str])
def documents():
    return sorted(pdf.name for pdf in PDF_DIR.glob("*.pdf"))


@protected.get("/chats")
def recent_chats(user=Depends(get_current_user), db=Depends(get_db)):
    return list_chats(db, user["id"])


@protected.post("/chats/draft")
def new_chat_draft():
    return {"chat_id": str(uuid4())}



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
def chat_stream(request: ChatRequest, http: Request,user=Depends(get_current_user), db=Depends(get_db)):

    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be blank")
    if not save_chat_if_needed(db, request.chat_id, user["id"], request.question):
        raise HTTPException(status_code=404, detail="Chat not found")
    db.commit()

    rag_app = http.app.state.rag_app

    def events():
        try:
            for kind, payload in stream_answer(rag_app, user["id"], str(request.chat_id), request.question):
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

@protected.get("/chats/{chat_id}/messages")
def chat_messages(chat_id: UUID, http: Request,user=Depends(get_current_user), db=Depends(get_db)):

    if not owns_chat(db, chat_id, user["id"]):
        raise HTTPException(status_code=404, detail="Chat not found")

    state = http.app.state.rag_app.get_state(make_config(user["id"], str(chat_id)))
    
    history = []

    for msg in state.values.get("messages",[]):

        if msg.type not in ("human", "ai"):
            continue
        history.append(
            {"role":"user" if msg.type == "human" else "assistant", "content": msg.content}

        )
        
    return history
