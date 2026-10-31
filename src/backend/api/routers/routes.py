import asyncio
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, HTTPException, Request,
    UploadFile, status
)
from fastapi.responses import FileResponse, StreamingResponse

from src.backend.api.dependencies import CurrentUser, DB, get_current_user
from src.backend.api.schemas import ChatRequest, UploadResponse
from src.backend.api.sse import sse
from src.backend.db.users import (
    clear_user_vault, delete_user_document, get_user_document,
    list_user_documents, savefiles_metadata_in_db,
)
from src.backend.services.chat import (
    list_chats, make_config, owns_chat,
    save_chat_if_needed, stream_answer,
)
from src.backend.services.vault import UserVault
from src.logger import logger

PDF_DIR = Path(__file__).resolve().parents[4] / "PDF_DATA"

router = APIRouter()                                              # public
protected = APIRouter(dependencies=[Depends(get_current_user)])   # needs a token

# public
@router.get("/")
async def welcome():
    return {"msg": "welcome to the legalsathi api"}


@router.get("/health")
async def health():
    return {"status": "ready"}


@protected.get("/documents", response_model=list[str])
async def documents():
    return sorted(pdf.name for pdf in PDF_DIR.glob("*.pdf"))


@protected.get("/chats")
async def recent_chats(user=Depends(get_current_user), db=DB):
    return await asyncio.to_thread(list_chats, db, user["id"])


@protected.post("/chats/draft")
async def new_chat_draft():
    return {"chat_id": str(uuid4())}


@protected.get("/documents/{filename}")
async def read_document(filename: str):
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
async def chat_stream(request: ChatRequest, http: Request, user=CurrentUser, db=DB):

    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be blank")

    if not await asyncio.to_thread(save_chat_if_needed, db, request.chat_id, user["id"], request.question):
        raise HTTPException(status_code=404, detail="Chat not found")
        
    await asyncio.to_thread(db.commit)

    rag_app = http.app.state.rag_app

    async def events():
        try:
            async for kind, payload in stream_answer(rag_app, user["id"], str(request.chat_id), request.question):
                if await http.is_disconnected():
                    logger.info(f"Client disconnected early from chat {request.chat_id}")
                    break
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
async def chat_messages(chat_id: UUID, http: Request, user=CurrentUser, db=DB):

    if not await asyncio.to_thread(owns_chat, db, chat_id, user["id"]):
        raise HTTPException(status_code=404, detail="Chat not found")

    state = await http.app.state.rag_app.aget_state(make_config(user["id"], str(chat_id)))

    history = []

    for msg in state.values.get("messages", []):
        if msg.type not in ("human", "ai") or not msg.content:
            continue

        history.append(
            {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
        )

    return history


VAULT_BASE_DIR = Path("user_data/data_vault")


@protected.post("/upload_documents/", response_model=UploadResponse)
async def uplod_documents(files: list[UploadFile] = File(...), user=CurrentUser, db=DB):
    MAX_FILES = 5
    MAX_SIZE = 20 * 1024 * 1024  # 20 MB
    MAGIC = {".pdf": b"%PDF-", ".docx": b"PK"}

    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Maximum {MAX_FILES} files allowed per upload.")

    folder = VAULT_BASE_DIR / str(user["id"])
    folder.mkdir(parents=True, exist_ok=True)

    uploaded, failed = [], []
    vault = UserVault()

    for file in files:
        name = Path(file.filename or "document.pdf").name
        ext = Path(name).suffix.lower()

        if ext not in MAGIC:
            failed.append(f"{name} (unsupported format)")
            continue

        content = await file.read(MAX_SIZE + 1)
        if len(content) > MAX_SIZE:
            failed.append(f"{name} (exceeds 20 MB)")
            continue

        if not content.startswith(MAGIC[ext]):
            failed.append(f"{name} (corrupted or invalid {ext[1:].upper()})")
            continue

        path = folder / f"{uuid4().hex}{ext}"

        # Per-document atomic processing
        try:
            await asyncio.to_thread(path.write_bytes, content)

            saved_name = await asyncio.to_thread(
                savefiles_metadata_in_db,
                db=db,
                user_id=user["id"],
                file_content={
                    "original_name": name,
                    "stored_path": str(path),
                    "file_size": len(content),
                    "file_type": ext.lstrip("."),
                },
            )

            # Ingest, chunk, and embed into Chroma vault for this document
            res = await asyncio.to_thread(
                vault.process_and_store, file_path=path, user_id=user["id"], doc_name=saved_name
            )
            if res.get("status") == "error":
                raise RuntimeError(res.get("message", "Ingestion failed"))

            # Commit to database only when disk and Chroma both succeed for this file
            await asyncio.to_thread(db.commit)
            uploaded.append(saved_name)

        except Exception as e:
            await asyncio.to_thread(db.rollback)
            path.unlink(missing_ok=True)
            logger.error(f"Failed processing file '{name}': {e}")
            failed.append(f"{name} (interrupted or failed)")
            break

    if not uploaded:
        raise HTTPException(400, f"Upload failed: {', '.join(failed)}")

    msg = f"Successfully uploaded and indexed {len(uploaded)} document(s)"
    if failed:
        msg += f". Interrupted: {', '.join(failed)}"
    return {"message": msg, "uploaded_files": uploaded}


@protected.get("/user/documents")
async def get_user_vault_documents(user=CurrentUser, db=DB):
    """Returns metadata for all documents in the user's personal vault."""
    return await asyncio.to_thread(list_user_documents, db, user["id"])


@protected.get("/user/documents/{doc_id}")
async def stream_user_vault_document(doc_id: str, user=CurrentUser, db=DB):
    """Streams a single document on-demand to protect server and client RAM."""
    doc = await asyncio.to_thread(get_user_document, db, user["id"], doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = Path(doc["stored_path"])
    if not file_path.exists():
        candidate = Path("user_data/data_vault") / str(user["id"]) / file_path.name
        if candidate.exists():
            file_path = candidate
        else:
            candidate2 = Path("data/user_vaults") / str(user["id"]) / file_path.name
            if candidate2.exists():
                file_path = candidate2
            else:
                raise HTTPException(status_code=404, detail="File not found on disk")

    media_type = (
        "application/pdf"
        if doc["file_type"] == "pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=doc["display_name"],
        content_disposition_type="inline",
    )


def _bg_cleanup_document(user_id: int, doc_name: str, stored_path: str):
    """Deletes vector embeddings and server disk file in background."""
    try:
        vault = UserVault()
        vault.delete_document(user_id=user_id, doc_name=doc_name)
    except Exception as e:
        logger.error(f"Error removing vectors for {doc_name}: {e}")
    Path(stored_path).unlink(missing_ok=True)


def _bg_clear_vault(user_id: int, paths: list[str], folder: Path):
    """Clears all user vector embeddings and server directory in background."""
    try:
        vault = UserVault()
        vault.clear_vault(user_id=user_id)
    except Exception as e:
        logger.error(f"Error clearing user vault vectors: {e}")
    for p in paths:
        Path(p).unlink(missing_ok=True)
    if folder.exists():
        import shutil
        shutil.rmtree(folder, ignore_errors=True)


@protected.delete("/user/documents/{doc_id}")
async def delete_vault_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    user=CurrentUser,
    db=DB,
):
    """Deletes metadata from PostgreSQL immediately, then cleans Chroma & disk in background."""
    doc = await asyncio.to_thread(delete_user_document, db, user["id"], doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    await asyncio.to_thread(db.commit)

    # Sync background functions are run in a thread by Starlette automatically
    background_tasks.add_task(
        _bg_cleanup_document,
        user_id=user["id"],
        doc_name=doc["display_name"],
        stored_path=doc["stored_path"],
    )

    return {"message": f"Successfully deleted '{doc['display_name']}' from vault"}


@protected.delete("/user/vault")
async def clear_personal_vault(
    background_tasks: BackgroundTasks,
    user=CurrentUser,
    db=DB,
):
    """Clears all records from PostgreSQL immediately, then cleans Chroma & disk in background."""
    docs = await asyncio.to_thread(clear_user_vault, db, user["id"])
    await asyncio.to_thread(db.commit)

    paths = [d["stored_path"] for d in docs]
    folder = VAULT_BASE_DIR / str(user["id"])

    background_tasks.add_task(_bg_clear_vault, user_id=user["id"], paths=paths, folder=folder)

    return {"message": f"Personal vault cleared ({len(docs)} documents removed)"}