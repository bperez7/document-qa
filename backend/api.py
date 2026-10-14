"""Local HTTP adapter for the existing Document Q&A CLI functions."""

import os
from pathlib import Path
import tempfile
import threading

from fastapi import FastAPI, HTTPException, UploadFile
from openai import APIError, OpenAI
from pydantic import BaseModel, Field

from backend import qa

app = FastAPI(title="Document Q&A")
lock = threading.Lock()
MAX_UPLOAD = 20 * 1024 * 1024
SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def client():
    qa.load_dotenv(qa.BACKEND / ".env", override=False)
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or key == "your-api-key-here":
        raise HTTPException(503, "Set OPENAI_API_KEY in backend/.env and retry.")
    return OpenAI(api_key=key, timeout=60, max_retries=2)


@app.exception_handler(APIError)
async def api_error(request, error):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=502, content={"detail":
        "OpenAI request failed. Check backend credentials, model access, quota, and network. Retry or use CLI cleanup."})


@app.exception_handler(qa.CLIError)
async def processing_error(request, error):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.get("/api/documents")
def documents():
    with lock:
        state = qa.load_state()
        return [{"id": digest, "filename": entry["filename"]}
                for digest, entry in state["documents"].items() if entry.get("ready")]


@app.post("/api/documents")
def upload(file: UploadFile):
    filename = Path((file.filename or "").replace("\\", "/")).name
    if Path(filename).suffix.lower() not in SUPPORTED:
        raise HTTPException(400, "Choose a PDF, DOCX, TXT, or Markdown document.")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / filename
        size = 0
        with path.open("wb") as target:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "Documents must be 20 MB or smaller.")
                target.write(chunk)
        if size == 0:
            raise HTTPException(400, "The document is empty.")
        with lock, client() as sdk:
            state = qa.load_state()
            digest = qa.prepare_document(sdk, state, path)
            return {"id": digest, "filename": state["documents"][digest]["filename"]}


class Question(BaseModel):
    document_id: str
    question: str = Field(min_length=1, max_length=10000)


@app.post("/api/questions")
def question(body: Question):
    if not body.question.strip():
        raise HTTPException(400, "Enter a question.")
    with lock, client() as sdk:
        state = qa.load_state()
        entry = state["documents"].get(body.document_id)
        if not entry or not entry.get("ready") or not state["vector_store_id"]:
            raise HTTPException(404, "Upload this document again to finish processing.")
        response = qa.ask(sdk, state, body.document_id, body.question,
                          os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
        return {"answer": response.output_text or "No answer text returned.",
                "citations": qa.citations(response)}
