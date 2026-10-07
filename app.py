"""
app.py
------
FastAPI server that:
  1. Exposes the Self-RAG pipeline (rag_pipeline.py) as a JSON API.
  2. Serves the static frontend (chat UI) so the whole thing can be
     deployed as a single service.

Run locally:
    uvicorn backend.app:app --reload --port 8000

Then open http://localhost:8000
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import rag_pipeline as rag

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Constitution of India — Self-RAG API",
    description="Ask questions about the Constitution of India, answered by a retrieval-augmented, self-critiquing pipeline.",
    version="1.0.0",
)

# Allow the frontend to call the API from any origin (tighten this in
# production if the frontend is hosted on a different domain).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    verbose: bool = False


class AskResponse(BaseModel):
    answer: str
    decision: str
    scores: dict | None = None
    sources: list[str] = []


@app.on_event("startup")
def startup_event() -> None:
    # Build/load the index and load the LLM in a background thread so the
    # server can start responding to health checks immediately instead of
    # blocking the deploy platform's boot/health-check window.
    threading.Thread(target=_safe_init, daemon=True).start()


def _safe_init() -> None:
    try:
        rag.init_pipeline()
    except Exception:
        # Error is captured in rag.init_error() and surfaced via /api/health
        pass


@app.get("/api/health")
def health():
    return {
        "ready": rag.is_ready(),
        "error": rag.init_error(),
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if not rag.is_ready():
        if rag.init_error():
            raise HTTPException(status_code=500, detail=f"Pipeline failed to initialize: {rag.init_error()}")
        raise HTTPException(status_code=503, detail="Pipeline is still starting up. Please retry in a few seconds.")

    result = rag.self_rag_part(payload.question, verbose=payload.verbose)
    return AskResponse(**result)


# --- Static frontend -------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
