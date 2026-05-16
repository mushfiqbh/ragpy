from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.responses import HTMLResponse # type: ignore
from pydantic import BaseModel, Field # type: ignore

from pipeline.chunker import chunk_text
from pipeline.embedder import embed
from pipeline.parser import parse_document
from pipeline.qa import answer
from pipeline.store import default_store
from pipeline.drive_service import default_drive_service

app = FastAPI(title="RAG Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_indexed: int
    total_chunks: int


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = Path("index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return index_path.read_text(encoding="utf-8")


@app.get("/health")
async def health():
    stats = default_store.get_stats()
    return {"status": "ok", "stats": stats}


@app.post("/ask")
async def ask(payload: AskRequest):
    stats = default_store.get_stats()
    if stats.get("total_chunks", 0) == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet")

    result = answer(payload.question, top_k=payload.top_k)
    return result


@app.post("/drive-sync")
async def drive_sync(background_tasks: BackgroundTasks):
    """
    Endpoint to trigger recursive download and indexing from the configured Google Drive folder ID.
    Uses DRIVE_FOLDER_ID from environment variables.
    Runs as a background worker job.
    """
    folder_id = os.getenv("DRIVE_FOLDER_ID")
    if not folder_id:
        raise HTTPException(status_code=500, detail="DRIVE_FOLDER_ID not configured in environment")

    # Background job function
    def sync_worker():
        try:
            print(f"Starting background sync for folder: {folder_id}")
            default_drive_service.process_folder(folder_id)
            print("Background sync completed successfully")
        except Exception as e:
            print(f"Background sync failed: {e}")

    background_tasks.add_task(sync_worker)
    
    return {
        "message": "Drive sync job started in background",
        "folder_id": folder_id
    }

