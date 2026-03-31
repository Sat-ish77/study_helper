"""
backend/routers/rag.py
RAG endpoints — the core of Study Helper.

POST /api/v1/rag/ask          → standard Q&A (returns full answer)
POST /api/v1/rag/ask/stream   → streaming Q&A (SSE, word by word)
POST /api/v1/rag/ingest       → upload file, chunk, embed, store
DELETE /api/v1/rag/documents/{source_file} → delete user's document

How RAG works:
1. User asks a question
2. We embed the question using text-embedding-3-small (1536 dims)
3. We search sh_document_chunks for similar chunks (pgvector)
4. We build a prompt with the retrieved chunks as context
5. LLM generates an answer grounded in the docs
6. If docs aren't sufficient, optionally fall back to Tavily web search
"""

import os
import io
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dependencies import get_current_user, get_db
from services.rag_service import (
    retrieve_docs,
    build_answer,
    build_answer_stream,
    tavily_search,
    context_is_sufficient,
)
from model_manager import get_llm

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    model: str = "Llama 3.3 70B"
    mode: str = "medium"          # short / medium / long
    web_fallback: bool = True
    language: str = "English"
    history: list = []            # last N exchanges for context


class AskResponse(BaseModel):
    answer: str
    file_sources: list
    web_sources: list
    raw_sources: list
    used_web: bool


# ── Standard Q&A ─────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Main RAG Q&A endpoint.
    Retrieves relevant document chunks for this user,
    builds context, calls LLM, returns answer + sources.
    """
    llm = get_llm(body.model)
    result = build_answer(
        question=body.question,
        user_id=user_id,
        llm=llm,
        mode=body.mode,
        web_fallback=body.web_fallback,
        language=body.language,
        history=body.history,
    )
    return AskResponse(**result)


# ── Streaming Q&A (SSE) ───────────────────────────────────────────────────────

@router.post("/ask/stream")
async def ask_stream(
    body: AskRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Streaming RAG Q&A using Server-Sent Events.
    React frontend reads chunks as they arrive — ChatGPT-style.
    
    Response format:
    data: {"chunk": "word "}\n\n
    data: {"chunk": "by "}\n\n
    data: {"done": true, "sources": [...]}\n\n
    """
    llm = get_llm(body.model)

    async def generate():
        async for event in build_answer_stream(
            question=body.question,
            user_id=user_id,
            llm=llm,
            mode=body.mode,
            web_fallback=body.web_fallback,
            language=body.language,
            history=body.history,
        ):
            yield event

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # needed for nginx/Cloud Run
        }
    )


# ── File Ingest ───────────────────────────────────────────────────────────────

@router.post("/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    """
    Upload a PDF, DOCX, or PPTX file.
    Extracts text (Mistral OCR for PDFs), chunks it,
    embeds each chunk with text-embedding-3-small,
    stores in sh_document_chunks with user_id.
    
    Returns: {filename, chunks_created, status}
    """
    from services.ingest_service import process_file

    allowed = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ]

    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use PDF, DOCX, or PPTX."
        )

    file_bytes = await file.read()
    result = await process_file(
        file_bytes=file_bytes,
        filename=file.filename,
        content_type=file.content_type,
        user_id=user_id,
    )
    return result


# ── Delete Document ───────────────────────────────────────────────────────────

@router.delete("/documents/{source_file}")
async def delete_document(
    source_file: str,
    user_id: str = Depends(get_current_user)
):
    """
    Delete all chunks for a specific file belonging to this user.
    Only deletes chunks where user_id matches — users can't delete each other's docs.
    """
    sb = get_db()
    result = sb.table("sh_document_chunks")\
        .delete()\
        .eq("user_id", user_id)\
        .eq("source_file", source_file)\
        .execute()

    return {"deleted": True, "source_file": source_file}


# ── List Documents ────────────────────────────────────────────────────────────

@router.get("/documents")
async def list_documents(
    user_id: str = Depends(get_current_user)
):
    """
    List all unique source files uploaded by this user.
    """
    sb = get_db()
    result = sb.table("sh_document_chunks")\
        .select("source_file, filetype")\
        .eq("user_id", user_id)\
        .execute()

    # Deduplicate
    seen = set()
    docs = []
    for row in (result.data or []):
        sf = row["source_file"]
        if sf not in seen:
            seen.add(sf)
            docs.append({"source_file": sf, "filetype": row.get("filetype", "")})

    return {"documents": docs, "count": len(docs)}