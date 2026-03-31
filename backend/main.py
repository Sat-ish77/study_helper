"""
backend/main.py — Study Helper v2 FastAPI Backend
Entry point. Mounts all routers, sets up CORS, middleware.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from routers import rag, speech, quiz, flashcards, canvas, conversations, dashboard, models

app = FastAPI(
    title="Study Helper API",
    description="RAG-powered study assistant backend",
    version="2.0.0"
)

# ── CORS — allow React frontend ───────────────────────────────────────────────
# In production replace * with your Vercel domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ─────────────────────────────────────────────────────────────
app.include_router(rag.router,           prefix="/api/v1/rag",           tags=["RAG"])
app.include_router(speech.router,        prefix="/api/v1/speech",        tags=["Speech"])
app.include_router(quiz.router,          prefix="/api/v1/quiz",          tags=["Quiz"])
app.include_router(flashcards.router,    prefix="/api/v1/flashcards",    tags=["Flashcards"])
app.include_router(canvas.router,        prefix="/api/v1/canvas",        tags=["Canvas"])
app.include_router(conversations.router, prefix="/api/v1/conversations", tags=["Conversations"])
app.include_router(dashboard.router,     prefix="/api/v1/dashboard",     tags=["Dashboard"])
app.include_router(models.router,        prefix="/api/v1/models",        tags=["Models"])

@app.get("/")
def root():
    return {"status": "ok", "app": "Study Helper API v2"}

@app.get("/health")
def health():
    return {"status": "healthy"}