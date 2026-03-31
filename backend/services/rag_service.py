"""
backend/services/rag_service.py
Core RAG logic — no Streamlit imports.
Ported from study_helper/main.py with Streamlit removed.

Key concepts:
- retrieve_docs(): pgvector similarity search via match_sh_documents() SQL function
- context_is_sufficient(): checks if retrieved docs are good enough to answer
- build_answer(): full RAG pipeline → returns dict
- build_answer_stream(): async generator for SSE streaming
- tavily_search(): web fallback when docs aren't sufficient
"""

import os
import json
import re
from typing import AsyncGenerator
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
from tavily import TavilyClient

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────

def _get_supabase():
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )

def _get_openai():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _get_tavily():
    return TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ── Constants ─────────────────────────────────────────────────────────────────

TOP_K = 10
FETCH_K = 30
EMBEDDING_MODEL = "text-embedding-3-small"  # NEVER CHANGE — 1536 dims

THRESHOLDS = {
    "short":  0.30,
    "medium": 0.35,
    "long":   0.40,
}

LENGTH_INSTRUCTIONS = {
    "short":  "Answer in 2-3 sentences maximum. Be concise.",
    "medium": "Answer in 1-2 clear paragraphs. Moderate detail.",
    "long":   "Answer in detail with examples and full context. Use structured paragraphs.",
}

SYSTEM_RULES = """You are Study Helper, an AI study assistant.
Answer questions based on the student's uploaded study materials.
Always cite sources using [S1], [S2] for file sources and [W1], [W2] for web sources.
Be accurate, educational, and encouraging."""

CANT_ANSWER = [
    "does not contain", "do not contain", "cannot provide",
    "no information", "not found", "not mentioned",
    "unable to find", "couldn't find", "could not find"
]


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    """
    Embed a query using text-embedding-3-small.
    Returns 1536-dim vector. NEVER change this model —
    all stored chunks use 1536 dims.
    """
    client = _get_openai()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_docs(user_id: str, question: str) -> dict:
    """
    Retrieve relevant document chunks for this user.
    Uses match_sh_documents() pgvector SQL function.
    Returns dict with docs list and metadata.
    """
    sb = _get_supabase()
    embedding = embed_query(question)

    result = sb.rpc("match_sh_documents", {
        "query_embedding": embedding,
        "match_count": TOP_K,
        "filter_user_id": user_id,
    }).execute()

    docs = result.data or []
    return {"docs": docs, "count": len(docs)}


def context_is_sufficient(docs: list, mode: str) -> bool:
    """
    Check if retrieved docs are good enough to answer.
    Uses best score + top-3 average — not average of all 10.
    This prevents low-scoring docs from dragging down good ones.
    """
    if not docs:
        return False

    scores = [d.get("similarity", 0) for d in docs if d.get("similarity", 0) > 0]
    if not scores:
        return True  # no score metadata, assume sufficient

    best = max(scores)
    top3_avg = sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores))
    threshold = THRESHOLDS.get(mode, 0.35)

    return best >= threshold or top3_avg >= threshold


# ── Web Search ────────────────────────────────────────────────────────────────

def tavily_search(question: str) -> list[dict]:
    """
    Web search fallback using Tavily API.
    Returns list of {title, url, content} dicts.
    """
    try:
        client = _get_tavily()
        results = client.search(question, max_results=5)
        return results.get("results", [])
    except Exception as e:
        print(f"[rag_service] Tavily error: {e}")
        return []


# ── Context builders ──────────────────────────────────────────────────────────

def build_file_context(docs: list) -> tuple[str, list, list]:
    """
    Build tagged context string from retrieved docs.
    Returns (context_text, file_cites, raw_sources)
    """
    parts = []
    file_cites = []
    raw_sources = []

    for i, doc in enumerate(docs[:TOP_K], 1):
        content = doc.get("content", "")
        source = doc.get("source_file", "Unknown")
        page = doc.get("page_num")
        sim = doc.get("similarity", 0)

        parts.append(f"[S{i}] {content}")
        file_cites.append(f"{source} — Page {page}" if page else source)
        raw_sources.append({
            "source_file": source,
            "page_num": page,
            "similarity": round(sim, 3),
        })

    return "\n\n".join(parts), file_cites, raw_sources


def build_web_context(results: list) -> tuple[str, list]:
    """
    Build tagged context string from web search results.
    Returns (context_text, web_cites)
    """
    parts = []
    web_cites = []

    for i, r in enumerate(results[:5], 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")[:500]

        parts.append(f"[W{i}] {content}")
        web_cites.append(f"{title} — {url}")

    return "\n\n".join(parts), web_cites


def build_history_context(history: list) -> str:
    """
    Build conversation history string for LLM context.
    Only includes if 2+ real exchanges exist — prevents hallucination.
    """
    if not history:
        return ""

    assistant_msgs = [m for m in history if m.get("role") == "assistant"]
    if len(assistant_msgs) < 2:
        return ""

    recent = history[-6:]
    lines = []
    for m in recent:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = m.get("content", "")[:300]
        lines.append(f"{role}: {content}")

    return "CONVERSATION HISTORY:\n" + "\n".join(lines) + "\n\n"


# ── Main RAG pipeline ─────────────────────────────────────────────────────────

def build_answer(
    question: str,
    user_id: str,
    llm,
    mode: str = "medium",
    web_fallback: bool = True,
    language: str = "English",
    history: list = [],
) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve docs
    2. Check if sufficient
    3. If yes → answer from docs
    4. If no → web fallback (if enabled)
    5. Return answer + sources
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    length_instr = LENGTH_INSTRUCTIONS.get(mode, LENGTH_INSTRUCTIONS["medium"])
    system_prompt = f"{SYSTEM_RULES}\n\n{length_instr}"
    history_ctx = build_history_context(history)

    retrieval = retrieve_docs(user_id, question)
    docs = retrieval["docs"]

    file_cites = []
    web_cites = []
    raw_sources = []
    used_web = False
    answer = ""

    if not docs:
        # No docs at all → web fallback
        if web_fallback:
            web_results = tavily_search(question)
            if web_results:
                web_ctx, web_cites = build_web_context(web_results)
                prompt = f"{history_ctx}Web sources:\n{web_ctx}\n\nQuestion: {question}"
                resp = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt)
                ])
                answer = resp.content.strip()
                used_web = True
            else:
                answer = "I couldn't find anything in your documents or on the web."
        else:
            answer = "I couldn't find this in your uploaded documents."

    elif context_is_sufficient(docs, mode):
        # Good docs found → answer from docs
        file_ctx, file_cites, raw_sources = build_file_context(docs)
        prompt = f"{history_ctx}Study materials:\n{file_ctx}\n\nQuestion: {question}"
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        answer = resp.content.strip()

        # Check if LLM says it can't answer despite having docs
        if any(p in answer.lower() for p in CANT_ANSWER) and web_fallback:
            web_results = tavily_search(question)
            if web_results:
                web_ctx, web_cites = build_web_context(web_results)
                prompt = f"{history_ctx}Web sources:\n{web_ctx}\n\nQuestion: {question}"
                resp = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt)
                ])
                answer = "📄 Not found in your docs.\n🌐 From the web:\n\n" + resp.content.strip()
                file_cites = []
                used_web = True

    else:
        # Docs found but not sufficient → web fallback
        if web_fallback:
            web_results = tavily_search(question)
            if web_results:
                web_ctx, web_cites = build_web_context(web_results)
                prompt = f"{history_ctx}Web sources:\n{web_ctx}\n\nQuestion: {question}"
                resp = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt)
                ])
                answer = "📄 Not enough in your docs.\n🌐 From the web:\n\n" + resp.content.strip()
                used_web = True
            else:
                answer = "Couldn't find enough relevant info. Try uploading more study materials."
        else:
            answer = "Couldn't find enough relevant info in your documents."

    # Translate if needed
    if language != "English" and answer:
        answer = translate_answer(answer, language, llm)

    return {
        "answer": answer,
        "file_sources": file_cites,
        "web_sources": web_cites,
        "raw_sources": raw_sources,
        "used_web": used_web,
    }


async def build_answer_stream(
    question: str,
    user_id: str,
    llm,
    mode: str = "medium",
    web_fallback: bool = True,
    language: str = "English",
    history: list = [],
) -> AsyncGenerator[str, None]:
    """
    Streaming version of build_answer.
    Yields SSE-formatted chunks for React frontend.
    
    Format:
    data: {"chunk": "word "}\n\n
    data: {"done": true, "sources": {...}}\n\n
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    length_instr = LENGTH_INSTRUCTIONS.get(mode, LENGTH_INSTRUCTIONS["medium"])
    system_prompt = f"{SYSTEM_RULES}\n\n{length_instr}"
    history_ctx = build_history_context(history)

    retrieval = retrieve_docs(user_id, question)
    docs = retrieval["docs"]
    file_cites, web_cites, raw_sources = [], [], []
    used_web = False

    if docs and context_is_sufficient(docs, mode):
        file_ctx, file_cites, raw_sources = build_file_context(docs)
        prompt = f"{history_ctx}Study materials:\n{file_ctx}\n\nQuestion: {question}"
    elif web_fallback:
        web_results = tavily_search(question)
        if web_results:
            web_ctx, web_cites = build_web_context(web_results)
            prompt = f"{history_ctx}Web sources:\n{web_ctx}\n\nQuestion: {question}"
            used_web = True
        else:
            prompt = question
    else:
        prompt = question

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]

    # Stream chunks
    async for chunk in llm.astream(messages):
        text = chunk.content
        if text:
            yield f"data: {json.dumps({'chunk': text})}\n\n"

    # Final done event with sources
    yield f"data: {json.dumps({'done': True, 'file_sources': file_cites, 'web_sources': web_cites, 'raw_sources': raw_sources, 'used_web': used_web})}\n\n"


def translate_answer(answer: str, language: str, llm) -> str:
    """
    Translate answer to target language.
    Keeps technical terms in English.
    """
    from langchain_core.messages import HumanMessage

    prompt = f"""Translate to {language}. Rules:
1. Simple everyday {language} words
2. Keep technical terms in English (algorithm, DNA, HTTP, RAM)
3. Keep code, formulas, equations in English
4. Natural translation — not word-for-word
5. Never translate text in backticks

Text: {answer}"""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception as e:
        print(f"[rag_service] Translation error: {e}")
        return answer