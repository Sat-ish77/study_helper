"""
main.py — Study Helper v2
RAG pipeline using Supabase pgvector.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient

# ── Conversational Intent Detection ────────────────────────────────────────────

CONVERSATIONAL_TRIGGERS = [
    "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye",
    "how are you", "what are you", "who are you", "good morning",
    "good evening", "good night", "what can you do", "help",
    "ok", "okay", "cool", "great", "nice", "awesome", "got it",
    "sounds good", "perfect", "sure", "alright", "no problem"
]

FOLLOW_UP_TRIGGERS = [
    "go deeper", "deeper", "more detail", "explain more", "clarify",
    "what about", "tell me more", "expand on", "elaborate", "can you explain that",
    "what do you mean", "explain that", "more on this", "continue",
    "more on that", "keep going", "go on",
]

def is_conversational(query: str) -> bool:
    """Detect if query is conversational (greeting, thanks, etc.)"""
    q = query.lower().strip().rstrip("!?.")
    if len(q.split()) <= 4 and any(t in q for t in CONVERSATIONAL_TRIGGERS):
        return True
    return False

def is_follow_up(query: str, history: List[Dict]) -> bool:
    """Detect if query is a follow-up to previous conversation.
    
    STRICT rules — only match when the user clearly refers to the
    previous answer, NOT when they ask a new standalone question.
    """
    if not history or not any(m["role"] == "assistant" for m in history):
        return False

    q = query.lower().strip().rstrip("!?.")
    words = q.split()

    # 1. Explicit follow-up phrases (multi-word, unambiguous)
    if any(trigger in q for trigger in FOLLOW_UP_TRIGGERS):
        return True

    # 2. Very short (≤3 words) pronoun-only questions that clearly
    #    reference the previous answer — e.g. "explain it", "why is that"
    pronoun_refs = {"it", "that", "this", "them", "those", "these"}
    if len(words) <= 3 and any(w in pronoun_refs for w in words):
        return True

    return False

def answer_follow_up(query: str, history: List[Dict], llm) -> str:
    """Handle follow-up questions using previous context without RAG search"""
    try:
        # Build context from last 3-4 exchanges
        recent_history = history[-6:] if len(history) > 6 else history
        
        context_text = "\n\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in recent_history
        ])
        
        prompt = f"""You're continuing a conversation about a study topic.

PREVIOUS CONVERSATION:
{context_text}

FOLLOW-UP QUESTION: {query}

Respond naturally based on the previous context. Don't search for new information.
If the previous context doesn't contain enough information to answer, say so clearly.
Keep your answer focused and helpful."""
        
        return llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception as e:
        print(f"[answer_follow_up] Error: {e}")
        return "I'm having trouble continuing. Could you rephrase your question?"

def answer_conversationally(query: str, llm) -> str:
    """Respond conversationally without searching documents"""
    try:
        prompt = """You are a warm, friendly AI study assistant called Study Helper.
    Respond naturally to this casual message. Keep it to 1-2 sentences.
    Do NOT search any documents. Do NOT mention documents or sources.
    If they seem ready to study, invite them to ask a study question.
    Message: """ + query
        return llm.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        # Fallback responses if LLM fails
        query_lower = query.lower().strip()
        if any(greeting in query_lower for greeting in ["hi", "hello", "hey"]):
            return f"Hello! I'm your Study Helper. How can I help you learn today?"
        elif any(thanks in query_lower for thanks in ["thanks", "thank you"]):
            return "You're welcome! Feel free to ask if you need anything else."
        elif any(bye in query_lower for bye in ["bye", "goodbye"]):
            return "Goodbye! Happy studying!"
        elif "how are you" in query_lower:
            return "I'm doing great and ready to help you study! What would you like to learn about?"
        else:
            return "Hi! I'm here to help you study. Ask me anything from your notes or documents."

# ── Config ────────────────────────────────────────────────────────────────────

TOP_K  =10
FETCH_K = 30

MIN_CONTEXT_CHARS_SHORT = 350
MIN_CONTEXT_CHARS_MED   = 700
MIN_CONTEXT_CHARS_LONG  = 1400

MIN_AVG_SCORE_SHORT = 0.30
MIN_AVG_SCORE_MED   = 0.35
MIN_AVG_SCORE_LONG  = 0.40

WEB_FALLBACK_DEFAULT  = False
TAVILY_MAX_RESULTS    = 5
TAVILY_SEARCH_DEPTH   = "advanced"

SYSTEM_RULES = """You are a study helper.

Hard rules:
- Use ONLY the provided context snippets (Sources [S#] and optional Web [W#]).
- If the context does not contain enough information, say so clearly.
- Do NOT guess, do NOT invent definitions, and do NOT add outside facts unless Web [W#] is provided.
- When you state a fact, cite it using [S#] or [W#] tags inline.

Style:
- Be clear and exam-friendly.
- If the user asks for a long answer, write structured paragraphs + bullet points if helpful.
"""

# ── Embeddings ────────────────────────────────────────────────────────────────

def get_embeddings():
    """
    Always use OpenAI text-embedding-3-small (1536 dims).
    MUST match ingest.py _get_embeddings() — same model, same dimensions.
    Databricks BGE removed — it produces 1024 dims which mismatches stored 1536 dims.
    """
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


# ── Utilities ─────────────────────────────────────────────────────────────────

def detect_length_mode(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ["long answer", "in detail", "detailed", "essay", "500", "800", "in-depth"]):
        return "long"
    if any(x in q for x in ["brief", "short answer", "in 2 lines", "in two lines"]):
        return "short"
    return "medium"


def is_comparison_question(question: str) -> bool:
    q = question.lower()
    patterns = [r"\bvs\b", r"\bversus\b", r"\bcompare\b", r"\bdifference\b",
                r"\bdifferentiate\b", r"\bcontrast\b", r"\bpros and cons\b"]
    return any(re.search(p, q) for p in patterns)


def extract_comparison_terms(question: str) -> Optional[Tuple[str, str]]:
    q = question.strip()
    m = re.search(r"(.+?)\s+\b(vs|versus)\b\s+(.+)", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" ?.,:;"), m.group(3).strip(" ?.,:;")
    m = re.search(r"between\s+(.+?)\s+and\s+(.+)", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" ?.,:;"), m.group(2).strip(" ?.,:;")
    m = re.search(r"(differentiate|difference|compare|contrast)\s+(.+?)\s+and\s+(.+)", q, flags=re.IGNORECASE)
    if m:
        return m.group(2).strip(" ?.,:;"), m.group(3).strip(" ?.,:;")
    return None


def normalize_term(term: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9\s\-_/]", "", term).lower()
    return re.sub(r"\s+", " ", t).strip()


# ── Citations ─────────────────────────────────────────────────────────────────

def format_citation(meta: Dict[str, Any]) -> str:
    filename = meta.get("filename", meta.get("source", "unknown_file"))
    filetype = meta.get("filetype", "")
    if filetype == "pdf" and "page" in meta:
        try:
            page_num = int(meta["page"]) + 1
        except Exception:
            page_num = meta["page"]
        return f"{filename} (page {page_num})"
    if filetype == "pptx" and "slide" in meta:
        return f"{filename} (slide {meta['slide']})"
    return filename


def build_tagged_context(docs: List[Any]) -> Tuple[str, List[str]]:
    parts, citations = [], []
    for idx, d in enumerate(docs, start=1):
        tag  = f"[S{idx}]"
        cite = format_citation(d.metadata if hasattr(d, "metadata") else {})
        citations.append(f"{tag} {cite}")
        parts.append(f"{tag}\n{(d.page_content or '').strip()}")
    return "\n\n".join(parts), citations


def build_tagged_web_context(results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    parts, cites = [], []
    for idx, r in enumerate(results, start=1):
        tag     = f"[W{idx}]"
        url     = r.get("url", "")
        title   = (r.get("title") or "").strip()
        snippet = (r.get("content") or "").strip()[:1200]
        cites.append(f"{tag} {title} — {url}".strip(" —"))
        parts.append(f"{tag}\nTITLE: {title}\nURL: {url}\nSNIPPET: {snippet}")
    return "\n\n".join(parts), cites


# ── Retrieval ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    docs: List[Any]
    avg_score: Optional[float] = None
    context_chars: int = 0


def retrieve_docs(user_id: str, question: str) -> RetrievalResult:
    """
    Retrieve top-K chunks for this user from Supabase pgvector.
    Uses service role key to bypass RLS.
    """
    try:
        embeddings  = get_embeddings()
        embed_query = embeddings.embed_query(question)

        from supabase_client import get_supabase

        def _client():
            return get_supabase()

        supabase = _client()

        response = supabase.rpc(
            "match_sh_documents",
            {
                "query_embedding": embed_query,
                "match_count":     TOP_K,
                "filter_user_id":  user_id,
            }
        ).execute()

        rows = response.data or []

        # Debug — remove after confirming retrieval works
        print(f"[retrieve_docs] user_id={user_id} | rows returned={len(rows)} | embedding_model=text-embedding-3-small")

        docs, scores = [], []
        for row in rows:
            sim = float(row["similarity"]) if "similarity" in row else 0.0
            doc = Document(
                page_content=row.get("content", ""),
                metadata={
                    "filename":  row.get("source_file", ""),
                    "filetype":  row.get("filetype", ""),
                    "page":      row.get("page_num"),
                    "slide":     row.get("slide_num"),
                    "user_id":   row.get("user_id"),
                    "similarity": sim,
                }
            )
            docs.append(doc)
            if sim:
                scores.append(sim)

        context_chars = sum(len(d.page_content) for d in docs)
        avg_score = (sum(scores) / len(scores)) if scores else None
        return RetrievalResult(docs=docs, avg_score=avg_score, context_chars=context_chars)
    except Exception as e:
        print(f"[retrieve_docs] Error: {e}")
        return RetrievalResult(docs=[], avg_score=None, context_chars=0)


# ── Sufficiency checks ────────────────────────────────────────────────────────

def context_is_sufficient(rr: RetrievalResult, length_mode: str) -> bool:
    """Check if retrieved docs are sufficient based on best score or top-3 average."""
    if not rr.docs:
        return False
    # Extract similarity scores from doc metadata
    scores = []
    for d in rr.docs:
        meta = d.metadata if hasattr(d, "metadata") else {}
        sim = meta.get("similarity", 0)
        if sim > 0:
            scores.append(sim)
    # If no scores available, assume sufficient (legacy behavior)
    if not scores:
        return True
    best_score = max(scores)
    top_3_avg = sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores))
    # Thresholds: 0.30/0.35/0.40 for short/medium/long
    thresholds = {"short": 0.30, "medium": 0.35, "long": 0.40}
    threshold = thresholds.get(length_mode, 0.35)
    return best_score >= threshold or top_3_avg >= threshold


def coverage_check_for_comparison(
    question: str, docs: List[Any]
) -> Tuple[bool, Optional[str]]:
    terms = extract_comparison_terms(question)
    if not terms:
        return True, None
    a, b   = terms
    blob   = "\n".join((d.page_content or "").lower() for d in docs)
    has_a  = normalize_term(a) in normalize_term(blob)
    has_b  = normalize_term(b) in normalize_term(blob)
    if has_a and has_b:
        return True, None
    missing = [x.strip() for x, found in [(a, has_a), (b, has_b)] if not found]
    return False, f"Weak coverage for: {', '.join(missing)} in your uploaded files."


# ── Web fallback ──────────────────────────────────────────────────────────────

def tavily_search(query: str) -> List[Dict[str, Any]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        client = TavilyClient(api_key=api_key)
        resp   = client.search(query=query, max_results=TAVILY_MAX_RESULTS,
                               search_depth=TAVILY_SEARCH_DEPTH)
        return resp.get("results", []) if isinstance(resp, dict) else []
    except Exception:
        return []


# ── Prompt builder ────────────────────────────────────────────────────────────

def make_user_prompt(
    question: str,
    length_mode: str,
    file_context: str,
    web_context: Optional[str] = None,
    language: str = "English",
) -> str:
    length_instruction = {
        "short":  "Answer briefly (3–6 bullet points).",
        "medium": "Answer clearly with a few short paragraphs and bullets if helpful.",
        "long":   "Write a long exam-style answer: definition, explanation, comparison/examples, conclusion.",
    }[length_mode]

    lang_note = f"\nRespond in {language}." if language != "English" else ""
    web_block = f"\n\nWEB CONTEXT:\n{web_context}\n" if web_context else ""

    return f"""You must answer using ONLY the provided context.

FILE CONTEXT:
{file_context}
{web_block}

QUESTION:
{question}

{length_instruction}{lang_note}

CITATION RULE: Every important claim must include [S#] or [W#] inline.
If context is insufficient, say exactly what is missing — do not guess.
"""