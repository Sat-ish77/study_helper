from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

# Tavily (web fallback)
from tavily import TavilyClient


# -----------------------
# Config
# -----------------------
DB_DIR = "vectordb"

DEFAULT_MODEL = "gpt-4o"
TEMPERATURE = 0

# Retrieval
TOP_K = 8                 # how many chunks we finally use
FETCH_K = 24              # how many we fetch before MMR selects diverse subset
USE_MMR = True

# Safety thresholds (tune later if you want)
MIN_CONTEXT_CHARS_SHORT = 350
MIN_CONTEXT_CHARS_MED = 700
MIN_CONTEXT_CHARS_LONG = 1400

# relevance score thresholds (higher = stricter, only truly relevant chunks pass)
# Chroma scores typically range 0.0–1.0 where higher is more relevant
# Tested: irrelevant queries can score ~0.64, so we set bar at 0.70+
MIN_AVG_SCORE_SHORT = 0.65
MIN_AVG_SCORE_MED = 0.70
MIN_AVG_SCORE_LONG = 0.75

# Web fallback defaults
WEB_FALLBACK_DEFAULT = False
TAVILY_MAX_RESULTS = 5
TAVILY_SEARCH_DEPTH = "advanced"


SYSTEM_RULES = """You are a study helper.

Hard rules:
- Use ONLY the provided context snippets (Sources [S#] and optional Web [W#]).
- If the context does not contain enough information, say so clearly.
- Do NOT guess, do NOT invent definitions, and do NOT add outside facts unless Web [W#] is provided.
- When you state a fact, cite it using [S#] or [W#] tags inline.

Style:
- Be clear and exam-friendly.
- If the user asks for a "long answer", write structured paragraphs + bullet points if helpful.
"""


# -----------------------
# Utilities: question intent
# -----------------------
def detect_length_mode(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ["long answer", "in detail", "detailed", "essay", "500", "800", "in-depth", "in depth"]):
        return "long"
    if any(x in q for x in ["brief", "short answer", "in 2 lines", "in two lines"]):
        return "short"
    return "medium"


def is_comparison_question(question: str) -> bool:
    q = question.lower()
    patterns = [
        r"\bvs\b", r"\bversus\b", r"\bcompare\b", r"\bdifference\b", r"\bdifferentiate\b",
        r"\bcontrast\b", r"\bpros and cons\b"
    ]
    return any(re.search(p, q) for p in patterns)


def extract_comparison_terms(question: str) -> Optional[Tuple[str, str]]:
    """
    Best-effort extraction of (A, B) for questions like:
    - "ReLU vs Sigmoid"
    - "difference between ReLU and Sigmoid"
    - "differentiate relu and sigmoid"
    """
    q = question.strip()

    # A vs B
    m = re.search(r"(.+?)\s+\b(vs|versus)\b\s+(.+)", q, flags=re.IGNORECASE)
    if m:
        a = m.group(1).strip(" ?.,:;")
        b = m.group(3).strip(" ?.,:;")
        return (a, b)

    # difference between A and B
    m = re.search(r"between\s+(.+?)\s+and\s+(.+)", q, flags=re.IGNORECASE)
    if m:
        a = m.group(1).strip(" ?.,:;")
        b = m.group(2).strip(" ?.,:;")
        return (a, b)

    # differentiate A and B
    m = re.search(r"(differentiate|difference|compare|contrast)\s+(.+?)\s+and\s+(.+)", q, flags=re.IGNORECASE)
    if m:
        a = m.group(2).strip(" ?.,:;")
        b = m.group(3).strip(" ?.,:;")
        return (a, b)

    return None


def normalize_term(term: str) -> str:
    # keep it simple: remove punctuation, collapse spaces, lowercase
    t = re.sub(r"[^a-zA-Z0-9\s\-_/]", "", term).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


# -----------------------
# Utilities: citations
# -----------------------
def format_citation(meta: Dict[str, Any]) -> str:
    filename = meta.get("filename", meta.get("source", "unknown_file"))
    filetype = meta.get("filetype", "unknown_type")

    if filetype == "pdf" and "page" in meta:
        try:
            page_num = int(meta["page"]) + 1  # PyPDFLoader sometimes 0-based
        except Exception:
            page_num = meta["page"]
        return f"{filename} (page {page_num})"

    if filetype == "pptx" and "slide" in meta:
        return f"{filename} (slide {meta['slide']})"

    return f"{filename}"


def build_tagged_context(docs: List[Any]) -> Tuple[str, List[str]]:
    """
    Returns:
      context_text: chunks with [S1], [S2] tags
      citations: list where citations[i] corresponds to [S{i+1}]
    """
    parts: List[str] = []
    citations: List[str] = []

    for idx, d in enumerate(docs, start=1):
        tag = f"[S{idx}]"
        cite = format_citation(d.metadata if hasattr(d, "metadata") else {})
        citations.append(f"{tag} {cite}")
        text = (d.page_content or "").strip()
        parts.append(f"{tag}\n{text}")

    return "\n\n".join(parts), citations


def build_tagged_web_context(results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    parts: List[str] = []
    cites: List[str] = []
    for idx, r in enumerate(results, start=1):
        tag = f"[W{idx}]"
        url = r.get("url", "")
        title = (r.get("title") or "").strip()
        snippet = (r.get("content") or "").strip()
        # Keep snippet reasonably short
        snippet = snippet[:1200]
        cites.append(f"{tag} {title} — {url}".strip(" —"))
        parts.append(f"{tag}\nTITLE: {title}\nURL: {url}\nSNIPPET: {snippet}")
    return "\n\n".join(parts), cites


# -----------------------
# Retrieval + safety checks
# -----------------------
@dataclass
class RetrievalResult:
    docs: List[Any]
    avg_score: Optional[float] = None
    context_chars: int = 0


def retrieve_docs(vectordb: Chroma, question: str) -> RetrievalResult:
    # Prefer relevance scoring if available
    docs: List[Any] = []
    scores: List[float] = []

    # Try to use relevance scores (not always available on every vectorstore/version)
    try:
        pairs = vectordb.similarity_search_with_relevance_scores(question, k=TOP_K)
        for d, s in pairs:
            docs.append(d)
            scores.append(float(s))
    except Exception:
        # Fallback: retriever without scores
        if USE_MMR:
            retriever = vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": 0.5},
            )
        else:
            retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
        docs = retriever.get_relevant_documents(question)

    context_chars = sum(len((d.page_content or "")) for d in docs)
    avg_score = (sum(scores) / len(scores)) if scores else None
    return RetrievalResult(docs=docs, avg_score=avg_score, context_chars=context_chars)


def context_is_sufficient(rr: RetrievalResult, length_mode: str) -> bool:
    # If we couldn't measure relevance scores, we can't trust the results
    # Be conservative: assume insufficient and let web fallback handle it
    if rr.avg_score is None:
        return False

    if length_mode == "short":
        if rr.context_chars < MIN_CONTEXT_CHARS_SHORT:
            return False
        if rr.avg_score < MIN_AVG_SCORE_SHORT:
            return False
        return True

    if length_mode == "long":
        if rr.context_chars < MIN_CONTEXT_CHARS_LONG:
            return False
        if rr.avg_score < MIN_AVG_SCORE_LONG:
            return False
        return True

    # medium
    if rr.context_chars < MIN_CONTEXT_CHARS_MED:
        return False
    if rr.avg_score < MIN_AVG_SCORE_MED:
        return False
    return True


def coverage_check_for_comparison(question: str, docs: List[Any]) -> Tuple[bool, Optional[str]]:
    """
    If the question is A vs B, ensure the retrieved context mentions both A and B somewhere.
    If not, return (False, reason).
    """
    terms = extract_comparison_terms(question)
    if not terms:
        return True, None

    a, b = terms
    a_norm, b_norm = normalize_term(a), normalize_term(b)

    blob = "\n".join((d.page_content or "").lower() for d in docs)

    has_a = a_norm and a_norm in normalize_term(blob)
    has_b = b_norm and b_norm in normalize_term(blob)

    if has_a and has_b:
        return True, None

    missing = []
    if not has_a:
        missing.append(a.strip())
    if not has_b:
        missing.append(b.strip())

    return (
        False,
        f"I found weak/no coverage for: {', '.join(missing)} in your uploaded files.",
    )


# -----------------------
# Web fallback
# -----------------------
def tavily_search(query: str) -> List[Dict[str, Any]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    client = TavilyClient(api_key=api_key)
    try:
        resp = client.search(
            query=query,
            max_results=TAVILY_MAX_RESULTS,
            search_depth=TAVILY_SEARCH_DEPTH,
        )
        return resp.get("results", []) if isinstance(resp, dict) else []
    except Exception:
        return []


# -----------------------
# Answering
# -----------------------
def make_user_prompt(
    question: str,
    length_mode: str,
    file_context: str,
    web_context: Optional[str] = None,
) -> str:
    length_instruction = {
        "short": "Answer briefly (3–6 bullet points).",
        "medium": "Answer clearly with a few short paragraphs and bullets if helpful.",
        "long": "Write a long exam-style answer with structure: definition, explanation, comparison/examples, and a short conclusion.",
    }[length_mode]

    web_block = f"\n\nWEB CONTEXT:\n{web_context}\n" if web_context else ""

    return f"""You must answer using ONLY the provided context.

FILE CONTEXT:
{file_context}
{web_block}

QUESTION:
{question}

{length_instruction}

CITATION RULE:
- Every important claim must include a citation tag like [S1] or [W2] at the end of the sentence.
- If the context is insufficient, say exactly what is missing and do not guess.
"""


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in your .env file.")

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=TEMPERATURE)

    web_fallback = WEB_FALLBACK_DEFAULT

    print(" Study Helper (files-first) ready.")
    print("Commands:")
    print("  :web on   -> enable web fallback (needs TAVILY_API_KEY)")
    print("  :web off  -> disable web fallback")
    print("  :mode short|medium|long -> default answer length")
    print("  exit      -> quit\n")

    default_mode = "medium"

    while True:
        raw = input("Q: ").strip()
        if not raw:
            continue

        if raw.lower() in {"exit", "quit"}:
            break

        # Commands
        if raw.startswith(":web "):
            val = raw.split(":web ", 1)[1].strip().lower()
            web_fallback = (val == "on")
            print(f" Web fallback is now: {'ON' if web_fallback else 'OFF'}\n")
            continue

        if raw.startswith(":mode "):
            val = raw.split(":mode ", 1)[1].strip().lower()
            if val in {"short", "medium", "long"}:
                default_mode = val
                print(f" Default mode is now: {default_mode}\n")
            else:
                print(" Mode must be: short, medium, or long\n")
            continue

        question = raw
        length_mode = detect_length_mode(question) or default_mode
        # If user didn't explicitly ask, use default
        if detect_length_mode(question) == "medium":
            length_mode = default_mode

        # 1) Retrieve from files
        rr = retrieve_docs(vectordb, question)

        # Debug: see relevance scores (remove this line later if you want)
        score_str = f"{rr.avg_score:.3f}" if rr.avg_score is not None else "None"
        print(f"[DEBUG] {len(rr.docs)} chunks, {rr.context_chars} chars, avg_score={score_str}")

        if not rr.docs:
            # no retrieval at all
            if web_fallback:
                web_results = tavily_search(question)
                if not web_results:
                    print("\nA: I can't find anything in your uploaded files, and web search is unavailable.\n")
                    continue
                web_context, web_cites = build_tagged_web_context(web_results)
                user_prompt = make_user_prompt(question, length_mode, file_context="(No file context found)", web_context=web_context)
                resp = llm.invoke([SystemMessage(content=SYSTEM_RULES), HumanMessage(content=user_prompt)])
                print("\nA:", resp.content.strip(), "\n")
                print("Web Sources:")
                for c in web_cites:
                    print("-", c)
                print()
                continue

            print("\nA: I can't find this in your uploaded files.\n")
            continue

        # 2) Comparison coverage check
        if is_comparison_question(question):
            ok, reason = coverage_check_for_comparison(question, rr.docs)
            if not ok:
                if web_fallback:
                    # try web to fill the missing concept
                    web_results = tavily_search(question)
                    web_context, web_cites = build_tagged_web_context(web_results) if web_results else ("", [])
                    file_context, file_cites = build_tagged_context(rr.docs)

                    # If web is still empty, be honest
                    if not web_results:
                        print("\nA:", reason)
                        print("Tip: enable Tavily key or upload notes covering the missing topic.\n")
                        continue

                    user_prompt = make_user_prompt(question, length_mode, file_context=file_context, web_context=web_context)
                    resp = llm.invoke([SystemMessage(content=SYSTEM_RULES), HumanMessage(content=user_prompt)])

                    print("\nA:", resp.content.strip(), "\n")
                    print("File Sources:")
                    for c in file_cites:
                        print("-", c)
                    print("\nWeb Sources:")
                    for c in web_cites:
                        print("-", c)
                    print()
                    continue

                # No web: refuse to guess
                print("\nA:", reason)
                print("I can answer the part that exists in your files, but I won't guess the missing side.")
                print("Options: upload more notes OR turn on web fallback (:web on).\n")
                continue

        # 3) General sufficiency check (especially for long answers)
        sufficient = context_is_sufficient(rr, length_mode)
        if not sufficient:
            if web_fallback:
                web_results = tavily_search(question)
                web_context, web_cites = build_tagged_web_context(web_results) if web_results else ("", [])
                file_context, file_cites = build_tagged_context(rr.docs)

                if not web_results:
                    # still insufficient + no web
                    print("\nA: I found some related info in your files, but not enough to answer confidently.")
                    print("Try a more specific question OR upload more relevant notes.\n")
                    continue

                user_prompt = make_user_prompt(question, length_mode, file_context="(No relevant file context)", web_context=web_context)
                resp = llm.invoke([SystemMessage(content=SYSTEM_RULES), HumanMessage(content=user_prompt)])

                print("\n📄 I couldn't find relevant information in your documents.")
                print("🌐 Here's what I found from the web instead:\n")
                print("A:", resp.content.strip(), "\n")
                print("Web Sources:")
                for c in web_cites:
                    print("-", c)
                print()
                continue

            # No web: be strict
            print("\nA: I couldn't find relevant information in your documents for this question.")
            print("Options:")
            print("  1. Upload notes that cover this topic")
            print("  2. Enable web fallback: :web on\n")
            continue

        # 4) Answer from files (normal path)
        file_context, file_cites = build_tagged_context(rr.docs)
        user_prompt = make_user_prompt(question, length_mode, file_context=file_context, web_context=None)
        resp = llm.invoke([SystemMessage(content=SYSTEM_RULES), HumanMessage(content=user_prompt)])

        answer_text = resp.content.strip()

        # Check if LLM said it couldn't find the answer (post-answer safety check)
        cant_answer_phrases = [
            "does not contain", "do not contain", "cannot provide", "can't provide",
            "no information", "not found", "not mentioned", "doesn't mention",
            "unable to find", "couldn't find", "could not find"
        ]
        llm_says_no_answer = any(phrase in answer_text.lower() for phrase in cant_answer_phrases)

        if llm_says_no_answer and web_fallback:
            # LLM couldn't answer from files → try web
            web_results = tavily_search(question)
            if web_results:
                web_context, web_cites = build_tagged_web_context(web_results)
                user_prompt = make_user_prompt(question, length_mode, file_context="(No relevant file context)", web_context=web_context)
                resp = llm.invoke([SystemMessage(content=SYSTEM_RULES), HumanMessage(content=user_prompt)])

                print("\n📄 I couldn't find this specific answer in your documents.")
                print("🌐 Here's what I found from the web instead:\n")
                print("A:", resp.content.strip(), "\n")
                print("Web Sources:")
                for c in web_cites:
                    print("-", c)
                print()
                continue

        print("\nA:", answer_text, "\n")
        print("File Sources:")
        for c in file_cites:
            print("-", c)
        print()


if __name__ == "__main__":
    main()
