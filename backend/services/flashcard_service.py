"""
backend/services/flashcard_service.py
Flashcard business logic ported from flashcard_db.py.
Uses exact same table names and column names as original.

Table: sh_flashcards
Columns: id, user_id, question, answer, source_file, ease_factor,
         interval_days, repetitions, next_review, last_review, created_at
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from dependencies import get_db


# ── Stats ────────────────────────────────────────────────────────────────────

def get_flashcard_stats(user_id: str) -> Dict:
    """
    Get flashcard statistics for dashboard.
    Ported from flashcard_db.get_flashcard_stats().
    """
    try:
        sb = get_db()

        # Total cards
        total_result = sb.table("sh_flashcards") \
            .select("id", count="exact") \
            .eq("user_id", user_id) \
            .execute()
        total_cards = total_result.count or 0

        # Due cards
        now = datetime.now(timezone.utc).isoformat()
        due_result = sb.table("sh_flashcards") \
            .select("id", count="exact") \
            .eq("user_id", user_id) \
            .lte("next_review", now) \
            .execute()
        due_cards = due_result.count or 0

        # Learned cards (repetitions >= 3)
        learned_result = sb.table("sh_flashcards") \
            .select("id", count="exact") \
            .eq("user_id", user_id) \
            .gte("repetitions", 3) \
            .execute()
        learned_cards = learned_result.count or 0

        # New cards (repetitions == 0)
        new_result = sb.table("sh_flashcards") \
            .select("id", count="exact") \
            .eq("user_id", user_id) \
            .eq("repetitions", 0) \
            .execute()
        new_cards = new_result.count or 0

        return {
            "total": total_cards,
            "due": due_cards,
            "learned": learned_cards,
            "new": new_cards,
            "retention": round(learned_cards / total_cards * 100, 1) if total_cards > 0 else 0
        }

    except Exception as e:
        print(f"[flashcard_service] Stats error: {e}")
        return {
            "total": 0,
            "due": 0,
            "learned": 0,
            "new": 0,
            "retention": 0
        }


# ── Save from Q&A ───────────────────────────────────────────────────────────

PREAMBLE_LINES = [
    "📄 I couldn't find this in your documents.",
    "🌐 Here's what I found from the web:",
    "📄 Couldn't find enough in your documents.",
    "📄 I can't find this in your uploaded files.",
    "🌐 Here's what I found from the web:",
]


def _clean_answer_preamble(answer: str) -> str:
    """Remove RAG preamble lines from flashcard answers."""
    answer_lines = answer.split('\n')
    clean_lines: List[str] = []
    skip_next = False

    for line in answer_lines:
        is_preamble = any(preamble in line for preamble in PREAMBLE_LINES)
        if is_preamble:
            skip_next = True
        elif skip_next:
            skip_next = False
        else:
            clean_lines.append(line)

    return '\n'.join(clean_lines).strip()


def save_flashcard_from_qa(
    user_id: str,
    question: str,
    answer: str,
    source_file: str = ""
) -> Optional[str]:
    """
    Save a flashcard from a chat Q&A pair.
    Cleans preamble lines from the answer.
    Returns card ID if successful, None otherwise.
    Ported from flashcard_db.save_flashcard_from_qa().
    """
    question = question.strip()
    clean_answer = _clean_answer_preamble(answer)

    try:
        sb = get_db()
        row = {
            "user_id": user_id,
            "question": question,
            "answer": clean_answer,
            "source_file": source_file.strip(),
            "ease_factor": 2.5,
            "interval_days": 1,
            "repetitions": 0,
            "next_review": datetime.now(timezone.utc).date().isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        result = sb.table("sh_flashcards").insert(row).execute()
        if result.data:
            return result.data[0]["id"]
        return None
    except Exception as e:
        print(f"[flashcard_service] Save from QA error: {e}")
        return None