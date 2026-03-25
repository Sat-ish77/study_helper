"""
quiz_db.py — Study Helper v2
Quiz history CRUD using Supabase postgres.
"""
from __future__ import annotations
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


def _client():
    from supabase import create_client
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))


def save_score(
    user_id: str,
    topic: str,
    source_file: str,
    score: int,
    total: int,
    difficulty: str,
    types_used: list[str],
    language: str = "English",
) -> bool:
    try:
        _client().table("sh_quiz_scores").insert({
            "user_id":     user_id,
            "topic":       topic,
            "source_file": source_file,
            "score":       score,
            "total":       total,
            "difficulty":  difficulty,
            "types_used":  types_used,
            "language":    language,
        }).execute()
        return True
    except Exception as e:
        print(f"[quiz_db.save_score] {e}")
        return False


def get_user_history(user_id: str, limit: int = 20) -> list[dict]:
    try:
        res = _client().table("sh_quiz_scores") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        return res.data or []
    except Exception:
        return []


def get_user_stats(user_id: str) -> dict:
    """Returns {total_questions, accuracy_pct, total_quizzes}"""
    try:
        res = _client().table("sh_quiz_scores") \
            .select("score, total") \
            .eq("user_id", user_id) \
            .execute()
        rows = res.data or []
        if not rows:
            return {"total_questions": 0, "accuracy_pct": 0, "total_quizzes": 0}
        total_q  = sum(r["total"] for r in rows)
        total_s  = sum(r["score"] for r in rows)
        accuracy = round((total_s / total_q * 100) if total_q else 0, 1)
        return {
            "total_questions": total_q,
            "accuracy_pct":    accuracy,
            "total_quizzes":   len(rows),
        }
    except Exception:
        return {"total_questions": 0, "accuracy_pct": 0, "total_quizzes": 0}