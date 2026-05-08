"""
backend/services/quiz_service.py
Quiz business logic ported from quiz_db.py.

Table: sh_quiz_scores
Columns: id, user_id, topic, source_file, score, total, difficulty,
         types_used, language, course_name, created_at
"""
from __future__ import annotations

from typing import Dict

from dependencies import get_db


def get_quiz_stats(user_id: str) -> Dict:
    """
    Get comprehensive quiz statistics for a user.
    Ported from quiz_db.get_quiz_stats().
    """
    try:
        sb = get_db()

        result = sb.table("sh_quiz_scores") \
            .select("score, total, created_at") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .execute()

        rows = result.data or []

        if not rows:
            return {
                "average_score": 0,
                "total_taken": 0,
                "total_questions": 0,
                "best_score": 0,
                "recent_scores": []
            }

        total_quizzes = len(rows)
        total_questions = sum(r["total"] for r in rows)
        total_correct = sum(r["score"] for r in rows)
        average_score = round((total_correct / total_questions * 100) if total_questions else 0, 1)
        best_score = max(r["score"] / r["total"] * 100 for r in rows) if rows else 0

        return {
            "average_score": average_score,
            "total_taken": total_quizzes,
            "total_questions": total_questions,
            "best_score": round(best_score, 1),
            "recent_scores": rows[:5]
        }

    except Exception as e:
        print(f"[quiz_service] Stats error: {e}")
        return {
            "average_score": 0,
            "total_taken": 0,
            "total_questions": 0,
            "best_score": 0,
            "recent_scores": []
        }


def get_user_stats(user_id: str) -> Dict:
    """
    Simplified stats for dashboard.
    Ported from quiz_db.get_user_stats().
    Returns {total_questions, accuracy_pct, total_quizzes}.
    """
    try:
        sb = get_db()
        res = sb.table("sh_quiz_scores") \
            .select("score, total") \
            .eq("user_id", user_id) \
            .execute()
        rows = res.data or []
        if not rows:
            return {"total_questions": 0, "accuracy_pct": 0, "total_quizzes": 0}
        total_q = sum(r["total"] for r in rows)
        total_s = sum(r["score"] for r in rows)
        accuracy = round((total_s / total_q * 100) if total_q else 0, 1)
        return {
            "total_questions": total_q,
            "accuracy_pct": accuracy,
            "total_quizzes": len(rows),
        }
    except Exception:
        return {"total_questions": 0, "accuracy_pct": 0, "total_quizzes": 0}