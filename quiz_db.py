"""
quiz_db.py — Study Helper v2
Quiz history CRUD using Supabase postgres.
"""
from __future__ import annotations
import os
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv
from supabase_client import get_supabase

load_dotenv()


def _client():
    return get_supabase()


def save_score(
    user_id: str,
    topic: str,
    source_file: str,
    score: int,
    total: int,
    difficulty: str,
    types_used: list[str],
    language: str = "English",
    course_name: str = None,
) -> bool:
    try:
        data = {
            "user_id":     user_id,
            "topic":       topic,
            "source_file": source_file,
            "score":       score,
            "total":       total,
            "difficulty":  difficulty,
            "types_used":  types_used,
            "language":    language,
        }
        if course_name:
            data["course_name"] = course_name
        
        _client().table("sh_quiz_scores").insert(data).execute()
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


def get_recent_quiz_scores(user_id: str, limit: int = 10) -> List[Dict]:
    """
    Get recent quiz scores for a user.
    Returns list of quiz score dictionaries.
    """
    try:
        supabase = get_supabase()
        
        result = supabase.table("sh_quiz_scores") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        return result.data or []
        
    except Exception as e:
        print(f"[quiz_db] Recent scores error: {e}")
        return []


def get_quiz_stats(user_id: str) -> Dict:
    """
    Get comprehensive quiz statistics for a user.
    Returns dict with average_score, total_taken, and other stats.
    """
    try:
        supabase = get_supabase()
        
        result = supabase.table("sh_quiz_scores") \
            .select("score, total, created_at") \
            .eq("user_id", user_id) \
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
            "recent_scores": rows[:5]  # Last 5 scores
        }
        
    except Exception as e:
        print(f"[quiz_db] Stats error: {e}")
        return {
            "average_score": 0,
            "total_taken": 0,
            "total_questions": 0,
            "best_score": 0,
            "recent_scores": []
        }


def get_user_stats(user_id: str) -> Dict:
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