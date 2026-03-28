"""
failed_queries_db.py — Study Helper v2
Track failed RAG queries for debugging.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from supabase_client import get_supabase


def log_failed_query(user_id: str, query: str, top_score: Optional[float] = None) -> bool:
    """Log a failed RAG query. Returns True if successful."""
    try:
        supabase = get_supabase()
        response = supabase.table("sh_failed_queries").insert({
            "user_id": user_id,
            "query": query,
            "top_score": top_score,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        
        return len(response.data) > 0 if response.data else False
    except Exception as e:
        print(f"[failed_queries_db] Log error: {e}")
        return False


def get_failed_queries(user_id: str, limit: int = 50) -> list:
    """Get user's failed queries sorted by created_at desc."""
    try:
        supabase = get_supabase()
        response = supabase.table("sh_failed_queries").select(
            "*"
        ).eq("user_id", user_id).order(
            "created_at", desc=True
        ).limit(limit).execute()
        
        return response.data if response.data else []
    except Exception as e:
        print(f"[failed_queries_db] Get error: {e}")
        return []


def delete_failed_query(query_id: str) -> bool:
    """Delete a failed query log. Returns True if successful."""
    try:
        supabase = get_supabase()
        response = supabase.table("sh_failed_queries").delete().eq(
            "id", query_id
        ).execute()
        
        return len(response.data) > 0 if response.data else False
    except Exception as e:
        print(f"[failed_queries_db] Delete error: {e}")
        return False
