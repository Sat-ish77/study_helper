"""
conversation_db.py — Study Helper v2
Manage saved conversations in Supabase.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from supabase_client import get_supabase


def create_conversation(user_id: str, title: str, messages: List[Dict]) -> Optional[str]:
    """Create a new conversation. Returns conversation ID if successful."""
    try:
        supabase = get_supabase()
        response = supabase.table("sh_conversations").insert({
            "user_id": user_id,
            "title": title,
            "messages": json.dumps(messages),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        
        if response.data:
            return response.data[0]["id"]
        return None
    except Exception as e:
        print(f"[conversation_db] Create error: {e}")
        return None


def update_conversation(conv_id: str, messages: List[Dict]) -> bool:
    """Update conversation messages and timestamp. Returns True if successful."""
    try:
        supabase = get_supabase()
        response = supabase.table("sh_conversations").update({
            "messages": json.dumps(messages),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", conv_id).execute()
        
        return len(response.data) > 0 if response.data else False
    except Exception as e:
        print(f"[conversation_db] Update error: {e}")
        return False


def list_conversations(user_id: str, limit: int = 20) -> List[Dict]:
    """Get list of user's conversations sorted by updated_at desc."""
    try:
        supabase = get_supabase()
        response = supabase.table("sh_conversations").select(
            "id, title, created_at, updated_at"
        ).eq("user_id", user_id).order(
            "updated_at", desc=True
        ).limit(limit).execute()
        
        return response.data if response.data else []
    except Exception as e:
        print(f"[conversation_db] List error: {e}")
        return []


def load_conversation(conv_id: str) -> Optional[Dict]:
    """Load full conversation including messages."""
    try:
        supabase = get_supabase()
        response = supabase.table("sh_conversations").select(
            "*"
        ).eq("id", conv_id).execute()
        
        if response.data and len(response.data) > 0:
            conv = response.data[0]
            # Parse messages JSON
            if isinstance(conv.get("messages"), str):
                conv["messages"] = json.loads(conv["messages"])
            return conv
        return None
    except Exception as e:
        print(f"[conversation_db] Load error: {e}")
        return None


def delete_conversation(conv_id: str) -> bool:
    """Delete a conversation. Returns True if successful."""
    try:
        supabase = get_supabase()
        response = supabase.table("sh_conversations").delete().eq(
            "id", conv_id
        ).execute()
        
        return len(response.data) > 0 if response.data else False
    except Exception as e:
        print(f"[conversation_db] Delete error: {e}")
        return False


def auto_title(first_question: str, llm) -> str:
    """Generate a short title from the first question."""
    try:
        from langchain_core.messages import HumanMessage
        
        prompt = f"""Generate a very short title (max 5 words) for this question:
        
Question: {first_question}

Title:"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        title = response.content.strip()
        
        # Clean up the title
        title = title.strip('"').strip("'").strip()
        if len(title) > 50:
            title = title[:47] + "..."
        
        return title if title else "New Chat"
    except Exception:
        # Fallback to truncated question
        return first_question[:30] + "..." if len(first_question) > 30 else first_question
