"""
backend/routers/conversations.py
Saved conversation endpoints.

GET    /api/v1/conversations        → list user's conversations
POST   /api/v1/conversations        → create new conversation
GET    /api/v1/conversations/{id}   → get full conversation with messages
PUT    /api/v1/conversations/{id}   → update messages
DELETE /api/v1/conversations/{id}   → delete conversation

conversations use sh_conversations table:
  id UUID, user_id TEXT, title TEXT, messages JSONB, 
  created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from dependencies import get_current_user, get_db

router = APIRouter()


class CreateConversationRequest(BaseModel):
    title: str
    messages: list = []


class UpdateConversationRequest(BaseModel):
    messages: list


@router.get("")
async def list_conversations(
    limit: int = 20,
    user_id: str = Depends(get_current_user)
):
    sb = get_db()
    result = sb.table("sh_conversations")\
        .select("id, title, created_at, updated_at")\
        .eq("user_id", user_id)\
        .order("updated_at", desc=True)\
        .limit(limit)\
        .execute()
    return {"conversations": result.data or [], "count": len(result.data or [])}


@router.post("")
async def create_conversation(
    body: CreateConversationRequest,
    user_id: str = Depends(get_current_user)
):
    sb = get_db()
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "user_id": user_id,
        "title": body.title[:100],
        "messages": body.messages,
        "created_at": now,
        "updated_at": now,
    }
    result = sb.table("sh_conversations").insert(row).execute()
    return {"created": True, "id": result.data[0]["id"] if result.data else None}


@router.get("/{conv_id}")
async def get_conversation(
    conv_id: str,
    user_id: str = Depends(get_current_user)
):
    sb = get_db()
    result = sb.table("sh_conversations")\
        .select("*")\
        .eq("id", conv_id)\
        .eq("user_id", user_id)\
        .execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return result.data[0]


@router.put("/{conv_id}")
async def update_conversation(
    conv_id: str,
    body: UpdateConversationRequest,
    user_id: str = Depends(get_current_user)
):
    sb = get_db()
    sb.table("sh_conversations").update({
        "messages": body.messages,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }).eq("id", conv_id).eq("user_id", user_id).execute()
    return {"updated": True}


@router.delete("/{conv_id}")
async def delete_conversation(
    conv_id: str,
    user_id: str = Depends(get_current_user)
):
    sb = get_db()
    sb.table("sh_conversations")\
        .delete()\
        .eq("id", conv_id)\
        .eq("user_id", user_id)\
        .execute()
    return {"deleted": True}