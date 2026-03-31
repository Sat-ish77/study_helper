"""
backend/routers/flashcards.py
Flashcard endpoints with SM-2 spaced repetition.

GET    /api/v1/flashcards           → list all cards (or due today)
POST   /api/v1/flashcards           → create a card
PUT    /api/v1/flashcards/{id}      → update after review (SM-2)
DELETE /api/v1/flashcards/{id}      → delete a card
POST   /api/v1/flashcards/generate  → generate from document chunks
POST   /api/v1/flashcards/bulk      → bulk save generated cards

SM-2 Algorithm:
- ease_factor: how easy the card is (default 2.5)
- interval_days: how many days until next review
- repetitions: how many times reviewed correctly in a row

After each review:
- quality 0-2 (forgot): reset to day 1, decrease ease
- quality 3-4 (hard but remembered): increase interval slowly
- quality 5 (easy): increase interval significantly
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone, date, timedelta
from dependencies import get_current_user, get_db
from model_manager import get_llm

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class CreateCardRequest(BaseModel):
    question: str
    answer: str
    source_file: Optional[str] = None


class ReviewCardRequest(BaseModel):
    quality: int   # 0-5 scale (0=complete blackout, 5=perfect recall)


class GenerateCardsRequest(BaseModel):
    source_file: str          # which uploaded file to generate from
    num_cards: int = 10
    model: str = "Llama 3.3 70B"


class BulkSaveRequest(BaseModel):
    cards: list   # list of {question, answer, source_file}


# ── List Cards ────────────────────────────────────────────────────────────────

@router.get("")
async def list_cards(
    due_only: bool = False,
    user_id: str = Depends(get_current_user)
):
    """
    List flashcards for this user.
    due_only=True: only cards due for review today (next_review <= today)
    """
    sb = get_db()
    query = sb.table("sh_flashcards")\
        .select("*")\
        .eq("user_id", user_id)

    if due_only:
        today = date.today().isoformat()
        query = query.lte("next_review", today)

    result = query.order("next_review").execute()
    cards = result.data or []
    return {"cards": cards, "count": len(cards)}


# ── Create Card ───────────────────────────────────────────────────────────────

@router.post("")
async def create_card(
    body: CreateCardRequest,
    user_id: str = Depends(get_current_user)
):
    """Create a single flashcard."""
    sb = get_db()
    row = {
        "user_id": user_id,
        "question": body.question,
        "answer": body.answer,
        "source_file": body.source_file or "",
        "ease_factor": 2.5,
        "interval_days": 1,
        "repetitions": 0,
        "next_review": date.today().isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = sb.table("sh_flashcards").insert(row).execute()
    return {"created": True, "card": result.data[0] if result.data else None}


# ── Review Card (SM-2) ────────────────────────────────────────────────────────

@router.put("/{card_id}")
async def review_card(
    card_id: str,
    body: ReviewCardRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Update flashcard after a review session using SM-2 algorithm.
    quality: 0-5
      0-2 = forgot (reset)
      3 = remembered with difficulty
      4 = remembered with hesitation
      5 = perfect recall
    """
    sb = get_db()

    # Fetch current card state
    result = sb.table("sh_flashcards")\
        .select("*")\
        .eq("id", card_id)\
        .eq("user_id", user_id)\
        .execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Flashcard not found")

    card = result.data[0]
    q = max(0, min(5, body.quality))

    ease = card.get("ease_factor", 2.5)
    interval = card.get("interval_days", 1)
    reps = card.get("repetitions", 0)

    # SM-2 algorithm
    if q < 3:
        # Forgot — reset
        reps = 0
        interval = 1
    else:
        if reps == 0:
            interval = 1
        elif reps == 1:
            interval = 6
        else:
            interval = round(interval * ease)
        reps += 1

    # Update ease factor
    ease = ease + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    ease = max(1.3, ease)  # minimum ease factor

    next_review = (date.today() + timedelta(days=interval)).isoformat()

    update = {
        "ease_factor": round(ease, 2),
        "interval_days": interval,
        "repetitions": reps,
        "next_review": next_review,
        "last_review": date.today().isoformat(),
    }

    sb.table("sh_flashcards")\
        .update(update)\
        .eq("id", card_id)\
        .execute()

    return {
        "updated": True,
        "next_review": next_review,
        "interval_days": interval,
        "ease_factor": round(ease, 2),
    }


# ── Delete Card ───────────────────────────────────────────────────────────────

@router.delete("/{card_id}")
async def delete_card(
    card_id: str,
    user_id: str = Depends(get_current_user)
):
    """Delete a flashcard."""
    sb = get_db()
    sb.table("sh_flashcards")\
        .delete()\
        .eq("id", card_id)\
        .eq("user_id", user_id)\
        .execute()
    return {"deleted": True}


# ── Generate from Document ────────────────────────────────────────────────────

@router.post("/generate")
async def generate_cards(
    body: GenerateCardsRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Generate flashcards from a specific uploaded document.
    Fetches chunks for that file, combines them,
    asks LLM to create Q&A pairs.
    Returns cards as preview — does NOT auto-save.
    Frontend shows preview, user confirms before saving.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    import json

    sb = get_db()

    # Fetch chunks for this file
    result = sb.table("sh_document_chunks")\
        .select("content")\
        .eq("user_id", user_id)\
        .eq("source_file", body.source_file)\
        .limit(20)\
        .execute()

    chunks = result.data or []
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No content found for {body.source_file}"
        )

    # Combine up to 3000 chars of content
    combined = " ".join(c["content"] for c in chunks)[:3000]

    llm = get_llm(body.model)
    prompt = f"""Generate {body.num_cards} flashcards from this study material.

Content:
{combined}

Return ONLY valid JSON:
{{
  "flashcards": [
    {{
      "question": "Clear, specific question",
      "answer": "Concise, accurate answer"
    }}
  ]
}}

Rules:
- Questions must test actual concepts, not page numbers or structure
- Answers should be 1-3 sentences
- Each flashcard must be unique
- Keep technical terms in English"""

    try:
        resp = llm.invoke([
            SystemMessage(content="You are a flashcard generator. Return only valid JSON."),
            HumanMessage(content=prompt)
        ])
        content = resp.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        cards = data.get("flashcards", [])

        return {
            "cards": cards,
            "count": len(cards),
            "source_file": body.source_file,
            "preview": True  # signal to frontend to show preview before saving
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ── Bulk Save ─────────────────────────────────────────────────────────────────

@router.post("/bulk")
async def bulk_save(
    body: BulkSaveRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Save multiple flashcards at once.
    Called after user confirms generated cards preview.
    """
    sb = get_db()
    today = date.today().isoformat()
    now = datetime.now(timezone.utc).isoformat()

    rows = []
    for card in body.cards:
        rows.append({
            "user_id": user_id,
            "question": card.get("question", ""),
            "answer": card.get("answer", ""),
            "source_file": card.get("source_file", ""),
            "ease_factor": 2.5,
            "interval_days": 1,
            "repetitions": 0,
            "next_review": today,
            "created_at": now,
        })

    if not rows:
        return {"saved": 0}

    result = sb.table("sh_flashcards").insert(rows).execute()
    return {"saved": len(result.data or []), "cards": result.data}