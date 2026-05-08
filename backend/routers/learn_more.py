"""
backend/routers/learn_more.py
Learn More endpoints — YouTube + Wikimedia integration.

GET /api/v1/learn-more?topic=X          → get videos + images for topic
GET /api/v1/learn-more/ambiguity?topic=X → check if topic is ambiguous
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from dependencies import get_current_user

router = APIRouter()


@router.get("")
async def learn_more(
    topic: str = Query(..., min_length=1),
    user_id: str = Depends(get_current_user)
):
    """
    Get educational videos and images for a topic.
    Checks cache first, fetches fresh if needed.
    """
    from services.learn_more_service import get_learn_more
    if not topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    result = get_learn_more(topic.strip())
    return result


@router.get("/ambiguity")
async def check_ambiguity(
    topic: str = Query(..., min_length=1),
    answer: str = Query(""),
    user_id: str = Depends(get_current_user)
):
    """
    Check if a topic is ambiguous and return disambiguation options.
    """
    from services.learn_more_service import detect_ambiguity, get_ambiguity_options
    ambiguous_term = detect_ambiguity(topic.strip(), answer)
    if ambiguous_term:
        options = get_ambiguity_options(ambiguous_term)
        return {
            "ambiguous": True,
            "term": ambiguous_term,
            "options": options
        }
    return {"ambiguous": False, "term": None, "options": []}
