"""
backend/routers/canvas.py
Canvas iCal calendar endpoints.

GET  /api/v1/canvas/events   → fetch and parse iCal events
POST /api/v1/canvas/url      → save user's iCal URL
DELETE /api/v1/canvas/url    → clear iCal URL

Unlike Streamlit which only showed 14 days,
the React frontend can show ALL events and filter client-side.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from dependencies import get_current_user, get_db

router = APIRouter()


class SaveUrlRequest(BaseModel):
    url: str


@router.post("/url")
async def save_url(
    body: SaveUrlRequest,
    user_id: str = Depends(get_current_user)
):
    """Save user's Canvas iCal URL to sh_user_settings."""
    if not body.url or not body.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    sb = get_db()
    sb.table("sh_user_settings").upsert({
        "user_id": user_id,
        "ical_url": body.url.strip(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }, on_conflict="user_id").execute()

    return {"saved": True}


@router.get("/url")
async def get_url(user_id: str = Depends(get_current_user)):
    """Get user's saved iCal URL."""
    sb = get_db()
    result = sb.table("sh_user_settings")\
        .select("ical_url")\
        .eq("user_id", user_id)\
        .execute()

    if result.data:
        return {"url": result.data[0].get("ical_url")}
    return {"url": None}


@router.delete("/url")
async def clear_url(user_id: str = Depends(get_current_user)):
    """Clear user's iCal URL."""
    sb = get_db()
    sb.table("sh_user_settings").upsert({
        "user_id": user_id,
        "ical_url": None,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }, on_conflict="user_id").execute()
    return {"cleared": True}


@router.get("/events")
async def get_events(
    days_ahead: int = 30,
    user_id: str = Depends(get_current_user)
):
    """
    Fetch and parse Canvas iCal events.
    Returns ALL events (React filters client-side).
    Caches in sh_canvas_cache for 1 hour.
    """
    import hashlib
    import json
    from datetime import timedelta

    sb = get_db()

    # Get iCal URL
    result = sb.table("sh_user_settings")\
        .select("ical_url")\
        .eq("user_id", user_id)\
        .execute()

    if not result.data or not result.data[0].get("ical_url"):
        return {"events": [], "error": "No Canvas URL saved"}

    ical_url = result.data[0]["ical_url"]
    cache_key = f"canvas_{hashlib.md5(ical_url.encode()).hexdigest()}_{user_id}"

    # Check cache
    cache = sb.table("sh_canvas_cache")\
        .select("data, expires_at")\
        .eq("cache_key", cache_key)\
        .execute()

    now = datetime.now(timezone.utc)

    if cache.data:
        expires = datetime.fromisoformat(cache.data[0]["expires_at"])
        if expires > now:
            return {"events": cache.data[0]["data"], "cached": True}

    # Fetch and parse iCal
    try:
        import requests
        from icalendar import Calendar

        resp = requests.get(ical_url, timeout=10)
        cal = Calendar.from_ical(resp.content)

        events = []
        for component in cal.walk():
            if component.name == "VEVENT":
                start = component.get("dtstart")
                end = component.get("dtend")
                events.append({
                    "uid": str(component.get("uid", "")),
                    "title": str(component.get("summary", "")),
                    "start": start.dt.isoformat() if start else None,
                    "end": end.dt.isoformat() if end else None,
                    "description": str(component.get("description", "")),
                    "location": str(component.get("location", "")),
                })

        # Sort by start date
        events.sort(key=lambda e: e.get("start") or "")

        # Cache for 1 hour
        expires_at = (now + timedelta(hours=1)).isoformat()
        sb.table("sh_canvas_cache").upsert({
            "cache_key": cache_key,
            "data": events,
            "created_at": now.isoformat(),
            "expires_at": expires_at,
        }, on_conflict="cache_key").execute()

        return {"events": events, "count": len(events), "cached": False}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch calendar: {str(e)}")